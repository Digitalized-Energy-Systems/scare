"""Unit tests for the collaborators extracted from EnergyConverterRole.

These cover the previously-untested pure logic now living in cp_envelope.py,
cp_flex.py, and cp_l3.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from scare.base.model import AvailableFlexAnswer, Sector
from scare.service.coupling.cp_envelope import CoalitionEnvelope
from scare.service.coupling.cp_flex import (
    aggregate_flex_answers,
    compute_sector_priorities,
)
from scare.service.coupling.cp_l3 import CPComponentView, compute_cp_setpoint

# Flat ADMM result layout [el, heat, gas], mirroring cp._RESULT_INDEX.
RESULT_INDEX = {Sector.ELECTRICITY: 0, Sector.HEAT: 1, Sector.GAS: 2}


# --------------------------------------------------------------------------- #
# CoalitionEnvelope
# --------------------------------------------------------------------------- #


def test_envelope_inactive_by_default():
    env = CoalitionEnvelope()
    assert env.active(now=0.0) is False
    assert env.clamp([1.0, 2.0, 3.0], RESULT_INDEX, now=0.0) is None


def test_envelope_active_within_ttl_and_expires():
    env = CoalitionEnvelope()
    env.set({Sector.ELECTRICITY: 5.0}, ttl_s=10.0, coalition_id="c1", now=0.0)
    assert env.active(now=5.0) is True
    assert env.coalition_id == "c1"
    # Past expiry: inactive, and the flows are cleared (matches original).
    assert env.active(now=10.1) is False
    assert env.flows_mw is None


def test_envelope_clamp_overwrites_only_present_sectors():
    env = CoalitionEnvelope()
    env.set(
        {Sector.ELECTRICITY: 5.0, Sector.GAS: -2.0},
        ttl_s=10.0,
        coalition_id="c1",
        now=0.0,
    )
    result = [1.0, 2.0, 3.0]
    pre = env.clamp(result, RESULT_INDEX, now=1.0)
    assert pre == [1.0, 2.0, 3.0]  # pre-clamp snapshot
    assert result == [5.0, 2.0, -2.0]  # heat (absent) untouched; in place


def test_envelope_clamp_returns_none_when_expired():
    env = CoalitionEnvelope()
    env.set({Sector.ELECTRICITY: 5.0}, ttl_s=1.0, coalition_id="c1", now=0.0)
    result = [1.0, 2.0, 3.0]
    assert env.clamp(result, RESULT_INDEX, now=2.0) is None
    assert result == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# cp_flex
# --------------------------------------------------------------------------- #


def _answer(sector, balance, **kw):
    return AvailableFlexAnswer(
        flex=kw.get("flex", 0.0),
        balance=balance,
        shedded=kw.get("shedded", 0.0),
        sector=sector,
        demand_by_priority=kw.get("demand_by_priority", {}),
        served_by_priority=kw.get("served_by_priority", {}),
        demand_by_sector_priority=kw.get("demand_by_sector_priority", {}),
        served_by_sector_priority=kw.get("served_by_sector_priority", {}),
        unmet_by_sector=kw.get("unmet_by_sector", {}),
    )


def test_aggregate_sums_balance_and_unmet_per_sector():
    answers = [
        _answer(Sector.ELECTRICITY, balance=2.0, unmet_by_sector={"electricity": 1.0}),
        _answer(Sector.ELECTRICITY, balance=-0.5, unmet_by_sector={"electricity": 0.5}),
    ]
    agg = aggregate_flex_answers(answers)
    assert agg.imbalance_by_sector[Sector.ELECTRICITY] == 1.5
    assert agg.unmet_by_sector_total[Sector.ELECTRICITY] == 1.5


def test_aggregate_tracks_lowest_unmet_tier():
    # tier 2 unmet on electricity, tier 1 unmet on heat -> heat outranks.
    answers = [
        _answer(
            Sector.ELECTRICITY,
            balance=0.0,
            demand_by_sector_priority={"electricity": {2: 4.0}},
            served_by_sector_priority={"electricity": {2: 1.0}},
        ),
        _answer(
            Sector.HEAT,
            balance=0.0,
            demand_by_sector_priority={"heat": {1: 3.0}},
            served_by_sector_priority={"heat": {1: 0.0}},
        ),
    ]
    agg = aggregate_flex_answers(answers)
    assert agg.top_unmet_tier_per_sector[Sector.ELECTRICITY] == 2
    assert agg.top_unmet_tier_per_sector[Sector.HEAT] == 1
    assert agg.top_unmet_mag_per_sector[Sector.ELECTRICITY] == 3.0


def test_sector_priorities_normalised_and_top_tier_dominates():
    answers = [
        _answer(
            Sector.ELECTRICITY,
            balance=0.0,
            demand_by_sector_priority={"electricity": {3: 4.0}},
            served_by_sector_priority={"electricity": {3: 0.0}},
        ),
        _answer(
            Sector.HEAT,
            balance=0.0,
            demand_by_sector_priority={"heat": {1: 1.0}},
            served_by_sector_priority={"heat": {1: 0.0}},
        ),
    ]
    agg = aggregate_flex_answers(answers)
    pr = compute_sector_priorities(np, agg)
    assert pr.shape == (3,)
    assert float(pr.min()) >= 0.01
    assert float(pr.max()) <= 1.0 + 1e-12
    # heat (tier 1) must outrank electricity (tier 3) and gas (no demand).
    assert pr[1] > pr[0]
    assert pr[1] > pr[2]


# --------------------------------------------------------------------------- #
# cp_l3.compute_cp_setpoint
# --------------------------------------------------------------------------- #


def test_compute_cp_setpoint_zero_capacity_or_no_ratios():
    assert compute_cp_setpoint({"capacity_mw": 0.0}, {}) == {}
    assert compute_cp_setpoint({"capacity_mw": 5.0}, {}) == {}
    assert compute_cp_setpoint({"capacity_mw": 5.0, "coupling_ratios": {}}, {}) == {}


def test_compute_cp_setpoint_runs_when_output_more_stressed():
    meta = {"capacity_mw": 4.0, "coupling_ratios": {("electricity", "heat"): 1.0}}
    # heat stressed (1.0), electricity slack (0.0): step = 1*1 - 0 = 1.
    out = compute_cp_setpoint(meta, {"electricity": 0.0, "heat": 1.0})
    assert out["electricity"] == 4.0  # consumes from source
    assert out["heat"] == -4.0  # produces into destination


def test_compute_cp_setpoint_balanced_pair_no_commitment():
    meta = {"capacity_mw": 4.0, "coupling_ratios": {("electricity", "heat"): 1.0}}
    assert compute_cp_setpoint(meta, {"electricity": 1.0, "heat": 1.0}) == {}


# --------------------------------------------------------------------------- #
# cp_l3.CPComponentView
# --------------------------------------------------------------------------- #


class _FakeMirror:
    def __init__(self, reachable):
        self._reachable = set(reachable)

    def reachable_from(self, node, sector=None, allow_cp_bridges=True):
        return set(self._reachable)


def test_component_view_disabled_until_wired():
    view = CPComponentView()
    assert view.enabled() is False
    view.wire(
        topology_mirror=_FakeMirror({1, 2}),
        my_node_id=1,
        cp_meta_by_aid={"cp1": {"node_id": 1}},
        leader_addrs_by_sector={},
        leader_node_ids={},
    )
    assert view.enabled() is True


def test_component_view_peers_and_coordinator_election():
    view = CPComponentView()
    view.wire(
        topology_mirror=_FakeMirror({1, 2}),
        my_node_id=1,
        cp_meta_by_aid={
            "cp1": {"node_id": 1},  # self
            "cp2": {"node_id": 2},  # reachable
            "cp3": {"node_id": 9},  # unreachable
        },
        leader_addrs_by_sector={},
        leader_node_ids={},
    )
    peers = view.cp_peers("cp1")
    assert set(peers) == {"cp1", "cp2"}  # cp3 excluded (unreachable)
    assert view.is_coordinator("cp1") is True  # lex-smallest
    assert view.is_coordinator("cp2") is False


def test_component_view_leader_addrs_filtered_by_reachability():
    view = CPComponentView()
    view.wire(
        topology_mirror=_FakeMirror({1, 2}),
        my_node_id=1,
        cp_meta_by_aid={"cp1": {"node_id": 1}},
        leader_addrs_by_sector={
            Sector.ELECTRICITY: {"L_near": "addr_near", "L_far": "addr_far"},
        },
        leader_node_ids={"L_near": 2, "L_far": 99},
    )
    out = view.leader_addrs("cp1")
    assert out == {Sector.ELECTRICITY: {"L_near": "addr_near"}}


# --------------------------------------------------------------------------- #
# CpActuator + _reply_ask_energy (extracted from EnergyConverterRole, T1)
# --------------------------------------------------------------------------- #


class _FakeBehavior:
    def __init__(self, obs):
        self._obs = obs

    def observe(self, aid):
        return self._obs


def test_cp_actuator_drives_strongest_signal_sector(monkeypatch):
    from scare.service.coupling import cp as cp_mod

    monkeypatch.setattr(cp_mod, "clamp_to_constraints", lambda v, obs, sector: v)
    calls: list = []
    monkeypatch.setattr(
        cp_mod,
        "apply_regulate",
        lambda behavior, aid, factor, **kw: calls.append((aid, factor, kw)),
    )
    obs = {"el_mw": 10.0, "heat_mw": 4.0, "gas_mass_flow_kgs": 5.0}
    act = cp_mod.CpActuator(_FakeBehavior(obs))
    # result [el=2, heat=3, gas=0]: |weights| 2,3,0 -> heat wins; factor 3/4.
    factor = act.apply("cp1", [2.0, 3.0, 0.0], timestamp=7.0)
    assert factor == pytest.approx(0.75)
    assert len(calls) == 1
    aid, f, kw = calls[0]
    assert aid == "cp1"
    assert f == pytest.approx(0.75)
    assert kw == {"sector": "cp", "reason": "cp_admm", "timestamp": 7.0}


def test_cp_actuator_returns_none_without_applicable_sector(monkeypatch):
    from scare.service.coupling import cp as cp_mod

    calls: list = []
    monkeypatch.setattr(cp_mod, "apply_regulate", lambda *a, **k: calls.append(a))
    assert (
        cp_mod.CpActuator(_FakeBehavior({})).apply("cp1", [1.0, 2.0, 3.0], 0.0) is None
    )
    # zero cap -> that sector is skipped too.
    assert (
        cp_mod.CpActuator(_FakeBehavior({"el_mw": 0.0})).apply(
            "cp1", [1.0, 0.0, 0.0], 0.0
        )
        is None
    )
    assert calls == []


def test_cp_actuator_does_not_touch_cp_version(monkeypatch):
    # Guards the extraction contract: the actuator has no access to _cp_version,
    # so it can never advance it (the double-bump the plan review flagged).
    from scare.service.coupling import cp as cp_mod

    monkeypatch.setattr(cp_mod, "apply_regulate", lambda *a, **k: None)
    role = cp_mod.EnergyConverterRole(
        _FakeBehavior({"el_mw": 10.0}),
        flex_actor=object(),
        sectors=[Sector.ELECTRICITY],
    )
    assert role._cp_version.current == 0
    role._actuator.apply("cp1", [5.0, 0.0, 0.0], timestamp=1.0)
    assert role._cp_version.current == 0


def test_reply_ask_energy_uses_sector_setpoint():
    import asyncio

    from scare.base.model import AskEnergyMessage
    from scare.service.coupling.cp import _reply_ask_energy

    sent: list = []

    class _Ctx:
        aid = "cp1"

        async def send_message(self, msg, receiver_addr):
            sent.append((msg, receiver_addr))

    behavior = _FakeBehavior({"el_mw": 8.0, "regulation": 0.5})
    asyncio.run(
        _reply_ask_energy(
            behavior,
            _Ctx(),
            AskEnergyMessage(negotiation_id="n1", sector=Sector.ELECTRICITY),
            meta={"sender_addr": ("leader-addr",), "sender_id": "leader"},
        )
    )
    assert len(sent) == 1
    reply, _ = sent[0]
    assert reply.available == 0.0
    assert reply.setpoint == pytest.approx(4.0)  # 8.0 * 0.5


# --------------------------------------------------------------------------- #
# FlexRound (C8 buffer extracted from EnergyConverterRole)
# --------------------------------------------------------------------------- #


def test_flex_round_open_add_complete_and_drain():
    from types import SimpleNamespace

    from scare.service.coupling.cp import FlexRound

    fr = FlexRound()
    rid = fr.open("cp1", 2)
    assert rid == "cp1-flex-1"
    assert fr.expected == 2 and fr.answers == [] and fr.round_id == rid
    # Stale round id is rejected and not buffered.
    assert fr.add(SimpleNamespace(round_id="other")) is False
    assert fr.answers == []
    # First matching answer: incomplete; second: complete.
    assert fr.add(SimpleNamespace(round_id=rid)) is False
    assert fr.add(SimpleNamespace(round_id=rid)) is True
    assert len(fr.answers) == 2
    drained = fr.drain()
    assert len(drained) == 2
    # drain() atomically closes the round.
    assert fr.expected == 0 and fr.answers == [] and fr.round_id == ""


def test_flex_round_rejects_late_answer_after_drain():
    # The _flex_expected<=0 double-run guard: a late same-round answer arriving
    # after dispatch must not re-complete the round.
    from types import SimpleNamespace

    from scare.service.coupling.cp import FlexRound

    fr = FlexRound()
    rid = fr.open("cp1", 1)
    assert fr.add(SimpleNamespace(round_id=rid)) is True
    fr.drain()
    assert fr.add(SimpleNamespace(round_id=rid)) is False


def test_flex_round_counter_monotonic_and_close_empty():
    from scare.service.coupling.cp import FlexRound

    fr = FlexRound()
    assert fr.open("cp1", 3) == "cp1-flex-1"
    assert fr.open("cp1", 1) == "cp1-flex-2"  # counter persists across rounds
    fr.close_empty()
    assert fr.expected == 0
