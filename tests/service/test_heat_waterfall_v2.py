"""Waterfall v2: sufficiency-gated defer, defer budget, meaningful-target
floor, hydraulic-component scoping, and the multi-target peer-shed actuator.

Regression anchor: eval_full_v2 just_heat — the v1 existence-based defer guard
plus the multiplicative peer shed deadlocked cold nodes (peer reducible decays
geometrically, never hits zero, the guard never releases, the own shed freezes
out-of-band to sim end).
"""

from __future__ import annotations

import pytest
from mango import RoleAgent, create_world
from mango.simulation.world import step_simulation

from scare.base.model import ConstraintStateMessage, Sector
from scare.service.control.constraints import GridConstraintMonitor
from scare.service.control.heat_frontier import HeatFrontierController
from tests.conftest import MockBehavior

_LO = 313.15  # heat t_k floor; target = lo + MARGIN(3) = 316.15


def _decide(ctrl, *, t=300.0, cap=0.05, cur=1.0, my_tier=1, now=0.0):
    return ctrl.decide(
        t=t,
        lo=_LO,
        cap=cap,
        cur=cur,
        sensitivity=660.0,
        now=now,
        my_tier=my_tier,
        has_lock=False,
        waterfall_enabled=True,
    )


# --------------------------------------------------------------------------- #
# Controller: sufficiency gate + defer budget + target floor + component scope
# --------------------------------------------------------------------------- #


def test_deadlock_regression_kilowatt_remnants_do_not_hold_the_shed():
    """Peers whose reducible decayed below the meaningful floor must not
    defer the own shed (the v1 deadlock: existence-based guard on remnants)."""
    c = HeatFrontierController(peer_freshness_s=50.0)
    c.note_peer_state("p4a", 0.0, 4, 5e-4)
    c.note_peer_state("p4b", 0.0, 4, 8e-4)
    out = _decide(c)
    assert out is not None and out.reason == "curtail"
    assert out.new_reg < 1.0


def test_sufficiency_gate_insufficient_peers_shed_self():
    """Lower-priority reducible below the own step's MW-equivalent ⇒ no
    defer (existence is not sufficiency)."""
    c = HeatFrontierController(peer_freshness_s=50.0)
    # own step: MAX_STEP(0.15) * cap(0.05) = 0.0075 MW needed
    c.note_peer_state("p4", 0.0, 4, 0.002)  # meaningful, but too small
    out = _decide(c)
    assert out is not None and out.reason == "curtail"


def test_sufficiency_gate_sufficient_peers_defer():
    c = HeatFrontierController(peer_freshness_s=50.0)
    c.note_peer_state("p4", 0.0, 4, 0.05)  # >= 0.0075 needed
    out = _decide(c)
    assert out is not None and out.reason == "defer_waterfall"
    assert out.new_reg == 1.0
    assert out.needed_mw == pytest.approx(0.15 * 0.05)


def test_defer_budget_times_out_without_warming():
    """Deferring must stop after WATERFALL_DEFER_POLLS polls without the node
    warming — the own shed resumes (t_k feasibility wins over ordering)."""
    c = HeatFrontierController(peer_freshness_s=50.0)
    c.note_peer_state("p4", 0.0, 4, 1.0)  # effectively unlimited reducible
    for i in range(c.WATERFALL_DEFER_POLLS):
        out = _decide(c, now=float(i))
        assert out is not None and out.reason == "defer_waterfall", f"poll {i}"
    # Budget exhausted at the same temperature -> shed self, and stay shedding.
    for i in range(2):
        out = _decide(c, now=float(c.WATERFALL_DEFER_POLLS + i))
        assert out is not None and out.reason == "curtail", f"post-budget {i}"


def test_defer_budget_rearms_on_real_warming():
    c = HeatFrontierController(peer_freshness_s=50.0)
    c.note_peer_state("p4", 0.0, 4, 1.0)
    for i in range(c.WATERFALL_DEFER_POLLS + 1):
        _decide(c, now=float(i))  # exhaust the budget at t=300.0
    # Node warmed by more than DEFER_IMPROVE_K since the anchor -> deferral is
    # working again; the budget re-arms.
    out = _decide(c, t=300.0 + c.WATERFALL_DEFER_IMPROVE_K + 0.1, now=10.0)
    assert out is not None and out.reason == "defer_waterfall"


def test_defer_state_resets_when_back_in_band():
    c = HeatFrontierController(peer_freshness_s=50.0)
    c.note_peer_state("p4", 0.0, 4, 1.0)
    for i in range(c.WATERFALL_DEFER_POLLS + 1):
        _decide(c, now=float(i))
    assert c._defer_exhausted
    assert _decide(c, t=317.0, now=20.0) is None  # in band
    assert not c._defer_exhausted
    out = _decide(c, now=21.0)  # cold again -> fresh budget
    assert out is not None and out.reason == "defer_waterfall"


def test_target_floor_scales_with_needed_relief():
    c = HeatFrontierController(peer_freshness_s=50.0)
    c.note_peer_state("small", 0.0, 4, 0.004)
    c.note_peer_state("big", 0.0, 4, 0.006)
    # needed 0.05 -> floor max(1e-3, 0.005) = 0.005: only "big" qualifies.
    assert c.waterfall_request_targets(1, 0.0, needed_mw=0.05) == [
        ("big", 4, 0.006)
    ]
    # No need context -> absolute floor only: both qualify, biggest first.
    assert c.waterfall_request_targets(1, 0.0) == [
        ("big", 4, 0.006),
        ("small", 4, 0.004),
    ]


def test_component_scoping_filters_foreign_peers():
    c = HeatFrontierController(peer_freshness_s=50.0, component_id=7)
    c.note_peer_state("same", 0.0, 4, 0.05, component_id=7)
    c.note_peer_state("other", 0.0, 4, 0.05, component_id=8)
    c.note_peer_state("legacy", 0.0, 4, 0.02, component_id=None)
    targets = c.waterfall_request_targets(1, 0.0)
    assert [t[0] for t in targets] == ["same", "legacy"]
    assert c.region_has_lower_priority_reducible(1, 0.0) == pytest.approx(0.07)


def test_component_scoping_unscoped_controller_admits_all():
    c = HeatFrontierController(peer_freshness_s=50.0)  # component unknown
    c.note_peer_state("tagged", 0.0, 4, 0.05, component_id=3)
    assert c.waterfall_request_targets(1, 0.0) == [("tagged", 4, 0.05)]


# --------------------------------------------------------------------------- #
# Monitor: multi-target actuator + escalation while self-shedding
# --------------------------------------------------------------------------- #


def _heat_monitor(behavior, aid, *, priority, t_k, component_id=None):
    behavior.set_obs(
        aid,
        {"q_mw_heat": 0.2, "regulation": 1.0, "t_k": t_k, "priority": priority},
    )
    behavior.add_action(aid, "regulate")
    return GridConstraintMonitor(
        behavior,
        Sector.HEAT,
        node_id=aid,
        max_hops=1,
        enable_curtailment_auction=False,
        enable_multihop_constraint=False,
        enable_heat_frontier=False,
        enable_heat_priority_waterfall=True,
        heat_component_id=component_id,
    )


def _seed_peer(m0, peer_monitor, *, tier, reducible):
    now = m0.context.current_timestamp
    origin = str(peer_monitor.context.addr)
    m0._heat_frontier.note_peer_state(origin, now, tier, reducible)
    m0._neighbour_state[(origin, "t_k")] = ConstraintStateMessage(
        sector=Sector.HEAT,
        variable="t_k",
        value=330.0,
        utilization=0.1,
        hops_remaining=1,
        origin_addr=peer_monitor.context.addr,
        priority_tier=tier,
        reducible=reducible,
    )


@pytest.mark.asyncio
async def test_waterfall_multi_target_requests_up_to_cap():
    """A deferring cold load fans requests over several peers per poll (up to
    the target cap) when one peer's expected relief can't cover its step."""
    behavior = MockBehavior()
    world = create_world()
    m0 = _heat_monitor(behavior, "agent-0", priority=1, t_k=300.0)
    peers = []
    for i in range(1, 5):
        m = _heat_monitor(behavior, f"agent-{i}", priority=4, t_k=330.0)
        peers.append(m)
    for i, m in enumerate([m0] + peers):
        agent = world.register(RoleAgent(), suggested_aid=f"agent-{i}")
        agent.add_role(m)

    async with world:
        m0._sens._value = 660.0
        # needed = 0.15 * 0.2 = 0.03 MW; each peer covers 0.5 * 0.01 = 0.005.
        for m in peers:
            _seed_peer(m0, m, tier=4, reducible=0.01)
        await m0._heat_frontier_control()
        await step_simulation(world, step_size_s=1.0)

    shed_peers = {
        a[0]
        for a in behavior.action_log
        if a[1] == "regulate" and a[0] != "agent-0"
    }
    assert len(shed_peers) == GridConstraintMonitor._HEAT_WATERFALL_MAX_TARGETS
    own = [a for a in behavior.action_log if a[0] == "agent-0" and a[1] == "regulate"]
    assert not own, "sufficient peers -> the tier-1 load must defer, not shed"


@pytest.mark.asyncio
async def test_waterfall_escalates_while_self_shedding():
    """Insufficient peers ⇒ the cold load sheds itself (safety valve) AND
    still requests what the lower tier can give."""
    behavior = MockBehavior()
    world = create_world()
    m0 = _heat_monitor(behavior, "agent-0", priority=1, t_k=300.0)
    m1 = _heat_monitor(behavior, "agent-1", priority=4, t_k=330.0)
    for i, m in enumerate([m0, m1]):
        agent = world.register(RoleAgent(), suggested_aid=f"agent-{i}")
        agent.add_role(m)

    async with world:
        m0._sens._value = 660.0
        # needed = 0.03 MW; peer holds only 0.002 -> insufficient to defer,
        # but above the 1e-3 targeting floor.
        _seed_peer(m0, m1, tier=4, reducible=0.002)
        await m0._heat_frontier_control()
        await step_simulation(world, step_size_s=1.0)

    own = [a for a in behavior.action_log if a[0] == "agent-0" and a[1] == "regulate"]
    assert own and own[-1][2][0] < 1.0, "must shed self on insufficient peers"
    peer = [a for a in behavior.action_log if a[0] == "agent-1" and a[1] == "regulate"]
    assert peer, "escalation request must still reach the lower-priority peer"


@pytest.mark.asyncio
async def test_fully_shed_cold_load_still_requests_peers():
    """regulation=0 leaves nothing to shed locally; the peer request is the
    only lever and must keep firing while the node is cold."""
    behavior = MockBehavior()
    world = create_world()
    m0 = _heat_monitor(behavior, "agent-0", priority=1, t_k=300.0)
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.2, "regulation": 0.0, "t_k": 300.0, "priority": 1},
    )
    m1 = _heat_monitor(behavior, "agent-1", priority=4, t_k=330.0)
    for i, m in enumerate([m0, m1]):
        agent = world.register(RoleAgent(), suggested_aid=f"agent-{i}")
        agent.add_role(m)

    async with world:
        m0._sens._value = 660.0
        _seed_peer(m0, m1, tier=4, reducible=0.05)
        await m0._heat_frontier_control()
        await step_simulation(world, step_size_s=1.0)

    peer = [a for a in behavior.action_log if a[0] == "agent-1" and a[1] == "regulate"]
    assert peer, "fully-shed cold load stopped requesting peer sheds"


@pytest.mark.asyncio
async def test_constraint_state_carries_component_id_into_cache():
    behavior = MockBehavior()
    world = create_world()
    monitor = _heat_monitor(behavior, "agent-0", priority=1, t_k=300.0, component_id=2)
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    async with world:
        msg = ConstraintStateMessage(
            sector=Sector.HEAT,
            variable="t_k",
            value=300.0,
            utilization=1.2,
            hops_remaining=1,
            origin_addr="peer-9",
            priority_tier=4,
            reducible=0.03,
            component_id=5,
        )
        await monitor._handle_constraint_state(
            msg, {"sender_addr": "peer-sender", "sender_id": "s0"}
        )
    _t, tier, reducible, component = monitor._heat_frontier._peer_state["peer-9"]
    assert (tier, reducible, component) == (4, pytest.approx(0.03), 5)
    # Foreign component -> never a waterfall partner of this monitor.
    assert monitor._heat_frontier.waterfall_request_targets(1, 0.0) == []
