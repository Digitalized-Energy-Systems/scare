"""Unit tests for the L2.5 cross-sector coalition extension.

Drives :class:`HolonSummaryRole`'s cross-sector detection + allocation
path with a stub mango context to verify the
``enable_cross_sector_coalitions`` knob:

* Flag off: an injected cross-sector inversion (electricity tier-1 at
  30 % while heat tier-5 is at 100 %, with a P2H between them) produces
  no ``CPCommitment`` and no cross-sector ``StartBalanceNegotiation``.
* Flag on: the same inversion produces one ``CPCommitment`` to the P2H
  plus the per-sector service-fraction commits, both registered in the
  shared :class:`CoalitionConstraintStore` so L2/L3 see them.

No real ADMM solve runs — the path uses the greedy priority-aware
allocator (deterministic transfer from CP rated capacity + coupling
ratio).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from scare.base.channel import (
    CPCommitment,
    HolonSummary,
    MonotonicVersion,
)
from scare.base.model import Sector, StartBalanceNegotiation
from scare.community.coalition_store import CoalitionConstraintStore
from scare.community.summary import HolonSummaryRole, _xs_registry


# ---------------------------------------------------------------------------
# Fake mango context: records outbound messages, carries an aid + sim
# clock. No scheduler — async role methods are driven via asyncio.run.
# ---------------------------------------------------------------------------


@dataclass
class _SentMessage:
    payload: Any
    receiver_addr: Any


class _Addr:
    """Lightweight ``addr`` substitute with an ``aid`` attribute."""

    def __init__(self, aid: str) -> None:
        self.aid = aid

    def __repr__(self) -> str:  # pragma: no cover
        return f"<addr {self.aid}>"


class _FakeContext:
    def __init__(self, aid: str, t: float = 100.0) -> None:
        self.aid = aid
        self.addr = _Addr(aid)
        self.current_timestamp = t
        self.sent: list[_SentMessage] = []

    async def send_message(self, payload: Any, *, receiver_addr: Any) -> None:
        self.sent.append(_SentMessage(payload=payload, receiver_addr=receiver_addr))

    # Capture the instant task so the test can await it deterministically.
    def schedule_instant_task(self, coro: Any) -> None:
        self._pending = coro

    def schedule_periodic_task(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Helpers to build the synthetic cross-sector inversion fixture
# ---------------------------------------------------------------------------


def _make_summary(
    aid: str,
    sector: Sector,
    *,
    demand_by_tier: dict[int, float],
    served_by_tier: dict[int, float],
    version: int = 1,
) -> HolonSummary:
    return HolonSummary(
        publisher=aid,
        version=version,
        caused_by={},
        timestamp_s=0.0,
        sector=sector,
        per_tier_served_mw=dict(served_by_tier),
        per_tier_demand_mw=dict(demand_by_tier),
    )


def _build_role(
    behavior: Any,
    aid: str,
    sector: Sector,
    *,
    enable_cross_sector_coalitions: bool,
    cp_meta: dict[str, dict[str, Any]] | None = None,
    peer_leader_addrs: dict[Sector, dict[str, Any]] | None = None,
    constraint_store: CoalitionConstraintStore | None = None,
) -> HolonSummaryRole:
    role = HolonSummaryRole(
        behavior,
        sector,
        period_s=1.0,
        inversion_tol=1e-3,
        enable_coalition=True,
        coalition_constraint_ttl_s=4.0,
        priority_tiers=10,
        constraint_store=constraint_store,
        enable_cross_sector_coalitions=enable_cross_sector_coalitions,
        cp_meta=cp_meta or {},
        peer_leader_addrs=peer_leader_addrs or {},
    )
    role._context = _FakeContext(aid)
    # Force cooldown into the past so the first tick may fire.
    role._last_xs_inversion_emit_t = -1e9
    role._inversion_cooldown_s = 0.0
    return role


def _inject_inversion(behavior: Any) -> None:
    """Populate the shared registry: electricity tier-1 at 30 % served,
    heat tier-5 at 100 % — the inversion the coalition should fix.
    """
    registry = _xs_registry(behavior)
    registry[Sector.ELECTRICITY] = {
        "leader-el-1": _make_summary(
            "leader-el-1", Sector.ELECTRICITY,
            demand_by_tier={1: 2.0},
            served_by_tier={1: 0.6},  # 30 % served
        ),
    }
    registry[Sector.HEAT] = {
        "leader-heat-1": _make_summary(
            "leader-heat-1", Sector.HEAT,
            demand_by_tier={5: 1.0},
            served_by_tier={5: 1.0},  # 100 % served
        ),
    }


def _p2h_meta(cp_addr: _Addr) -> dict[str, dict[str, Any]]:
    """Single CP bridging heat → electricity. The role picks the
    direction that pushes into the under-served sector (electricity), so
    the coupling ratio is keyed ``(heat, electricity)``. η = 0.5 keeps
    the expected transfer round-numbered.
    """
    return {
        "p2h-1": {
            "sectors": [Sector.ELECTRICITY, Sector.HEAT],
            # Keyed (in_sector, out_sector): push into electricity from heat.
            "coupling_ratios": {("heat", "electricity"): 0.5},
            "rated_capacity_mw": {"electricity": 1.0, "heat": 1.0},
            "addr": cp_addr,
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossSectorCoalitionFlagDisabled:
    """Flag off: the inversion produces no cross-sector dispatch; only
    the intra-sector detector runs (and sees no inversion here).
    """

    def test_no_dispatch_when_disabled(self) -> None:
        behavior = SimpleNamespace()
        _inject_inversion(behavior)
        cp_addr = _Addr("p2h-1")
        store = CoalitionConstraintStore()
        role = _build_role(
            behavior,
            "leader-el-1",
            Sector.ELECTRICITY,
            enable_cross_sector_coalitions=False,
            cp_meta=_p2h_meta(cp_addr),
            constraint_store=store,
        )
        # _tick skips the cross-sector branch when the flag is False.
        if role.enable_cross_sector_coalitions:
            role._check_cross_sector_invariants()
        assert not role.context.sent
        assert not role._active_xs_coalitions
        assert not store._cp_envelopes


class TestCrossSectorCoalitionFlagEnabled:
    """Flag on: the inversion fires a coalition. Asserts the
    CPCommitment is dispatched to the P2H with greedy-allocated
    ``target_flows_mw``, two per-sector ``StartBalanceNegotiation``
    payloads carry the raised/reduced fractions, and the shared store
    carries both the tier records and the CP envelope.
    """

    def test_dispatch_when_enabled(self) -> None:
        behavior = SimpleNamespace()
        _inject_inversion(behavior)
        cp_addr = _Addr("p2h-1")
        store = CoalitionConstraintStore()
        # Peer-sector leader address for the heat-side dispatch.
        peer_addrs = {
            Sector.HEAT: {"leader-heat-1": _Addr("leader-heat-1")},
        }
        role = _build_role(
            behavior,
            "leader-el-1",
            Sector.ELECTRICITY,
            enable_cross_sector_coalitions=True,
            cp_meta=_p2h_meta(cp_addr),
            peer_leader_addrs=peer_addrs,
            constraint_store=store,
        )
        # The detector schedules an instant task; the fake context
        # captures it. Drive it to completion to observe the dispatches.
        role._check_cross_sector_invariants()
        pending = getattr(role.context, "_pending", None)
        assert pending is not None, "cross-sector detector did not schedule a task"
        asyncio.run(pending)

        # ---- CPCommitment was dispatched -------------------------------
        cp_msgs = [
            m for m in role.context.sent if isinstance(m.payload, CPCommitment)
        ]
        assert len(cp_msgs) == 1
        cp_commit = cp_msgs[0].payload
        assert cp_msgs[0].receiver_addr is cp_addr
        assert cp_commit.cp_id == "p2h-1"
        # Greedy allocation:
        #   deficit_el_tier1 = 2 - 0.6 = 1.4
        #   peer_freeable    = served_heat_tier5 * eta = 1.0 * 0.5 = 0.5
        #   cp_cap_out       = 1.0
        # transfer_out = min(1.4, 0.5, 1.0) = 0.5  (electricity, +)
        # transfer_in  = 0.5 / 0.5 = 1.0           (heat, -)
        flows = cp_commit.target_flows_mw
        assert flows["electricity"] == pytest.approx(0.5)
        assert flows["heat"] == pytest.approx(-1.0)
        assert cp_commit.ttl_s == pytest.approx(4.0)

        # ---- Per-sector StartBalanceNegotiation was dispatched ---------
        sb_msgs = [
            m for m in role.context.sent
            if isinstance(m.payload, StartBalanceNegotiation)
        ]
        # One dispatch per sector (heat leader + own electricity leader).
        sectors_dispatched = set()
        for m in sb_msgs:
            frac_map = m.payload.service_fraction_by_sector_priority or {}
            sectors_dispatched.update(frac_map.keys())
        assert "electricity" in sectors_dispatched
        assert "heat" in sectors_dispatched
        # Heat tier-5 fraction = (1.0 - 1.0) / 1.0 = 0.0
        # Electricity tier-1 fraction = (0.6 + 0.5) / 2.0 = 0.55
        heat_frac = None
        el_frac = None
        for m in sb_msgs:
            fm = m.payload.service_fraction_by_sector_priority or {}
            if "electricity" in fm:
                el_frac = fm["electricity"].get(1)
            if "heat" in fm:
                heat_frac = fm["heat"].get(5)
        assert el_frac == pytest.approx(0.55)
        assert heat_frac == pytest.approx(0.0)

        # ---- Coalition store written ----------------------------------
        env = store.cp_envelope_for("p2h-1", now=100.0)
        assert env is not None
        assert env["electricity"] == pytest.approx(0.5)
        assert env["heat"] == pytest.approx(-1.0)
        assert store.has_active_cp_envelope("p2h-1", now=100.0)
        # Envelope is time-bounded — gone after TTL, which stops
        # oscillation against the underlying L2/L3 paths.
        assert not store.has_active_cp_envelope("p2h-1", now=200.0)

        assert len(role._active_xs_coalitions) == 1

    def test_branch_failure_invalidates(self) -> None:
        """A branch failure event drops the cross-sector coalition + CP
        envelope so the post-failure topology is free to redecide.
        """
        behavior = SimpleNamespace()
        _inject_inversion(behavior)
        cp_addr = _Addr("p2h-1")
        store = CoalitionConstraintStore()
        peer_addrs = {Sector.HEAT: {"leader-heat-1": _Addr("leader-heat-1")}}
        role = _build_role(
            behavior,
            "leader-el-1",
            Sector.ELECTRICITY,
            enable_cross_sector_coalitions=True,
            cp_meta=_p2h_meta(cp_addr),
            peer_leader_addrs=peer_addrs,
            constraint_store=store,
        )
        role._check_cross_sector_invariants()
        pending = role.context._pending
        asyncio.run(pending)
        assert role._active_xs_coalitions
        assert store.has_active_cp_envelope("p2h-1", now=100.0)

        role.on_branch_failure(("b1", 0))

        assert not role._active_xs_coalitions
        assert not store.has_active_cp_envelope("p2h-1", now=100.0)


class TestCrossSectorCoalitionFlagSideBySide:
    """The flag must produce different observable behaviour for the same
    inputs.
    """

    def test_flag_changes_dispatch_count(self) -> None:
        behavior_off = SimpleNamespace()
        behavior_on = SimpleNamespace()
        _inject_inversion(behavior_off)
        _inject_inversion(behavior_on)

        cp_addr = _Addr("p2h-1")
        store_off = CoalitionConstraintStore()
        store_on = CoalitionConstraintStore()

        role_off = _build_role(
            behavior_off,
            "leader-el-1",
            Sector.ELECTRICITY,
            enable_cross_sector_coalitions=False,
            cp_meta=_p2h_meta(cp_addr),
            constraint_store=store_off,
        )
        role_on = _build_role(
            behavior_on,
            "leader-el-1",
            Sector.ELECTRICITY,
            enable_cross_sector_coalitions=True,
            cp_meta=_p2h_meta(cp_addr),
            peer_leader_addrs={
                Sector.HEAT: {"leader-heat-1": _Addr("leader-heat-1")},
            },
            constraint_store=store_on,
        )

        if role_off.enable_cross_sector_coalitions:
            role_off._check_cross_sector_invariants()
        role_on._check_cross_sector_invariants()
        if hasattr(role_on.context, "_pending"):
            asyncio.run(role_on.context._pending)

        cp_msgs_off = [m for m in role_off.context.sent
                       if isinstance(m.payload, CPCommitment)]
        cp_msgs_on = [m for m in role_on.context.sent
                      if isinstance(m.payload, CPCommitment)]
        assert len(cp_msgs_off) == 0
        assert len(cp_msgs_on) == 1
