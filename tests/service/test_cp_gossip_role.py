"""Tests for the gossip-mode wiring of :class:`CPPriorityAdmmRole`.

Two contract properties are pinned:

1. **Initiator gate** — under stable topology, only one CP per
   cross-sector connected component fires a round per tick, using the
   lowest reachable cp_id as the deterministic initiator.
2. **Commit callback** — when the gossip participant resolves a round,
   the role invokes :func:`apply_regulate` exactly once with the CP's
   own factor.

The role runs directly against a fake mango context (no agent world).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from scare.base.channel import CPSummary
from scare.base.model import Sector
from scare.service.coupling.cp_priority_admm_role import CPPriorityAdmmRole

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


@dataclass
class _MockBehavior:
    """Just enough of the RestorationEnvironmentBehavior surface for
    :func:`apply_regulate` and ``has_action`` checks."""

    actions: dict[str, set[str]] = field(default_factory=dict)
    obs: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_log: list[tuple[str, str, float]] = field(default_factory=list)

    def add_action(self, aid: str, kind: str) -> None:
        self.actions.setdefault(aid, set()).add(kind)

    def set_obs(self, aid: str, obs: dict[str, Any]) -> None:
        self.obs[aid] = dict(obs)

    def has_action(self, aid: str, kind: str) -> bool:
        return kind in self.actions.get(aid, set())

    def observe(self, aid: str) -> dict[str, Any] | None:
        return self.obs.get(aid)

    def act(self, aid: str, kind: str, value: float) -> None:
        self.action_log.append((aid, kind, float(value)))


class _Addr:
    def __init__(self, aid: str) -> None:
        self.aid = aid


class _FakeContext:
    def __init__(self, aid: str, t: float = 100.0) -> None:
        self.aid = aid
        self.addr = _Addr(aid)
        self.current_timestamp = t
        self.sent: list[tuple[Any, Any]] = []

    async def send_message(self, payload: Any, *, receiver_addr: Any) -> None:
        self.sent.append((payload, receiver_addr))

    def schedule_instant_task(self, coro: Any) -> None:
        # Tests await coroutines manually.
        pass

    def schedule_periodic_task(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def subscribe_message(self, *_args: Any, **_kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_role(
    cp_id: str,
    *,
    capacity_by_sector: dict[str, float],
    bridged_sectors: list[Sector],
    peer_addrs: dict[str, Any],
    peer_node_ids: dict[str, Any] | None = None,
) -> tuple[CPPriorityAdmmRole, _FakeContext, _MockBehavior]:
    behavior = _MockBehavior()
    behavior.add_action(cp_id, "regulate")
    behavior.set_obs(cp_id, {"regulation": 0.0})
    role = CPPriorityAdmmRole(
        behavior=behavior,
        cp_id=cp_id,
        capacity_by_sector=capacity_by_sector,
        bridged_sectors=bridged_sectors,
        rebalance_min_gap_s=0.0,
        admm_max_iters=50,
        algorithm="gossip",
    )
    ctx = _FakeContext(cp_id)
    role._context = ctx  # type: ignore[attr-defined]
    # Populate peer state directly so the initiator gate evaluates
    # against a known reachable set (bypasses the gossiped CPSummary path).
    for peer_id in peer_addrs:
        role._peer_cps[peer_id] = CPSummary(
            publisher=peer_id,
            version=1,
            caused_by={},
            timestamp_s=0.0,
            capacity_by_sector={},
            home_node_id=None,
        )
    role._peer_cp_addrs = dict(peer_addrs)
    role._peer_cp_node_ids = dict(peer_node_ids or {})
    return role, ctx, behavior


# ---------------------------------------------------------------------------
# Initiator gate
# ---------------------------------------------------------------------------


def test_initiator_is_lowest_cp_id_among_reachable_peers():
    """Only the lowest reachable cp_id evaluates to initiator."""
    addrs = {"cp-002": _Addr("cp-002"), "cp-003": _Addr("cp-003")}
    role_a, _, _ = _make_role(
        "cp-001",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
        bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        peer_addrs=addrs,
    )
    addrs_b = {"cp-001": _Addr("cp-001"), "cp-003": _Addr("cp-003")}
    role_b, _, _ = _make_role(
        "cp-002",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
        bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        peer_addrs=addrs_b,
    )
    assert role_a._am_gossip_initiator() is True
    assert role_b._am_gossip_initiator() is False


def test_only_one_initiator_per_tick_across_a_full_mesh():
    """Run the gate on every CP in a 3-CP component; exactly one
    returns True."""
    cp_ids = ["cp-x", "cp-y", "cp-z"]
    roles = []
    for cp in cp_ids:
        peer_addrs = {p: _Addr(p) for p in cp_ids if p != cp}
        role, _, _ = _make_role(
            cp,
            capacity_by_sector={"electricity": 1.0, "heat": -1.0},
            bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
            peer_addrs=peer_addrs,
        )
        roles.append(role)
    initiators = [r for r in roles if r._am_gossip_initiator()]
    assert len(initiators) == 1
    assert initiators[0].cp_id == min(cp_ids)


def test_initiator_handover_when_lowest_cp_dies():
    """Dropping the previous initiator from the peer set promotes the
    next-lowest cp_id, with no handover protocol."""
    addrs = {"cp-x": _Addr("cp-x"), "cp-z": _Addr("cp-z")}
    role, _, _ = _make_role(
        "cp-y",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
        bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        peer_addrs=addrs,
    )
    assert role._am_gossip_initiator() is False  # cp-x is lower
    role._peer_cps.pop("cp-x")  # cp-x dies
    role._peer_cp_addrs.pop("cp-x")
    assert role._am_gossip_initiator() is True  # cp-y is now lowest


# ---------------------------------------------------------------------------
# Commit callback
# ---------------------------------------------------------------------------


def test_commit_callback_writes_apply_regulate_once():
    """When the participant resolves a round, the role's on_commit
    hook fires apply_regulate for this CP only."""
    role, ctx, behavior = _make_role(
        "cp-001",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
        bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        peer_addrs={},
    )
    role._on_gossip_commit(
        r=np.array([0.62]),
        converged=True,
        iterations=12,
    )
    assert len(behavior.action_log) == 1
    aid, kind, value = behavior.action_log[0]
    assert aid == "cp-001"
    assert kind == "regulate"
    assert value == pytest.approx(0.62, abs=1e-6)


def test_commit_callback_clamps_to_zero_one():
    """Out-of-range r is clamped before apply_regulate (defensive)."""
    role, _, behavior = _make_role(
        "cp-001",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
        bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        peer_addrs={},
    )
    role._on_gossip_commit(
        r=np.array([1.4]),
        converged=False,
        iterations=200,
    )
    role._on_gossip_commit(
        r=np.array([-0.1]),
        converged=False,
        iterations=200,
    )
    assert behavior.action_log[0][2] == pytest.approx(1.0)
    assert behavior.action_log[1][2] == pytest.approx(0.0)


def test_commit_skipped_on_empty_factor():
    """A degenerate empty factor is silently skipped, no apply_regulate."""
    role, _, behavior = _make_role(
        "cp-001",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
        bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        peer_addrs={},
    )
    role._on_gossip_commit(
        r=np.array([]),
        converged=True,
        iterations=0,
    )
    assert behavior.action_log == []


# ---------------------------------------------------------------------------
# Round-id monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_gossip_round_bumps_round_id():
    """The initiator bumps its round_id every time it fires."""
    role, _, _ = _make_role(
        "cp-001",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
        bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        peer_addrs={},
    )
    # Manually install participant + carrier (setup() not invoked).
    from distributed_resource_optimization.algorithm.admm.lexicographic import (
        create_gossip_cascade_participant,
    )

    from scare.service.coupling.cp_priority_admm_role import _ReachableCPCarrier

    role._gossip_carrier = _ReachableCPCarrier(role)
    role._gossip_participant = create_gossip_cascade_participant(
        cp_id="cp-001",
        capacity_by_sector=role.capacity_by_sector,
        on_commit=role._on_gossip_commit,
    )

    # No demands ⇒ early-return without bumping (we want to assert the
    # bump only happens on a real round).  Force a demand source.
    role._leader_summaries[Sector.ELECTRICITY.value]["leader-el"] = type(
        "FakeHS",
        (),
        {
            "version": 1,
            "supply_by_sector": {Sector.ELECTRICITY.value: 5.0},
            "demand_by_sector_priority": {Sector.ELECTRICITY.value: {1: 1.0}},
            "served_by_sector_priority": None,
            "slack_budget_by_sector": None,
        },
    )()
    role._leader_summaries[Sector.HEAT.value]["leader-heat"] = type(
        "FakeHS",
        (),
        {
            "version": 1,
            "supply_by_sector": {Sector.HEAT.value: 5.0},
            "demand_by_sector_priority": {Sector.HEAT.value: {1: 1.0}},
            "served_by_sector_priority": None,
            "slack_budget_by_sector": None,
        },
    )()

    assert role._gossip_round_id == 0
    await role._run_gossip_round()
    assert role._gossip_round_id == 1
    # Give the cascade time to wind down so we don't double-run.
    await asyncio.sleep(0.05)
    await role._run_gossip_round()
    assert role._gossip_round_id == 2
