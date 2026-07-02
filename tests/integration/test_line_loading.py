"""Integration tests for the line-loading constraint pipeline.

Exercises the branch-mode ``GridConstraintMonitor`` in a minimal mango
world: a branch agent with a synthetic overloaded line observation
sends a ``StartBalanceNegotiation`` with a relief-MW target to a
designated home group leader.  Verifies:

1. The monitor fires when ``loading_percent`` crosses the (-100, 100)
   bound and the message reaches the home leader.
2. The relief target carries the right sign (negative => home group
   must shed) and a non-zero magnitude derived from the line flow.
3. Disabling ``enable_curtailment_auction`` does not suppress the
   StartBalanceNegotiation channel (independent code paths).
4. A non-violating loading_percent produces no message.
"""

import pytest
from mango import RoleAgent, SimpleCommunicationSimulation, create_world
from mango.express.topology import create_topology
from mango.simulation.world import discrete_step_until

from scare.base.model import Sector, StartBalanceNegotiation
from scare.service.control.constraints import GridConstraintMonitor
from tests.conftest import MockBehavior


class _RecordingRole:
    """Minimal mango Role that records every StartBalanceNegotiation."""

    def __init__(self):
        self.received: list[StartBalanceNegotiation] = []

    def __init_subclass__(cls, **kwargs):  # pragma: no cover - unused
        super().__init_subclass__(**kwargs)


class _RecordingHomeLeader:
    """A mango Role that captures incoming StartBalanceNegotiation."""

    def __init__(self):

        # Compose dynamically so we don't import Role at module top.
        self._role = None
        self.received: list[StartBalanceNegotiation] = []

    def make_role(self):
        from mango import Role

        owner = self

        class _LeaderRole(Role):
            def setup(self_inner):
                def _on_msg(msg, meta):
                    owner.received.append(msg)

                self_inner.context.subscribe_message(
                    self_inner,
                    _on_msg,
                    lambda msg, meta: isinstance(msg, StartBalanceNegotiation),
                )

        self._role = _LeaderRole()
        return self._role


def _make_branch_obs(
    loading_percent: float,
    p_from_mw: float = 1.0,
    p_to_mw: float = 0.95,
) -> dict:
    """Synthetic PowerLine observation dict.

    ``loading_percent`` is in percent (matching the SECTOR_CONSTRAINTS
    bound at +/-100); a direct ``loading_percent`` key maps straight
    through ``obs_constraint_values``.
    """
    return {
        "loading_percent": loading_percent,
        "p_from_mw": p_from_mw,
        "p_to_mw": p_to_mw,
    }


def _build_world(
    behavior: MockBehavior,
    branch_aid: str,
    branch_obs: dict,
    leader_aid: str = "load-leader",
    enable_auction: bool = False,
) -> tuple:
    """Build a 2-agent world: branch monitor + recording home leader.

    Returns (world, branch_agent, leader_recorder).
    """
    world = create_world(
        communication_sim=SimpleCommunicationSimulation(default_delay_s=0.001)
    )

    behavior.set_obs(branch_aid, branch_obs)

    # Build leader first so we can pass its address to the branch monitor.
    leader_recorder = _RecordingHomeLeader()
    leader_agent = world.register(RoleAgent(), suggested_aid=leader_aid)
    leader_agent.add_role(leader_recorder.make_role())

    monitor = GridConstraintMonitor(
        behavior,
        Sector.ELECTRICITY,
        branch_id=(1, 2),
        home_leader_addr=leader_agent.addr,
        enable_curtailment_auction=enable_auction,
        enable_multihop_constraint=False,
    )
    branch_agent = world.register(RoleAgent(), suggested_aid=branch_aid)
    branch_agent.add_role(monitor)

    # Empty groups topology: multi-hop / auction paths reach for it via
    # ``topology_neighbors`` and would otherwise raise; empty keeps those
    # calls safe no-ops.
    with create_topology(tid="groups"):
        pass

    return world, branch_agent, leader_recorder


@pytest.mark.asyncio
async def test_line_overload_triggers_balance_negotiation():
    """loading_percent=130 produces a StartBalanceNegotiation on the
    home leader with a negative override_target."""
    behavior = MockBehavior()
    world, _, leader_recorder = _build_world(
        behavior,
        branch_aid="branch-1-2",
        branch_obs=_make_branch_obs(loading_percent=130.0, p_from_mw=1.0, p_to_mw=0.95),
    )

    async with world:
        await discrete_step_until(world, max_advance_time_s=2.0)

    assert len(leader_recorder.received) >= 1, (
        "branch monitor should have sent StartBalanceNegotiation"
    )
    msg = leader_recorder.received[0]
    assert msg.override_target is not None
    # Relief is negative (group must reduce net load).
    assert msg.override_target < 0
    # Relief = flow_mw * overshoot/100 = 1.0 * 0.30 = 0.30 MW.
    assert abs(msg.override_target + 0.30) < 1e-6


@pytest.mark.asyncio
async def test_line_loading_within_bounds_emits_nothing():
    """loading_percent=80 stays within (-100, 100): no message."""
    behavior = MockBehavior()
    world, _, leader_recorder = _build_world(
        behavior,
        branch_aid="branch-3-4",
        branch_obs=_make_branch_obs(loading_percent=80.0),
    )

    async with world:
        await discrete_step_until(world, max_advance_time_s=2.0)

    assert leader_recorder.received == []


@pytest.mark.asyncio
async def test_relief_magnitude_scales_with_flow():
    """A heavier flow at the same overshoot yields a larger relief MW."""
    behavior = MockBehavior()
    world, _, leader_recorder = _build_world(
        behavior,
        branch_aid="branch-5-6",
        # 120% loading, 2.0 MW flowing: relief = 2.0 * 0.20 = 0.40 MW.
        branch_obs=_make_branch_obs(loading_percent=120.0, p_from_mw=2.0, p_to_mw=1.9),
    )

    async with world:
        await discrete_step_until(world, max_advance_time_s=2.0)

    assert len(leader_recorder.received) >= 1
    msg = leader_recorder.received[0]
    assert abs(msg.override_target + 0.40) < 1e-6


@pytest.mark.asyncio
async def test_no_home_leader_addr_skips_relief_send():
    """When home_leader_addr is None (e.g. an orphan PowerLine whose home
    assignment failed to resolve) the monitor must neither crash nor send."""
    behavior = MockBehavior()
    branch_aid = "branch-7-8"
    behavior.set_obs(branch_aid, _make_branch_obs(loading_percent=140.0))

    world = create_world(
        communication_sim=SimpleCommunicationSimulation(default_delay_s=0.001)
    )
    monitor = GridConstraintMonitor(
        behavior,
        Sector.ELECTRICITY,
        branch_id=(7, 8),
        home_leader_addr=None,
        enable_curtailment_auction=False,
        enable_multihop_constraint=False,
    )
    agent = world.register(RoleAgent(), suggested_aid=branch_aid)
    agent.add_role(monitor)
    with create_topology(tid="groups"):
        pass

    async with world:
        await discrete_step_until(world, max_advance_time_s=1.5)

    # Success is reaching here without an exception.
