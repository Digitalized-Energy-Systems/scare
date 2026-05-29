"""Component tests for GridConstraintMonitor role."""

import pytest

from mango import RoleAgent, create_world
from mango.simulation.world import step_simulation

from scare.base.model import ConstraintViolation, ConstraintWarning, Sector
from scare.service.constraints import GridConstraintMonitor
from tests.conftest import MockBehavior


def _make_monitor(
    behavior: MockBehavior,
    aid: str = "agent-0",
    sector: Sector = Sector.ELECTRICITY,
    vm_pu: float = 1.0,
    max_hops: int = 1,
) -> GridConstraintMonitor:
    behavior.set_obs(aid, {"p_mw": 5.0, "vm_pu": vm_pu})
    behavior.add_action(aid, "regulate")
    return GridConstraintMonitor(behavior, sector, node_id=0, max_hops=max_hops)


@pytest.mark.asyncio
async def test_no_event_within_bounds():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)  # center of [0.95, 1.05]

    events = []
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    # Capture events by adding a listener role
    class EventCapture:
        pass

    async with world:
        await step_simulation(world, step_size_s=1.0)

    assert monitor.is_locally_feasible()


@pytest.mark.asyncio
async def test_violation_emitted_when_out_of_bounds():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.06)  # above 1.05

    violations = []

    class ViolationCapture(RoleAgent):
        def handle_message(self, content, meta):
            pass

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    # Subscribe to violation events on the same agent
    from mango.agent.role import Role

    class Listener(Role):
        def setup(self):
            self.context.subscribe_event(
                self, ConstraintViolation, self._on_violation
            )

        def _on_violation(self, event, src):
            violations.append(event)

    listener = Listener()
    agent.add_role(listener)

    async with world:
        # Advance past the poll period (0.5s for electricity)
        await step_simulation(world, step_size_s=1.0)

    assert len(violations) >= 1
    assert violations[0].variable == "vm_pu"
    assert violations[0].value == 1.06
    assert not monitor.is_locally_feasible()


@pytest.mark.asyncio
async def test_warning_emitted_near_bound():
    # vm_pu=1.04 => util = |1.04 - 1.0| / 0.05 = 0.8 < 1.0 (no violation)
    # but > PROACTIVE_WARNING_FRACTION (0.85)?  0.8 < 0.85, so no warning.
    # vm_pu=1.044 => util = 0.88 > 0.85 => warning
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.044)

    warnings = []

    from mango.agent.role import Role

    class Listener(Role):
        def setup(self):
            self.context.subscribe_event(
                self, ConstraintWarning, self._on_warning
            )

        def _on_warning(self, event, src):
            warnings.append(event)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    agent.add_role(Listener())

    async with world:
        await step_simulation(world, step_size_s=1.0)

    assert len(warnings) >= 1
    assert warnings[0].variable == "vm_pu"
    assert warnings[0].utilization > 0.85


@pytest.mark.asyncio
async def test_is_locally_feasible_default():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        pass

    assert monitor.is_locally_feasible()


@pytest.mark.asyncio
async def test_local_sensitivity_ema_update():
    """Sensitivity estimator should converge via EMA from repeated (P, V) samples."""
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    initial = monitor.local_sensitivity()

    async with world:
        # First sample establishes baseline; subsequent samples move the EMA.
        for p, v in [(5.0, 1.00), (6.0, 1.02), (7.0, 1.04), (8.0, 1.06)]:
            behavior.set_obs("agent-0", {"p_mw": p, "vm_pu": v})
            await step_simulation(world, step_size_s=1.0)

    # Each |ΔV/ΔP| = 0.02 / 1.0 = 0.02; EMA should move toward this.
    assert monitor.local_sensitivity() != initial
    assert monitor.local_sensitivity() > 0.0


@pytest.mark.asyncio
async def test_heat_sensitivity_updates_for_mw_scale_dp():
    """Heat sensitivity must update for MW-scale setpoint deltas.

    Regression for the ``_SENSITIVITY_MIN_DP[HEAT]`` unit mismatch: it was
    0.5 (as if P were in W), but heat ``obs_setpoint`` is ``q_mw_heat`` in
    MW (~0.0075–0.05), so no regulation step could ever exceed it — the
    estimate stayed pinned at the 1e-5 default and the curtailment-auction
    willingness lost its sensitivity term entirely.
    """
    behavior = MockBehavior()
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    initial = monitor.local_sensitivity()

    async with world:
        # setpoint = q_mw_heat * regulation; each step moves P by ~0.02 MW
        # (≫ the 5e-4 MW floor) and t_k by a few K -> the EMA must move.
        for reg, t in [(1.0, 360.0), (0.6, 366.0), (0.3, 372.0)]:
            behavior.set_obs(
                "agent-0", {"q_mw_heat": 0.05, "regulation": reg, "t_k": t}
            )
            await step_simulation(world, step_size_s=6.0)

    assert monitor.local_sensitivity() != initial
    assert monitor.local_sensitivity() > 0.0


@pytest.mark.asyncio
async def test_curtail_willingness_sensitivity_is_bounded():
    """The sensitivity multiplier in the auction willingness is clamped to
    ``[_SENS_MULT_MIN, _SENS_MULT_MAX]`` so it ranks within a tier but can
    never overcome the 1e4 priority tier step (no waterfall inversion)."""
    from scare.service.constraints import _SENS_MULT_MAX, _SENS_MULT_MIN

    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.04, "regulation": 1.0, "t_k": 360.0, "priority": 4},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        obs = behavior.observe("agent-0")
        monitor._sensitivity = 1e9   # absurdly high
        w_hi = monitor._own_curtail_willingness(obs)
        monitor._sensitivity = 1e-9  # absurdly low
        w_lo = monitor._own_curtail_willingness(obs)

    # Same tier + reducible in both calls, so the ratio is purely the
    # clamped sensitivity-multiplier span (16x), not the raw 1e18.
    assert w_hi / w_lo == pytest.approx(_SENS_MULT_MAX / _SENS_MULT_MIN)


@pytest.mark.asyncio
async def test_heat_frontier_sheds_to_partial_not_zero():
    """A cold heat node is curtailed to a PARTIAL feasible frontier, not
    bang-bang to 0: with a learned dT/dreg sensitivity the frontier
    controller takes a proportional step toward t_k = floor + margin."""
    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 305.0, "priority": 3},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        monitor._sensitivity = 660.0  # learned dT/dP (K per MW): frontier ~0.8
        await monitor._heat_frontier_control()

    regs = [a for a in behavior.action_log if a[1] == "regulate"]
    assert regs, "frontier controller should have written a regulate"
    factor = regs[-1][2][0]
    assert 0.0 < factor < 1.0  # partial frontier, not bang-bang 0


@pytest.mark.asyncio
async def test_heat_frontier_applies_to_tier1():
    """Tier-1 heat is NOT exempt from the frontier controller — a critical
    heat load at an infeasible temperature is curtailed to its feasible
    partial (serving >0 beats the barrier crediting 0 at full draw)."""
    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 300.0, "priority": 1},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        monitor._sensitivity = 660.0
        await monitor._heat_frontier_control()

    regs = [a for a in behavior.action_log if a[1] == "regulate"]
    assert regs and regs[-1][2][0] < 1.0  # tier-1 heat was curtailed


@pytest.mark.asyncio
async def test_heat_frontier_holds_in_band():
    """A feasible heat node comfortably inside the hold band is not touched."""
    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 318.0, "priority": 3},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        monitor._sensitivity = 660.0
        await monitor._heat_frontier_control()

    assert not [a for a in behavior.action_log if a[1] == "regulate"]


@pytest.mark.asyncio
async def test_worst_neighbour_utilization_default():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        pass

    assert monitor.worst_neighbour_utilization() == 0.0
