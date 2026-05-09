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
async def test_worst_neighbour_utilization_default():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        pass

    assert monitor.worst_neighbour_utilization() == 0.0
