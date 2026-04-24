"""Component tests for GenerationController role."""

import pytest

from mango import RoleAgent, create_world
from mango.simulation.world import step_simulation

from scare.base.model import NegotiationFinishedEvent, Sector
from scare.service.stability import GenerationController
from tests.conftest import MockBehavior, make_electricity_gen


def _setup_controller(
    behavior: MockBehavior,
    aid: str = "agent-0",
    sector: Sector = Sector.ELECTRICITY,
    obs: dict | None = None,
) -> tuple[GenerationController, RoleAgent]:
    """Create a GenerationController attached to a RoleAgent in a world."""
    if obs is None:
        obs = make_electricity_gen(p_mw=-10.0)
    behavior.set_obs(aid, obs)
    behavior.add_action(aid, "regulate")
    role = GenerationController(behavior, sector)
    return role, None  # agent created in test


@pytest.mark.asyncio
async def test_applies_factor_on_negotiation_finished():
    behavior = MockBehavior()
    behavior.set_obs("agent-0", make_electricity_gen(p_mw=-10.0))
    behavior.add_action("agent-0", "regulate")

    role = GenerationController(behavior, Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(role)

    async with world:
        # Emit event: setpoint = -7.0, capacity = -10.0 => factor = 0.7
        role.context.emit_event(
            NegotiationFinishedEvent(new_setpoint=-7.0, sector=Sector.ELECTRICITY)
        )

    assert len(behavior.action_log) == 1
    aid, action, args, _ = behavior.action_log[0]
    assert aid == "agent-0"
    assert action == "regulate"
    assert args[0] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_ignores_wrong_sector():
    behavior = MockBehavior()
    behavior.set_obs("agent-0", make_electricity_gen(p_mw=-10.0))
    behavior.add_action("agent-0", "regulate")

    role = GenerationController(behavior, Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(role)

    async with world:
        role.context.emit_event(
            NegotiationFinishedEvent(new_setpoint=-5.0, sector=Sector.HEAT)
        )

    assert len(behavior.action_log) == 0


@pytest.mark.asyncio
async def test_clamps_factor_to_one():
    behavior = MockBehavior()
    behavior.set_obs("agent-0", make_electricity_gen(p_mw=-5.0))
    behavior.add_action("agent-0", "regulate")

    role = GenerationController(behavior, Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(role)

    async with world:
        # Setpoint exceeds capacity => factor clamped to 1.0
        role.context.emit_event(
            NegotiationFinishedEvent(new_setpoint=-20.0, sector=Sector.ELECTRICITY)
        )

    assert len(behavior.action_log) == 1
    _, _, args, _ = behavior.action_log[0]
    assert args[0] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_no_action_when_no_capacity():
    behavior = MockBehavior()
    behavior.set_obs("agent-0", {"p_mw": 0.0})
    behavior.add_action("agent-0", "regulate")

    role = GenerationController(behavior, Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(role)

    async with world:
        role.context.emit_event(
            NegotiationFinishedEvent(new_setpoint=-5.0, sector=Sector.ELECTRICITY)
        )

    assert len(behavior.action_log) == 0
