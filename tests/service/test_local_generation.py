"""Component tests for LocalGenerationFallbackRole."""

import pytest

from mango import RoleAgent, agent_composed_of, create_world
from mango.express.topology import create_topology
from mango.simulation.world import step_simulation

from scare.base.model import LocalGenerationApproval, Sector
from scare.service.local_generation import LocalGenerationFallbackRole
from tests.conftest import MockBehavior, make_electricity_gen, make_electricity_load


@pytest.mark.asyncio
async def test_local_gen_ramps_generator():
    behavior = MockBehavior()
    behavior.set_obs("leader", make_electricity_gen(p_mw=-10.0, regulation=0.5))
    behavior.add_action("leader", "regulate")

    role = LocalGenerationFallbackRole(behavior, Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="leader")
    agent.add_role(role)

    # Minimal topology: the leader is alone, so it checks itself.
    with create_topology(tid="groups") as topo:
        topo.add_node(agent)

    async with world:
        role.context.emit_event(
            LocalGenerationApproval(sector=Sector.ELECTRICITY, residual_deficit=2.0)
        )
        await step_simulation(world, step_size_s=0.1)

    regulate_calls = [c for c in behavior.action_log if c[1] == "regulate"]
    assert len(regulate_calls) >= 1


@pytest.mark.asyncio
async def test_local_gen_ignores_wrong_sector():
    behavior = MockBehavior()
    behavior.set_obs("leader", make_electricity_gen(p_mw=-10.0, regulation=0.5))
    behavior.add_action("leader", "regulate")

    role = LocalGenerationFallbackRole(behavior, Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="leader")
    agent.add_role(role)

    with create_topology(tid="groups") as topo:
        topo.add_node(agent)

    async with world:
        role.context.emit_event(
            LocalGenerationApproval(sector=Sector.GAS, residual_deficit=2.0)
        )
        await step_simulation(world, step_size_s=0.1)

    assert len(behavior.action_log) == 0


@pytest.mark.asyncio
async def test_local_gen_no_generators():
    behavior = MockBehavior()
    # A load, not a generator — nothing for the fallback to ramp.
    behavior.set_obs("leader", make_electricity_load(p_mw=3.0, regulation=0.5))
    behavior.add_action("leader", "regulate")

    role = LocalGenerationFallbackRole(behavior, Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="leader")
    agent.add_role(role)

    with create_topology(tid="groups") as topo:
        topo.add_node(agent)

    async with world:
        role.context.emit_event(
            LocalGenerationApproval(sector=Sector.ELECTRICITY, residual_deficit=2.0)
        )
        await step_simulation(world, step_size_s=0.1)

    # No regulate calls — loads are not ramped by the fallback
    assert len(behavior.action_log) == 0


@pytest.mark.asyncio
async def test_local_gen_zero_deficit_ignored():
    behavior = MockBehavior()
    behavior.set_obs("leader", make_electricity_gen(p_mw=-10.0, regulation=0.5))
    behavior.add_action("leader", "regulate")

    role = LocalGenerationFallbackRole(behavior, Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="leader")
    agent.add_role(role)

    with create_topology(tid="groups") as topo:
        topo.add_node(agent)

    async with world:
        role.context.emit_event(
            LocalGenerationApproval(sector=Sector.ELECTRICITY, residual_deficit=0.0)
        )
        await step_simulation(world, step_size_s=0.1)

    assert len(behavior.action_log) == 0
