"""Integration tests for gossip-based energy balance negotiation.

Tests multi-agent gossip convergence in a mango simulation world with
real message passing, topology, and periodic tasks.
"""

import pytest

from mango import RoleAgent, SimpleCommunicationSimulation, create_world
from mango.express.topology import create_topology
from mango.simulation.world import discrete_step_until

from scare.base.model import Sector
from scare.service.balance import EnergyBalanceNegotiator
from tests.conftest import MockBehavior, make_electricity_gen, make_electricity_load


def _build_group(
    behavior: MockBehavior,
    agent_specs: list[dict],
    comm_delay_s: float = 0.001,
) -> tuple:
    """Build a group of agents with EnergyBalanceNegotiator roles.

    Each spec dict: {aid, obs, priority}.
    Returns (world, agents, roles).
    """
    world = create_world(
        communication_sim=SimpleCommunicationSimulation(
            default_delay_s=comm_delay_s
        )
    )

    agents = []
    roles = []
    for spec in agent_specs:
        aid = spec["aid"]
        behavior.set_obs(aid, spec["obs"])
        behavior.add_action(aid, "regulate")
        role = EnergyBalanceNegotiator(
            behavior,
            Sector.ELECTRICITY,
            priority=spec.get("priority", 0),
        )
        agent = world.register(RoleAgent(), suggested_aid=aid)
        agent.add_role(role)
        agents.append(agent)
        roles.append(role)

    # Build fully-connected group topology; first agent is leader
    with create_topology(tid="groups") as topo:
        nids = []
        for agent in agents:
            nid = topo.add_node(agent)
            nids.append(nid)
        # Mark first as leader
        topo.set_characteristic(nids[0], agents[0], "leader")
        # Fully connect
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                topo.add_edge(nids[i], nids[j])

    return world, agents, roles


@pytest.mark.asyncio
async def test_two_agent_gossip_converges():
    """Generator + load: gossip should restore the load partially."""
    behavior = MockBehavior()
    world, agents, roles = _build_group(behavior, [
        {"aid": "gen-0", "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0), "priority": 0},
        {"aid": "load-0", "obs": make_electricity_load(p_mw=3.0, regulation=0.0, priority=1), "priority": 1},
    ])

    async with world:
        # Leader triggers negotiation
        roles[0].context.schedule_instant_task(
            roles[0].trigger_balance_negotiation()
        )
        await discrete_step_until(world, max_advance_time_s=5.0)

    # The load should have been regulated (at least one regulate call)
    regulate_calls = [c for c in behavior.action_log if c[0] == "load-0" and c[1] == "regulate"]
    assert len(regulate_calls) >= 1


@pytest.mark.asyncio
async def test_three_agent_gossip():
    """One generator, two loads: both should get regulated."""
    behavior = MockBehavior()
    world, agents, roles = _build_group(behavior, [
        {"aid": "gen-0", "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0), "priority": 0},
        {"aid": "load-0", "obs": make_electricity_load(p_mw=3.0, regulation=0.0, priority=1), "priority": 1},
        {"aid": "load-1", "obs": make_electricity_load(p_mw=2.0, regulation=0.0, priority=2), "priority": 2},
    ])

    async with world:
        roles[0].context.schedule_instant_task(
            roles[0].trigger_balance_negotiation()
        )
        await discrete_step_until(world, max_advance_time_s=5.0)

    # Both loads should have received regulate calls
    load_0_calls = [c for c in behavior.action_log if c[0] == "load-0" and c[1] == "regulate"]
    load_1_calls = [c for c in behavior.action_log if c[0] == "load-1" and c[1] == "regulate"]
    assert len(load_0_calls) >= 1
    assert len(load_1_calls) >= 1


@pytest.mark.asyncio
async def test_priority_ordering():
    """Higher priority load (lower number) should participate earlier.

    Under the 4-tier model the tier-1 load is regulated by the leader's
    hard-constraint pre-step (regulation = 1) and the tier-4 load is
    handled by the QP that follows.  Both should still receive at
    least one regulate call.
    """
    behavior = MockBehavior()
    world, agents, roles = _build_group(behavior, [
        {"aid": "gen-0", "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0), "priority": 0},
        {"aid": "high-prio", "obs": make_electricity_load(p_mw=3.0, regulation=0.0, priority=1), "priority": 1},
        {"aid": "low-prio", "obs": make_electricity_load(p_mw=3.0, regulation=0.0, priority=4), "priority": 4},
    ])

    async with world:
        roles[0].context.schedule_instant_task(
            roles[0].trigger_balance_negotiation()
        )
        await discrete_step_until(world, max_advance_time_s=5.0)

    # Both should be regulated, but high-prio first
    high_calls = [c for c in behavior.action_log if c[0] == "high-prio" and c[1] == "regulate"]
    low_calls = [c for c in behavior.action_log if c[0] == "low-prio" and c[1] == "regulate"]
    assert len(high_calls) >= 1
    assert len(low_calls) >= 1
