"""Regression test for the L2 supply-priority allocation being silently
dropped while an L1 gossip is in flight.

``EnergyBalanceNegotiator._handle_start_balance`` returns early when
``self._sess.active`` is True. The supply-priority dispatch path doesn't
gossip (it just calls ``apply_regulate`` on the leader's group), so that
guard lost authoritative L2 priority decisions whenever they collided
with an in-flight L1 curtailment gossip.
"""

from __future__ import annotations

import pytest
from mango import RoleAgent, SimpleCommunicationSimulation, create_world
from mango.express.topology import create_topology
from mango.simulation.world import discrete_step_until

from scare.base.model import Sector, StartBalanceNegotiation
from scare.service.balance.balance import EnergyBalanceNegotiator
from tests.conftest import MockBehavior, make_electricity_gen, make_electricity_load


def _build_leader_with_two_loads(
    behavior: MockBehavior, *, comm_delay_s: float = 0.001
):
    """Leader (tier-0 gen) + 2 tier-2 loads, fully connected group topology."""
    world = create_world(
        communication_sim=SimpleCommunicationSimulation(default_delay_s=comm_delay_s)
    )

    specs = [
        {
            "aid": "leader-0",
            "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0),
            "priority": 0,
        },
        {
            "aid": "load-A",
            "obs": make_electricity_load(p_mw=3.0, regulation=1.0, priority=2),
            "priority": 2,
        },
        {
            "aid": "load-B",
            "obs": make_electricity_load(p_mw=2.0, regulation=1.0, priority=2),
            "priority": 2,
        },
    ]

    agents = []
    roles = []
    for spec in specs:
        aid = spec["aid"]
        behavior.set_obs(aid, spec["obs"])
        behavior.add_action(aid, "regulate")
        role = EnergyBalanceNegotiator(
            behavior,
            Sector.ELECTRICITY,
            priority=spec["priority"],
        )
        agent = world.register(RoleAgent(), suggested_aid=aid)
        agent.add_role(role)
        agents.append(agent)
        roles.append(role)

    with create_topology(tid="groups") as topo:
        nids = [topo.add_node(a) for a in agents]
        topo.set_characteristic(nids[0], agents[0], "leader")
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                topo.add_edge(nids[i], nids[j])

    return world, agents, roles


@pytest.mark.asyncio
async def test_l2_supply_priority_lands_while_gossip_active():
    """L2's supply-priority dispatch must reach the leader's regulate
    even when an L1 gossip is in flight (else the StartBalanceNegotiation
    is dropped because ``self._sess.active`` is True).
    """
    behavior = MockBehavior()
    world, agents, roles = _build_leader_with_two_loads(behavior)
    leader = roles[0]

    async with world:
        # Simulate an in-flight L1 gossip on the leader.
        leader._sess.active = True

        # Deliver an L2 supply-priority allocation; the dispatch must
        # shed both tier-2 loads to factor=0.0 despite _active.
        await leader.context.send_message(
            StartBalanceNegotiation(
                service_fraction_by_sector_priority={
                    Sector.ELECTRICITY.value: {2: 0.0},
                },
            ),
            receiver_addr=leader.context.addr,
        )
        await discrete_step_until(world, max_advance_time_s=2.0)

    # Both tier-2 loads should have been shed to factor ~0.0 by the L2
    # dispatch, independent of the in-flight gossip.
    def shed_calls(aid: str):
        return [
            c
            for c in behavior.action_log
            if c[0] == aid
            and c[1] == "regulate"
            and c[2]
            and abs(float(c[2][0])) < 1e-3
        ]

    assert shed_calls("load-A"), (
        "L2 supply-priority dispatch was silently dropped: load-A never "
        "received the shed action (race against in-flight L1 gossip)."
    )
    assert shed_calls("load-B"), (
        "L2 supply-priority dispatch was silently dropped: load-B never "
        "received the shed action (race against in-flight L1 gossip)."
    )
