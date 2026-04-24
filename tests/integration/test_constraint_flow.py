"""Integration tests for constraint violation → rebalance pipeline.

Tests that a constraint violation on one agent triggers a BalanceProblem
event, which the leader picks up and triggers rebalance negotiation.
"""

import pytest

from mango import RoleAgent, SimpleCommunicationSimulation, create_world
from mango.agent.role import Role
from mango.express.topology import create_topology
from mango.simulation.world import discrete_step_until, step_simulation

from scare.base.model import (
    BalanceProblem,
    ConstraintViolation,
    Sector,
)
from scare.service.balance import EnergyBalanceNegotiator
from scare.service.constraints import GridConstraintMonitor
from tests.conftest import MockBehavior, make_electricity_gen, make_electricity_load


def _build_constraint_group(
    behavior: MockBehavior,
    agent_specs: list[dict],
    comm_delay_s: float = 0.001,
) -> tuple:
    """Build a group with both GridConstraintMonitor and EnergyBalanceNegotiator.

    Each spec dict: {aid, obs, priority, node_id (optional)}.
    Returns (world, agents, monitors, negotiators).
    """
    world = create_world(
        communication_sim=SimpleCommunicationSimulation(
            default_delay_s=comm_delay_s
        )
    )

    agents = []
    monitors = []
    negotiators = []
    for spec in agent_specs:
        aid = spec["aid"]
        behavior.set_obs(aid, spec["obs"])
        behavior.add_action(aid, "regulate")

        monitor = GridConstraintMonitor(
            behavior,
            Sector.ELECTRICITY,
            node_id=spec.get("node_id", 0),
        )
        negotiator = EnergyBalanceNegotiator(
            behavior,
            Sector.ELECTRICITY,
            priority=spec.get("priority", 0),
        )

        agent = world.register(RoleAgent(), suggested_aid=aid)
        agent.add_role(monitor)
        agent.add_role(negotiator)
        agents.append(agent)
        monitors.append(monitor)
        negotiators.append(negotiator)

    with create_topology(tid="groups") as topo:
        nids = []
        for agent in agents:
            nid = topo.add_node(agent)
            nids.append(nid)
        topo.set_characteristic(nids[0], agents[0], "leader")
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                topo.add_edge(nids[i], nids[j])

    return world, agents, monitors, negotiators


@pytest.mark.asyncio
async def test_violation_triggers_rebalance():
    """Voltage violation on gen → BalanceProblem → leader triggers gossip → load regulated."""
    behavior = MockBehavior()
    world, agents, monitors, negotiators = _build_constraint_group(
        behavior,
        [
            {
                "aid": "gen-0",
                "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0, vm_pu=1.06),
                "priority": 0,
            },
            {
                "aid": "load-0",
                "obs": make_electricity_load(p_mw=3.0, regulation=0.0, priority=1, vm_pu=1.0),
                "priority": 1,
            },
        ],
    )

    violations = []

    class ViolationCapture(Role):
        def setup(self):
            self.context.subscribe_event(
                self, ConstraintViolation, self._on_violation
            )

        def _on_violation(self, event, src):
            violations.append(event)

    agents[0].add_role(ViolationCapture())

    async with world:
        # Let the periodic monitor fire (poll period for electricity is 0.5s)
        await discrete_step_until(world, max_advance_time_s=3.0)

    # Should have detected the voltage violation
    assert len(violations) >= 1
    assert violations[0].variable == "vm_pu"

    # The rebalance triggered by BalanceProblem should have regulated the load
    regulate_calls = [
        c for c in behavior.action_log if c[0] == "load-0" and c[1] == "regulate"
    ]
    assert len(regulate_calls) >= 1


@pytest.mark.asyncio
async def test_constraint_state_propagates_to_neighbour():
    """ConstraintStateMessage reaches a 1-hop neighbour."""
    behavior = MockBehavior()

    # Two agents: gen-0 has a constraint issue, load-0 is normal
    world, agents, monitors, _ = _build_constraint_group(
        behavior,
        [
            {
                "aid": "gen-0",
                "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0, vm_pu=1.03),
                "priority": 0,
            },
            {
                "aid": "load-0",
                "obs": make_electricity_load(p_mw=3.0, regulation=0.0, priority=1, vm_pu=1.0),
                "priority": 1,
            },
        ],
    )

    async with world:
        # Advance past the poll period so constraint state propagates
        await discrete_step_until(world, max_advance_time_s=3.0)

    # The load's monitor should have received constraint state from gen-0
    load_monitor = monitors[1]
    assert len(load_monitor._neighbour_state) >= 1
    # Check the cached state contains vm_pu from gen-0
    has_vm_pu = any(
        key[1] == "vm_pu" for key in load_monitor._neighbour_state
    )
    assert has_vm_pu
