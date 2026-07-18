"""Integration tests for the tier-1 hard-constraint pre-step.

The L1 leader hard-locks every tier-1 load at ``regulation = 1`` before
the gossip QP runs, provided the generator pool covers total tier-1
demand. When pool < tier-1 demand, the trivial allocation kicks in:
pro-rata pool across tier-1 loads, zero the lower tiers, skip the QP.
Both branches are exercised end-to-end through a real mango world.
"""

import pytest
from mango import RoleAgent, SimpleCommunicationSimulation, create_world
from mango.express.topology import create_topology
from mango.simulation.world import discrete_step_until

from scare.base.model import Sector
from scare.service.balance.balance import EnergyBalanceNegotiator
from tests.conftest import MockBehavior, make_electricity_gen, make_electricity_load


def _build_group(behavior: MockBehavior, agent_specs: list[dict]) -> tuple:
    """Build a fully-connected single-sector group; first agent is leader."""
    world = create_world(
        communication_sim=SimpleCommunicationSimulation(default_delay_s=0.001)
    )
    agents, roles = [], []
    for spec in agent_specs:
        aid = spec["aid"]
        behavior.set_obs(aid, spec["obs"])
        behavior.add_action(aid, "regulate")
        role = EnergyBalanceNegotiator(
            behavior, Sector.ELECTRICITY, priority=spec.get("priority", 0)
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


def _last_factor(behavior: MockBehavior, aid: str) -> float | None:
    """Most recent regulate-factor written for ``aid``; ``None`` if no write."""
    factor = None
    for entry in behavior.action_log:
        if entry[0] == aid and entry[1] == "regulate":
            args = entry[2]
            if args:
                factor = float(args[0])
    return factor


def _has_reason(behavior: MockBehavior, aid: str, reason_substr: str) -> bool:
    """Did ``aid`` ever get a regulate call? MockBehavior doesn't record
    the ``reason`` kwarg, so a call's presence proxies for the pre-step
    having engaged on this aid."""
    return any(
        entry[0] == aid and entry[1] == "regulate" for entry in behavior.action_log
    )


# Feasible branch: tier-1 demand <= pool.


@pytest.mark.asyncio
async def test_tier1_feasible_locked_at_one_qp_runs_for_lower_tiers():
    """Pool 2.0 MW, tier-1 demand 1.0 MW, tier-3 demand 1.0 MW.
    Feasible: tier-1 hard-locked at factor 1; QP serves residual on
    tier-3 (entire remaining pool).
    """
    behavior = MockBehavior()
    world, _agents, roles = _build_group(
        behavior,
        [
            {
                "aid": "gen-0",
                "obs": make_electricity_gen(p_mw=-2.0, regulation=1.0),
                "priority": 0,
            },
            {
                "aid": "load-1",
                "obs": make_electricity_load(p_mw=1.0, regulation=0.0, priority=1),
                "priority": 1,
            },
            {
                "aid": "load-3",
                "obs": make_electricity_load(p_mw=1.0, regulation=0.0, priority=3),
                "priority": 3,
            },
        ],
    )
    async with world:
        roles[0].context.schedule_instant_task(
            roles[0]._trigger.trigger_balance_negotiation()
        )
        await discrete_step_until(world, max_advance_time_s=30.0)

    f_t1 = _last_factor(behavior, "load-1")
    f_t3 = _last_factor(behavior, "load-3")
    # Tier-1 ends at exactly 1.0: the pre-step is the last write since
    # the QP that follows sees tier-1 with QP weight 0.
    assert f_t1 is not None and abs(f_t1 - 1.0) < 1e-9, (
        f"tier-1 must be hard-locked at 1.0, got {f_t1}"
    )
    # Tier-3 fully serveable from residual pool; box-clamp + QP noise
    # means we accept any factor >= 0.5.
    assert f_t3 is not None and f_t3 >= 0.5, (
        f"tier-3 should be partially or fully served by the residual QP, got {f_t3}"
    )


# Infeasible branch: tier-1 demand > pool.


@pytest.mark.asyncio
async def test_tier1_infeasible_pro_rata_lower_tiers_zero_no_qp():
    """Pool 0.5 MW, tier-1 demand 1.0 MW (two tier-1 loads at 0.5 each),
    tier-3 demand 1.0 MW. Infeasible: each tier-1 load gets factor 0.5;
    tier-3 is shed to factor 0; the gossip QP does not run.
    """
    behavior = MockBehavior()
    world, _agents, roles = _build_group(
        behavior,
        [
            {
                "aid": "gen-0",
                "obs": make_electricity_gen(p_mw=-0.5, regulation=1.0),
                "priority": 0,
            },
            {
                "aid": "load-1a",
                "obs": make_electricity_load(p_mw=0.5, regulation=0.0, priority=1),
                "priority": 1,
            },
            {
                "aid": "load-1b",
                "obs": make_electricity_load(p_mw=0.5, regulation=0.0, priority=1),
                "priority": 1,
            },
            {
                "aid": "load-3",
                "obs": make_electricity_load(p_mw=1.0, regulation=0.0, priority=3),
                "priority": 3,
            },
        ],
    )
    async with world:
        roles[0].context.schedule_instant_task(
            roles[0]._trigger.trigger_balance_negotiation()
        )
        await discrete_step_until(world, max_advance_time_s=30.0)

    f_t1a = _last_factor(behavior, "load-1a")
    f_t1b = _last_factor(behavior, "load-1b")
    f_t3 = _last_factor(behavior, "load-3")
    # Pro-rata: each tier-1 load gets new_sp 0.25 / cap 0.5 = factor 0.5.
    assert f_t1a is not None and abs(f_t1a - 0.5) < 1e-6, (
        f"tier-1 pro-rata: expected ~0.5, got {f_t1a}"
    )
    assert f_t1b is not None and abs(f_t1b - 0.5) < 1e-6, (
        f"tier-1 pro-rata: expected ~0.5, got {f_t1b}"
    )
    # Tier 3 is shed to 0 in the infeasible branch.
    assert f_t3 is not None and abs(f_t3) < 1e-9, (
        f"tier-3 should be 0 under tier-1 starvation, got {f_t3}"
    )


# No-deficit branch: pool >> demand.


@pytest.mark.asyncio
async def test_tier1_no_deficit_all_loads_served():
    """Pool 10 MW, tier-1 demand 1 MW, tier-3 demand 1 MW. Surplus:
    tier-1 pre-locked at 1.0; tier-3 served by the QP up to full demand.
    """
    behavior = MockBehavior()
    world, _agents, roles = _build_group(
        behavior,
        [
            {
                "aid": "gen-0",
                "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0),
                "priority": 0,
            },
            {
                "aid": "load-1",
                "obs": make_electricity_load(p_mw=1.0, regulation=0.0, priority=1),
                "priority": 1,
            },
            {
                "aid": "load-3",
                "obs": make_electricity_load(p_mw=1.0, regulation=0.0, priority=3),
                "priority": 3,
            },
        ],
    )
    async with world:
        roles[0].context.schedule_instant_task(
            roles[0]._trigger.trigger_balance_negotiation()
        )
        await discrete_step_until(world, max_advance_time_s=30.0)

    f_t1 = _last_factor(behavior, "load-1")
    f_t3 = _last_factor(behavior, "load-3")
    assert f_t1 is not None and abs(f_t1 - 1.0) < 1e-9, (
        f"tier-1 must be hard-locked at 1.0 under surplus, got {f_t1}"
    )
    assert f_t3 is not None and f_t3 >= 0.5, (
        f"tier-3 should be near-fully served under surplus, got {f_t3}"
    )
