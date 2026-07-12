"""Heat L2 reconnect (``enable_heat_l2_dispatch``): heat leaders actuate the
holon's Route-A per-tier service fractions (dispatch-only — gossip stays
heat-excluded), delivered heat is reported as the sector's flex supply pool,
and the heat curtail-lock becomes direction-aware (L2 raises defer, L2 sheds
pass).
"""

from __future__ import annotations

import pytest
from mango import Role, RoleAgent, create_world
from mango.express.topology import create_topology
from mango.simulation.world import discrete_step_until

from scare.base.model import (
    AskForAvailableFlex,
    AvailableFlexAnswer,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.util import apply_regulate, has_heat_curtail_lock
from scare.service.balance.balance import EnergyBalanceNegotiator
from tests.conftest import MockBehavior


def _heat_load_obs(q_mw_heat, regulation, priority):
    return {
        "q_mw_heat": q_mw_heat,
        "regulation": regulation,
        "t_k": 330.0,
        "priority": priority,
    }


def _build_heat_group(behavior, *, heat_l2_dispatch):
    """Heat leader (gen) + tier-1 and tier-4 heat loads, group topology."""
    world = create_world()
    specs = [
        ("leader-0", {"q_mw_heat": -5.0, "regulation": 1.0, "t_k": 340.0}, 0),
        ("load-t1", _heat_load_obs(1.0, 0.8, 1), 1),
        ("load-t4", _heat_load_obs(1.0, 0.5, 4), 4),
    ]
    agents, roles = [], []
    for aid, obs, priority in specs:
        behavior.set_obs(aid, obs)
        behavior.add_action(aid, "regulate")
        role = EnergyBalanceNegotiator(
            behavior,
            Sector.HEAT,
            priority=priority,
            enable_heat_l2_dispatch=heat_l2_dispatch,
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


def _regulates(behavior, aid):
    return [
        c for c in behavior.action_log if c[0] == aid and c[1] == "regulate"
    ]


@pytest.mark.asyncio
async def test_heat_leader_dispatches_service_fractions_when_enabled():
    """The tier-graded allocation reaches heat loads: tier-4 shed to its
    fraction, tier-1 raised toward full service."""
    behavior = MockBehavior()
    world, agents, roles = _build_heat_group(behavior, heat_l2_dispatch=True)
    leader = roles[0]

    async with world:
        await leader.context.send_message(
            StartBalanceNegotiation(
                service_fraction_by_sector_priority={
                    Sector.HEAT.value: {1: 1.0, 4: 0.2},
                },
            ),
            receiver_addr=leader.context.addr,
        )
        await discrete_step_until(world, max_advance_time_s=2.0)

    t4 = _regulates(behavior, "load-t4")
    assert t4 and t4[-1][2][0] == pytest.approx(0.2), (
        "tier-4 heat load never received its L2 shed"
    )
    t1 = _regulates(behavior, "load-t1")
    assert t1 and t1[-1][2][0] == pytest.approx(1.0), (
        "tier-1 heat load never raised to its full allocation"
    )


@pytest.mark.asyncio
async def test_heat_leader_drops_allocation_when_disabled():
    behavior = MockBehavior()
    world, agents, roles = _build_heat_group(behavior, heat_l2_dispatch=False)
    leader = roles[0]

    async with world:
        await leader.context.send_message(
            StartBalanceNegotiation(
                service_fraction_by_sector_priority={
                    Sector.HEAT.value: {1: 1.0, 4: 0.2},
                },
            ),
            receiver_addr=leader.context.addr,
        )
        await discrete_step_until(world, max_advance_time_s=2.0)

    assert not _regulates(behavior, "load-t4")
    assert not _regulates(behavior, "load-t1")


@pytest.mark.asyncio
async def test_all_zero_heat_allocation_is_refused():
    """The waterfall's degenerate no-supply branch (all tiers 0.0) must not
    black out the heat sector — a dark region would freeze the delivered-heat
    supply estimate at zero."""
    behavior = MockBehavior()
    world, agents, roles = _build_heat_group(behavior, heat_l2_dispatch=True)
    leader = roles[0]

    async with world:
        await leader.context.send_message(
            StartBalanceNegotiation(
                service_fraction_by_sector_priority={
                    Sector.HEAT.value: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                },
            ),
            receiver_addr=leader.context.addr,
        )
        await discrete_step_until(world, max_advance_time_s=2.0)

    assert not _regulates(behavior, "load-t4")
    assert not _regulates(behavior, "load-t1")


class _FlexCollector(Role):
    def __init__(self) -> None:
        super().__init__()
        self.answers: list[AvailableFlexAnswer] = []

    def setup(self) -> None:
        self.context.subscribe_message(
            self,
            lambda msg, meta: self.answers.append(msg),
            lambda msg, meta: isinstance(msg, AvailableFlexAnswer),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_flex_reports_delivered_heat_as_supply(enabled):
    """With the reconnect on, the flex answer's heat supply pool is the
    delivered heat (0.8 + 0.5) plus the upward probe share of the unserved
    gap — without it, heat supply is the raw generator ledger."""
    behavior = MockBehavior()
    world, agents, roles = _build_heat_group(behavior, heat_l2_dispatch=enabled)
    leader = roles[0]

    collector = _FlexCollector()
    asker = world.register(RoleAgent(), suggested_aid="asker-0")
    asker.add_role(collector)

    async with world:
        await collector.context.send_message(
            AskForAvailableFlex(), receiver_addr=leader.context.addr
        )
        await discrete_step_until(world, max_advance_time_s=2.0)

    assert collector.answers, "leader never answered the flex ask"
    from scare.service.balance.balance import _HEAT_L2_PROBE_SHARE

    supply = collector.answers[-1].supply_by_sector
    delivered, demand = 0.8 + 0.5, 1.0 + 1.0
    expected = delivered + _HEAT_L2_PROBE_SHARE * (demand - delivered)
    if enabled:
        assert supply.get(Sector.HEAT.value) == pytest.approx(expected)
    else:
        # Leader gen delivers |sp| = 5.0 into the legacy ledger; the point is
        # the delivered-heat reframe only applies under the flag.
        assert supply.get(Sector.HEAT.value) != pytest.approx(expected)


def test_heat_curtail_lock_is_direction_aware():
    """L2 raises defer on a frontier-locked load; L2 deeper sheds pass."""
    behavior = MockBehavior()
    behavior.set_obs("h1", _heat_load_obs(1.0, 1.0, 4))
    behavior.add_action("h1", "regulate")

    assert apply_regulate(
        behavior, "h1", 0.5, sector="heat", reason="curtail", timestamp=1.0
    )
    assert has_heat_curtail_lock(behavior, "h1")

    # Raise while locked -> deferred (recovery belongs to the frontier).
    assert not apply_regulate(
        behavior,
        "h1",
        0.9,
        sector="heat",
        reason="holon_supply_priority",
        timestamp=2.0,
    )
    # Deeper shed while locked -> passes (only helps t_k feasibility).
    assert apply_regulate(
        behavior,
        "h1",
        0.2,
        sector="heat",
        reason="holon_supply_priority",
        timestamp=3.0,
    )
    regs = [c for c in behavior.action_log if c[1] == "regulate"]
    assert regs[-1][2][0] == pytest.approx(0.2)
    assert has_heat_curtail_lock(behavior, "h1")
