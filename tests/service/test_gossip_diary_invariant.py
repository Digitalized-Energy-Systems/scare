"""Regression test for the diary ``started == Σ terminals`` invariant
under overlapping negotiation triggers.

The eval_full_small_20260527-015015 run leaked one ``started`` record
per slack child (e.g. child-118): a slack-budget override gossip and a
balance-round gossip raced, and the second ``_start_gossip`` overwrote
``self._gossip`` for the first originator gossip without recording a
terminal.  ``_start_gossip`` now retires any in-flight originator gossip
as ``abandoned`` before overwriting it.
"""

from __future__ import annotations

import pytest

from mango import RoleAgent, SimpleCommunicationSimulation, create_world
from mango.express.topology import create_topology

from scare.base import diagnostics
from scare.base.model import Sector
from scare.service.balance import EnergyBalanceNegotiator
from tests.conftest import MockBehavior, make_electricity_gen, make_electricity_load


def _build_group(behavior: MockBehavior, specs):
    world = create_world(
        communication_sim=SimpleCommunicationSimulation(default_delay_s=0.001)
    )
    agents, roles = [], []
    for spec in specs:
        behavior.set_obs(spec["aid"], spec["obs"])
        behavior.add_action(spec["aid"], "regulate")
        role = EnergyBalanceNegotiator(
            behavior, Sector.ELECTRICITY, priority=spec.get("priority", 0)
        )
        agent = world.register(RoleAgent(), suggested_aid=spec["aid"])
        agent.add_role(role)
        agents.append(agent)
        roles.append(role)
    with create_topology(tid="groups") as topo:
        nids = [topo.add_node(a) for a in agents]
        topo.set_characteristic(nids[0], agents[0], "leader")
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                topo.add_edge(nids[i], nids[j])
    return world, roles


def _terminals(diary):
    return {
        r.nid for r in diary
        if r.event in ("finished", "timed_out", "cancelled", "abandoned", "stalled")
    }


@pytest.mark.asyncio
async def test_start_gossip_retires_superseded_originator():
    """A second ``_start_gossip`` while a previous originator gossip is
    still live must record a terminal for the first — otherwise its
    ``started`` leaks and breaks ``started == Σ terminals``.
    """
    diagnostics.arm()
    behavior = MockBehavior()
    world, roles = _build_group(behavior, [
        {"aid": "leader-0", "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0), "priority": 0},
        {"aid": "load-A", "obs": make_electricity_load(p_mw=3.0, regulation=1.0, priority=2), "priority": 2},
    ])
    leader = roles[0]

    async with world:
        # First originator gossip; multi-member group so a real
        # ``started`` is recorded (not the singleton skip path).
        await leader._start_gossip(-0.05)
        first_nid = leader._gossip.negotiation_id
        assert leader._gossip.is_originator

        # Second trigger arrives before the first terminated.
        await leader._start_gossip(-0.04)
        assert leader._gossip.negotiation_id != first_nid

        # Sim-end flush retires the last still-in-flight gossip, just
        # like _flush_pending_negotiations in the real runner.
        leader.flush_pending()

    diary = diagnostics.negotiation_log()
    started = {r.nid for r in diary if r.event == "started"}
    orphans = started - _terminals(diary)
    assert not orphans, f"started without terminal: {orphans}"
    assert first_nid in _terminals(diary)


@pytest.mark.asyncio
async def test_diary_counts_balance_after_supersede():
    """End-to-end count check: started total equals terminal total."""
    diagnostics.arm()
    behavior = MockBehavior()
    world, roles = _build_group(behavior, [
        {"aid": "leader-0", "obs": make_electricity_gen(p_mw=-10.0, regulation=1.0), "priority": 0},
        {"aid": "load-A", "obs": make_electricity_load(p_mw=3.0, regulation=1.0, priority=2), "priority": 2},
    ])
    leader = roles[0]

    async with world:
        await leader._start_gossip(-0.05)
        await leader._start_gossip(-0.04)
        await leader._start_gossip(-0.03)
        leader.flush_pending()

    diary = diagnostics.negotiation_log()
    started = sum(1 for r in diary if r.event == "started")
    terminals = sum(
        1 for r in diary
        if r.event in ("finished", "timed_out", "cancelled", "abandoned", "stalled")
    )
    assert started == terminals, (
        f"diary invariant broken: started={started} terminals={terminals}"
    )
