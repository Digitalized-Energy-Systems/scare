"""Unit tests for :class:`MultiCommunityCPRole`.

The role is the CP-side of the ``component_level`` baseline.  It
collects per-community :class:`NegotiationFinishedEvent` deliveries
arriving via the existing ``cps``↔``groups`` cross-topology link,
blends them with an EMA, and commits via ``apply_regulate`` under a
deadband + cooldown guard.  These tests exercise that state machine in
isolation.
"""

from __future__ import annotations

import pytest
from mango import RoleAgent, create_world
from mango.express.topology import create_topology
from mango.simulation.world import step_simulation

from scare.base.model import (
    AskEnergyMessage,
    NegotiationFinishedEvent,
    Sector,
)
from scare.service.coupling.cp import MultiCommunityCPRole
from tests.conftest import MockBehavior


def _cp_obs(p_mw: float = 1.0, regulation: float = 1.0) -> dict:
    """Synthetic CP obs.

    The CP role reads sector-specific keys via ``_ACCESS_KEYS``
    (``el_mw`` / ``gas_kgps`` / ``heat_mw``); we stamp both these and
    the generic ``p_mw`` so the role can resolve a non-zero
    capacity regardless of which sector's path it walks.
    """
    return {
        "p_mw": p_mw,
        "el_mw": p_mw,
        "gas_kgps": p_mw,
        "heat_mw": p_mw,
        "regulation": regulation,
    }


def _make_role(
    *,
    ema_alpha: float = 0.3,
    deadband_mw: float = 0.05,
    min_interval_s: float = 1.0,
    p_mw: float = 1.0,
    regulation: float = 1.0,
) -> tuple[MultiCommunityCPRole, MockBehavior, RoleAgent, object]:
    behavior = MockBehavior()
    behavior.set_obs("cp1", _cp_obs(p_mw=p_mw, regulation=regulation))
    behavior.add_action("cp1", "regulate")
    role = MultiCommunityCPRole(
        behavior,
        [Sector.ELECTRICITY, Sector.GAS],
        ema_alpha=ema_alpha,
        deadband_mw=deadband_mw,
        min_interval_s=min_interval_s,
    )
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="cp1")
    agent.add_role(role)
    with create_topology(tid="groups") as topo:
        topo.add_node(agent)
    with create_topology(tid="cps") as cps_topo:
        cps_topo.add_node(agent)
    return role, behavior, agent, world


def _regulate_calls(behavior: MockBehavior) -> list[float]:
    return [c[2][0] for c in behavior.action_log if c[1] == "regulate"]


# ---------------------------------------------------------------------------
# EMA blending + initial seeding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_event_seeds_ema_and_commits():
    role, behavior, _, world = _make_role(p_mw=2.0, regulation=1.0)
    async with world:
        await step_simulation(world, step_size_s=0.1)
        # Patch the role's idea of "now" to a value past any cooldown.
        # The role reads self.context.current_timestamp; we make a fresh
        # event arrive while sim time is advanced past min_interval_s.
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.ELECTRICITY),
            meta={},
        )

    # Target after EMA seeding == new_setpoint = 1.0
    # |target − committed (0.0)| = 1.0 > deadband 0.05 → commit
    # factor = clamp(|1.0 / 2.0|, 0, 1) = 0.5
    calls = _regulate_calls(behavior)
    assert calls == pytest.approx([0.5], rel=1e-3)
    assert role._target_by_sector[Sector.ELECTRICITY] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_second_event_blends_via_ema():
    role, behavior, _, world = _make_role(ema_alpha=0.3, min_interval_s=0.0, p_mw=2.0)
    async with world:
        await step_simulation(world, step_size_s=0.1)
        # Seed with 1.0 then blend in 0.0; EMA at α=0.3 gives 0.7.
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.ELECTRICITY),
            meta={},
        )
        await step_simulation(world, step_size_s=2.0)  # past cooldown
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=0.0, sector=Sector.ELECTRICITY),
            meta={},
        )

    assert role._target_by_sector[Sector.ELECTRICITY] == pytest.approx(0.7, rel=1e-3)


# ---------------------------------------------------------------------------
# Deadband suppresses no-op commits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deadband_suppresses_small_drift():
    role, behavior, _, world = _make_role(deadband_mw=0.1, p_mw=2.0)
    async with world:
        await step_simulation(world, step_size_s=0.1)
        # Seed and commit at 1.0.
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.ELECTRICITY),
            meta={},
        )
        await step_simulation(world, step_size_s=2.0)  # past cooldown
        # Tiny perturbation: EMA target shifts by ~0.3 * (1.02 - 1.0) = 0.006,
        # well under the 0.1 deadband.
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.02, sector=Sector.ELECTRICITY),
            meta={},
        )

    # Only the initial commit lands.
    calls = _regulate_calls(behavior)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Cooldown blocks rapid commits even when the deadband would allow them
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_blocks_rapid_commits():
    role, behavior, _, world = _make_role(deadband_mw=0.0, min_interval_s=5.0, p_mw=2.0)
    async with world:
        await step_simulation(world, step_size_s=0.1)
        # First commit lands.
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.ELECTRICITY),
            meta={},
        )
        # Same-tick second event should be gated by cooldown.
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=0.0, sector=Sector.ELECTRICITY),
            meta={},
        )

    calls = _regulate_calls(behavior)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Per-sector independence: signals on different sectors update separately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_sector_targets_are_independent():
    role, behavior, _, world = _make_role(min_interval_s=0.0, p_mw=2.0)
    async with world:
        await step_simulation(world, step_size_s=0.1)
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.ELECTRICITY),
            meta={},
        )
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=0.5, sector=Sector.GAS),
            meta={},
        )

    # Both sectors seed independently.
    assert role._target_by_sector[Sector.ELECTRICITY] == pytest.approx(1.0)
    assert role._target_by_sector[Sector.GAS] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_event_for_unsubscribed_sector_is_ignored():
    behavior = MockBehavior()
    behavior.set_obs("cp1", _cp_obs(p_mw=2.0))
    behavior.add_action("cp1", "regulate")
    role = MultiCommunityCPRole(
        behavior,
        [Sector.ELECTRICITY],  # gas not subscribed
        min_interval_s=0.0,
    )
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="cp1")
    agent.add_role(role)
    with create_topology(tid="groups") as topo:
        topo.add_node(agent)

    async with world:
        await step_simulation(world, step_size_s=0.1)
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.GAS),
            meta={},
        )

    assert Sector.GAS not in role._target_by_sector
    assert _regulate_calls(behavior) == []


# ---------------------------------------------------------------------------
# Branch-failure EMA reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_branch_failure_clears_ema_targets():
    role, _, _, world = _make_role(min_interval_s=0.0, p_mw=2.0)
    async with world:
        await step_simulation(world, step_size_s=0.1)
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.ELECTRICITY),
            meta={},
        )
        assert role._target_by_sector  # seeded
        role.on_branch_failure(("b1",))
        assert role._target_by_sector == {}


@pytest.mark.asyncio
async def test_on_branch_failure_preserves_committed_setpoint():
    # The physical setpoint at the CP doesn't move on a branch failure,
    # so the deadband must keep anchoring against the live committed
    # value even after the EMA targets are wiped.
    role, behavior, _, world = _make_role(
        ema_alpha=0.3, deadband_mw=0.05, min_interval_s=0.0, p_mw=2.0
    )
    async with world:
        await step_simulation(world, step_size_s=0.1)
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.ELECTRICITY),
            meta={},
        )
        # Branch failure wipes EMA target but leaves committed = 1.0.
        role.on_branch_failure(("b1",))
        assert role._committed_by_sector[Sector.ELECTRICITY] == pytest.approx(1.0)
        # New event re-seeds EMA at the proposed value (no carry-over).
        await role._handle_negotiation_finished(
            NegotiationFinishedEvent(new_setpoint=1.02, sector=Sector.ELECTRICITY),
            meta={},
        )
    # Target is reset to 1.02 (no 0.3 * 1.02 + 0.7 * 1.0 blend);
    # |1.02 − 1.0| = 0.02 < deadband 0.05 → no new commit.
    assert role._target_by_sector[Sector.ELECTRICITY] == pytest.approx(1.02)
    # Total commits remain 1 (the initial one).
    assert len(_regulate_calls(behavior)) == 1


# ---------------------------------------------------------------------------
# AskEnergyMessage handler keeps gossip alive (available = 0)
# ---------------------------------------------------------------------------


def test_ask_energy_handler_replies_with_zero_available():
    # Direct unit test on the role's reply construction — no world needed.
    behavior = MockBehavior()
    behavior.set_obs("cp1", _cp_obs(p_mw=2.0, regulation=0.5))
    role = MultiCommunityCPRole(behavior, [Sector.ELECTRICITY])

    # Build a stub context just enough to read aid / send_message.
    sent: list = []

    class _Ctx:
        aid = "cp1"
        current_timestamp = 0.0

        async def send_message(self, msg, receiver_addr):
            sent.append((msg, receiver_addr))

    role._context = _Ctx()  # mango stores on attr _context

    import asyncio

    asyncio.run(
        role._handle_ask_energy(
            AskEnergyMessage(negotiation_id="n1", sector=Sector.ELECTRICITY),
            meta={"sender_addr": ("leader-addr",), "sender_id": "leader"},
        )
    )
    assert len(sent) == 1
    reply, _ = sent[0]
    # available = 0 mirrors EnergyConverterRole — the CP has no spare flex.
    assert reply.available == 0.0
    # setpoint = p_mw * regulation = 2.0 * 0.5 = 1.0
    assert reply.setpoint == pytest.approx(1.0)
