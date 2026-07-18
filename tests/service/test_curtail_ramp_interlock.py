"""Tests for the curtail-vs-ramp interlock.

The interlock is enforced centrally in ``apply_regulate`` via a generator
over-voltage curtail-lock: while the curtailment auction holds a generator's
active power below full for a live node violation (PV over-voltage), the
local-gen RESTORE writes (``self_local_gen`` inline self-dispatch,
``local_gen_fallback`` role) DEFER instead of ramping it back to 1.0. Without
it the multiplicative ``_apply_curtail`` always restarts from full and the
auction/restore pair limit-cycles, so the over-voltage never clears.

Also guards the ``_find_constraint_monitor`` lookup, which must use mango's
``context.get_role`` — a prior version used a non-existent ``context.roles``
attribute and silently returned ``None``.
"""

from __future__ import annotations

import pytest

from scare.base.config import RestorationConfiguration
from scare.base.model import Sector
from scare.base.util import apply_regulate, has_gen_curtail_lock
from scare.service.balance.balance import EnergyBalanceNegotiator
from tests.conftest import MockBehavior, make_electricity_gen


def _gen_behavior(*, interlock: bool) -> MockBehavior:
    b = MockBehavior()
    b.set_obs("pv", make_electricity_gen(p_mw=-10.0, regulation=1.0))
    b.add_action("pv", "regulate")
    b._scare_config = RestorationConfiguration(enable_curtail_ramp_interlock=interlock)
    return b


def _regulate_values(b: MockBehavior) -> list[float]:
    return [c[2][0] for c in b.action_log if c[1] == "regulate"]


def test_curtail_sets_lock_and_restore_defers():
    b = _gen_behavior(interlock=True)
    # Auction sheds the PV below full -> sets the lock and applies.
    assert apply_regulate(
        b, "pv", 0.925, sector="electricity", reason="curtail", timestamp=0.0
    )
    assert has_gen_curtail_lock(b, "pv", now=0.1)
    # Both restore paths defer while the lock is fresh.
    assert not apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="self_local_gen", timestamp=0.1
    )
    assert not apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="local_gen_fallback", timestamp=0.2
    )
    # The PV was never ramped back to full.
    assert 1.0 not in _regulate_values(b)


def test_lock_is_freshness_lifted_after_ttl():
    b = _gen_behavior(interlock=True)
    apply_regulate(
        b, "pv", 0.925, sector="electricity", reason="curtail", timestamp=0.0
    )
    # Well past the lock TTL (3 s) with no re-assert: restore is allowed again.
    assert not has_gen_curtail_lock(b, "pv", now=10.0)
    assert apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="self_local_gen", timestamp=10.0
    )


def test_curtail_to_full_clears_lock():
    b = _gen_behavior(interlock=True)
    apply_regulate(
        b, "pv", 0.925, sector="electricity", reason="curtail", timestamp=0.0
    )
    # Auction releases (writes ~1.0) -> lock cleared.
    apply_regulate(b, "pv", 1.0, sector="electricity", reason="curtail", timestamp=0.5)
    assert not has_gen_curtail_lock(b, "pv", now=0.6)


def test_disabled_flag_allows_restore():
    b = _gen_behavior(interlock=False)
    apply_regulate(
        b, "pv", 0.925, sector="electricity", reason="curtail", timestamp=0.0
    )
    assert not has_gen_curtail_lock(b, "pv", now=0.1)
    assert apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="self_local_gen", timestamp=0.1
    )
    assert 1.0 in _regulate_values(b)


# --------------------------------------------------------------------------- #
# _find_constraint_monitor must use get_role (regression for the .roles bug)
# --------------------------------------------------------------------------- #


class _FakeMonitor:
    def __init__(self, sector: Sector) -> None:
        self.sector = sector


class _FakeCtxWithGetRole:
    """Stand-in for mango RoleContext: exposes ``get_role`` (the real API),
    not the non-existent ``.roles`` attribute the buggy version reached for."""

    def __init__(self, monitor) -> None:
        self._monitor = monitor
        self.roles = []  # the buggy lookup used this; must NOT be consulted

    def get_role(self, cls):
        return self._monitor


def test_find_constraint_monitor_uses_get_role():
    monitor = _FakeMonitor(Sector.ELECTRICITY)
    role = EnergyBalanceNegotiator(MockBehavior(), Sector.ELECTRICITY)
    role._context = _FakeCtxWithGetRole(monitor)  # type: ignore[attr-defined]
    # Found via get_role, sector matches.
    assert role._listener.find_constraint_monitor() is monitor


def test_find_constraint_monitor_sector_guard():
    monitor = _FakeMonitor(Sector.HEAT)  # wrong sector
    role = EnergyBalanceNegotiator(MockBehavior(), Sector.ELECTRICITY)
    role._context = _FakeCtxWithGetRole(monitor)  # type: ignore[attr-defined]
    assert role._listener.find_constraint_monitor() is None


# --------------------------------------------------------------------------- #
# Gossip _apply_setpoint clamps a curtail-locked generator to its held level.
# The gossip path writes via behavior.act directly (bypassing apply_regulate),
# so it must honour the lock itself — as a CLAMP, not a deferral: the clamped
# applied_sp feeds the actuated-ledger writeback, which marks the gen
# saturated so the dual reallocates. (A deferral skips the writeback and
# leaves phantom gen supply on the ledger — A/B-validated worse.)
# --------------------------------------------------------------------------- #


class _FakeGossipCtx:
    def __init__(self, aid: str, now: float) -> None:
        self.aid = aid
        self.current_timestamp = now


def _gossip_role(b: MockBehavior, now: float) -> EnergyBalanceNegotiator:
    role = EnergyBalanceNegotiator(
        b,
        Sector.ELECTRICITY,
        constraint_aware=False,
        enable_l2_priority_floor=False,
    )
    role._context = _FakeGossipCtx("pv", now)  # type: ignore[attr-defined]
    return role


def test_gossip_apply_setpoint_clamps_restore_to_curtail_lock():
    b = _gen_behavior(interlock=True)
    apply_regulate(b, "pv", 0.5, sector="electricity", reason="curtail", timestamp=0.0)
    role = _gossip_role(b, now=0.5)
    applied = role._actuator.apply_setpoint(-10.0)  # gossip asks for full output
    assert applied == pytest.approx(-5.0)  # held at 0.5 * cap, not deferred
    assert 1.0 not in _regulate_values(b)
    assert _regulate_values(b)[-1] == pytest.approx(0.5)


def test_gossip_apply_setpoint_shed_passes_under_gen_lock():
    b = _gen_behavior(interlock=True)
    apply_regulate(b, "pv", 0.5, sector="electricity", reason="curtail", timestamp=0.0)
    role = _gossip_role(b, now=0.5)
    applied = role._actuator.apply_setpoint(-2.0)  # deeper shed is allowed
    assert applied == pytest.approx(-2.0)
    assert _regulate_values(b)[-1] == pytest.approx(0.2)


def test_gossip_apply_setpoint_restores_after_lock_expiry():
    b = _gen_behavior(interlock=True)
    apply_regulate(b, "pv", 0.5, sector="electricity", reason="curtail", timestamp=0.0)
    role = _gossip_role(b, now=10.0)  # past the 3 s lock TTL
    applied = role._actuator.apply_setpoint(-10.0)
    assert applied == pytest.approx(-10.0)
    assert _regulate_values(b)[-1] == pytest.approx(1.0)
