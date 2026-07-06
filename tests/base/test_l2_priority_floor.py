"""Unit tests for the L2 priority-floor helpers in scare.base.util.

The floor stops an L1 reactive shed (gossip ``balance`` / ``stability``)
from pushing a load below the component ADMM's per-tier allocation,
while yielding to local physical constraints (so curtailment still
works).  See ``RestorationConfiguration.enable_l2_priority_floor``.
"""

from __future__ import annotations

import pytest

from scare.base.model import Sector
from scare.base.util import (
    _l2_floor_store,
    apply_regulate,
    constraint_allowed_fraction,
    l2_effective_floor,
)


class _Behavior:
    """Minimal behavior double: records act() calls, holds a config."""

    def __init__(self, config=None, obs=None):
        self._scare_config = config
        self._obs = obs or {}
        self.acted: list[tuple[str, float]] = []
        self._net_results = object()

    def observe(self, aid):
        return dict(self._obs)

    def has_action(self, aid, action):
        return True

    def act(self, aid, action, factor):
        self.acted.append((aid, factor))


class _Cfg:
    def __init__(self, floor=True):
        self.enable_l2_priority_floor = floor
        self.cooldown_s = 0.0


# --- constraint_allowed_fraction ---------------------------------------


def test_allowed_fraction_no_pressure_is_one():
    obs = {"p_mw": 10.0, "vm_pu": 1.0}
    assert constraint_allowed_fraction(obs, Sector.ELECTRICITY, tier=3) == 1.0


def test_allowed_fraction_tier1_immune():
    obs = {"p_mw": 10.0, "vm_pu": 1.09}  # well past any deadband
    assert constraint_allowed_fraction(obs, Sector.ELECTRICITY, tier=1) == 1.0


def test_allowed_fraction_drops_under_voltage():
    # UNDER-voltage: serving the load (which pulls V down) worsens it, so the
    # served fraction is capped. (Over-voltage does NOT cap a load — see below.)
    obs = {"p_mw": 10.0, "vm_pu": 0.953}
    frac = constraint_allowed_fraction(obs, Sector.ELECTRICITY, tier=4)
    assert 0.0 <= frac < 1.0


def test_allowed_fraction_load_not_capped_by_overvoltage():
    # Direction-aware: over-voltage is relieved by serving load, so a load is
    # NOT capped by a high-side reading (the old symmetric cap wrongly shed it).
    obs = {"p_mw": 10.0, "vm_pu": 1.09}
    assert constraint_allowed_fraction(obs, Sector.ELECTRICITY, tier=4) == 1.0


# --- l2_effective_floor ------------------------------------------------


def test_effective_floor_none_without_allocation():
    b = _Behavior()
    assert l2_effective_floor(b, "load-1", {"p_mw": 1.0}, Sector.ELECTRICITY, 3) is None


def test_effective_floor_equals_alloc_without_constraint():
    b = _Behavior()
    _l2_floor_store(b)["load-1"] = 0.8
    obs = {"p_mw": 1.0, "vm_pu": 1.0}
    assert l2_effective_floor(b, "load-1", obs, Sector.ELECTRICITY, 3) == 0.8


def test_effective_floor_capped_by_constraint():
    # Allocation says serve 1.0, but a near-bound voltage caps the
    # achievable fraction below 1.0 → floor relaxes to the constraint.
    b = _Behavior()
    _l2_floor_store(b)["load-1"] = 1.0
    obs = {"p_mw": 1.0, "vm_pu": 0.953}
    eff = l2_effective_floor(b, "load-1", obs, Sector.ELECTRICITY, 4)
    assert eff < 1.0
    assert eff == constraint_allowed_fraction(obs, Sector.ELECTRICITY, tier=4)


# --- apply_regulate enforcement ----------------------------------------


def test_l2_write_sets_floor_then_stability_is_clamped_up():
    cfg = _Cfg(floor=True)
    obs = {"p_mw": 1.0, "vm_pu": 1.0}
    b = _Behavior(config=cfg, obs=obs)
    # L2 allocates 0.6 to a tier-3 load.
    apply_regulate(
        b,
        "load-1",
        0.6,
        sector="electricity",
        reason="holon_supply_priority",
        timestamp=1.0,
        priority_tier=3,
    )
    assert _l2_floor_store(b)["load-1"] == pytest.approx(0.6)
    # A stability shed to 0.0 must be clamped up to the 0.6 floor.
    apply_regulate(
        b,
        "load-1",
        0.0,
        sector="electricity",
        reason="stability",
        timestamp=1.1,
        priority_tier=3,
    )
    assert b.acted[-1] == ("load-1", pytest.approx(0.6))


def test_floor_disabled_lets_stability_shed():
    cfg = _Cfg(floor=False)
    obs = {"p_mw": 1.0, "vm_pu": 1.0}
    b = _Behavior(config=cfg, obs=obs)
    apply_regulate(
        b,
        "load-1",
        0.6,
        sector="electricity",
        reason="holon_supply_priority",
        timestamp=1.0,
        priority_tier=3,
    )
    apply_regulate(
        b,
        "load-1",
        0.0,
        sector="electricity",
        reason="stability",
        timestamp=1.1,
        priority_tier=3,
    )
    assert b.acted[-1] == ("load-1", pytest.approx(0.0))


def test_tier1_stability_is_floored_hardlock():
    # Tier 1 is floored against reactive sheds: its constraint-allowed is
    # always 1.0 (immune), so the floor is its L2 allocation. The
    # curtailment auction (reason="curtail") can still shed it, but
    # stability/balance cannot.
    cfg = _Cfg(floor=True)
    obs = {"p_mw": 1.0, "vm_pu": 1.0}
    b = _Behavior(config=cfg, obs=obs)
    apply_regulate(
        b,
        "load-1",
        1.0,
        sector="electricity",
        reason="holon_supply_priority",
        timestamp=1.0,
        priority_tier=1,
    )
    apply_regulate(
        b,
        "load-1",
        0.0,
        sector="electricity",
        reason="stability",
        timestamp=1.1,
        priority_tier=1,
    )
    assert b.acted[-1] == ("load-1", pytest.approx(1.0))

    # ...but a constraint-driven curtail can still shed tier 1.
    applied = apply_regulate(
        b,
        "load-1",
        0.0,
        sector="electricity",
        reason="curtail",
        timestamp=1.2,
        priority_tier=1,
    )
    assert applied is True
    assert b.acted[-1] == ("load-1", pytest.approx(0.0))


def test_constraint_relaxes_floor_for_stability():
    # L2 says 1.0, but a near-bound voltage means the achievable floor is
    # below 1.0 — a stability shed toward the constraint limit is allowed.
    cfg = _Cfg(floor=True)
    obs = {"p_mw": 1.0, "vm_pu": 1.09}
    b = _Behavior(config=cfg, obs=obs)
    apply_regulate(
        b,
        "load-1",
        1.0,
        sector="electricity",
        reason="holon_supply_priority",
        timestamp=1.0,
        priority_tier=4,
    )
    eff = constraint_allowed_fraction(obs, Sector.ELECTRICITY, tier=4)
    apply_regulate(
        b,
        "load-1",
        0.0,
        sector="electricity",
        reason="stability",
        timestamp=1.1,
        priority_tier=4,
    )
    # Clamped up only to the constraint-allowed fraction, not to 1.0.
    assert b.acted[-1][1] == pytest.approx(eff)


# --- note_actuated_factor (gossip → apply_regulate dedup sync) ---------


def test_external_write_unsticks_apply_regulate_dedup():
    """A gossip ``_apply_setpoint`` write bypasses apply_regulate; without
    syncing the dedup cache, a later L2 re-dispatch to the load's
    allocation is silently deduped against the stale value and the
    gossip-shed load is never restored."""
    from scare.base.util import note_actuated_factor

    b = _Behavior(config=_Cfg(floor=False), obs={"p_mw": 1.0})
    # L2 set the load to 1.0 via apply_regulate (cache := 1.0).
    apply_regulate(
        b,
        "load-1",
        1.0,
        sector="electricity",
        reason="holon_supply_priority",
        timestamp=1.0,
        priority_tier=3,
    )
    # Gossip sheds it to 0.0 via a direct actuator write + cache sync.
    b.act("load-1", "regulate", 0.0)
    note_actuated_factor(b, "load-1", 0.0)
    # L2 re-dispatches the allocation (1.0).  Must actuate, not dedup.
    applied = apply_regulate(
        b,
        "load-1",
        1.0,
        sector="electricity",
        reason="holon_supply_priority",
        timestamp=2.0,
        priority_tier=3,
    )
    assert applied is True
    assert b.acted[-1] == ("load-1", 1.0)


def test_dedup_still_fires_when_cache_truthful():
    """If the cache reflects the true value, an identical re-dispatch is
    still deduped (no redundant solve)."""
    from scare.base.util import note_actuated_factor

    b = _Behavior(config=_Cfg(floor=False), obs={"p_mw": 1.0})
    b.act("load-1", "regulate", 0.5)
    note_actuated_factor(b, "load-1", 0.5)
    applied = apply_regulate(
        b,
        "load-1",
        0.5,
        sector="electricity",
        reason="stability",
        timestamp=2.0,
        priority_tier=3,
    )
    assert applied is False  # same value → deduped
