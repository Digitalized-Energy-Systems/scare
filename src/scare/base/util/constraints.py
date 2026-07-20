"""Local constraint math and the direction-aware serving cap.

Owns ``_CAP_STATE`` (the single-writer toggle) so its setter and reader share one
module namespace. Depends only on ``obs`` (leaf) + ``model`` — no import of the
``util`` package.
"""

from __future__ import annotations

import math

from scare.base.model import SECTOR_CONSTRAINTS, Sector
from scare.base.util.obs import obs_capacity


def constraint_utilization(
    value: float, bound_low: float, bound_high: float, *, unclamped: bool = False
) -> float:
    """Return how close *value* is to violating a bound.

    0.0 = at the centre of the feasible range.
    1.0 = at or beyond a bound.

    Clamped to ``[0, 1]`` by default. Pass ``unclamped=True`` to let values past a
    bound exceed 1.0 (the fractional overshoot), e.g. for integrating the
    out-of-bounds area; every other caller relies on the ``1.0`` ceiling.
    """
    span = bound_high - bound_low
    if span <= 0:
        return 1.0
    mid = (bound_low + bound_high) / 2.0
    u = abs(value - mid) / (span / 2.0)
    return u if unclamped else min(1.0, u)


_CLAMP_TIER_DEADBAND: dict[int, float] = {
    # Tier 1's entry is reached ONLY with ``tier1_immune=False`` (grading):
    # without it, tier 1 fell to the 0.85 default — the HARSHEST band — so a
    # tier-1 row was more readily excluded as constraint-throttled than the
    # tier-2 row it is compared against, which can both hide a real t1<t2
    # inversion and manufacture a fake one on the fatal priority claim.
    # Strictest tier ⇒ most protective band (above tier 2's).
    1: 0.97,
    2: 0.95,
    3: 0.90,
    4: 0.85,
}


_CLAMP_DEFAULT_DEADBAND: float = 0.85  # untagged / out-of-range tiers


def clamp_to_constraints(
    setpoint: float,
    obs: dict,
    sector: Sector,
    *,
    tier: int | None = None,
) -> float:
    """Clamp a proposed setpoint within local constraint bounds.

    Past a tier-dependent deadband the allowed fraction ramps linearly to zero
    (``(1-util)/(1-DEADBAND)``); the deadband stops normal LV drift from cutting
    every load and overriding the gossip waterfall. Tier 1 is immune (its
    pre-step lock must not be overruled; a true ConstraintViolation re-checks
    it). Tiers 2/3/4 → 0.95/0.90/0.85; ``None`` → 0.85.
    """
    cap = obs_capacity(obs)
    if cap == 0.0:
        return setpoint

    tightest_fraction = constraint_allowed_fraction(obs, sector, tier=tier)
    if tightest_fraction < 1.0:
        max_abs = tightest_fraction * abs(cap)
        setpoint = max(-max_abs, min(max_abs, setpoint))

    return setpoint


class _DirectionalCapState:
    def __init__(self) -> None:
        self.enabled = True


_CAP_STATE = _DirectionalCapState()


def set_directional_constraint_cap(enabled: bool) -> None:
    """Toggle the direction-aware serving cap in :func:`constraint_allowed_fraction`."""
    _CAP_STATE.enabled = bool(enabled)


def constraint_allowed_fraction(
    obs: dict,
    sector: Sector,
    *,
    tier: int | None = None,
    tier1_immune: bool = True,
) -> float:
    """Tightest constraint-allowed served fraction ``∈ [0, 1]`` from local
    measurements (same tier deadband as :func:`clamp_to_constraints`).

    The capacity fraction the actor may be served at given local physics, before
    the priority decision. Shared with the L2 priority-floor so the floor relaxes
    by exactly the amount the clamp sheds.

    DIRECTION-AWARE: serving pushes state vars DOWN for a load, UP for a
    generator; only the bound serving moves TOWARD may cap. Canonical case:
    over-voltage on a PV feeder is relieved by serving load, so a load must not be
    capped by it (the old symmetric ``constraint_utilization`` stranded the shed);
    over-voltage still caps generators. Tier-1 immune.
    """
    # Tier 1 immune to the soft clamp; a true ConstraintViolation re-checks it.
    # ``tier1_immune=False`` (grading only): the immunity is a runtime
    # protection policy, not physics — the eval's feasible-subset filter needs
    # the physical value or tier 1 reads as always-servable and every
    # physically-forced tier-1 shortfall is misread as a priority inversion.
    if tier1_immune and tier is not None and int(tier) == 1:
        return 1.0
    if tier is not None and int(tier) >= 1:
        deadband = _CLAMP_TIER_DEADBAND.get(int(tier), _CLAMP_DEFAULT_DEADBAND)
    else:
        deadband = _CLAMP_DEFAULT_DEADBAND
    width = max(1e-9, 1.0 - deadband)

    # Generators (cap < 0) inject → serving raises state vars; loads → lowers.
    serving_raises = obs_capacity(obs) < 0

    tightest_fraction = 1.0
    for var, (lo, hi) in SECTOR_CONSTRAINTS.get(sector, {}).items():
        if var not in obs:
            continue
        val = float(obs[var])
        if not math.isfinite(val):
            continue
        if _CAP_STATE.enabled:
            half = (hi - lo) / 2.0
            if half <= 0.0:
                continue
            mid = (lo + hi) / 2.0
            # One-sided utilization in the WORSENING direction only. Serving
            # raises val (generator) ⇒ the HIGH bound worsens; serving lowers
            # val (load) ⇒ the LOW bound worsens. The opposite side is relieved
            # by serving → no cap.
            util = (val - mid) / half if serving_raises else (mid - val) / half
            util = max(0.0, min(1.0, util))
        else:
            # Legacy symmetric behaviour: caps on proximity to EITHER bound.
            util = constraint_utilization(val, lo, hi)
        if util <= deadband:
            allowed = 1.0
        else:
            allowed = max(0.0, (1.0 - util) / width)
        tightest_fraction = min(tightest_fraction, allowed)
    return tightest_fraction
