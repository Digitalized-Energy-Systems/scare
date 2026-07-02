"""Pure decision kernel for the curtailment auction.

Side-effect-free functions over scalars / bid dicts (willingness, proximity,
allocation), unit-testable without a mango context. Tuning constants stay in
``constraints.py`` and are passed in.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from scare.base.util import tier_priority_weight


def curtail_willingness(
    *,
    priority_tier: int,
    capacity: float,
    reducible: float,
    sensitivity: float,
    sensitivity_ref: float,
    priority_tiers: int,
    sens_mult_min: float,
    sens_mult_max: float,
) -> float:
    """Curtailment willingness for one load (bigger = more able to absorb).

    Product of priority-tier weight (dominant), a bounded sensitivity
    multiplier (within-tier tiebreaker), and reducible output. Tier-1 LOADS
    (``capacity > 0``) return exactly 0.0 (not the 1e-9 floor) so they can't
    dispatch to self and break the hard lock; generators keep the floor so PV
    stays shed-eligible under overvoltage.
    """
    if priority_tier <= 1 and capacity > 0:
        return 0.0
    prio_weight = tier_priority_weight(
        priority_tier,
        regime=-1,
        priority_tiers=priority_tiers,
    )
    sens_mult = sensitivity / sensitivity_ref if sensitivity_ref > 0.0 else 1.0
    if not math.isfinite(sens_mult) or sens_mult <= 0.0:
        sens_mult = 1.0
    sens_mult = max(sens_mult_min, min(sens_mult_max, sens_mult))
    willingness = prio_weight * sens_mult * reducible
    if not math.isfinite(willingness) or willingness <= 0.0:
        willingness = 1e-9
    return willingness


def proximity_from_hops(
    hops_remaining: float,
    max_hops: int,
    *,
    prox_min: float,
    prox_max: float,
) -> float:
    """Bounded proximity multiplier in ``[prox_min, prox_max]``: larger
    ``hops_remaining`` => fewer hops from origin => electrically closer."""
    frac = max(0.0, min(1.0, float(hops_remaining) / float(max_hops)))
    return prox_min + (prox_max - prox_min) * frac


class AllocationPlan(NamedTuple):
    """Output of :func:`plan_auction_allocation`.

    ``dispatches`` is the list of ``(bidder_key, addr, share)`` to send;
    ``tier1_exhausted`` flags the waterfall terminal state (only tier-1
    reducible bidders remain), surfaced as a ``line_relief_tier1_residual`` event.
    """

    dispatches: list[tuple[str, Any, float]]
    tier1_exhausted: bool


def plan_auction_allocation(
    bids: dict[str, float],
    bidders: dict[str, Any],
    bid_meta: dict[str, tuple],
    total: float,
    *,
    waterfall: bool,
    min_reducible: float,
) -> AllocationPlan:
    """Compute the per-bidder curtailment shares. Two modes:

    - **waterfall** (line relief): shed the lowest-priority tier with reducible
      draw first; tier 1 is never shed (``tier1_exhausted=True`` when only it remains).
    - **willingness-proportional** otherwise; zero-willingness bidders (tier-1
      hard-locked loads bid exactly 0.0) are excluded, and an all-zero field
      yields an empty plan rather than an even split that would shed them.
    """
    if not bids:
        return AllocationPlan([], False)

    if waterfall:
        eligible = {
            k: tier
            for k, (tier, red) in bid_meta.items()
            if tier >= 2 and red > min_reducible
        }
        if not eligible:
            return AllocationPlan([], True)
        target_tier = max(eligible.values())  # lowest priority present
        dispatches = [
            (k, bidders.get(k), total)
            for k, tier in eligible.items()
            if tier == target_tier
        ]
        return AllocationPlan(dispatches, False)

    positive = {k: w for k, w in bids.items() if w > 0.0}
    if not positive:
        return AllocationPlan([], False)
    sum_w = sum(positive.values())
    return AllocationPlan(
        [(k, bidders.get(k), total * (w / sum_w)) for k, w in positive.items()],
        False,
    )
