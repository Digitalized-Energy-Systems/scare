"""Pure decision kernel for the curtailment auction.

Extracted from :class:`scare.service.constraints.GridConstraintMonitor`. These
are side-effect-free functions over scalars / bid dicts: the role keeps the
async messaging, scheduling and ``apply_regulate`` plumbing and delegates the
*math* (willingness, proximity, allocation) here so it can be unit-tested
without a mango context. Tuning constants stay in ``constraints.py`` (some are
imported by tests) and are passed in.
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
    """Curtailment willingness for one load (bigger = more able to absorb
    curtailment). Product of priority-tier weight (dominant, lexicographic),
    a bounded sensitivity multiplier (within-tier tiebreaker), and current
    reducible output.

    Tier-1 LOADS (``capacity > 0``) return exactly 0.0 — not the 1e-9 floor —
    so a tier-1 self-only auction can't dispatch to self and break the hard
    lock. Generators (``capacity < 0``) keep the floor so PV stays
    shed-eligible under overvoltage.
    """
    if priority_tier <= 1 and capacity > 0:
        return 0.0
    prio_weight = tier_priority_weight(
        priority_tier, regime=-1, priority_tiers=priority_tiers,
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
    """Bounded proximity multiplier in ``[prox_min, prox_max]`` from cached
    multi-hop distance: larger ``hops_remaining`` => fewer hops from origin =>
    electrically closer => larger ∂constraint/∂Q."""
    frac = max(0.0, min(1.0, float(hops_remaining) / float(max_hops)))
    return prox_min + (prox_max - prox_min) * frac


class AllocationPlan(NamedTuple):
    """Output of :func:`plan_auction_allocation`.

    ``dispatches`` is the list of ``(bidder_key, addr, share)`` to send;
    ``tier1_exhausted`` flags the line-relief waterfall's terminal state (only
    tier-1 reducible bidders remain — relieving further would break the hard
    lock), which the role surfaces as a ``line_relief_tier1_residual`` event.
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
    """Compute the per-bidder curtailment shares.

    Three modes, mirroring the original ``_allocate_auction``:
      - **waterfall** (branch-downstream line relief): shed the lowest-priority
        tier with reducible draw first — every bidder in the highest tier number
        present (lowest priority) gets the full ``total``. Tier 1 is never shed;
        when only tier-1 reducible remains, return ``tier1_exhausted=True``.
      - **even split** when all willingness is zero (so something still
        curtails).
      - **willingness-proportional** otherwise.
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

    sum_w = sum(bids.values())
    if sum_w <= 0.0:
        share = total / len(bids)
        return AllocationPlan(
            [(k, bidders.get(k), share) for k in bidders], False,
        )
    return AllocationPlan(
        [(k, bidders.get(k), total * (w / sum_w)) for k, w in bids.items()],
        False,
    )
