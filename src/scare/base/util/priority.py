"""Priority-tier weight schedules and waterfall / aggregate share math."""

from __future__ import annotations


def clamp_tier_monotonic(fraction_by_tier: dict[int, float]) -> dict[int, float]:
    """Clamp per-tier service fractions non-increasing in tier number (tier 1
    = highest priority). Priority-safe: only ever lowers a lower-priority tier
    to its higher tier's level. Mutates and returns ``fraction_by_tier``.
    Shared by the component-allocation and coalition dispatch paths so a
    coalition merge can't reintroduce a tier inversion."""
    cap = 1.0
    for tier in sorted(t for t in fraction_by_tier if t >= 1):
        fraction_by_tier[tier] = min(fraction_by_tier[tier], cap)
        cap = fraction_by_tier[tier]
    return fraction_by_tier


def compute_priority_weighted_shares(
    demand_by_priority_per_group: list[dict[int, float]],
    served_by_priority_per_group: list[dict[int, float]],
    total_available: float,
) -> list[float]:
    """Compute each group's share of *total_available* via waterfall allocation.

    From the highest tier down, allocate proportionally to unserved demand
    until the budget is exhausted. Returns one share per group, summing to at
    most *total_available*.
    """
    n = len(demand_by_priority_per_group)
    shares = [0.0] * n
    if total_available <= 0 or n == 0:
        return shares

    all_tiers = sorted({t for d in demand_by_priority_per_group for t in d})
    remaining = total_available

    for tier in all_tiers:
        if remaining <= 1e-9:
            break
        tier_unserved = []
        for i in range(n):
            demand = demand_by_priority_per_group[i].get(tier, 0.0)
            served = served_by_priority_per_group[i].get(tier, 0.0)
            tier_unserved.append(max(0.0, demand - served))

        total_tier = sum(tier_unserved)
        if total_tier <= 1e-9:
            continue

        allocatable = min(remaining, total_tier)
        for i in range(n):
            share = allocatable * (tier_unserved[i] / total_tier)
            shares[i] += share
        remaining -= allocatable

    return shares


def aggregate_priority_weight(
    demand_by_priority: dict[int, float],
    served_by_priority: dict[int, float],
) -> float:
    """Scalar urgency weight from a priority-tier demand breakdown.

    Higher tiers weigh more per unit unserved demand. Used by the L3 CP
    S-coefficient. Uses the strict-monotone schedule, not the L1 QP schedule
    which returns 0 for tier 1 and would mask tier-1 unmet demand.
    """
    weight = 0.0
    for tier, demand in demand_by_priority.items():
        served = served_by_priority.get(tier, 0.0)
        unserved = max(0.0, demand - served)
        weight += unserved * tier_priority_weight_strict(int(tier))
    return weight


# 4-tier priority model with hard tier-1 enforcement. Tier 1 is pre-locked at
# ``regulation = 1`` off-QP; tiers 2-4 are QP-weighted with steep exponents so
# the equilibrium is effectively strict. Generators (tier <= 0) keep unit weight.
DEFAULT_PRIORITY_TIERS: int = 4

# Restoration (target > 0): higher tiers get higher weight. Tier 1 weight is 0
# (hard-locked at the pre-step, must not enter the QP or the dual normaliser).
_TIER_WEIGHT_RESTORATION: dict[int, float] = {
    1: 0.0,
    2: 1e8,
    3: 1e4,
    4: 1.0,
}

# Curtailment (target < 0): lowest tier sheds first. Tier 1 pre-locked at full.
_TIER_WEIGHT_CURTAILMENT: dict[int, float] = {
    1: 0.0,
    2: 1.0,
    3: 1e4,
    4: 1e8,
}


def tier_priority_weight(
    tier: int,
    *,
    regime: int = 1,
    priority_tiers: int = DEFAULT_PRIORITY_TIERS,
) -> float:
    """Single source of truth for the per-tier QP weight (L1 gossip).

    ``regime > 0`` restoration: tier 2→1e8, 3→1e4, 4→1. ``regime < 0``
    curtailment: 4→1e8 (sheds first), 3→1e4, 2→1. ``regime == 0``: 1.0.
    Tier 1 returns 0.0 (hard-locked off-QP; must not enter the QP or the dual
    normaliser). ``priority_tiers`` kept for API compatibility; schedule is
    fixed at 4 tiers, inputs clamped to ``[1, 4]``.
    """
    p = max(0, int(tier))
    if regime == 0 or p <= 0:
        return 1.0
    p = min(p, 4)
    if regime > 0:
        return _TIER_WEIGHT_RESTORATION.get(p, 1.0)
    return _TIER_WEIGHT_CURTAILMENT.get(p, 1.0)


def tier_priority_weight_strict(
    tier: int,
    *,
    priority_tiers: int = DEFAULT_PRIORITY_TIERS,
) -> float:
    """Strictly-monotone tier weight (tier 1 → P, tier P → 1) for
    waterfall-style sorts; tier 1 must sort first, which the QP schedule's low
    tier-1 weight breaks. Avoids the QP's wild magnitudes that would
    destabilise the ADMM sharing-distance objective.
    """
    P = max(1, int(priority_tiers))
    p = max(1, min(P, int(tier)))
    return float(P - p + 1)
