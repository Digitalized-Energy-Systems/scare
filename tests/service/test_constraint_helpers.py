"""Unit tests for the collaborators extracted from GridConstraintMonitor.

Covers the previously-untested pure logic now in curtailment.py (auction
willingness / proximity / allocation) and heat_frontier.py (the frontier step
decision + priority-waterfall peer gate).
"""

from __future__ import annotations

from scare.service.control.curtailment import (
    AllocationPlan,
    curtail_willingness,
    plan_auction_allocation,
    proximity_from_hops,
)
from scare.service.control.heat_frontier import FrontierDecision, HeatFrontierController

# Mirror constraints.py defaults so the willingness math matches the role.
_SENS_MIN, _SENS_MAX = 0.25, 4.0
_PROX_MIN, _PROX_MAX = 0.25, 4.0


def _willingness(tier, cap, reducible, *, sens=0.01, sens_ref=0.01):
    return curtail_willingness(
        priority_tier=tier,
        capacity=cap,
        reducible=reducible,
        sensitivity=sens,
        sensitivity_ref=sens_ref,
        priority_tiers=4,
        sens_mult_min=_SENS_MIN,
        sens_mult_max=_SENS_MAX,
    )


# --------------------------------------------------------------------------- #
# curtail_willingness
# --------------------------------------------------------------------------- #


def test_tier1_load_is_unwilling():
    # Tier-1 LOAD (cap > 0) returns exactly 0 — protects the hard lock.
    assert _willingness(1, cap=1.0, reducible=0.5) == 0.0


def test_tier1_generator_keeps_floor():
    # Tier-1 GENERATOR (cap < 0) stays shed-eligible at the 1e-9 floor.
    assert _willingness(1, cap=-1.0, reducible=0.5) == 1e-9


def test_lower_priority_more_willing():
    # Higher tier number = lower priority = more willing to shed.
    w_t2 = _willingness(2, cap=1.0, reducible=0.5)
    w_t3 = _willingness(3, cap=1.0, reducible=0.5)
    assert w_t3 > w_t2 > 0.0


def test_willingness_scales_with_reducible():
    w1 = _willingness(2, cap=1.0, reducible=1.0)
    w2 = _willingness(2, cap=1.0, reducible=2.0)
    assert abs(w2 - 2.0 * w1) < 1e-12


def test_sensitivity_multiplier_is_clamped():
    # Huge sensitivity clamps the multiplier at _SENS_MAX (tiebreaker stays
    # bounded so priority remains lexicographic).
    base = _willingness(2, cap=1.0, reducible=1.0, sens=0.01, sens_ref=0.01)
    huge = _willingness(2, cap=1.0, reducible=1.0, sens=1e6, sens_ref=0.01)
    assert abs(huge - base * _SENS_MAX) < 1e-9


# --------------------------------------------------------------------------- #
# proximity_from_hops
# --------------------------------------------------------------------------- #


def test_proximity_monotonic_and_bounded():
    far = proximity_from_hops(0, 3, prox_min=_PROX_MIN, prox_max=_PROX_MAX)
    near = proximity_from_hops(3, 3, prox_min=_PROX_MIN, prox_max=_PROX_MAX)
    mid = proximity_from_hops(1, 3, prox_min=_PROX_MIN, prox_max=_PROX_MAX)
    assert far == _PROX_MIN
    assert near == _PROX_MAX
    assert far < mid < near


# --------------------------------------------------------------------------- #
# plan_auction_allocation
# --------------------------------------------------------------------------- #


def test_allocation_empty_bids():
    plan = plan_auction_allocation({}, {}, {}, 1.0, waterfall=False, min_reducible=5e-4)
    assert plan == AllocationPlan([], False)


def test_allocation_proportional_to_willingness():
    bids = {"a": 1.0, "b": 3.0}
    bidders = {"a": "addr_a", "b": "addr_b"}
    plan = plan_auction_allocation(
        bids, bidders, {}, 1.0, waterfall=False, min_reducible=5e-4
    )
    shares = {k: s for k, _addr, s in plan.dispatches}
    assert abs(shares["a"] - 0.25) < 1e-12
    assert abs(shares["b"] - 0.75) < 1e-12
    assert not plan.tier1_exhausted


def test_allocation_even_split_when_all_zero():
    bids = {"a": 0.0, "b": 0.0}
    bidders = {"a": "addr_a", "b": "addr_b"}
    plan = plan_auction_allocation(
        bids, bidders, {}, 1.0, waterfall=False, min_reducible=5e-4
    )
    shares = {k: s for k, _addr, s in plan.dispatches}
    assert shares == {"a": 0.5, "b": 0.5}


def test_waterfall_sheds_lowest_priority_tier_first_full_amount():
    # tier 2 and tier 4 present with reducible draw -> only tier 4 (lowest
    # priority) sheds, each eligible gets the FULL total.
    bids = {"hi": 1.0, "lo": 1.0}
    bidders = {"hi": "addr_hi", "lo": "addr_lo"}
    bid_meta = {"hi": (2, 0.5), "lo": (4, 0.5)}
    plan = plan_auction_allocation(
        bids, bidders, bid_meta, 0.8, waterfall=True, min_reducible=5e-4
    )
    assert plan.dispatches == [("lo", "addr_lo", 0.8)]
    assert not plan.tier1_exhausted


def test_waterfall_tier1_exhausted_when_only_tier1_reducible():
    # Only a tier-1 reducible bidder left -> nothing eligible, residual flagged.
    bid_meta = {"crit": (1, 0.5)}
    plan = plan_auction_allocation(
        {"crit": 1.0}, {"crit": "a"}, bid_meta, 0.8, waterfall=True, min_reducible=5e-4
    )
    assert plan.dispatches == []
    assert plan.tier1_exhausted is True


def test_waterfall_ignores_exhausted_reducible():
    # tier-4 bidder below the reducible threshold is not eligible.
    bid_meta = {"lo": (4, 1e-9)}
    plan = plan_auction_allocation(
        {"lo": 1.0}, {"lo": "a"}, bid_meta, 0.8, waterfall=True, min_reducible=5e-4
    )
    assert plan.tier1_exhausted is True


# --------------------------------------------------------------------------- #
# HeatFrontierController
# --------------------------------------------------------------------------- #

_LO = 313.15  # heat t_k floor; target = lo + MARGIN(3) = 316.15


def test_region_reducible_fresh_stale_and_tier_filter():
    c = HeatFrontierController(peer_freshness_s=50.0)
    c.note_peer_state("p4", t_received=0.0, tier=4, reducible=0.05)
    # Lower-priority (tier 4 > my tier 1) and fresh -> counted.
    assert c.region_has_lower_priority_reducible(my_tier=1, now=10.0) == 0.05
    # Same/higher priority peer (tier 4 not > my tier 4) -> not counted.
    assert c.region_has_lower_priority_reducible(my_tier=4, now=10.0) == 0.0
    # Stale (now - t_received > freshness) -> aged out.
    assert c.region_has_lower_priority_reducible(my_tier=1, now=100.0) == 0.0


def test_decide_holds_in_band():
    c = HeatFrontierController(peer_freshness_s=50.0)
    # t between deadband and restore band -> no move.
    out = c.decide(
        t=318.0,
        lo=_LO,
        cap=0.05,
        cur=1.0,
        sensitivity=660.0,
        now=1.0,
        my_tier=2,
        has_lock=False,
        waterfall_enabled=False,
    )
    assert out is None


def test_decide_sheds_when_too_cold():
    c = HeatFrontierController(peer_freshness_s=50.0)
    out = c.decide(
        t=300.0,
        lo=_LO,
        cap=0.05,
        cur=1.0,
        sensitivity=660.0,
        now=1.0,
        my_tier=2,
        has_lock=False,
        waterfall_enabled=False,
    )
    assert isinstance(out, FrontierDecision)
    assert out.new_reg < 1.0
    assert out.reason == "curtail"


def test_decide_waterfall_defers_to_lower_priority_peer():
    c = HeatFrontierController(peer_freshness_s=50.0)
    c.note_peer_state("p4", t_received=1.0, tier=4, reducible=0.05)
    out = c.decide(
        t=300.0,
        lo=_LO,
        cap=0.05,
        cur=1.0,
        sensitivity=660.0,
        now=1.0,
        my_tier=1,
        has_lock=False,
        waterfall_enabled=True,
    )
    assert out is None  # deferred while lower-priority reducible remains


def test_decide_restore_requires_lock():
    c = HeatFrontierController(peer_freshness_s=50.0)
    kw = dict(
        t=325.0,
        lo=_LO,
        cap=0.05,
        cur=0.8,
        sensitivity=660.0,
        now=1.0,
        my_tier=2,
        waterfall_enabled=False,
    )
    # Warm + below 1.0 but no curtail-lock -> do not claw back.
    assert c.decide(has_lock=False, **kw) is None
    # With the lock held, restore toward 1.0.
    out = c.decide(has_lock=True, **kw)
    assert isinstance(out, FrontierDecision)
    assert out.new_reg > 0.8
    assert out.reason == "heat_recovery"


def test_decide_below_commit_threshold_returns_none():
    c = HeatFrontierController(peer_freshness_s=50.0)
    # Enormous sensitivity -> sub-1e-3 step -> no commit.
    out = c.decide(
        t=314.0,
        lo=_LO,
        cap=0.05,
        cur=1.0,
        sensitivity=1e6,
        now=1.0,
        my_tier=2,
        has_lock=False,
        waterfall_enabled=False,
    )
    assert out is None
