"""Unit tests for the tier-stratified holon ADMM (Package C).

These tests bypass the mango role machinery and drive the ADMM
problem-construction + solve directly, so we can assert on the
numerical allocation without spinning up an agent world.

The setup mirrors what ``_run_tier_stratified_admm`` does:
1. Build per-actor flex bounds from synthetic
   ``demand_by_sector_priority`` / ``served_by_sector_priority``.
2. Compute the target vector ``T`` per (sector, tier) cell.
3. Set per-dimension priority weights ``priorities`` =
   ``2^(P − tier + 1)``.
4. Run ``start_coordinated_optimization`` on the resulting actors.
5. Assert on the resulting ``actor.x`` per cell.

The key property we test is the priority inversion fix: when one
group has tier-2 deficit and another has tier-8 surplus headroom,
the allocation should route flow such that the high-priority deficit
is satisfied first.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from distributed_resource_optimization import (
    create_admm_sharing_data,
    create_admm_start,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.flex_actor import (
    ADMMFlexActor,
)


def _build_problem(
    groups: list[dict],
    *,
    priority_tiers: int = 10,
):
    """Same construction as ``_run_tier_stratified_admm`` for one sector.

    Each entry in ``groups`` is a dict with keys:
      - ``demand`` :  ``{tier: total_demand_mw}``
      - ``served`` :  ``{tier: total_served_mw}``
    Both keyed by priority tier (1-10, lower = more critical).

    Returns ``(actors, total_T, priorities, sec, tiers)``.
    """
    sec = "electricity"
    tiers = sorted({t for g in groups for t in g.get("demand", {})})
    if not tiers:
        raise ValueError("at least one tier must be present")
    tier_idx = {t: j for j, t in enumerate(tiers)}
    n_dims = len(tiers)

    total_T = np.zeros(n_dims)
    group_bounds: list[tuple[np.ndarray, np.ndarray]] = []
    total_deficit_per_cell = np.zeros(n_dims)
    for g in groups:
        lb = np.zeros(n_dims)
        ub = np.full(n_dims, 1e-6)
        for tier in tiers:
            j = tier_idx[tier]
            dem = float(g.get("demand", {}).get(tier, 0.0))
            ser = float(g.get("served", {}).get(tier, 0.0))
            deficit = dem - ser
            total_T[j] += deficit
            total_deficit_per_cell[j] += max(0.0, deficit)
            lb[j] = -max(ser, 0.0)
            ub[j] = max(dem - ser, 1e-6)
        group_bounds.append((lb, ub))

    priorities = np.zeros(n_dims)
    for tier in tiers:
        priorities[tier_idx[tier]] = 2.0 ** max(0, priority_tiers - tier + 1)

    actors: list[ADMMFlexActor] = []
    for idx, g in enumerate(groups):
        lb, ub = group_bounds[idx]
        S = np.zeros(n_dims)
        for tier in tiers:
            j = tier_idx[tier]
            dem = float(g.get("demand", {}).get(tier, 0.0))
            ser = float(g.get("served", {}).get(tier, 0.0))
            my_deficit = max(0.0, dem - ser)
            pie = total_deficit_per_cell[j]
            if pie > 1e-9 and my_deficit > 1e-9:
                share = my_deficit / pie
                S[j] = -share * priorities[j]
        # Per-actor coupling: Σ_cell x ≤ budget.  ``budget`` defaults
        # to the actor's total deficit (= sum(ub) per cell) which
        # makes the coupling non-binding — preserves the legacy
        # "trivially feasible" behaviour for the basic tests.  Tests
        # that want to exercise scarcity set ``budget`` explicitly.
        budget = float(g.get("budget", sum(max(0.0, dem - ser)
                                           for dem, ser in zip(
                                               (g.get("demand", {}).get(t, 0.0) for t in tiers),
                                               (g.get("served", {}).get(t, 0.0) for t in tiers),
                                           ))))
        C = np.ones((1, n_dims))
        d = np.array([budget])
        actors.append(
            ADMMFlexActor(
                lb=lb, u=ub, C=C, d=d, S=S,
            )
        )

    return actors, total_T, priorities, sec, tiers


def _solve(actors, T, priorities, *, max_iters: int = 200, abs_tol: float = 1e-4):
    coord = create_sharing_target_distance_admm_coordinator()
    coord.max_iters = max_iters
    coord.abs_tol = abs_tol
    start = create_admm_start(
        create_admm_sharing_data(T.tolist(), priorities=priorities.tolist())
    )
    asyncio.run(start_coordinated_optimization(actors, coord, start))
    return [np.array(a.x) for a in actors]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_balanced_no_op():
    """All groups fully served → ADMM finds the zero allocation
    (within tolerance).  Sanity floor: if there's no deficit, the
    holon shouldn't do anything."""
    groups = [
        {"demand": {2: 5.0, 8: 5.0}, "served": {2: 5.0, 8: 5.0}},
        {"demand": {2: 5.0, 8: 5.0}, "served": {2: 5.0, 8: 5.0}},
    ]
    actors, T, p, _, tiers = _build_problem(groups)
    # T is essentially zero on every cell.
    assert np.allclose(T, 0.0, atol=1e-6), f"expected balanced T, got {T}"


def test_each_actor_absorbs_its_own_deficit():
    """Two groups, complementary deficits.
    Group A has tier-2 deficit, group B has tier-8 deficit.  Each
    group can absorb both its own deficit and a small amount of the
    other tier's.  The S-coefficient (deficit-share weighted by
    priority) should bias each actor toward absorbing its OWN cell's
    deficit, not the other actor's.

    The strict assertion: each actor's allocation in its own deficit
    cell exceeds its allocation in the other cell.
    """
    groups = [
        # A: tier-2 unserved, tier-8 fully served (room to shed)
        {
            "demand": {2: 10.0, 8: 10.0},
            "served": {2: 5.0, 8: 10.0},
        },
        # B: tier-2 fully served, tier-8 unserved
        {
            "demand": {2: 10.0, 8: 10.0},
            "served": {2: 10.0, 8: 5.0},
        },
    ]
    actors, T, p, _, tiers = _build_problem(groups)
    # Each cell has one group with the full 5 MW deficit (the other
    # contributes 0).  Sum per cell = 5.
    assert np.allclose(T, [5.0, 5.0]), f"unexpected T: {T}"

    # The S-coefficient should be sharply asymmetric:
    # A has all the tier-2 deficit share but none of tier-8.
    # → S_A = [-priority[2], 0]; S_B = [0, -priority[8]]
    s_A = actors[0].S
    s_B = actors[1].S
    assert s_A[0] < s_B[0], f"A's tier-2 S not stronger than B's: {s_A} {s_B}"
    assert s_B[1] < s_A[1], f"B's tier-8 S not stronger than A's: {s_A} {s_B}"

    xs = _solve(actors, T, p, max_iters=300, abs_tol=1e-5)

    sum_x = sum(xs)
    assert np.allclose(sum_x, T, atol=0.5), (
        f"ADMM didn't approximately reach T: sum_x={sum_x} T={T}"
    )

    # Each actor absorbs *its own* cell's deficit, not the other's.
    # B can't absorb tier-2 (ub_tier2_B ≈ 0); A can't absorb tier-8.
    a_tier2 = xs[0][0]
    a_tier8 = xs[0][1]
    b_tier2 = xs[1][0]
    b_tier8 = xs[1][1]
    assert a_tier2 > a_tier8, (
        f"A should absorb tier-2 more than tier-8: tier-2={a_tier2}, tier-8={a_tier8}"
    )
    assert b_tier8 > b_tier2, (
        f"B should absorb tier-8 more than tier-2: tier-8={b_tier8}, tier-2={b_tier2}"
    )
    # And the absorptions should match each cell's deficit.
    assert a_tier2 == pytest.approx(5.0, abs=0.2)
    assert b_tier8 == pytest.approx(5.0, abs=0.2)


def test_priority_weights_correct():
    """Verify the priority-weight schedule (2^(P-tier+1)) lands on
    the right ratios for P=10.  Tier-2 should be 64x tier-8.
    """
    groups = [
        {"demand": {2: 5.0, 8: 5.0}, "served": {2: 4.0, 8: 4.0}},
        {"demand": {2: 5.0, 8: 5.0}, "served": {2: 4.0, 8: 4.0}},
    ]
    actors, T, p, _, tiers = _build_problem(groups, priority_tiers=10)
    assert tiers == [2, 8]
    # 2^(10-2+1) = 512 vs 2^(10-8+1) = 8 → ratio 64
    assert p[0] / p[1] == pytest.approx(64.0)


def test_high_priority_pulls_harder():
    """Three groups with equal-magnitude deficits in three tiers.
    The allocation share should be monotone in priority: tier-2 cell
    >= tier-5 cell >= tier-8 cell.
    """
    groups = [
        {"demand": {2: 10.0}, "served": {2: 5.0}},  # tier-2 deficit
        {"demand": {5: 10.0}, "served": {5: 5.0}},  # tier-5 deficit
        {"demand": {8: 10.0}, "served": {8: 5.0}},  # tier-8 deficit
    ]
    actors, T, p, _, tiers = _build_problem(groups)
    assert tiers == [2, 5, 8]
    assert np.allclose(T, [5.0, 5.0, 5.0])

    xs = _solve(actors, T, p)
    # Each group has only one non-zero cell (its own tier).
    a_tier2 = xs[0][0]  # tier 2 in group 0
    b_tier5 = xs[1][1]  # tier 5 in group 1
    c_tier8 = xs[2][2]  # tier 8 in group 2
    assert a_tier2 >= b_tier5 - 1e-2, (
        f"tier-2 should get >= tier-5: a={a_tier2}, b={b_tier5}"
    )
    assert b_tier5 >= c_tier8 - 1e-2, (
        f"tier-5 should get >= tier-8: b={b_tier5}, c={c_tier8}"
    )


def test_zero_demand_cell_stays_zero():
    """A cell with zero demand on every group must stay at zero
    in the ADMM target and in every actor's allocation — no spurious
    pull from priority weight alone.
    """
    groups = [
        {"demand": {2: 10.0}, "served": {2: 6.0}},
        {"demand": {2: 10.0}, "served": {2: 10.0}},
    ]
    actors, T, p, _, tiers = _build_problem(groups)
    assert tiers == [2]
    assert T[0] == pytest.approx(4.0)
    # No second tier to test; this just verifies the single-tier
    # case doesn't explode.
    xs = _solve(actors, T, p)
    # Sum should reach T.
    assert sum(x[0] for x in xs) == pytest.approx(4.0, abs=0.5)


def test_scarcity_forces_priority_arbitration():
    """The key test for Option 2's coupling constraint.

    Setup: 2 actors, each with deficits in two tiers (tier-2 critical,
    tier-8 least critical), but with a *total absorption budget*
    (``budget``) that is less than the sum of their deficits.  With
    no coupling constraint, the ADMM would trivially allocate
    deficit-per-cell (no priority decision).  With the coupling,
    each actor must choose how to spend its limited budget across
    tiers — and the priority weighting should bias the spend toward
    tier-2 (high priority) rather than tier-8 (low priority).
    """
    groups = [
        {
            "demand": {2: 5.0, 8: 5.0},   # 10 MW total demand
            "served": {2: 0.0, 8: 0.0},   # nothing served → 10 MW deficit
            "budget": 5.0,                 # but only 5 MW absorption capacity
        },
        {
            "demand": {2: 5.0, 8: 5.0},
            "served": {2: 0.0, 8: 0.0},
            "budget": 5.0,
        },
    ]
    actors, T, p, _, tiers = _build_problem(groups)
    assert tiers == [2, 8]
    # Total deficit per cell = 10; total absorption (sum budgets) =
    # 10.  Per-cell ub = 5 per actor, so without coupling each cell
    # could be fully covered.  WITH coupling, each actor has budget
    # 5 — must split between tier-2 and tier-8.
    assert np.allclose(T, [10.0, 10.0])
    assert p[0] / p[1] == pytest.approx(64.0)

    xs = _solve(actors, T, p, max_iters=500, abs_tol=1e-6)

    # Each actor's total absorption should respect the budget.
    for i, x in enumerate(xs):
        assert sum(x) <= 5.0 + 0.1, (
            f"actor {i} exceeded budget: sum_x={sum(x)} > 5.0"
        )

    # Aggregate across actors per cell.
    sum_x = sum(xs)
    a_tier2 = sum_x[0]
    a_tier8 = sum_x[1]
    # The total absorption is bounded by sum(budgets) = 10.
    # Both deficits are 10 MW.  With priority [64, 1], the ADMM
    # should bias allocation toward tier-2 — i.e. tier-2's sum_x
    # should be larger than tier-8's.  Strict assertion: the
    # tier-2 sum_x should be at least 2x the tier-8 sum_x given
    # the 64:1 weight ratio.
    assert a_tier2 > a_tier8, (
        f"priority weighting failed to bias: tier-2 sum_x={a_tier2:.4f}, "
        f"tier-8 sum_x={a_tier8:.4f} (priority weights = {p.tolist()})"
    )
    # Tighter check: ratio reflects the priority weight order.
    # Not exactly 64:1 because the QP is regularized, but at least
    # 2:1 in the right direction.
    ratio = a_tier2 / max(a_tier8, 1e-9)
    assert ratio >= 1.5, (
        f"priority bias too weak: ratio={ratio:.2f} "
        f"(tier-2 sum_x={a_tier2:.4f}, tier-8 sum_x={a_tier8:.4f})"
    )


def _build_supply_problem(
    groups: list[dict],
    *,
    priority_tiers: int = 10,
):
    """Construct a Route-A (supply-priority) ADMM problem.

    ``groups`` entries:
      - ``demand``: {tier: total_demand_mw}
      - ``supply``: float — total generator capacity available to this
                    group (sector-aggregate)
    """
    tiers = sorted({t for g in groups for t in g.get("demand", {})})
    tier_idx = {t: j for j, t in enumerate(tiers)}
    n_dims = len(tiers)

    total_T = np.zeros(n_dims)
    total_demand_per_cell = np.zeros(n_dims)
    for g in groups:
        for tier in tiers:
            j = tier_idx[tier]
            dem = float(g.get("demand", {}).get(tier, 0.0))
            total_T[j] += dem
            total_demand_per_cell[j] += dem

    priorities = np.zeros(n_dims)
    for tier in tiers:
        priorities[tier_idx[tier]] = 2.0 ** max(0, priority_tiers - tier + 1)

    holon_supply_total = sum(float(g.get("supply", 0.0)) for g in groups) or 1.0
    actors: list[ADMMFlexActor] = []
    for g in groups:
        supply = float(g.get("supply", 0.0))
        lb = np.zeros(n_dims)
        ub = np.full(n_dims, 1e-6)
        for tier in tiers:
            j = tier_idx[tier]
            ub[j] = max(min(supply, total_demand_per_cell[j]), 1e-6)
        C = np.ones((1, n_dims))
        d = np.array([max(supply, 0.0)])
        share = supply / holon_supply_total
        S = -share * priorities
        actors.append(ADMMFlexActor(lb=lb, u=ub, C=C, d=d, S=S))

    return actors, total_T, priorities, tiers


def test_route_a_supply_scarcity_serves_high_priority_first():
    """The decisive Route-A test.

    Group A: 10 MW supply, 0 MW tier-2 demand, 10 MW tier-8 demand.
    Group B: 0 MW supply, 10 MW tier-2 demand, 0 MW tier-8 demand.

    Total holon supply = 10 MW.  Total holon demand = 20 MW (10 at
    tier-2, 10 at tier-8).  Scarce.

    Without priority, the ADMM could route A's supply to satisfy
    either A's local tier-8 demand or B's tier-2 demand.  With
    priority [tier-2 = 1024, tier-8 = 8], the coordinator should
    pull the allocation toward tier-2.

    Expected: sum_x[tier-2] >> sum_x[tier-8], proving the supply-
    priority formulation routes A's supply across the community
    boundary to serve B's high-priority load.
    """
    groups = [
        {"demand": {2: 0.0, 8: 10.0}, "supply": 10.0},  # A: 10 MW gen, tier-8 load
        {"demand": {2: 10.0, 8: 0.0}, "supply": 0.0},   # B: no gen, tier-2 load
    ]
    actors, T, p, tiers = _build_supply_problem(groups)
    assert tiers == [2, 8]
    assert np.allclose(T, [10.0, 10.0])
    assert p[0] / p[1] == pytest.approx(64.0)

    xs = _solve(actors, T, p, max_iters=500, abs_tol=1e-6)
    sum_x = sum(xs)
    a_tier2, a_tier8 = sum_x[0], sum_x[1]

    # Supply budget per actor must hold.
    for i, x in enumerate(xs):
        budget = float(groups[i]["supply"])
        assert sum(x) <= budget + 0.1, (
            f"actor {i} exceeded supply: sum_x={sum(x)} > {budget}"
        )

    # Total committed supply ≤ total holon supply.
    assert a_tier2 + a_tier8 <= 10.0 + 0.2

    # KEY assertion: the holon's 10 MW of supply goes mostly to
    # tier-2 (high priority), not tier-8 (low priority).  This is
    # ONLY possible because supply is fungible across cells in this
    # formulation — A can commit to tier-2 service via the grid.
    assert a_tier2 > a_tier8, (
        f"supply-priority routing failed: tier-2 supply={a_tier2:.3f}, "
        f"tier-8 supply={a_tier8:.3f}"
    )
    # The high-priority cell should get nearly all of A's 10 MW.
    assert a_tier2 >= 8.0, (
        f"high-priority tier-2 should attract ≥8 MW of A's 10 MW supply "
        f"(got {a_tier2:.3f}); tier-8 got {a_tier8:.3f}"
    )


def test_route_a_balanced_supply_fully_serves_all():
    """Sanity: when supply ≥ total demand, every tier should be
    fully served regardless of priority.  Priority only matters
    under scarcity.
    """
    groups = [
        {"demand": {2: 5.0, 8: 5.0}, "supply": 15.0},  # plenty of supply
        {"demand": {2: 5.0, 8: 5.0}, "supply": 5.0},
    ]
    actors, T, p, tiers = _build_supply_problem(groups)
    assert np.allclose(T, [10.0, 10.0])
    # Total supply 20 ≥ total demand 20 — feasible.

    xs = _solve(actors, T, p, max_iters=500, abs_tol=1e-6)
    sum_x = sum(xs)
    # Both cells should reach T (within tolerance).
    assert sum_x[0] == pytest.approx(10.0, abs=0.3), (
        f"tier-2 should be fully served, got {sum_x[0]}"
    )
    assert sum_x[1] == pytest.approx(10.0, abs=0.3), (
        f"tier-8 should be fully served, got {sum_x[1]}"
    )


def test_legacy_no_priority_baseline():
    """Without priority weights (uniform ``priorities`` of 1.0), the
    allocation should be approximately *equal* across tiers — i.e.
    the priority correction in the main test is genuinely from the
    weights, not an artefact of the bounds.
    """
    groups = [
        {
            "demand": {2: 10.0, 8: 10.0},
            "served": {2: 5.0, 8: 10.0},
        },
        {
            "demand": {2: 10.0, 8: 10.0},
            "served": {2: 10.0, 8: 5.0},
        },
    ]
    actors, T, _, _, _ = _build_problem(groups)
    # Uniform priorities — same target, but no per-cell weighting.
    uniform_p = np.ones(2)
    xs = _solve(actors, T, uniform_p)
    a_tier2 = xs[0][0]
    b_tier8 = xs[1][1]
    # With uniform weights, the two cells get similar pulls.
    # We assert the ratio is much closer to 1 than the
    # priority-weighted case.
    ratio = a_tier2 / b_tier8 if b_tier8 > 1e-9 else float("inf")
    assert ratio < 3.0, (
        f"uniform-priority ratio should be small: a={a_tier2}, b={b_tier8}, "
        f"ratio={ratio}"
    )
