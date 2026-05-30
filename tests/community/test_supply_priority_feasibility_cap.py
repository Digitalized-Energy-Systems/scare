"""Regression test for the supply-priority ADMM feasibility cap.

Reproduces the structural infeasibility that produced the misleading
"ADMM reached max iterations" warnings observed on the cp_coalition_eval
campaign (task 2, child-103): total cross-tier demand was ~8× the
holon's available supply, the L1 sharing-distance term could never
drive the primal residual below tolerance, and the library logged
"not converged" despite the dual residual having collapsed to zero.

The fix scales ``total_T`` proportionally when ``sum(demand) >
holon_supply_total``, giving the ADMM a reachable target.  Service
fractions are still computed against the *original* demand so the
dispatch semantics are unchanged.

These tests assert two properties:
1. The ADMM library logger does NOT emit a "reached max iterations"
   warning on the scarcity scenario.
2. Service fractions still respect the priority ordering — tier 1
   gets fully (or maximally) served, low-priority tiers get shed.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from scare.community.supply_priority_admm import allocate_supply_priority


def _drive(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures: scarcity scenario reproducing the bug
# ---------------------------------------------------------------------------


def _scarcity_scenario():
    """Two actors, one sector ("electricity"), demand across tiers
    1/2/4/5 totalling 0.088 MW, total supply 0.011 MW (~13% coverage).

    Matches the shape of the failing case in task 2 (child-103).
    """
    sectors = ["electricity"]
    tiers = [1, 2, 4, 5]
    actor_supplies = [
        {"electricity": 0.007},
        {"electricity": 0.004},
    ]
    actor_demands = [
        {"electricity": {1: 0.004, 2: 0.020, 4: 0.010, 5: 0.008}},
        {"electricity": {1: 0.005, 2: 0.022, 4: 0.008, 5: 0.011}},
    ]
    return sectors, tiers, actor_supplies, actor_demands


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSupplyPriorityFeasibilityCap:
    def test_scarcity_no_max_iters_warning(self, caplog) -> None:
        """The ADMM library must not log a 'reached max iterations'
        warning on the scarcity scenario.  Pre-fix this produced one
        warning per supply-priority call on any deficit-bearing holon.
        """
        sectors, tiers, supplies, demands = _scarcity_scenario()
        caplog.set_level(
            logging.WARNING,
            logger="distributed_resource_optimization.algorithm.admm.core",
        )
        service_fraction, _x, meta = _drive(allocate_supply_priority(
            sectors=sectors,
            tiers=tiers,
            actor_supplies=supplies,
            actor_demands=demands,
            max_iters=50,
            abs_tol=1e-3,
        ))
        max_iter_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "reached max iterations" in r.getMessage()
        ]
        assert not max_iter_warns, (
            f"feasibility cap should silence ADMM max-iter warnings under "
            f"scarcity; got {len(max_iter_warns)}"
        )
        # Sanity: meta carries both the scaled target and the original
        # demand so callers can inspect either.
        assert "demand_per_cell" in meta
        assert "T_per_cell" in meta

    def test_priority_ordering_preserved(self) -> None:
        """Service fraction at tier 1 must be ≥ service fraction at
        tier 5 — the priority schedule must still steer the limited
        supply toward critical loads after the feasibility cap.
        """
        sectors, tiers, supplies, demands = _scarcity_scenario()
        service_fraction, _x, _meta = _drive(allocate_supply_priority(
            sectors=sectors,
            tiers=tiers,
            actor_supplies=supplies,
            actor_demands=demands,
            max_iters=200,
            abs_tol=1e-4,
        ))
        sec_frac = service_fraction["electricity"]
        # Tier 1 (highest priority) ≥ tier 5 (lowest) — the priority
        # invariant the supply-priority ADMM is supposed to enforce.
        assert sec_frac[1] >= sec_frac[5] - 1e-6, sec_frac
        # And tier 1 should be served at a non-trivial fraction even
        # though total supply is ~13% of total demand — the priority
        # weighting concentrates supply on high tiers.
        assert sec_frac[1] > 0.5, sec_frac

    def test_zero_supply_returns_all_zero_fractions(self) -> None:
        """When every actor reports zero controllable supply (the
        orphan-island case after a failure splits a sub-component off
        every grid-forming source), the allocator must return
        ``service_fraction == 0.0`` for every (sector, tier).

        Pre-fix: ``holon_supply_total = sum(supplies) or 1.0`` quietly
        substituted a 1 MW phantom pool; the waterfall short-circuit
        then produced ``frac = demand/demand = 1.0`` across all tiers
        — the L2 told every leader "serve everything", no shed
        happened, and the slack backstopping the island stayed
        over budget.

        Real-world manifestation:
        ``eval_full_small_20260529-181310/tasks/000088`` child-12's
        orphan sub-coord (n_communities=7, supply=0) published
        ``T2=T3=T4=1.0`` every round; result: ``slack__electricity__
        child-39`` settled +10.6% over its 0.10948 MW budget.
        """
        sectors = ["electricity"]
        tiers = [1, 2, 3, 4]
        actor_supplies = [{"electricity": 0.0}, {"electricity": 0.0}]
        actor_demands = [
            {"electricity": {1: 0.02, 2: 0.03, 3: 0.04, 4: 0.02}},
            {"electricity": {1: 0.01, 2: 0.02, 3: 0.01, 4: 0.01}},
        ]
        service_fraction, x_per_actor, meta = _drive(allocate_supply_priority(
            sectors=sectors,
            tiers=tiers,
            actor_supplies=actor_supplies,
            actor_demands=actor_demands,
            max_iters=50,
        ))
        for tier in tiers:
            assert service_fraction["electricity"][tier] == 0.0, (
                f"zero-supply scenario must shed tier {tier}; "
                f"got fraction={service_fraction['electricity'][tier]}"
            )
        # No per-actor commitment — every actor's allocation is zero.
        for row in x_per_actor:
            assert all(v == 0.0 for v in row), row
        # Meta surfaces the degenerate branch for caller introspection.
        assert meta.get("degenerate_no_supply") is True
        assert meta["holon_supply_total"] == 0.0

    def test_near_zero_supply_uses_waterfall_not_phantom_pool(self) -> None:
        """A tiny-but-positive supply pool must still trigger the
        waterfall cap (not the legacy phantom default) and serve only
        what the actual supply covers.

        Tier 1 gets the supply first; tiers 2-4 get zero.  This is the
        scarcity ordering the supply-priority schedule guarantees, and
        is the boundary check that the no-supply fix doesn't kick in
        on a legitimately small (but positive) pool.
        """
        sectors = ["electricity"]
        tiers = [1, 2, 3, 4]
        actor_supplies = [{"electricity": 0.005}, {"electricity": 0.0}]
        actor_demands = [
            {"electricity": {1: 0.01, 2: 0.02, 3: 0.01, 4: 0.01}},
            {"electricity": {1: 0.0,  2: 0.01, 3: 0.0,  4: 0.0}},
        ]
        service_fraction, _x, meta = _drive(allocate_supply_priority(
            sectors=sectors, tiers=tiers,
            actor_supplies=actor_supplies, actor_demands=actor_demands,
            max_iters=50,
        ))
        # Tier 1 served at supply/tier_1_demand = 0.005/0.01 = 0.5.
        sec = service_fraction["electricity"]
        assert sec[1] == pytest.approx(0.5, abs=1e-3), sec
        # Tier 2 onwards: pool exhausted on tier 1, so frac = 0.
        for tier in (2, 3, 4):
            assert sec[tier] == 0.0, sec
        # Definitely not the degenerate-no-supply branch.
        assert meta.get("degenerate_no_supply") is not True
        assert meta["holon_supply_total"] == pytest.approx(0.005)

    def test_abundant_supply_unchanged(self) -> None:
        """When supply ≥ demand the feasibility cap is a no-op: every
        tier should be fully served and the meta target equals demand.
        """
        sectors = ["electricity"]
        tiers = [1, 2, 4]
        actor_supplies = [{"electricity": 1.0}, {"electricity": 1.0}]
        actor_demands = [
            {"electricity": {1: 0.1, 2: 0.2, 4: 0.3}},
            {"electricity": {1: 0.1, 2: 0.1, 4: 0.2}},
        ]
        service_fraction, _x, meta = _drive(allocate_supply_priority(
            sectors=sectors,
            tiers=tiers,
            actor_supplies=actor_supplies,
            actor_demands=actor_demands,
            max_iters=100,
        ))
        # All tiers served at 1.0 — supply abundant.
        for t in tiers:
            assert service_fraction["electricity"][t] == pytest.approx(
                1.0, abs=1e-3,
            )
        # No scaling: T_per_cell == demand_per_cell.
        for t_val, d_val in zip(meta["T_per_cell"], meta["demand_per_cell"]):
            assert t_val == pytest.approx(d_val, abs=1e-9)
