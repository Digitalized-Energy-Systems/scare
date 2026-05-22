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
