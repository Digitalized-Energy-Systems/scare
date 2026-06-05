"""Regression tests for the supply-priority ADMM feasibility cap.

When total cross-tier demand exceeds holon supply, the L1
sharing-distance term can never drive the primal residual below
tolerance, so the library spuriously logs "reached max iterations". The
fix scales ``total_T`` proportionally so the ADMM has a reachable
target; service fractions are still computed against the original
demand, leaving dispatch semantics unchanged.

Asserts: (1) no "reached max iterations" warning under scarcity, and
(2) priority ordering is preserved (high tiers served, low tiers shed).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from scare.community.supply_priority_admm import allocate_supply_priority


def _drive(coro):
    return asyncio.run(coro)


def _scarcity_scenario():
    """Two actors, one sector, demand across tiers 1/2/4/5 totalling
    0.088 MW against 0.011 MW supply (~13% coverage).
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


class TestSupplyPriorityFeasibilityCap:
    def test_scarcity_no_max_iters_warning(self, caplog) -> None:
        """No 'reached max iterations' warning under scarcity."""
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
        # meta carries both the scaled target and the original demand.
        assert "demand_per_cell" in meta
        assert "T_per_cell" in meta

    def test_priority_ordering_preserved(self) -> None:
        """Tier-1 service fraction >= tier-5 after the feasibility cap:
        the schedule still steers limited supply toward critical loads.
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
        assert sec_frac[1] >= sec_frac[5] - 1e-6, sec_frac
        # Priority weighting concentrates supply: tier 1 served > 0.5
        # even at ~13% total coverage.
        assert sec_frac[1] > 0.5, sec_frac

    def test_zero_supply_returns_all_zero_fractions(self) -> None:
        """Zero controllable supply (orphan-island case) must yield
        ``service_fraction == 0.0`` for every (sector, tier).

        A prior ``sum(supplies) or 1.0`` substituted a 1 MW phantom pool,
        so the waterfall short-circuit produced ``frac = 1.0`` across all
        tiers — L2 told every leader to serve everything and no shed
        happened.
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
        # Every actor's allocation is zero.
        for row in x_per_actor:
            assert all(v == 0.0 for v in row), row
        # meta surfaces the degenerate branch.
        assert meta.get("degenerate_no_supply") is True
        assert meta["holon_supply_total"] == 0.0

    def test_near_zero_supply_uses_waterfall_not_phantom_pool(self) -> None:
        """A tiny-but-positive pool triggers the waterfall cap (not the
        phantom default): tier 1 gets the supply first, tiers 2-4 zero.
        Boundary check that the no-supply branch doesn't fire on a small
        positive pool.
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
        # Not the degenerate-no-supply branch.
        assert meta.get("degenerate_no_supply") is not True
        assert meta["holon_supply_total"] == pytest.approx(0.005)

    def test_abundant_supply_unchanged(self) -> None:
        """When supply >= demand the cap is a no-op: every tier fully
        served and the meta target equals demand.
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
        # Supply abundant: all tiers served at 1.0.
        for t in tiers:
            assert service_fraction["electricity"][t] == pytest.approx(
                1.0, abs=1e-3,
            )
        # No scaling: T_per_cell == demand_per_cell.
        for t_val, d_val in zip(meta["T_per_cell"], meta["demand_per_cell"]):
            assert t_val == pytest.approx(d_val, abs=1e-9)
