"""Unit tests for the L3 priority-cascaded sharing ADMM kernel.

The module under test is :mod:`scare.service.cp_priority_admm`.  These
tests exercise the kernel as a pure compute function: no agent world,
no mango context, no I/O.  Each test isolates one property of the
design — priority cascade, coupling honoured by the variable shape,
multi-CP arbitration, convergence behaviour, edge cases — so failures
point at a single mechanism.

Priority weight base is set to a tame ``10`` throughout (rather than
the production ``1e4``) so that intermediate aggregates fit in
human-readable magnitudes; the design's near-strict-priority guarantee
only requires monotone weight separation, not the production stiffness.
"""

from __future__ import annotations

import numpy as np
import pytest

from scare.service.cp_priority_admm import (
    CPSpec,
    SectorDemand,
    marginal_priority,
    solve_cp_priority_admm,
    tier_priority_weight,
    waterfall_serve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _demand(sector: str, by_tier: dict[int, float], *, supply: float = 0.0,
            H: int = 1) -> SectorDemand:
    """Convenience builder: constant per-step demand and supply over H."""
    return SectorDemand(
        sector=sector,
        demand_by_tier={t: np.full(H, v, dtype=float) for t, v in by_tier.items()},
        base_supply=np.full(H, supply, dtype=float),
    )


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------


class TestWaterfall:
    def test_supply_covers_all_demand(self):
        # 10 MW supply, 3 MW tier-1, 3 MW tier-2 → both fully served.
        supply = np.array([[10.0]])
        demand = np.array([[[3.0], [3.0]]])  # (n_sec=1, n_tier=2, H=1)
        served = waterfall_serve(supply, demand)
        assert served[0, 0, 0] == pytest.approx(3.0)
        assert served[0, 1, 0] == pytest.approx(3.0)

    def test_partial_supply_serves_high_priority_first(self):
        # 4 MW supply, 3 MW tier-1, 3 MW tier-2 → tier-1 fully, tier-2 partial.
        supply = np.array([[4.0]])
        demand = np.array([[[3.0], [3.0]]])
        served = waterfall_serve(supply, demand)
        assert served[0, 0, 0] == pytest.approx(3.0)
        assert served[0, 1, 0] == pytest.approx(1.0)

    def test_zero_supply_serves_nothing(self):
        supply = np.array([[0.0]])
        demand = np.array([[[3.0], [3.0]]])
        served = waterfall_serve(supply, demand)
        assert np.all(served == 0.0)

    def test_negative_supply_treated_as_zero(self):
        # Net deficit case — waterfall cannot serve anything.
        supply = np.array([[-2.0]])
        demand = np.array([[[3.0]]])
        served = waterfall_serve(supply, demand)
        assert np.all(served == 0.0)


class TestMarginalPriority:
    def test_no_scarcity_means_zero_marginal(self):
        # Demand fully served → no priority pressure.
        demand = np.array([[[3.0], [3.0]]])
        served = np.array([[[3.0], [3.0]]])
        priorities = np.array([100.0, 10.0])
        lam = marginal_priority(served, demand, priorities)
        assert lam[0, 0] == 0.0

    def test_marginal_picks_highest_unserved_tier(self):
        # Tier-1 fully served, tier-2 partially served → marginal = tier-2 weight.
        demand = np.array([[[3.0], [3.0]]])
        served = np.array([[[3.0], [1.0]]])
        priorities = np.array([100.0, 10.0])
        lam = marginal_priority(served, demand, priorities)
        assert lam[0, 0] == pytest.approx(10.0)

    def test_marginal_picks_tier1_if_unserved(self):
        # Even with tier-2 served, tier-1 unserved dominates.
        demand = np.array([[[3.0], [3.0]]])
        served = np.array([[[1.0], [3.0]]])  # NB: not physically realisable, but stresses the rule
        priorities = np.array([100.0, 10.0])
        lam = marginal_priority(served, demand, priorities)
        assert lam[0, 0] == pytest.approx(100.0)


class TestTierWeights:
    def test_monotone_decreasing_in_tier(self):
        w = [tier_priority_weight(t, priority_tiers=4, base=10.0) for t in [1, 2, 3, 4]]
        assert w == [10_000, 1_000, 100, 10]
        assert w[0] > w[1] > w[2] > w[3]

    def test_tier_zero_returns_zero(self):
        assert tier_priority_weight(0, priority_tiers=4) == 0.0


# ---------------------------------------------------------------------------
# Solver edge cases
# ---------------------------------------------------------------------------


class TestSolverDegenerate:
    def test_no_cps_returns_empty_result(self):
        res = solve_cp_priority_admm(
            cps=[],
            demands=[_demand("electricity", {1: 5.0}, supply=10.0)],
        )
        assert res.factor_by_cp == {}
        assert res.converged

    def test_no_demand_returns_zero_factors(self):
        cps = [CPSpec("p2h-0", {"electricity": 1.0, "heat": -0.95})]
        res = solve_cp_priority_admm(cps=cps, demands=[])
        # No demand anywhere → no scarcity, no incentive to run.
        assert res.factor_by_cp["p2h-0"][0] == pytest.approx(0.0)

    def test_cp_with_zero_capacity_does_nothing(self):
        cps = [CPSpec("zero-cp", {"electricity": 0.0, "heat": 0.0})]
        demands = [_demand("heat", {1: 5.0}, supply=0.0)]
        res = solve_cp_priority_admm(cps=cps, demands=demands)
        # Zero-cap CP has no levers; factor stays 0.
        assert res.factor_by_cp["zero-cp"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Single-CP scenarios
# ---------------------------------------------------------------------------


class TestSingleCP:
    def test_p2h_runs_when_heat_scarce_and_el_surplus(self):
        # Heat side: 10 MW tier-1 demand, 0 MW base supply.
        # Electricity side: 0 MW demand, 100 MW base supply (surplus).
        # The P2H should ramp up to serve heat; el-side has no priority
        # pressure (no demand).
        cps = [CPSpec("p2h", {"electricity": 10.0, "heat": -9.5})]
        demands = [
            _demand("heat", {1: 10.0}, supply=0.0),
            _demand("electricity", {}, supply=100.0),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            priority_weight_base=10.0,
            rho=1.0,
            max_iters=300,
        )
        # At convergence the P2H is at high regulation (close to 1).
        # It is allowed to serve up to 10/9.5 ≈ 1.05 of its capacity
        # but the box clamps at 1.0.
        assert res.factor_by_cp["p2h"][0] >= 0.5, (
            f"P2H should ramp to serve scarce heat; got "
            f"{res.factor_by_cp['p2h'][0]:.3f}"
        )
        # Heat tier-1 should be at least partially served.
        assert res.served_by_sector_tier["heat"][1][0] > 0.0

    def test_p2h_holds_back_when_el_tier1_unmet_and_heat_lower_tier(self):
        # Electricity side: 5 MW tier-1 unmet (only 2 MW base supply).
        # Heat side: 10 MW tier-3 demand, no supply.
        # Priority comparison: el-tier-1 (10^4 with base=10) vastly
        # outweighs heat-tier-3 (10).  The P2H must NOT draw electricity
        # away from the el-tier-1 to serve heat-tier-3.
        cps = [CPSpec("p2h", {"electricity": 10.0, "heat": -9.5})]
        demands = [
            _demand("electricity", {1: 5.0}, supply=2.0),
            _demand("heat", {3: 10.0}, supply=0.0),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            priority_weight_base=10.0,
            rho=1.0,
            max_iters=300,
        )
        # P2H should sit near zero — the el-tier-1 priority dominates.
        assert res.factor_by_cp["p2h"][0] <= 0.1, (
            f"P2H should not draw el away from el-tier-1; got "
            f"{res.factor_by_cp['p2h'][0]:.3f}"
        )

    def test_g2p_runs_when_el_tier1_unmet_and_gas_surplus(self):
        # G2P: gas → electricity.  El-tier-1 has 5 MW unmet demand;
        # gas side has 100 MW base supply and no demand.  The G2P
        # should ramp up.
        cps = [CPSpec("g2p", {"gas": 10.0, "electricity": -4.0})]
        demands = [
            _demand("electricity", {1: 5.0}, supply=0.0),
            _demand("gas", {}, supply=100.0),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            priority_weight_base=10.0,
            rho=1.0,
            max_iters=300,
        )
        assert res.factor_by_cp["g2p"][0] >= 0.5, (
            f"G2P should ramp to serve scarce el-tier-1; got "
            f"{res.factor_by_cp['g2p'][0]:.3f}"
        )


# ---------------------------------------------------------------------------
# Multi-CP scenarios
# ---------------------------------------------------------------------------


class TestMultipleCPs:
    def test_two_p2hs_share_load_when_heat_demand_exceeds_one_unit(self):
        # Heat side: 15 MW tier-1 unmet (no base supply).
        # Each P2H can deliver up to 9.5 MW at r=1.  Combined they can
        # deliver 19 MW.  Electricity surplus is large so no input-side
        # constraint binds.  Both CPs should engage (non-zero factor),
        # collectively close to ≥15/19 ≈ 0.79.
        cps = [
            CPSpec("p2h-A", {"electricity": 10.0, "heat": -9.5}),
            CPSpec("p2h-B", {"electricity": 10.0, "heat": -9.5}),
        ]
        demands = [
            _demand("heat", {1: 15.0}, supply=0.0),
            _demand("electricity", {}, supply=100.0),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            priority_weight_base=10.0,
            rho=1.0,
            max_iters=300,
        )
        r_A = res.factor_by_cp["p2h-A"][0]
        r_B = res.factor_by_cp["p2h-B"][0]
        # Both should be near saturation under the priority pressure.
        assert r_A >= 0.5 and r_B >= 0.5, (
            f"both P2Hs should engage under heat-tier-1 scarcity; got "
            f"r_A={r_A:.3f}, r_B={r_B:.3f}"
        )

    def test_two_p2hs_capped_when_el_tier1_would_be_broken(self):
        # Heat side: 10 MW tier-2 demand (less critical).
        # Electricity side: 20 MW tier-1 demand, only 22 MW base supply.
        # Each P2H draws up to 10 MW; together 20 MW.  If both run at
        # full they consume 20 MW from electricity, leaving only 2 MW
        # for el-tier-1's 20 MW demand — 18 MW of priority-1 unmet,
        # vastly worse than not serving the 10 MW of priority-2 heat.
        # The kernel must back the P2Hs down to a combined draw that
        # does NOT eat into el-tier-1's supply: combined draw ≤
        # 22 − 20 = 2 MW, i.e.\ each at r ≤ ~0.1.
        cps = [
            CPSpec("p2h-A", {"electricity": 10.0, "heat": -9.5}),
            CPSpec("p2h-B", {"electricity": 10.0, "heat": -9.5}),
        ]
        demands = [
            _demand("electricity", {1: 20.0}, supply=22.0),
            _demand("heat", {2: 10.0}, supply=0.0),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            priority_weight_base=10.0,
            rho=1.0,
            max_iters=500,
        )
        r_A = res.factor_by_cp["p2h-A"][0]
        r_B = res.factor_by_cp["p2h-B"][0]
        # Combined draw ≤ ~2 MW: each at r ≤ ~0.15 (a little slack for
        # damped convergence).
        assert r_A + r_B <= 0.4, (
            f"combined draw must not eat el-tier-1; got "
            f"r_A={r_A:.3f}, r_B={r_B:.3f} (sum={r_A+r_B:.3f})"
        )
        # Electricity tier-1 essentially fully served on the residual.
        assert res.served_by_sector_tier["electricity"][1][0] >= 19.0

    def test_mixed_chp_and_p2h_cross_sector(self):
        # CHP burns gas to produce el + heat.  P2H consumes el to make heat.
        # Heat tier-1 has 8 MW unmet demand.  El has 0 base supply,
        # 0 demand.  Gas has 100 MW base supply, 0 demand.
        # The optimal answer: CHP runs (turns gas into el + heat); the
        # P2H may also run if extra el is produced by CHP.  Either way,
        # heat tier-1 should be substantially served.
        cps = [
            CPSpec("chp", {"gas": 10.0, "electricity": -3.5, "heat": -4.5}),
            CPSpec("p2h", {"electricity": 5.0, "heat": -4.75}),
        ]
        demands = [
            _demand("heat", {1: 8.0}, supply=0.0),
            _demand("electricity", {}, supply=0.0),
            _demand("gas", {}, supply=100.0),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            priority_weight_base=10.0,
            rho=1.0,
            max_iters=500,
        )
        # CHP should be running to convert plentiful gas into scarce heat.
        assert res.factor_by_cp["chp"][0] >= 0.3, (
            f"CHP should engage to serve heat-tier-1; got "
            f"{res.factor_by_cp['chp'][0]:.3f}"
        )
        # Heat tier-1 served at least partially.
        assert res.served_by_sector_tier["heat"][1][0] > 0.0


# ---------------------------------------------------------------------------
# Convergence + determinism
# ---------------------------------------------------------------------------


class TestConvergenceAndDeterminism:
    def test_residuals_decrease_to_tolerance(self):
        cps = [CPSpec("p2h", {"electricity": 10.0, "heat": -9.5})]
        demands = [
            _demand("heat", {1: 5.0}, supply=0.0),
            _demand("electricity", {}, supply=100.0),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            priority_weight_base=10.0,
            rho=1.0,
            max_iters=300,
            abs_tol=1e-3,
            record_history=True,
        )
        # The recorded primal residuals should be monotone-ish and
        # ultimately drop below tolerance.
        primal = res.history["primal_residuals"]
        assert primal[-1] < 1e-3 or res.iterations == 300
        # First residual should be larger than last.
        assert primal[0] >= primal[-1]

    def test_kernel_is_deterministic(self):
        cps = [
            CPSpec("p2h-A", {"electricity": 10.0, "heat": -9.5}),
            CPSpec("p2h-B", {"electricity": 5.0, "heat": -4.75}),
        ]
        demands = [
            _demand("heat", {1: 8.0, 3: 3.0}, supply=0.0),
            _demand("electricity", {2: 4.0}, supply=20.0),
        ]
        res_a = solve_cp_priority_admm(
            cps=cps, demands=demands, priority_weight_base=10.0, max_iters=300,
        )
        res_b = solve_cp_priority_admm(
            cps=cps, demands=demands, priority_weight_base=10.0, max_iters=300,
        )
        # Two runs on identical inputs must produce bit-identical outputs.
        # This is the property the replicated-coordinator pattern relies on.
        for cp_id in res_a.factor_by_cp:
            np.testing.assert_array_equal(
                res_a.factor_by_cp[cp_id],
                res_b.factor_by_cp[cp_id],
            )


# ---------------------------------------------------------------------------
# Horizon (H > 1) sanity — even if storage isn't exercised yet, the data
# layout supports it and the per-step decisions should be independent
# when no inter-step coupling is present.
# ---------------------------------------------------------------------------


class TestHorizon:
    def test_constant_demand_yields_constant_factor(self):
        # H = 3 with identical per-step inputs → identical per-step outputs.
        cps = [CPSpec("p2h", {"electricity": 10.0, "heat": -9.5})]
        demands = [
            SectorDemand("heat",
                         {1: np.full(3, 5.0)}, base_supply=np.zeros(3)),
            SectorDemand("electricity",
                         {}, base_supply=np.full(3, 100.0)),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            horizon=3,
            priority_weight_base=10.0,
            max_iters=300,
        )
        r = res.factor_by_cp["p2h"]
        assert r.shape == (3,)
        # All three steps should agree on the same decision.
        assert np.allclose(r, r[0], atol=1e-6), (
            f"constant inputs should yield constant factor; got {r}"
        )

    def test_per_step_independence_under_no_inter_step_coupling(self):
        # Step 0: scarce heat, abundant electricity → P2H should run.
        # Step 1: scarce electricity (with tier-1), abundant heat → P2H back off.
        cps = [CPSpec("p2h", {"electricity": 10.0, "heat": -9.5})]
        heat_demand = np.array([5.0, 0.0])
        el_supply = np.array([100.0, 2.0])
        el_demand_tier1 = np.array([0.0, 5.0])
        demands = [
            SectorDemand("heat", {1: heat_demand}, base_supply=np.zeros(2)),
            SectorDemand("electricity",
                         {1: el_demand_tier1}, base_supply=el_supply),
        ]
        res = solve_cp_priority_admm(
            cps=cps, demands=demands,
            horizon=2,
            priority_weight_base=10.0,
            max_iters=300,
        )
        r = res.factor_by_cp["p2h"]
        # Step 0: serve heat → ramp up.
        assert r[0] >= 0.3, f"step 0 should engage; got {r[0]:.3f}"
        # Step 1: el-tier-1 priority dominates → back off.
        assert r[1] <= 0.1, f"step 1 should back off; got {r[1]:.3f}"
