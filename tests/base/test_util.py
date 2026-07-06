"""Unit tests for scare.base.util — pure functions with no framework deps."""

import math

import pytest

from scare.base.model import Sector
from scare.base.util import (
    aggregate_priority_weight,
    clamp_to_constraints,
    compute_priority_weighted_shares,
    constraint_utilization,
    kgps_to_mw,
    mw_to_kgps,
    obs_capacity,
    obs_constraint_values,
    obs_min_max,
    obs_priority,
    obs_sector,
    obs_setpoint,
)

# ===================================================================
# obs_sector
# ===================================================================


class TestObsSector:
    def test_electricity_p_mw(self):
        assert obs_sector({"p_mw": 5.0}) == Sector.ELECTRICITY

    def test_electricity_p_kw(self):
        assert obs_sector({"p_kw": 5000.0}) == Sector.ELECTRICITY

    def test_electricity_p_mw_capacity(self):
        assert obs_sector({"p_mw_capacity": 10.0}) == Sector.ELECTRICITY

    def test_heat_q_w_set(self):
        assert obs_sector({"q_w_set": 1000.0}) == Sector.HEAT

    def test_heat_q_mvar(self):
        assert obs_sector({"q_mvar": 1.0}) == Sector.HEAT

    def test_empty_returns_none(self):
        assert obs_sector({}) is None

    def test_mass_flow_alone_is_ambiguous(self):
        # Gas and water junctions share obs shape (mass_flow_kgs / pressure_pu),
        # so the heuristic cannot distinguish them and must return None.
        assert obs_sector({"mass_flow_kgs": 0.1, "pressure_pu": 1.0}) is None


# ===================================================================
# obs_capacity
# ===================================================================


class TestObsCapacity:
    def test_p_mw(self):
        assert obs_capacity({"p_mw": -5.0}) == -5.0

    def test_q_w_set(self):
        assert obs_capacity({"q_w_set": 1000.0}) == 1000.0

    def test_mass_flow(self):
        assert obs_capacity({"mass_flow_kgs": 0.3}) == 0.3

    def test_missing_returns_zero(self):
        assert obs_capacity({}) == 0.0

    def test_p_mw_takes_precedence(self):
        assert obs_capacity({"p_mw": 3.0, "mass_flow_kgs": 0.5}) == 3.0


# ===================================================================
# obs_setpoint
# ===================================================================


class TestObsSetpoint:
    def test_with_regulation(self):
        assert obs_setpoint({"p_mw": 10.0, "regulation": 0.5}) == pytest.approx(5.0)

    def test_no_regulation_defaults_to_one(self):
        assert obs_setpoint({"p_mw": 10.0}) == pytest.approx(10.0)

    def test_zero_regulation(self):
        assert obs_setpoint({"p_mw": 10.0, "regulation": 0.0}) == pytest.approx(0.0)


# ===================================================================
# obs_min_max
# ===================================================================


class TestObsMinMax:
    def test_generator_full_regulation(self):
        # cap=-5, sp=-5 => dmin=0, dmax=5
        dmin, dmax = obs_min_max({"p_mw": -5.0, "regulation": 1.0})
        assert dmin == pytest.approx(0.0)
        assert dmax == pytest.approx(5.0)

    def test_load_zero_regulation(self):
        # cap=3, sp=0 => dmin=0, dmax=3
        dmin, dmax = obs_min_max({"p_mw": 3.0, "regulation": 0.0})
        assert dmin == pytest.approx(0.0)
        assert dmax == pytest.approx(3.0)

    def test_load_partial_regulation(self):
        # cap=4, sp=2 => dmin=-2, dmax=2
        dmin, dmax = obs_min_max({"p_mw": 4.0, "regulation": 0.5})
        assert dmin == pytest.approx(-2.0)
        assert dmax == pytest.approx(2.0)


# ===================================================================
# constraint_utilization
# ===================================================================


class TestConstraintUtilization:
    def test_center_is_zero(self):
        assert constraint_utilization(1.0, 0.95, 1.05) == pytest.approx(0.0)

    def test_at_upper_bound(self):
        assert constraint_utilization(1.05, 0.95, 1.05) == pytest.approx(1.0)

    def test_at_lower_bound(self):
        assert constraint_utilization(0.95, 0.95, 1.05) == pytest.approx(1.0)

    def test_beyond_bound_clamped(self):
        assert constraint_utilization(1.10, 0.95, 1.05) == 1.0

    def test_half_way(self):
        # (1.025 - 1.0) / 0.05 = 0.5
        assert constraint_utilization(1.025, 0.95, 1.05) == pytest.approx(0.5)

    def test_zero_span(self):
        assert constraint_utilization(5.0, 5.0, 5.0) == 1.0


# ===================================================================
# clamp_to_constraints
# ===================================================================


class TestClampToConstraints:
    def test_no_constraint_keys_in_obs(self):
        assert clamp_to_constraints(5.0, {"p_mw": 10.0}, Sector.ELECTRICITY) == 5.0

    def test_at_center_no_clamping(self):
        obs = {"p_mw": 10.0, "vm_pu": 1.0}
        assert clamp_to_constraints(5.0, obs, Sector.ELECTRICITY) == 5.0

    def test_undervoltage_load_reduces(self):
        # Load, vm_pu=0.952 => below centre => serving (which pulls V down)
        # worsens it => util=0.96, deadband 0.85 => allowed=(1-.96)/.15=0.267.
        obs = {"p_mw": 10.0, "vm_pu": 0.952}
        result = clamp_to_constraints(5.0, obs, Sector.ELECTRICITY)
        assert result == pytest.approx(2.667, abs=1e-2)

    def test_overvoltage_load_not_reduced(self):
        # Direction-aware: over-voltage is RELIEVED by serving load (it draws V
        # down), so a load is NOT capped by a high-side reading.
        obs = {"p_mw": 10.0, "vm_pu": 1.048}
        assert clamp_to_constraints(5.0, obs, Sector.ELECTRICITY) == 5.0

    def test_zero_capacity(self):
        obs = {"p_mw": 0.0, "vm_pu": 1.04}
        assert clamp_to_constraints(5.0, obs, Sector.ELECTRICITY) == 5.0

    def test_negative_setpoint(self):
        # Under-voltage load, negative setpoint clamps symmetrically in magnitude.
        obs = {"p_mw": 10.0, "vm_pu": 0.952}
        result = clamp_to_constraints(-5.0, obs, Sector.ELECTRICITY)
        assert result == pytest.approx(-2.667, abs=1e-2)


# ===================================================================
# obs_priority
# ===================================================================


class TestObsPriority:
    def test_explicit(self):
        assert obs_priority({"p_mw": 3.0, "priority": 5}) == 5

    def test_generator_implicit(self):
        # Generator with partial regulation has dmin < 0 (can reduce output)
        assert obs_priority({"p_mw": -5.0, "regulation": 0.5}) == 0

    def test_load_implicit(self):
        # Unannotated loads default to tier 4 (sheddable) under the
        # 4-tier model — see ``obs_priority`` for the rationale: tier 1
        # is hard-locked at L1, so falling back to tier 1 would
        # catastrophically over-assign critical priority to loads
        # whose actual priority was never registered.
        assert obs_priority({"p_mw": 3.0, "regulation": 0.0}) == 4


# ===================================================================
# obs_constraint_values
# ===================================================================


class TestObsConstraintValues:
    def test_electricity_present(self):
        result = obs_constraint_values({"vm_pu": 1.02, "p_mw": 5.0}, Sector.ELECTRICITY)
        assert result == {"vm_pu": 1.02}

    def test_electricity_missing(self):
        result = obs_constraint_values({"p_mw": 5.0}, Sector.ELECTRICITY)
        assert result == {}

    def test_gas(self):
        result = obs_constraint_values({"pressure_pu": 0.95}, Sector.GAS)
        assert result == {"pressure_pu": 0.95}

    def test_heat(self):
        result = obs_constraint_values({"t_k": 370.0}, Sector.HEAT)
        assert result == {"t_k": 370.0}

    def test_loading_direct_key_is_percent(self):
        result = obs_constraint_values(
            {"loading_percent": 87.5}, Sector.ELECTRICITY
        )
        assert result == {"loading_percent": 87.5}

    def test_loading_line_from_side_fraction(self):
        # PowerLine: no MVA rating, sane per-unit fraction -> x100.
        result = obs_constraint_values(
            {
                "loading_from_pu": 0.9421,
                "loading_to_pu": 0.9421,
                "max_i_ka": 0.276,
                "max_s_mva": None,
                "p_from_mw": 0.06,
                "q_from_mvar": 0.01,
            },
            Sector.ELECTRICITY,
        )
        assert result["loading_percent"] == pytest.approx(94.21)

    def test_loading_trafo_mva_basis_ignores_inflated_to_side(self):
        # 20/0.4 kV trafo: loading_to_pu is inflated by the voltage ratio
        # (50x) because max_i_ka is from-side based; the MVA basis from the
        # solved flows is the graded truth (~80%), not 4009%.
        result = obs_constraint_values(
            {
                "loading_from_pu": 0.8019,
                "loading_to_pu": 40.0966,
                "max_i_ka": 0.011547,
                "max_s_mva": 0.4,
                "p_from_mw": 0.3186,
                "q_from_mvar": 0.032,
                "p_to_mw": -0.317,
                "q_to_mvar": -0.03,
            },
            Sector.ELECTRICITY,
        )
        assert result["loading_percent"] == pytest.approx(
            100.0 * math.hypot(0.3186, 0.032) / 0.4
        )
        assert result["loading_percent"] < 100.0

    def test_loading_unbounded_rating_sentinel_skipped(self):
        result = obs_constraint_values(
            {"loading_from_pu": 0.001, "loading_to_pu": 0.001, "max_i_ka": 999.0},
            Sector.ELECTRICITY,
        )
        assert "loading_percent" not in result

    def test_loading_nan_fraction_skipped(self):
        result = obs_constraint_values(
            {"loading_from_pu": float("nan"), "max_i_ka": 0.2},
            Sector.ELECTRICITY,
        )
        assert "loading_percent" not in result


# ===================================================================
# Unit conversions
# ===================================================================


class TestConversions:
    def test_mw_kgps_roundtrip(self):
        assert kgps_to_mw(mw_to_kgps(10.0)) == pytest.approx(10.0)

    def test_kgps_mw_roundtrip(self):
        assert mw_to_kgps(kgps_to_mw(0.5)) == pytest.approx(0.5)


# ===================================================================
# compute_priority_weighted_shares
# ===================================================================


class TestComputePriorityWeightedShares:
    def test_single_group_gets_everything(self):
        shares = compute_priority_weighted_shares(
            [{1: 5.0}], [{1: 0.0}], total_available=5.0
        )
        assert shares == [pytest.approx(5.0)]

    def test_two_groups_same_tier_proportional(self):
        """Two groups with same priority tier split proportionally."""
        shares = compute_priority_weighted_shares(
            [{1: 3.0}, {1: 7.0}],
            [{1: 0.0}, {1: 0.0}],
            total_available=10.0,
        )
        assert shares[0] == pytest.approx(3.0)
        assert shares[1] == pytest.approx(7.0)

    def test_high_priority_served_first(self):
        """Group A has tier-1 demand, group B has tier-3 demand.
        With only 5 MW available, group A gets its full 3 MW first."""
        shares = compute_priority_weighted_shares(
            [{1: 3.0}, {3: 5.0}],
            [{1: 0.0}, {3: 0.0}],
            total_available=5.0,
        )
        assert shares[0] == pytest.approx(3.0)
        assert shares[1] == pytest.approx(2.0)

    def test_mixed_tiers_waterfall(self):
        """Both groups have tier-1 and tier-3 demand.
        Available = 6 MW, tier-1 total = 4 MW, so tier-1 fully served,
        then 2 MW left for tier-3."""
        shares = compute_priority_weighted_shares(
            [{1: 2.0, 3: 4.0}, {1: 2.0, 3: 6.0}],
            [{1: 0.0, 3: 0.0}, {1: 0.0, 3: 0.0}],
            total_available=6.0,
        )
        # Tier 1: 2+2=4, fully served. Remaining=2
        # Tier 3: 4+6=10, proportional: A=2*4/10=0.8, B=2*6/10=1.2
        assert shares[0] == pytest.approx(2.0 + 0.8)
        assert shares[1] == pytest.approx(2.0 + 1.2)

    def test_already_served_demand_skipped(self):
        """Demand that is already served doesn't consume allocation."""
        shares = compute_priority_weighted_shares(
            [{1: 5.0}, {1: 5.0}],
            [{1: 5.0}, {1: 0.0}],  # group A fully served
            total_available=5.0,
        )
        assert shares[0] == pytest.approx(0.0)
        assert shares[1] == pytest.approx(5.0)

    def test_zero_available(self):
        shares = compute_priority_weighted_shares(
            [{1: 5.0}], [{1: 0.0}], total_available=0.0
        )
        assert shares == [0.0]

    def test_empty_groups(self):
        assert compute_priority_weighted_shares([], [], total_available=10.0) == []

    def test_no_demand_groups(self):
        """Groups with empty demand dicts get nothing."""
        shares = compute_priority_weighted_shares(
            [{}, {}], [{}, {}], total_available=10.0
        )
        assert shares == [0.0, 0.0]

    def test_scarce_resource_all_to_critical(self):
        """Only 1 MW available, group A has 10 MW tier-1, group B has 10 MW tier-4.
        All goes to group A (tier-1 first)."""
        shares = compute_priority_weighted_shares(
            [{1: 10.0}, {4: 10.0}],
            [{1: 0.0}, {4: 0.0}],
            total_available=1.0,
        )
        assert shares[0] == pytest.approx(1.0)
        assert shares[1] == pytest.approx(0.0)


# ===================================================================
# aggregate_priority_weight
# ===================================================================


class TestAggregatePriorityWeight:
    def test_fully_unserved_high_priority(self):
        w = aggregate_priority_weight({1: 5.0}, {1: 0.0})
        assert w > 0

    def test_fully_served_zero_weight(self):
        w = aggregate_priority_weight({1: 5.0}, {1: 5.0})
        assert w == pytest.approx(0.0)

    def test_high_priority_heavier(self):
        w_high = aggregate_priority_weight({1: 1.0}, {1: 0.0})
        w_low = aggregate_priority_weight({4: 1.0}, {4: 0.0})
        assert w_high > w_low

    def test_empty_dicts(self):
        assert aggregate_priority_weight({}, {}) == pytest.approx(0.0)

    def test_partially_served(self):
        w_full = aggregate_priority_weight({1: 10.0}, {1: 0.0})
        w_half = aggregate_priority_weight({1: 10.0}, {1: 5.0})
        assert w_full > w_half > 0
