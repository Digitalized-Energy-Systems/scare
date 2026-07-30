import numpy as np
import pandas as pd
import pytest

from experiment.eval.compliance import (
    _t_crit_95,
    cluster_ci95,
    compliant_mask,
    frozen_net_mask,
    mean_ci95,
    slack_gate_vacuous_mask,
    slack_pass_mask,
    stale_physics_mask,
)

TEMP = "claims__constraint_compliance__detail__by_variable__temperature__n_checked"
VOLT = "claims__constraint_compliance__detail__by_variable__voltage__n_checked"
UTIL_A = (
    "claims__slack_budget_compliance__detail__per_slack__slack__gas__child-287"
    "__utilization"
)
UTIL_B = (
    "claims__slack_budget_compliance__detail__per_slack__slack__electricity"
    "__child-118__utilization"
)
ORACLE_DRAW = (
    "claims__slack_budget_compliance__detail__per_slack__slack__gas__child-287"
    "__draw_kgs"
)


def test_t_crit_exact_above_the_old_bands():
    """Each old band returned the value at its ceiling, so every CI with df > 30
    was one-directionally too narrow."""
    assert _t_crit_95(31) == pytest.approx(2.0395, abs=5e-4)
    assert _t_crit_95(61) == pytest.approx(1.9996, abs=5e-4)
    assert _t_crit_95(200) == pytest.approx(1.9719, abs=5e-4)
    assert _t_crit_95(31) > 2.021
    assert _t_crit_95(200) > 1.96


def test_t_crit_table_retained_below_31():
    assert _t_crit_95(1) == 12.706
    assert _t_crit_95(30) == 2.042
    assert _t_crit_95(0) == float("inf")


def test_cluster_ci_reduces_to_iid_when_clusters_are_singletons():
    vals = [0.1, 0.4, 0.35, 0.9, 0.55, 0.2]
    m_i, ci_i = mean_ci95(vals)
    m_c, ci_c = cluster_ci95(vals, range(len(vals)))
    assert m_c == pytest.approx(m_i)
    assert ci_c == pytest.approx(ci_i)


def test_cluster_ci_widens_on_replicated_draws():
    vals = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    _, ci_i = mean_ci95(vals)
    _, ci_c = cluster_ci95(vals, ["a", "a", "a", "b", "b", "b"])
    assert ci_c > ci_i


def test_cluster_ci_drops_nans_pairwise():
    m, _ = cluster_ci95([1.0, float("nan"), 3.0], ["a", "a", "b"])
    assert m == pytest.approx(2.0)


def test_frozen_net_mask_needs_a_completed_status():
    df = pd.DataFrame(
        {
            TEMP: [np.nan, 12.0, np.nan],
            VOLT: [40.0, 40.0, 40.0],
            "status": ["ok", "ok", "error"],
        }
    )
    assert list(frozen_net_mask(df)) == [True, False, False]


def test_frozen_net_mask_needs_another_variable_actually_scanned():
    """vm_pu/pressure carry constructor defaults so they are scanned even on an
    unsolved net; t_k is solver-only. But a grid with NO heat sector also scans
    nothing thermal, and firing there would drop it on the opposite signal."""
    df = pd.DataFrame({TEMP: [np.nan], VOLT: [np.nan], "status": ["ok"]})
    assert not frozen_net_mask(df).any()


def test_slack_gate_vacuous_marks_passes_with_no_budget_recorded():
    """pv_peak scenarios declare no slack_budget_pct, so the claim passes with
    nothing measured. Correct, but it must not read as a verified pass."""
    df = pd.DataFrame(
        {
            "claims__slack_budget_compliance__passed": [True, True, False],
            UTIL_A: [np.nan, 0.8, np.nan],
            "status": ["ok"] * 3,
        }
    )
    assert list(slack_gate_vacuous_mask(df)) == [True, False, False]


def test_slack_gate_vacuous_does_not_flag_the_lp_enforced_oracle():
    """The oracle emits budget_mw/draw_kgs and no ``utilization``. Keying off
    ``utilization`` alone marked its whole arm vacuous — inverting the truth,
    since enforced_at_lp is a bound the solve cannot violate."""
    df = pd.DataFrame(
        {
            "claims__slack_budget_compliance__passed": [True, True],
            ORACLE_DRAW: [1.4, np.nan],
            "claims__slack_budget_compliance__detail__enforced_at_lp": [True, True],
            "status": ["ok", "ok"],
        }
    )
    assert not slack_gate_vacuous_mask(df).any()


def test_stale_physics_mask_thresholds_on_failure_fraction():
    df = pd.DataFrame(
        {
            "outcomes__physics_solves__failed": [30.0, 10.0, 0.0, 5.0],
            "outcomes__physics_solves__ok": [70.0, 90.0, 100.0, 0.0],
            "status": ["ok"] * 4,
        }
    )
    assert list(stale_physics_mask(df)) == [True, False, False, True]


def test_masks_are_all_false_when_columns_absent():
    df = pd.DataFrame({"status": ["ok", "ok"]})
    assert not frozen_net_mask(df).any()
    assert not stale_physics_mask(df).any()
    assert not slack_gate_vacuous_mask(df).any()


def test_compliant_mask_excludes_unsolved_rows():
    df = pd.DataFrame(
        {
            "claims__slack_budget_compliance__passed": [True, True],
            "claims__constraint_compliance__passed": [True, True],
            TEMP: [np.nan, 8.0],
            VOLT: [40.0, 40.0],
            "status": ["ok", "ok"],
        }
    )
    assert list(compliant_mask(df)) == [False, True]


def test_slack_pass_is_vacuous_on_rows_with_no_recorded_utilisation():
    """Reproduces the shipped grader: an unrecorded slack cannot exceed its
    budget. ``U.max() <= thr`` reverses this for all-NaN rows."""
    df = pd.DataFrame({UTIL_A: [np.nan], UTIL_B: [np.nan]})
    assert bool(slack_pass_mask(df, tol=0.0).iloc[0])


def test_slack_pass_any_semantics_across_columns():
    df = pd.DataFrame({UTIL_A: [1.02, 1.02, np.nan], UTIL_B: [0.4, np.nan, 0.4]})
    assert list(slack_pass_mask(df, tol=0.05)) == [True, True, True]
    assert list(slack_pass_mask(df, tol=0.0)) == [False, False, True]


def test_slack_eps_admits_utilisation_recorded_as_exactly_one():
    df = pd.DataFrame({UTIL_A: [1.0]})
    assert bool(slack_pass_mask(df, tol=0.0, eps=1e-12).iloc[0])
    assert not bool(slack_pass_mask(df, tol=0.0, eps=-1e-9).iloc[0])
