import csv

import pytest

from experiment.eval.claims import _check_constraint_compliance
from experiment.eval.metrics import is_t_pu_floor_artifact
from scare.base.model import is_energised_reading

VAR_FLOOR_T_K = 106.8  # monee t_pu Var floor 0.3 * t_ref 356


@pytest.mark.parametrize("value", [VAR_FLOOR_T_K, 106.8004, 106.7996])
def test_var_floor_pin_is_an_artifact(value):
    assert is_t_pu_floor_artifact("t_k", value)


@pytest.mark.parametrize(
    "value",
    [
        106.81,  # outside the 1e-3 tolerance
        230.4,  # the below-ambient tail — a different defect, still gates
        296.15,  # ambient
        303.0,  # the median cold junction; genuinely shed, still gates
        313.15,  # the service floor itself
    ],
)
def test_genuinely_cold_readings_still_gate(value):
    assert not is_t_pu_floor_artifact("t_k", value)


@pytest.mark.parametrize("variable", ["vm_pu", "pressure_pu", "loading_percent"])
def test_only_t_k_is_affected(variable):
    assert not is_t_pu_floor_artifact(variable, VAR_FLOOR_T_K)


@pytest.mark.parametrize("value", [None, "", "abc"])
def test_non_numeric_is_not_an_artifact(value):
    assert not is_t_pu_floor_artifact("t_k", value)


def test_predicate_does_not_widen_the_simulator_facing_band():
    """``is_energised_reading`` drives live actuators. A Var-floor pin must
    still read as energised there, or control behaviour changes and every
    recorded run is invalidated."""
    assert is_energised_reading("t_k", VAR_FLOOR_T_K)


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "kind",
                "id",
                "sector",
                "variable",
                "value",
                "lo",
                "hi",
                "overshoot",
                "violated",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def _row(value, violated=1, variable="t_k", sector="heat"):
    return {
        "kind": "node",
        "id": 37,
        "sector": sector,
        "variable": variable,
        "value": f"{value:.6f}",
        "lo": "313.150000",
        "hi": "403.150000",
        "overshoot": "4.585556",
        "violated": violated,
    }


def test_regrade_passes_a_task_whose_only_breaches_are_var_floor_pins(tmp_path):
    """Task 003250's exact shape: three violated rows, all t_k = 106.8."""
    p = tmp_path / "constraints_final.csv"
    _write(p, [_row(VAR_FLOOR_T_K) for _ in range(3)])
    out = _check_constraint_compliance(p)
    assert out["passed"]
    assert out["detail"]["n_violations"] == 0
    assert out["detail"]["n_checked"] == 0


def test_regrade_keeps_failing_a_genuinely_cold_junction(tmp_path):
    p = tmp_path / "constraints_final.csv"
    _write(p, [_row(VAR_FLOOR_T_K), _row(303.0)])
    out = _check_constraint_compliance(p)
    assert not out["passed"]
    assert out["detail"]["n_violations"] == 1
    assert out["detail"]["n_checked"] == 1


def test_artifact_rows_are_excluded_from_n_checked(tmp_path):
    p = tmp_path / "constraints_final.csv"
    _write(p, [_row(VAR_FLOOR_T_K, violated=0), _row(350.0, violated=0)])
    out = _check_constraint_compliance(p)
    assert out["passed"]
    assert out["detail"]["n_checked"] == 1
