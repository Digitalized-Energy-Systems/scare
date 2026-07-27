"""``variant>oracle`` must be reported split by whether the oracle converged.

Two separate reasons a variant can out-score the oracle, and only one is a
result: the oracle maximises a near-lexicographic tier ladder
(1e6/1e4/1e2/1) but is scored on PWSF's 8:4:2:1, so an honest win exists; and
the oracle may simply have hit the solver time limit. On eval_full_v2 the second
dominates — 469/1435 oracle runs (32.7%) were time-limited, and SCARE beat the
oracle in 5.08% of those pairs versus 0.23% of certified ones.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiment.hpc.aggregate import _as_bool, _paired_vs_oracle_section


@pytest.mark.parametrize("value", [True, "True", "true", 1, 1.0, "1", "yes"])
def test_as_bool_accepts_truthy_forms(value):
    assert _as_bool(value)


@pytest.mark.parametrize("value", [False, "False", "false", 0, 0.0, "", None])
def test_as_bool_rejects_falsy_forms(value):
    """``bool("False")`` is True, so the string form must be special-cased or a
    time-limited oracle would be scored as certified."""
    assert not _as_bool(value)


def test_as_bool_rejects_nan():
    assert not _as_bool(float("nan"))


def _frame(rows):
    return pd.DataFrame(rows)


def _row(variant, seed, pwsf, optimal=None):
    return {
        "grid": "g",
        "variant": variant,
        "seed": seed,
        "scenario": "s",
        "experiment": "e",
        "status": "ok",
        "outcomes__priority_weighted_fraction": pwsf,
        "outcomes__oracle_solve_optimal": optimal,
        "claims__slack_budget_compliance__passed": True,
        "claims__constraint_compliance__passed": True,
        "claims__constraint_compliance__detail__by_variable__temperature__n_checked": 1,
        "outcomes__physics_solves__ok": 10,
        "outcomes__physics_solves__failed": 0,
    }


def test_certified_columns_exclude_time_limited_oracle_wins():
    # seed 0: variant beats a TIME-LIMITED oracle -> counted overall, not certified.
    # seed 1: variant beats a CERTIFIED oracle    -> counted in both.
    # seed 2: variant trails a CERTIFIED oracle   -> counted in neither.
    df = _frame(
        [
            _row("scare", 0, 0.90),
            _row("oracle", 0, 0.80, optimal=False),
            _row("scare", 1, 0.95),
            _row("oracle", 1, 0.90, optimal=True),
            _row("scare", 2, 0.50),
            _row("oracle", 2, 0.90, optimal=True),
        ]
    )
    out = "\n".join(_paired_vs_oracle_section(df))
    assert "n_certified" in out and "variant>oracle (certified)" in out
    body = [ln for ln in out.splitlines() if ln.startswith("| S") or "scare" in ln.lower()]
    line = next(ln for ln in body if "|" in ln and "grid" not in ln)
    # 2 of 3 pairs won overall; only 1 of the 2 certified pairs won.
    assert "2/3" in line, line
    assert "1/2" in line, line


def test_missing_optimal_column_degrades_to_dashes_not_a_crash():
    """Campaigns predating the oracle solver-stats columns must still render."""
    df = _frame([_row("scare", 0, 0.9), _row("oracle", 0, 0.8)])
    df = df.drop(columns=["outcomes__oracle_solve_optimal"])
    out = "\n".join(_paired_vs_oracle_section(df))
    assert "n_certified" in out
