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
    assert "n_certified" in out and "n_usable" in out


# --- three-way verdict ---------------------------------------------------
#
# ``certified`` is a BIASED filter — certified pairs are the shallow-deficit
# tail — so discarding every uncertified pair is not neutral. But an uncertified
# oracle still reports a FEASIBLE incumbent, so where the oracle already matches
# or beats the variant the true optimum is at least as good and the conclusion
# survives. Only "uncertified AND variant ahead" is genuinely indeterminate.


def _row_of(out, needle="scare"):
    return next(
        ln
        for ln in out.splitlines()
        if ln.startswith("|") and needle in ln.lower() and "grid" not in ln
    )


def test_uncertified_pair_the_oracle_wins_is_still_usable():
    # seed 0: uncertified, but the oracle leads -> usable (the optimum is only
    # better, so "variant trails oracle" holds regardless of convergence).
    # seed 1: uncertified and the VARIANT leads -> indeterminate.
    # seed 2: certified -> usable by definition.
    df = _frame(
        [
            _row("scare", 0, 0.70),
            _row("oracle", 0, 0.80, optimal=False),
            _row("scare", 1, 0.95),
            _row("oracle", 1, 0.80, optimal=False),
            _row("scare", 2, 0.60),
            _row("oracle", 2, 0.90, optimal=True),
        ]
    )
    line = _row_of("\n".join(_paired_vs_oracle_section(df)))
    cells = [c.strip() for c in line.strip("|").split("|")]
    # certified keeps 1 of 3 pairs; the three-way verdict keeps 2 and marks 1.
    assert cells[7] == "1", line  # n_certified
    assert cells[10] == "2", line  # n_usable
    assert cells[12] == "1", line  # n_indeterminate


def test_usable_subset_recovers_pairs_certified_discards():
    """The bias case: every pair is uncertified but the oracle leads on all of
    them, so certified reports nothing while usable reports the whole set."""
    df = _frame(
        [
            _row("scare", 0, 0.70),
            _row("oracle", 0, 0.80, optimal=False),
            _row("scare", 1, 0.60),
            _row("oracle", 1, 0.90, optimal=False),
        ]
    )
    line = _row_of("\n".join(_paired_vs_oracle_section(df)))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[7] == "0" and cells[8] == "—", line  # certified: nothing to say
    assert cells[10] == "2", line  # usable: both pairs
    assert cells[11] == "-0.2000", line  # mean(0.65) - mean(0.85)
    assert cells[12] == "0", line


def test_a_variant_win_over_a_certified_oracle_stays_usable():
    """Certified wins are honest (tier ladder vs PWSF weights) and must not be
    filtered out as indeterminate."""
    df = _frame([_row("scare", 0, 0.95), _row("oracle", 0, 0.90, optimal=True)])
    line = _row_of("\n".join(_paired_vs_oracle_section(df)))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[10] == "1" and cells[12] == "0", line
