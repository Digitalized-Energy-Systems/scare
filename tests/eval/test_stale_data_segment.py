"""Tests for the stale-data detection used by the constraint-envelope
plot.

The plot relies on ``_stale_data_segment`` to figure out from which
timestamp onward the recorded observation columns are the same
held-over snapshot (because the underlying energyflow recompute
returned infeasible and the behavior kept the previous
``_net_results``).
"""
from __future__ import annotations

import pandas as pd

from experiment.eval.plots import _stale_data_segment


def test_no_last_feasible_solve_column_returns_none() -> None:
    df = pd.DataFrame({"time_s": [0.0, 1.0, 2.0]})
    assert _stale_data_segment(df) == (None, 0)


def test_all_fresh_returns_none() -> None:
    # last_feasible_solve_t advances in lockstep with time_s → no stale rows.
    df = pd.DataFrame({
        "time_s":                [0.0, 1.0, 2.0, 3.0],
        "last_feasible_solve_t": [0.0, 1.0, 2.0, 3.0],
    })
    assert _stale_data_segment(df) == (None, 0)


def test_partial_stale_returns_first_stale_t_and_count() -> None:
    # Recompute succeeded at t=0, t=1, t=2; failed thereafter — every
    # row past t=2 carries last_feasible_solve_t=2.0.
    df = pd.DataFrame({
        "time_s":                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "last_feasible_solve_t": [0.0, 1.0, 2.0, 2.0, 2.0, 2.0],
    })
    stale_from, count = _stale_data_segment(df)
    assert stale_from == 3.0
    assert count == 3  # rows at t=3, 4, 5


def test_immediate_freeze_after_init() -> None:
    # Pathological case (the one observed in eval_full_smoke task 000000):
    # only the initial solve succeeded, every recompute infeasible.
    df = pd.DataFrame({
        "time_s":                [0.0, 0.5, 1.0, 1.5, 2.0],
        "last_feasible_solve_t": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    stale_from, count = _stale_data_segment(df)
    assert stale_from == 0.5
    assert count == 4


def test_empty_dataframe_returns_none() -> None:
    df = pd.DataFrame({"time_s": [], "last_feasible_solve_t": []})
    assert _stale_data_segment(df) == (None, 0)
