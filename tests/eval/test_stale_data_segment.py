"""Tests for ``_stale_data_segment``, used by the constraint-envelope plot.

Detects from which timestamp onward the recorded observations are a stale
held-over snapshot (energyflow recompute returned infeasible, so the prior
``_net_results`` was kept).
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
    # Recompute failed after t=2, so every later row freezes
    # last_feasible_solve_t at 2.0.
    df = pd.DataFrame({
        "time_s":                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "last_feasible_solve_t": [0.0, 1.0, 2.0, 2.0, 2.0, 2.0],
    })
    stale_from, count = _stale_data_segment(df)
    assert stale_from == 3.0
    assert count == 3  # rows at t=3, 4, 5


def test_immediate_freeze_after_init() -> None:
    # Only the initial solve succeeded; every recompute infeasible.
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
