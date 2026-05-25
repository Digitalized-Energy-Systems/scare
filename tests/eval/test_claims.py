"""Regression tests for the claim-check helpers in
:mod:`experiment.eval.claims` that were previously unguarded:

- :func:`_check_diary_invariant`
- :func:`_check_monotonic_progress`
- :func:`_check_slack_budget`

Recent commits ("fixing monotonic claim", "fixing infeasibilities by
protecting slack regulation") cite fixes to these very paths; without
unit coverage a future refactor can re-break them silently.  The tests
here build minimal CSV artefacts on-disk to exercise the helpers
without spinning up a full simulation.
"""

from __future__ import annotations

import csv
from pathlib import Path

from experiment.eval.claims import (
    _check_diary_invariant,
    _check_monotonic_progress,
    _check_slack_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, cols: tuple[str, ...], rows: list[dict]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    return path


def _write_diary(path: Path, events: list[str]) -> Path:
    return _write_csv(
        path, ("event",), [{"event": ev} for ev in events],
    )


def _write_events(path: Path, rows: list[dict]) -> Path:
    return _write_csv(
        path, ("t", "kind", "aid", "sector", "detail"), rows,
    )


def _write_timeseries(
    path: Path,
    cols: tuple[str, ...],
    rows: list[tuple[float, ...]],
) -> Path:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("time_s",) + cols)
        for row in rows:
            w.writerow(row)
    return path


# ---------------------------------------------------------------------------
# _check_diary_invariant
# ---------------------------------------------------------------------------


class TestDiaryInvariant:
    def test_balanced_started_and_terminals_passes(self, tmp_path):
        # 3 started, 3 terminals (mix of finished / timed_out / cancelled).
        diary = _write_diary(
            tmp_path / "diary.csv",
            ["started", "started", "started", "finished", "timed_out", "cancelled"],
        )
        res = _check_diary_invariant(diary)
        assert res["passed"] is True
        assert res["detail"]["started"] == 3
        assert res["detail"]["terminals"] == 3

    def test_missing_terminal_fails(self, tmp_path):
        # 3 started, 2 terminals — one negotiation leaked.
        diary = _write_diary(
            tmp_path / "diary.csv",
            ["started", "started", "started", "finished", "finished"],
        )
        res = _check_diary_invariant(diary)
        assert res["passed"] is False
        assert res["detail"]["started"] == 3
        assert res["detail"]["terminals"] == 2

    def test_skipped_singleton_does_not_count_as_terminal(self, tmp_path):
        # ``skipped_singleton`` is emitted without a prior ``started``
        # — counting it as a terminal would invert the invariant.
        diary = _write_diary(
            tmp_path / "diary.csv",
            ["skipped_singleton", "skipped_singleton"],
        )
        res = _check_diary_invariant(diary)
        assert res["passed"] is True
        assert res["detail"]["started"] == 0
        assert res["detail"]["terminals"] == 0

    def test_missing_file_passes_vacuously(self, tmp_path):
        res = _check_diary_invariant(tmp_path / "does_not_exist.csv")
        assert res["passed"] is True
        assert res["detail"] == "no diary"

    def test_stalled_counts_as_terminal(self, tmp_path):
        # ``stalled`` is one of the recognized terminal states; if the
        # check forgot it, started=1 / terminals=0 would falsely fail.
        diary = _write_diary(
            tmp_path / "diary.csv", ["started", "stalled"],
        )
        res = _check_diary_invariant(diary)
        assert res["passed"] is True


# ---------------------------------------------------------------------------
# _check_monotonic_progress
# ---------------------------------------------------------------------------


class TestMonotonicProgress:
    def test_steady_climb_passes(self, tmp_path):
        # Monotone increasing electrical_balance after the warmup window.
        ts = _write_timeseries(
            tmp_path / "timeseries.csv",
            ("electrical_balance",),
            [(t / 10.0, 0.5 + 0.05 * t) for t in range(0, 30)],
        )
        ev = _write_events(tmp_path / "events.csv", [])
        res = _check_monotonic_progress(ts, ev)
        assert res["passed"] is True
        assert res["detail"]["per_sector_relative_drop"]["electricity"] < 1e-9

    def test_drop_outside_warmup_and_failure_window_fails(self, tmp_path):
        # Build a series with a big drop at t=5 s — well past warmup,
        # no failure event to excuse it.
        rows = [(t * 0.5, 1.0) for t in range(0, 20)]
        rows.append((10.5, 0.2))  # 80 % aggregate drop
        ts = _write_timeseries(
            tmp_path / "timeseries.csv", ("electrical_balance",), rows,
        )
        ev = _write_events(tmp_path / "events.csv", [])
        res = _check_monotonic_progress(ts, ev)
        assert res["passed"] is False
        assert res["detail"]["per_sector_relative_drop"]["electricity"] > 0.5

    def test_drop_inside_failure_window_is_excused(self, tmp_path):
        # Same drop, but a branch_failure at t=10 s opens a 2 s
        # quiescence window covering t=10.5 s — the drop is physical,
        # not a SCARE regression.
        rows = [(t * 0.5, 1.0) for t in range(0, 20)]
        rows.append((10.5, 0.2))
        ts = _write_timeseries(
            tmp_path / "timeseries.csv", ("electrical_balance",), rows,
        )
        ev = _write_events(
            tmp_path / "events.csv",
            [{"t": 10.0, "kind": "branch_failure", "aid": "", "sector": "", "detail": ""}],
        )
        res = _check_monotonic_progress(ts, ev)
        assert res["passed"] is True

    def test_drop_inside_warmup_is_excused(self, tmp_path):
        # Drop at t=0.5 s is inside the 1 s warmup; everyone starts
        # at regulation=1.0 and the first MAS dispatch lands here.
        ts = _write_timeseries(
            tmp_path / "timeseries.csv",
            ("electrical_balance",),
            [(0.0, 1.0), (0.5, 0.3), (5.0, 0.4), (10.0, 0.4)],
        )
        ev = _write_events(tmp_path / "events.csv", [])
        res = _check_monotonic_progress(ts, ev)
        assert res["passed"] is True

    def test_constraint_violation_window_excuses_drop(self, tmp_path):
        # Drop happens during an active electricity-sector constraint
        # violation — defensive shed, not a regression.
        rows = [(t * 1.0, 1.0) for t in range(0, 6)]
        rows.append((6.0, 0.3))
        ts = _write_timeseries(
            tmp_path / "timeseries.csv", ("electrical_balance",), rows,
        )
        ev = _write_events(
            tmp_path / "events.csv",
            [{"t": 5.5, "kind": "constraint_violation",
              "aid": "", "sector": "electricity", "detail": ""}],
        )
        res = _check_monotonic_progress(ts, ev)
        assert res["passed"] is True

    def test_no_timeseries_passes_vacuously(self, tmp_path):
        res = _check_monotonic_progress(
            tmp_path / "missing.csv", tmp_path / "events.csv",
        )
        assert res["passed"] is True


# ---------------------------------------------------------------------------
# _check_slack_budget
# ---------------------------------------------------------------------------


class TestSlackBudgetCompliance:
    def test_no_events_csv_passes_vacuously(self, tmp_path):
        res = _check_slack_budget(tmp_path / "missing.csv")
        assert res["passed"] is True

    def test_empty_events_csv_passes_vacuously(self, tmp_path):
        ev = _write_events(tmp_path / "events.csv", [])
        res = _check_slack_budget(ev)
        assert res["passed"] is True

    def test_no_violation_events_passes(self, tmp_path):
        ev = _write_events(
            tmp_path / "events.csv",
            [
                {"t": 1.0, "kind": "regulate", "aid": "child-1",
                 "sector": "electricity", "detail": "factor=0.5"},
                {"t": 2.0, "kind": "branch_failure", "aid": "",
                 "sector": "", "detail": "(3,4)"},
            ],
        )
        res = _check_slack_budget(ev)
        assert res["passed"] is True
        assert res["detail"]["n_violations"] == 0

    def test_violation_event_fails(self, tmp_path):
        # A single ``slack_budget_violation`` is enough to fail the
        # operator-policy claim for the run.
        ev = _write_events(
            tmp_path / "events.csv",
            [
                {"t": 1.5, "kind": "slack_budget_violation",
                 "aid": "child-7", "sector": "electricity",
                 "detail": "p_mw=12.3 budget=10.0"},
            ],
        )
        res = _check_slack_budget(ev)
        assert res["passed"] is False
        assert res["detail"]["n_violations"] == 1
        assert res["detail"]["first_t"] == "1.5"
        assert res["detail"]["peaks_sample"][0]["aid"] == "child-7"

    def test_multiple_violations_reports_first_and_last(self, tmp_path):
        rows = [
            {"t": str(t), "kind": "slack_budget_violation",
             "aid": "child-1", "sector": "electricity",
             "detail": f"p_mw={10 + t}"}
            for t in range(1, 8)
        ]
        ev = _write_events(tmp_path / "events.csv", rows)
        res = _check_slack_budget(ev)
        assert res["passed"] is False
        assert res["detail"]["n_violations"] == 7
        assert res["detail"]["first_t"] == "1"
        assert res["detail"]["last_t"] == "7"
        # Peaks sample is capped at 5 to keep the report small.
        assert len(res["detail"]["peaks_sample"]) == 5
