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
import json
from pathlib import Path

from experiment.eval.claims import (
    _check_diary_invariant,
    _check_heat_priority,
    _check_monotonic_progress,
    _check_priority_invariant,
    _check_slack_budget,
)


_SBL_COLS = (
    "aid", "sector", "tier", "node_id", "component",
    "demand", "served", "fraction", "disconnected", "constraint_allowed",
)


def _write_served_by_load(path: Path, loads: list[dict]) -> Path:
    rows = []
    for ld in loads:
        demand = ld["demand"]
        served = ld["served"]
        rows.append({
            "aid": ld.get("aid", "x"),
            # Default to a checked sector: heat is excluded from the
            # priority-ordering claim (locally temperature-governed), so
            # the mechanics tests (throttle/strand) use electricity.
            "sector": ld.get("sector", "electricity"),
            "tier": ld["tier"],
            "node_id": ld.get("node_id", 0),
            "component": ld.get("component", "0"),
            "demand": demand,
            "served": served,
            "fraction": served / demand if demand else 0.0,
            "disconnected": ld.get("disconnected", 0),
            "constraint_allowed": ld.get("constraint_allowed", 1.0),
        })
    return _write_csv(path, _SBL_COLS, rows)


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


class TestSlackBudgetSteadyState:
    """Steady-state path: judge the slack draw over the final settling
    window from timeseries.csv against slack_meta.json budgets."""

    def _slack_meta(self, tmp_path, budget=10.0, sector="electricity", aid="child-7"):
        meta = {aid: {"sector": sector, "obs_key": "p_mw", "budget": budget,
                      "lp_envelope": budget * 10, "node_id": 1}}
        p = tmp_path / "slack_meta.json"
        p.write_text(json.dumps(meta))
        return p, f"slack__{sector}__{aid}"

    def test_transient_spike_that_recovers_passes(self, tmp_path):
        # Over budget early (cold start), recovers under budget by the
        # settling window → should PASS (the A-type false-positive fix).
        meta, col = self._slack_meta(tmp_path, budget=10.0)
        ts = _write_timeseries(
            tmp_path / "timeseries.csv", (col,),
            [(0.5, -13.0), (1.0, -12.0), (2.0, -10.4),
             (8.5, -9.8), (9.0, -9.7), (10.0, -9.6)],
        )
        # An event was recorded for the early spike, but it recovered.
        ev = _write_events(tmp_path / "events.csv", [
            {"t": 0.5, "kind": "slack_budget_violation", "aid": "child-7",
             "sector": "electricity", "detail": "p_mw=-13.0 budget=10.0"},
        ])
        res = _check_slack_budget(ev, timeseries_path=ts, slack_meta_path=meta)
        assert res["passed"] is True, res["detail"]
        assert res["detail"]["mode"] == "steady_state"
        assert res["detail"]["n_transient_events"] == 1

    def test_sustained_over_budget_fails(self, tmp_path):
        # Over threshold (budget*1.05=10.5) through the settling window.
        meta, col = self._slack_meta(tmp_path, budget=10.0)
        ts = _write_timeseries(
            tmp_path / "timeseries.csv", (col,),
            [(0.5, -11.0), (2.0, -10.8), (8.5, -10.8), (9.0, -10.9), (10.0, -10.85)],
        )
        ev = _write_events(tmp_path / "events.csv", [])
        res = _check_slack_budget(ev, timeseries_path=ts, slack_meta_path=meta)
        assert res["passed"] is False, res["detail"]
        assert res["detail"]["n_steady_breaches"] == 1
        assert res["detail"]["breaches"][0]["slack"] == col

    def test_within_tolerance_passes(self, tmp_path):
        # Draw at budget*1.04 < threshold budget*1.05 → within tol.
        meta, col = self._slack_meta(tmp_path, budget=10.0)
        ts = _write_timeseries(
            tmp_path / "timeseries.csv", (col,),
            [(8.5, -10.4), (9.0, -10.3), (10.0, -10.4)],
        )
        ev = _write_events(tmp_path / "events.csv", [])
        res = _check_slack_budget(ev, timeseries_path=ts, slack_meta_path=meta)
        assert res["passed"] is True, res["detail"]

    def test_falls_back_to_legacy_without_timeseries(self, tmp_path):
        # No timeseries/meta → legacy "any event fails" path.
        ev = _write_events(tmp_path / "events.csv", [
            {"t": 1.5, "kind": "slack_budget_violation", "aid": "child-7",
             "sector": "electricity", "detail": "p_mw=-13"},
        ])
        res = _check_slack_budget(ev)
        assert res["passed"] is False
        assert res["detail"]["mode"] == "legacy"


class TestPriorityInvariantConstraintThrottle:
    """Physics-aware exclusion: a load capped by a local constraint (and
    serving at/near that cap) is not a priority inversion; a load shed
    *below* its physical cap still is."""

    def test_throttled_high_tier_excluded_physics_aware(self, tmp_path):
        # tier-2 load physically capped at 0.0 (constraint_allowed=0),
        # tier-3 fully served.  Strict sees tier2<tier3 inversion;
        # physics-aware excludes the throttled tier-2 load.
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            {"aid": "a", "tier": 2, "demand": 1.0, "served": 0.0, "constraint_allowed": 0.0},
            {"aid": "b", "tier": 3, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
            {"aid": "c", "tier": 3, "demand": 1.0, "served": 0.4, "constraint_allowed": 1.0},
        ])
        strict = _check_priority_invariant(p, exclude_constraint_throttled=False)
        physical = _check_priority_invariant(p, exclude_constraint_throttled=True)
        assert strict["passed"] is False           # raw inversion present
        assert physical["passed"] is True           # throttled tier-2 dropped
        assert physical["detail"]["n_loads_throttled"] == 1

    def test_priority_shed_below_cap_still_counts(self, tmp_path):
        # tier-2 served 0.2 while its physical cap is 0.9 → shed below
        # what physics allows ⇒ a real priority/balance shed, kept even
        # in physics-aware mode; tier-3 fully served ⇒ inversion stands.
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            {"aid": "a", "tier": 2, "demand": 1.0, "served": 0.2, "constraint_allowed": 0.9},
            {"aid": "b", "tier": 3, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
        ])
        physical = _check_priority_invariant(p, exclude_constraint_throttled=True)
        assert physical["passed"] is False
        assert physical["detail"]["n_loads_throttled"] == 0

    def test_unconstrained_inversion_counts_both_modes(self, tmp_path):
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            {"aid": "a", "tier": 2, "demand": 1.0, "served": 0.5, "constraint_allowed": 1.0},
            {"aid": "b", "tier": 3, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
        ])
        assert _check_priority_invariant(p, exclude_constraint_throttled=False)["passed"] is False
        assert _check_priority_invariant(p, exclude_constraint_throttled=True)["passed"] is False

    def test_legacy_csv_without_column_falls_back(self, tmp_path):
        # No constraint_allowed column (old artefact) → physics-aware
        # mode behaves like strict (no exclusion), never crashes.
        cols = ("aid", "sector", "tier", "node_id", "component",
                "demand", "served", "fraction", "disconnected")
        rows = [
            {"aid": "a", "sector": "electricity", "tier": 2, "node_id": 0,
             "component": "0", "demand": 1.0, "served": 0.5, "fraction": 0.5,
             "disconnected": 0},
            {"aid": "b", "sector": "electricity", "tier": 3, "node_id": 0,
             "component": "0", "demand": 1.0, "served": 1.0, "fraction": 1.0,
             "disconnected": 0},
        ]
        p = _write_csv(tmp_path / "served_by_load.csv", cols, rows)
        res = _check_priority_invariant(p, exclude_constraint_throttled=True)
        assert res["passed"] is False
        assert res["detail"]["n_loads_throttled"] == 0


class TestPriorityInvariantHeatExcluded:
    """Heat is locally temperature-governed, so it is dropped from the
    gating priority-ordering claim (``_LOCAL_PHYSICS_SECTORS``)."""

    def test_heat_inversion_does_not_fail_claim(self, tmp_path):
        # A blatant heat inversion (tier-2 shed below tier-3) must NOT
        # flip the gating claim — heat is skipped by default.
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            {"aid": "a", "sector": "heat", "tier": 2, "demand": 1.0, "served": 0.2, "constraint_allowed": 1.0},
            {"aid": "b", "sector": "heat", "tier": 3, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
        ])
        res = _check_priority_invariant(p)
        assert res["passed"] is True
        assert res["detail"]["n_loads_sector_skipped"] == 2
        assert "heat" in res["detail"]["skipped_sectors"]

    def test_electricity_inversion_still_fails_with_heat_present(self, tmp_path):
        # Heat skipped, but a genuine electricity inversion still fails.
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            {"aid": "h1", "sector": "heat", "tier": 2, "demand": 1.0, "served": 0.0, "constraint_allowed": 1.0},
            {"aid": "h2", "sector": "heat", "tier": 3, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
            {"aid": "e1", "sector": "electricity", "tier": 2, "demand": 1.0, "served": 0.5, "constraint_allowed": 1.0},
            {"aid": "e2", "sector": "electricity", "tier": 3, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
        ])
        res = _check_priority_invariant(p)
        assert res["passed"] is False
        assert all(inv["sector"] == "electricity" for inv in res["detail"]["inversions"])

    def test_explicit_empty_skip_set_restores_heat_check(self, tmp_path):
        # Passing skip_sectors=frozenset() opts back into checking heat
        # (used by the diagnostic / strict-validation callers).
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            {"aid": "a", "sector": "heat", "tier": 2, "demand": 1.0, "served": 0.2, "constraint_allowed": 1.0},
            {"aid": "b", "sector": "heat", "tier": 3, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
        ])
        res = _check_priority_invariant(p, skip_sectors=frozenset())
        assert res["passed"] is False


class TestHeatPriorityDiagnostic:
    """The non-gating ``heat_priority`` metric: per-tier feasible-heat
    served fraction, with controllable inversions flagged for the oracle
    diff."""

    def test_reports_per_tier_feasible_and_flags_inversion(self, tmp_path):
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            # tier-1 feasible but shed to 0.3 (controllable error)
            {"aid": "a", "sector": "heat", "tier": 1, "demand": 1.0, "served": 0.3, "constraint_allowed": 1.0},
            # tier-2 feasible fully served → inversion vs tier-1
            {"aid": "b", "sector": "heat", "tier": 2, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
        ])
        res = _check_heat_priority(p)
        assert res["passed"] is False
        assert res["detail"]["n_feasible_inversions"] == 1
        assert res["detail"]["per_tier_served_feasible"]["1"] == 0.3

    def test_temperature_infeasible_load_excluded_from_feasible(self, tmp_path):
        # A tier-1 load capped by temperature (constraint_allowed<1) drops
        # out of the feasible subset → no controllable inversion flagged.
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            {"aid": "a", "sector": "heat", "tier": 1, "demand": 1.0, "served": 0.0, "constraint_allowed": 0.0},
            {"aid": "b", "sector": "heat", "tier": 2, "demand": 1.0, "served": 1.0, "constraint_allowed": 1.0},
        ])
        res = _check_heat_priority(p)
        assert res["passed"] is True
        assert res["detail"]["n_heat_loads_feasible"] == 1

    def test_no_heat_loads_passes_vacuously(self, tmp_path):
        p = _write_served_by_load(tmp_path / "served_by_load.csv", [
            {"aid": "e", "sector": "electricity", "tier": 1, "demand": 1.0, "served": 0.5, "constraint_allowed": 1.0},
        ])
        res = _check_heat_priority(p)
        assert res["passed"] is True
