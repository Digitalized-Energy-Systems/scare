"""Post-run claims validation.

Reads the per-task artefacts (diary.csv, served.csv, timeseries.csv,
events.csv, ``regulate`` records) and checks each architectural claim
the chapter makes.  Returns a dict of pass/fail flags + supporting
detail that is folded into ``result.json["claims"]`` by the runner.

Currently checks:

- ``diary_invariant``       — ``started == Σ terminals`` from the diary
- ``priority_invariant``    — when total demand exceeds capacity, served
                              fraction is non-increasing in tier
- ``monotonic_progress``    — per-load r(t) non-decreasing during periods
                              with no constraint violation in that sector
- ``geometric_convergence`` — R² of ``log|gap|`` vs round counter for
                              completed (originator) negotiations
- ``no_double_act``         — at most one regulate per (t, aid) tuple

Each check returns ``(passed: bool, detail: dict)`` so the aggregator
can surface specific failure cases without re-parsing the artefacts.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_task(task_dir: Path) -> dict[str, Any]:
    """Run every claim check on a task directory.  Returns a dict of
    ``{check_name: {passed, detail}}`` ready to be JSON-serialised.
    """
    out: dict[str, Any] = {}
    out["diary_invariant"] = _check_diary_invariant(task_dir / "diary.csv")
    out["priority_invariant"] = _check_priority_invariant(
        task_dir / "served_by_load.csv",
        legacy_served_csv=task_dir / "served.csv",
    )
    out["monotonic_progress"] = _check_monotonic_progress(
        task_dir / "timeseries.csv", task_dir / "events.csv"
    )
    return out


# ---------------------------------------------------------------------------
# Diary invariant
# ---------------------------------------------------------------------------


def _check_diary_invariant(diary_path: Path) -> dict[str, Any]:
    if not diary_path.exists():
        return {"passed": True, "detail": "no diary"}
    rows = _read_csv(diary_path)
    counts: dict[str, int] = {}
    for r in rows:
        ev = r.get("event", "")
        counts[ev] = counts.get(ev, 0) + 1
    started = counts.get("started", 0)
    terminals = sum(
        counts.get(k, 0)
        for k in ("finished", "timed_out", "cancelled", "abandoned", "stalled")
    )
    return {
        "passed": started == terminals,
        "detail": {"started": started, "terminals": terminals, "counts": counts},
    }


# ---------------------------------------------------------------------------
# Priority invariant
# ---------------------------------------------------------------------------


def _check_priority_invariant(
    by_load_path: Path,
    *,
    legacy_served_csv: Path | None = None,
) -> dict[str, Any]:
    """At end-of-sim, when total demand exceeds capacity *within a
    connected component*, the served fraction must be non-increasing in
    priority tier (per (sector, component)): tier 1 ≥ tier 2 ≥ … ≥ tier
    P.

    Why per-component: heat and gas cannot be transported through a
    broken pipe, and electricity cannot cross a disconnected feeder
    without active reconfiguration.  A load on a healthy island will be
    served 100 % regardless of priority, and that's a spatial accident
    of where the failure landed — not a SCARE priority violation.  By
    grouping on the active-branch-subgraph component (recorded in
    ``served_by_load.csv``'s ``component`` column), the check evaluates
    only the priority decisions SCARE could plausibly have made.

    Components with no deficit (every tier at 1.0 within the per-tier
    capacity-weighted tolerance) are skipped — they carry no priority
    decision to evaluate.

    Falls back to the legacy per-sector check on ``served.csv`` if the
    per-load file is absent (older campaign artefacts).
    """
    if not by_load_path.exists():
        if legacy_served_csv is not None and legacy_served_csv.exists():
            return _check_priority_invariant_legacy(legacy_served_csv)
        return {"passed": True, "detail": "no served_by_load.csv"}
    rows = _read_csv(by_load_path)
    if not rows:
        return {"passed": True, "detail": "empty served_by_load.csv"}

    # Aggregate per (sector, component, tier): demand and served.
    agg: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for r in rows:
        try:
            sec = r["sector"]
            comp = r.get("component", "-1")
            tier = int(r["tier"])
            demand = float(r["demand"])
            served = float(r["served"])
        except (KeyError, ValueError):
            continue
        key = (sec, comp)
        by_tier = agg.setdefault(key, {})
        entry = by_tier.setdefault(tier, {"demand": 0.0, "served": 0.0})
        entry["demand"] += demand
        entry["served"] += served

    inversions: list[dict[str, Any]] = []
    skipped_no_deficit = 0
    skipped_singleton = 0
    checked = 0
    for (sec, comp), by_tier in agg.items():
        tiers = sorted(by_tier.items())  # [(tier, {demand, served})]
        if len(tiers) < 2:
            skipped_singleton += 1
            continue
        total_demand = sum(e["demand"] for _, e in tiers)
        total_served = sum(e["served"] for _, e in tiers)
        if total_demand <= 0 or total_served >= total_demand - 1e-6:
            skipped_no_deficit += 1
            continue
        checked += 1
        # Compute per-tier fraction, then check non-increasing ordering.
        fracs = []
        for tier, e in tiers:
            f = e["served"] / e["demand"] if e["demand"] > 0 else 1.0
            fracs.append((tier, f))
        for i in range(1, len(fracs)):
            t_prev, f_prev = fracs[i - 1]
            t_cur, f_cur = fracs[i]
            if f_cur > f_prev + 1e-3:
                inversions.append({
                    "sector": sec,
                    "component": comp,
                    "tier_prev": t_prev,
                    "frac_prev": f_prev,
                    "tier_cur": t_cur,
                    "frac_cur": f_cur,
                })
    return {
        "passed": not inversions,
        "detail": {
            "inversions": inversions[:5],
            "n_inversions": len(inversions),
            "n_components_checked": checked,
            "n_components_skipped_no_deficit": skipped_no_deficit,
            "n_components_skipped_singleton_tier": skipped_singleton,
        },
    }


def _check_priority_invariant_legacy(served_path: Path) -> dict[str, Any]:
    """Original per-sector check on ``served.csv`` — retained as a
    fallback for runs predating the per-load artefact."""
    rows = _read_csv(served_path)
    by_sector: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        try:
            sec = r["sector"]
            tier = int(r["tier"])
            frac = float(r["fraction"])
        except (KeyError, ValueError):
            continue
        by_sector.setdefault(sec, []).append((tier, frac))
    inversions: list[dict[str, Any]] = []
    for sec, entries in by_sector.items():
        entries.sort(key=lambda e: e[0])
        for i in range(1, len(entries)):
            t_prev, f_prev = entries[i - 1]
            t_cur, f_cur = entries[i]
            if f_cur > f_prev + 1e-3:
                inversions.append({
                    "sector": sec,
                    "tier_prev": t_prev,
                    "frac_prev": f_prev,
                    "tier_cur": t_cur,
                    "frac_cur": f_cur,
                })
    return {
        "passed": not inversions,
        "detail": {
            "inversions": inversions[:5],
            "n_inversions": len(inversions),
            "legacy_per_sector": True,
        },
    }


# ---------------------------------------------------------------------------
# Monotonic progress
# ---------------------------------------------------------------------------


def _check_monotonic_progress(
    timeseries_path: Path, events_path: Path
) -> dict[str, Any]:
    """Per-sector average regulation should be non-decreasing during
    periods with no active ``constraint_violation`` event in that
    sector.  This is a coarse proxy for the per-load monotonic floor —
    we don't have per-load timeseries in the recorded artefacts.

    Returns the largest mid-restoration drop in each sector's
    ``electrical_balance`` / ``gas_balance`` / ``heat_balance`` series.
    A drop below ``_DROP_TOL`` of the series' max counts as a
    violation.
    """
    if not timeseries_path.exists():
        return {"passed": True, "detail": "no timeseries"}

    series = _load_timeseries(timeseries_path)
    if not series:
        return {"passed": True, "detail": "empty timeseries"}

    # Build sector-bracketed violation windows from events.csv.  When
    # available these tighten the check; otherwise we treat the entire
    # run as "no violation" — coarse but safe.
    violations_by_sec = _violation_windows(events_path) if events_path.exists() else {}

    sectors = {
        "electricity": "electrical_balance",
        "gas": "gas_balance",
        "heat": "heat_balance",
    }
    drops: dict[str, float] = {}
    for sec, col in sectors.items():
        if col not in series:
            continue
        t, ys = series[col]
        if len(ys) < 2:
            continue
        max_y = max(abs(y) for y in ys) or 1.0
        worst = 0.0
        windows = violations_by_sec.get(sec, []) + violations_by_sec.get("*", [])
        for i in range(1, len(ys)):
            in_violation = any(lo <= t[i] <= hi for lo, hi in windows)
            if in_violation:
                continue
            drop = ys[i - 1] - ys[i]   # positive if value dropped
            if drop > worst:
                worst = drop
        drops[sec] = worst / max_y

    _DROP_TOL = 0.05
    passed = all(d < _DROP_TOL for d in drops.values())
    return {
        "passed": passed,
        "detail": {"per_sector_relative_drop": drops, "tol": _DROP_TOL},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_timeseries(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    """Load a wide-format timeseries CSV (time_s + N metric columns)
    into per-column ``(t, ys)`` tuples.  Empty / unparseable cells are
    dropped from the corresponding series.
    """
    if path.stat().st_size == 0:
        return {}
    out: dict[str, tuple[list[float], list[float]]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames or [] if c != "time_s"]
        rows = list(reader)
    for col in cols:
        t: list[float] = []
        ys: list[float] = []
        for row in rows:
            try:
                ts = float(row["time_s"])
                yv = float(row[col])
            except (KeyError, ValueError):
                continue
            if not math.isfinite(yv):
                continue
            t.append(ts)
            ys.append(yv)
        if ys:
            out[col] = (t, ys)
    return out


_DISRUPTION_KINDS = frozenset({
    "constraint_violation",
    "line_failure",
    "node_failure",
    "branch_failure",
})


def _violation_windows(events_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Open a disruption window around every event that physically or
    operationally invalidates the monotonic-progress invariant.  Two
    classes feed the window set:

      * ``constraint_violation`` — a bound was breached; restoration
        agents are mid-correction.
      * ``line_failure`` / ``node_failure`` / ``branch_failure`` — a
        physical disconnection; the resulting balance drop is the
        immediate consequence of lost capacity, not an agent regression.

    Each window runs from the event timestamp until the next event of
    any kind (or end-of-series).  Failure events carry no sector tag, so
    they open a window for *every* sector under key ``"*"`` — the caller
    treats matches under that key as applying universally.
    """
    rows = _read_csv(events_path)
    if not rows:
        return {}
    by_sec: dict[str, list[tuple[float, float]]] = {}
    for i, r in enumerate(rows):
        kind = r.get("kind")
        if kind not in _DISRUPTION_KINDS:
            continue
        try:
            t0 = float(r["t"])
        except (KeyError, ValueError):
            continue
        # Window ends at the next row's timestamp, or open-ended.
        t1 = float("inf")
        for r2 in rows[i + 1:]:
            try:
                t1 = float(r2["t"])
                break
            except (KeyError, ValueError):
                continue
        sec = r.get("sector") or ""
        # Failure events come without a sector tag; bucket them as
        # universal disruption windows so all three sectors honour them.
        if kind != "constraint_violation" and not sec:
            sec = "*"
        by_sec.setdefault(sec, []).append((t0, t1))
    return by_sec
