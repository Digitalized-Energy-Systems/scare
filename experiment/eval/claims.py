"""Post-run claims validation.

Reads per-task artefacts (diary.csv, served*.csv, timeseries.csv,
events.csv, constraints_final.csv) and checks each architectural claim.
Returns a dict of pass/fail flags + supporting detail folded into
``result.json["claims"]`` by the runner.

Each check returns ``{"passed": bool, "detail": dict}`` so the
aggregator can surface specific failure cases without re-parsing the
artefacts.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from experiment.eval.metrics import (
    NON_GATING_CONSTRAINT_VARIABLES,
    _variable_tally_entry,
)


def evaluate_task(task_dir: Path) -> dict[str, Any]:
    """Run every claim check on a task directory.  Returns a dict of
    ``{check_name: {passed, detail}}`` ready to be JSON-serialised.
    """
    out: dict[str, Any] = {}
    out["diary_invariant"] = _check_diary_invariant(task_dir / "diary.csv")
    # Physics-aware (gating): excludes loads capped by a local constraint
    # — physically limited, not priority-shed.
    out["priority_invariant"] = _check_priority_invariant(
        task_dir / "served_by_load.csv",
        legacy_served_csv=task_dir / "served.csv",
        exclude_constraint_throttled=True,
        near_full_exempt=True,
    )
    # Strict (diagnostic, non-gating): raw inversion signal with no
    # constraint exclusion.
    out["priority_invariant_strict"] = _check_priority_invariant(
        task_dir / "served_by_load.csv",
        legacy_served_csv=task_dir / "served.csv",
        exclude_constraint_throttled=False,
        near_full_exempt=False,
    )
    out["monotonic_progress"] = _check_monotonic_progress(
        task_dir / "timeseries.csv", task_dir / "events.csv"
    )
    # Diagnostic (non-gating): heat is excluded from the priority-ordering
    # claims; this tracks the per-tier feasible-heat served fraction so the
    # controllable heat-priority gap stays measurable against the oracle.
    out["heat_priority"] = _check_heat_priority(task_dir / "served_by_load.csv")
    out["slack_budget_compliance"] = _check_slack_budget(
        task_dir / "events.csv",
        timeseries_path=task_dir / "timeseries.csv",
        slack_meta_path=task_dir / "slack_meta.json",
    )
    # Grid-feasibility half of the compliance gate: no hard-bound violation
    # (voltage / pressure / temperature / line loading) at end-of-sim.  Paired
    # with ``slack_budget_compliance`` so a run is "compliant" only when it
    # solved the same operator- and physics-constrained problem the oracle did.
    out["constraint_compliance"] = _check_constraint_compliance(
        task_dir / "constraints_final.csv"
    )
    return out


# Diary invariant


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


# Priority invariant


# A load is physically constraint-throttled (excluded from the physics-aware
# check) when its local cap is below ~full AND it serves at/near that cap —
# limited by physics, not a priority/balance shed.  A load served below its
# cap was shed by a decision and still counts.
_THROTTLE_CAP_TOL: float = 0.02

# Sectors excluded from the priority-ordering claims (``priority_invariant``,
# its strict variant, ``monotonic_progress``).  Heat is governed by *local*
# junction-temperature feasibility, not a global priority allocation: a heat
# load on a flow-starved node must be shed regardless of tier, and shedding it
# can recover the node so end-of-sim ``constraint_allowed`` reads ~1.0 —
# making a temperature-forced shed indistinguishable from a priority shed.
# Heat priority is instead tracked by the non-gating ``heat_priority``
# diagnostic.
_LOCAL_PHYSICS_SECTORS: frozenset[str] = frozenset({"heat"})

# Near-full exemption (physics-aware check only).  A higher-priority tier
# served at/above this fraction is treated as essentially fully served: a
# marginal shortfall below a fully-served lower tier is a ramp/convergence
# residual, not a priority decision.  This guards the *higher* tier's
# absolute service, not the gap width — a genuine inversion where both tiers
# are under-served still has the higher tier below this floor and is flagged.
# The strict variant keeps the raw 1e-3 signal (``near_full_exempt=False``).
_NEAR_FULL_FRAC: float = 0.95


def _check_priority_invariant(
    by_load_path: Path,
    *,
    legacy_served_csv: Path | None = None,
    exclude_constraint_throttled: bool = True,
    near_full_exempt: bool = True,
    skip_sectors: frozenset[str] = _LOCAL_PHYSICS_SECTORS,
) -> dict[str, Any]:
    """At end-of-sim, when total demand exceeds capacity within a connected
    component, served fraction must be non-increasing in priority tier per
    (sector, component): tier 1 >= tier 2 >= ... >= tier P.

    ``skip_sectors`` (default ``{"heat"}``): sectors governed by local physics
    rather than global priority are dropped entirely (see
    ``_LOCAL_PHYSICS_SECTORS``); count reported via ``n_loads_sector_skipped``.

    ``exclude_constraint_throttled`` (physics-aware, default): also drop loads
    capped by a local constraint (``served`` at/near ``constraint_allowed``) —
    unservable regardless of priority, so counting them as inversions would
    penalise correctly serving the physically-servable loads.  A load served
    below its cap is still a genuine shed and is kept.  Set False for the
    strict raw-inversion signal.

    Per-component because flow cannot cross a broken pipe / disconnected
    feeder: a load on a healthy island served 100% is a spatial accident of
    where the failure landed, not a priority decision.  Grouping on the
    active-branch-subgraph ``component`` column evaluates only the priority
    decisions SCARE could plausibly have made.

    Loads marked ``disconnected=1`` (no path to a grid-forming source via
    active branches, per monee's ``find_ignored_nodes``) are split into a
    separate ``stranded`` bucket: physically unservable, not a shedding
    decision.  Reported via ``n_loads_stranded``.

    Components with no deficit (every tier at 1.0 within tolerance) are
    skipped — no priority decision to evaluate.

    Falls back to the legacy per-sector check on ``served.csv`` when the
    per-load file is absent.
    """
    if not by_load_path.exists():
        if legacy_served_csv is not None and legacy_served_csv.exists():
            return _check_priority_invariant_legacy(legacy_served_csv)
        return {"passed": True, "detail": "no served_by_load.csv"}
    rows = _read_csv(by_load_path)
    if not rows:
        return {"passed": True, "detail": "empty served_by_load.csv"}

    # Aggregate demand/served per (sector, component, tier); stranded
    # (disconnected) loads go to a separate bucket.
    agg: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    stranded_by_sector: dict[str, dict[int, dict[str, float]]] = {}
    n_loads_stranded = 0
    stranded_demand_mw = 0.0
    n_loads_throttled = 0
    throttled_demand_mw = 0.0
    n_loads_sector_skipped = 0
    sector_skipped_demand_mw = 0.0
    for r in rows:
        try:
            sec = r["sector"]
            comp = r.get("component", "-1")
            tier = int(r["tier"])
            demand = float(r["demand"])
            served = float(r["served"])
        except (KeyError, ValueError):
            continue
        # Locally-governed sectors (heat): excluded — shedding follows
        # junction-temperature feasibility, not tier.
        if sec in skip_sectors:
            n_loads_sector_skipped += 1
            sector_skipped_demand_mw += demand
            continue
        # Stranded loads (no source path) — separate bucket.
        disc_raw = (r.get("disconnected") or "0").strip()
        if disc_raw in ("1", "true", "True"):
            n_loads_stranded += 1
            stranded_demand_mw += demand
            sec_strand = stranded_by_sector.setdefault(sec, {})
            entry = sec_strand.setdefault(tier, {"demand": 0.0, "served": 0.0})
            entry["demand"] += demand
            entry["served"] += served
            continue
        # Constraint-throttled loads (physics-aware mode): capped by a
        # local constraint and serving at/near it — physically limited,
        # not priority-shed.  Excluded like stranded loads.
        if exclude_constraint_throttled and "constraint_allowed" in r:
            try:
                allowed = float(r["constraint_allowed"])
            except (TypeError, ValueError):
                allowed = 1.0
            frac = served / demand if demand > 0 else 1.0
            if (
                allowed < 1.0 - _THROTTLE_CAP_TOL
                and frac >= allowed - _THROTTLE_CAP_TOL
            ):
                n_loads_throttled += 1
                throttled_demand_mw += demand
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
        # Per-tier fraction, then check non-increasing ordering.
        fracs = []
        for tier, e in tiers:
            f = e["served"] / e["demand"] if e["demand"] > 0 else 1.0
            fracs.append((tier, f))
        for i in range(1, len(fracs)):
            t_prev, f_prev = fracs[i - 1]
            t_cur, f_cur = fracs[i]
            if f_cur <= f_prev + 1e-3:
                continue
            # Near-full exemption: a higher tier that is itself essentially
            # fully served is mid-ramp, not shed in favour of the lower one.
            if near_full_exempt and f_prev >= _NEAR_FULL_FRAC:
                continue
            inversions.append(
                {
                    "sector": sec,
                    "component": comp,
                    "tier_prev": t_prev,
                    "frac_prev": f_prev,
                    "tier_cur": t_cur,
                    "frac_cur": f_cur,
                }
            )
    return {
        "passed": not inversions,
        "detail": {
            "inversions": inversions[:5],
            "n_inversions": len(inversions),
            "n_components_checked": checked,
            "n_components_skipped_no_deficit": skipped_no_deficit,
            "n_components_skipped_singleton_tier": skipped_singleton,
            "n_loads_stranded": n_loads_stranded,
            "stranded_demand_mw": stranded_demand_mw,
            "excluded_constraint_throttled": exclude_constraint_throttled,
            "n_loads_throttled": n_loads_throttled,
            "throttled_demand_mw": throttled_demand_mw,
            "skipped_sectors": sorted(skip_sectors),
            "n_loads_sector_skipped": n_loads_sector_skipped,
            "sector_skipped_demand_mw": sector_skipped_demand_mw,
        },
    }


def _check_heat_priority(by_load_path: Path) -> dict[str, Any]:
    """Diagnostic (non-gating) heat-priority metric.

    Heat is excluded from the gating ``priority_invariant`` claim (shedding
    tracks local junction-temperature feasibility, not tier).  This reports
    the per-tier heat served fraction over the *feasible* subset
    (``constraint_allowed`` at full — servable regardless of temperature),
    where served fraction should be non-increasing in tier; a residual
    inversion there is a controllable priority error.

    Honest only relative to the oracle: a flow-starved node is
    temperature-infeasible for SCARE and oracle alike, so the aggregator
    compares ``per_tier_served_feasible`` against the oracle's.  Never gating.
    """
    if not by_load_path.exists():
        return {"passed": True, "detail": "no served_by_load.csv"}
    rows = _read_csv(by_load_path)
    if not rows:
        return {"passed": True, "detail": "empty served_by_load.csv"}
    return heat_priority_from_rows(rows)


def heat_priority_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows-based core of :func:`_check_heat_priority`.

    Accepts the same per-load records ``served_by_load`` produces (str- or
    number-typed ``sector``/``tier``/``demand``/``served``/``disconnected``/
    ``constraint_allowed``), so the oracle path can compute the diagnostic
    in-memory off its solved network without round-tripping through a CSV.
    """
    if not rows:
        return {"passed": True, "detail": "empty served_by_load rows"}

    # Per-tier aggregation over feasible (non-throttled, connected) heat
    # loads, and separately over all heat loads (for the oracle diff).
    feasible: dict[int, dict[str, float]] = {}
    all_heat: dict[int, dict[str, float]] = {}
    n_feasible = 0
    n_total = 0
    for r in rows:
        try:
            if r["sector"] != "heat":
                continue
            tier = int(r["tier"])
            demand = float(r["demand"])
            served = float(r["served"])
        except (KeyError, ValueError):
            continue
        disc_raw = str(r.get("disconnected") or "0").strip()
        if disc_raw in ("1", "true", "True"):
            continue
        n_total += 1
        a = all_heat.setdefault(tier, {"demand": 0.0, "served": 0.0})
        a["demand"] += demand
        a["served"] += served
        try:
            allowed = float(r.get("constraint_allowed", 1.0))
        except (TypeError, ValueError):
            allowed = 1.0
        if allowed >= 1.0 - _THROTTLE_CAP_TOL:
            n_feasible += 1
            f = feasible.setdefault(tier, {"demand": 0.0, "served": 0.0})
            f["demand"] += demand
            f["served"] += served

    if not all_heat:
        return {"passed": True, "detail": "no heat loads"}

    def _fracs(agg: dict[int, dict[str, float]]) -> dict[int, float]:
        return {
            t: (e["served"] / e["demand"] if e["demand"] > 0 else 1.0)
            for t, e in sorted(agg.items())
        }

    per_tier_feasible = _fracs(feasible)
    per_tier_all = _fracs(all_heat)

    # Inversions among the feasible subset (controllable priority errors).
    inversions: list[dict[str, Any]] = []
    tiers = sorted(per_tier_feasible)
    for i in range(1, len(tiers)):
        t_prev, t_cur = tiers[i - 1], tiers[i]
        f_prev, f_cur = per_tier_feasible[t_prev], per_tier_feasible[t_cur]
        if f_cur > f_prev + 1e-3:
            inversions.append(
                {
                    "tier_prev": t_prev,
                    "frac_prev": round(f_prev, 4),
                    "tier_cur": t_cur,
                    "frac_cur": round(f_cur, 4),
                }
            )

    return {
        "passed": not inversions,
        "detail": {
            "per_tier_served_feasible": {
                str(t): round(v, 4) for t, v in per_tier_feasible.items()
            },
            "per_tier_served_all": {
                str(t): round(v, 4) for t, v in per_tier_all.items()
            },
            "n_heat_loads_feasible": n_feasible,
            "n_heat_loads_total": n_total,
            "feasible_inversions": inversions,
            "n_feasible_inversions": len(inversions),
            "gating": False,
        },
    }


def _check_priority_invariant_legacy(served_path: Path) -> dict[str, Any]:
    """Fallback per-sector check on ``served.csv`` for runs lacking the
    per-load artefact."""
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
                inversions.append(
                    {
                        "sector": sec,
                        "tier_prev": t_prev,
                        "frac_prev": f_prev,
                        "tier_cur": t_cur,
                        "frac_cur": f_cur,
                    }
                )
    return {
        "passed": not inversions,
        "detail": {
            "inversions": inversions[:5],
            "n_inversions": len(inversions),
            "legacy_per_sector": True,
        },
    }


# Monotonic progress


_WARMUP_S: float = 1.0
_DROP_TOL: float = 0.05
# A lower-priority tier counts as "still served" (so shedding a higher tier
# above it is a priority-order violation) when its regulation sum exceeds this
# fraction of its own peak.  Below it the tier is treated as already shed, so
# dropping a higher tier is legitimate bottom-up shedding, not a regret switch.
_SHED_EPS: float = 0.1
# Loading-percent above which an electricity line counts as overloaded for the
# line-relief exemption (mirrors the constraint monitor's 100% bound plus the
# numerical-wiggle tolerance ``metrics._LINE_LOADING_TOL_PCT``).
_LINE_OVERLOAD_PCT: float = 101.0


def _check_monotonic_progress(
    timeseries_path: Path, events_path: Path
) -> dict[str, Any]:
    """No-regret-switching check.

    Restoration must not regret-switch — un-serve a load it had restored —
    except to free supply for a higher-priority tier (the intended L3->L2->L1
    re-shed; ``enable_monotonic_floor`` is off by design).

    Primary check (when the per-(sector, tier) ``tier_balance__<sector>__
    <tier>`` series are present): within a sector, a drop in a tier's
    regulation sum is a violation only when a strictly lower-priority tier in
    the *same* sector is still served (above ``_SHED_EPS`` of its peak).
    Bottom-up shedding passes; shedding a higher tier while a lower one stays
    served fails.  Same-sector comparison so cross-sector independence is
    never mis-flagged.

    Falls back to the legacy per-sector aggregate-drop check on artefacts
    lacking the per-tier series.

    Both paths exclude (1) the warm-up window ``t < _WARMUP_S``, (2) active
    ``constraint_violation`` windows, and (3) post-failure ringing
    (``dt <= _POST_FAILURE_S``).
    """
    if not timeseries_path.exists():
        return {"passed": True, "detail": "no timeseries"}

    series = _load_timeseries(timeseries_path)
    if not series:
        return {"passed": True, "detail": "empty timeseries"}

    violations_by_sec = _violation_windows(events_path) if events_path.exists() else {}
    failure_windows = _failure_windows(events_path) if events_path.exists() else []

    # Line-overload exemption (electricity): while a power line is over its
    # thermal rating, the line-relief lever sheds through-load to restore
    # feasibility — a constraint-driven shed (the oracle does the same), not a
    # priority regret.  Derived from the ``max_line_loading_percent`` series
    # rather than the deduped ``constraint_violation`` event so every overloaded
    # instant is exempt.  The line analogue of the ``_LOCAL_PHYSICS_SECTORS``
    # heat exemption.
    line_overload_windows: list[tuple[float, float]] = []
    ml = series.get("max_line_loading_percent")
    if ml:
        t_ml, y_ml = ml
        for tt, yy in zip(t_ml, y_ml):
            if yy > _LINE_OVERLOAD_PCT:
                line_overload_windows.append((tt, tt + _DISRUPTION_GRACE_S))

    def _excluded(sec: str, ti: float) -> bool:
        if ti < _WARMUP_S:
            return True
        windows = violations_by_sec.get(sec, []) + violations_by_sec.get("*", [])
        if any(lo <= ti <= hi for lo, hi in windows):
            return True
        if any(lo <= ti <= hi for lo, hi in failure_windows):
            return True
        if sec == "electricity" and any(
            lo <= ti <= hi for lo, hi in line_overload_windows
        ):
            return True
        return False

    # Primary: per-(sector, tier) priority-order check
    by_sector_tier: dict[str, dict[int, tuple[list[float], list[float]]]] = {}
    for col, (t, ys) in series.items():
        if not col.startswith("tier_balance__"):
            continue
        try:
            _, sec, tier_s = col.split("__", 2)
            tier = int(tier_s)
        except (ValueError, IndexError):
            continue
        by_sector_tier.setdefault(sec, {})[tier] = (t, ys)

    def _legacy_worst_drop(sec: str, col: str) -> float | None:
        """Worst excluded-window-filtered relative drop of the legacy
        per-sector aggregate series, or None when the series is unusable."""
        if col not in series:
            return None
        t, ys = series[col]
        if len(ys) < 2:
            return None
        max_y = max(abs(y) for y in ys) or 1.0
        worst = 0.0
        for i in range(1, len(ys)):
            if _excluded(sec, t[i]):
                continue
            drop = ys[i - 1] - ys[i]
            if drop > worst:
                worst = drop
        return worst / max_y

    if by_sector_tier:
        violations: dict[str, dict[str, Any]] = {}
        skipped_sectors: list[str] = []
        for sec, tier_series in by_sector_tier.items():
            # Heat shedding follows local junction-temperature feasibility,
            # not tier order, so a tier can correctly drop while a lower tier
            # stays served.
            if sec in _LOCAL_PHYSICS_SECTORS:
                skipped_sectors.append(sec)
                continue
            maxes = {
                k: (max(abs(y) for y in ys) or 1.0)
                for k, (_t, ys) in tier_series.items()
            }
            tiers_sorted = sorted(tier_series)
            # _load_timeseries drops empty/NaN cells per column, so positional
            # indices can desync across tiers — look lower tiers up by timestamp.
            at_t = {j: dict(zip(*tier_series[j])) for j in tiers_sorted}
            for k in tiers_sorted:
                t_k, ys_k = tier_series[k]
                lower = [j for j in tiers_sorted if j > k]  # lower-priority tiers
                for i in range(1, len(ys_k)):
                    if _excluded(sec, t_k[i]):
                        continue
                    rel_drop = (ys_k[i - 1] - ys_k[i]) / maxes[k]
                    if rel_drop <= _DROP_TOL:
                        continue
                    # A lower-priority tier still served at this instant?
                    for j in lower:
                        y_j = at_t[j].get(t_k[i])
                        if y_j is not None and y_j > _SHED_EPS * maxes[j]:
                            prev = violations.get(sec, {}).get("rel_drop", 0.0)
                            if rel_drop > prev:
                                violations[sec] = {
                                    "rel_drop": round(rel_drop, 4),
                                    "tier": k,
                                    "lower_tier_served": j,
                                    "t": round(t_k[i], 3),
                                }
                            break
        # Gas tier series exist only on artifacts recorded with gas-sector
        # Sink tiers; older artifacts fall back to the legacy aggregate check.
        gas_drop: float | None = None
        if "gas" not in by_sector_tier:
            gas_drop = _legacy_worst_drop("gas", "gas_balance")
            if gas_drop is not None and gas_drop >= _DROP_TOL:
                violations["gas"] = {
                    "rel_drop": round(gas_drop, 4),
                    "legacy_aggregate": True,
                }
        return {
            "passed": not violations,
            "detail": {
                "mode": "per_tier",
                "violations": violations,
                "tol": _DROP_TOL,
                "shed_eps": _SHED_EPS,
                "warmup_s": _WARMUP_S,
                "skipped_sectors": sorted(set(skipped_sectors)),
                **(
                    {"gas_legacy_relative_drop": round(gas_drop, 4)}
                    if gas_drop is not None
                    else {}
                ),
            },
        }

    # Fallback: legacy per-sector aggregate drop
    # Heat omitted: its shedding tracks local temperature feasibility, not
    # tier order.
    sectors = {
        "electricity": "electrical_balance",
        "gas": "gas_balance",
    }
    drops: dict[str, float] = {}
    for sec, col in sectors.items():
        d = _legacy_worst_drop(sec, col)
        if d is not None:
            drops[sec] = d

    passed = all(d < _DROP_TOL for d in drops.values())
    return {
        "passed": passed,
        "detail": {
            "mode": "legacy_aggregate",
            "per_sector_relative_drop": drops,
            "tol": _DROP_TOL,
            "warmup_s": _WARMUP_S,
        },
    }


_POST_FAILURE_S: float = 2.0


def _failure_windows(events_path: Path) -> list[tuple[float, float]]:
    """Return ``[(t_fail, t_fail + _POST_FAILURE_S)]`` per failure event.
    Branch / node / custom / line failures qualify — each opens a short
    quiescence window in which a monotonic-progress drop is physical, not a
    regression.

    Coalition / holon-priority re-allocations are *not* excluded: silencing
    the aggregate-regulation drop they produce would mask exactly the
    regret-switching this claim detects.  Small per-tick redistributions are
    tolerated by the metric; a coalition flagged here redistributed enough to
    deserve attention.
    """
    rows = _read_csv(events_path)
    windows: list[tuple[float, float]] = []
    for r in rows:
        kind = r.get("kind", "")
        if kind not in (
            "branch_failure",
            "node_failure",
            "custom_failure",
            "line_failure",
        ):
            continue
        try:
            t = float(r.get("t", ""))
        except (TypeError, ValueError):
            continue
        windows.append((t, t + _POST_FAILURE_S))
    return windows


# Slack budget compliance


# Settling window: the slack draw must be within budget at steady state, not
# at every instant.  The initial-dispatch transient (LP draws full import
# before the first MAS dispatch lands) fires a ``slack_budget_violation`` that
# the controller clears within a rebalance round, so judge the draw over the
# final ``_SLACK_SETTLE_S`` seconds: a recovering spike passes, a sustained
# over-draw still fails.
_SLACK_SETTLE_S: float = 2.0
_SLACK_TOL: float = 0.05


def _check_slack_budget(
    events_path: Path,
    *,
    timeseries_path: Path | None = None,
    slack_meta_path: Path | None = None,
) -> dict[str, Any]:
    """Operator slack-budget constraint must hold at steady state.

    The steady-state draw from ``timeseries.csv``'s ``slack__<sector>__<aid>``
    columns is compared against the per-slack budget in ``slack_meta.json``.
    Passed iff the peak draw over the final ``_SLACK_SETTLE_S`` seconds is
    within ``budget * (1 + _SLACK_TOL)`` for every budgeted slack.  Falls back
    to the legacy "any ``slack_budget_violation`` event" check when those
    artefacts are absent.

    Vacuously True when no slack child carries an explicit budget.

    Mirrors the oracle's budget claim so passing both means SCARE and oracle
    solved the same operator-constrained problem and the PWSF gap is a real
    allocation gap, not a policy violation by one side.
    """
    # Always collect the event ledger for diagnostics / fallback.
    rows = _read_csv(events_path) if events_path.exists() else []
    violations = [r for r in rows if r.get("kind") == "slack_budget_violation"]

    budgets = _load_slack_budgets(slack_meta_path)
    series = (
        _load_timeseries(timeseries_path)
        if timeseries_path is not None and timeseries_path.exists()
        else {}
    )

    # Steady-state path (preferred)
    if budgets and series:
        # Settling window taken from the longest slack series.
        t_end = 0.0
        for col in budgets:
            if col in series:
                t, _ys = series[col]
                if t:
                    t_end = max(t_end, t[-1])
        cutoff = max(0.0, t_end - _SLACK_SETTLE_S)

        breaches: list[dict[str, Any]] = []
        per_slack: dict[str, Any] = {}
        for col, budget in budgets.items():
            if col not in series or budget <= 0:
                continue
            t, ys = series[col]
            tail = [abs(y) for ti, y in zip(t, ys) if ti >= cutoff]
            if not tail:
                continue
            peak = max(tail)
            threshold = budget * (1.0 + _SLACK_TOL)
            per_slack[col] = {
                "budget": budget,
                "steady_peak": peak,
                "threshold": threshold,
                "violated": peak > threshold,
            }
            if peak > threshold:
                breaches.append(
                    {
                        "slack": col,
                        "steady_peak": round(peak, 6),
                        "budget": round(budget, 6),
                        "threshold": round(threshold, 6),
                        "over_pct": round(100.0 * (peak / budget - 1.0), 1),
                    }
                )
        return {
            "passed": not breaches,
            "detail": {
                "mode": "steady_state",
                "settle_window_s": _SLACK_SETTLE_S,
                "tol": _SLACK_TOL,
                "n_steady_breaches": len(breaches),
                "breaches": breaches[:5],
                "per_slack": per_slack,
                "n_transient_events": len(violations),
            },
        }

    # Legacy fallback: any violation event fails
    if not events_path.exists():
        return {"passed": True, "detail": "no events.csv (claim vacuous)"}
    if not rows:
        return {"passed": True, "detail": "empty events.csv (claim vacuous)"}
    if not violations:
        return {
            "passed": True,
            "detail": {"n_violations": 0, "enforced_at": "agent", "mode": "legacy"},
        }
    peaks: list[dict[str, Any]] = []
    for r in violations[:5]:
        peaks.append(
            {
                "t": r.get("t"),
                "aid": r.get("aid"),
                "sector": r.get("sector"),
                "detail": (r.get("detail") or "")[:120],
            }
        )
    return {
        "passed": False,
        "detail": {
            "mode": "legacy",
            "n_violations": len(violations),
            "first_t": violations[0].get("t"),
            "last_t": violations[-1].get("t"),
            "peaks_sample": peaks,
        },
    }


# Constraint compliance (end-of-sim grid feasibility)


def _check_constraint_compliance(constraints_path: Path) -> dict[str, Any]:
    """End-of-sim hard-bound feasibility must hold: no active, connected node
    or branch may breach its GATING ``SECTOR_CONSTRAINTS`` envelope (voltage,
    pressure, line/transformer loading).

    Reads ``constraints_final.csv`` (one row per checked variable, off the
    final solved network).  Passed iff no GATING row is flagged ``violated``.

    Heat junction temperature (``t_k``) is NON-GATING (see
    ``NON_GATING_CONSTRAINT_VARIABLES``): a temperature-infeasible heat node
    already serves no load (its ``constraint_allowed`` collapses to ~0), so the
    served-fraction metric already penalises it.  Counting the ``t_k`` breach
    here as well would punish the heat grid twice for the same local physics.
    The ``t_k`` breaches are still reported in the detail (``by_sector`` and
    ``nongating_violations``) but do not affect ``passed``.

    Grid-feasibility companion to ``slack_budget_compliance``: together they
    make a run "compliant" only when it honoured both the operator slack budget
    and the gating physical envelope the oracle LP enforces by construction.
    Without this gate a variant could post a higher PWSF by leaving voltages /
    lines out of bounds — feasibility the oracle cannot buy.

    Vacuously True when the artefact is absent / empty.
    """
    if not constraints_path.exists():
        return {"passed": True, "detail": "no constraints_final.csv (claim vacuous)"}
    rows = _read_csv(constraints_path)
    if not rows:
        return {"passed": True, "detail": "empty constraints_final.csv (claim vacuous)"}

    by_sector: dict[str, dict[str, Any]] = {}
    by_variable: dict[str, dict[str, Any]] = {}
    gating: list[dict[str, Any]] = []
    nongating: list[dict[str, Any]] = []
    for r in rows:
        sec = r.get("sector", "")
        var = r.get("variable", "")
        entry = by_sector.setdefault(
            sec,
            {
                "n_checked": 0,
                "n_violations": 0,
                "worst_overshoot": 0.0,
                "n_nongating_violations": 0,
            },
        )
        var_entry = _variable_tally_entry(by_variable, var)
        entry["n_checked"] += 1
        var_entry["n_checked"] += 1
        if (r.get("violated") or "0").strip() not in ("1", "true", "True"):
            continue
        try:
            overshoot = float(r.get("overshoot", 0.0))
        except (TypeError, ValueError):
            overshoot = 0.0
        entry["n_violations"] += 1
        entry["worst_overshoot"] = max(entry["worst_overshoot"], overshoot)
        var_entry["n_violations"] += 1
        var_entry["worst_overshoot"] = max(var_entry["worst_overshoot"], overshoot)
        rec = {
            "kind": r.get("kind", ""),
            "id": r.get("id", ""),
            "sector": sec,
            "variable": var,
            "value": r.get("value", ""),
            "overshoot": round(overshoot, 6),
        }
        # Temperature (t_k) breaches are tracked but do NOT gate the run — the
        # cold node is already penalised via the served-load metric.
        if var in NON_GATING_CONSTRAINT_VARIABLES:
            entry["n_nongating_violations"] += 1
            nongating.append(rec)
        else:
            gating.append(rec)

    gating.sort(key=lambda d: d["overshoot"], reverse=True)
    nongating.sort(key=lambda d: d["overshoot"], reverse=True)
    return {
        "passed": not gating,
        "detail": {
            "n_checked": len(rows),
            # ``n_violations`` is the GATING count (drives ``passed``); the
            # non-gating (``t_k``) breaches are surfaced separately.
            "n_violations": len(gating),
            "n_nongating_violations": len(nongating),
            "nongating_variables": sorted(NON_GATING_CONSTRAINT_VARIABLES),
            "by_sector": by_sector,
            # Per-variable-type tally (voltage / pressure / temperature /
            # line_load) — a compliance-accompanying diagnostic; ``temperature``
            # is non-gating. Slack violations are tallied by the separate
            # ``slack_budget_compliance`` claim.
            "by_variable": by_variable,
            "violations": gating[:5],
            "nongating_violations": nongating[:5],
        },
    }


def _load_slack_budgets(slack_meta_path: Path | None) -> dict[str, float]:
    """Map ``slack__<sector>__<aid>`` timeseries column -> budget, from
    ``slack_meta.json``.  Skips slacks with a ``null`` budget (heat-side
    ``ExtHydrGrid`` is intentionally unbudgeted)."""
    if slack_meta_path is None or not slack_meta_path.exists():
        return {}
    try:
        import json

        meta = json.loads(slack_meta_path.read_text())
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, float] = {}
    for aid, m in (meta or {}).items():
        budget = m.get("budget")
        sector = m.get("sector")
        if budget is None or sector is None:
            continue
        try:
            out[f"slack__{sector}__{aid}"] = float(budget)
        except (TypeError, ValueError):
            continue
    return out


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


_DISRUPTION_KINDS = frozenset(
    {
        "constraint_violation",
        "line_failure",
        "node_failure",
        "branch_failure",
    }
)


_DISRUPTION_GRACE_S = 2.0


def _violation_windows(events_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Open a disruption window around every event that invalidates the
    monotonic-progress invariant:

      * ``constraint_violation`` — a bound was breached; agents mid-correction.
      * ``line_failure`` / ``node_failure`` / ``branch_failure`` — a physical
        disconnection; the balance drop is lost capacity, not a regression.

    Each window runs from the event timestamp for ``_DISRUPTION_GRACE_S``
    seconds (a fixed grace period rather than "until next event", which dense
    same-timestamp events would collapse to zero).

    Failure events carry no sector tag, so they open a window for every sector
    under key ``"*"``, which the caller treats as universal.
    """
    rows = _read_csv(events_path)
    if not rows:
        return {}
    by_sec: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        kind = r.get("kind")
        if kind not in _DISRUPTION_KINDS:
            continue
        try:
            t0 = float(r["t"])
        except (KeyError, ValueError):
            continue
        t1 = t0 + _DISRUPTION_GRACE_S
        sec = r.get("sector") or ""
        # Failure events come without a sector tag; bucket them as universal.
        if kind != "constraint_violation" and not sec:
            sec = "*"
        by_sec.setdefault(sec, []).append((t0, t1))
    return by_sec
