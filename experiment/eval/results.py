"""Per-task result writers for the evaluation harness.

Called once at the end of a run (from the runner), produces:

- ``result.json``    extended schema with outcomes, diary summary, events
- ``served.csv``     per-(sector, tier) demand / served / fraction
- ``diary.csv``      one row per ``NegotiationRecord``
- ``events.csv``     one row per ``EventRecord``
- ``messages.csv``   per-message-type counts (off by default)

The schema is the source of truth for the aggregator and the claims
checker; both read these artefacts back without going near the in-process
diagnostics state.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from monee.model.child import ExtHydrGrid, ExtPowerGrid

from scare.base import diagnostics
from scare.base.model import Sector
from scare.base.util import sector_from_grid

from experiment.eval.metrics import (
    constraint_rows,
    constraint_violation_integral,
    constraint_violations_final,
    restoration_breakdown,
    served_breakdown,
    served_by_load,
    time_to_stabilise_s,
)


# ---------------------------------------------------------------------------
# Result composition
# ---------------------------------------------------------------------------


def compose_result(
    *,
    world: Any,
    monee_net: Any,
    behavior: Any,
    task_meta: dict[str, Any],
    wallclock_s: float,
    completed: bool,
    extra_metrics: dict[str, Any] | None = None,
    priorities: dict[str, int] | None = None,
    baseline_served: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured result.json payload for a scare-variant run."""

    served = served_breakdown(monee_net, behavior, priorities=priorities)
    restoration = restoration_breakdown(served, baseline_served)
    integral = constraint_violation_integral(world)
    # Accurate end-of-sim feasibility: the during-run integral above only sees
    # per-sector averages, so a run with individual buses / junctions / lines
    # out of bounds can still integrate near-zero.  This scans the final solved
    # network node-by-node and branch-by-branch against the same envelope the
    # oracle enforces; the ``constraint_compliance`` claim (claims.py) gates on
    # it so PWSF stays comparable across variants and the oracle.
    constraints_final = constraint_violations_final(monee_net)
    t_stable = time_to_stabilise_s(world)

    diary_summary = diagnostics.negotiation_summary()
    diary_invariant = _diary_invariant_holds(diary_summary)

    event_summary = diagnostics.event_summary()

    # Per-message-type counts via mango's recorded_messages, if any.
    msg_counts: dict[str, int] = {}
    for rec in getattr(world, "recorded_messages", []) or []:
        # mango records typically expose ``content_type`` or similar;
        # fall back to repr of the message content if not.
        msg_type = (
            getattr(rec, "content_type", None)
            or type(getattr(rec, "content", rec)).__name__
        )
        msg_counts[str(msg_type)] = msg_counts.get(str(msg_type), 0) + 1

    # Per-reason regulate counts.
    regulates_by_reason: Counter[str] = Counter()
    for rec in diagnostics._log:  # type: ignore[attr-defined]
        if rec.kind == "regulate":
            regulates_by_reason[rec.reason] += 1

    clock_t = float(getattr(getattr(world, "clock", None), "time", 0.0))

    payload: dict[str, Any] = {
        "task": task_meta,
        "wallclock_s": wallclock_s,
        "completed": completed,
        "sim_time_final": clock_t,
        "outcomes": {
            "priority_weighted_demand": served["priority_weighted_demand"],
            "priority_weighted_served": served["priority_weighted_served"],
            "priority_weighted_fraction": served["priority_weighted_fraction"],
            "served_by_sector": served["by_sector"],
            "served_by_tier": served["by_tier"],
            "served_by_tier_sector": served["by_tier_sector"],
            "n_loads": served["n_loads"],
            "n_loads_served_zero": served["n_loads_served_zero"],
            "constraint_violation_integral": integral,
            "constraint_violations_final": constraints_final,
            "time_to_stabilise_s": t_stable,
            "regulates_total": sum(regulates_by_reason.values()),
            "regulates_by_reason": dict(regulates_by_reason),
            "restoration": restoration,
        },
        "diary": {**diary_summary, "invariant_holds": diary_invariant},
        "events": event_summary,
        "messages": msg_counts,
    }
    if extra_metrics:
        payload.update(extra_metrics)
    return payload


def _diary_invariant_holds(summary: dict[str, int]) -> bool:
    started = int(summary.get("started", 0))
    terminals = sum(
        int(summary.get(k, 0))
        for k in ("finished", "timed_out", "cancelled", "abandoned", "stalled")
    )
    return started == terminals


# ---------------------------------------------------------------------------
# Artefact writers
# ---------------------------------------------------------------------------


def write_result_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def write_served_csv(
    path: Path,
    monee_net: Any,
    behavior: Any,
    priorities: dict[str, int] | None = None,
) -> None:
    """One row per (sector, tier): demand, served, fraction."""
    served = served_breakdown(monee_net, behavior, priorities=priorities)
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sector", "tier", "demand", "served", "fraction"])
        for sec, tiers in sorted(served["by_tier_sector"].items()):
            for tier, entry in sorted(tiers.items()):
                w.writerow([
                    sec,
                    tier,
                    f"{entry['demand']:.6f}",
                    f"{entry['served']:.6f}",
                    f"{entry['fraction']:.6f}",
                ])


def write_served_by_load_csv(
    path: Path,
    monee_net: Any,
    behavior: Any,
    priorities: dict[str, int] | None = None,
) -> None:
    """Per-load detail with sector, tier, node_id, component, demand,
    served, fraction, disconnected.  The ``component`` column carries
    the active-branch-subgraph connected-component index so the
    priority-invariant claim check can compare tiers within each
    post-failure island fairly (cross-island deficits are spatial
    accidents and shouldn't count as priority inversions).
    """
    rows = served_by_load(monee_net, behavior, priorities=priorities)
    cols = (
        "aid", "sector", "tier", "node_id", "component",
        "demand", "served", "fraction", "disconnected", "constraint_allowed",
    )
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["aid"], r["sector"], r["tier"], r["node_id"], r["component"],
                f"{r['demand']:.6f}", f"{r['served']:.6f}", f"{r['fraction']:.6f}",
                r["disconnected"], f"{r.get('constraint_allowed', 1.0):.6f}",
            ])


def write_constraints_final_csv(path: Path, monee_net: Any) -> None:
    """Per-node / per-branch end-of-sim hard-bound readings.

    One row per checked constraint variable: ``kind`` (node|branch), ``id``,
    ``sector``, ``variable``, ``value``, ``lo``, ``hi``, ``overshoot``,
    ``violated``.  The ``constraint_compliance`` claim
    (``experiment.eval.claims._check_constraint_compliance``) reads this back
    and passes iff no row is ``violated`` — the grid-feasibility half of the
    compliance gate that keeps PWSF comparable to the oracle.
    """
    rows = constraint_rows(monee_net)
    cols = (
        "kind", "id", "sector", "variable",
        "value", "lo", "hi", "overshoot", "violated",
    )
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["kind"], r["id"], r["sector"], r["variable"],
                f"{r['value']:.6f}", f"{r['lo']:.6f}", f"{r['hi']:.6f}",
                f"{r['overshoot']:.6f}", int(bool(r["violated"])),
            ])


def write_slack_meta(path: Path, monee_net: Any) -> None:
    """Persist per-slack-child metadata (budget + LP envelope) so the
    post-run plot tooling can overlay the operator policy on the
    measured trajectory.

    Output schema (one entry per slack-class child):

        {
          "<aid>": {
            "sector":       "electricity" | "gas" | "heat",
            "obs_key":      "p_mw" | "mass_flow",
            "budget":       <float | null>,   # |operator policy|, null = heat
            "lp_envelope":  <float | null>,   # |LP Var bound|, null = unbounded
            "node_id":      <child's parent node id>
          },
          ...
        }

    Budget is the value stamped on the model by ``apply_slack_budget``.
    LP-envelope is the absolute Var bound after ``apply_slack_budget``
    widened it (typically ``10 × budget``).  Both are ``null`` for
    slack children left intentionally unbudgeted (heat-side
    ``ExtHydrGrid``); the plot then skips the overlay for that sector.
    """

    meta: dict[str, dict[str, Any]] = {}
    for child in monee_net.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid):
            obs_key = "p_mw"
            budget = getattr(m, "_scare_slack_budget_mw", None)
            var = getattr(m, "p_mw", None)
        elif isinstance(m, ExtHydrGrid):
            obs_key = "mass_flow"
            budget = getattr(m, "_scare_slack_budget_kgs", None)
            var = getattr(m, "mass_flow", None)
        else:
            continue
        try:
            node = monee_net.node_by_id(child.node_id)
            sector = sector_from_grid(node.grid)
        except Exception:  # noqa: BLE001
            sector = None
        if sector is None:
            continue
        envelope: float | None = None
        if var is not None:
            v_min = getattr(var, "min", None)
            v_max = getattr(var, "max", None)
            mags = [abs(float(b)) for b in (v_min, v_max) if b is not None]
            if mags:
                envelope = max(mags)
        aid = f"child-{child.id}"
        meta[aid] = {
            "sector": sector.value if isinstance(sector, Sector) else str(sector),
            "obs_key": obs_key,
            "budget": float(budget) if budget is not None else None,
            "lp_envelope": envelope,
            "node_id": child.node_id,
        }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def write_diary_csv(path: Path) -> None:
    rows = diagnostics.negotiation_log()
    _write_dataclass_csv(path, rows)


def write_events_csv(path: Path) -> None:
    rows = diagnostics.event_log()
    _write_dataclass_csv(path, rows)


def write_trajectories_csv(path: Path) -> None:
    """Pivot the per-aid trajectory log into a wide CSV.

    Output layout:

        time_s, <aid_1>, <aid_2>, ..., <aid_n>
        t_0,    f_0_1,   f_0_2,   ...,  f_0_n
        ...

    Forward-fills missing values per aid (regulate is event-driven, so
    aids report at different timestamps).  The result is consumed by
    the C.5 cluster-synchronisation analysis in
    :mod:`experiment.eval.adaptive_network_analysis`.
    """
    rows = diagnostics.trajectory_log()
    if not rows:
        return
    aids: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r.aid not in seen:
            seen.add(r.aid)
            aids.append(r.aid)
    times = sorted({round(r.t, 6) for r in rows})

    # last-observed factor per aid, replayed at each unique timestamp.
    pending: dict[str, float] = {}
    by_t: dict[float, dict[str, float]] = {}
    for r in rows:
        t = round(r.t, 6)
        by_t.setdefault(t, {})[r.aid] = float(r.factor)

    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", *aids])
        for t in times:
            updates = by_t.get(t, {})
            pending.update(updates)
            row = [f"{t:.6f}"]
            for aid in aids:
                v = pending.get(aid)
                row.append(f"{v:.6f}" if v is not None else "")
            w.writerow(row)


def write_messages_csv(path: Path, world: Any) -> None:
    """Best-effort dump of mango.recorded_messages.  Off by default in
    the evaluation campaigns (high volume); turned on for the
    communication-cost campaign only.
    """
    recs = list(getattr(world, "recorded_messages", []) or [])
    if not recs:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "type", "sender", "recipient"])
        for rec in recs:
            w.writerow([
                getattr(rec, "time", ""),
                type(getattr(rec, "content", rec)).__name__,
                getattr(rec, "sender", ""),
                getattr(rec, "receiver", ""),
            ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_dataclass_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        Path(path).write_text("")
        return
    sample = rows[0]
    if not is_dataclass(sample):
        Path(path).write_text("")
        return
    headers = [f.name for f in fields(sample)]
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            d = asdict(r)
            w.writerow([d.get(h, "") for h in headers])
