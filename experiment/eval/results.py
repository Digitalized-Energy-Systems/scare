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

from scare.base import diagnostics

from experiment.eval.metrics import (
    constraint_violation_integral,
    served_breakdown,
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
) -> dict[str, Any]:
    """Build the structured result.json payload for a scare-variant run."""
    served = served_breakdown(monee_net, behavior, priorities=priorities)
    integral = constraint_violation_integral(world)
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
            "time_to_stabilise_s": t_stable,
            "regulates_total": sum(regulates_by_reason.values()),
            "regulates_by_reason": dict(regulates_by_reason),
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


def write_diary_csv(path: Path) -> None:
    rows = diagnostics.negotiation_log()
    _write_dataclass_csv(path, rows)


def write_events_csv(path: Path) -> None:
    rows = diagnostics.event_log()
    _write_dataclass_csv(path, rows)


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
