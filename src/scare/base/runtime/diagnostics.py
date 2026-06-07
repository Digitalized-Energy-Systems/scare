"""Runtime diagnostics for monee non-convergence.

Global ring buffer of recent regulate/switch actions, dumped alongside a
solver infeasibility warning. No-op until ``arm()`` is called.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Ring buffer of recent regulate/switch/failure actions (<1 MB).
_MAX_ACTIONS = 10000
_armed: bool = False
_log: deque[ActionRecord] = deque(maxlen=_MAX_ACTIONS)

# Negotiation lifecycle ledger; unbounded, cleared on arm().
_negotiation_log: list[NegotiationRecord] = []

# Domain event ledger; unbounded, cleared on arm().
_event_log: list[EventRecord] = []

# Per-aid regulation-factor trajectory; opt-in, can grow large.
_trajectory_log: list[TrajectoryRecord] = []
_trajectory_armed: bool = False


@dataclass(frozen=True)
class ActionRecord:
    t: float
    kind: str  # "regulate" | "switch" | "failure"
    aid: str
    sector: str  # "" if N/A
    value: float  # factor for regulate, NaN for switch
    reason: str  # short tag


@dataclass(frozen=True)
class EventRecord:
    """One domain event. ``kind`` is a short tag; ``detail`` is free-form."""

    t: float
    kind: str
    aid: str
    sector: str
    detail: str


@dataclass(frozen=True)
class TrajectoryRecord:
    """One sample of a device's regulation factor (aggregated to trajectories.csv)."""

    t: float
    aid: str
    sector: str
    factor: float


@dataclass(frozen=True)
class NegotiationRecord:
    """One step in a gossip negotiation's lifecycle.

    ``event`` is the lifecycle position (started / skipped_balanced /
    skipped_singleton / finished / timed_out / cancelled / abandoned). Each
    negotiation yields one ``started`` plus exactly one terminal record.
    """

    t: float
    aid: str
    sector: str
    nid: str
    event: str
    target: float
    residual: float
    group_size: int


def arm() -> None:
    """Enable recording and reset every per-run log; call per task so counts
    are not polluted by earlier tasks sharing the worker process.
    """
    global _armed
    _armed = True
    _log.clear()
    _negotiation_log.clear()
    _event_log.clear()
    _trajectory_log.clear()


def set_trajectory_logging(enabled: bool) -> None:
    """Toggle per-aid regulation-factor trajectory logging (opt-in: it can
    grow to thousands of rows per device on long runs).
    """
    global _trajectory_armed
    _trajectory_armed = bool(enabled)
    if not _trajectory_armed:
        _trajectory_log.clear()


def record_regulate(
    *, t: float, aid: str, sector: str, factor: float, reason: str
) -> None:
    if not _armed:
        return
    _log.append(
        ActionRecord(
            t=t, kind="regulate", aid=aid, sector=sector, value=factor, reason=reason
        )
    )
    if _trajectory_armed:
        _trajectory_log.append(
            TrajectoryRecord(t=t, aid=aid, sector=sector, factor=factor)
        )


def record_switch(*, t: float, aid: str, reason: str) -> None:
    if not _armed:
        return
    _log.append(
        ActionRecord(
            t=t, kind="switch", aid=aid, sector="", value=float("nan"), reason=reason
        )
    )
    # Mirror into the event ledger so the aggregator counts tie-switch closes.
    _event_log.append(
        EventRecord(t=t, kind=f"switch:{reason}", aid=aid, sector="", detail="")
    )


def record_failure(*, t: float, branch_id: Any) -> None:
    if not _armed:
        return
    _log.append(
        ActionRecord(
            t=t,
            kind="failure",
            aid=str(branch_id),
            sector="",
            value=float("nan"),
            reason="branch_failure",
        )
    )


def record_negotiation(
    *,
    t: float,
    aid: str,
    sector: str,
    nid: str,
    event: str,
    target: float = float("nan"),
    residual: float = float("nan"),
    group_size: int = 0,
) -> None:
    """Append a gossip-lifecycle record.  No-op until ``arm()`` is called."""
    if not _armed:
        return
    _negotiation_log.append(
        NegotiationRecord(
            t=t,
            aid=aid,
            sector=sector,
            nid=nid,
            event=event,
            target=target,
            residual=residual,
            group_size=group_size,
        )
    )


def negotiation_log() -> list[NegotiationRecord]:
    """Snapshot of the lifecycle ledger.  Used by the post-run aggregator."""
    return list(_negotiation_log)


def negotiation_summary() -> dict[str, int]:
    """Aggregate counts per lifecycle event.

    Healthy: ``started == finished + timed_out + cancelled + abandoned +
    stalled + skipped_singleton`` (skipped_balanced does not count as
    started). ``stalled``: gap-window range fell below tolerance while the
    gap was still above threshold.
    """
    summary: dict[str, int] = {}
    for r in _negotiation_log:
        summary[r.event] = summary.get(r.event, 0) + 1
    return summary


def record_event(
    *,
    t: float,
    kind: str,
    aid: str = "",
    sector: str = "",
    detail: str = "",
) -> None:
    """Append a domain-level event record. No-op until ``arm()`` is called."""
    if not _armed:
        return
    _event_log.append(
        EventRecord(t=t, kind=kind, aid=aid, sector=sector, detail=detail)
    )


def event_log() -> list[EventRecord]:
    return list(_event_log)


def event_summary() -> dict[str, int]:
    summary: dict[str, int] = {}
    for r in _event_log:
        summary[r.kind] = summary.get(r.kind, 0) + 1
    return summary


def trajectory_log() -> list[TrajectoryRecord]:
    """Snapshot of the per-aid regulation factor trajectory ledger."""
    return list(_trajectory_log)


def _format_record(r: ActionRecord) -> str:
    if r.kind == "regulate":
        return (
            f"  t={r.t:7.3f} regulate {r.aid:<14s} sec={r.sector:<11s} "
            f"factor={r.value:.4f} ({r.reason})"
        )
    if r.kind == "switch":
        return f"  t={r.t:7.3f} switch   {r.aid:<14s} ({r.reason})"
    return f"  t={r.t:7.3f} FAILURE  branch={r.aid} ({r.reason})"


def dump_recent(limit: int = _MAX_ACTIONS) -> str:
    if not _log:
        return "  (no actions recorded)"
    items = list(_log)[-limit:]
    return "\n".join(_format_record(r) for r in items)


class _SolverWarningHandler(logging.Filter):
    """On a solve-failure warning, log recent action history. Returns True so
    the original record still propagates."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.levelno >= logging.WARNING
            and "Pyomo solve failed" in record.getMessage()
        ):
            logger.warning(
                "Recent agent actions before solver failure:\n%s",
                dump_recent(limit=20),
            )
        return True


def install_solver_failure_dump() -> None:
    """Wire the solver warning hook.  Idempotent."""
    arm()
    target = logging.getLogger("monee.solver.pyo")
    for f in target.filters:
        if isinstance(f, _SolverWarningHandler):
            return
    target.addFilter(_SolverWarningHandler())
