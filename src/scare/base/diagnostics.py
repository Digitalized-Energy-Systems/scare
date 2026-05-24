"""Lightweight runtime diagnostics for monee non-convergence.

Provides a global ring buffer that captures the recent stream of
``regulate`` / ``switch`` actions that scare roles applied to the
network, so that when monee's solver logs an infeasibility warning we
can dump the action context that led there.

Off by default: the recording is a no-op until ``arm()`` is called
(typically from an experiment script), so unit tests stay quiet.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

logger = logging.getLogger(__name__)

# Keep enough history to capture every regulate action across a typical
# smoke run: 5 s sim × ~200 agents × a few regulate calls each easily
# pushes past 64 (the original cap, which silently truncated child-1's
# regulate trace during the priority-inversion investigation).  10k is
# still well below 1 MB of memory and lets diagnostics.txt remain a
# faithful audit log of every actuation.
_MAX_ACTIONS = 10000
_armed: bool = False
_log: Deque["ActionRecord"] = deque(maxlen=_MAX_ACTIONS)

# Negotiation lifecycle ledger.  Unbounded list because the post-run
# aggregator wants the full picture, not just the last few entries.
# Cleared by ``arm()`` so each experiment run starts fresh.
_negotiation_log: list["NegotiationRecord"] = []

# Event ledger — failures, reconfigurations, local-gen fallback requests,
# constraint violations.  One row per occurrence so the aggregator can
# count by type and reconstruct timing.  Same lifecycle as the
# negotiation log: unbounded, cleared on ``arm()``.
_event_log: list["EventRecord"] = []

# Per-aid trajectory log — every applied regulation factor with timestamp.
# Powers the C.5 cluster-synchronisation analysis.  Off by default; arm
# turns it on, ``set_trajectory_logging`` toggles independently because
# this log can grow large on long runs and is therefore opt-in.
_trajectory_log: list["TrajectoryRecord"] = []
_trajectory_armed: bool = False


@dataclass(frozen=True)
class ActionRecord:
    t: float
    kind: str       # "regulate" | "switch" | "failure"
    aid: str
    sector: str     # "" if N/A
    value: float    # factor for regulate, NaN for switch
    reason: str     # short tag


@dataclass(frozen=True)
class EventRecord:
    """One global / domain event.  ``kind`` is a short tag; ``detail``
    is a free-form string for the aggregator to display verbatim or
    parse if interested (branch_id, sector, residual, switch aid …).
    """
    t: float
    kind: str
    aid: str
    sector: str
    detail: str


@dataclass(frozen=True)
class TrajectoryRecord:
    """One sample of a device's regulation factor.

    Generated whenever a role calls ``record_regulate`` and trajectory
    logging is armed.  Aggregated downstream into per-task
    ``trajectories.csv`` (one column per aid, one row per timestamp,
    factor in [0, 1] with the last seen value forward-filled).
    """

    t: float
    aid: str
    sector: str
    factor: float


@dataclass(frozen=True)
class NegotiationRecord:
    """One step in the lifecycle of a single gossip negotiation.

    ``event`` enumerates the lifecycle position:

    - ``"started"``           → ``_start_gossip`` past the threshold check
    - ``"skipped_balanced"``  → target below the per-group threshold
    - ``"skipped_singleton"`` → no group neighbours; emitted LocalGenerationRequest
    - ``"finished"``          → ``_finish_negotiation`` ran with a residual
    - ``"timed_out"``         → wallclock deadline forced ``_finish_negotiation``
    - ``"cancelled"``         → ``ConstraintViolation`` aborted active gossip
    - ``"abandoned"``         → still active when the world tore down

    A single negotiation contributes one ``"started"`` record and exactly
    one terminal record (any of finished/timed_out/cancelled/abandoned),
    so a healthy run satisfies ``started == sum(terminals)``.
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
    """Enable recording and reset every per-run log to empty.

    Called once at process start by ``install_solver_failure_dump`` and
    again at the top of each task by the campaign runner so per-task
    ``result.json`` event counts are not polluted by earlier tasks
    sharing the same worker process.
    """
    global _armed
    _armed = True
    _log.clear()
    _negotiation_log.clear()
    _event_log.clear()
    _trajectory_log.clear()


def set_trajectory_logging(enabled: bool) -> None:
    """Toggle per-aid regulation-factor trajectory logging.

    Default is OFF: the log can grow to thousands of rows per device
    on long runs, and only the C.5 cluster-synchronisation analysis
    needs it.  The eval runner enables it via the corresponding plan
    flag.
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
        ActionRecord(t=t, kind="regulate", aid=aid, sector=sector,
                     value=factor, reason=reason)
    )
    if _trajectory_armed:
        _trajectory_log.append(
            TrajectoryRecord(t=t, aid=aid, sector=sector, factor=factor)
        )


def record_switch(*, t: float, aid: str, reason: str) -> None:
    if not _armed:
        return
    _log.append(
        ActionRecord(t=t, kind="switch", aid=aid, sector="",
                     value=float("nan"), reason=reason)
    )
    # Also surface in the event ledger so the post-run aggregator can
    # count tie-switch closes alongside other domain events.
    _event_log.append(
        EventRecord(t=t, kind=f"switch:{reason}", aid=aid, sector="", detail="")
    )


def record_failure(*, t: float, branch_id: Any) -> None:
    if not _armed:
        return
    _log.append(
        ActionRecord(t=t, kind="failure", aid=str(branch_id), sector="",
                     value=float("nan"), reason="branch_failure")
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
    """Append a gossip-lifecycle record.

    Cheap (single list append).  No-op until ``arm()`` is called, so unit
    tests that don't enable diagnostics aren't affected.
    """
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
    """Aggregate counts per lifecycle event.  Useful for at-a-glance
    accounting at the end of a run.  An invariant a healthy run should
    satisfy: ``started == finished + timed_out + cancelled + abandoned
    + stalled + skipped_singleton`` (plus ``skipped_balanced`` which
    doesn't count as "started").  ``stalled`` is the P2 early-termination
    terminal: gossip's gap-window range fell below tolerance while the
    gap was still above the per-group threshold.
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
    """Append a domain-level event record (branch_failure, line_failure,
    reconfiguration_completed, tie_switch_close, local_gen_request,
    local_gen_covered, constraint_violation, constraint_warning,
    holon_formed, holon_admm_result, holon_admm_failed).

    Cheap (single list append).  No-op until ``arm()`` is called.
    """
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
    """Attached to ``monee.solver.pyo`` — on every warning, append the
    action history to the log record so the dump appears alongside the
    infeasibility report.  Returns ``True`` so the original record still
    propagates."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING and "Pyomo solve failed" in record.getMessage():
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
