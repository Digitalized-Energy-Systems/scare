"""Runtime diagnostics for monee non-convergence.

Ring buffer of recent regulate/switch actions plus negotiation/event/trajectory
ledgers, dumped alongside a solver infeasibility warning. No-op until ``arm()``.

State lives in a single :class:`DiagnosticsRecorder` (the module singleton
``_RECORDER``); the module-level ``record_*`` / ``*_log`` functions are thin
delegators that resolve ``_RECORDER`` at CALL TIME. ``arm()`` clears the ledgers
in place and NEVER rebinds ``_RECORDER``, so already-captured ``from ... import
record_event`` bindings stay valid across a per-task re-arm.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Ring buffer of recent regulate/switch actions (<1 MB).
_MAX_ACTIONS = 10000


@dataclass(frozen=True)
class ActionRecord:
    t: float
    kind: str  # "regulate" | "switch"
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
    skipped_singleton / finished / stalled / timed_out / cancelled /
    abandoned). ``skipped_balanced`` and ``skipped_singleton`` emit a single
    record with no preceding ``started``; only negotiations that reach
    ``started`` also emit exactly one terminal record.
    """

    t: float
    aid: str
    sector: str
    nid: str
    event: str
    target: float
    residual: float
    group_size: int


def _format_record(r: ActionRecord) -> str:
    if r.kind == "regulate":
        return (
            f"  t={r.t:7.3f} regulate {r.aid:<14s} sec={r.sector:<11s} "
            f"factor={r.value:.4f} ({r.reason})"
        )
    if r.kind == "switch":
        return f"  t={r.t:7.3f} switch   {r.aid:<14s} ({r.reason})"
    return f"  t={r.t:7.3f} FAILURE  branch={r.aid} ({r.reason})"


class DiagnosticsRecorder:
    """Owns the arm flag, action ring buffer, and negotiation/event/trajectory
    ledgers, with all record/snapshot/summary methods. Single point of per-task
    reset: ``arm()`` clears in place. The module singleton is never rebound."""

    def __init__(self) -> None:
        self._armed: bool = False
        self._log: deque[ActionRecord] = deque(maxlen=_MAX_ACTIONS)
        self._negotiation_log: list[NegotiationRecord] = []
        self._event_log: list[EventRecord] = []
        self._trajectory_log: list[TrajectoryRecord] = []
        self._trajectory_armed: bool = False

    def arm(self) -> None:
        self._armed = True
        self._log.clear()
        self._negotiation_log.clear()
        self._event_log.clear()
        self._trajectory_log.clear()

    def set_trajectory_logging(self, enabled: bool) -> None:
        self._trajectory_armed = bool(enabled)
        if not self._trajectory_armed:
            self._trajectory_log.clear()

    def record_regulate(
        self, *, t: float, aid: str, sector: str, factor: float, reason: str
    ) -> None:
        if not self._armed:
            return
        self._log.append(
            ActionRecord(
                t=t,
                kind="regulate",
                aid=aid,
                sector=sector,
                value=factor,
                reason=reason,
            )
        )
        if self._trajectory_armed:
            self._trajectory_log.append(
                TrajectoryRecord(t=t, aid=aid, sector=sector, factor=factor)
            )

    def record_switch(self, *, t: float, aid: str, reason: str) -> None:
        if not self._armed:
            return
        self._log.append(
            ActionRecord(
                t=t,
                kind="switch",
                aid=aid,
                sector="",
                value=float("nan"),
                reason=reason,
            )
        )
        # Mirror into the event ledger so the aggregator counts tie-switch closes.
        self._event_log.append(
            EventRecord(t=t, kind=f"switch:{reason}", aid=aid, sector="", detail="")
        )

    def record_negotiation(
        self,
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
        if not self._armed:
            return
        self._negotiation_log.append(
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

    def negotiation_log(self) -> list[NegotiationRecord]:
        return list(self._negotiation_log)

    def negotiation_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for r in self._negotiation_log:
            summary[r.event] = summary.get(r.event, 0) + 1
        return summary

    def record_event(
        self, *, t: float, kind: str, aid: str = "", sector: str = "", detail: str = ""
    ) -> None:
        if not self._armed:
            return
        self._event_log.append(
            EventRecord(t=t, kind=kind, aid=aid, sector=sector, detail=detail)
        )

    def event_log(self) -> list[EventRecord]:
        return list(self._event_log)

    def event_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for r in self._event_log:
            summary[r.kind] = summary.get(r.kind, 0) + 1
        return summary

    def trajectory_log(self) -> list[TrajectoryRecord]:
        return list(self._trajectory_log)

    def action_log(self) -> list[ActionRecord]:
        return list(self._log)

    def dump_recent(self, limit: int = _MAX_ACTIONS) -> str:
        if not self._log:
            return "  (no actions recorded)"
        items = list(self._log)[-limit:]
        return "\n".join(_format_record(r) for r in items)


_RECORDER = DiagnosticsRecorder()


# --- Module-level delegators (resolve _RECORDER at call time) ---------------


def arm() -> None:
    """Enable recording and reset every per-run log; call per task so counts are
    not polluted by earlier tasks sharing the worker process."""
    _RECORDER.arm()


def set_trajectory_logging(enabled: bool) -> None:
    """Toggle per-aid regulation-factor trajectory logging (opt-in: it can grow
    to thousands of rows per device on long runs)."""
    _RECORDER.set_trajectory_logging(enabled)


def record_regulate(
    *, t: float, aid: str, sector: str, factor: float, reason: str
) -> None:
    _RECORDER.record_regulate(t=t, aid=aid, sector=sector, factor=factor, reason=reason)


def record_switch(*, t: float, aid: str, reason: str) -> None:
    _RECORDER.record_switch(t=t, aid=aid, reason=reason)


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
    _RECORDER.record_negotiation(
        t=t,
        aid=aid,
        sector=sector,
        nid=nid,
        event=event,
        target=target,
        residual=residual,
        group_size=group_size,
    )


def negotiation_log() -> list[NegotiationRecord]:
    """Snapshot of the lifecycle ledger.  Used by the post-run aggregator."""
    return _RECORDER.negotiation_log()


def negotiation_summary() -> dict[str, int]:
    """Aggregate counts per lifecycle event.

    Healthy: ``started == finished + timed_out + cancelled + abandoned +
    stalled + skipped_singleton`` (skipped_balanced does not count as started).
    """
    return _RECORDER.negotiation_summary()


def record_event(
    *, t: float, kind: str, aid: str = "", sector: str = "", detail: str = ""
) -> None:
    """Append a domain-level event record. No-op until ``arm()`` is called."""
    _RECORDER.record_event(t=t, kind=kind, aid=aid, sector=sector, detail=detail)


def event_log() -> list[EventRecord]:
    return _RECORDER.event_log()


def event_summary() -> dict[str, int]:
    return _RECORDER.event_summary()


def trajectory_log() -> list[TrajectoryRecord]:
    """Snapshot of the per-aid regulation factor trajectory ledger."""
    return _RECORDER.trajectory_log()


def action_log() -> list[ActionRecord]:
    """Snapshot of the recent-action ring buffer."""
    return _RECORDER.action_log()


def dump_recent(limit: int = _MAX_ACTIONS) -> str:
    return _RECORDER.dump_recent(limit)


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
    """Wire the solver warning hook (install-only, idempotent). Arming is the
    caller's explicit responsibility (runner and scenarios/__main__ both arm)."""
    target = logging.getLogger("monee.solver.pyo")
    for f in target.filters:
        if isinstance(f, _SolverWarningHandler):
            return
    target.addFilter(_SolverWarningHandler())
