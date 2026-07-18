"""Lightweight tracing helpers for diagnosing the eval timeout.

Two facilities:

* **Simulation-time log enrichment**
* **Optimization bracket logging**
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import time
from collections import Counter

# --- current simulation time (updated by the sim driver) -------------------
_SIM_T: dict[str, float] = {"t": 0.0}


def set_sim_time(t: float) -> None:
    """Record the current simulation time for log enrichment."""
    try:
        _SIM_T["t"] = float(t)
    except (TypeError, ValueError):
        pass


class SimTimeLogFilter(logging.Filter):
    """Stamp every log record with the current sim time as ``record.sim_t``.

    Installed on the log *handlers* so every record (including third-party
    ones) gets the attribute before formatting.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.sim_t = _SIM_T["t"]
        return True


_solve_logger = logging.getLogger("scare.solve")


@contextlib.contextmanager
def optimization(label: str, *, logger: logging.Logger | None = None, **ctx):
    """Bracket a monee energyflow or distributed/ADMM solve (sync call or awaited
    coroutine) with START/DONE/FAIL log lines. A solve that never returns leaves a
    ``SOLVE-START`` with no matching ``SOLVE-DONE``.
    """
    log = logger or _solve_logger
    detail = " ".join(f"{k}={v}" for k, v in ctx.items())
    log.info("SOLVE-START %s %s", label, detail)
    t0 = time.monotonic()
    try:
        yield
    except BaseException as exc:  # noqa: BLE001
        log.info(
            "SOLVE-FAIL  %s after %.3fs: %s: %s",
            label,
            time.monotonic() - t0,
            type(exc).__name__,
            exc,
        )
        raise
    else:
        log.info("SOLVE-DONE  %s in %.3fs", label, time.monotonic() - t0)


# --- discrete-event clock-freeze watchdog ----------------------------------
_stall_logger = logging.getLogger("scare.stall")


def _unsettled_scheduled_tasks(world):
    """Yield (aid, source, coro, asyncio_task) for every scheduled agent task
    that is neither sleeping nor done — i.e. the tasks that keep mango's
    ``tasks_complete_or_sleeping`` from returning and freeze the clock.
    """
    for aid_key, agent in list(getattr(world, "_agents", {}).items()):
        sched = getattr(agent, "scheduler", None) or getattr(agent, "_scheduler", None)
        if sched is None:
            continue
        entries = list(getattr(sched, "_scheduled_tasks", [])) + list(
            getattr(sched, "_scheduled_process_tasks", [])
        )
        for entry in entries:
            scheduled_task, atask, coro, src = entry[0], entry[1], entry[2], entry[3]
            try:
                if scheduled_task._is_sleeping.done() or scheduled_task._is_done.done():
                    continue
            except AttributeError:
                continue
            aid = getattr(getattr(agent, "context", None), "aid", aid_key)
            yield aid, src, coro, atask


def _inbox_backlog(world) -> tuple[int, int]:
    """(container_inbox, summed agent inbox) backlog — detects message churn."""
    container_q = 0
    cib = getattr(world, "inbox", None)
    if cib is not None and hasattr(cib, "qsize"):
        with contextlib.suppress(Exception):
            container_q = cib.qsize()
    agent_q = 0
    for agent in list(getattr(world, "_agents", {}).values()):
        ib = getattr(agent, "inbox", None)
        if ib is not None and hasattr(ib, "qsize"):
            with contextlib.suppress(Exception):
                agent_q += ib.qsize()
    return container_q, agent_q


async def sim_stall_watchdog(
    world,
    *,
    interval_s: float = 10.0,
    min_freeze_s: float = 60.0,
    report_every_s: float = 60.0,
    max_stuck_dumped: int = 12,
    logger: logging.Logger | None = None,
) -> None:
    """Real-time watchdog that logs *why* the sim clock is frozen."""
    log = logger or _stall_logger
    last_t: float | None = None
    static_s = 0.0
    last_report_s = 0.0
    while True:
        await asyncio.sleep(interval_s)
        try:
            now_t = float(world.clock.time)
        except Exception:  # noqa: BLE001
            return

        if last_t is None or now_t != last_t:
            last_t = now_t
            static_s = 0.0
            last_report_s = 0.0
            continue

        # Clock unchanged since the last tick.
        static_s += interval_s
        if static_s < min_freeze_s:
            continue  # likely just a slow step; not yet a freeze
        if static_s - last_report_s < report_every_s and last_report_s > 0:
            continue
        last_report_s = static_s

        unsettled = list(_unsettled_scheduled_tasks(world))
        container_q, agent_q = _inbox_backlog(world)
        log.warning(
            "watchdog: SIM CLOCK FROZEN at t=%.3f for ~%.0fs | unsettled_tasks=%d "
            "container_inbox=%d agent_inbox=%d",
            now_t,
            static_s,
            len(unsettled),
            container_q,
            agent_q,
        )
        if unsettled:
            by_src = Counter(src for _, src, _, _ in unsettled)
            log.warning("watchdog: unsettled by source: %s", dict(by_src))
            for aid, src, coro, atask in unsettled[:max_stuck_dumped]:
                buf = io.StringIO()
                with contextlib.suppress(Exception):
                    atask.print_stack(limit=12, file=buf)
                log.warning(
                    "watchdog: STUCK agent=%s src=%s coro=%r\n%s",
                    aid,
                    src,
                    coro,
                    buf.getvalue().rstrip(),
                )
        else:
            log.warning(
                "watchdog: no unsettled scheduled tasks — clock-advance/"
                "convergence-loop stall, not a never-settling task"
            )
