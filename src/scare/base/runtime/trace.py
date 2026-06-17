"""Lightweight tracing helpers for diagnosing the eval timeout.

Two facilities:

* **Simulation-time log enrichment** — :class:`SimTimeLogFilter` stamps every
  log record with the current sim clock time (``record.sim_t``), kept current
  by the sim driver via :func:`set_sim_time`. The runner's log format prints it
  so every line carries ``t=<sim_seconds>``.

* **Optimization bracket logging** — :func:`optimization` wraps a (possibly
  hanging) solve with ``SOLVE-START`` / ``SOLVE-DONE`` / ``SOLVE-FAIL`` lines.
  If a monee energyflow or a distributed/ADMM solve is entered and never
  returns (the clock-freeze failure mode), the log ends on a ``SOLVE-START``
  with no matching ``SOLVE-DONE`` — naming exactly which solve hung.
"""

from __future__ import annotations

import contextlib
import logging
import time

# --- current simulation time (updated by the sim driver) -------------------
_SIM_T: dict[str, float] = {"t": 0.0}


def set_sim_time(t: float) -> None:
    """Record the current simulation time for log enrichment."""
    try:
        _SIM_T["t"] = float(t)
    except (TypeError, ValueError):
        pass


def get_sim_time() -> float:
    return _SIM_T["t"]


class SimTimeLogFilter(logging.Filter):
    """Stamp every log record with the current sim time as ``record.sim_t``.

    Installed on the log *handlers* so every record (including third-party
    ones) gets the attribute before formatting.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.sim_t = _SIM_T["t"]
        return True


# --- optimization bracket logging ------------------------------------------
_solve_logger = logging.getLogger("scare.solve")


@contextlib.contextmanager
def optimization(label: str, *, logger: logging.Logger | None = None, **ctx):
    """Bracket an optimization solve with START/DONE/FAIL log lines.

    Use around any monee energyflow or distributed/ADMM solve — sync call or
    ``await``ed coroutine. A solve that never returns leaves a ``SOLVE-START``
    with no ``SOLVE-DONE``::

        with optimization("energyflow", n_childs=len(net.childs)):
            result = run_energy_flow(net)
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
