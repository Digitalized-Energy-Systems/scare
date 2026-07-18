"""Solver-failure log counter and file/stderr logging setup for the runner."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from scare.base.runtime.trace import SimTimeLogFilter

# Solver-failure emitters the per-task counter attaches to; detached in run_task's
# finally since run_local reuses worker processes across tasks.
_SOLVER_FAILURE_LOGGERS: tuple[str, ...] = (
    "monee.solver.pyo",
    "pyomo.core",
    "monee.solver.gurobipy",
    "monee.simulation.stepper",
)


class _SolverFailureCounter(logging.Filter):
    """Count solver-status escalations for per-task health.

    An infeasible solve fires as a monee-ERROR + pyomo-WARNING pair, deduped
    within ``_DEDUPE_WINDOW_S``; Gurobi/Pyomo env strings are also caught to
    separate env issues from algorithm bugs.
    """

    _SOLVER_ERROR_MARKERS: tuple[str, ...] = (
        "GurobiError",
        "HostID mismatch",
        "License",  # Gurobi LicenseError
    )
    _DEDUPE_WINDOW_S: float = 1.0  # min spacing between distinct solves

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.infeasible_count = 0
        self.warning_count = 0
        self._last_infeasible_t: float = float("-inf")

    def _is_infeasible_msg(self, msg: str) -> bool:
        # monee.solver.pyo ERROR path.
        if "infeasible (status=" in msg or "Pyomo solve infeasible" in msg:
            return True
        # pyomo.core load_solutions WARNING path (both substrings, one record).
        if (
            "Loading a SolverResults object" in msg
            and "termination condition: infeasible" in msg
        ):
            return True
        # monee.solver.gurobipy (islanding backend) + monee.simulation.stepper
        # skip-mode absorption — without these, stepper-path failures leave
        # solver_failures at 0.
        if "Gurobi solve failed without a usable solution" in msg:
            return True
        if "Stepper step" in msg and "failed" in msg:
            return True
        return False

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.levelno < logging.WARNING:
            return True
        msg = record.getMessage()
        if self._is_infeasible_msg(msg):
            if record.created - self._last_infeasible_t >= self._DEDUPE_WINDOW_S:
                self.infeasible_count += 1
                self.count += 1
            self._last_infeasible_t = record.created
        elif "returned non-ok status" in msg:
            self.warning_count += 1
            self.count += 1
        elif any(marker in msg for marker in self._SOLVER_ERROR_MARKERS):
            # Gurobi env/license/host-id errors: still solver failures.
            self.warning_count += 1
            self.count += 1
        return True


def _setup_logging(log_path: Path) -> tuple[logging.FileHandler, _SolverFailureCounter]:
    # Stamp every record with the current sim time (record.sim_t) so the
    # LOG_FORMAT's t=... field is always populated, including third-party logs.
    sim_time_filter = SimTimeLogFilter()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(sim_time_filter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Drop pre-existing handlers to avoid double-logging (e.g. earlier basicConfig).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    # Keep WARN+ on stderr so Slurm captures show-stoppers per array task.
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(logging.Formatter(LOG_FORMAT))
    stderr.addFilter(sim_time_filter)
    root.addHandler(stderr)

    # Suppress third-party DEBUG/INFO chatter (mango alone emits ~60k lines
    # per 30s sim). At package root so new submodules stay quiet; WARN+ surfaces.
    for noisy in (
        "pyomo",
        "gurobipy",
        "mango",
        "mango_energy_environments",
        "simbench",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    counter = _SolverFailureCounter()
    # Listen on every infeasibility emitter; the counter dedupes pairs (an
    # absorbed stepper failure fires as gurobipy/pyo ERROR + stepper WARNING
    # within the dedupe window).
    for logger_name in _SOLVER_FAILURE_LOGGERS:
        logging.getLogger(logger_name).addFilter(counter)
    return handler, counter


LOG_FORMAT = "%(asctime)s t=%(sim_t)8.3f %(levelname)s [%(name)s] %(message)s"
