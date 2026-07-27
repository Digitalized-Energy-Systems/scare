"""Solver-failure log counter and file/stderr logging setup for the runner."""

from __future__ import annotations

import logging
import re
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

    One infeasible physics step fires several records (a monee backend ERROR
    carrying the IIS, then the stepper's skip-mode WARNING), so they must be
    deduplicated to one event. The stepper names the step —
    ``Stepper step 19 failed`` — and that identity is what dedupes: records
    carrying a step number collapse per step, and the backend ERROR that
    precedes one is folded into it.

    A wall-clock window cannot do this job. At ``_DEDUPE_WINDOW_S = 1.0`` it
    also swallowed *genuinely distinct* steps whenever solves ran faster than a
    second, which on eval_full_v2 reported 2277 events against 3202 real failed
    steps — a 29 % undercount concentrated exactly on the fast-solving grids.

    Gurobi/Pyomo env strings are counted separately to keep environment issues
    apart from algorithm bugs.
    """

    _SOLVER_ERROR_MARKERS: tuple[str, ...] = (
        "GurobiError",
        "HostID mismatch",
        "License",  # Gurobi LicenseError
    )
    _STEP_RE = re.compile(r"Stepper step (\d+) failed")
    _BACKEND_LOGGERS: tuple[str, ...] = ("monee.solver.pyo", "monee.solver.gurobipy")

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.infeasible_count = 0
        self.warning_count = 0
        #: Step indices whose physics solve failed, in first-seen order.
        self.failed_steps: list[int] = []
        self._seen_steps: set[int] = set()
        # A backend failure is counted only once it is known not to be the
        # first half of a pair: either the stepper names its step, or the next
        # backend failure arrives. :meth:`finalize` flushes the last one.
        self._pending = False

    @property
    def first_failed_step(self) -> int | None:
        return self.failed_steps[0] if self.failed_steps else None

    @staticmethod
    def _is_backend_infeasible(msg: str) -> bool:
        return (
            "infeasible (status=" in msg
            or "Pyomo solve infeasible" in msg
            or "Gurobi solve failed without a usable solution" in msg
        )

    @staticmethod
    def _is_pyomo_echo(msg: str) -> bool:
        """Pyomo's ``load_solutions`` warning — always the echo of a backend
        ERROR for the same solve, never an event of its own."""
        return (
            "Loading a SolverResults object" in msg
            and "termination condition: infeasible" in msg
        )

    def _count_infeasible(self) -> None:
        self.infeasible_count += 1
        self.count += 1

    def _flush(self) -> None:
        if self._pending:
            self._pending = False
            self._count_infeasible()

    def finalize(self) -> None:
        """Count a trailing backend failure the stepper never attributed.

        The oracle path solves without a Stepper, so its infeasibility has no
        step record to close it; call once when the task ends.
        """
        self._flush()

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.levelno < logging.WARNING:
            return True
        msg = record.getMessage()

        step = self._STEP_RE.search(msg)
        if step is not None:
            # Authoritative: one failed physics step, identified by index.
            index = int(step.group(1))
            self._pending = False
            if index not in self._seen_steps:
                self._seen_steps.add(index)
                self.failed_steps.append(index)
                self._count_infeasible()
        elif record.name in self._BACKEND_LOGGERS and self._is_backend_infeasible(msg):
            self._flush()
            self._pending = True
        elif self._is_pyomo_echo(msg):
            # Counts only when the backend logger stayed silent.
            self._pending = True
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
