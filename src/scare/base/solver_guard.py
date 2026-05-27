"""Defensive per-solve time-limit guard for energyflow MISOCP solves.

Why this exists
---------------

``mango_energy_environments.base.monee.energyflow`` ultimately invokes
``monee.solver.pyo.PyomoSolver.solve``, which constructs the Pyomo
solver and seeds it from ``PER_SOLVER_OPTIONS``.  Newer monee builds
include ``TimeLimit: 300`` for Gurobi; older builds (the one currently
pinned in the HPC ``cmres_env`` that drove eval_full_small_20260526-
165742) include nothing for ``scip``, so a single SCIP MISOCP solve on
a hard problem (``voltage_stress`` / ``cp_heavy_dependent``) can run
for an unbounded number of wall-clock seconds.

That matters because ``asyncio.wait_for(..., timeout=task_timeout_s)``
in :func:`experiment.hpc.runner._run_simulation` is cooperative: a
synchronous solve holds the Python interpreter and the cancellation
fired by the timer cannot land until the next ``await``.  Eight of the
138 tasks in the campaign exited as ``killed`` at exactly the SLURM
wall (2700 s ≈ ``00:45:00``) rather than at the configured
``task_timeout_s=1500`` because the cancellation could not preempt a
SCIP solve that had already exceeded the runner timeout.

What this module does
---------------------

:func:`install_solver_time_limit` patches the relevant entry in
``monee.solver.pyo.PER_SOLVER_OPTIONS`` so every subsequent
``PyomoSolver.solve`` call seeds the solver with the right per-vendor
time-limit option:

* SCIP — ``limits/time`` (seconds)
* Gurobi — ``TimeLimit`` (seconds)
* HiGHS — ``time_limit`` (seconds)
* CBC — ``seconds`` (seconds)

The default is conservative (60 s per solve): even on the LV-50 grid
the typical Gurobi MISOCP solve is well under a second, so 60 s only
trips when the solver is genuinely diverging.  Callers can override
via the ``limit_s`` argument or the ``SCARE_SOLVER_TIMELIMIT_S`` env
var.

The patch is idempotent and additive: it never lowers an existing
limit a downstream library has chosen, so a newer monee that already
ships ``TimeLimit: 300`` keeps its value.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Default per-solve wall-clock cap.  60 s is plenty for LV-50 with
# Gurobi (~0.5 s typical, ~5 s for hard MISOCPs) and bounds the
# worst-case SCIP solve to a value asyncio.wait_for can preempt
# around.
_DEFAULT_LIMIT_S: float = 60.0


# Per-solver option name carrying the seconds-of-wall-clock budget.
# These follow each solver's native option naming, which is what
# Pyomo's ``solver.options[k] = v`` forwards to the underlying CLI.
_TIME_LIMIT_OPTION: dict[str, str] = {
    "scip": "limits/time",
    "gurobi": "TimeLimit",
    "gurobi_direct": "TimeLimit",
    "gurobi_persistent": "TimeLimit",
    "appsi_gurobi": "TimeLimit",
    "highs": "time_limit",
    "appsi_highs": "time_limit",
    "cbc": "seconds",
    "glpk": "tmlim",
    "ipopt": "max_cpu_time",
}


_INSTALLED: dict[str, Any] = {"done": False, "limit_s": None}


def _resolve_limit_s(limit_s: float | None) -> float:
    if limit_s is not None:
        return float(limit_s)
    env = os.environ.get("SCARE_SOLVER_TIMELIMIT_S")
    if env:
        try:
            return float(env)
        except ValueError:
            logger.warning(
                "SCARE_SOLVER_TIMELIMIT_S=%r is not a float — using default %.0fs",
                env, _DEFAULT_LIMIT_S,
            )
    return _DEFAULT_LIMIT_S


def install_solver_time_limit(limit_s: float | None = None) -> None:
    """Seed a per-solve wall-clock limit for every Pyomo solver used
    by ``monee``'s energyflow path.

    Safe to call multiple times; the patch is idempotent and never
    *reduces* a limit a downstream library has already chosen.
    Returns silently when ``monee.solver.pyo`` is not importable
    (test environments / shims that don't ship monee).
    """
    seconds = _resolve_limit_s(limit_s)
    try:
        from monee.solver import pyo as _pyo
    except ImportError:
        logger.debug("monee.solver.pyo not importable — solver guard skipped")
        return

    per_solver = getattr(_pyo, "PER_SOLVER_OPTIONS", None)
    if per_solver is None:
        logger.debug(
            "monee.solver.pyo.PER_SOLVER_OPTIONS missing — solver guard "
            "skipped (likely an older monee API; patch the host site instead)"
        )
        return

    applied: list[tuple[str, str, float]] = []
    for solver_name, option_key in _TIME_LIMIT_OPTION.items():
        existing = per_solver.setdefault(solver_name, {})
        current = existing.get(option_key)
        if current is None:
            existing[option_key] = seconds
            applied.append((solver_name, option_key, seconds))
            continue
        # Never raise a limit downstream already chose — but never let
        # an absent / disabled one (0, None) override our floor either.
        try:
            current_f = float(current)
        except (TypeError, ValueError):
            current_f = 0.0
        if current_f <= 0 or current_f > seconds:
            existing[option_key] = seconds
            applied.append((solver_name, option_key, seconds))

    _INSTALLED["done"] = True
    _INSTALLED["limit_s"] = seconds
    if applied:
        logger.info(
            "Solver per-solve time limits installed: %s",
            ", ".join(f"{s}.{k}={v:.0f}s" for s, k, v in applied),
        )
    else:
        logger.debug("Solver per-solve time limits already at-or-below %.0fs", seconds)


def installed_limit_s() -> float | None:
    """Return the active per-solve limit if :func:`install_solver_time_limit`
    has been called; ``None`` otherwise.  Used by tests to assert the
    guard is in place.
    """
    return _INSTALLED["limit_s"] if _INSTALLED["done"] else None
