"""Defensive per-solve time-limit guard for energyflow MISOCP solves.

The energyflow path invokes ``monee.solver.pyo.PyomoSolver.solve``, which
seeds the solver from ``PER_SOLVER_OPTIONS``.  Some monee builds ship no
time limit for SCIP, so a hard MISOCP solve can run unbounded.  That
matters because the runner's ``asyncio.wait_for`` cancellation is
cooperative: a synchronous solve holds the interpreter and cannot be
preempted until the next ``await``, letting a task overrun its timeout.

:func:`install_solver_time_limit` patches ``PER_SOLVER_OPTIONS`` so every
subsequent solve seeds the right per-vendor time-limit option (SCIP
``limits/time``, Gurobi ``TimeLimit``, HiGHS ``time_limit``, CBC
``seconds``, ...).  Default is 60 s per solve — well above the typical
sub-second solve, so it only trips on genuine divergence.  Override via
``limit_s`` or ``SCARE_SOLVER_TIMELIMIT_S``.

The patch is idempotent and additive: it never lowers a limit a
downstream library already chose.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Default per-solve wall-clock cap.  60 s far exceeds the typical
# sub-second MISOCP solve and bounds the worst case for asyncio.wait_for.
_DEFAULT_LIMIT_S: float = 60.0


# Per-solver option name for the wall-clock budget, in each solver's native
# naming as forwarded by Pyomo's ``solver.options[k] = v``.
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
        # Never raise a limit downstream already chose, but treat an
        # absent / disabled one (<= 0) as overridable by our floor.
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
