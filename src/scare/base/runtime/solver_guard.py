"""Per-solve time-limit guard for energyflow MISOCP solves.

Some monee builds ship no SCIP time limit, so a hard solve can run
unbounded — and asyncio.wait_for can't preempt a synchronous solve until
the next ``await``, letting a task overrun. :func:`install_solver_time_limit`
patches ``PER_SOLVER_OPTIONS`` with the per-vendor time-limit option
(default 60 s; override via ``limit_s`` / ``SCARE_SOLVER_TIMELIMIT_S``).
Idempotent and additive — never lowers a limit downstream already chose.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Default per-solve wall-clock cap; bounds the asyncio.wait_for worst case.
_DEFAULT_LIMIT_S: float = 60.0


# Per-solver wall-clock budget option, in each solver's native naming.
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
                env,
                _DEFAULT_LIMIT_S,
            )
    return _DEFAULT_LIMIT_S


def install_solver_time_limit(limit_s: float | None = None) -> None:
    """Seed a per-solve wall-clock limit for every Pyomo solver.

    Idempotent; never reduces a limit downstream already chose. Returns
    silently when ``monee.solver.pyo`` is not importable.
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
        # Never raise an existing limit, but treat <= 0 (disabled) as overridable.
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
    """Return the active per-solve limit, or ``None`` if not yet installed."""
    return _INSTALLED["limit_s"] if _INSTALLED["done"] else None
