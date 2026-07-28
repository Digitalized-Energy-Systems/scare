"""Per-solve time-limit guard for energyflow MISOCP solves.

Some monee builds ship no SCIP time limit, so a hard solve can run
unbounded — and asyncio.wait_for can't preempt a synchronous solve until
the next ``await``, letting a task overrun. :func:`install_solver_time_limit`
patches ``PER_SOLVER_OPTIONS`` with the per-vendor time-limit option
(default 60 s; override via ``limit_s`` / ``SCARE_SOLVER_TIMELIMIT_S``).
Idempotent; caps a larger or disabled (<=0) limit down to the target but
never raises one a downstream already set lower.

It must patch BOTH monee solver backends. ``PER_SOLVER_OPTIONS`` only reaches
the Pyomo round-trip, but ``solver="gurobi"`` resolves to monee's *native*
gurobipy backend (``dispatch._auto_backend``), which is constructed with no
params and copies ``gurobipy.DEFAULT_GUROBI_PARAMS`` — ``TimeLimit`` 300 —
at ``__init__``. Patching only Pyomo left every physics solve uncapped in
practice: on eval_full_v2_20260727 the runner logged "Scaled per-solve cap to
47s" while LV-S solves ran 300.7 s each, and 18 infeasible solves x 300 s hit
``task_timeout_s=5400`` exactly, killing two tasks. Callers that pass an
explicit ``TimeLimit`` (the oracle) still win: per-instance params are merged
OVER the defaults.
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


def _should_lower(current: Any, seconds: float) -> bool:
    """True when ``current`` is absent, disabled (<=0), or above ``seconds``."""
    if current is None:
        return True
    try:
        current_f = float(current)
    except (TypeError, ValueError):
        current_f = 0.0
    return current_f <= 0 or current_f > seconds


def _install_pyomo(seconds: float) -> list[tuple[str, str, float]]:
    try:
        from monee.solver import pyo as _pyo
    except ImportError:
        logger.debug("monee.solver.pyo not importable — Pyomo guard skipped")
        return []

    per_solver = getattr(_pyo, "PER_SOLVER_OPTIONS", None)
    if per_solver is None:
        logger.debug(
            "monee.solver.pyo.PER_SOLVER_OPTIONS missing — Pyomo guard "
            "skipped (likely an older monee API; patch the host site instead)"
        )
        return []

    applied: list[tuple[str, str, float]] = []
    for solver_name, option_key in _TIME_LIMIT_OPTION.items():
        existing = per_solver.setdefault(solver_name, {})
        if _should_lower(existing.get(option_key), seconds):
            existing[option_key] = seconds
            applied.append((solver_name, option_key, seconds))
    return applied


def _install_native_gurobipy(seconds: float) -> list[tuple[str, str, float]]:
    """Cap monee's native gurobipy backend, which ignores PER_SOLVER_OPTIONS.

    ``GurobipySolver.__init__`` copies this dict, so the patch must land before
    the solver is constructed — the runner installs the guard before building
    the world, and the oracle's explicit ``TimeLimit`` still overrides it.
    """
    try:
        from monee.solver import gurobipy as _gp
    except ImportError:
        logger.debug("monee.solver.gurobipy not importable — native guard skipped")
        return []

    defaults = getattr(_gp, "DEFAULT_GUROBI_PARAMS", None)
    if defaults is None:
        logger.debug(
            "monee.solver.gurobipy.DEFAULT_GUROBI_PARAMS missing — native "
            "guard skipped (likely an older monee API)"
        )
        return []

    if _should_lower(defaults.get("TimeLimit"), seconds):
        defaults["TimeLimit"] = seconds
        return [("gurobipy", "TimeLimit", seconds)]
    return []


def install_solver_time_limit(limit_s: float | None = None) -> None:
    """Seed a per-solve wall-clock limit for every monee solver backend.

    Idempotent; caps a larger or disabled limit down to ``seconds`` but
    never raises a lower one. Returns silently when monee is not importable.
    """
    seconds = _resolve_limit_s(limit_s)
    applied = _install_pyomo(seconds) + _install_native_gurobipy(seconds)

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
