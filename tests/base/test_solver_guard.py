"""Unit tests for :mod:`scare.base.solver_guard`.

Validates the per-solve wall-clock cap is installed defensively
(catches the eval_full_small task-64 / task-67 / task-98 case where
a synchronous SCIP MISOCP solve exceeded ``task_timeout_s`` and
blocked ``asyncio.wait_for`` cancellation until the SLURM SIGTERM).
"""

from __future__ import annotations

import importlib

import pytest

# All tests rewrite PER_SOLVER_OPTIONS; restore the original snapshot
# after each test so other suites don't see leaked state.
@pytest.fixture(autouse=True)
def _restore_per_solver_options():
    from monee.solver import pyo as _pyo
    original = {k: dict(v) for k, v in _pyo.PER_SOLVER_OPTIONS.items()}
    # Reset the module-level "_INSTALLED" so each test sees a clean run.
    from scare.base import solver_guard
    solver_guard._INSTALLED["done"] = False
    solver_guard._INSTALLED["limit_s"] = None
    yield
    _pyo.PER_SOLVER_OPTIONS.clear()
    _pyo.PER_SOLVER_OPTIONS.update(original)
    solver_guard._INSTALLED["done"] = False
    solver_guard._INSTALLED["limit_s"] = None


def test_install_seeds_scip_time_limit():
    """SCIP carries no default TimeLimit in older monee builds.  The
    guard must install ``limits/time`` so a SCIP MISOCP cannot run
    unbounded and block asyncio cancellation.
    """
    from monee.solver import pyo as _pyo
    from scare.base.solver_guard import install_solver_time_limit, installed_limit_s

    # Strip any pre-existing scip entry so we can verify the seed lands.
    _pyo.PER_SOLVER_OPTIONS.pop("scip", None)

    install_solver_time_limit(45.0)

    assert _pyo.PER_SOLVER_OPTIONS["scip"]["limits/time"] == 45.0
    assert installed_limit_s() == 45.0


def test_install_seeds_gurobi_time_limit():
    """Gurobi uses ``TimeLimit`` (camelCase) — different option key."""
    from monee.solver import pyo as _pyo
    from scare.base.solver_guard import install_solver_time_limit

    _pyo.PER_SOLVER_OPTIONS.pop("gurobi", None)

    install_solver_time_limit(45.0)

    assert _pyo.PER_SOLVER_OPTIONS["gurobi"]["TimeLimit"] == 45.0


def test_install_does_not_raise_existing_limit():
    """When monee already pins ``TimeLimit=300`` for Gurobi (newer
    builds), calling the guard with a smaller cap must respect the
    smaller of the two — we tighten, never loosen, but also never
    raise an explicit user-chosen lower limit.
    """
    from monee.solver import pyo as _pyo
    from scare.base.solver_guard import install_solver_time_limit

    _pyo.PER_SOLVER_OPTIONS["gurobi"] = {"TimeLimit": 30.0}

    install_solver_time_limit(60.0)

    # 30 was already lower than our 60 cap — must NOT be raised to 60.
    assert _pyo.PER_SOLVER_OPTIONS["gurobi"]["TimeLimit"] == 30.0


def test_install_tightens_an_overly_generous_limit():
    """A pre-existing TimeLimit of 600s is *higher* than our guard cap;
    the guard tightens it so the cancellation window we promised the
    runner is actually honoured.
    """
    from monee.solver import pyo as _pyo
    from scare.base.solver_guard import install_solver_time_limit

    _pyo.PER_SOLVER_OPTIONS["gurobi"] = {"TimeLimit": 600.0}

    install_solver_time_limit(60.0)

    assert _pyo.PER_SOLVER_OPTIONS["gurobi"]["TimeLimit"] == 60.0


def test_install_idempotent():
    """Calling twice with the same limit is a no-op the second time."""
    from monee.solver import pyo as _pyo
    from scare.base.solver_guard import install_solver_time_limit

    _pyo.PER_SOLVER_OPTIONS.pop("scip", None)

    install_solver_time_limit(60.0)
    first = dict(_pyo.PER_SOLVER_OPTIONS["scip"])
    install_solver_time_limit(60.0)
    second = dict(_pyo.PER_SOLVER_OPTIONS["scip"])

    assert first == second


def test_env_var_override(monkeypatch):
    """``SCARE_SOLVER_TIMELIMIT_S`` overrides the default when no
    explicit limit is passed.
    """
    from monee.solver import pyo as _pyo
    from scare.base.solver_guard import install_solver_time_limit, installed_limit_s

    monkeypatch.setenv("SCARE_SOLVER_TIMELIMIT_S", "15")
    _pyo.PER_SOLVER_OPTIONS.pop("scip", None)

    install_solver_time_limit()

    assert installed_limit_s() == 15.0
    assert _pyo.PER_SOLVER_OPTIONS["scip"]["limits/time"] == 15.0


def test_install_survives_missing_monee():
    """If ``monee.solver.pyo`` is not importable (lightweight test
    environment), the guard must no-op rather than crash.
    """
    import sys
    from scare.base import solver_guard

    # Hide monee.solver.pyo for the duration of this test.
    real = sys.modules.get("monee.solver.pyo")
    sys.modules["monee.solver.pyo"] = None  # type: ignore[assignment]
    try:
        # Should not raise.
        solver_guard.install_solver_time_limit(30.0)
    finally:
        if real is not None:
            sys.modules["monee.solver.pyo"] = real
        else:
            sys.modules.pop("monee.solver.pyo", None)
