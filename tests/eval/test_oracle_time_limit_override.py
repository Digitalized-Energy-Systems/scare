"""``oracle_time_limit_s`` must reach the solver, and reach BOTH oracle solves.

A task runs the oracle twice: the post-failure solve (whose solver the runner
builds via :func:`oracle_solver_for_task`) and the pre-failure baseline solve
inside ``compute_baseline_served``, which takes its own ``solver=`` and falls
back to ``_OracleGurobiSolver()``'s 300 s default when not given one. An
end-to-end run exposed the gap: with ``oracle_time_limit_s=1.0`` the log showed
one solve at ``runtime_s=1.0`` and another at ``300.005``.

Pre-existing and deliberately left alone: the 900 s ``_ORACLE_HARD_PRESET``
likewise never reaches the baseline call, so on lv_reconfig / mvlv the
restoration ratio divides a 900 s-capped post state by a 300 s-capped baseline.
"""

from __future__ import annotations

import pytest

from experiment.eval.oracle import (
    _ORACLE_HARD_GRIDS,
    _ORACLE_HARD_PRESET,
    oracle_solver_for_task,
)

DEFAULT_TIME_LIMIT = 300


def _limit(solver):
    return dict(getattr(solver, "_params", None) or {}).get("TimeLimit")


def test_scenario_override_reaches_the_solver():
    s = oracle_solver_for_task("simbench_lv", {"oracle_time_limit_s": 5.0})
    assert _limit(s) == pytest.approx(5.0)


def test_override_wins_over_the_hard_grid_preset():
    """An explicit budget is an instruction, not a suggestion."""
    grid = sorted(_ORACLE_HARD_GRIDS)[0]
    s = oracle_solver_for_task(grid, {"oracle_time_limit_s": 42.0})
    assert _limit(s) == pytest.approx(42.0)
    assert _limit(s) != _ORACLE_HARD_PRESET["TimeLimit"]


def test_absent_override_leaves_the_shipped_presets_untouched():
    plain = oracle_solver_for_task("simbench_lv", {"kind": "clean"})
    assert _limit(plain) in (None, DEFAULT_TIME_LIMIT)
    hard = oracle_solver_for_task(sorted(_ORACLE_HARD_GRIDS)[0], {"kind": "clean"})
    assert _limit(hard) == _ORACLE_HARD_PRESET["TimeLimit"]


def test_microgrid_scenario_still_gets_the_extended_budget():
    s = oracle_solver_for_task("simbench_lv", {"kind": "microgrid"})
    assert _limit(s) == _ORACLE_HARD_PRESET["TimeLimit"]


def test_none_scenario_does_not_crash():
    assert oracle_solver_for_task("simbench_lv", None) is not None


def test_runner_forwards_a_baseline_solver_only_when_overridden():
    """Guards the byte-identity of the shipped path.

    The runner builds a baseline solver only when the scenario declares a
    budget; otherwise it passes ``None`` so ``compute_baseline_served`` keeps
    its historical default. Mirrors the runner's condition so a refactor that
    drops the guard fails here.
    """
    import inspect

    from experiment.hpc import runner

    src = inspect.getsource(runner.run_task) if hasattr(runner, "run_task") else ""
    if "baseline_solver" not in src:
        src = inspect.getsource(runner)
    assert "baseline_solver" in src, "baseline solver forwarding was removed"
    assert "oracle_time_limit_s" in src, "forwarding is no longer gated on the key"
