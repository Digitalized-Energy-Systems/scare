"""The preflight refuses to submit a campaign that cannot solve step 0."""

import logging

import pytest

from experiment.hpc.config import TaskSpec
from experiment.hpc.preflight import (
    PreflightResult,
    _distinct_pairs,
    _IISCapture,
    _normalise,
    assert_preflight_clean,
    preflight_scenarios,
)


def _task(task_id, grid, scenario, variant="scare"):
    return TaskSpec(
        task_id=task_id, grid=grid, seed=task_id, n_failures=1,
        variant=variant, scenario=scenario,
    )


def test_pairs_collapse_by_grid_and_scenario():
    pv = {"kind": "pv_peak", "max_failures": 1}
    tasks = [
        _task(0, "g1", pv),
        _task(1, "g1", dict(reversed(list(pv.items())))),  # same dict, other order
        _task(2, "g1", {"kind": "clean"}),
        _task(3, "g2", pv),
    ]
    assert [t.task_id for t in _distinct_pairs(tasks)] == [0, 2, 3]


def test_oracle_tasks_are_not_preflighted():
    # The oracle solves its own min-load-shedding LP with curtailable loads,
    # so a simulation-solve verdict does not apply to it.
    tasks = [_task(0, "g1", {"kind": "pv_peak"}, variant="oracle")]
    assert _distinct_pairs(tasks) == []


def test_seeds_do_not_multiply_the_work():
    pv = {"kind": "pv_peak"}
    tasks = [_task(i, "g1", pv) for i in range(500)]
    assert len(_distinct_pairs(tasks)) == 1


@pytest.mark.parametrize(
    "member,expected",
    [
        ("branch_12_42_0__current_pu_squared [LB]", "branch_*__current_pu_squared [LB]"),
        ("node_47__el_mw [UB]", "node_*__el_mw [UB]"),
        ("child_118__p_mw [LB]", "child_*__p_mw [LB]"),
        ("node_388_eq_35", "node_*_eq_35"),
        ("R11693", "R#"),
    ],
)
def test_signature_normalisation_collapses_ids(member, expected):
    assert _normalise(member) == expected


def test_iis_capture_reads_the_monee_block():
    capture = _IISCapture()
    record = logging.LogRecord(
        "monee.solver.gurobipy", logging.ERROR, __file__, 0,
        "Gurobi solve infeasible (status=3).  Diagnostic report:\n"
        "Irreducible Inconsistent Subsystem (Gurobi IIS):\n"
        "  Variable bounds in IIS (1):\n"
        "    node_47__el_mw [UB]\n"
        "  Constraints in IIS (2):\n"
        "    node_2_eq_0\n"
        "    branch_1_5_0_eq_4\n",
        None, None,
    )
    capture.emit(record)
    assert capture.members == ["node_47__el_mw [UB]", "node_2_eq_0", "branch_1_5_0_eq_4"]


def test_clean_preflight_returns_results(monkeypatch):
    ok = [PreflightResult("g1", {"kind": "clean"}, True)]
    monkeypatch.setattr(
        "experiment.hpc.preflight.preflight_scenarios", lambda *a, **k: ok
    )
    assert assert_preflight_clean([]) == ok


def test_infeasible_pair_aborts_submission(monkeypatch):
    bad = [
        PreflightResult("g1", {"kind": "clean"}, True),
        PreflightResult("simbench_lv_small", {"kind": "pv_peak"}, False, "IIS: node_*__el_mw [UB]"),
    ]
    monkeypatch.setattr(
        "experiment.hpc.preflight.preflight_scenarios", lambda *a, **k: bad
    )
    with pytest.raises(SystemExit) as exc:
        assert_preflight_clean([])
    message = str(exc.value)
    assert "1 of 2" in message
    assert "simbench_lv_small" in message
    assert "node_*__el_mw [UB]" in message
    assert "--no-preflight" in message


def test_failure_is_reported_per_pair(monkeypatch):
    calls = []

    def fake_check(task, **kwargs):
        calls.append(task.grid)
        return PreflightResult(task.grid, task.scenario, task.grid != "bad")

    monkeypatch.setattr("experiment.hpc.preflight.check_pair", fake_check)
    results = preflight_scenarios(
        [_task(0, "good", {"kind": "clean"}), _task(1, "bad", {"kind": "pv_peak"})]
    )
    assert calls == ["good", "bad"]
    assert [r.ok for r in results] == [True, False]
