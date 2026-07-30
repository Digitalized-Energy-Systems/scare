"""The preflight refuses to submit a campaign that cannot solve step 0."""

import logging
from types import SimpleNamespace

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
        task_id=task_id,
        grid=grid,
        seed=task_id,
        n_failures=1,
        variant=variant,
        scenario=scenario,
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
        (
            "branch_12_42_0__current_pu_squared [LB]",
            "branch_*__current_pu_squared [LB]",
        ),
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
        "monee.solver.gurobipy",
        logging.ERROR,
        __file__,
        0,
        "Gurobi solve infeasible (status=3).  Diagnostic report:\n"
        "Irreducible Inconsistent Subsystem (Gurobi IIS):\n"
        "  Variable bounds in IIS (1):\n"
        "    node_47__el_mw [UB]\n"
        "  Constraints in IIS (2):\n"
        "    node_2_eq_0\n"
        "    branch_1_5_0_eq_4\n",
        None,
        None,
    )
    capture.emit(record)
    assert capture.members == [
        "node_47__el_mw [UB]",
        "node_2_eq_0",
        "branch_1_5_0_eq_4",
    ]


def test_clean_preflight_returns_results(monkeypatch):
    ok = [PreflightResult("g1", {"kind": "clean"}, True)]
    monkeypatch.setattr(
        "experiment.hpc.preflight.preflight_scenarios", lambda *a, **k: ok
    )
    assert assert_preflight_clean([]) == ok


def test_infeasible_pair_aborts_submission(monkeypatch):
    bad = [
        PreflightResult("g1", {"kind": "clean"}, True),
        PreflightResult(
            "simbench_lv_small", {"kind": "pv_peak"}, False, "IIS: node_*__el_mw [UB]"
        ),
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


# --- margin + contingency reporting -------------------------------------
#
# Step-0 solvability is necessary but not sufficient: LV-S passed it while
# sitting at exactly 100% of the LP's hard ampacity bound and carrying a
# single-branch kill switch. Both are reported, neither blocks submission.


def test_tight_pair_solves_but_is_flagged():
    loaded = PreflightResult(
        "simbench_lv_small", {"kind": "pv_peak"}, True, max_loading_pct=300.0
    )
    roomy = PreflightResult(
        "simbench_lv", {"kind": "clean"}, True, max_loading_pct=81.0
    )
    assert loaded.tight and not roomy.tight
    assert "300.0%" in loaded.margin_detail


def test_missing_loading_is_not_treated_as_tight():
    """An unsolved net's Var defaults are phantoms — absent means unknown."""
    assert not PreflightResult("g", {}, True, max_loading_pct=None).tight


def test_kill_switch_is_reported_in_margin_detail():
    res = PreflightResult(
        "simbench_lv_small",
        {"kind": "clean"},
        True,
        max_loading_pct=90.0,
        kill_switches=[[37, 36, 0]],
        n_contingencies_scanned=47,
    )
    assert res.tight is False  # loading alone is fine ...
    assert "1/47 single-branch contingencies infeasible" in res.margin_detail
    assert "[37, 36, 0]" in res.margin_detail


def test_tight_and_kill_switches_warn_but_do_not_abort(monkeypatch, caplog):
    risky = [
        PreflightResult(
            "simbench_lv_small",
            {"kind": "pv_peak"},
            True,
            max_loading_pct=300.0,
            kill_switches=[[37, 36, 0]],
            n_contingencies_scanned=47,
        ),
    ]
    monkeypatch.setattr(
        "experiment.hpc.preflight.preflight_scenarios", lambda *a, **k: risky
    )
    with caplog.at_level(logging.WARNING):
        assert assert_preflight_clean([]) == risky  # no SystemExit
    assert "no margin" in caplog.text
    assert "simbench_lv_small" in caplog.text


def test_contingency_scan_finds_the_branch_that_breaks_the_lp(monkeypatch):
    """One solve per branch; a branch whose removal makes the LP infeasible is
    a kill switch, while one that merely sheds load is the experiment working."""
    from experiment.hpc import preflight as pf

    class _Branch:
        def __init__(self, bid):
            self.id, self.active = bid, True

    def fake_build(task):
        return SimpleNamespace(branches=[_Branch((1, 2, 0)), _Branch((37, 36, 0))])

    def fake_solve(net, _limit):
        dead = [b.id for b in net.branches if not b.active]
        return (dead == [(37, 36, 0)], "infeasible" if dead else "", [])

    monkeypatch.setattr(pf, "_build_net", fake_build)
    monkeypatch.setattr(pf, "_solve_step0", fake_solve)

    kills, n = pf.scan_branch_contingencies(_task(0, "simbench_lv_small", {}))
    assert n == 2
    assert kills == [[37, 36, 0]]


def test_contingency_scan_respects_the_limit(monkeypatch):
    from experiment.hpc import preflight as pf

    class _Branch:
        def __init__(self, bid):
            self.id, self.active = bid, True

    monkeypatch.setattr(
        pf,
        "_build_net",
        lambda task: SimpleNamespace(
            branches=[_Branch((i, i + 1, 0)) for i in range(9)]
        ),
    )
    monkeypatch.setattr(pf, "_solve_step0", lambda net, _l: (False, "", []))

    _, n = pf.scan_branch_contingencies(_task(0, "g", {}), limit=3)
    assert n == 3
