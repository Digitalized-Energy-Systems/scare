"""One failed physics step must count as exactly one solver infeasibility.

The old counter deduped on a 1 s wall-clock window, which merged genuinely
distinct steps whenever solves ran faster than a second: eval_full_v2 reported
2277 infeasibilities against 3202 real failed steps, a 29 % undercount biased
toward the fast-solving grids. Identity now comes from the step index.
"""

import logging

from experiment.hpc.runner_logging import _SolverFailureCounter


def _rec(name: str, level: int, msg: str) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 0, msg, None, None)


def _feed(counter: _SolverFailureCounter, records) -> None:
    for r in records:
        counter.filter(r)


def _backend(step_note: str = "") -> logging.LogRecord:
    return _rec(
        "monee.solver.gurobipy",
        logging.ERROR,
        f"Gurobi solve infeasible (status=3).  Diagnostic report:{step_note}",
    )


def _stepper(index: int) -> logging.LogRecord:
    return _rec(
        "monee.simulation.stepper",
        logging.WARNING,
        f"Stepper step {index} failed: Step {index}: solver reported an "
        "unsuccessful solve (success=False)",
    )


def test_backend_error_and_stepper_warning_are_one_event():
    c = _SolverFailureCounter()
    _feed(c, [_backend(), _stepper(19)])
    c.finalize()
    assert c.infeasible_count == 1
    assert c.failed_steps == [19]


def test_distinct_steps_within_one_second_all_count():
    c = _SolverFailureCounter()
    for step in range(5):
        _feed(c, [_backend(), _stepper(step)])
    c.finalize()
    assert c.infeasible_count == 5
    assert c.failed_steps == [0, 1, 2, 3, 4]


def test_repeated_record_for_the_same_step_counts_once():
    c = _SolverFailureCounter()
    _feed(c, [_backend(), _stepper(7), _stepper(7)])
    c.finalize()
    assert c.infeasible_count == 1


def test_first_failed_step_zero_flags_a_born_infeasible_net():
    c = _SolverFailureCounter()
    _feed(c, [_backend(), _stepper(0), _backend(), _stepper(1)])
    c.finalize()
    assert c.first_failed_step == 0
    assert c.infeasible_count == 2


def test_first_failed_step_is_none_when_nothing_failed():
    c = _SolverFailureCounter()
    c.finalize()
    assert c.first_failed_step is None
    assert c.infeasible_count == 0


def test_pyomo_echo_does_not_double_count():
    c = _SolverFailureCounter()
    _feed(
        c,
        [
            _rec("monee.solver.pyo", logging.ERROR, "Pyomo solve infeasible"),
            _rec(
                "pyomo.core",
                logging.WARNING,
                "Loading a SolverResults object with a warning status into "
                "model; termination condition: infeasible",
            ),
        ],
    )
    c.finalize()
    assert c.infeasible_count == 1


def test_oracle_path_without_a_stepper_still_counts():
    # The oracle solves via run_energy_flow, so no step record ever closes it.
    c = _SolverFailureCounter()
    _feed(c, [_rec("monee.solver.pyo", logging.ERROR, "Pyomo solve infeasible")])
    assert c.infeasible_count == 0, "counted before finalize()"
    c.finalize()
    assert c.infeasible_count == 1
    assert c.failed_steps == []


def test_consecutive_unattributed_backend_failures_each_count():
    c = _SolverFailureCounter()
    _feed(c, [_backend(), _backend(), _backend()])
    c.finalize()
    assert c.infeasible_count == 3


def test_env_errors_land_in_warnings_not_infeasibilities():
    c = _SolverFailureCounter()
    _feed(
        c,
        [
            _rec("monee.solver.pyo", logging.ERROR, "GurobiError: HostID mismatch"),
            _rec("monee.solver.pyo", logging.WARNING, "solver returned non-ok status"),
        ],
    )
    c.finalize()
    assert c.infeasible_count == 0
    assert c.warning_count == 2
    assert c.count == 2


def test_info_records_are_ignored():
    c = _SolverFailureCounter()
    _feed(c, [_rec("monee.simulation.stepper", logging.INFO, "Stepper step 3 failed")])
    c.finalize()
    assert c.count == 0
