"""``is_nonphysical_t_k`` quantifies the sub-freezing heat readings without
changing any verdict.

A heat junction below 273.15 K is not a temperature the water was at: monee's
nodal heat balance loses rank in T_n once the mass-flow terms vanish, because
the conduction regulariser is scaled by ``grid.node_heat_reg_kgs``, which is
never set anywhere in the stack. T_n is then pinned only by its Var box,
``t_pu in [0.3, 2.0]`` = [106.8, 712] K — and the observed minimum across the
campaign is exactly 106.800 = 0.3 * 356.

The predicate stays out of the compliance gate on purpose: excluding these
readings would flip 14 eval_full_v2 tasks to compliant, all of them SCARE.
"""

from __future__ import annotations

import pytest

from experiment.eval.metrics import (
    is_nonphysical_t_k,
    is_t_pu_floor_artifact,
)
from scare.base.model import is_energised_reading

VAR_FLOOR_T_K = 106.8


@pytest.mark.parametrize("value", [VAR_FLOOR_T_K, 230.4, 238.418, 260.0, 273.14])
def test_subfreezing_readings_are_flagged_nonphysical(value):
    assert is_nonphysical_t_k("t_k", value)


@pytest.mark.parametrize(
    "value",
    [
        273.15,  # the bound itself is physical (ice point)
        296.15,  # ambient — implausible for DHS but not impossible
        303.0,  # the median cold junction
        313.15,  # the service floor
        396.0,  # a hot junction
    ],
)
def test_plausible_readings_are_not_flagged(value):
    assert not is_nonphysical_t_k("t_k", value)


@pytest.mark.parametrize("variable", ["vm_pu", "pressure_pu", "loading_percent"])
def test_only_t_k_is_affected(variable):
    assert not is_nonphysical_t_k(variable, 10.0)


@pytest.mark.parametrize("value", [None, "", "abc", float("nan"), float("inf")])
def test_non_numeric_and_non_finite_are_not_flagged(value):
    assert not is_nonphysical_t_k("t_k", value)


def test_it_subsumes_the_var_floor_artifact():
    """The Var-floor pin is a strict subset — 106.8 K is below freezing."""
    assert is_t_pu_floor_artifact("t_k", VAR_FLOOR_T_K)
    assert is_nonphysical_t_k("t_k", VAR_FLOOR_T_K)


def test_it_does_not_widen_the_gating_predicate():
    """Guards the deliberate scope split: the gate still drops only the exact
    Var-floor pin, so the sub-freezing tail keeps counting as a violation and
    no published verdict moves."""
    for value in (230.4, 260.0, 272.0):
        assert is_nonphysical_t_k("t_k", value)
        assert not is_t_pu_floor_artifact("t_k", value)


def test_it_does_not_touch_the_simulator_facing_band():
    """``is_energised_reading`` drives live actuators; widening it would change
    control behaviour and invalidate every recorded run."""
    assert is_energised_reading("t_k", VAR_FLOOR_T_K)
    assert is_energised_reading("t_k", 260.0)
