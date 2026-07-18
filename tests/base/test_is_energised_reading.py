"""Boundary lock for the authoritative de-energised/energised reading predicate."""

from __future__ import annotations

import math

from scare.base.model import (
    DEENERGISED_PRESSURE_HIGH_PU,
    DEENERGISED_PRESSURE_PU,
    DEENERGISED_VM_PU,
    is_energised_reading,
)

EPS = 1e-9


def test_vm_pu_band():
    assert is_energised_reading("vm_pu", DEENERGISED_VM_PU + EPS) is True
    assert is_energised_reading("vm_pu", DEENERGISED_VM_PU) is False  # strict >
    assert is_energised_reading("vm_pu", DEENERGISED_VM_PU - EPS) is False
    assert is_energised_reading("vm_pu", 0.0) is False  # collapsed reading


def test_pressure_pu_band_both_sided():
    assert is_energised_reading("pressure_pu", DEENERGISED_PRESSURE_PU + EPS) is True
    assert is_energised_reading("pressure_pu", 1.0) is True
    assert is_energised_reading("pressure_pu", DEENERGISED_PRESSURE_PU) is False
    assert is_energised_reading("pressure_pu", DEENERGISED_PRESSURE_HIGH_PU) is False
    # sqrt(3) ~ 1.732 relaxed-Weymouth high-pressure artefact -> not energised.
    assert is_energised_reading("pressure_pu", math.sqrt(3)) is False


def test_t_k_band():
    assert is_energised_reading("t_k", EPS) is True
    assert is_energised_reading("t_k", 0.0) is False
    assert is_energised_reading("t_k", -5.0) is False


def test_unknown_variable_is_always_energised():
    assert is_energised_reading("loading_percent", 137.0) is True
    assert is_energised_reading("loading_percent", 0.0) is True


def test_none_returns_false_without_raising():
    # Locks the None-before-isfinite ordering that _branch_loading depends on.
    assert is_energised_reading("vm_pu", None) is False


def test_non_finite_is_not_energised():
    assert is_energised_reading("vm_pu", float("nan")) is False
    assert is_energised_reading("pressure_pu", float("inf")) is False
    assert is_energised_reading("t_k", float("-inf")) is False


def test_non_numeric_returns_false():
    assert is_energised_reading("vm_pu", "0.6") is True  # numeric string coerces
    assert is_energised_reading("vm_pu", object()) is False
