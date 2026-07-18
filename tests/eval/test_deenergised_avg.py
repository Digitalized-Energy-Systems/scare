"""Characterization lock for metrics._is_deenergised_avg after it was routed
through the shared is_energised_reading predicate.

The subtle invariant: a non-finite average (nan) must stay KEPT (not scored as a
de-energisation), which the ``math.isfinite(val) and ...`` prefix preserves.
"""

from __future__ import annotations

import math

from experiment.eval.metrics import _is_deenergised_avg


def test_bands_match_the_energised_predicate():
    assert _is_deenergised_avg("vm_pu", 0.05) is True  # collapsed reading
    assert _is_deenergised_avg("vm_pu", 0.98) is False  # genuine served
    assert _is_deenergised_avg("pressure_pu", 0.05) is True  # low artefact
    assert _is_deenergised_avg("pressure_pu", math.sqrt(3)) is True  # high artefact
    assert _is_deenergised_avg("pressure_pu", 1.0) is False  # genuine
    assert _is_deenergised_avg("t_k", 0.0) is True
    assert _is_deenergised_avg("t_k", 350.0) is False
    assert _is_deenergised_avg("loading_percent", 120.0) is False  # no artefact axis


def test_nan_average_is_kept_not_scored_as_deenergised():
    assert _is_deenergised_avg("vm_pu", float("nan")) is False
    assert _is_deenergised_avg("pressure_pu", float("nan")) is False
    assert _is_deenergised_avg("t_k", float("nan")) is False
