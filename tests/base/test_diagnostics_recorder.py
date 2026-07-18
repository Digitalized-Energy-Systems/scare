"""Invariant lock for DiagnosticsRecorder and its module delegators.

The load-bearing invariants: instances are independent; ``arm()`` clears all
ledgers IN PLACE and never rebinds the singleton; and the module delegators
resolve the singleton at CALL TIME, so a captured ``import record_event`` still
lands in the ledger after a per-task re-arm.
"""

from __future__ import annotations

import pytest

import scare.base.runtime.diagnostics as diag
from scare.base.runtime.diagnostics import (
    DiagnosticsRecorder,
    action_log,
    arm,
    event_log,
    negotiation_summary,
    record_event,
    record_negotiation,
    record_regulate,
    set_trajectory_logging,
    trajectory_log,
)


@pytest.fixture(autouse=True)
def _reset_recorder():
    yield
    r = diag._RECORDER
    r._armed = False
    r._trajectory_armed = False
    r._log.clear()
    r._negotiation_log.clear()
    r._event_log.clear()
    r._trajectory_log.clear()


def test_two_recorders_are_independent():
    r1, r2 = DiagnosticsRecorder(), DiagnosticsRecorder()
    r1.arm()
    r1.record_event(t=1.0, kind="k")
    assert len(r1.event_log()) == 1
    assert r2.event_log() == []


def test_arm_clears_all_ledgers_in_place_without_rebinding():
    before = id(diag._RECORDER)
    arm()
    set_trajectory_logging(True)
    record_regulate(t=1.0, aid="child-1", sector="e", factor=0.5, reason="x")
    record_event(t=1.0, kind="ev")
    record_negotiation(t=1.0, aid="a", sector="e", nid="n", event="started")
    assert action_log() and event_log() and trajectory_log() and negotiation_summary()
    arm()  # per-task re-arm
    assert id(diag._RECORDER) == before  # singleton NOT rebound
    assert action_log() == []
    assert event_log() == []
    assert trajectory_log() == []
    assert negotiation_summary() == {}


def test_call_time_binding_survives_arm():
    # A name imported BEFORE arm() must still land in the singleton after arm().
    from scare.base.runtime.diagnostics import record_event as captured

    arm()
    captured(t=2.0, kind="captured_kind")
    assert any(e.kind == "captured_kind" for e in diag.event_log())


def test_noop_until_armed():
    diag._RECORDER._armed = False
    diag._RECORDER._event_log.clear()
    record_event(t=1.0, kind="ignored")
    assert event_log() == []
