"""Lock for the shared diary terminal-event constant + invariant."""

from __future__ import annotations

from experiment.eval.claims import DIARY_TERMINAL_EVENTS, diary_invariant_holds
from experiment.eval.results import _diary_invariant_holds


def test_terminal_events_constant():
    assert DIARY_TERMINAL_EVENTS == (
        "finished", "timed_out", "cancelled", "abandoned", "stalled",
    )


def test_invariant():
    assert diary_invariant_holds({"started": 3, "finished": 2, "stalled": 1}) is True
    assert diary_invariant_holds({"started": 3, "finished": 2}) is False
    assert diary_invariant_holds({}) is True  # 0 == 0


def test_results_delegates_to_claims():
    assert _diary_invariant_holds({"started": 1, "cancelled": 1}) is True
    assert _diary_invariant_holds({"started": 2, "cancelled": 1}) is False
