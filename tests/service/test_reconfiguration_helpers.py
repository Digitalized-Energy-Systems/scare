"""Unit tests for the pure reconfiguration toggle-race guards.

The env "switch" action is a toggle, so two endpoints acting on the same open
tie would close and then re-open it. These cover the two message-free guards:
initiator dedup and the act-time skip-if-not-known-open decision.
"""

from __future__ import annotations

from scare.service.reconfiguration import (
    is_initiating_endpoint,
    should_close_tie,
    switch_state_from_obs,
)


def test_exactly_one_endpoint_initiates():
    # Each endpoint sees (its own node, the other node); exactly one wins.
    assert is_initiating_endpoint(3, 7) is True
    assert is_initiating_endpoint(7, 3) is False
    assert is_initiating_endpoint(3, 7) != is_initiating_endpoint(7, 3)


def test_initiator_dedup_mixed_types_is_deterministic():
    a, b = "node-a", 7
    assert is_initiating_endpoint(a, b) != is_initiating_endpoint(b, a)


def test_switch_state_from_obs():
    assert switch_state_from_obs({"on_off": 0}) == 0
    assert switch_state_from_obs({"on_off": 1}) == 1
    assert switch_state_from_obs({"loading_percent": 50.0}) is None  # no reading
    assert switch_state_from_obs({}) is None
    assert switch_state_from_obs(None) is None
    assert switch_state_from_obs({"on_off": "bogus"}) is None


def test_should_close_tie_only_when_known_open():
    assert should_close_tie(0) is True
    assert should_close_tie(1) is False  # already closed: toggle would open it
    assert should_close_tie(None) is False  # unknown: never toggle
