"""Characterization tests for NeighbourRouter, the trust-weighted routing
collaborator extracted from EnergyBalanceNegotiator (C2)."""

from __future__ import annotations

from scare.service.balance.neighbour_router import NeighbourRouter
from scare.service.balance.trust import TrustLedger, TrustParams


def _router() -> NeighbourRouter:
    return NeighbourRouter(
        TrustLedger(
            TrustParams(
                decay_rate_per_s=0.1,
                recover_rate=0.6,
                liveness_threshold=0.5,
                initial=1.0,
            )
        )
    )


def test_next_hop_empty_returns_none():
    assert _router().next_hop([], "nid", 0, now=0.0) is None


def test_next_hop_is_deterministic_and_in_set():
    r = _router()
    peers = ["a", "b", "c"]
    h1 = r.next_hop(peers, "nid", 0, now=0.0)
    h2 = r.next_hop(peers, "nid", 0, now=0.0)
    assert h1 == h2  # same (nid, counter) routes identically
    assert h1 in peers


def test_next_hop_counter_varies_routing_key():
    r = _router()
    peers = ["a", "b", "c", "d", "e"]
    # Distinct counters feed distinct hash keys; each still resolves in-set.
    assert all(r.next_hop(peers, "nid", c, now=0.0) in peers for c in range(5))


def test_live_bootstraps_unknown_neighbours_optimistically():
    # Unknown neighbours start at K = initial = 1.0 > threshold 0.5, so all live.
    r = _router()
    peers = ["a", "b"]
    assert r.live(peers, now=0.0) == peers


def test_record_sender_and_touch_keep_routing_functional():
    r = _router()
    r.record_sender("a", now=0.0)
    r.touch(["a", "b"], now=0.0)
    assert r.live(["a", "b"], now=1.0) == ["a", "b"]
    assert r.next_hop(["a", "b"], "nid", 1, now=1.0) in ["a", "b"]
