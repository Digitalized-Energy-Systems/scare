"""Boundary + invariant lock for FreshnessStore (backs the 7 util TTL registries).

The load-bearing assertions: strict ``<`` staleness (exactly-at-ttl is stale),
non-evicting reads (a stale key survives so stamp can re-arm it), no-op stamp on
an absent key, and a per-call ttl override distinct from the construction ttl.
"""

from __future__ import annotations

from scare.base.freshness import FreshnessStore


def _store(ttl=3.0):
    return FreshnessStore({}, ttl)


def test_fresh_stale_boundary_is_strict_less_than():
    s = _store(ttl=3.0)
    s.put("k", 42, now=0.0)
    assert s.get("k", now=2.999) == 42  # just under ttl -> fresh
    assert s.get("k", now=3.0) is None  # exactly at ttl -> STALE
    assert s.get("k", now=3.001) is None


def test_get_is_non_evicting_and_stamp_re_arms():
    s = _store(ttl=3.0)
    s.put("k", 42, now=0.0)
    assert s.get("k", now=5.0) is None  # stale
    # ... but the entry survives the stale read, so stamp revives it:
    s.stamp("k", now=5.0)
    assert s.get("k", now=6.0) == 42  # re-armed, payload preserved


def test_stamp_is_no_op_when_absent():
    s = _store(ttl=3.0)
    s.stamp("ghost", now=1.0)  # must not insert
    assert s.get("ghost", now=1.0) is None


def test_per_call_ttl_override():
    s = _store(ttl=3.0)
    s.put("k", 1.0, now=0.0)
    # construction ttl says stale at 4.0, but a wider per-call ttl keeps it fresh
    assert s.get("k", now=4.0) is None
    assert s.get("k", now=4.0, ttl=10.0) == 1.0
    # a narrower per-call ttl makes it stale earlier
    assert s.get("k", now=2.0, ttl=1.0) is None


def test_pop_removes():
    s = _store()
    s.put("k", 1, now=0.0)
    s.pop("k")
    assert s.get("k", now=0.0) is None
    s.pop("missing")  # no raise


def test_items_fresh_filters_and_honors_ttl_override():
    s = _store(ttl=3.0)
    s.put("a", "pa", now=0.0)
    s.put("b", "pb", now=2.0)
    assert dict(s.items_fresh(now=2.5)) == {"a": "pa", "b": "pb"}
    assert dict(s.items_fresh(now=3.5)) == {"b": "pb"}  # a is stale
    # per-call ttl override widens the window
    assert dict(s.items_fresh(now=3.5, ttl=10.0)) == {"a": "pa", "b": "pb"}


def test_none_payload_presence_only():
    # gen-lock stores payload=None: presence+freshness, no value.
    s = _store(ttl=3.0)
    s.put("gen", None, now=0.0)
    # a missing key and a fresh-None key both read as None -> use items/membership
    assert "gen" in dict(s.items_fresh(now=1.0))
    assert "gen" not in dict(s.items_fresh(now=3.0))
