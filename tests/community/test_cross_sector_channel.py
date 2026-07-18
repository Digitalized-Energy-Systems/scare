"""Publish/read round-trip lock for CrossSectorChannel (the typed facade over the
per-behavior cross-sector HolonSummary bus)."""

from __future__ import annotations

from types import SimpleNamespace

from scare.base.model import Sector
from scare.community.summary import CrossSectorChannel


def test_publish_read_roundtrip_per_sector():
    b = SimpleNamespace()
    ch = CrossSectorChannel.for_behavior(b)
    ch.publish(Sector.ELECTRICITY, "child-1", "A")
    ch.publish(Sector.ELECTRICITY, "child-2", "B")
    ch.publish(Sector.HEAT, "child-3", "C")
    assert ch.read(Sector.ELECTRICITY) == {"child-1": "A", "child-2": "B"}
    assert ch.read(Sector.HEAT) == {"child-3": "C"}
    assert ch.read(Sector.GAS) == {}  # absent sector -> empty


def test_store_is_shared_across_wrappers_via_behavior():
    b = SimpleNamespace()
    CrossSectorChannel.for_behavior(b).publish(Sector.GAS, "k", "v")
    # a fresh wrapper reads the same behavior-backed store
    assert CrossSectorChannel.for_behavior(b).read(Sector.GAS) == {"k": "v"}


def test_latest_publish_wins_per_key():
    b = SimpleNamespace()
    ch = CrossSectorChannel.for_behavior(b)
    ch.publish(Sector.ELECTRICITY, "child-1", "old")
    ch.publish(Sector.ELECTRICITY, "child-1", "new")
    assert ch.read(Sector.ELECTRICITY) == {"child-1": "new"}
