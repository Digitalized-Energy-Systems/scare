"""Unit tests for the soft congestion-price store (v1 line-loading relief).

Covers the additive per-branch price -> generation-ceiling kernel in
``scare.base.util``: sum across branches, freshness expiry, clear-on-zero, and
the [0,1] ceiling clamp.
"""

from __future__ import annotations

from scare.base.util import (
    line_congestion_ceiling,
    publish_line_congestion_price,
)


class _Beh:
    """Minimal behavior stub: the util store helpers only need attribute bags."""


def test_no_price_means_no_cap():
    b = _Beh()
    assert line_congestion_ceiling(b, "child-1", now=0.0, ttl=3.0) == 1.0


def test_single_branch_price_sets_ceiling():
    b = _Beh()
    publish_line_congestion_price(b, "branch-a", "child-1", 0.4, now=0.0)
    assert abs(line_congestion_ceiling(b, "child-1", 0.0, 3.0) - 0.6) < 1e-12


def test_prices_sum_across_branches():
    b = _Beh()
    publish_line_congestion_price(b, "branch-a", "child-1", 0.3, now=0.0)
    publish_line_congestion_price(b, "branch-b", "child-1", 0.4, now=0.0)
    # 1 - (0.3 + 0.4)
    assert abs(line_congestion_ceiling(b, "child-1", 0.0, 3.0) - 0.3) < 1e-12


def test_ceiling_clamped_nonnegative():
    b = _Beh()
    publish_line_congestion_price(b, "branch-a", "child-1", 0.8, now=0.0)
    publish_line_congestion_price(b, "branch-b", "child-1", 0.8, now=0.0)
    assert line_congestion_ceiling(b, "child-1", 0.0, 3.0) == 0.0


def test_stale_price_drops_out():
    b = _Beh()
    publish_line_congestion_price(b, "branch-a", "child-1", 0.5, now=0.0)
    # ttl=3.0: a read at t=5 is stale -> no cap.
    assert line_congestion_ceiling(b, "child-1", now=5.0, ttl=3.0) == 1.0


def test_zero_price_clears_entry():
    b = _Beh()
    publish_line_congestion_price(b, "branch-a", "child-1", 0.5, now=0.0)
    publish_line_congestion_price(b, "branch-a", "child-1", 0.0, now=0.1)
    assert line_congestion_ceiling(b, "child-1", 0.1, 3.0) == 1.0


def test_price_is_per_generator():
    b = _Beh()
    publish_line_congestion_price(b, "branch-a", "child-1", 0.4, now=0.0)
    # A different downstream gen with no price is uncapped.
    assert line_congestion_ceiling(b, "child-2", 0.0, 3.0) == 1.0
