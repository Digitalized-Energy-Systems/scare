"""Unit tests for the priority-aware home-endpoint picker that
controls which group a PowerLine branch joins (decision 3a in the
plan).

The picker (``_line_home_endpoint``) sums priority-weighted demand
``Σ 2^(P − π) · |cap|`` over the loads at each endpoint and returns
the endpoint with the *lower* sum so an overload-driven shed lands
on the less-critical side.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scare.scenario.restoration import (
    _line_home_endpoint,
    _node_priority_weighted_demand,
)


def _mk_child(child_id: int, p_mw: float) -> SimpleNamespace:
    """A monee-shaped child stub: ``.model.values`` is the obs dict."""
    return SimpleNamespace(
        id=child_id,
        model=SimpleNamespace(values={"p_mw": p_mw}),
    )


def _mk_net(node_children: dict[int, list[SimpleNamespace]]) -> SimpleNamespace:
    """Build a monee-shaped network stub from ``{node_id: [child, ...]}``.

    Provides ``node_by_id`` and ``child_by_id`` and packs ``child_ids``
    on each node — the only API surface ``_line_home_endpoint`` uses.
    """
    children_index: dict[int, SimpleNamespace] = {}
    nodes: dict[int, SimpleNamespace] = {}
    for node_id, children in node_children.items():
        cids = [c.id for c in children]
        nodes[node_id] = SimpleNamespace(id=node_id, child_ids=cids)
        for c in children:
            children_index[c.id] = c

    return SimpleNamespace(
        node_by_id=lambda nid: nodes[nid],
        child_by_id=lambda cid: children_index[cid],
    )


class TestNodePriorityWeightedDemand:
    def test_single_tier1_load(self):
        net = _mk_net({5: [_mk_child(child_id=10, p_mw=2.0)]})
        priorities = {"child-10": 1}
        pwd = _node_priority_weighted_demand(5, net, priorities)
        # tier 1 weight = 2^(10-1) = 512; 512 * 2.0 = 1024
        assert pwd == pytest.approx(1024.0)

    def test_single_tier10_load(self):
        net = _mk_net({5: [_mk_child(child_id=10, p_mw=2.0)]})
        priorities = {"child-10": 10}
        pwd = _node_priority_weighted_demand(5, net, priorities)
        # tier 10 weight = 2^0 = 1; 1 * 2.0 = 2
        assert pwd == pytest.approx(2.0)

    def test_generator_skipped(self):
        # Generators (cap < 0) contribute nothing.
        net = _mk_net({5: [_mk_child(child_id=10, p_mw=-3.0)]})
        priorities = {"child-10": 1}
        pwd = _node_priority_weighted_demand(5, net, priorities)
        assert pwd == 0.0

    def test_missing_priority_defaults_tier_one(self):
        # Loads without an entry in ``priorities`` are treated as tier 1.
        net = _mk_net({5: [_mk_child(child_id=10, p_mw=2.0)]})
        priorities: dict[str, int] = {}
        pwd = _node_priority_weighted_demand(5, net, priorities)
        assert pwd == pytest.approx(512.0 * 2.0)

    def test_empty_node(self):
        net = _mk_net({5: []})
        pwd = _node_priority_weighted_demand(5, net, {})
        assert pwd == 0.0


class TestLineHomeEndpoint:
    def test_lower_priority_side_wins(self):
        # Endpoint 1 has tier-1 load; endpoint 2 has tier-10 load.
        # Home should be endpoint 2 (cheaper to shed there).
        net = _mk_net(
            {
                1: [_mk_child(child_id=100, p_mw=2.0)],
                2: [_mk_child(child_id=200, p_mw=2.0)],
            }
        )
        priorities = {"child-100": 1, "child-200": 10}
        branch = SimpleNamespace(id=(1, 2))
        home = _line_home_endpoint(branch, net, priorities)
        assert home == 2

    def test_symmetric_choice_breaks_to_smaller_id(self):
        # Both endpoints carry identical tier-5 demand → deterministic
        # tie-break by smaller node id.
        net = _mk_net(
            {
                3: [_mk_child(child_id=300, p_mw=1.0)],
                7: [_mk_child(child_id=700, p_mw=1.0)],
            }
        )
        priorities = {"child-300": 5, "child-700": 5}
        branch = SimpleNamespace(id=(7, 3))
        home = _line_home_endpoint(branch, net, priorities)
        assert home == 3

    def test_one_side_empty_one_side_loaded(self):
        # Empty side has pwd = 0 < the loaded side, so home is the
        # empty endpoint.  This is the "shed on the side with nothing
        # to shed" degenerate case — protects the loaded side from
        # spurious StartBalanceNegotiation on local overload.
        net = _mk_net(
            {
                9: [],
                10: [_mk_child(child_id=1000, p_mw=2.0)],
            }
        )
        priorities = {"child-1000": 1}
        branch = SimpleNamespace(id=(9, 10))
        home = _line_home_endpoint(branch, net, priorities)
        assert home == 9

    def test_quantity_outweighs_tier(self):
        # Five tier-5 loads (5 × 32 = 160 weight units per MW) on
        # endpoint 1 should still lose to one tier-1 load (512 weight
        # units per MW) of comparable size on endpoint 2.  Home = 1.
        net = _mk_net(
            {
                1: [_mk_child(child_id=i, p_mw=1.0) for i in range(10, 15)],
                2: [_mk_child(child_id=20, p_mw=1.0)],
            }
        )
        priorities = {
            "child-10": 5, "child-11": 5, "child-12": 5,
            "child-13": 5, "child-14": 5,
            "child-20": 1,
        }
        branch = SimpleNamespace(id=(1, 2))
        # endpoint 1: 5 * 32 * 1.0 = 160
        # endpoint 2: 1 * 512 * 1.0 = 512
        # Home = endpoint 1 (lower pwd).
        home = _line_home_endpoint(branch, net, priorities)
        assert home == 1
