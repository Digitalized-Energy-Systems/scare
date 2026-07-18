"""Round-trip + fallback lock for the centralized aid grammar.

The crux: the two parse fallbacks are DISTINCT and must stay so — node parsing
returns None on a malformed id (detection drops it), trailing-id parsing returns
0 (reconfiguration tolerates it).
"""

from __future__ import annotations

from types import SimpleNamespace

from scare.base.addressing import (
    addr_aid,
    branch_aid,
    branch_aid_from_addrs,
    child_aid,
    id_from_aid,
    is_child_aid,
    is_node_aid,
    node_aid,
    node_id_from_aid,
)


def test_builders():
    assert node_aid(5) == "node-5"
    assert child_aid(12) == "child-12"


def test_membership_predicates():
    assert is_node_aid("node-5") and not is_node_aid("child-5")
    assert is_child_aid("child-5") and not is_child_aid("node-5")


def test_branch_aid_orders_hi_lo_and_ignores_extra():
    assert branch_aid((3, 7)) == "branch-7-3"
    assert branch_aid((7, 3)) == "branch-7-3"
    assert branch_aid((1, 9, 0)) == "branch-9-1"


def test_node_id_from_aid_roundtrip_and_none_fallback():
    assert node_id_from_aid(node_aid(42)) == 42
    assert node_id_from_aid("child-42") is None  # not a node aid
    assert node_id_from_aid("node-x") is None  # malformed -> None


def test_id_from_aid_zero_fallback():
    assert id_from_aid("branch-7-3") == 3
    assert id_from_aid("child-12") == 12
    assert id_from_aid("weird") == 0  # malformed -> 0, NOT None


def test_addr_aid():
    assert addr_aid(SimpleNamespace(aid="node-1")) == "node-1"
    assert addr_aid("node-2") == "node-2"  # no .aid -> str(addr)


def test_branch_aid_from_addrs_uses_zero_fallback():
    a = SimpleNamespace(aid="node-3")
    b = SimpleNamespace(aid="node-7")
    assert branch_aid_from_addrs(a, b) == "branch-7-3"
    c = SimpleNamespace(aid="weird")
    assert branch_aid_from_addrs(a, c) == "branch-3-0"


def test_the_two_fallbacks_are_distinct():
    assert node_id_from_aid("node-x") is None
    assert id_from_aid("node-x") == 0
