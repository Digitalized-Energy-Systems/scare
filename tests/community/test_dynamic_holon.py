"""Unit tests for the L2 dynamic holon-membership filter.

Two layers:

1. :class:`DynamicHolonRole`'s ``is_live`` predicate after a simulated
   branch failure, driven through the shared :class:`GridTopologyMirror`.
2. :class:`HolonicCommunityRole._live_members`, exercised with a stub
   ``LivePeerFilter`` to confirm peer iteration honours the filter.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from scare.base.model import Sector
from scare.base.topology_mirror import GridTopologyMirror
from scare.community.dynamic_holon import DynamicHolonRole
from scare.community.holonic import HolonicCommunityRole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _three_leader_grid() -> tuple[GridTopologyMirror, dict[str, Any]]:
    """A 3-leader electrical line: L1 at node 1, L2 at node 2, L3 at node 4.

    ::

        L1 (n1) --b1-- n2/L2 --b2-- n3 --b3-- n4/L3

    Breaking ``b2`` isolates L3 from L1+L2.
    """
    branches = {
        ("b1",): (1, 2),
        ("b2",): (2, 3),
        ("b3",): (3, 4),
    }
    sector_tag = {bid: "electricity" for bid in branches}
    mirror = GridTopologyMirror(branches=branches, branch_sector=sector_tag)
    aid_to_node = {"leader-1": 1, "leader-2": 2, "leader-3": 4}
    return mirror, aid_to_node


class _StubAddr:
    """Minimal ``addr`` stand-in for the ``LivePeerFilter`` contract.
    The filter only reads ``.aid``.
    """

    def __init__(self, aid: str) -> None:
        self.aid = aid

    def __repr__(self) -> str:
        return f"<addr {self.aid}>"


class _NoOpContext:
    """Minimal context surface for _reassess_membership without scheduling."""

    def __init__(self, aid: str, t: float = 100.0) -> None:
        self.aid = aid
        self.current_timestamp = t


# ---------------------------------------------------------------------------
# DynamicHolonRole: is_live semantics
# ---------------------------------------------------------------------------


class TestDynamicHolonRoleIsLive:
    def test_unknown_addrs_admitted_by_default(self) -> None:
        mirror, aid_to_node = _three_leader_grid()
        role = DynamicHolonRole(
            behavior=SimpleNamespace(),
            sector=Sector.ELECTRICITY,
            my_node_id=1,
            aid_to_node_id=aid_to_node,
            mirror=mirror,
        )
        # Additive filter: everything is alive until declared dead.
        assert role.is_live(_StubAddr("leader-2"))
        assert role.is_live(_StubAddr("stranger"))

    def test_drop_marks_unreachable_only(self) -> None:
        mirror, aid_to_node = _three_leader_grid()
        role = DynamicHolonRole(
            behavior=SimpleNamespace(),
            sector=Sector.ELECTRICITY,
            my_node_id=1,
            aid_to_node_id=aid_to_node,
            mirror=mirror,
        )
        # Populate the unreachable set directly, as reassess would.
        role._unreachable_aids.add("leader-3")
        assert role.is_live(_StubAddr("leader-2"))
        assert not role.is_live(_StubAddr("leader-3"))

    def test_self_aware_filter_after_break(self) -> None:
        """After breaking middle branch ``b2``, node 4 (leader-3) is
        unreachable from node 1; replaying the role's BFS marks it dead.
        """
        mirror, aid_to_node = _three_leader_grid()
        role = DynamicHolonRole(
            behavior=SimpleNamespace(),
            sector=Sector.ELECTRICITY,
            my_node_id=1,
            aid_to_node_id=aid_to_node,
            mirror=mirror,
        )
        mirror.mark_broken(("b2",))

        reachable_nodes = mirror.reachable_from(1, sector=Sector.ELECTRICITY)
        for aid, node_id in aid_to_node.items():
            if aid != "leader-1" and node_id not in reachable_nodes:
                role._unreachable_aids.add(aid)

        assert role.is_live(_StubAddr("leader-1"))
        assert role.is_live(_StubAddr("leader-2"))
        assert not role.is_live(_StubAddr("leader-3"))

    def test_never_drops_self(self) -> None:
        """Reassess must never drop the leader's own aid — that would
        orphan the holon."""
        mirror, aid_to_node = _three_leader_grid()
        role = DynamicHolonRole(
            behavior=SimpleNamespace(),
            sector=Sector.ELECTRICITY,
            my_node_id=1,
            aid_to_node_id=aid_to_node,
            mirror=mirror,
        )
        assert role.is_live(_StubAddr("leader-1"))


# ---------------------------------------------------------------------------
# HolonicCommunityRole._live_members: filter wiring
# ---------------------------------------------------------------------------


class _DroppingFilter:
    """LivePeerFilter that drops a configured set of aids."""

    def __init__(self, dropped: set[str]) -> None:
        self.dropped = set(dropped)

    def is_live(self, addr: Any) -> bool:
        aid = getattr(addr, "aid", None) or str(addr)
        return aid not in self.dropped


class TestHolonicCommunityRoleLiveMembers:
    def _build(self, filt) -> HolonicCommunityRole:
        return HolonicCommunityRole(
            sector=Sector.ELECTRICITY,
            live_member_filter=filt,
        )

    def test_no_filter_is_passthrough(self) -> None:
        role = self._build(filt=None)
        members = [_StubAddr("a"), _StubAddr("b"), _StubAddr("c")]
        assert role._live_members(members) == members

    def test_filter_drops_unreachable(self) -> None:
        role = self._build(filt=_DroppingFilter({"b"}))
        members = [_StubAddr("a"), _StubAddr("b"), _StubAddr("c")]
        kept = role._live_members(members)
        assert [m.aid for m in kept] == ["a", "c"]

    def test_filter_drops_all(self) -> None:
        role = self._build(filt=_DroppingFilter({"a", "b", "c"}))
        members = [_StubAddr("a"), _StubAddr("b"), _StubAddr("c")]
        assert role._live_members(members) == []
