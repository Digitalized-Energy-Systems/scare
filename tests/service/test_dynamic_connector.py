"""Unit tests for the L3 dynamic CP-connector filter (Concept C).

Symmetric to the L2 tests: drive the reachability path through the
shared :class:`GridTopologyMirror` and verify the resulting
``is_live`` predicate, then verify that
:class:`EnergyConverterRole._live_connectors` honours the filter.

The cross-sector flavour of L3 reachability — same-sector edges *and*
CP bridges — is exercised separately so the L2/L3 distinction is
explicit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from scare.base.model import Sector
from scare.base.topology.topology_mirror import GridTopologyMirror
from scare.service.coupling.cp import EnergyConverterRole
from scare.service.coupling.dynamic_connector import DynamicConnectorRole

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cp_grid() -> tuple[GridTopologyMirror, dict[str, Any]]:
    """A CP bridging electricity and gas, with two leaders per sector.

    ::

        EL:   leader-el-a (n1) -b1- n2/CP -b2- (n3) leader-el-b
                                     |
                                     bridge (CP)
                                     |
        GAS:  leader-gs-a (n10) -b3- n11/CP-side -b4- (n12) leader-gs-b

    ``b1`` failing isolates ``leader-el-a`` from the CP (n2).
    The bridge edge failing islands the gas side entirely from the CP.
    """
    branches = {
        ("b1",): (1, 2),
        ("b2",): (2, 3),
        ("bridge",): (2, 11),
        ("b3",): (10, 11),
        ("b4",): (11, 12),
    }
    sector_tag = {
        ("b1",): "electricity",
        ("b2",): "electricity",
        ("bridge",): "cp",
        ("b3",): "gas",
        ("b4",): "gas",
    }
    mirror = GridTopologyMirror(branches=branches, branch_sector=sector_tag)
    leader_node = {
        "leader-el-a": 1,
        "leader-el-b": 3,
        "leader-gs-a": 10,
        "leader-gs-b": 12,
    }
    return mirror, leader_node


class _StubAddr:
    def __init__(self, aid: str) -> None:
        self.aid = aid


# ---------------------------------------------------------------------------
# DynamicConnectorRole: cross-sector is_live semantics
# ---------------------------------------------------------------------------


class TestDynamicConnectorRoleIsLive:
    def test_default_admits_all(self) -> None:
        mirror, leaders = _cp_grid()
        role = DynamicConnectorRole(
            behavior=SimpleNamespace(),
            my_node_id=2,
            leader_aid_to_node_id=leaders,
            mirror=mirror,
        )
        for aid in leaders:
            assert role.is_live(_StubAddr(aid))

    def test_same_sector_branch_failure(self) -> None:
        """A failure on b1 (electricity) islands ``leader-el-a`` from
        the CP through electricity edges, and there is no other
        sector-bridge path back to node 1 — so the leader should be
        classified unreachable.
        """
        mirror, leaders = _cp_grid()
        role = DynamicConnectorRole(
            behavior=SimpleNamespace(),
            my_node_id=2,
            leader_aid_to_node_id=leaders,
            mirror=mirror,
        )
        mirror.mark_broken(("b1",))

        reach = mirror.reachable_from(2, sector=None, allow_cp_bridges=True)
        for aid, node in leaders.items():
            if node not in reach:
                role._unreachable_aids.add(aid)

        assert not role.is_live(_StubAddr("leader-el-a"))
        # leader-el-b stays reachable through b2.
        assert role.is_live(_StubAddr("leader-el-b"))
        # Gas side stays reachable through the CP bridge.
        assert role.is_live(_StubAddr("leader-gs-a"))
        assert role.is_live(_StubAddr("leader-gs-b"))

    def test_cp_bridge_failure_isolates_far_sector(self) -> None:
        """Breaking the CP bridge itself islands the entire gas side
        from the CP's perspective (no other CP edge in this fixture).
        """
        mirror, leaders = _cp_grid()
        role = DynamicConnectorRole(
            behavior=SimpleNamespace(),
            my_node_id=2,
            leader_aid_to_node_id=leaders,
            mirror=mirror,
        )
        mirror.mark_broken(("bridge",))

        reach = mirror.reachable_from(2, sector=None, allow_cp_bridges=True)
        for aid, node in leaders.items():
            if node not in reach:
                role._unreachable_aids.add(aid)

        # EL side still reachable through b1/b2.
        assert role.is_live(_StubAddr("leader-el-a"))
        assert role.is_live(_StubAddr("leader-el-b"))
        # Gas side now islanded.
        assert not role.is_live(_StubAddr("leader-gs-a"))
        assert not role.is_live(_StubAddr("leader-gs-b"))

    def test_cross_sector_reachability_uses_bridge(self) -> None:
        """Sanity: from the CP at node 2, the gas side is *only*
        reachable via the CP bridge.  Confirms L3 must use the
        cross-sector reachability flavour, not L2's sector-bounded one.
        """
        mirror, _ = _cp_grid()
        # Sector-bounded query would *not* see the gas side.
        el_only = mirror.reachable_from(2, sector=Sector.ELECTRICITY)
        assert 10 not in el_only and 12 not in el_only
        # Cross-sector with bridges *does*.
        cross = mirror.reachable_from(2, sector=None, allow_cp_bridges=True)
        assert {10, 12}.issubset(cross)


# ---------------------------------------------------------------------------
# EnergyConverterRole._live_connectors: filter wiring
# ---------------------------------------------------------------------------


class _DroppingFilter:
    def __init__(self, dropped: set[str]) -> None:
        self.dropped = set(dropped)

    def is_live(self, addr: Any) -> bool:
        aid = getattr(addr, "aid", None) or str(addr)
        return aid not in self.dropped


class _NullActor:
    """Stand-in for ADMMFlexActor — never invoked by the filter helper."""


class TestEnergyConverterLiveConnectors:
    def _build(self, filt) -> EnergyConverterRole:
        return EnergyConverterRole(
            behavior=SimpleNamespace(),
            flex_actor=_NullActor(),
            sectors=[Sector.ELECTRICITY, Sector.GAS],
            live_connector_filter=filt,
        )

    def test_no_filter_passthrough(self) -> None:
        role = self._build(filt=None)
        connectors = [_StubAddr("g-a"), _StubAddr("g-b")]
        assert role._live_connectors(connectors) == connectors

    def test_filter_drops_islanded(self) -> None:
        role = self._build(filt=_DroppingFilter({"g-a"}))
        connectors = [_StubAddr("g-a"), _StubAddr("g-b"), _StubAddr("g-c")]
        kept = role._live_connectors(connectors)
        assert [c.aid for c in kept] == ["g-b", "g-c"]

    def test_filter_drops_all(self) -> None:
        role = self._build(filt=_DroppingFilter({"g-a", "g-b"}))
        kept = role._live_connectors([_StubAddr("g-a"), _StubAddr("g-b")])
        assert kept == []
