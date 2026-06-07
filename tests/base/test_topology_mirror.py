"""Unit tests for :mod:`scare.base.topology.topology_mirror`.

The mirror is a pure data structure — no mango, no async — so these
tests build a small synthetic grid by hand and exercise the BFS
reachability paths directly.

Grid layout (numbers are node ids, letters are branch ids):

::

    sector EL:   1 --a-- 2 --b-- 3 --c-- 4
                              \\
                               d (CP, EL <-> GAS)
                                \\
    sector GAS:                  10 --e-- 11

The CP bridge ``d`` connects el-node 3 to gas-node 10.  ``a``/``b``/``c``
are electricity branches, ``e`` is a gas branch.
"""

from __future__ import annotations

import pytest

from scare.base.model import Sector
from scare.base.topology.topology_mirror import GridTopologyMirror


def _grid() -> GridTopologyMirror:
    branches = {
        ("a",): (1, 2),
        ("b",): (2, 3),
        ("c",): (3, 4),
        ("d",): (3, 10),
        ("e",): (10, 11),
    }
    branch_sector = {
        ("a",): "electricity",
        ("b",): "electricity",
        ("c",): "electricity",
        ("d",): "cp",
        ("e",): "gas",
    }
    return GridTopologyMirror(branches=branches, branch_sector=branch_sector)


class TestReachability:
    def test_all_live_same_sector(self) -> None:
        m = _grid()
        # All EL nodes reachable from node 1 through electricity edges.
        assert m.reachable_from(1, sector=Sector.ELECTRICITY) == {1, 2, 3, 4}

    def test_cp_bridges_blocked_when_sector_bounded(self) -> None:
        m = _grid()
        # Sector-bounded query must not cross the CP edge into gas.
        reach = m.reachable_from(1, sector=Sector.ELECTRICITY)
        assert 10 not in reach
        assert 11 not in reach

    def test_cross_sector_via_cp_bridge(self) -> None:
        m = _grid()
        # Cross-sector query with CP bridges admitted reaches the gas side.
        reach = m.reachable_from(1, sector=None, allow_cp_bridges=True)
        assert {1, 2, 3, 4, 10, 11}.issubset(reach)

    def test_cross_sector_without_cp_isolates_sectors(self) -> None:
        m = _grid()
        # No CP bridge -> EL and GAS are disjoint islands.
        reach = m.reachable_from(1, sector=None, allow_cp_bridges=False)
        assert reach == {1, 2, 3, 4}
        assert m.reachable_from(10, sector=None, allow_cp_bridges=False) == {10, 11}

    def test_isolated_node_returns_self(self) -> None:
        m = _grid()
        # A node not incident to any live edge in the requested sector
        # is its own connected component.
        assert m.reachable_from(99, sector=Sector.ELECTRICITY) == {99}


class TestBranchFailure:
    def test_break_splits_sector(self) -> None:
        m = _grid()
        m.mark_broken(("b",))
        # 1,2 on one side; 3,4 on the other.
        assert m.reachable_from(1, sector=Sector.ELECTRICITY) == {1, 2}
        assert m.reachable_from(4, sector=Sector.ELECTRICITY) == {3, 4}

    def test_break_idempotent(self) -> None:
        m = _grid()
        m.mark_broken(("b",))
        m.mark_broken(("b",))
        assert m.is_broken(("b",))
        assert m.reachable_from(1, sector=Sector.ELECTRICITY) == {1, 2}

    def test_restore_reconnects(self) -> None:
        m = _grid()
        m.mark_broken(("b",))
        m.mark_restored(("b",))
        assert not m.is_broken(("b",))
        assert m.reachable_from(1, sector=Sector.ELECTRICITY) == {1, 2, 3, 4}

    def test_cp_bridge_failure_islanding(self) -> None:
        m = _grid()
        m.mark_broken(("d",))
        # CP bridge down: gas side no longer reachable from EL side
        # even with CP traversal admitted, because the bridge edge
        # itself is gone.
        reach = m.reachable_from(1, sector=None, allow_cp_bridges=True)
        assert reach == {1, 2, 3, 4}

    def test_is_reachable_short_circuits_self(self) -> None:
        m = _grid()
        m.mark_broken(("a",))
        assert m.is_reachable(1, 1, sector=Sector.ELECTRICITY)

    def test_unknown_branch_mark_is_noop(self) -> None:
        m = _grid()
        # A branch the mirror doesn't know about should not corrupt state.
        m.mark_broken(("zzz",))
        assert m.reachable_from(1, sector=Sector.ELECTRICITY) == {1, 2, 3, 4}


class TestApiContract:
    def test_sector_and_cp_combination_rejected(self) -> None:
        m = _grid()
        with pytest.raises(ValueError):
            m.reachable_from(1, sector=Sector.ELECTRICITY, allow_cp_bridges=True)

    def test_live_branches_filtering(self) -> None:
        m = _grid()
        # Same-sector only.
        el = set(m.live_branches(sector=Sector.ELECTRICITY))
        assert el == {("a",), ("b",), ("c",)}
        # Cross-sector with CP.
        cross = set(m.live_branches(sector=None, include_cp=True))
        assert ("d",) in cross
        # Cross-sector without CP excludes the bridge.
        cross_no_cp = set(m.live_branches(sector=None, include_cp=False))
        assert ("d",) not in cross_no_cp
