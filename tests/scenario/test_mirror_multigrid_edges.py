"""A branch leaving a multi-grid node must still be traversable.

``sector_from_grid`` returns None for a multi-grid node — its docstring says so
explicitly, "must be resolved by context" — and ``_branch_sector_str`` reads the
from-node alone, so every branch leaving one goes untagged. ``mirror_from_monee``
drops untagged branches (``if not tag: continue``), which deletes them from the
reachability graph entirely.

On ``simbench_lv_el_dependent`` that is 26 ``GenericTransferBranch`` stubs — one
per ``CHPHGControlNode``, carrying its electrical output into its bus. Without
them each CHP sits in the gas component alone (sector-scoped electricity reach:
1 node, itself), the 47-CP fleet splits into a 21-CP electricity+heat round and a
26-CP **gas-only** round, and a single-sector CP round has no coupling to
optimise — so ``minimize_usage`` pins all 26 CHPs at r=0 under every kernel
realisation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scare.base.topology.topology_mirror import mirror_from_monee
from scare.scenario.restoration import _branch_sector_str, _mirror_sector_str


class _Node:
    def __init__(self, nid, grid):
        self.id = nid
        self.grid = grid


class _Model:
    def __init__(self, cp=False):
        self._cp = cp

    def is_cp(self):
        return self._cp


class _Branch:
    def __init__(self, a, b, cp=False):
        self.id = (a, b, 0)
        self.model = _Model(cp)


_POWER = SimpleNamespace(name="power")
_GAS = SimpleNamespace(name="gas")
_WATER = SimpleNamespace(name="water")


def _net(nodes, branches):
    by_id = {n.id: n for n in nodes}
    return SimpleNamespace(
        nodes=nodes, branches=branches, node_by_id=lambda i: by_id[i]
    )


def _chp_net():
    """A CHP control node ([power, gas]) with a gas pipe and a transfer stub to
    a power bus, plus a second power bus one line further out."""
    nodes = [
        _Node("chp", [_POWER, _GAS]),  # multi-grid: sector_from_grid -> None
        _Node("bus", _POWER),
        _Node("bus2", _POWER),
        _Node("gasjunction", _GAS),
    ]
    branches = [
        _Branch("chp", "bus"),  # the transfer stub: from-node is multi-grid
        # Pipe INTO the CHP: from-node is the junction, so it tags "gas" either
        # way. Mirrors the measured grid, where a CHP control node's incident
        # branches are exactly one gas-tagged pipe and one untagged stub.
        _Branch("gasjunction", "chp"),
        _Branch("bus", "bus2"),  # ordinary power line
    ]
    return _net(nodes, branches)


def test_from_node_only_tagging_drops_the_multigrid_branch():
    """The defect, at the tag level."""
    net = _chp_net()
    stub = net.branches[0]
    assert _branch_sector_str(stub, net) == ""  # dropped by mirror_from_monee
    assert _mirror_sector_str(stub, net) == "electricity"  # resolved by context


def test_cp_branches_still_tag_as_bridges():
    """A real CP branch keeps its ``cp`` tag — the fallback must not shadow it."""
    net = _net([_Node("a", _POWER), _Node("b", _WATER)], [_Branch("a", "b", cp=True)])
    assert _mirror_sector_str(net.branches[0], net) == "cp"


def test_a_branch_with_no_resolvable_endpoint_stays_untagged():
    net = _net([_Node("a", None), _Node("b", None)], [_Branch("a", "b")])
    assert _mirror_sector_str(net.branches[0], net) == ""


@pytest.mark.parametrize(
    ("resolver", "expect_reach"),
    [(_branch_sector_str, False), (_mirror_sector_str, True)],
)
def test_the_chp_node_reaches_the_power_grid_only_with_the_fallback(
    resolver, expect_reach
):
    """The consequence: whether the CHP is electrically connected at all."""
    net = _chp_net()
    mirror = mirror_from_monee(net, branch_sector_resolver=lambda b: resolver(b, net))
    reach = mirror.reachable_from("chp", sector=None, allow_cp_bridges=True)
    assert ("bus2" in reach) is expect_reach
    # Without the fallback the CHP's only live edge is its gas pipe.
    assert ("gasjunction" in reach) is True


def test_the_fallback_does_not_relabel_ordinary_lines():
    """Every branch whose from-node resolves keeps exactly its old tag, so the
    other consumers of the sector tag see no change."""
    net = _chp_net()
    for branch in net.branches:
        old = _branch_sector_str(branch, net)
        if old:
            assert _mirror_sector_str(branch, net) == old
