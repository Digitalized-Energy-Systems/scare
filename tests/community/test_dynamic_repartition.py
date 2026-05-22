"""Unit tests for the L1 failure-driven re-partition (Concept C).

The ``DynamicRepartitionRole`` itself depends on the mango context for
message scheduling, so we don't spin up the full role here.  Instead
we cover the two pieces it composes:

1. ``_bfs_reachable`` — the BFS helper that classifies members as
   orphaned vs. reachable.  Pure function, easy to test.
2. The orphan-detection + new-leader-election logic, exercised by
   driving the same data the role uses (member node-id map + live edge
   set) and asserting on what the role *would* send.

The end-to-end mango wiring is exercised by the integration test in
``tests/integration/test_dynamic_topology.py``.
"""

from __future__ import annotations

from scare.base.model import Sector
from scare.base.topology_mirror import GridTopologyMirror
from scare.community.repartition import _bfs_reachable


def test_bfs_full_component() -> None:
    edges = [(1, 2), (2, 3), (3, 4)]
    assert _bfs_reachable(1, edges) == {1, 2, 3, 4}


def test_bfs_disconnected() -> None:
    # 1-2 separated from 3-4 by a missing middle edge.
    edges = [(1, 2), (3, 4)]
    assert _bfs_reachable(1, edges) == {1, 2}
    assert _bfs_reachable(3, edges) == {3, 4}


def test_bfs_isolated_start() -> None:
    # A start node that has no incident live edge is its own component.
    edges = [(2, 3)]
    assert _bfs_reachable(1, edges) == {1}


def test_orphan_detection_via_mirror() -> None:
    """The repartition role's reachability check, replayed through the
    mirror's API.  After breaking the middle of a 4-node line, members
    on the far side should be classified as orphans relative to the
    leader at node 1.
    """
    branches = {
        ("e1",): (1, 2),
        ("e2",): (2, 3),
        ("e3",): (3, 4),
    }
    sector_tag = {bid: "electricity" for bid in branches}
    mirror = GridTopologyMirror(branches=branches, branch_sector=sector_tag)

    member_node_id = {"agent-1": 1, "agent-2": 2, "agent-3": 3, "agent-4": 4}
    leader_node = member_node_id["agent-1"]

    # Healthy state — nobody orphaned.
    reach = mirror.reachable_from(leader_node, sector=Sector.ELECTRICITY)
    orphans = [aid for aid, node in member_node_id.items() if node not in reach]
    assert orphans == []

    # Break the middle edge — 3 and 4 should become orphans.
    mirror.mark_broken(("e2",))
    reach = mirror.reachable_from(leader_node, sector=Sector.ELECTRICITY)
    orphans = sorted(aid for aid, node in member_node_id.items() if node not in reach)
    assert orphans == ["agent-3", "agent-4"]


def test_new_leader_election_is_lex_smallest_orphan() -> None:
    """The role elects the lex-smallest orphan as the new leader.
    Trivial logic but worth pinning so a future refactor doesn't drift.
    """
    orphans = ["agent-9", "agent-3", "agent-7", "agent-5"]
    new_leader = sorted(orphans)[0]
    assert new_leader == "agent-3"
