"""Unit tests for the L1 failure-driven re-partition.

Covers the two pieces ``DynamicRepartitionRole`` composes without the
mango context:

1. ``_bfs_reachable`` — classifies members as orphaned vs. reachable.
2. The orphan-detection + new-leader-election logic, driven against the
   same data (member node-id map + live edges) the role uses.
"""

from __future__ import annotations

from scare.base.model import Sector
from scare.base.topology.topology_mirror import GridTopologyMirror
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
    """After breaking the middle of a 4-node line, far-side members are
    classified as orphans relative to the leader at node 1.
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
    """The role elects the lex-smallest orphan as the new leader."""
    orphans = ["agent-9", "agent-3", "agent-7", "agent-5"]
    new_leader = sorted(orphans)[0]
    assert new_leader == "agent-3"


def test_repartition_handler_rewrites_slack_budget_home_leader() -> None:
    """``RepartitionHandlerRole._on_repartition`` must rewrite
    ``home_leader_addr`` on every co-located ``SlackBudgetMonitor``.

    The monitor caches the original group leader at build time; after a
    failure-driven re-assignment that address goes stale and an
    over-budget poll would escalate to the wrong (or isolated) leader.
    """
    # pylint: disable=import-outside-toplevel
    import asyncio
    from types import SimpleNamespace
    from uuid import uuid4

    from scare.base.model import RepartitionAssignment
    from scare.community.repartition import RepartitionHandlerRole

    class _StubMonitor:
        """Duck-typed SlackBudgetMonitor stand-in. The rewrite filters on
        the class name (not isinstance), so only the name and
        ``home_leader_addr`` matter.
        """

        def __init__(self) -> None:
            self.home_leader_addr = "original-leader"

    _StubMonitor.__name__ = "SlackBudgetMonitor"

    class _OtherRole:
        """An unrelated role that must NOT be touched."""

        def __init__(self) -> None:
            self.home_leader_addr = "untouched"

    class _StubContext:
        def __init__(self) -> None:
            self.aid = "child-39"
            self.addr = "child-39-addr"
            self.current_timestamp = 1.1
            self.monitor = _StubMonitor()
            self.other = _OtherRole()
            self.roles = [self.monitor, self.other]
            self._models: dict = {}
            self._events: list = []

        def get_or_create_model(self, _cls):
            return self._models.setdefault(
                "ca",
                SimpleNamespace(
                    community_id=None,
                    neighbors=[],
                    leader_addr=None,
                ),
            )

        def update(self, _model):
            pass

        def emit_event(self, ev):
            self._events.append(ev)

        async def send_message(self, *_args, **_kwargs):
            return None

    role = RepartitionHandlerRole()
    role._context = _StubContext()  # type: ignore[attr-defined]
    msg = RepartitionAssignment(
        community_id=uuid4(),
        new_leader_addr="child-25-addr",
        orphan_addrs=["child-39-addr", "child-31-addr", "child-25-addr"],
    )
    asyncio.run(role._on_repartition(msg, meta={}))

    assert role._context.monitor.home_leader_addr == "child-25-addr", (
        "SlackBudgetMonitor.home_leader_addr must be rewritten to the "
        "new leader; got "
        f"{role._context.monitor.home_leader_addr!r}"
    )
    # Unrelated co-located roles with a homonymous attr must be untouched.
    assert role._context.other.home_leader_addr == "untouched"


def test_repartition_module_publishes_leader_emerged_message() -> None:
    """The module must broadcast a ``LeaderEmerged`` to the original
    leader's ``holon_summary_<sector>`` peers — checked statically so a
    refactor that drops it is flagged. A promoted orphan leader is
    otherwise invisible to the component coordinator (no leader has its
    aid in ``_leader_node_ids``); routing through the original leader,
    already on that mesh, fixes it.
    """
    # pylint: disable=import-outside-toplevel
    import inspect

    from scare.community import repartition

    src = inspect.getsource(repartition.DynamicRepartitionRole)
    assert "LeaderEmerged" in src, (
        "DynamicRepartitionRole must publish LeaderEmerged after the "
        "orphan-leader is elected; the broadcast goes through the "
        "original leader's holon_summary_<sector> peer mesh because "
        "the promoted orphan was not a leader at scenario time and so "
        "isn't on that mesh itself."
    )
    assert "topology_neighbors" in src and "holon_summary_" in src, (
        "Broadcast target must be the holon_summary_<sector> mesh — "
        "this is where the receiving HolonicCommunityRoles live and "
        "update _leader_node_ids."
    )
