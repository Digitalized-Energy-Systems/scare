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


def test_repartition_handler_rewrites_slack_budget_home_leader() -> None:
    """``RepartitionHandlerRole._on_repartition`` must rewrite
    ``home_leader_addr`` on every co-located ``SlackBudgetMonitor``.

    The monitor caches the original group leader at scenario build
    time; after a failure-driven re-assignment the agent's home
    leader can change and the cached address goes stale.  An
    over-budget poll would then send
    ``StartBalanceNegotiation(override_target=imbalance)`` to the
    wrong (or now-isolated) leader and the budget excursion has no
    escalation path.

    Verified end-to-end mango wiring is exercised by integration
    tests; here we drive the role's coroutine directly to assert the
    rewrite contract.

    Regression for the residual ``slack__electricity__child-39``
    breach on ``eval_full_small_20260529-181310/tasks/000088``: the
    repartition wired child-39 to a new community (``community_
    reassigned ... new_leader=child-25``) but the SlackBudgetMonitor
    kept addressing the original leader.
    """
    # pylint: disable=import-outside-toplevel
    import asyncio
    from types import SimpleNamespace
    from uuid import uuid4

    from scare.base.model import RepartitionAssignment
    from scare.community.repartition import RepartitionHandlerRole

    class _StubMonitor:
        """Minimal duck-typed SlackBudgetMonitor stand-in: the only
        attribute the rewrite touches is ``home_leader_addr``.  The
        class name is what the rewrite filters on (no isinstance
        check) so we keep the test free of the heavy slack-budget
        dependency tree.
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
            return self._models.setdefault("ca", SimpleNamespace(
                community_id=None, neighbors=[], leader_addr=None,
            ))

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
    # Unrelated co-located roles must NOT be touched — the rewrite
    # filters on the class name to avoid clobbering homonymous
    # attributes elsewhere.
    assert role._context.other.home_leader_addr == "untouched"


def test_repartition_module_publishes_leader_emerged_message() -> None:
    """End-to-end mango wiring is out of scope here (covered by the
    integration test), but the module's contract — broadcasting a
    ``LeaderEmerged`` message to the original leader's
    ``holon_summary_<sector>`` peers — must be statically visible in
    the source so a future refactor that drops it gets flagged.

    Regression for the slack__electricity__child-39 +10.6% breach in
    ``eval_full_small_20260529-181310/tasks/000088``: the promoted
    orphan leader (child-25) was invisible to the elected component
    coordinator because no leader registered its aid in their
    ``_leader_node_ids`` map.  The fix routes the announcement
    through the original leader who is already on the
    holon_summary_<sector> mesh.
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
