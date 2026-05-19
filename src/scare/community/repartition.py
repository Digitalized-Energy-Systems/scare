"""Failure-driven dynamic re-partitioning of Level-1 communities.

The Level-1 partition (``communities_from_topology`` label propagation
in [scare.base.community]) is computed once at scenario build time.
When a branch fails and disconnects part of a community from its
leader, those orphaned members continue to identify with their
original community even though they can no longer reach it through
the surviving sector subgraph.  This module re-partitions on failure:
the leader detects which members are now unreachable, elects a new
leader among them, and pushes a fresh ``CommunityAssignment`` to each
orphan via a ``RepartitionAssignment`` message.

Design notes:

- The trigger is the global ``BranchFailureEvent`` (wired by
  ``_add_system_behaviors`` in the scenario builder) — no comm
  topology required, the event delivers the failed branch_id to
  every interested role.
- Each leader maintains a local view of the sector subgraph at
  the *branch* level (precomputed at scenario build), and on each
  failure runs a small BFS to find which of its members remain
  reachable through live branches.
- The election is the simplest possible: lex-smallest aid in the
  orphan set is the new leader.  No exchange round.  Same
  determinism as the static label-propagation seed selection.
- Reachability uses physical (sector) topology *only*, ignoring
  the message bus.  This is the right model for islanding —
  reachability is about energy flow, not message delivery.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from mango import Role
from mango.express.topology import topology_characteristic

from scare.base.model import (
    CommunityAssignment,
    CommunityReassignedEvent,
    RepartitionAssignment,
    Sector,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leader-side: detect orphans, elect new leader, push reassignments
# ---------------------------------------------------------------------------


class DynamicRepartitionRole(Role):
    """Installed on each group leader (in the ``groups`` topology).

    Watches for ``BranchFailureEvent`` (wired globally by the
    scenario builder), maintains a local view of which branches in
    its sector are live, and after a short debounce window runs a
    BFS to identify members that have become physically unreachable.
    Orphans are removed from the leader's own
    ``CommunityAssignment.neighbors`` and informed of their new
    membership via ``RepartitionAssignment``.

    The role intentionally does *not* mutate the mango topology
    (``groups`` / ``sector_grid_*``) at runtime — mango exposes no
    public API for that from a role context.  The post-repartition
    membership lives on ``CommunityAssignment``; downstream roles
    that need to be repartition-aware should read it via
    ``context.get_or_create_model(CommunityAssignment)``.
    """

    def __init__(
        self,
        behavior: "RestorationEnvironmentBehavior",
        sector: Sector,
        my_node_id: Any,
        member_aid_to_node_id: dict[str, Any],
        member_aid_to_addr: dict[str, Any],
        sector_branches: dict[tuple, tuple[Any, Any]],
        *,
        debounce_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self._my_node_id = my_node_id
        self._member_node_id = dict(member_aid_to_node_id)
        self._member_addr = dict(member_aid_to_addr)
        # Edge view: ``{branch_id: (node_a, node_b)}`` for every
        # branch in this leader's sector.  Live edges live in
        # ``_live_branches``; broken ones get removed when a failure
        # is observed.
        self._live_branches: dict[tuple, tuple[Any, Any]] = dict(sector_branches)
        self._broken_branches: set[tuple] = set()
        self._debounce_s = debounce_s
        # Members already moved off into an orphan partition — don't
        # try to re-orphan them on a later failure (their original
        # leader is no longer authoritative for them).
        self._already_orphaned: set[str] = set()
        self._reassess_pending: bool = False

    def setup(self) -> None:
        # No subscribe_message: the trigger is the global
        # ``BranchFailureEvent`` registered by the scenario builder
        # via ``behavior_in(world, ..., on_global_event=BranchFailureEvent)``.
        # That dispatch lands on ``on_branch_failure`` below.
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Global-event callback wired by the scenario builder.

        Records the failure, debounces, then schedules
        ``_reassess_membership``.  Multiple failures arriving in
        close succession collapse into one reassess.
        """
        if branch_id in self._broken_branches:
            return
        self._broken_branches.add(branch_id)
        if branch_id in self._live_branches:
            del self._live_branches[branch_id]
        if self._reassess_pending:
            return
        self._reassess_pending = True
        # ``schedule_timestamp_task`` aligns with the simulation
        # clock; the debounce window collapses a burst of
        # simultaneous failures into one reassess pass.
        try:
            self.context.schedule_timestamp_task(
                self._reassess_membership(),
                timestamp=self.context.current_timestamp + self._debounce_s,
            )
        except Exception:
            # Defensive fallback for early invocations before the
            # scheduler is fully online — just schedule instantly.
            self.context.schedule_instant_task(self._reassess_membership())

    async def _reassess_membership(self) -> None:
        self._reassess_pending = False
        if topology_characteristic(self, tid="groups") != "leader":
            return

        live_members = {
            aid: node_id for aid, node_id in self._member_node_id.items()
            if aid not in self._already_orphaned
        }
        if not live_members:
            return

        reachable_nodes = _bfs_reachable(
            self._my_node_id, self._live_branches.values()
        )
        orphaned_aids: list[str] = []
        for aid, node_id in live_members.items():
            if node_id not in reachable_nodes:
                orphaned_aids.append(aid)

        if not orphaned_aids:
            logger.debug(
                "[%s] repartition reassess: %d members all reachable, no split",
                self.context.aid, len(live_members),
            )
            return

        # Build orphan sub-community: lex-smallest aid leads.
        orphaned_aids.sort()
        new_leader_aid = orphaned_aids[0]
        new_leader_addr = self._member_addr.get(new_leader_aid)
        if new_leader_addr is None:
            logger.warning(
                "[%s] repartition: orphan leader %s has no recorded addr, abort",
                self.context.aid, new_leader_aid,
            )
            return
        new_community_id = uuid4()
        orphan_addrs = [self._member_addr[aid] for aid in orphaned_aids]

        from scare.base.diagnostics import record_event

        record_event(
            t=self.context.current_timestamp,
            kind="community_repartitioned",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=(
                f"orphans={len(orphaned_aids)} new_leader={new_leader_aid} "
                f"remaining={len(live_members) - len(orphaned_aids)}"
            ),
        )
        logger.info(
            "[%s] repartition: %d/%d members orphaned (new_leader=%s, sector=%s)",
            self.context.aid,
            len(orphaned_aids),
            len(live_members),
            new_leader_aid,
            self.sector.value,
        )

        # Notify each orphan.  Send concurrently for prompt convergence;
        # delivery still goes through mango's per-message scheduler.
        assignment = RepartitionAssignment(
            community_id=new_community_id,
            new_leader_addr=new_leader_addr,
            orphan_addrs=orphan_addrs,
        )
        await asyncio.gather(
            *[
                self.context.send_message(assignment, receiver_addr=addr)
                for addr in orphan_addrs
            ]
        )

        # Update our own assignment — drop orphans from the surviving
        # community's neighbour list.
        own = self.context.get_or_create_model(CommunityAssignment)
        own.neighbors = [
            addr for aid, addr in self._member_addr.items()
            if aid not in self._already_orphaned and aid not in orphaned_aids
        ]
        self.context.update(own)
        self._already_orphaned.update(orphaned_aids)


# ---------------------------------------------------------------------------
# Member-side: accept a repartition assignment
# ---------------------------------------------------------------------------


class RepartitionHandlerRole(Role):
    """Installed on every community-member agent.  Receives a
    ``RepartitionAssignment`` after a failure-driven re-election,
    updates the local ``CommunityAssignment`` model, and emits a
    ``CommunityReassignedEvent`` so co-located roles can react.
    """

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        self.context.subscribe_message(
            self,
            _wrap(self._on_repartition),
            lambda msg, meta: isinstance(msg, RepartitionAssignment),
        )

    async def _on_repartition(
        self, message: RepartitionAssignment, meta: dict
    ) -> None:
        assignment = self.context.get_or_create_model(CommunityAssignment)
        assignment.community_id = message.community_id
        # Neighbours are the other orphans, NOT including self.
        my_addr = self.context.addr
        assignment.neighbors = [
            addr for addr in message.orphan_addrs if addr != my_addr
        ]
        assignment.leader_addr = message.new_leader_addr
        self.context.update(assignment)

        from scare.base.diagnostics import record_event

        record_event(
            t=self.context.current_timestamp,
            kind="community_reassigned",
            aid=self.context.aid,
            sector="",
            detail=(
                f"new_leader={message.new_leader_addr} "
                f"n_neighbors={len(assignment.neighbors)}"
            ),
        )
        logger.info(
            "[%s] reassigned to new community: leader=%s neighbors=%d",
            self.context.aid,
            message.new_leader_addr,
            len(assignment.neighbors),
        )

        try:
            self.context.emit_event(
                CommunityReassignedEvent(
                    new_leader_addr=message.new_leader_addr,
                    n_neighbors=len(assignment.neighbors),
                )
            )
        except KeyError:
            # No local subscribers — fine, downstream code can read
            # the model directly.
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bfs_reachable(
    start: Any, live_edges: Iterable[tuple[Any, Any]]
) -> set[Any]:
    """Return the set of nodes reachable from ``start`` traversing only
    ``live_edges`` (an iterable of unordered node-pair tuples).

    Edges are undirected.  No radius bound — the surviving subgraph
    around the leader is small after a failure burst and we want the
    full connected component for the membership check.
    """
    adj: dict[Any, list[Any]] = {}
    for a, b in live_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    seen: set[Any] = {start}
    frontier = [start]
    while frontier:
        nxt: list[Any] = []
        for node in frontier:
            for neigh in adj.get(node, ()):
                if neigh in seen:
                    continue
                seen.add(neigh)
                nxt.append(neigh)
        frontier = nxt
    return seen
