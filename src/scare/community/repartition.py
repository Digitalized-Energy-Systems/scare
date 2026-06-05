"""Failure-driven dynamic re-partitioning of Level-1 communities.

The L1 partition (``communities_from_topology`` in scare.base.community)
is computed once at build time. When a branch failure disconnects
members from their leader, those orphans can no longer reach it through
the surviving sector subgraph. This module re-partitions: the leader
detects unreachable members, elects a new leader among them, and pushes
a fresh ``CommunityAssignment`` via ``RepartitionAssignment``.

Design notes:

- Triggered by the global ``BranchFailureEvent`` (wired by the scenario
  builder) — no comm topology required.
- Each leader keeps a branch-level view of its sector subgraph
  (precomputed at build) and runs a BFS over live branches per failure
  to find which members remain reachable.
- Election: lex-smallest aid in the orphan set, no exchange round (same
  determinism as the static label-propagation seed selection).
- Reachability uses physical (sector) topology only, not the message
  bus — islanding is about energy flow, not message delivery.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from mango import Role
from mango.express.topology import topology_characteristic, topology_neighbors

from scare.base.diagnostics import record_event
from scare.base.model import (
    CommunityAssignment,
    CommunityReassignedEvent,
    LeaderEmerged,
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

    On ``BranchFailureEvent`` (wired globally by the scenario builder)
    tracks live branches in its sector, then after a debounce window
    BFS-checks which members became physically unreachable. Orphans are
    dropped from the leader's ``CommunityAssignment.neighbors`` and told
    their new membership via ``RepartitionAssignment``.

    Does not mutate the mango topology at runtime (no public API from a
    role context); post-repartition membership lives only on
    ``CommunityAssignment``, which repartition-aware downstream roles
    read via ``context.get_or_create_model(CommunityAssignment)``.
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
        # Edge view ``{branch_id: (node_a, node_b)}``; broken branches
        # move out of ``_live_branches`` when a failure is observed.
        self._live_branches: dict[tuple, tuple[Any, Any]] = dict(sector_branches)
        self._broken_branches: set[tuple] = set()
        self._debounce_s = debounce_s
        # Members already moved to an orphan partition: don't re-orphan
        # them later (this leader is no longer authoritative for them).
        self._already_orphaned: set[str] = set()
        self._reassess_pending: bool = False

    def setup(self) -> None:
        # No subscribe_message: the global ``BranchFailureEvent`` wired by
        # the scenario builder dispatches to ``on_branch_failure`` below.
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Global-event callback: record the failure, debounce, then
        schedule ``_reassess_membership``. A burst of failures collapses
        into one reassess.
        """
        if branch_id in self._broken_branches:
            return
        self._broken_branches.add(branch_id)
        if branch_id in self._live_branches:
            del self._live_branches[branch_id]
        if self._reassess_pending:
            return
        self._reassess_pending = True
        # Debounce on the simulation clock so simultaneous failures
        # collapse into one reassess pass.
        try:
            self.context.schedule_timestamp_task(
                self._reassess_membership(),
                timestamp=self.context.current_timestamp + self._debounce_s,
            )
        except Exception:
            # Fallback for early invocations before the scheduler is online.
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

        # Orphan sub-community: lex-smallest aid leads.
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

        # Notify each orphan, concurrently for prompt convergence.
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

        # Drop orphans from the surviving community's neighbour list.
        own = self.context.get_or_create_model(CommunityAssignment)
        own.neighbors = [
            addr for aid, addr in self._member_addr.items()
            if aid not in self._already_orphaned and aid not in orphaned_aids
        ]
        self.context.update(own)
        self._already_orphaned.update(orphaned_aids)

        # Broadcast ``LeaderEmerged`` on the ``holon_summary_<sector>`` mesh
        # so every L2 role adds the promoted orphan leader to its
        # ``_leader_node_ids`` map. Without it, the component coordinator
        # drops the new leader's ``ComponentAdmmReport`` (unknown-sender
        # filter) and the ``SlackBudgetMonitor`` override has no escalation
        # path. Sent from THIS original-leader role because it is already on
        # the mesh; the orphan leader was a member at build time and cannot
        # be added dynamically (no mango API; see module header).
        new_leader_node_id = self._member_node_id.get(new_leader_aid)
        if new_leader_node_id is not None:
            announcement = LeaderEmerged(
                leader_aid=new_leader_aid,
                leader_addr=new_leader_addr,
                node_id=new_leader_node_id,
                sector=self.sector,
            )
            try:
                summary_peers = list(
                    topology_neighbors(
                        self, tid=f"holon_summary_{self.sector.value}",
                    )
                )
            except Exception:
                summary_peers = []
            for peer_addr in summary_peers:
                try:
                    await self.context.send_message(
                        announcement, receiver_addr=peer_addr,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "[%s] leader-emerged broadcast to %s failed: %s",
                        self.context.aid, peer_addr, exc,
                    )


# ---------------------------------------------------------------------------
# Member-side: accept a repartition assignment
# ---------------------------------------------------------------------------


class RepartitionHandlerRole(Role):
    """Installed on every community-member agent. On a
    ``RepartitionAssignment`` (failure-driven re-election) updates the
    local ``CommunityAssignment`` and emits ``CommunityReassignedEvent``
    for co-located roles.
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
        # Neighbours are the other orphans, excluding self.
        my_addr = self.context.addr
        assignment.neighbors = [
            addr for addr in message.orphan_addrs if addr != my_addr
        ]
        assignment.leader_addr = message.new_leader_addr
        self.context.update(assignment)

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
            # No local subscribers — downstream reads the model directly.
            pass

        # Repoint any co-located ``SlackBudgetMonitor.home_leader_addr``:
        # it caches the original leader (wired at build time), which goes
        # stale on reassignment, so its override
        # ``StartBalanceNegotiation`` would land at the wrong leader and
        # silently drop if that leader is isolated. Duck-typed on class
        # name to avoid a community→service import dependency.
        for role in getattr(self.context, "roles", []):
            if type(role).__name__ == "SlackBudgetMonitor":
                role.home_leader_addr = message.new_leader_addr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bfs_reachable(
    start: Any, live_edges: Iterable[tuple[Any, Any]]
) -> set[Any]:
    """Return nodes reachable from ``start`` over ``live_edges`` (an
    iterable of unordered, undirected node-pair tuples).

    No radius bound — the membership check needs the full connected
    component.
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
