"""Failure-driven dynamic re-partitioning of Level-1 communities.

The L1 partition is built once. When a branch failure disconnects members
from their leader, this module re-partitions: the leader detects unreachable
members (BFS over live branches), elects a new leader (lex-smallest orphan
aid), and pushes a fresh ``CommunityAssignment`` via ``RepartitionAssignment``.
Reachability uses physical sector topology, not the message bus.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mango import Role
from mango.express.topology import topology_characteristic, topology_neighbors

from scare.base.model import (
    CommunityAssignment,
    CommunityReassignedEvent,
    LeaderEmerged,
    RepartitionAssignment,
    Sector,
)
from scare.base.runtime.diagnostics import record_event

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leader-side: detect orphans, elect new leader, push reassignments
# ---------------------------------------------------------------------------


class DynamicRepartitionRole(Role):
    """Installed on each group leader (``groups`` topology).

    On ``BranchFailureEvent`` tracks live branches, then after a debounce
    BFS-checks which members became unreachable. Orphans are dropped from
    ``CommunityAssignment.neighbors`` and told their new membership via
    ``RepartitionAssignment``. The mango topology is not mutated at runtime;
    post-repartition membership lives only on ``CommunityAssignment``.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
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
        # Edge view ``{branch_id: (node_a, node_b)}``; failures move branches
        # out of ``_live_branches``.
        self._live_branches: dict[tuple, tuple[Any, Any]] = dict(sector_branches)
        self._broken_branches: set[tuple] = set()
        self._debounce_s = debounce_s
        # Members already orphaned: don't re-orphan (no longer authoritative).
        self._already_orphaned: set[str] = set()
        self._reassess_pending: bool = False

    def setup(self) -> None:
        # The global ``BranchFailureEvent`` dispatches to on_branch_failure.
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Record failure, debounce, then schedule ``_reassess_membership``."""
        if branch_id in self._broken_branches:
            return
        self._broken_branches.add(branch_id)
        if branch_id in self._live_branches:
            del self._live_branches[branch_id]
        if self._reassess_pending:
            return
        self._reassess_pending = True
        # Debounce on the sim clock so simultaneous failures collapse into one.
        try:
            self.context.schedule_timestamp_task(
                self._reassess_membership(),
                timestamp=self.context.current_timestamp + self._debounce_s,
            )
        except Exception:
            # Fallback before the scheduler is online.
            self.context.schedule_instant_task(self._reassess_membership())

    async def _reassess_membership(self) -> None:
        self._reassess_pending = False
        if topology_characteristic(self, tid="groups") != "leader":
            return

        live_members = {
            aid: node_id
            for aid, node_id in self._member_node_id.items()
            if aid not in self._already_orphaned
        }
        if not live_members:
            return

        reachable_nodes = _bfs_reachable(self._my_node_id, self._live_branches.values())
        orphaned_aids: list[str] = []
        for aid, node_id in live_members.items():
            if node_id not in reachable_nodes:
                orphaned_aids.append(aid)

        if not orphaned_aids:
            logger.debug(
                "[%s] repartition reassess: %d members all reachable, no split",
                self.context.aid,
                len(live_members),
            )
            return

        # Orphan sub-community: lex-smallest aid leads.
        orphaned_aids.sort()
        new_leader_aid = orphaned_aids[0]
        new_leader_addr = self._member_addr.get(new_leader_aid)
        if new_leader_addr is None:
            logger.warning(
                "[%s] repartition: orphan leader %s has no recorded addr, abort",
                self.context.aid,
                new_leader_aid,
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
            addr
            for aid, addr in self._member_addr.items()
            if aid not in self._already_orphaned and aid not in orphaned_aids
        ]
        self.context.update(own)
        self._already_orphaned.update(orphaned_aids)

        # Broadcast ``LeaderEmerged`` on the ``holon_summary_<sector>`` mesh so
        # L2 roles register the promoted orphan leader; otherwise its
        # ``ComponentAdmmReport`` is dropped and SlackBudgetMonitor can't
        # escalate. Sent from this role since it is already on the mesh.
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
                        self,
                        tid=f"holon_summary_{self.sector.value}",
                    )
                )
            except Exception:
                summary_peers = []
            for peer_addr in summary_peers:
                try:
                    await self.context.send_message(
                        announcement,
                        receiver_addr=peer_addr,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "[%s] leader-emerged broadcast to %s failed: %s",
                        self.context.aid,
                        peer_addr,
                        exc,
                    )


# ---------------------------------------------------------------------------
# Member-side: accept a repartition assignment
# ---------------------------------------------------------------------------


class RepartitionHandlerRole(Role):
    """Installed on every community member.

    On a ``RepartitionAssignment`` updates the local ``CommunityAssignment``
    and emits ``CommunityReassignedEvent`` for co-located roles.
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

    async def _on_repartition(self, message: RepartitionAssignment, meta: dict) -> None:
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

        # Repoint co-located ``SlackBudgetMonitor.home_leader_addr``: its cached
        # original leader goes stale on reassignment. Duck-typed on class name
        # to avoid a community→service import dependency.
        for role in getattr(self.context, "roles", []):
            if type(role).__name__ == "SlackBudgetMonitor":
                role.home_leader_addr = message.new_leader_addr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bfs_reachable(start: Any, live_edges: Iterable[tuple[Any, Any]]) -> set[Any]:
    """Nodes reachable from ``start`` over ``live_edges`` (undirected pairs).

    No radius bound — membership needs the full connected component.
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
