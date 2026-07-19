"""Peer and membership resolution for the holonic community role.

Owns the deliverability wiring (mirror + leader node ids), the live-member
filter, the sector/component peer sets, and the component coordinator
election.
"""

from __future__ import annotations

import logging
from typing import Any

from mango.express.topology import topology_neighbors

from scare.base.model import LeaderEmerged, Sector
from scare.base.topology.topology_mirror import LivePeerFilter

logger = logging.getLogger(__name__)


class PeerResolver:
    """Resolves who this leader coordinates with.

    Reads the sector/topology through its owning role; owns the leader
    node-id registry that gates the component peer set.
    """

    def __init__(
        self,
        role: Any,
        *,
        live_member_filter: LivePeerFilter | None,
        my_node_id: Any,
        leader_node_ids: dict[str, Any] | None,
        topology_mirror: Any,
    ) -> None:
        self._role = role
        self.live_member_filter = live_member_filter
        self.my_node_id = my_node_id
        self.leader_node_ids: dict[str, Any] = dict(leader_node_ids or {})
        self.topology_mirror = topology_mirror

    @property
    def _sector(self) -> Sector:
        return self._role.sector

    def register_leader(self, message: LeaderEmerged) -> bool:
        """Register a promoted orphan-community leader. True iff newly added."""
        aid = str(message.leader_aid)
        if not aid:
            return False
        prior = self.leader_node_ids.get(aid)
        self.leader_node_ids[aid] = message.node_id
        return prior is None

    def live_members(self, members: list[Any]) -> list[Any]:
        """Subset of ``members`` the :class:`LivePeerFilter` deems reachable;
        passthrough when no filter is wired.
        """
        if self.live_member_filter is None:
            return members
        kept: list[Any] = []
        dropped: list[Any] = []
        for m in members:
            if self.live_member_filter.is_live(m):
                kept.append(m)
            else:
                dropped.append(m)
        if dropped and logger.isEnabledFor(logging.DEBUG):
            # ``role.context`` may be None in unit-test construction.
            ctx = getattr(self._role, "context", None)
            logger.debug(
                "[%s] holon filter dropped %d unreachable members (kept=%d)",
                getattr(ctx, "aid", "<detached>"),
                len(dropped),
                len(kept),
            )
        return kept

    def sector_peers(self) -> dict[str, Any]:
        """``{leader_aid: leader_addr}`` for every same-sector leader on the
        ``holon_summary_<sector>`` topology (incl. self). Unfiltered baseline.
        """
        ctx = self._role.context
        addrs: dict[str, Any] = {ctx.aid: ctx.addr}
        try:
            peers = list(
                topology_neighbors(
                    self._role, tid=f"holon_summary_{self._sector.value}"
                )
            )
        except Exception:
            return addrs
        for addr in peers:
            aid = getattr(addr, "aid", None)
            if aid is None:
                aid = str(addr)
            addrs[str(aid)] = addr
        return addrs

    def component_peers(self) -> dict[str, Any]:
        """``{leader_aid: leader_addr}`` for same-sector leaders in this
        leader's connected component. Falls back to the unfiltered sector peer
        set when the mirror or own node id is unavailable.

        Filter to known leader aids (self always included) to keep CP/branch
        agents out of the coordinator election (a report routed to one is
        dropped); empty ``leader_node_ids`` ⇒ unfiltered fallback.
        """
        ctx = self._role.context
        # Through the role so a test/subclass override of the sector peer set
        # is honoured.
        sector_peers = self._role._resolve_sector_peer_addrs()
        leader_aids = set(self.leader_node_ids)
        if leader_aids:
            sector_peers = {
                aid: addr
                for aid, addr in sector_peers.items()
                if aid == ctx.aid or aid in leader_aids
            }
        mirror = self.topology_mirror
        my_node = self.my_node_id
        if mirror is None or my_node is None:
            return sector_peers
        try:
            reachable = mirror.reachable_from(my_node, sector=self._sector)
        except Exception:
            return sector_peers
        out: dict[str, Any] = {}
        for aid, addr in sector_peers.items():
            if aid == ctx.aid:
                # Always include self even if ``leader_node_ids`` lacks it.
                out[aid] = addr
                continue
            node_id = self.leader_node_ids.get(aid)
            if node_id is None or node_id in reachable:
                out[aid] = addr
        return out

    def coordinator_aid(self) -> str | None:
        """Lex-smallest component-peer aid — the coordinator. None only when
        even self is missing (caller falls back to the per-holon path).
        """
        peers = self._role._resolve_component_peer_addrs()
        if not peers:
            return None
        return min(peers.keys())
