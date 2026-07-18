from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors
from mango_energy_environments import BranchFailureEvent

from scare.base.addressing import addr_aid, node_id_from_aid
from scare.base.model import FailureNotice, LineFailure, Sector
from scare.base.runtime.diagnostics import record_event
from scare.base.util import async_dispatch

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Initial FailureNotice TTL. Same-sector hop costs 1, CP-bridge hop costs
# _CP_BRIDGE_COST. Tuned to a single-branch outage's reach without flooding.
_INITIAL_HOPS: int = 3
_CP_BRIDGE_COST: int = 2


class ProblemDetector(Role):
    """Per-node detector: on a ``BranchFailureEvent`` at an endpoint, emits a local
    ``LineFailure`` for the reconfigurator and gossips a sector-tagged, TTL-bounded
    ``FailureNotice`` through grid neighbours to each node's children. State:
    ``neighbour_branch_sectors`` (per-edge cost), ``child_addrs`` (local recipients).
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        node_id: Any,
        *,
        neighbour_branch_sectors: dict[Any, str] | None = None,
        child_addrs: list[Any] | None = None,
        enable_distributed_failure_notice: bool = True,
        ttl_hops: int = _INITIAL_HOPS,
        cp_bridge_cost: int = _CP_BRIDGE_COST,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.node_id = node_id
        self.enable_distributed_failure_notice = enable_distributed_failure_notice
        self.ttl_hops = ttl_hops
        self.cp_bridge_cost = cp_bridge_cost
        # Sector of the grid edge to each neighbour, by node id; missing keys
        # are untraversable.
        self.neighbour_branch_sectors: dict[Any, str] = (
            dict(neighbour_branch_sectors) if neighbour_branch_sectors else {}
        )
        # Children at this node — recipients of local notices.
        self.child_addrs: list[Any] = list(child_addrs) if child_addrs else []

        # Max hops_remaining already forwarded per (origin, branch_id); a
        # strictly higher TTL overrides, equal/lower are suppressed.
        self._forwarded_ttl: dict[tuple, int] = {}
        # Same key; ensures each child is delivered exactly once per failure.
        self._delivered: set[tuple] = set()

    def setup(self) -> None:
        _wrap = async_dispatch(self)

        self.context.subscribe_message(
            self,
            _wrap(self._handle_failure_notice),
            lambda msg, meta: isinstance(msg, FailureNotice),
        )

    def on_global_event(self, event: Any) -> None:
        if isinstance(event, BranchFailureEvent):
            self._on_branch_failure(event)

    # ------------------------------------------------------------------
    # Endpoint origination
    # ------------------------------------------------------------------

    def _on_branch_failure(self, event: BranchFailureEvent) -> None:
        from_id, to_id = event.branch_id[0], event.branch_id[1]
        if self.node_id not in (from_id, to_id):
            return

        logger.info(
            "[%s] forwarding branch failure %s",
            self.context.aid,
            event.branch_id,
        )
        record_event(
            t=self.context.current_timestamp,
            kind="line_failure",
            aid=self.context.aid,
            detail=str(event.branch_id),
        )
        self.context.emit_event(
            LineFailure(
                source_node_id=self.node_id,
                target_node_id=to_id if self.node_id == from_id else from_id,
                branch_id=event.branch_id,
            )
        )

        sector_str = self._lookup_branch_sector(event.branch_id)
        sector = _sector_from_str(sector_str)
        if sector is None:
            logger.debug(
                "[%s] branch %s has unknown sector — skipping FailureNotice",
                self.context.aid,
                event.branch_id,
            )
            return

        if not self.enable_distributed_failure_notice:
            # Propagation disabled; local LineFailure already emitted above.
            return

        notice = FailureNotice(
            branch_id=event.branch_id,
            sector=sector,
            hops_remaining=self.ttl_hops,
            origin_addr=self.context.addr,
        )
        # Mark originator delivered so a later inbound copy isn't re-pushed.
        key = (str(notice.origin_addr), notice.branch_id)
        self._delivered.add(key)
        self._forwarded_ttl[key] = notice.hops_remaining
        self.context.schedule_instant_task(self._propagate(notice, exclude_addr=None))

    # ------------------------------------------------------------------
    # Inbound notice
    # ------------------------------------------------------------------

    async def _handle_failure_notice(self, message: FailureNotice, meta: dict) -> None:
        key = (str(message.origin_addr), message.branch_id)
        # Only forward copies fresher (higher TTL) than already forwarded.
        prev_ttl = self._forwarded_ttl.get(key)
        if prev_ttl is not None and message.hops_remaining <= prev_ttl:
            return
        self._forwarded_ttl[key] = message.hops_remaining

        # Gated separately so each child receives the notice exactly once.
        if key not in self._delivered:
            self._delivered.add(key)
            await self._deliver_local(message)

        sender = mango_sender_addr(meta)
        await self._propagate(message, exclude_addr=sender)

    # ------------------------------------------------------------------
    # Forwarding + local delivery
    # ------------------------------------------------------------------

    async def _propagate(self, notice: FailureNotice, *, exclude_addr: Any) -> None:
        """Forward ``notice`` to every grid neighbour whose edge is traversable
        for this sector, skipping the sender and the origin.
        """
        for neigh_addr in topology_neighbors(self, tid="grid"):
            if exclude_addr is not None and neigh_addr == exclude_addr:
                continue
            if neigh_addr == notice.origin_addr:
                continue
            neigh_node_id = _node_id_from_addr(neigh_addr)
            if neigh_node_id is None:
                continue
            edge_sector = self.neighbour_branch_sectors.get(neigh_node_id)
            if edge_sector is None:
                continue
            cost = _edge_cost(
                edge_sector, notice.sector.value, cp_bridge_cost=self.cp_bridge_cost
            )
            if cost is None:
                continue
            new_hops = notice.hops_remaining - cost
            if new_hops < 1:
                continue
            await self.context.send_message(
                FailureNotice(
                    branch_id=notice.branch_id,
                    sector=notice.sector,
                    hops_remaining=new_hops,
                    origin_addr=notice.origin_addr,
                ),
                receiver_addr=neigh_addr,
            )

    async def _deliver_local(self, notice: FailureNotice) -> None:
        """Send the notice to every child; each decides per sector whether to
        trigger a balance negotiation.
        """
        for child_addr in self.child_addrs:
            await self.context.send_message(notice, receiver_addr=child_addr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lookup_branch_sector(self, branch_id: tuple) -> str:
        store = getattr(self.behavior, "_scare_branch_sector", None)
        if store is None:
            return ""
        return store.get(branch_id) or store.get(_normalise_branch_id(branch_id), "")


def _normalise_branch_id(branch_id: tuple) -> tuple:
    """Order-insensitive lookup key: branches are undirected, so map to lex-min form."""
    if len(branch_id) < 2:
        return branch_id
    a, b = branch_id[0], branch_id[1]
    if a <= b:
        return branch_id
    rest = tuple(branch_id[2:])
    return (b, a, *rest)


def _node_id_from_addr(addr: Any) -> Any:
    return node_id_from_aid(addr_aid(addr))


def _sector_from_str(s: str) -> Sector | None:
    if not s:
        return None
    if s == Sector.ELECTRICITY.value:
        return Sector.ELECTRICITY
    if s == Sector.GAS.value:
        return Sector.GAS
    if s == Sector.HEAT.value:
        return Sector.HEAT
    return None


def _edge_cost(
    edge_sector: str,
    failure_sector: str,
    *,
    cp_bridge_cost: int = _CP_BRIDGE_COST,
) -> int | None:
    """Hop cost to traverse ``edge_sector`` while propagating a
    ``failure_sector`` failure; ``None`` means untraversable.
    """
    if edge_sector == failure_sector:
        return 1
    if edge_sector == "cp":
        return cp_bridge_cost
    return None
