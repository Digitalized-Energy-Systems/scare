from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors
from mango_energy_environments import BranchFailureEvent

from scare.base.diagnostics import record_event
from scare.base.model import FailureNotice, LineFailure, Sector

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Initial TTL stamped on FailureNotice when an endpoint detects the
# failure.  Each same-sector hop costs 1; each CP-bridge hop costs
# ``_CP_BRIDGE_COST``.  At K=3 with bridge cost 2, a failure can reach
# directly-affected groups up to 3 same-sector hops away, plus the
# heat/gas neighbourhood one hop beyond a CP that sits within 1 hop of
# the failure — which matches the physical reach of a single-branch
# outage on the simbench LV grids without flooding the rest of the
# network.
_INITIAL_HOPS: int = 3
_CP_BRIDGE_COST: int = 2


class ProblemDetector(Role):
    """Per-node failure detector and distributed propagation hub.

    Two responsibilities:

    1. **Local conversion** — when a global ``BranchFailureEvent`` lands
       on one of the two endpoint nodes, emit a local ``LineFailure``
       event so the co-located ``GridReconfigurator`` can start its
       path search.  (Unchanged from before.)
    2. **Distributed propagation** — replaces the previous centralised
       pre-filter that used a global ``nx.Graph`` snapshot.  The
       endpoint detectors stamp a ``FailureNotice`` and gossip it
       through grid-topology neighbours, sector-tagged and TTL-bounded.
       Each detector along the way notifies the children at its node
       so their negotiators can react locally.

    The detector knows two pieces of locally-acquirable state passed at
    construction:

    - ``neighbour_branch_sectors`` — the sector of each grid-edge
      leaving this node.  Drives the per-edge forwarding cost.
    - ``child_addrs`` — agent addresses of the children sitting at
      this node.  The notice is delivered locally so the children's
      ``EnergyBalanceNegotiator`` can trigger a balance round.
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
        # Sector of the branch connecting this node to each grid
        # neighbour, keyed by neighbour node id.  Values: ``"electricity"``
        # / ``"gas"`` / ``"heat"`` / ``"cp"``.  Populated at scenario
        # build time; missing keys are treated as untraversable.
        self.neighbour_branch_sectors: dict[Any, str] = (
            dict(neighbour_branch_sectors) if neighbour_branch_sectors else {}
        )
        # Children at this node — recipients of local notices so their
        # negotiators can trigger without going through a global event.
        self.child_addrs: list[Any] = list(child_addrs) if child_addrs else []

        # Dedup table: maximum ``hops_remaining`` we have already
        # forwarded for each ``(origin_addr_str, branch_id)``.  A later
        # notice with strictly higher TTL overrides (because it can
        # reach farther); equal/lower TTLs are suppressed.  Same shape
        # as ``GridConstraintMonitor._state_forwarded`` for consistency.
        self._forwarded_ttl: dict[tuple, int] = {}
        # Same key, separate ledger for *delivered* (to local children)
        # so each child gets exactly one notice per unique failure.
        self._delivered: set[tuple] = set()

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

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

        # Determine the failing branch's sector.  The lookup table on
        # the behavior is populated once at scenario build; defensive
        # fallback to the sector of the local node's grid.
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
            # Distributed propagation is disabled (ablation comparison
            # against the centralised pre-filter).  The local
            # ``LineFailure`` event has already been emitted above for
            # the reconfigurator; nothing else to do here.
            return

        notice = FailureNotice(
            branch_id=event.branch_id,
            sector=sector,
            hops_remaining=self.ttl_hops,
            origin_addr=self.context.addr,
        )
        # Originator counts as "delivered" so a subsequent inbound
        # notice for the same failure isn't redundantly re-pushed.
        key = (str(notice.origin_addr), notice.branch_id)
        self._delivered.add(key)
        self._forwarded_ttl[key] = notice.hops_remaining
        self.context.schedule_instant_task(
            self._propagate(notice, exclude_addr=None)
        )

    # ------------------------------------------------------------------
    # Inbound notice
    # ------------------------------------------------------------------

    async def _handle_failure_notice(
        self, message: FailureNotice, meta: dict
    ) -> None:
        key = (str(message.origin_addr), message.branch_id)
        # Forwarding gate: only proceed if this copy is fresher than
        # what we have already forwarded.  Standard TTL-based dedup —
        # fresher copies (higher hops_remaining) reach farther.
        prev_ttl = self._forwarded_ttl.get(key)
        if prev_ttl is not None and message.hops_remaining <= prev_ttl:
            return
        self._forwarded_ttl[key] = message.hops_remaining

        # Delivery to local children is independent of forwarding gate:
        # we want to deliver exactly once per failure regardless of how
        # many copies traverse this node.  But the dedup table is
        # already updated above; gate on a separate ``_delivered`` set.
        if key not in self._delivered:
            self._delivered.add(key)
            await self._deliver_local(message)

        sender = mango_sender_addr(meta)
        await self._propagate(message, exclude_addr=sender)

    # ------------------------------------------------------------------
    # Forwarding + local delivery
    # ------------------------------------------------------------------

    async def _propagate(
        self, notice: FailureNotice, *, exclude_addr: Any
    ) -> None:
        """Forward ``notice`` to every grid neighbour whose connecting
        edge is traversable for this sector.  Skip the sender (no
        reflection) and the origin (closure of the propagation).
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
        """Send the notice to every child agent at this node.  The
        children's ``EnergyBalanceNegotiator`` decides — based on its
        sector — whether to trigger a balance negotiation.
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
    """Order-insensitive key for lookups: branches are undirected, so
    swap ``(from, to, idx)`` to its lex-min form for a stable key."""
    if len(branch_id) < 2:
        return branch_id
    a, b = branch_id[0], branch_id[1]
    if a <= b:
        return branch_id
    rest = tuple(branch_id[2:])
    return (b, a, *rest)


def _node_id_from_addr(addr: Any) -> Any:
    aid = getattr(addr, "aid", str(addr))
    if not aid.startswith("node-"):
        return None
    try:
        return int(aid.split("-", 1)[1])
    except ValueError:
        return None


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
    """Return the hop cost for traversing ``edge_sector`` while
    propagating a failure of ``failure_sector``.  ``None`` means the
    edge is untraversable (different-sector physical branch).
    """
    if edge_sector == failure_sector:
        return 1
    if edge_sector == "cp":
        return cp_bridge_cost
    return None
