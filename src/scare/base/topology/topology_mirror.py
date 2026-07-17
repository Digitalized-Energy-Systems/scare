"""Central physical-grid mirror for the dynamic-topology layer.

Query-only mirror of the grid's live-edge state, built once from the monee
network and updated on branch failure/restoration. Queried by the L1/L2/L3
dynamic roles for peer reachability — the only centralised piece, since grid
state has one authoritative source (the simulator).

Two reachability flavours: sector-bounded (same-sector edges only, L1/L2) and
cross-sector with CP bridges (L3). Both share one BFS implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Protocol

from scare.base.model import Sector

logger = logging.getLogger(__name__)


class LivePeerFilter(Protocol):
    """Peer-filter interface a dynamic-topology role exposes to its host role.

    Implemented by L2/L3 roles; a ``None`` filter preserves static-topology
    behaviour, making the dynamic layer purely additive.
    """

    def is_live(self, addr: Any) -> bool: ...


# Sentinel sector tag for a cross-sector coupling-point branch (kept a plain
# string, not a Sector enum value, for hashable JSON-serialisable diagnostics).
_CP_BRIDGE: str = "cp"


class GridTopologyMirror:
    """Shared mirror of the physical branch-live set.

    Built once from the per-branch endpoint+sector table.
    ``mark_broken``/``mark_restored`` update the live set; BFS helpers query
    the current live subgraph. A plain data structure — callers drive it.
    """

    def __init__(
        self,
        *,
        branches: dict[tuple, tuple[Any, Any]],
        branch_sector: dict[tuple, str],
    ) -> None:
        """Build the mirror.

        ``branches`` maps ``branch_id`` to ``(node_a, node_b)``.
        ``branch_sector`` maps it to a sector value or the ``"cp"`` sentinel;
        unknown-sector branches are omitted (non-traversable anyway).
        """
        self._endpoints: dict[tuple, tuple[Any, Any]] = dict(branches)
        # branch_id -> sector tag ("electricity" / "heat" / "gas" / "cp").
        self._sector_tag: dict[tuple, str] = dict(branch_sector)
        # Live set: every branch not marked broken.
        self._live: set[tuple] = {
            bid for bid in self._endpoints if bid in self._sector_tag
        }
        self._broken: set[tuple] = set()
        # Per-sector (+CP) adjacency, rebuilt in full on each live-set change
        # (failures are sparse, so O(E) rebuild beats incremental bookkeeping).
        self._adj_by_sector: dict[str, dict[Any, list[Any]]] = {}
        self._rebuild_adjacency()

    # ------------------------------------------------------------------
    # State mutation
    # ------------------------------------------------------------------

    def mark_broken(self, branch_id: tuple) -> None:
        """Record a physical branch as broken.  Idempotent."""
        if branch_id not in self._sector_tag:
            # Unknown-sector edge is already non-traversable.
            return
        if branch_id in self._broken:
            return
        self._broken.add(branch_id)
        self._live.discard(branch_id)
        self._rebuild_adjacency()
        logger.debug(
            "mirror: branch %s marked broken; live=%d broken=%d",
            branch_id,
            len(self._live),
            len(self._broken),
        )

    def mark_restored(self, branch_id: tuple) -> None:
        """Record a previously-broken branch as restored.  Idempotent."""
        if branch_id not in self._broken:
            return
        self._broken.discard(branch_id)
        self._live.add(branch_id)
        self._rebuild_adjacency()
        logger.debug(
            "mirror: branch %s restored; live=%d broken=%d",
            branch_id,
            len(self._live),
            len(self._broken),
        )

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def is_broken(self, branch_id: tuple) -> bool:
        return branch_id in self._broken

    # ------------------------------------------------------------------
    # Reachability queries
    # ------------------------------------------------------------------

    def reachable_from(
        self,
        start_node: Any,
        *,
        sector: Sector | None = None,
        allow_cp_bridges: bool = False,
    ) -> set[Any]:
        """BFS the live subgraph and return all reachable node ids.

        ``sector`` restricts traversal to one sector; ``None`` admits all
        (L3 cross-sector). ``allow_cp_bridges`` additionally admits CP
        branches — meaningful only when ``sector is None``.
        """
        if sector is not None and allow_cp_bridges:
            # A sector-bounded query cannot cross a CP; reject the combination.
            raise ValueError(
                "reachable_from: allow_cp_bridges only makes sense when "
                f"sector is None (got sector={sector})"
            )

        if sector is not None:
            adj = self._adj_by_sector.get(sector.value, {})
        else:
            adj = self._merged_adjacency(include_cp=allow_cp_bridges)

        if start_node not in adj:
            # No live incident edges: node is its own component.
            return {start_node}

        seen: set[Any] = {start_node}
        frontier: list[Any] = [start_node]
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

    def is_reachable(
        self,
        from_node: Any,
        to_node: Any,
        *,
        sector: Sector | None = None,
        allow_cp_bridges: bool = False,
    ) -> bool:
        """Convenience wrapper.  Same semantics as ``reachable_from``."""
        if from_node == to_node:
            return True
        return to_node in self.reachable_from(
            from_node, sector=sector, allow_cp_bridges=allow_cp_bridges
        )

    def live_branches(
        self, *, sector: Sector | None = None, include_cp: bool = False
    ) -> Iterable[tuple]:
        """Iterate branch ids of every live edge matching the filter.

        ``sector`` restricts to one sector; ``include_cp`` admits CP bridges
        too (only meaningful when ``sector is None``).
        """
        for bid in self._live:
            tag = self._sector_tag.get(bid)
            if tag is None:
                continue
            if sector is not None:
                if tag == sector.value:
                    yield bid
                continue
            if tag == _CP_BRIDGE and not include_cp:
                continue
            yield bid

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_adjacency(self) -> None:
        """Rebuild the per-sector adjacency buckets from the live set."""
        buckets: dict[str, dict[Any, list[Any]]] = {}
        for bid in self._live:
            tag = self._sector_tag.get(bid)
            if tag is None:
                continue
            a, b = self._endpoints[bid]
            adj = buckets.setdefault(tag, {})
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        self._adj_by_sector = buckets

    def _merged_adjacency(self, *, include_cp: bool) -> dict[Any, list[Any]]:
        """Union of per-sector adjacency buckets (optionally CP), for L3's
        cross-sector reachability view.
        """
        merged: dict[Any, list[Any]] = {}
        for tag, adj in self._adj_by_sector.items():
            if tag == _CP_BRIDGE and not include_cp:
                continue
            for node, neighs in adj.items():
                merged.setdefault(node, []).extend(neighs)
        return merged


def mirror_from_monee(
    monee_net: Any,
    *,
    branch_sector_resolver,
) -> GridTopologyMirror:
    """Construct a mirror from a monee network.

    ``branch_sector_resolver(branch) -> str`` tags each branch (passed in to
    avoid a circular import).
    """
    branches: dict[tuple, tuple[Any, Any]] = {}
    sector_tag: dict[tuple, str] = {}
    for branch in monee_net.branches:
        tag = branch_sector_resolver(branch)
        if not tag:
            continue
        a, b = branch.id[0], branch.id[1]
        branches[branch.id] = (a, b)
        sector_tag[branch.id] = tag
    return GridTopologyMirror(branches=branches, branch_sector=sector_tag)
