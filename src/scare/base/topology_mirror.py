"""Central physical-grid mirror for the dynamic-topology layer.

A query-only data structure mirroring the physical live-edge state of the
multi-energy grid.  Built once at scenario time from the monee network,
updated on branch failure/restoration, and queried by the per-layer
dynamic roles (L1/L2/L3) to decide which peers are reachable through live
infrastructure.

This is the only centralised piece in the dynamic-topology design: the
physical grid state has a single authoritative source (the simulator), so
mirroring it once and sharing by reference beats rebuilding an adjacency
table per role.  All cyber-organisational decisions (community membership,
leadership, ADMM participation) stay local; the mirror only answers
reachability questions.

Two reachability flavours are exposed:

- **Sector-bounded**: traverse only same-sector edges.  Used by L1
  (group membership is within one sector) and by L2 (a holon is a
  same-sector chunk of group leaders).
- **Cross-sector with CP bridges**: traverse same-sector edges *and*
  coupling-point branches.  Used by L3 (a CP's connected group leaders
  span all sectors the CP bridges).

Both share the same BFS implementation, parametrised by an
edge-traversability predicate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from scare.base.model import Sector

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LivePeerFilter(Protocol):
    """Query interface a dynamic-topology role exposes to its co-located
    host role.  ``addr`` is any mango address (carries an ``aid``).

    Implemented by ``DynamicHolonRole`` (L2) and ``DynamicConnectorRole``
    (L3) so the host role can filter peers.  A ``None`` filter at the host
    side preserves static-topology behaviour, making the dynamic layer
    purely additive.
    """

    def is_live(self, addr: Any) -> bool:
        ...


# Sentinel sector tag for a cross-sector coupling-point branch.  A string
# (not a ``Sector`` enum value, which are reserved for real sectors) so the
# dict keys stay hashable and JSON-serialisable for diagnostics.
_CP_BRIDGE: str = "cp"


class GridTopologyMirror:
    """Shared mirror of the physical branch-live set.

    Constructed once at scenario time with the full per-branch endpoint
    + sector table.  ``mark_broken`` / ``mark_restored`` update the
    live set; the BFS query helpers answer reachability questions on
    the current live subgraph.

    The mirror does not own any threading or async machinery — it's a
    plain data structure.  Callers (the scenario-level
    ``set_on_branch_failure`` callback, the per-role reassessment
    routines) drive it explicitly.
    """

    def __init__(
        self,
        *,
        branches: dict[tuple, tuple[Any, Any]],
        branch_sector: dict[tuple, str],
    ) -> None:
        """Build the mirror.

        ``branches`` maps ``branch_id`` to ``(node_a, node_b)``; both
        same-sector branches and CP bridges are included.
        ``branch_sector`` maps ``branch_id`` to a sector string —
        either a :class:`Sector` value ("electricity"/"heat"/"gas") for
        same-sector edges, or the sentinel ``"cp"`` for cross-sector
        coupling-point branches.  Branches with an unknown sector are
        omitted (they would be non-traversable anyway, so admitting them
        only clutters the live set).
        """
        self._endpoints: dict[tuple, tuple[Any, Any]] = dict(branches)
        # branch_id -> sector tag ("electricity" / "heat" / "gas" / "cp").
        self._sector_tag: dict[tuple, str] = dict(branch_sector)
        # Live set: every branch not marked broken.
        self._live: set[tuple] = {
            bid for bid in self._endpoints if bid in self._sector_tag
        }
        self._broken: set[tuple] = set()
        # Per-sector (plus CP-bridge) adjacency for fast reachable_from().
        # Rebuilt in full on each live-set change; failures arrive sparsely
        # so an O(E) rebuild beats incremental-update bookkeeping.
        self._adj_by_sector: dict[str, dict[Any, list[Any]]] = {}
        self._rebuild_adjacency()

    # ------------------------------------------------------------------
    # State mutation
    # ------------------------------------------------------------------

    def mark_broken(self, branch_id: tuple) -> None:
        """Record a physical branch as broken.  Idempotent."""
        if branch_id not in self._sector_tag:
            # An edge with no known sector is already non-traversable.
            return
        if branch_id in self._broken:
            return
        self._broken.add(branch_id)
        self._live.discard(branch_id)
        self._rebuild_adjacency()
        logger.debug(
            "mirror: branch %s marked broken; live=%d broken=%d",
            branch_id, len(self._live), len(self._broken),
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
            branch_id, len(self._live), len(self._broken),
        )

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def is_broken(self, branch_id: tuple) -> bool:
        return branch_id in self._broken

    def broken_branches(self) -> frozenset[tuple]:
        return frozenset(self._broken)

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

        ``sector`` restricts traversal to edges of that one sector.
        ``None`` admits all sectors (intended for cross-sector queries
        from L3).  ``allow_cp_bridges`` additionally admits CP coupling
        branches — meaningful only when ``sector is None`` (a same-
        sector reachability inside a single sector cannot cross a CP).
        """
        if sector is not None and allow_cp_bridges:
            # A sector-bounded query is within one sector graph and cannot
            # cross a CP; reject the contradictory combination loudly.
            raise ValueError(
                "reachable_from: allow_cp_bridges only makes sense when "
                "sector is None (got sector=%r)" % sector
            )

        if sector is not None:
            adj = self._adj_by_sector.get(sector.value, {})
        else:
            adj = self._merged_adjacency(include_cp=allow_cp_bridges)

        if start_node not in adj:
            # No live incident edges: the node is its own component.
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
        """Iterate the branch ids of every live edge matching the filter.

        ``sector`` restricts to a single sector; ``include_cp`` admits
        CP bridges in addition to same-sector edges (only meaningful
        when ``sector is None``).
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
        """Union of all per-sector adjacency buckets (optionally CP),
        built on demand for L3's cross-sector reachability view.
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

    ``branch_sector_resolver(branch) -> str`` is the same resolver
    ``scenario/restoration.py`` uses to tag branches as
    ``"electricity"`` / ``"heat"`` / ``"gas"`` / ``"cp"`` / ``""``
    (defensive fallback).  Passed in so we don't import the resolver
    here and create a circular dependency.
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
