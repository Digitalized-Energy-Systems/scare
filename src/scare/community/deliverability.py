"""Per-actor deliverability caps for the coalition supply-priority ADMM.

The base supply-priority ADMM treats each actor's supply as fungible
across all (sector, tier) cells where it has supply.  That holds for the
parent holon (chunk-mates share a connected subgraph) but fails for the
L2.5 coalition, which spans the full sector mesh and may include leaders
physically partitioned from each other after a branch failure.

This module computes, per coalition member ``g`` and (sector, tier) cell,
an upper bound on supply ``g`` can deliver there: the sum of tier-t demand
at nodes reachable from ``g``'s home node over live same-sector edges.
Supply that cannot reach any tier-t demand is capped at zero (1e-6 in
practice for conditioning), so the ADMM cannot allocate supply the
downstream LP cannot route.

These are delivery caps, not supply caps: the ADMM still enforces
``Σ_t x_g ≤ supply_g`` via the per-actor coupling matrix.  The cap only
narrows the per-cell ub to what's physically reachable, redistributing the
target onto reachable actors.
"""

from __future__ import annotations

import logging
from typing import Any

from scare.base.model import Sector
from scare.base.topology_mirror import GridTopologyMirror

logger = logging.getLogger(__name__)


def per_actor_deliverable_caps(
    *,
    actor_node_ids: list[Any | None],
    actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]],
    sector: Sector,
    mirror: GridTopologyMirror | None,
) -> list[dict[tuple[str, int], float] | None]:
    """Return per-actor (sector, tier) deliverability caps.

    Parameters
    ----------
    actor_node_ids
        Each actor's home node id in the grid (typically the
        leader's monee node).  ``None`` for an actor whose node is
        unknown — that actor gets ``None`` (no override → ADMM falls
        back to raw supply caps).
    actor_demand_nodes_by_tier
        Per-actor ``{tier: {node_id: demand_mw}}`` map.  These are
        the locations the actor's own loads sit at, keyed by tier,
        which become the "destination set" for other actors' supply.
    sector
        The sector the coalition operates in.  Reachability is
        evaluated in this sector's subgraph only — cross-sector
        delivery requires a CP, which is L3's concern.
    mirror
        Shared :class:`GridTopologyMirror`.  ``None`` short-circuits
        every entry to ``None`` (no override), preserving the raw-
        supply behaviour for callers running without dynamic
        topology.

    Returns
    -------
    list of optional ``{(sector_value, tier): cap_mw}`` maps, one per
    actor positionally.  ``None`` means "no override for this actor"
    (ADMM uses raw supply); an empty dict means "all cells capped at
    0" implicitly (no overrides → raw supply).  A populated entry
    with cap=0 for a cell tells the ADMM to set ub[cell] = 1e-6.
    """
    if mirror is None:
        return [None] * len(actor_node_ids)
    if len(actor_node_ids) != len(actor_demand_nodes_by_tier):
        raise ValueError(
            "per_actor_deliverable_caps: actor_node_ids and "
            "actor_demand_nodes_by_tier lengths differ"
        )

    sec_v = sector.value
    n_actors = len(actor_node_ids)

    # Per-tier global demand-node -> demand map, aggregated across actors:
    # supply at actor A can serve actor B's demand if the path is live,
    # which is exactly what the cap models.
    demand_by_tier_nodes: dict[int, dict[Any, float]] = {}
    for actor_map in actor_demand_nodes_by_tier:
        for tier, node_dem in actor_map.items():
            bucket = demand_by_tier_nodes.setdefault(tier, {})
            for node, dem in node_dem.items():
                bucket[node] = bucket.get(node, 0.0) + float(dem)

    caps: list[dict[tuple[str, int], float] | None] = []
    for g in range(n_actors):
        home_node = actor_node_ids[g]
        if home_node is None:
            caps.append(None)
            continue
        try:
            reachable = mirror.reachable_from(home_node, sector=sector)
        except Exception:
            # A malformed mirror entry must not crash the allocation;
            # None falls the ADMM back to raw supply for this actor.
            caps.append(None)
            continue
        actor_caps: dict[tuple[str, int], float] = {}
        for tier, node_dem in demand_by_tier_nodes.items():
            cap = 0.0
            for node, dem in node_dem.items():
                if node in reachable:
                    cap += dem
            actor_caps[(sec_v, tier)] = cap
        caps.append(actor_caps)
    return caps
