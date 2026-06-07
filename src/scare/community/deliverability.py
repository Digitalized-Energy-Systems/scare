"""Per-actor deliverability caps for the coalition supply-priority ADMM.

The base ADMM treats each actor's supply as fungible across its (sector, tier)
cells. That fails for the L2.5 coalition, which spans the full sector mesh and
may include leaders partitioned after a branch failure. Per member ``g`` and
cell this bounds deliverable supply by the tier-t demand reachable from ``g``'s
home node over live same-sector edges (unreachable → ~zero), so the ADMM can't
allocate supply the downstream LP can't route.

These are delivery caps, not supply caps: ``Σ_t x_g ≤ supply_g`` still holds via
the coupling matrix; the cap only narrows each cell's ub to what's reachable.
"""

from __future__ import annotations

import logging
from typing import Any

from scare.base.model import Sector
from scare.base.topology.topology_mirror import GridTopologyMirror

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
        Each actor's home node id; ``None`` → that actor gets ``None``
        (no override, ADMM uses raw supply).
    actor_demand_nodes_by_tier
        Per-actor ``{tier: {node_id: demand_mw}}``; the destination set
        other actors' supply can serve.
    sector
        Coalition's sector. Reachability is evaluated in this subgraph only
        (cross-sector delivery is L3's concern).
    mirror
        Shared :class:`GridTopologyMirror`; ``None`` → every entry ``None``
        (raw-supply fallback).

    Returns
    -------
    Per-actor optional ``{(sector_value, tier): cap_mw}`` maps. ``None`` = no
    override; a cap=0 cell tells the ADMM to set ub[cell] = 1e-6.
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

    # Per-tier demand aggregated across actors: A's supply can serve B's
    # demand when the path is live, which is what the cap models.
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
            # Malformed mirror entry: fall back to raw supply, don't crash.
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
