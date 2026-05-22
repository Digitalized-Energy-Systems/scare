"""Per-actor deliverability caps for the coalition supply-priority ADMM.

The base supply-priority ADMM (see :mod:`scare.community.supply_priority_admm`)
treats each actor's supply as fungible across all (sector, tier) cells
where that actor has supply in the sector.  That assumption holds for
the parent holon — its chunk-mates are all on the same connected
subgraph at scenario build time — but fails for the L2.5 coalition,
which spans the full sector mesh and may include leaders whose home
nodes are physically partitioned from each other after a branch
failure.

This module computes, for each coalition member ``g`` and (sector,
tier) cell, an upper bound on supply that ``g`` could plausibly
deliver to demand in that cell.  The bound is the sum of tier-t
demand at nodes physically reachable from ``g``'s home node through
live same-sector edges.  Supply at ``g`` that cannot reach any
tier-t demand is capped at zero (1e-6 in practice to keep the solver
well-conditioned), which prevents the ADMM from allocating
"impossible" supply that the downstream LP cannot route.

The caps are *delivery caps*, not *supply caps*: the ADMM still
enforces ``Σ_t x_g ≤ supply_g`` via the per-actor coupling matrix.
The cap only narrows the per-cell ub down from raw supply to what's
physically reachable, so an actor with abundant supply but no live
path to any demand contributes zero, and the ADMM redistributes the
target onto reachable actors only.
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

    # Pre-compute, per tier, the global demand-node → demand map.
    # Aggregating across actors lets each actor's reachability
    # check sum over all demand locations regardless of who owns
    # them (supply at actor A can deliver to actor B's demand if
    # the path is live, and that's exactly what the cap models).
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
            # Defensive: a malformed mirror entry should not crash
            # the coalition allocation.  Falling back to None means
            # the ADMM uses raw supply for this actor — same as the
            # no-deliverability path.
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
