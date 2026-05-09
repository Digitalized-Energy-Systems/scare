"""Community partitioning for hierarchical self-organisation.

Implements the Level-1 partitioning step described in
``docs/chapter_method.tex``: each connected sector subgraph is split into
multiple sub-communities so that the existing Level-2 (``HolonicCommunityRole``)
and Level-3 (``EnergyConverterRole``) coordination layers actually receive
multiple groups to aggregate.

The chosen scheme is **radius-bounded min-label propagation**.  Each node
starts as its own seed and in each synchronous round adopts the smallest
label it can see among its neighbours' labels, provided that label is
still within ``max_radius`` hops of its origin.  A node prefers a smaller
label even at greater distance, but never adopts a label whose distance
from its seed would exceed ``max_radius`` — that bound keeps each
community at most a ``max_radius``-hop ball around its seed and produces
multiple sub-communities per connected component as soon as the
component's diameter exceeds ``max_radius``.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import networkx as nx

from mango.express.topology import AGENT_NODE_KEY, Topology


def label_propagation_partition(
    graph: nx.Graph,
    *,
    max_radius: int = 2,
) -> dict[Hashable, Hashable]:
    """Return a deterministic radius-bounded partition of *graph*.

    Each entry maps a graph node id to the label (= seed node id) of the
    sub-community it belongs to.  Isolated nodes form a singleton
    community labelled by themselves.

    Within each round a node selects the smallest neighbour label it can
    reach without exceeding ``max_radius`` from that label's seed.  Ties
    on label resolve by the shorter distance.  After at most
    ``2 * max_radius + 1`` synchronous rounds the assignment stabilises:
    long-diameter components fragment into multiple ``max_radius``-balls
    centred on the lexicographically smallest reachable seeds.
    """
    if max_radius < 0:
        raise ValueError("max_radius must be non-negative")

    state: dict[Hashable, tuple[Hashable, int]] = {
        n: (n, 0) for n in graph.nodes
    }

    for _ in range(2 * max_radius + 1):
        next_state: dict[Hashable, tuple[Hashable, int]] = {}
        for node, current in state.items():
            best = current
            for neigh in graph.neighbors(node):
                n_label, n_dist = state[neigh]
                cand_dist = n_dist + 1
                if cand_dist > max_radius:
                    continue
                cand = (n_label, cand_dist)
                if (_label_key(cand[0]), cand[1]) < (_label_key(best[0]), best[1]):
                    best = cand
            next_state[node] = best
        if next_state == state:
            break
        state = next_state

    return {node: label for node, (label, _) in state.items()}


def _label_key(label: Hashable) -> Any:
    """Stable comparison key — labels can be ints, strings, or AIDs."""
    return (str(type(label).__name__), str(label))


def communities_from_topology(
    topology: Topology,
    *,
    max_radius: int = 2,
) -> list[list[Any]]:
    """Partition a per-sector physical *topology* into sub-communities.

    Returns a list of communities, each as a list of agents (the agents
    co-located on the topology node assigned to that community).  Within
    a community the agents are ordered by AID so the leader (first
    element) is deterministic.
    """
    label_by_node = label_propagation_partition(
        topology.graph, max_radius=max_radius
    )

    label_to_agents: dict[Hashable, list[Any]] = {}
    for tnode_id, label in label_by_node.items():
        agent_node = topology.graph.nodes[tnode_id][AGENT_NODE_KEY]
        if not agent_node.agents:
            continue
        label_to_agents.setdefault(label, []).extend(agent_node.agents)

    communities: list[list[Any]] = []
    for label in sorted(label_to_agents, key=_label_key):
        members = sorted(label_to_agents[label], key=lambda a: a.aid)
        communities.append(members)
    return communities
