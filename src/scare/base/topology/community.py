"""Community partitioning for hierarchical self-organisation.

Splits each connected sector subgraph into sub-communities for the L2/L3
layers. Default scheme is radius-bounded min-label propagation: each node
adopts the smallest neighbour label within ``max_radius`` hops of its seed.
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
    """Return a deterministic radius-bounded partition: ``{node_id: seed_label}``.

    Each node adopts the smallest neighbour label reachable within
    ``max_radius`` of that label's seed (ties broken by shorter distance).
    Stabilises in <= ``2 * max_radius + 1`` rounds; isolated nodes are singletons.
    """
    if max_radius < 0:
        raise ValueError("max_radius must be non-negative")

    state: dict[Hashable, tuple[Hashable, int]] = {n: (n, 0) for n in graph.nodes}

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


def modularity_partition(
    graph: nx.Graph,
    *,
    max_iterations: int = 10,
    resolution: float = 1.0,
) -> dict[Hashable, Hashable]:
    """Distributed-Louvain Phase 1: greedy local modularity-gain moves.

    Each node switches to the neighbour label maximising
    ``ΔQ = k_{i,c}/m − γ·(k_i·Σ_tot(c))/(2m²)``, iterating until no node moves
    or ``max_iterations``. ``resolution > 1`` gives finer partitions, ``< 1``
    coarser. Returns ``{node_id: label}``.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    m = graph.number_of_edges()
    if m == 0:
        return {n: n for n in graph.nodes}
    two_m_squared = 2.0 * m * m

    label: dict[Hashable, Hashable] = {n: n for n in graph.nodes}
    degree: dict[Hashable, int] = dict(graph.degree())
    # Σ_tot per community, maintained incrementally as nodes move.
    comm_degree: dict[Hashable, float] = {n: float(degree[n]) for n in graph.nodes}

    nodes_ordered = sorted(graph.nodes, key=_label_key)
    for _ in range(max_iterations):
        moved = False
        for node in nodes_ordered:
            k_i = degree[node]
            curr = label[node]

            # k_{i,c}: edges from node to nodes labelled c.
            edges_to_comm: dict[Hashable, int] = {}
            for neigh in graph.neighbors(node):
                cn = label[neigh]
                edges_to_comm[cn] = edges_to_comm.get(cn, 0) + 1

            # Remove node from its community before evaluating gains.
            comm_degree[curr] -= k_i

            candidates = set(edges_to_comm) | {curr}
            best_gain = float("-inf")
            best_label = curr
            for cand in candidates:
                k_i_in_c = edges_to_comm.get(cand, 0)
                sigma_tot = comm_degree.get(cand, 0.0)
                gain = k_i_in_c / m - resolution * (k_i * sigma_tot) / two_m_squared
                if gain > best_gain or (
                    gain == best_gain and _label_key(cand) < _label_key(best_label)
                ):
                    best_gain = gain
                    best_label = cand

            comm_degree[best_label] = comm_degree.get(best_label, 0.0) + k_i
            if best_label != curr:
                label[node] = best_label
                moved = True

        if not moved:
            break

    return label


def connected_component_partition(
    graph: nx.Graph,
) -> dict[Hashable, Hashable]:
    """Return a partition with one community per connected component.

    Each node's label is the lex-smallest node id in its component. Drives the
    ``component_level`` baseline. Returns ``{node_id: label}``.
    """
    label: dict[Hashable, Hashable] = {}
    for component in nx.connected_components(graph):
        seed = min(component, key=_label_key)
        for node in component:
            label[node] = seed
    return label


def modularity_of_partition(
    graph: nx.Graph,
    label_by_node: dict[Hashable, Hashable],
    *,
    resolution: float = 1.0,
) -> float:
    """Compute modularity ``Q = Σ_c [L_c/m − γ·(K_c/(2m))²]`` for a partition,
    where ``L_c`` is intra-community edges and ``K_c`` the degree sum.
    """
    m = graph.number_of_edges()
    if m == 0:
        return 0.0
    two_m = 2.0 * m

    degree = dict(graph.degree())
    members: dict[Hashable, list[Hashable]] = {}
    for n, c in label_by_node.items():
        members.setdefault(c, []).append(n)

    member_set: dict[Hashable, set] = {c: set(ns) for c, ns in members.items()}
    q = 0.0
    for c, nodes in members.items():
        l_c = 0
        for u in nodes:
            for v in graph.neighbors(u):
                if v in member_set[c] and v != u:
                    l_c += 1
        l_c //= 2  # each undirected edge counted twice above
        k_c = sum(degree[n] for n in nodes)
        q += (l_c / m) - resolution * (k_c / two_m) ** 2
    return q


def partition_label_by_node(
    topology: Topology,
    *,
    max_radius: int = 2,
    method: str = "label_propagation",
    modularity_iterations: int = 10,
    modularity_resolution: float = 1.0,
) -> dict[Hashable, Hashable]:
    """Return ``{node_id: label}`` for the given method.

    Same method semantics as :func:`communities_from_topology`; exposed for
    callers needing the per-node label assignment directly.
    """
    if method == "modularity":
        return modularity_partition(
            topology.graph,
            max_iterations=modularity_iterations,
            resolution=modularity_resolution,
        )
    if method == "label_propagation":
        return label_propagation_partition(topology.graph, max_radius=max_radius)
    if method == "connected_component":
        return connected_component_partition(topology.graph)
    raise ValueError(
        f"unknown community partition method: {method!r} "
        "(expected 'label_propagation', 'modularity', or "
        "'connected_component')"
    )


def agents_by_label_from_topology(
    topology: Topology,
    label_by_node: dict[Hashable, Hashable],
) -> list[list[Any]]:
    """Group ``topology``'s agents by the external ``label_by_node`` assignment.

    Returns communities (lists of agents), ordered by lex-smallest label;
    within each, agents sorted by AID so the leader (first) is deterministic.
    """
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


def communities_from_topology(
    topology: Topology,
    *,
    max_radius: int = 2,
    method: str = "label_propagation",
    modularity_iterations: int = 10,
    modularity_resolution: float = 1.0,
) -> list[list[Any]]:
    """Partition a per-sector physical *topology* into sub-communities.

    Methods: ``"label_propagation"`` (default, radius-bounded min-label balls),
    ``"modularity"`` (Louvain Phase 1, sizes vary), ``"connected_component"``
    (one per component; drives ``component_level`` baseline). Returns
    communities (lists of agents), each ordered by AID for a deterministic leader.
    """
    label_by_node = partition_label_by_node(
        topology,
        max_radius=max_radius,
        method=method,
        modularity_iterations=modularity_iterations,
        modularity_resolution=modularity_resolution,
    )
    return agents_by_label_from_topology(topology, label_by_node)
