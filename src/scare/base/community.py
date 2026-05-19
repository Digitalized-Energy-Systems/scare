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


def modularity_partition(
    graph: nx.Graph,
    *,
    max_iterations: int = 10,
    resolution: float = 1.0,
) -> dict[Hashable, Hashable]:
    """Distributed-Louvain Phase 1: greedy local modularity-gain moves.

    Each node greedily switches its community label to the one (among
    its neighbours' current labels) that maximises the local
    modularity gain

        ΔQ = k_{i,c} / m − γ · (k_i · Σ_tot(c)) / (2m²)

    where ``k_i`` is the node's degree, ``k_{i,c}`` is the number of
    edges from ``i`` to nodes currently labelled ``c``, ``Σ_tot(c)``
    is the sum of degrees in ``c`` (excluding ``i``), and ``γ`` is
    the resolution parameter.  Iterates synchronous rounds until no
    node moves or ``max_iterations`` is reached.

    The algorithm is the canonical Louvain Phase 1 — fully local at
    each step (everything in the formula except ``m`` is computable
    from the node's neighbour view); the centralised driver here is
    a faithful simulation of the distributed run, returning the same
    deterministic partition the distributed version would converge
    to.  ``m`` is a single scalar broadcast once at scenario time in
    the production distributed version.

    ``resolution > 1`` produces finer partitions (more, smaller
    communities); ``< 1`` coarser.  Default 1.0 = standard
    modularity.

    Returns ``{node_id: label}``, same shape as
    :func:`label_propagation_partition`, so call sites are
    interchangeable.
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
    # Σ_tot per community.  Maintained incrementally as nodes move.
    comm_degree: dict[Hashable, float] = {n: float(degree[n]) for n in graph.nodes}

    nodes_ordered = sorted(graph.nodes, key=_label_key)
    for _ in range(max_iterations):
        moved = False
        for node in nodes_ordered:
            k_i = degree[node]
            curr = label[node]

            # k_{i,c}: number of edges from node to nodes labelled c
            edges_to_comm: dict[Hashable, int] = {}
            for neigh in graph.neighbors(node):
                cn = label[neigh]
                edges_to_comm[cn] = edges_to_comm.get(cn, 0) + 1

            # Remove node from its current community for the
            # evaluation; we'll add it back to whichever it joins.
            comm_degree[curr] -= k_i

            candidates = set(edges_to_comm) | {curr}
            best_gain = float("-inf")
            best_label = curr
            for cand in candidates:
                k_i_in_c = edges_to_comm.get(cand, 0)
                sigma_tot = comm_degree.get(cand, 0.0)
                gain = (
                    k_i_in_c / m
                    - resolution * (k_i * sigma_tot) / two_m_squared
                )
                if gain > best_gain or (
                    gain == best_gain
                    and _label_key(cand) < _label_key(best_label)
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


def modularity_of_partition(
    graph: nx.Graph,
    label_by_node: dict[Hashable, Hashable],
    *,
    resolution: float = 1.0,
) -> float:
    """Compute modularity ``Q`` for a partition of ``graph``.

    ``Q = (1/2m) Σ_ij [A_ij − γ · k_i k_j / (2m)] · δ(c_i, c_j)``
    expanded per-community as
    ``Q = Σ_c [L_c/m − γ · (K_c/(2m))²]``
    where ``L_c`` is the number of intra-community edges and ``K_c``
    the degree sum.  Used for diagnostic comparison of partitions
    produced by different methods (label-propagation vs modularity).
    """
    m = graph.number_of_edges()
    if m == 0:
        return 0.0
    two_m = 2.0 * m

    degree = dict(graph.degree())
    # Group node sets per community label.
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
        l_c //= 2  # each edge counted twice in the undirected sum
        k_c = sum(degree[n] for n in nodes)
        q += (l_c / m) - resolution * (k_c / two_m) ** 2
    return q


def communities_from_topology(
    topology: Topology,
    *,
    max_radius: int = 2,
    method: str = "label_propagation",
    modularity_iterations: int = 10,
    modularity_resolution: float = 1.0,
) -> list[list[Any]]:
    """Partition a per-sector physical *topology* into sub-communities.

    Two methods are available:

    - ``"label_propagation"`` (default, original behaviour): radius-
      bounded min-label propagation — communities are ≤``max_radius``-
      hop balls centred on the lex-smallest reachable seed.
    - ``"modularity"``: distributed-Louvain Phase 1 — communities form
      to maximise local modularity gain, respecting the graph's
      natural cluster structure.  Sizes vary; not bounded by radius.

    Returns a list of communities, each as a list of agents (the agents
    co-located on the topology node assigned to that community).  Within
    a community the agents are ordered by AID so the leader (first
    element) is deterministic.
    """
    if method == "modularity":
        label_by_node = modularity_partition(
            topology.graph,
            max_iterations=modularity_iterations,
            resolution=modularity_resolution,
        )
    elif method == "label_propagation":
        label_by_node = label_propagation_partition(
            topology.graph, max_radius=max_radius
        )
    else:
        raise ValueError(
            f"unknown community partition method: {method!r} "
            "(expected 'label_propagation' or 'modularity')"
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
