"""Unit tests for the ``component_level`` baseline's connected-component
partition: each connected component collapses to a single community,
seeded on the lex-smallest node id.
"""

from __future__ import annotations

import networkx as nx

from scare.base.community import (
    agents_by_label_from_topology,
    communities_from_topology,
    connected_component_partition,
    label_propagation_partition,
    partition_label_by_node,
)


def test_single_component_collapses_to_one_label() -> None:
    g = nx.path_graph(6)  # 0-1-2-3-4-5
    labels = connected_component_partition(g)
    # One label across all six nodes — the lex-smallest seed (0).
    assert set(labels.values()) == {0}
    assert all(labels[n] == 0 for n in g.nodes)


def test_two_components_get_separate_labels() -> None:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])         # component A
    g.add_edges_from([(10, 11), (11, 12)])     # component B
    labels = connected_component_partition(g)
    # Each component's label is its lex-smallest node id.
    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == 0
    assert labels[10] == 10
    assert labels[11] == 10
    assert labels[12] == 10


def test_isolated_nodes_are_their_own_label() -> None:
    g = nx.Graph()
    g.add_nodes_from([5, 7, 9])
    labels = connected_component_partition(g)
    assert labels == {5: 5, 7: 7, 9: 9}


def test_component_partition_subsumes_label_propagation_on_small_diameter() -> None:
    # A 3-node path has diameter 2; label propagation at max_radius=2 also
    # collapses it to a single seed, so both partitions agree here.
    g = nx.path_graph(3)
    cc = connected_component_partition(g)
    lp = label_propagation_partition(g, max_radius=2)
    assert set(cc.values()) == set(lp.values()) == {0}


def test_component_partition_widens_label_propagation_balls() -> None:
    # A 6-node path has diameter 5; label propagation at max_radius=2
    # fragments it into multiple balls, while the connected-component
    # partition keeps it as one community.
    g = nx.path_graph(6)
    cc = connected_component_partition(g)
    lp = label_propagation_partition(g, max_radius=2)
    assert len(set(cc.values())) == 1
    assert len(set(lp.values())) >= 2


class _FakeTopology:
    """Minimal stand-in for mango's ``Topology``: exposes ``.graph`` only."""

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph


def test_partition_label_by_node_dispatches_to_connected_component() -> None:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (10, 11)])
    topo = _FakeTopology(g)
    labels = partition_label_by_node(topo, method="connected_component")
    assert labels[0] == 0 and labels[1] == 0
    assert labels[10] == 10 and labels[11] == 10


def test_partition_label_by_node_unknown_method_raises() -> None:
    import pytest

    g = nx.path_graph(3)
    topo = _FakeTopology(g)
    with pytest.raises(ValueError):
        partition_label_by_node(topo, method="not_a_method")
