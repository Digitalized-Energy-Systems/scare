"""Regression tests for the L2 component-scope vs priority-invariant
metric mismatch.

The metric's full-graph component index admits CP coupling branches, so
electricity-only-split nodes can still share one component via a
CP->heat->CP chain — while L2 coordinator election queries the topology
mirror per-sector (no CP bridges). The mismatch lets the
priority_invariant claim aggregate loads across a coordinator boundary
and report a spurious tier inversion. The fix aligns the metric scope
with L2's per-sector election scope.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from scare.base.model import Sector
from scare.base.topology.topology_mirror import GridTopologyMirror
from scare.community.holonic import HolonicCommunityRole

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubAddr:
    def __init__(self, aid: str) -> None:
        self.aid = aid

    def __repr__(self) -> str:
        return f"<addr {self.aid}>"


class _StubContext:
    def __init__(self, aid: str) -> None:
        self.aid = aid
        self.addr = _StubAddr(aid)
        # Some role paths read the sim clock; the stub has no scheduler.
        self.current_timestamp = 0.0


def _make_leader_role(
    *,
    aid: str,
    my_node_id: int,
    leader_node_ids: dict[str, int],
    mirror: GridTopologyMirror,
) -> HolonicCommunityRole:
    """Construct a HolonicCommunityRole stub wired with the bits the
    component-scope path reads."""
    role = HolonicCommunityRole(
        sector=Sector.ELECTRICITY,
        my_node_id=my_node_id,
        leader_node_ids=leader_node_ids,
        topology_mirror=mirror,
        admm_scope="component",
    )
    role._context = _StubContext(aid)  # Role.context is a property over _context
    return role


def _patch_sector_peers(
    role: HolonicCommunityRole, peer_addrs: dict[str, _StubAddr]
) -> None:
    """Inject the unfiltered sector-peer set directly so the topology
    mirror filter is the only variable under test."""
    role._resolve_sector_peer_addrs = lambda: dict(peer_addrs)  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Fixture: 2 electricity islands joined only by a CP/heat bridge.
# ---------------------------------------------------------------------------
#
#   Electricity island A:  el-1 --a1-- el-2  (leader-A at el-1)
#                                |
#                               cp1  (CP bridge, electricity <-> heat)
#                                |
#                               heat-10 --h-- heat-11
#                                              |
#                                             cp2  (CP bridge, heat <-> electricity)
#                                              |
#   Electricity island B:  el-20 --a2-- el-21  (leader-B at el-20)
#
# No electricity-only path connects the two islands, but the CP/heat
# chain joins them, so the full-graph metric (CP branches included) sees
# ONE component.
# ---------------------------------------------------------------------------


def _two_island_grid() -> tuple[GridTopologyMirror, dict[str, int]]:
    branches = {
        ("a1",): (1, 2),  # el
        ("a2",): (20, 21),  # el
        ("cp1",): (2, 10),  # cp bridge (el <-> heat)
        ("h",): (10, 11),  # heat
        ("cp2",): (11, 20),  # cp bridge (heat <-> el)
    }
    branch_sector = {
        ("a1",): "electricity",
        ("a2",): "electricity",
        ("cp1",): "cp",
        ("h",): "heat",
        ("cp2",): "cp",
    }
    mirror = GridTopologyMirror(branches=branches, branch_sector=branch_sector)
    leader_node_ids = {"leader-A": 1, "leader-B": 20}
    return mirror, leader_node_ids


def test_mirror_electricity_only_view_is_split() -> None:
    """Electricity-only reachability isolates each island."""
    mirror, _ = _two_island_grid()
    reach_A = mirror.reachable_from(1, sector=Sector.ELECTRICITY)
    reach_B = mirror.reachable_from(20, sector=Sector.ELECTRICITY)
    assert reach_A == {1, 2}
    assert reach_B == {20, 21}
    assert reach_A.isdisjoint(reach_B)


def test_metrics_active_components_legacy_full_graph_merges_via_cp_bridge() -> None:
    """Legacy full-graph mode (``sector=None``): every active branch
    contributes an edge, CP couplings included, so the two electricity
    islands collapse to one component — the property that let the
    priority_invariant claim aggregate the two coordinators' loads as a
    single group. Regression bookend for the per-sector test below.
    """
    mirror, _ = _two_island_grid()
    g = nx.Graph()
    for bid_endpoints, _ in [
        ((1, 2), "electricity"),
        ((20, 21), "electricity"),
        ((2, 10), "cp"),
        ((10, 11), "heat"),
        ((11, 20), "cp"),
    ]:
        g.add_edge(*bid_endpoints)
    components = list(nx.connected_components(g))
    assert len(components) == 1, (
        "legacy full-graph metric merges via CP bridges → one component; "
        f"got {len(components)}: {components}"
    )
    big = components[0]
    assert {1, 2, 20, 21}.issubset(big)


def test_metrics_active_components_per_sector_splits_electricity() -> None:
    """``_active_node_components(net, sector="electricity")`` excludes CP
    couplings, so the two electricity islands stay split — matching the
    L2 coordinator-election scope the priority_invariant aggregator now
    groups by. Stub network = two electricity islands + a CP/heat bridge.
    """
    # pylint: disable=import-outside-toplevel
    from experiment.eval.metrics import _active_node_components

    class _StubGrid:
        def __init__(self, name: str) -> None:
            # sector_from_grid picks the sector from grid.name by substring.
            self.name = name

    class _StubNode:
        def __init__(self, nid: int, grid_name: str) -> None:
            self.id = nid
            self.grid = _StubGrid(grid_name)

    class _StubModel:
        def __init__(self, is_cp: bool) -> None:
            self._cp = is_cp
            self.active = True

        def is_cp(self) -> bool:
            return self._cp

    class _StubBranch:
        def __init__(self, bid: tuple, is_cp: bool) -> None:
            self.id = bid
            self.active = True
            self.model = _StubModel(is_cp)

    class _StubNet:
        def __init__(self) -> None:
            self.nodes = [
                _StubNode(1, "power_grid"),
                _StubNode(2, "power_grid"),
                _StubNode(10, "water_grid"),
                _StubNode(11, "water_grid"),
                _StubNode(20, "power_grid"),
                _StubNode(21, "power_grid"),
            ]
            self.branches = [
                _StubBranch((1, 2, 0), is_cp=False),  # el line
                _StubBranch((20, 21, 0), is_cp=False),  # el line
                _StubBranch((2, 10, 0), is_cp=True),  # CP bridge (el<->heat)
                _StubBranch((10, 11, 0), is_cp=False),  # heat pipe
                _StubBranch((11, 20, 0), is_cp=True),  # CP bridge
            ]
            self._by_id = {n.id: n for n in self.nodes}

        def node_by_id(self, nid: int) -> _StubNode:
            return self._by_id[nid]

    net = _StubNet()
    legacy = _active_node_components(net)  # sector=None
    per_el = _active_node_components(net, sector="electricity")

    # Legacy: CP bridges merge everything into one component.
    assert len(set(legacy.values())) == 1, (
        f"legacy full-graph view should merge via CP; got {legacy}"
    )
    assert per_el[1] == per_el[2], (
        f"island A nodes 1 and 2 must share a component; got {per_el}"
    )
    assert per_el[20] == per_el[21], (
        f"island B nodes 20 and 21 must share a component; got {per_el}"
    )
    assert per_el[1] != per_el[20], (
        "electricity-only view must split islands A and B (the L2 "
        f"coordinator-election scope); got {per_el}"
    )


def test_component_allocation_carries_monotone_version_field() -> None:
    """``ComponentAllocation`` exposes ``version`` (default 0) and
    ``ComponentAdmmReport`` exposes ``last_applied_allocation_version``
    (default -1) — the implicit-ACK retry channel that makes
    fire-and-forget broadcasts reliable under packet loss.
    """
    # pylint: disable=import-outside-toplevel
    from scare.base.channel import ComponentAdmmReport, ComponentAllocation
    from scare.base.model import Sector

    alloc = ComponentAllocation(
        publisher="coord-A",
        sector=Sector.ELECTRICITY,
        service_fraction_by_tier={1: 1.0, 2: 0.5, 3: 0.0, 4: 0.0},
    )
    assert hasattr(alloc, "version")
    assert isinstance(alloc.version, int)
    assert alloc.version == 0

    # Explicit version round-trips.
    alloc2 = ComponentAllocation(
        publisher="coord-A",
        sector=Sector.ELECTRICITY,
        service_fraction_by_tier={},
        version=7,
    )
    assert alloc2.version == 7

    report = ComponentAdmmReport(
        publisher="leader-A",
        version=1,
        sector=Sector.ELECTRICITY,
    )
    assert hasattr(report, "last_applied_allocation_version")
    assert report.last_applied_allocation_version == -1

    report2 = ComponentAdmmReport(
        publisher="leader-A",
        version=1,
        sector=Sector.ELECTRICITY,
        last_applied_allocation_version=7,
    )
    assert report2.last_applied_allocation_version == 7


def test_leader_emerged_registers_promoted_orphan_aid() -> None:
    """``LeaderEmerged`` updates ``_leader_node_ids`` so the
    coordinator-election scope admits a promoted orphan leader. Without
    it, a repartition-spawned leader is unknown to every other leader
    and slack-budget overrides routed to it have no escalation path.
    """
    # pylint: disable=import-outside-toplevel
    from scare.base.model import LeaderEmerged

    mirror, leader_node_ids = _two_island_grid()
    role = _make_leader_role(
        aid="leader-A",
        my_node_id=1,
        leader_node_ids=leader_node_ids,
        mirror=mirror,
    )
    # Sanity: an unknown promoted-leader aid is not in the registry.
    assert "orphan-leader" not in role._leader_node_ids

    role._on_leader_emerged(
        LeaderEmerged(
            leader_aid="orphan-leader",
            leader_addr=_StubAddr("orphan-leader"),
            node_id=42,
            sector=Sector.ELECTRICITY,
        )
    )
    assert role._leader_node_ids.get("orphan-leader") == 42

    # Idempotent: re-applying the same emergence is a no-op.
    role._on_leader_emerged(
        LeaderEmerged(
            leader_aid="orphan-leader",
            leader_addr=_StubAddr("orphan-leader"),
            node_id=42,
            sector=Sector.ELECTRICITY,
        )
    )
    assert role._leader_node_ids.get("orphan-leader") == 42

    # Empty aid (defensive): ignored without raising.
    role._on_leader_emerged(
        LeaderEmerged(
            leader_aid="",
            leader_addr=_StubAddr(""),
            node_id=99,
            sector=Sector.ELECTRICITY,
        )
    )
    assert "" not in role._leader_node_ids


def test_resend_allocation_targets_only_stale_leader() -> None:
    """``ComponentCoordinator.resend_if_stale`` re-sends only to a leader whose
    echoed ``applied_version`` is behind the coordinator's latest
    counter, and skips leaders that are caught up (no O(N) re-broadcast).
    """
    # pylint: disable=import-outside-toplevel
    import asyncio

    from scare.base.channel import ComponentAllocation
    from scare.base.model import Sector

    mirror, leader_node_ids = _two_island_grid()
    role = _make_leader_role(
        aid="leader-A",
        my_node_id=1,
        leader_node_ids=leader_node_ids,
        mirror=mirror,
    )
    # Patch the stub context with a send-recording capability.
    sent: list[tuple[Any, Any]] = []

    async def _send(msg, receiver_addr):
        sent.append((msg, receiver_addr))

    role._context.send_message = _send  # type: ignore[attr-defined]
    # Stub out the diagnostics recorder, irrelevant here.
    role._record_event = lambda *args, **kwargs: None  # type: ignore[assignment]

    # Coordinator state as if version 3 had been dispatched.
    role._component.allocation_version_counter = 3
    role._component.last_dispatched_allocation = ComponentAllocation(
        publisher="leader-A",
        sector=Sector.ELECTRICITY,
        service_fraction_by_tier={1: 1.0, 2: 0.5, 3: 0.0, 4: 0.0},
        version=3,
    )
    leader_b_addr = _StubAddr("leader-B")

    # Stale leader echoed version=1 — should trigger a re-send.
    asyncio.run(role._component.resend_if_stale("leader-B", leader_b_addr, 1))
    assert len(sent) == 1, f"expected one re-send to leader-B, got {sent}"
    resent_msg, resent_addr = sent[0]
    assert resent_addr is leader_b_addr
    assert isinstance(resent_msg, ComponentAllocation)
    assert resent_msg.version == 3

    # Caught-up leader (applied_version == latest) — no re-send.
    sent.clear()
    asyncio.run(role._component.resend_if_stale("leader-B", leader_b_addr, 3))
    assert sent == []

    # Ahead-of-coordinator (shouldn't happen but defensive) — no re-send.
    sent.clear()
    asyncio.run(role._component.resend_if_stale("leader-B", leader_b_addr, 99))
    assert sent == []

    # Coordinator's own seat — never re-send to self.
    sent.clear()
    asyncio.run(role._component.resend_if_stale("leader-A", role._context.addr, 0))
    assert sent == []

    # No stashed allocation — degenerate first-round case.
    sent.clear()
    role._component.last_dispatched_allocation = None
    asyncio.run(role._component.resend_if_stale("leader-B", leader_b_addr, 0))
    assert sent == []


def test_l2_splits_two_coordinators_on_cp_bridged_islands() -> None:
    """L2 elects two coordinators on a grid where two electricity
    islands are bridged ONLY by a CP/heat chain: each leader queries
    ``reachable_from(my_node, sector=ELECTRICITY)``, gets a disjoint peer
    set, and picks itself (lex-smallest). This is correct once the metric
    aggregates per (sector, sector-subgraph component) rather than
    merging the islands via the CP chain.
    """
    mirror, leader_node_ids = _two_island_grid()

    role_A = _make_leader_role(
        aid="leader-A",
        my_node_id=1,
        leader_node_ids=leader_node_ids,
        mirror=mirror,
    )
    role_B = _make_leader_role(
        aid="leader-B",
        my_node_id=20,
        leader_node_ids=leader_node_ids,
        mirror=mirror,
    )
    peer_addrs = {
        "leader-A": _StubAddr("leader-A"),
        "leader-B": _StubAddr("leader-B"),
    }
    _patch_sector_peers(role_A, peer_addrs)
    _patch_sector_peers(role_B, peer_addrs)

    coord_A = role_A._component_coordinator_aid()
    coord_B = role_B._component_coordinator_aid()

    # Each leader is lex-smallest in its own sector-only peer set.
    assert coord_A == "leader-A"
    assert coord_B == "leader-B"
    assert coord_A != coord_B
