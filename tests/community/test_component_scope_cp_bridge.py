"""Regression test for the L2 component-scope vs priority-invariant
metric mismatch surfaced by eval_full_small task 88.

Bug (proven against the run artefacts at
``experiment/_runs/eval_full_small_20260529-181310/tasks/000088/``):

* Failures (14,5), (22,16), (106,94) split the simbench_lv_medium
  electricity feeder into multiple electricity-only components.
* ``experiment/eval/metrics.py:_active_node_components`` builds a graph
  from *all* active branches — including CP coupling branches that
  monee models as ``branch.model.is_cp() == True`` (see
  ``src/scare/scenario/restoration.py:124-142`` for the sector resolver
  that tags those as ``"cp"``).  So electricity-only-split nodes still
  share a single monee component index via the surviving CP→heat→CP
  chain.
* ``src/scare/community/holonic.py:2452`` (``_resolve_component_peer_addrs``)
  queries the topology mirror with ``sector=Sector.ELECTRICITY`` —
  electricity-only edges, NO CP bridges (the mirror explicitly rejects
  ``allow_cp_bridges`` when ``sector is not None``; see
  ``src/scare/base/topology_mirror.py:187``).
* Result: two coordinators (lex-smallest aid per electricity-only
  component) decide independently; the gating
  ``priority_invariant`` claim (``experiment/eval/claims.py:269``)
  groups loads by the monee full-graph index and reports a tier
  inversion across the coordinator boundary.

This test sets up the minimal grid that triggers the mismatch and
asserts the property the priority_invariant claim requires
(ONE coordinator across the merged-graph component).  It is expected
to FAIL until the L2 scope and the metric scope are aligned (see
the bug report for the two proposed fixes).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import networkx as nx
import pytest

from scare.base.model import Sector
from scare.base.topology_mirror import GridTopologyMirror
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
        # Some HolonicCommunityRole paths (e.g.
        # ``_on_leader_emerged``'s diagnostic ``record_event`` call)
        # touch the sim clock; mango sets this on real RoleContexts
        # but our stub has no scheduler.  0.0 is the natural default
        # for unit tests that drive the role directly.
        self.current_timestamp = 0.0


def _make_leader_role(
    *,
    aid: str,
    my_node_id: int,
    leader_node_ids: dict[str, int],
    mirror: GridTopologyMirror,
) -> HolonicCommunityRole:
    """Construct a HolonicCommunityRole stub wired with the bits the
    component-scope path actually reads.

    The role's `_resolve_sector_peer_addrs` walks
    `topology_neighbors(self, tid=...)` — without a mango runtime that
    raises and the method falls back to ``{self.context.aid: self.context.addr}``.
    We force the unfiltered baseline by populating ``_holon_member_addrs``-
    independent data via a monkeypatch below.
    """
    role = HolonicCommunityRole(
        sector=Sector.ELECTRICITY,
        my_node_id=my_node_id,
        leader_node_ids=leader_node_ids,
        topology_mirror=mirror,
        admm_scope="component",
    )
    role._context = _StubContext(aid)  # mango.agent.role.Role.context is a property over _context
    return role


def _patch_sector_peers(role: HolonicCommunityRole, peer_addrs: dict[str, _StubAddr]) -> None:
    """Force the unfiltered sector-peer set.  In the live system this is
    built from the holon_summary_<sector> mesh; in the test we inject it
    directly so the topology mirror filter is the only variable.
    """
    role._resolve_sector_peer_addrs = lambda: dict(peer_addrs)  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Fixture: 2 electricity islands joined by a CP bridge.
# Mirrors the task-88 topology in miniature.
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
# After: no electricity-only path connects {el-1, el-2} and {el-20, el-21};
# but a path through the CP/heat chain joins all of them, so monee's
# full-graph metric (which admits every branch including CP) sees ONE
# component.
# ---------------------------------------------------------------------------


def _two_island_grid() -> tuple[GridTopologyMirror, dict[str, int]]:
    branches = {
        ("a1",): (1, 2),         # el
        ("a2",): (20, 21),       # el
        ("cp1",): (2, 10),       # cp bridge (el <-> heat)
        ("h",): (10, 11),        # heat
        ("cp2",): (11, 20),      # cp bridge (heat <-> el)
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


# ---------------------------------------------------------------------------
# Sanity: mirror semantics match the bug.
# ---------------------------------------------------------------------------


def test_mirror_electricity_only_view_is_split() -> None:
    mirror, _ = _two_island_grid()
    # Electricity-only reachability: each island is isolated.
    reach_A = mirror.reachable_from(1, sector=Sector.ELECTRICITY)
    reach_B = mirror.reachable_from(20, sector=Sector.ELECTRICITY)
    assert reach_A == {1, 2}
    assert reach_B == {20, 21}
    assert reach_A.isdisjoint(reach_B)


def test_metrics_active_components_legacy_full_graph_merges_via_cp_bridge() -> None:
    """Replay the LEGACY ``_active_node_components`` behaviour
    (``sector=None``, the original full-graph mode): every active branch
    contributes an edge, CP couplings included, so the two electricity
    islands collapse to one component.

    This is the property that made the priority_invariant claim
    aggregate leader-A's and leader-B's loads as one (sector,
    component) group while L2 had split them into independent
    coordinators.  Kept here as the regression bookend — see the next
    test for the post-fix per-sector behaviour.
    """
    mirror, _ = _two_island_grid()
    # ``_active_node_components`` iterates ``monee_net.branches`` and adds
    # an edge for every active branch.  CP branches in monee carry
    # ``branch.model.is_cp() == True`` but still appear in ``branches``;
    # the legacy helper added them unconditionally.
    g = nx.Graph()
    for (bid_endpoints, _) in [
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
    """Post-fix behaviour of ``_active_node_components(monee_net,
    sector="electricity")``: CP couplings are excluded, so the two
    electricity islands stay split — matching the L2 coordinator-
    election scope.  ``served_by_load`` now stamps each row's
    ``component`` from this sector-specific map, so the
    ``priority_invariant`` aggregator groups loads by the same scope
    L2 actually arbitrates over.

    Uses a stub network that mirrors the simbench_lv_medium task-88
    topology (two electricity islands + a CP/heat bridge).
    """
    # pylint: disable=import-outside-toplevel
    from experiment.eval.metrics import _active_node_components

    class _StubGrid:
        def __init__(self, name: str) -> None:
            # ``sector_from_grid`` reads ``grid.name`` (lower-cased) and
            # picks ELECTRICITY/GAS/HEAT by substring.
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
            # Electricity grid -> Sector.ELECTRICITY via sector_from_grid.
            self.nodes = [
                _StubNode(1, "power_grid"),
                _StubNode(2, "power_grid"),
                _StubNode(10, "water_grid"),
                _StubNode(11, "water_grid"),
                _StubNode(20, "power_grid"),
                _StubNode(21, "power_grid"),
            ]
            self.branches = [
                _StubBranch((1, 2, 0), is_cp=False),    # el line
                _StubBranch((20, 21, 0), is_cp=False),  # el line
                _StubBranch((2, 10, 0), is_cp=True),    # CP bridge (el<->heat)
                _StubBranch((10, 11, 0), is_cp=False),  # heat pipe
                _StubBranch((11, 20, 0), is_cp=True),   # CP bridge
            ]
            self._by_id = {n.id: n for n in self.nodes}

        def node_by_id(self, nid: int) -> _StubNode:
            return self._by_id[nid]

    net = _StubNet()
    legacy = _active_node_components(net)               # sector=None
    per_el = _active_node_components(net, sector="electricity")

    # Legacy: one big component (CP bridges merge everything).
    assert len(set(legacy.values())) == 1, (
        f"legacy full-graph view should merge via CP; got {legacy}"
    )
    # Per-sector electricity: nodes 1,2 form one component;
    # 20,21 another.  Heat-only nodes (10,11) appear as singletons.
    el_comps_for_loads = {legacy_node: per_el[legacy_node] for legacy_node in (1, 2, 20, 21)}
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


# ---------------------------------------------------------------------------
# The actual regression: L2 elects 2 coordinators across the merged-graph
# component.  This is what the priority_invariant claim cannot tolerate.
# ---------------------------------------------------------------------------


def test_component_allocation_carries_monotone_version_field() -> None:
    """``ComponentAllocation`` must expose a ``version`` integer (default
    ``0``) and ``ComponentAdmmReport`` must expose a
    ``last_applied_allocation_version`` integer (default ``-1``).
    Together they implement the implicit-ACK retry channel that turns
    fire-and-forget broadcasts into reliable dispatches under packet
    loss — see ``channel.ComponentAllocation`` docstring (task 52).
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

    # Set explicit version — ensure the field round-trips.
    alloc2 = ComponentAllocation(
        publisher="coord-A",
        sector=Sector.ELECTRICITY,
        service_fraction_by_tier={},
        version=7,
    )
    assert alloc2.version == 7

    report = ComponentAdmmReport(
        publisher="leader-A", version=1, sector=Sector.ELECTRICITY,
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
    coordinator-election scope (``_resolve_component_peer_addrs``)
    admits the promoted orphan leader.

    Pre-fix manifestation: in task 88, ``community_repartitioned``
    spawns a new leader ``child-25`` that never appears as ``leader=``
    in any ``component_report_sent`` because no leader knows its aid;
    the slack-budget override routed to it has no L2/L3 escalation
    path and the breach plateaus 10.6% over budget.
    """
    # pylint: disable=import-outside-toplevel
    from scare.base.model import LeaderEmerged

    mirror, leader_node_ids = _two_island_grid()
    role = _make_leader_role(
        aid="leader-A", my_node_id=1,
        leader_node_ids=leader_node_ids, mirror=mirror,
    )
    # Sanity: an unknown promoted-leader aid is not in the registry.
    assert "orphan-leader" not in role._leader_node_ids

    role._on_leader_emerged(LeaderEmerged(
        leader_aid="orphan-leader",
        leader_addr=_StubAddr("orphan-leader"),
        node_id=42,
        sector=Sector.ELECTRICITY,
    ))
    assert role._leader_node_ids.get("orphan-leader") == 42

    # Idempotent: re-applying the same emergence (e.g. a retransmit)
    # is a no-op and doesn't double-fire the diagnostic event.
    role._on_leader_emerged(LeaderEmerged(
        leader_aid="orphan-leader",
        leader_addr=_StubAddr("orphan-leader"),
        node_id=42,
        sector=Sector.ELECTRICITY,
    ))
    assert role._leader_node_ids.get("orphan-leader") == 42

    # Empty aid (defensive): ignored without raising.
    role._on_leader_emerged(LeaderEmerged(
        leader_aid="",
        leader_addr=_StubAddr(""),
        node_id=99,
        sector=Sector.ELECTRICITY,
    ))
    assert "" not in role._leader_node_ids


def test_resend_allocation_targets_only_stale_leader() -> None:
    """``_resend_allocation_if_stale`` re-sends to a leader whose
    echoed ``applied_version`` is behind the coordinator's latest
    counter, and skips leaders that are caught up.  The retry
    targets only the stale leader (not a fresh broadcast), so a
    benign duplicate doesn't cost full O(N) under high
    coordinator-side report churn.

    Driven directly against ``HolonicCommunityRole`` to avoid the
    mango runtime.  Records the message sends on a stub context.
    """
    # pylint: disable=import-outside-toplevel
    import asyncio

    from scare.base.channel import ComponentAllocation
    from scare.base.model import Sector

    mirror, leader_node_ids = _two_island_grid()
    role = _make_leader_role(
        aid="leader-A", my_node_id=1,
        leader_node_ids=leader_node_ids, mirror=mirror,
    )
    # Patch the stub context with a send-recording capability.
    sent: list[tuple[Any, Any]] = []

    async def _send(msg, receiver_addr):
        sent.append((msg, receiver_addr))

    role._context.send_message = _send  # type: ignore[attr-defined]
    # Diagnostics event recorder shim: HolonicCommunityRole's
    # ``_record_event`` calls ``record_event``, which writes to the
    # behavior — irrelevant for this test; intercept it.
    role._record_event = lambda *args, **kwargs: None  # type: ignore[assignment]

    # Pre-populate coordinator state as if we'd dispatched version 3.
    role._allocation_version_counter = 3
    role._last_dispatched_allocation = ComponentAllocation(
        publisher="leader-A",
        sector=Sector.ELECTRICITY,
        service_fraction_by_tier={1: 1.0, 2: 0.5, 3: 0.0, 4: 0.0},
        version=3,
    )
    leader_b_addr = _StubAddr("leader-B")

    # Stale leader echoed version=1 — should trigger a re-send.
    asyncio.run(role._resend_allocation_if_stale("leader-B", leader_b_addr, 1))
    assert len(sent) == 1, f"expected one re-send to leader-B, got {sent}"
    resent_msg, resent_addr = sent[0]
    assert resent_addr is leader_b_addr
    assert isinstance(resent_msg, ComponentAllocation)
    assert resent_msg.version == 3

    # Caught-up leader (applied_version == latest) — no re-send.
    sent.clear()
    asyncio.run(role._resend_allocation_if_stale("leader-B", leader_b_addr, 3))
    assert sent == []

    # Ahead-of-coordinator (shouldn't happen but defensive) — no re-send.
    sent.clear()
    asyncio.run(role._resend_allocation_if_stale("leader-B", leader_b_addr, 99))
    assert sent == []

    # Coordinator's own seat — never re-send to self.
    sent.clear()
    asyncio.run(role._resend_allocation_if_stale("leader-A", role._context.addr, 0))
    assert sent == []

    # No stashed allocation — degenerate first-round case.
    sent.clear()
    role._last_dispatched_allocation = None
    asyncio.run(role._resend_allocation_if_stale("leader-B", leader_b_addr, 0))
    assert sent == []


def test_l2_splits_two_coordinators_on_cp_bridged_islands() -> None:
    """Documents the L2 coordinator-election behaviour on a CP-bridged
    grid.

    Each leader queries ``mirror.reachable_from(my_node,
    sector=Sector.ELECTRICITY)`` — electricity only.  When two
    electricity islands are bridged ONLY by a CP/heat chain, this
    returns two disjoint peer sets and **two coordinators are
    elected** (each leader picks itself as lex-smallest in its own
    peer set).

    This is the EXPECTED L2 behaviour after Bug 1 is fixed in the
    metric: the priority_invariant claim now aggregates per
    (sector, *sector-subgraph* component) so two coordinators making
    independent per-tier decisions is correct and does not surface as
    a spurious tier inversion.  Before the fix, the metric merged the
    two sub-areas via the CP chain, and the divergent per-tier
    fractions of the two coordinators looked like a priority order
    violation.

    Live-system manifestation (pre-fix):
    ``experiment/_runs/eval_full_small_20260529-181310/tasks/000088``.
    """
    mirror, leader_node_ids = _two_island_grid()

    role_A = _make_leader_role(
        aid="leader-A", my_node_id=1,
        leader_node_ids=leader_node_ids, mirror=mirror,
    )
    role_B = _make_leader_role(
        aid="leader-B", my_node_id=20,
        leader_node_ids=leader_node_ids, mirror=mirror,
    )
    peer_addrs = {
        "leader-A": _StubAddr("leader-A"),
        "leader-B": _StubAddr("leader-B"),
    }
    _patch_sector_peers(role_A, peer_addrs)
    _patch_sector_peers(role_B, peer_addrs)

    coord_A = role_A._component_coordinator_aid()
    coord_B = role_B._component_coordinator_aid()

    # Each leader is the lex-smallest in its own (sector-only) peer
    # set; the two coordinators are intentionally distinct.
    assert coord_A == "leader-A"
    assert coord_B == "leader-B"
    assert coord_A != coord_B
