"""Multi-sector L3 component view + per-CP setpoint heuristic.

Extracted from :class:`scare.service.cp.EnergyConverterRole`. ``CPComponentView``
owns the post-construction wiring the multi-sector L3 (Option B) path needs —
the shared topology mirror, this CP's host node, and the CP/leader address maps
— and answers reachability / coordinator-election queries scoped to the CP's
live multi-sector connected component. ``compute_cp_setpoint`` is the pure
marginal-gradient heuristic that turns per-sector L3 marginals into one CP's
flow setpoint. Both are testable without a mango context.
"""

from __future__ import annotations

import logging
from typing import Any

from scare.base.model import Sector

logger = logging.getLogger(__name__)


class CPComponentView:
    """The multi-sector L3 "who is in my component" view for one CP.

    The role injects wiring via :meth:`wire` (called from
    ``EnergyConverterRole.wire_multi_sector_l3`` after the topology mirror and
    every CP agent are built); before wiring, :meth:`enabled` is False and the
    role falls back to the legacy per-CP path. Queries take the caller's ``aid``
    explicitly since it is only known at runtime from the mango context.
    """

    def __init__(self) -> None:
        self.topology_mirror: Any = None
        self.my_node_id: Any = None
        # {cp_aid: {sectors, capacity_mw, coupling_ratios, addr, node_id}}.
        self.cp_meta_by_aid: dict[str, dict[str, Any]] = {}
        # {Sector: {leader_aid: addr}}.
        self.leader_addrs_by_sector: dict[Sector, dict[str, Any]] = {}
        # {leader_aid: node_id} for reachability filtering.
        self.leader_node_ids: dict[str, Any] = {}

    def wire(
        self,
        *,
        topology_mirror: Any,
        my_node_id: Any,
        cp_meta_by_aid: dict[str, dict[str, Any]],
        leader_addrs_by_sector: dict[Sector, dict[str, Any]],
        leader_node_ids: dict[str, Any],
    ) -> None:
        self.topology_mirror = topology_mirror
        self.my_node_id = my_node_id
        self.cp_meta_by_aid = dict(cp_meta_by_aid)
        self.leader_addrs_by_sector = dict(leader_addrs_by_sector)
        self.leader_node_ids = dict(leader_node_ids)

    def enabled(self) -> bool:
        """True iff this CP has the runtime state to drive the multi-sector L3
        path; False falls back to legacy per-CP (e.g. tests building the role
        without wiring the mirror)."""
        return (
            self.topology_mirror is not None
            and self.my_node_id is not None
            and bool(self.cp_meta_by_aid)
        )

    def reachable(self, aid: Any) -> set:
        """Node ids in this CP's multi-sector connected component: nodes
        mutually reachable on the active branch subgraph AND through active CP
        bridges. Scopes leader-flex collection and CP-allocation dispatch to the
        right physical island.
        """
        try:
            return self.topology_mirror.reachable_from(
                self.my_node_id, sector=None, allow_cp_bridges=True,
            )
        except Exception as exc:
            logger.debug("[%s] multi-sector reachable_from failed: %s", aid, exc)
            return {self.my_node_id}

    def cp_peers(self, aid: Any) -> dict[str, dict[str, Any]]:
        """{cp_aid: meta} for every CP whose host node is in this CP's
        multi-sector component. Always includes self so the lex-smallest aid is
        well-defined even for a singleton component.
        """
        reachable = self.reachable(aid)
        out: dict[str, dict[str, Any]] = {}
        for a, meta in self.cp_meta_by_aid.items():
            node = meta.get("node_id")
            if a == aid or node is None or node in reachable:
                out[a] = meta
        if aid not in out and aid in self.cp_meta_by_aid:
            out[aid] = self.cp_meta_by_aid[aid]
        return out

    def leader_addrs(self, aid: Any) -> dict[Sector, dict[str, Any]]:
        """{Sector: {leader_aid: addr}} filtered to leaders reachable from this
        CP on the active multi-sector subgraph. The L3 coord asks these for
        flex; like L2's per-component peer set but widened across every sector.
        """
        reachable = self.reachable(aid)
        out: dict[Sector, dict[str, Any]] = {}
        for sector, table in self.leader_addrs_by_sector.items():
            sec_out: dict[str, Any] = {}
            for a, addr in table.items():
                node = self.leader_node_ids.get(a)
                if node is not None and node in reachable:
                    sec_out[a] = addr
            if sec_out:
                out[sector] = sec_out
        return out

    def is_coordinator(self, aid: Any) -> bool:
        """True iff this CP has the lex-smallest aid among CPs in its
        multi-sector component. The L3 coord drives the joint ADMM; other CPs in
        the component wait for CPAllocation. If this CP isn't in the cp_meta
        table, returns True only for the trivial {self} case.
        """
        peers = self.cp_peers(aid)
        if not peers:
            return True
        return min(peers) == aid


def compute_cp_setpoint(
    cp_meta: dict[str, Any],
    marginal_by_sector: dict[str, float],
) -> dict[str, float]:
    """Pick a setpoint for one CP from the per-sector L3 marginal values.

    Heuristic gradient step: for each (in_sector, out_sector) coupling pair, run
    the CP iff marginal_out × ratio > marginal_in (conversion relieves a
    more-stressed cell at a less-stressed one's cost). Magnitude = capacity ×
    max(0, marginal_out × ratio − marginal_in), so a balanced pair gives a zero
    step (no commitment in zero-deficit scenarios).

    Returns {sector_value: signed_flow_mw}. Sign: positive = flow into sector
    (CP consumes/load-like), negative = flow out (produces). Matches
    CPSetpoint.sector_flows_mw so _apply_result consumes it unchanged.
    """
    capacity = float(cp_meta.get("capacity_mw") or 0.0)
    if capacity <= 0:
        return {}
    ratios = cp_meta.get("coupling_ratios") or {}
    if not ratios:
        return {}

    best_in: str | None = None
    best_out: str | None = None
    best_step: float = 0.0
    best_ratio: float = 1.0
    for (sec_in, sec_out), ratio in ratios.items():
        try:
            r = float(ratio)
        except (TypeError, ValueError):
            continue
        m_in = float(marginal_by_sector.get(str(sec_in), 0.0))
        m_out = float(marginal_by_sector.get(str(sec_out), 0.0))
        step = m_out * r - m_in
        if step > best_step:
            best_step = step
            best_in = str(sec_in)
            best_out = str(sec_out)
            best_ratio = r

    if best_in is None or best_step <= 0.0:
        return {}

    # step ∈ (0, 1] (marginals clipped to [0, 1]) so capacity × step ∈ (0, capacity].
    magnitude = capacity * min(1.0, best_step)
    return {
        best_in: float(magnitude),                 # consume from source
        best_out: -float(magnitude * best_ratio),  # produce into destination
    }
