"""Post-hoc outcome metrics computed from a finished restoration run.

Pulls final regulation factors and recorded timeseries from the world /
monee net. Outputs are plain scalars / nested dicts, serialisable straight
into ``result.json``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import networkx as nx
from mango.simulation.world import WorldRecording
from monee.model.child import ExtPowerGrid, HeatLoad, PowerLoad, Sink
from monee.model.core import value as _mvalue
from monee.model.grid import (
    DEFAULT_GAS_HHV_KWH_PER_KG,
    KGPS_KWHPERKG_TO_MW,
    GasGrid,
)
from monee.model.multi import (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    GasToHeatHG,
    GasToPower,
    PowerToGas,
    PowerToHeatControlNode,
    PowerToHeatHG,
)
from monee.solver.core import find_ignored_nodes

from scare.base.model import (
    DEENERGISED_PRESSURE_HIGH_PU,
    DEENERGISED_PRESSURE_PU,
    DEENERGISED_VM_PU,
    SECTOR_CONSTRAINTS,
    Sector,
)
from scare.base.util import (
    constraint_allowed_fraction,
    constraint_utilization,
    obs_capacity,
    obs_priority,
    obs_setpoint,
    sector_from_grid,
)

logger = logging.getLogger(__name__)


# Priority-tier weight schedule for the served-MAGNITUDE metric (PWSF):
# moderate geometric ``w(tier) = 2^(Pi - tier)`` over Pi=4 tiers => {1:8, 2:4,
# 3:2, 4:1} (8:1 across the range). Deliberately NOT a strict-priority weight
# (e.g. 1e12/1e8/1e4/1): a tier-1-dominant weight collapses PWSF to ~the tier-1
# fraction (since tier-1 is ~always served), erasing the tier 2-4 differences
# that distinguish configs. PWSF answers "how much priority-weighted load was
# served" (resolved across tiers); the SEPARATE gating ``priority_invariant``
# claim answers "was it served in priority order" — order is policed there, not
# by collapsing this scalar. PWSF is a weighted RATIO so only inter-tier ratios
# matter (this is identical to the legacy 2^(10-tier)). Tier 0 (generators)
# weighs 0; tiers clamp to [1, Pi].
_PI = 4


def _tier_weight(tier: int) -> float:
    if tier <= 0:
        return 0.0
    t = min(int(tier), _PI)
    return 2.0 ** (_PI - t)


# Final served fractions


def _disconnected_node_ids(monee_net: Any) -> set[int]:
    """Node IDs with no path to any grid-forming component through the active
    branch topology. Their loads are physically un-servable and must count as
    zero served, regardless of what the LP or agents report (a disconnected
    load can otherwise keep a default regulation of 1.0 and report
    ``served = cap``).

    The islanding config MUST be forwarded when present: the solve path
    passes it (a GridForming-anchored island is servable there), so grading
    without it would zero exactly the load the islanding extension restores.
    """
    try:
        return set(
            find_ignored_nodes(
                monee_net, getattr(monee_net, "islanding_config", None)
            )
        )
    except Exception:
        return set()


_ENERGISATION_KEYS = ("e_el", "e_gas", "e_water")


def _is_deenergised(obs: dict, node: Any) -> bool:
    """Whether the islanding MILP de-energised this node (energisation binary
    solved to 0). The binaries live on the solved node model; MAS observers
    merge node values into ``obs``, the oracle adapter does not — fall back to
    the node model for the latter (where the net IS the solved net)."""
    node_vals = None
    for key in _ENERGISATION_KEYS:
        v = obs.get(key)
        if v is None:
            if node_vals is None:
                node_vals = dict(getattr(node.model, "values", {}) or {})
            v = node_vals.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v < 0.5:
            return True
    return False


def _branch_carries_sector(branch: Any, monee_net: Any, sector: str | None) -> bool:
    """Whether this branch is admitted into the component subgraph for
    ``sector``.

    ``sector is None`` admits every branch (full-graph view). A named sector
    admits only same-sector edges; CP couplings (``model.is_cp()``) are
    excluded, matching the L2 coordinator-election scope (sector subgraph
    without CP bridges). This keeps the priority-invariant metric on the same
    partition the control uses, so two electricity sub-islands bridged only by
    a CP->heat->CP chain are not spuriously merged.
    """
    if sector is None:
        return True
    model = getattr(branch, "model", None)
    if model is not None and getattr(model, "is_cp", lambda: False)():
        return False
    # Same-sector check via one endpoint's grid -> sector.
    a = branch.id[0]
    try:
        node = monee_net.node_by_id(a)
    except Exception:
        return False
    sec = sector_from_grid(getattr(node, "grid", None))
    return sec is not None and sec.value == sector


def _active_node_components(
    monee_net: Any,
    *,
    sector: str | None = None,
) -> dict[Any, int]:
    """Map ``node_id -> component_index`` over the active-branch subgraph
    (failed branches removed), capturing the post-failure islands within
    which the priority-invariant check compares tiers.

    ``sector``: when set, restrict to same-sector branches (CP couplings
    excluded); ``None`` is the full-graph view. Disconnected single nodes get
    their own component. Indices are arbitrary but stable within a call.
    """
    graph = nx.Graph()
    for node in monee_net.nodes:
        graph.add_node(node.id)
    for branch in monee_net.branches:
        if not _branch_is_active(branch):
            continue
        if not _branch_carries_sector(branch, monee_net, sector):
            continue
        a, b = branch.id[0], branch.id[1]
        graph.add_edge(a, b)
    out: dict[Any, int] = {}
    for idx, comp in enumerate(nx.connected_components(graph)):
        for node_id in comp:
            out[node_id] = idx
    return out


def _component_label(comp_idx: int, sector: str | None) -> str:
    """Component label for the CSV ``component`` column. Sector-aware labels
    are prefixed (``"electricity:0"``) so the ``priority_invariant`` aggregator
    groups per-sector; ``sector is None`` yields the bare index.
    """
    if sector is None:
        return str(comp_idx)
    return f"{sector}:{comp_idx}"


def _branch_is_active(branch: Any) -> bool:
    """``branch.model.active`` when present, else ``branch.active`` — the same
    flag the simulator writes to."""
    model = getattr(branch, "model", None)
    if model is not None and "active" in getattr(model, "vars", {}):
        return bool(model.active)
    return bool(getattr(branch, "active", True))


# Heat served at a node whose temperature is outside the oracle's hard t_k
# bounds is not valid service. The live sim enforces t_k only softly, so this
# gate prevents crediting heat served at infeasible temperatures. Tolerance
# absorbs numerical wiggle at the bound.
_HEAT_T_TOL_K: float = 1.0


def _heat_served_feasible(obs: dict, sec: Any) -> bool:
    """False iff a heat load's node temperature is outside the oracle's hard
    ``t_k`` bounds (within ``_HEAT_T_TOL_K``). Non-heat loads, missing /
    non-finite readings, and configs without a heat ``t_k`` bound all pass."""
    if sec is not Sector.HEAT:
        return True
    bounds = SECTOR_CONSTRAINTS.get(Sector.HEAT, {}).get("t_k")
    if bounds is None:
        return True
    t = obs.get("t_k")
    try:
        t = float(t)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(t):
        return True
    lo, hi = bounds
    return (lo - _HEAT_T_TOL_K) <= t <= (hi + _HEAT_T_TOL_K)


# Line-side analogue of the heat gate: the oracle enforces loading_percent
# <= 100% and sheds through-load until lines sit at rating, while the live sim
# leaves lines at 100-120% and credits the downstream load in full. Tolerance
# absorbs numerical wiggle at exactly 100%.
_LINE_LOADING_TOL_PCT: float = 1.0


def _line_feasibility_factor(monee_net: Any) -> dict[Any, float]:
    """Map ``electricity node_id -> feasible-service factor in (0, 1]``.

    The factor is the bottleneck headroom of the *widest* supply path back to
    a grid-former: ``min(1, 100/loading)`` over the overloaded lines on that
    path (1.0 if none). Multiplying a load's served by its node's factor
    de-rates downstream service to what line ratings permit, matching the
    oracle. Radial feeders give the unique path; meshed sections the widest.
    """
    from collections import defaultdict, deque

    slack = {c.node_id for c in monee_net.childs if isinstance(c.model, ExtPowerGrid)}
    if not slack:
        return {}
    adj: dict[Any, list[tuple[Any, float]]] = defaultdict(list)
    for branch in monee_net.branches:
        if not _branch_is_active(branch):
            continue
        if not _branch_carries_sector(branch, monee_net, "electricity"):
            continue
        a, b = branch.id[0], branch.id[1]
        # Same exact-basis grading as the violation scan (``constraint_rows``)
        # so the served de-rate and the compliance gate see identical loadings.
        pct = _branch_loading_percent(branch, monee_net)
        loading = pct if pct is not None and math.isfinite(pct) else 0.0
        edge_f = (
            min(1.0, 100.0 / loading)
            if loading > 100.0 + _LINE_LOADING_TOL_PCT
            else 1.0
        )
        adj[a].append((b, edge_f))
        adj[b].append((a, edge_f))
    # Widest-path (maximin) relaxation from the slack set.
    factor: dict[Any, float] = {s: 1.0 for s in slack}
    dq = deque(slack)
    while dq:
        n = dq.popleft()
        fn = factor[n]
        for nbr, edge_f in adj.get(n, ()):
            cand = min(fn, edge_f)
            if cand > factor.get(nbr, -1.0):
                factor[nbr] = cand
                dq.append(nbr)
    return factor


def _is_gas_consumer_sink(child: Any, monee_net: Any) -> bool:
    """True when ``child`` is a ``Sink`` on a gas grid — a real terminal gas
    consumer with shedable demand.

    Mirror/inverse of ``scenario.restoration._is_heat_side_mass_flow_sink``:
    water/heat-side Sinks close monee's supply-return loop (topology artifact,
    excluded), but gas-sector Sinks are genuine consumption and must be counted
    as served load. ``obs_capacity`` reads their ``mass_flow_kgs`` rating.
    """
    if not isinstance(child.model, Sink):
        return False
    try:
        grid_name = str(getattr(monee_net.node_by_id(child.node_id).grid, "name", "")).lower()
    except Exception:  # noqa: BLE001
        return False
    return "gas" in grid_name


def _gas_load_mw_factor(node: Any) -> float:
    """kg/s -> MW energy-content factor for a gas load on ``node``'s grid.

    Uses that grid's higher heating value (``higher_heating_value_kwh_per_kg``,
    falling back to :data:`DEFAULT_GAS_HHV_KWH_PER_KG`) and the same
    ``KGPS_KWHPERKG_TO_MW`` constant :func:`_cp_output` applies to converter gas
    throughput, so terminal gas demand enters PWSF on the identical energy basis
    as the electricity and heat MW.
    """
    hhv = getattr(getattr(node, "grid", None), "higher_heating_value_kwh_per_kg", None)
    if not (isinstance(hhv, (int, float)) and hhv > 0):
        hhv = DEFAULT_GAS_HHV_KWH_PER_KG
    return float(hhv) * KGPS_KWHPERKG_TO_MW


def served_by_load(
    monee_net: Any,
    behavior: Any,
    priorities: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Per-load served / demand / fraction with node + component tag.

    Returns a list of dicts ready for CSV-writing or per-component
    aggregation. Shape: ``[{aid, sector, tier, node_id, component, demand,
    served, fraction, disconnected, constraint_allowed}, ...]``.

    Loads on physically disconnected nodes get ``served = 0``, matching
    :func:`served_breakdown`. ``component`` is the per-sector active-subgraph
    index so the priority-invariant claim can group by it.
    """
    disconnected = _disconnected_node_ids(monee_net)
    # Per-sector component maps so each row's ``component`` matches the L2
    # coordinator-election scope (sector subgraph), not a full-graph view that
    # would merge electricity sub-islands via CP couplings.
    components_by_sector: dict[str, dict[Any, int]] = {
        sec.value: _active_node_components(monee_net, sector=sec.value)
        for sec in Sector
    }

    # PowerLoad / HeatLoad are the electricity / heat consumers; gas consumers
    # are Sink children on a gas grid (heat-side return Sinks are excluded —
    # see ``_is_gas_consumer_sink``).
    _LOAD_CLASSES: tuple[type, ...] = (HeatLoad, PowerLoad)
    line_factor = _line_feasibility_factor(monee_net)

    rows: list[dict[str, Any]] = []
    for child in monee_net.childs:
        if not (
            isinstance(child.model, _LOAD_CLASSES)
            or _is_gas_consumer_sink(child, monee_net)
        ):
            continue
        aid = f"child-{child.id}"
        obs = behavior.observe(aid) or {}
        node = monee_net.node_by_id(child.node_id)
        deenergised = _is_deenergised(obs, node)
        cap = obs_capacity(obs)
        if not (cap > 0):
            # Injection-gated de-energised load: nominal demand stays counted
            # (mirrors served_breakdown).
            if deenergised:
                cap = obs_capacity(dict(getattr(child.model, "values", {}) or {}))
            if not (cap > 0):
                continue
        is_disconnected = (
            not getattr(child, "active", True)
            or child.node_id in disconnected
            or deenergised
        )
        sp = 0.0 if is_disconnected else obs_setpoint(obs)
        # min(cap, nan) returns cap — a NaN setpoint must not credit full served.
        served = max(0.0, min(cap, sp)) if math.isfinite(sp) else 0.0
        sec = sector_from_grid(node.grid)
        if sec is None:
            continue
        # Don't credit heat served at an out-of-bounds node temperature.
        temp_infeasible = not _heat_served_feasible(obs, sec)
        if temp_infeasible:
            served = 0.0
        # De-rate electricity served through an overloaded line to the feasible
        # level (see ``_line_feasibility_factor``).
        if sec is Sector.ELECTRICITY:
            served *= line_factor.get(child.node_id, 1.0)
        # Gas demand/served are mass flow (kg/s); convert to MW energy content
        # so every row shares the electricity/heat unit (see _gas_load_mw_factor).
        if sec is Sector.GAS:
            gas_mw = _gas_load_mw_factor(node)
            cap *= gas_mw
            served *= gas_mw
        if priorities is not None and aid in priorities:
            tier = int(priorities[aid])
        else:
            tier = obs_priority(obs)
        # Physical serve cap from local constraints (same util->fraction the
        # gossip clamp uses). A load at/near this cap is throttled by physics,
        # not priority, so the priority-invariant check excludes it (as it does
        # disconnected loads). A temperature-infeasible heat load is hard-capped
        # at 0 here so it reads as constraint-throttled, not a priority shed.
        if temp_infeasible:
            allowed = 0.0
        elif is_disconnected:
            allowed = 1.0
        else:
            allowed = constraint_allowed_fraction(obs, sec, tier=tier)
        sec_components = components_by_sector.get(sec.value, {})
        comp_idx = sec_components.get(child.node_id, -1)
        rows.append(
            {
                "aid": aid,
                "sector": sec.value,
                "tier": tier,
                "node_id": child.node_id,
                "component": _component_label(comp_idx, sec.value),
                "demand": cap,
                "served": served,
                "fraction": served / cap if cap > 0 else 0.0,
                "disconnected": int(is_disconnected),
                "constraint_allowed": allowed,
            }
        )
    return rows


def served_breakdown(
    monee_net: Any,
    behavior: Any,
    priorities: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Served / demand per (sector, tier) and per-sector aggregate, plus the
    priority-weighted scalar.

    Shape:

    ``{
        "by_tier_sector": {sector: {tier: {demand, served, fraction}}},
        "by_sector":      {sector: {demand, served, fraction}},
        "by_tier":        {tier:   {demand, served, fraction, weight}}
                          (all sectors in MW; gas HHV-converted from kg/s),
        "priority_weighted_served": float,
        "priority_weighted_demand": float,
        "priority_weighted_fraction": float,
        "n_loads": int,
        "n_loads_served_zero": int
    }``

    ``priority_weighted_fraction`` divides by total weighted demand, giving a
    value in [0, 1] comparable across grids of different size. Loads on
    disconnected nodes are forced to ``served = 0``.
    """
    by_tier_sector: dict[str, dict[int, dict[str, float]]] = {}
    by_sector: dict[str, dict[str, float]] = {}
    by_tier: dict[int, dict[str, float]] = {}
    pw_by_sector: dict[str, dict[str, float]] = {}
    n_loads = 0
    n_zero = 0
    pw_demand = 0.0
    pw_served = 0.0

    disconnected = _disconnected_node_ids(monee_net)
    line_factor = _line_feasibility_factor(monee_net)
    # Restrict to consumer loads: PowerLoad / HeatLoad (electricity / heat) plus
    # gas-grid Sinks (real gas consumers, see ``_is_gas_consumer_sink``). The
    # heat-side return Sink and ExtHydrGrid / ExtPowerGrid (slack injectors)
    # carry mass-flow / p_mw fields that ``obs_capacity`` would otherwise pick
    # up, inflating served past demand — they stay excluded.
    _LOAD_CLASSES: tuple[type, ...] = (HeatLoad, PowerLoad)

    for child in monee_net.childs:
        if not (
            isinstance(child.model, _LOAD_CLASSES)
            or _is_gas_consumer_sink(child, monee_net)
        ):
            continue
        aid = f"child-{child.id}"
        obs = behavior.observe(aid) or {}
        node = monee_net.node_by_id(child.node_id)
        deenergised = _is_deenergised(obs, node)
        cap = obs_capacity(obs)
        # Skip generators (cap < 0), zero-capacity placeholders, and
        # NaN-capacity entries (NaN propagates into compound port-childs on
        # disconnected nodes and carries no consumer demand).
        if not (cap > 0):
            # Injection gating zeroes a de-energised load's setpoint on the
            # solved net — its demand is real and must stay counted as
            # unserved, so recover the nominal rating from the child model.
            if deenergised:
                cap = obs_capacity(dict(getattr(child.model, "values", {}) or {}))
            if not (cap > 0):
                continue
        n_loads += 1
        # ``is_disconnected`` distinguishes the physical loss path
        # (priority-blind: no path to a grid-forming source, or islanding
        # de-energisation) from the agent-driven path (QP / ADMM priority
        # weighting decides who sheds). restoration_breakdown uses this to
        # split per-tier losses.
        is_disconnected = (
            not getattr(child, "active", True)
            or child.node_id in disconnected
            or deenergised
        )
        if is_disconnected:
            sp = 0.0
        else:
            sp = obs_setpoint(obs)
        # Clamp to [0, cap]: a load can't consume more than rated demand, and
        # MAS variants may set regulation > 1.0. Without this, over-served loads
        # mask genuine zero-served loads in the totals. min(cap, nan) returns
        # cap, so a NaN setpoint must be zeroed, not credited full served.
        served = max(0.0, min(cap, sp)) if math.isfinite(sp) else 0.0
        demand = cap
        # Demand on disconnected nodes — the priority-blind share of the
        # per-tier loss; restoration_breakdown uses it to attribute losses.
        demand_disc = demand if is_disconnected else 0.0

        sec = sector_from_grid(node.grid)
        if sec is None:
            continue
        # Don't credit heat served at an out-of-bounds node temperature.
        served_pre_gate = served
        if served > 0.0 and not _heat_served_feasible(obs, sec):
            served = 0.0
        # De-rate electricity served through an overloaded line to the feasible
        # level (see ``_line_feasibility_factor``).
        if served > 0.0 and sec is Sector.ELECTRICITY:
            served *= line_factor.get(child.node_id, 1.0)
        # Served the agents dispatched but the feasibility gates above removed —
        # physics-throttled, not an agent shedding decision. Tracked per tier so
        # restoration_breakdown can net it out of agent_shed (mirroring the
        # constraint-throttled exclusion in claims.py's priority invariant).
        served_constraint_capped = max(0.0, served_pre_gate - served)
        # Gas demand/served are mass flow (kg/s); convert to MW energy content
        # (grid HHV x KGPS_KWHPERKG_TO_MW, the same basis _cp_output uses) so gas
        # terminal load joins the electricity/heat MW aggregate consistently.
        if sec is Sector.GAS:
            gas_mw = _gas_load_mw_factor(node)
            demand *= gas_mw
            served *= gas_mw
            demand_disc *= gas_mw
            served_constraint_capped *= gas_mw
        if priorities is not None and aid in priorities:
            tier = int(priorities[aid])
        else:
            tier = obs_priority(obs)
        w = _tier_weight(tier)

        sec_key = sec.value
        by_sector.setdefault(
            sec_key,
            {
                "demand": 0.0,
                "served": 0.0,
                "demand_disconnected": 0.0,
                "served_constraint_capped": 0.0,
            },
        )
        by_sector[sec_key]["demand"] += demand
        by_sector[sec_key]["served"] += served
        by_sector[sec_key]["demand_disconnected"] += demand_disc
        by_sector[sec_key]["served_constraint_capped"] += served_constraint_capped

        # by_tier sums MW across sectors; gas is converted to MW above, so all
        # three sectors share the bucket.
        by_tier.setdefault(
            tier,
            {
                "demand": 0.0,
                "served": 0.0,
                "weight": w,
                "demand_disconnected": 0.0,
                "served_constraint_capped": 0.0,
            },
        )
        by_tier[tier]["demand"] += demand
        by_tier[tier]["served"] += served
        by_tier[tier]["demand_disconnected"] += demand_disc
        by_tier[tier]["served_constraint_capped"] += served_constraint_capped

        by_tier_sector.setdefault(sec_key, {})
        by_tier_sector[sec_key].setdefault(
            tier, {"demand": 0.0, "served": 0.0, "demand_disconnected": 0.0}
        )
        by_tier_sector[sec_key][tier]["demand"] += demand
        by_tier_sector[sec_key][tier]["served"] += served
        by_tier_sector[sec_key][tier]["demand_disconnected"] += demand_disc

        pw_by_sector.setdefault(sec_key, {"demand": 0.0, "served": 0.0})
        pw_by_sector[sec_key]["demand"] += w * demand
        pw_by_sector[sec_key]["served"] += w * served
        # Cross-sector aggregate is MW: electricity, heat, and (HHV-converted)
        # gas terminal load all contribute.
        pw_demand += w * demand
        pw_served += w * served
        if served < 1e-9 and demand > 1e-9:
            n_zero += 1

    for entry in by_sector.values():
        entry["fraction"] = (
            entry["served"] / entry["demand"] if entry["demand"] > 0 else 1.0
        )
    for entry in by_tier.values():
        entry["fraction"] = (
            entry["served"] / entry["demand"] if entry["demand"] > 0 else 1.0
        )
    for sec_entry in by_tier_sector.values():
        for entry in sec_entry.values():
            entry["fraction"] = (
                entry["served"] / entry["demand"] if entry["demand"] > 0 else 1.0
            )

    return {
        "by_tier_sector": by_tier_sector,
        "by_sector": by_sector,
        "by_tier": by_tier,
        # Aggregate PWSF over electricity + heat + gas, all in MW (gas HHV-
        # converted from kg/s to the same energy basis).
        "priority_weighted_demand": pw_demand,
        "priority_weighted_served": pw_served,
        "priority_weighted_fraction": (pw_served / pw_demand if pw_demand > 0 else 1.0),
        # Per-sector PWSF — same MW basis, split by carrier.
        "priority_weighted_demand_by_sector": {
            s: v["demand"] for s, v in pw_by_sector.items()
        },
        "priority_weighted_served_by_sector": {
            s: v["served"] for s, v in pw_by_sector.items()
        },
        "priority_weighted_fraction_by_sector": {
            s: (v["served"] / v["demand"] if v["demand"] > 0 else 1.0)
            for s, v in pw_by_sector.items()
        },
        "n_loads": n_loads,
        "n_loads_served_zero": n_zero,
    }


# Restoration breakdown — relative to the pre-failure baseline


def restoration_breakdown(
    post: dict[str, Any], baseline: dict[str, Any] | None
) -> dict[str, Any]:
    """Compare the post-restoration served breakdown against the pre-failure
    baseline, quantifying how much load was lost despite restoration.

    ``post`` and ``baseline`` are both ``served_breakdown`` dicts. When
    ``baseline`` is None or has zero demand the derived ratios collapse to
    ``1.0`` (no failure => no degradation).

    Output:

    ``{
        "absolute_load_lost_mw":   total_demand_baseline - total_served_post,
        "absolute_load_dropped_mw": total_served_baseline - total_served_post,
        "total_demand_mw":         total_demand_baseline,
        "total_served_baseline_mw": total_served_baseline,
        "total_served_post_mw":     total_served_post,
        "raw_restoration_ratio":   total_served_post / total_served_baseline,
        "pwsf_baseline":           baseline.priority_weighted_fraction,
        "pwsf_post":               post.priority_weighted_fraction,
        "pwsf_restoration_ratio":  pwsf_post / pwsf_baseline,
        "by_tier":   per-tier {demand_baseline_mw, served_baseline_mw,
                                served_post_mw, post_fraction (=post/demand),
                                baseline_fraction, ratio (=post/baseline,
                                clamped), ratio_unclamped, disconnect_lost_mw,
                                constraint_lost_mw, agent_shed_mw,
                                agent_only_ratio},
        "by_sector": per-sector {same fields, minus weight},
    }``

    The ``raw_*`` fields are unweighted MW (absolute load lost) alongside the
    priority-weighted fraction. ``by_tier.post_fraction`` is the headline
    per-tier view: the baseline-relative ``ratio`` divides by a priority-aware,
    slack-budget-bounded baseline that already sheds low tiers, so its clamp
    saturates them at 1.0 and can fabricate an apparent tier inversion.
    """
    if not baseline:
        return {"baseline_available": False}

    pw_demand = baseline.get("priority_weighted_demand", 0.0)
    pw_served_baseline = baseline.get("priority_weighted_served", 0.0)
    pw_served_post = post.get("priority_weighted_served", 0.0)

    sector_baseline = baseline.get("by_sector", {})
    sector_post = post.get("by_sector", {})
    # Raw MW totals sum all three sectors; gas by_sector is already HHV-converted
    # to MW in served_breakdown, so it adds on the same energy basis.
    _mw_sectors = (Sector.ELECTRICITY.value, Sector.HEAT.value, Sector.GAS.value)
    total_demand = sum(
        s.get("demand", 0.0) for k, s in sector_baseline.items() if k in _mw_sectors
    )
    total_served_baseline = sum(
        s.get("served", 0.0) for k, s in sector_baseline.items() if k in _mw_sectors
    )
    total_served_post = sum(
        s.get("served", 0.0) for k, s in sector_post.items() if k in _mw_sectors
    )

    by_tier_b = baseline.get("by_tier", {})
    by_tier_p = post.get("by_tier", {})
    by_tier_out: dict[str, dict[str, float]] = {}
    for tier in set(by_tier_b) | set(by_tier_p):
        tb = by_tier_b.get(tier, {})
        tp = by_tier_p.get(tier, {})
        d_b = float(tb.get("demand", 0.0))
        s_b = float(tb.get("served", 0.0))
        s_p = float(tp.get("served", 0.0))
        # Split the per-tier loss three ways: ``disconnect_lost`` is the
        # priority-blind share (load that physically lost its path to a
        # grid-former, irrecoverable); ``constraint_lost`` is served the agents
        # dispatched but the eval feasibility gates (heat t_k, line loading)
        # removed — physics-throttled, no agent lever; the remainder is
        # ``agent_shed`` (load the QP / ADMM chose to drop). Priority should
        # drive agent_shed only.
        disc_p = float(tp.get("demand_disconnected", 0.0))
        capped_p = float(tp.get("served_constraint_capped", 0.0))
        total_loss = max(0.0, s_b - s_p)
        disconnect_lost = max(0.0, min(disc_p, total_loss))
        # disconnect_lost > total_loss is arithmetically impossible; log + clamp
        # so the inversion is visible rather than masked by the outer min.
        if disc_p > total_loss + 1e-12:
            logger.warning(
                "Tier %s: demand_disconnected=%.4g > total_loss=%.4g; "
                "clamping disconnect_lost to total_loss.",
                tier,
                disc_p,
                total_loss,
            )
        constraint_lost = max(0.0, min(capped_p, total_loss - disconnect_lost))
        agent_shed = max(0.0, total_loss - disconnect_lost - constraint_lost)
        ratio = s_p / s_b if s_b > 1e-12 else (1.0 if d_b < 1e-12 else 0.0)
        # Agent-only ratio: share of the controllable baseline (physical
        # disconnect and gate-throttled load excluded) the agents kept.
        s_b_recoverable = max(1e-12, s_b - disconnect_lost - constraint_lost)
        agent_ratio = (s_b_recoverable - agent_shed) / s_b_recoverable
        by_tier_out[str(tier)] = {
            "demand_baseline_mw": d_b,
            "served_baseline_mw": s_b,
            "served_post_mw": s_p,
            "post_fraction": (s_p / d_b if d_b > 1e-12 else 1.0),
            "baseline_fraction": (s_b / d_b if d_b > 1e-12 else 1.0),
            "ratio": max(0.0, min(1.0, ratio)),
            "ratio_unclamped": ratio,
            "disconnect_lost_mw": disconnect_lost,
            "constraint_lost_mw": constraint_lost,
            "agent_shed_mw": agent_shed,
            "agent_only_ratio": max(0.0, min(1.0, agent_ratio)),
            "weight": float(tb.get("weight", 0.0)),
        }

    by_sector_out: dict[str, dict[str, float]] = {}
    for sec in set(sector_baseline) | set(sector_post):
        sb = sector_baseline.get(sec, {})
        sp = sector_post.get(sec, {})
        d_b = float(sb.get("demand", 0.0))
        s_b = float(sb.get("served", 0.0))
        s_p = float(sp.get("served", 0.0))
        ratio = s_p / s_b if s_b > 1e-12 else 1.0
        disc_p = float(sp.get("demand_disconnected", 0.0))
        capped_p = float(sp.get("served_constraint_capped", 0.0))
        total_loss = max(0.0, s_b - s_p)
        disconnect_lost = max(0.0, min(disc_p, total_loss))
        constraint_lost = max(0.0, min(capped_p, total_loss - disconnect_lost))
        agent_shed = max(0.0, total_loss - disconnect_lost - constraint_lost)
        by_sector_out[sec] = {
            "demand_baseline_mw": d_b,
            "served_baseline_mw": s_b,
            "served_post_mw": s_p,
            "post_fraction": (s_p / d_b if d_b > 1e-12 else 1.0),
            "baseline_fraction": (s_b / d_b if d_b > 1e-12 else 1.0),
            "ratio": max(0.0, min(1.0, ratio)),
            "ratio_unclamped": ratio,
            "disconnect_lost_mw": disconnect_lost,
            "constraint_lost_mw": constraint_lost,
            "agent_shed_mw": agent_shed,
        }

    # Campaign-level disconnect / constraint / agent split (per-sector sums);
    # all three sectors are MW (gas HHV-converted), same filter as the totals.
    total_disconnect_lost = sum(
        s.get("disconnect_lost_mw", 0.0)
        for k, s in by_sector_out.items()
        if k in _mw_sectors
    )
    total_constraint_lost = sum(
        s.get("constraint_lost_mw", 0.0)
        for k, s in by_sector_out.items()
        if k in _mw_sectors
    )
    total_agent_shed = sum(
        s.get("agent_shed_mw", 0.0)
        for k, s in by_sector_out.items()
        if k in _mw_sectors
    )

    return {
        "baseline_available": True,
        "absolute_load_lost_mw": max(0.0, total_demand - total_served_post),
        "absolute_load_dropped_mw": max(0.0, total_served_baseline - total_served_post),
        "disconnect_lost_mw": total_disconnect_lost,
        "constraint_lost_mw": total_constraint_lost,
        "agent_shed_mw": total_agent_shed,
        "total_demand_mw": total_demand,
        "total_served_baseline_mw": total_served_baseline,
        "total_served_post_mw": total_served_post,
        "raw_restoration_ratio": (
            total_served_post / total_served_baseline
            if total_served_baseline > 1e-12
            else 1.0
        ),
        "pwsf_baseline": (pw_served_baseline / pw_demand if pw_demand > 0 else 1.0),
        "pwsf_post": (pw_served_post / pw_demand if pw_demand > 0 else 1.0),
        "pwsf_restoration_ratio": (
            pw_served_post / pw_served_baseline if pw_served_baseline > 1e-12 else 1.0
        ),
        "by_tier": by_tier_out,
        "by_sector": by_sector_out,
    }


# Constraint violation integral


def _is_deenergised_avg(var: str, val: float) -> bool:
    """A sector-average reading low enough to mean the network (on average)
    collapsed/de-energised rather than violated a bound. Mirrors the per-node
    guards in ``constraint_rows`` (DEENERGISED_VM_PU / DEENERGISED_PRESSURE_PU
    low + DEENERGISED_PRESSURE_HIGH_PU high / t_k<=0) so the integral does not
    score a black-out as an over/under-voltage
    violation — without this a fully de-energised run (avg_vm_pu~0.05) integrates
    to a huge spurious value while the final scan reports zero violations.
    """
    if var == "vm_pu":
        return val <= DEENERGISED_VM_PU
    if var == "pressure_pu":
        # Low floor = region cut off from supply; high saturation (~sqrt(3))
        # = solver bound on an isolated region. Both are artefacts, not a real
        # average over-/under-pressure (matches the plot/model de-energised masks).
        return val <= DEENERGISED_PRESSURE_PU or val >= DEENERGISED_PRESSURE_HIGH_PU
    if var == "t_k":
        return val <= 0.0
    return False


def constraint_violation_integral(world: Any) -> dict[str, float]:
    """Per-sector ``integral of max(0, util(t) - 1) dt``, proxied by the
    recorded sector-average utilization series (avg_vm_pu / avg_pressure_pu /
    avg_t_k). Dimensionless; comparable across runs on the same grid. Being an
    average, it masks per-node violations (see ``constraint_violations_final``).

    De-energised samples (sector average below the DEENERGISED_* floor, i.e. the
    network collapsed) are NOT counted as violations — same convention as the
    final scan — so a black-out does not masquerade as a bound violation.
    """
    recordings = getattr(world, "data_collections", {}) or {}

    sector_var: dict[str, tuple[Sector, str]] = {
        "avg_vm_pu": (Sector.ELECTRICITY, "vm_pu"),
        "avg_pressure_pu": (Sector.GAS, "pressure_pu"),
        "avg_t_k": (Sector.HEAT, "t_k"),
    }

    out: dict[str, float] = {s.value: 0.0 for s in Sector}
    for name, rec in recordings.items():
        if name not in sector_var or not isinstance(rec, WorldRecording):
            continue
        sec, var = sector_var[name]
        bounds = SECTOR_CONSTRAINTS.get(sec, {}).get(var)
        if bounds is None:
            continue
        lo, hi = bounds
        ts = list(rec.timeseries)
        ts_t = list(rec.time)
        if len(ts) < 2:
            continue
        # Trapezoidal integration of max(0, util − 1) over time. The unclamped
        # utilization lets out-of-bounds readings exceed 1.0 (the clamped form
        # pins the integrand — and thus every integral — to 0). A de-energised
        # sample contributes 0 (a collapse is not a bound violation).
        integral = 0.0
        for i in range(1, len(ts)):
            va, vb = float(ts[i - 1]), float(ts[i])
            ov_a = (
                0.0
                if _is_deenergised_avg(var, va)
                else max(0.0, constraint_utilization(va, lo, hi, unclamped=True) - 1.0)
            )
            ov_b = (
                0.0
                if _is_deenergised_avg(var, vb)
                else max(0.0, constraint_utilization(vb, lo, hi, unclamped=True) - 1.0)
            )
            dt = float(ts_t[i]) - float(ts_t[i - 1])
            integral += 0.5 * (ov_a + ov_b) * dt
        out[sec.value] = integral
    return out


# Final constraint-violation scan (end-of-sim hard-bound feasibility)
# Node-by-node / branch-by-branch feasibility of the final solved network
# against the same ``SECTOR_CONSTRAINTS`` envelope the oracle LP enforces
# (bounds_el / _gas / _heat + ``max_line_loading``). Flags a SCARE run that
# "beats" the oracle's PWSF only by leaving voltages, pressures, temperatures,
# or line loadings out of bounds, so the aggregator can exclude it from the
# compliant-PWSF mean.

# Per-variable absolute tolerance: a reading violates only when it exceeds the
# bound by more than this. ``vm_pu`` / ``pressure_pu`` are p.u.; ``t_k`` and
# ``loading_percent`` reuse the heat-served / line-feasibility tolerances so the
# compliance gate and the served de-rating draw the line at the same place.
_CONSTRAINT_ABS_TOL: dict[str, float] = {
    "vm_pu": 0.005,
    # 0.01 (not 0.005) absorbs the MIQCQP solver residual on sqrt-derived
    # pressure_pu: feasible oracle solves land 1e-4..8e-4 below lo - 0.005 and
    # were ejected as non-compliant.
    "pressure_pu": 0.01,
    "t_k": _HEAT_T_TOL_K,
    "loading_percent": _LINE_LOADING_TOL_PCT,
}

# Variables with only a physical upper bound (idle = 0, limit = 100); the
# lower half of their ``SECTOR_CONSTRAINTS`` pair is a formula artefact and
# must not gate.
_ONE_SIDED_VARS: frozenset[str] = frozenset({"loading_percent"})


def _model_value(model: Any, key: str) -> float | None:
    """Return ``model.values[key]`` as a finite float, or ``None`` when the
    attribute is absent / non-numeric / not populated by the solver."""
    if model is None:
        return None
    try:
        vals = model.values if hasattr(model, "values") else {}
    except Exception:  # noqa: BLE001 — some models raise on access
        return None
    if key not in vals:
        return None
    try:
        return float(vals[key])
    except (TypeError, ValueError):
        return None


def _cp_param(model: Any, key: str, default: float = 0.0) -> float:
    """Numeric CP attribute (``Var`` or scalar), or *default* when absent."""
    if not hasattr(model, key):
        return default
    try:
        v = _mvalue(getattr(model, key))
    except Exception:  # noqa: BLE001
        return default
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _net_gas_hhv(monee_net: Any) -> float:
    for grid in getattr(monee_net, "grids", []) or []:
        if isinstance(grid, GasGrid):
            hhv = getattr(grid, "higher_heating_value_kwh_per_kg", None)
            if hhv:
                return float(hhv)
    return DEFAULT_GAS_HHV_KWH_PER_KG


def _cp_output(model: Any, gas_hhv: float) -> tuple[float, float, float]:
    """Delivered ``(el, heat, gas)`` MW for one CP model at its ACTUAL setpoint.

    Computed as ``nameplate_capacity * regulation`` from the model's fixed
    parameters — NOT from the solved ``el_mw`` / ``heat_mw`` Vars. The oracle
    solves ``regulation`` to a free Var and folds it into those output Vars, but
    the scare net only runs a power flow: it never re-derives the converter
    setpoints, so their Vars sit at the unregulated nameplate and reading them
    reports nameplate injection (e.g. 6 MW into a 1 MW grid). ``regulation`` is
    the single lever both variants carry (a solved Var for the oracle, the
    MAS-actuated scalar for scare), so scaling capacity by it reads the true
    dispatched setpoint consistently. Gas output is energy content (MW).
    """
    reg = _cp_param(model, "regulation", 1.0)
    k = KGPS_KWHPERKG_TO_MW
    if isinstance(model, (CHPControlNode, CHPHGControlNode)):
        hhv = _cp_param(model, "_hhv", gas_hhv)
        base = abs(_cp_param(model, "gas_mass_flow_kgs")) * k * hhv * reg
        return (
            _cp_param(model, "efficiency_power") * base,
            _cp_param(model, "efficiency_heat") * base,
            0.0,
        )
    if isinstance(model, GasToHeatControlNode):
        hhv = _cp_param(model, "_hhv", gas_hhv)
        heat = _cp_param(model, "efficiency_heat") * (
            abs(_cp_param(model, "gas_mass_flow_kgs")) * k * hhv * reg
        )
        return (0.0, heat, 0.0)
    if isinstance(model, PowerToHeatControlNode):
        # el_mw here is the electricity CONSUMPTION setpoint; only heat is output.
        return (0.0, _cp_param(model, "efficiency") * abs(_cp_param(model, "el_mw")) * reg, 0.0)
    if isinstance(model, (PowerToHeatHG, GasToHeatHG)):
        return (0.0, abs(_cp_param(model, "heat_energy_mw")) * reg, 0.0)
    if isinstance(model, GasToPower):
        return (abs(_cp_param(model, "el_mw")) * reg, 0.0, 0.0)
    if isinstance(model, PowerToGas):
        return (0.0, 0.0, abs(_cp_param(model, "gas_mass_flow_kgs")) * k * gas_hhv * reg)
    return (0.0, 0.0, 0.0)


def cp_generation_breakdown(monee_net: Any) -> dict[str, Any]:
    """Delivered coupling-point converter output (MW) at end of sim, summed
    over every CP unit — the direct "are the CPs contributing?" measure.

    Each CP's delivered output is its nameplate capacity scaled by its actual
    ``regulation`` setpoint (see :func:`_cp_output` for why the solved Vars are
    not read directly). CHP/CHPHG deliver el + heat off gas; P2H/G2H/`*HG`
    deliver heat; G2P delivers el; P2G delivers gas (energy content, MW). A
    failed/deactivated CP carries ``regulation ~ 0`` and contributes nothing;
    ``n_active`` counts units above a 1 kW floor.
    """
    el = heat = gas = 0.0
    n_cp = 0
    n_active = 0
    gas_hhv = _net_gas_hhv(monee_net)

    def _is_cp_model(model: Any) -> bool:
        fn = getattr(model, "is_cp", None)
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001
            return False

    for component in list(monee_net.nodes) + list(monee_net.branches):
        model = getattr(component, "model", None)
        if not _is_cp_model(model):
            continue
        c_el, c_heat, c_gas = _cp_output(model, gas_hhv)
        el += c_el
        heat += c_heat
        gas += c_gas
        n_cp += 1
        if c_el + c_heat + c_gas > 1e-3:
            n_active += 1

    return {
        "total_mw": el + heat + gas,
        "el_mw": el,
        "heat_mw": heat,
        "gas_mw": gas,
        "n_cp": n_cp,
        "n_active": n_active,
    }


def _branch_loading_mva_percent(model: Any) -> float | None:
    """Worst-side loading in the MVA basis: ``100 * sqrt(p^2 + q^2) /
    max_s_mva``. None when the rating or the solved p/q flows are
    unavailable."""
    try:
        mva = float(getattr(model, "max_s_mva", None))
    except (TypeError, ValueError):
        return None
    if not (mva > 0.0):
        return None
    s_sides: list[float] = []
    for side in ("from", "to"):
        p = _model_value(model, f"p_{side}_mw")
        q = _model_value(model, f"q_{side}_mvar")
        if p is not None and q is not None and math.isfinite(p) and math.isfinite(q):
            s_sides.append(math.hypot(p, q))
    if not s_sides:
        return None
    return 100.0 * max(s_sides) / mva


# Mirrors monee problem/utils.py: max_i_ka at/above this sentinel means the LP
# leaves the branch current unbounded, so there is no rating to grade against.
_UNBOUND_MAX_I_KA: float = 999.0


def _branch_loading_current_percent(branch: Any, monee_net: Any) -> float | None:
    """Worst-side loading in the exact current basis: ``100 * sqrt(p^2 + q^2)
    / (sqrt(3) * vm_pu * base_kv * max_i_ka)`` per side (S in MVA, voltage in
    kV, current in kA). This is the current the solved power flows imply —
    NOT the SOC-relaxed ``i_{from,to}_ka`` intermediates, which carry the
    relaxation gap. None when the rating, flows, or an energised end-voltage
    are unavailable."""
    model = getattr(branch, "model", None)
    try:
        max_i_ka = float(getattr(model, "max_i_ka", None))
    except (TypeError, ValueError):
        return None
    if not (0.0 < max_i_ka < _UNBOUND_MAX_I_KA):
        return None
    pcts: list[float] = []
    for side, node_id in (
        ("from", getattr(branch, "from_node_id", None)),
        ("to", getattr(branch, "to_node_id", None)),
    ):
        p = _model_value(model, f"p_{side}_mw")
        q = _model_value(model, f"q_{side}_mvar")
        if p is None or q is None or not (math.isfinite(p) and math.isfinite(q)):
            continue
        try:
            node_model = monee_net.node_by_id(node_id).model
        except Exception:  # noqa: BLE001
            continue
        vm = _model_value(node_model, "vm_pu")
        try:
            base_kv = float(getattr(node_model, "base_kv", None))
        except (TypeError, ValueError):
            continue
        if vm is None or not math.isfinite(vm) or vm <= DEENERGISED_VM_PU:
            continue
        if not (base_kv > 0.0):
            continue
        i_ka = math.hypot(p, q) / (math.sqrt(3.0) * vm * base_kv)
        pcts.append(100.0 * i_ka / max_i_ka)
    return max(pcts) if pcts else None


def _branch_loading_exact_percent(branch: Any, monee_net: Any) -> float | None:
    """Worst-side loading re-judged in the exact basis the oracle LP enforces:
    the MVA basis when the branch carries ``max_s_mva``, else the current
    basis monee's ``line_loading_limit`` falls back to (benchmark imports set
    only ``max_i_ka``)."""
    mva_pct = _branch_loading_mva_percent(getattr(branch, "model", None))
    if mva_pct is not None:
        return mva_pct
    return _branch_loading_current_percent(branch, monee_net)


def _branch_loading_percent(branch: Any, monee_net: Any) -> float | None:
    """Worst (from/to) thermal loading of a branch in *percent*.

    The reported ``loading_pu`` / ``loading_{from,to}_pu`` (per-unit fractions
    in every formulation; ``loading_pu`` is a Python property, not in
    ``model.values``, hence the per-side fallback) is the screen. An apparent
    overload is then re-judged in the exact basis the oracle LP enforces —
    MVA (``sqrt(p^2 + q^2) <= max_loading * max_s_mva``) when the branch is
    MVA-rated, else the exact current implied by the solved flows and end
    voltages against ``max_i_ka``: under MISOCP the loading intermediates
    derive from the SOC-RELAXED current, which only OVERSTATES loading
    (relaxation gap), so grading from them charged phantom overloads to
    solutions satisfying the LP's cap — while a genuine overload always shows
    in the screen. Re-judging (rather than grading the exact basis
    unconditionally) also keeps unsolved nets, whose p/q sit at Var defaults,
    from fabricating overloads.
    """
    model = getattr(branch, "model", None)
    if model is None:
        return None
    lp = _model_value(model, "loading_pu")
    if lp is None:
        lf = _model_value(model, "loading_from_pu")
        lt = _model_value(model, "loading_to_pu")
        mags = [abs(x) for x in (lf, lt) if x is not None]
        if not mags:
            return _branch_loading_exact_percent(branch, monee_net)
        lp = max(mags)
    pct = abs(lp) * 100.0
    if pct > 100.0 + _LINE_LOADING_TOL_PCT:
        exact_pct = _branch_loading_exact_percent(branch, monee_net)
        if exact_pct is not None:
            return exact_pct
    return pct


def _bound_overshoot(val: float, lo: float, hi: float, *, one_sided: bool) -> float:
    """How far ``val`` lies beyond its nearest bound, normalised by the
    half-span (0 = in-bounds, 1 = a full half-span over). Unlike
    :func:`constraint_utilization` (saturates at the bound), this ranks breach
    severity for the "worst violation" ordering. One-sided variables
    (``loading_percent``) only overshoot the upper bound.
    """
    half_span = (hi - lo) / 2.0
    if half_span <= 0:
        return 0.0
    if val > hi:
        return (val - hi) / half_span
    if val < lo and not one_sided:
        return (lo - val) / half_span
    return 0.0


def _violation_row(
    kind: str,
    cid: Any,
    sec: Sector,
    var: str,
    val: float,
    lo: float,
    hi: float,
) -> dict[str, Any]:
    tol = _CONSTRAINT_ABS_TOL.get(var, 0.0)
    one_sided = var in _ONE_SIDED_VARS
    if one_sided:
        violated = val > hi + tol
    else:
        violated = (val < lo - tol) or (val > hi + tol)
    overshoot = _bound_overshoot(val, lo, hi, one_sided=one_sided)
    return {
        "kind": kind,
        "id": cid,
        "sector": sec.value,
        "variable": var,
        "value": val,
        "lo": lo,
        "hi": hi,
        "overshoot": overshoot,
        "violated": violated,
    }


def constraint_rows(monee_net: Any) -> list[dict[str, Any]]:
    """Per-node / per-branch hard-bound readings off the final network state.

    Walks every active, connected node (``vm_pu`` / ``pressure_pu`` / ``t_k``
    per its sector) and every active electricity branch (``loading_percent``)
    against ``SECTOR_CONSTRAINTS``. One row per checked variable: ``{kind, id,
    sector, variable, value, lo, hi, overshoot, violated}``.

    Disconnected and inactive nodes/branches are skipped: their loads already
    count as served=0, their readings are meaningless (t_k ~0 on an isolated
    junction), and the oracle excludes them too.
    """
    disconnected = _disconnected_node_ids(monee_net)
    rows: list[dict[str, Any]] = []

    for node in monee_net.nodes:
        if not getattr(node, "active", True) or node.id in disconnected:
            continue
        # Islanding-de-energised nodes read ~0 like disconnected ones — their
        # excursions are de-energisation artifacts, not operating violations.
        if _is_deenergised({}, node):
            continue
        sec = sector_from_grid(getattr(node, "grid", None))
        if sec is None:
            continue
        for var, (lo, hi) in SECTOR_CONSTRAINTS.get(sec, {}).items():
            if var == "loading_percent":
                continue  # branch-level — handled below
            val = _model_value(node.model, var)
            if val is None or not math.isfinite(val):
                continue
            # Solver-unpopulated / de-energised junctions are not real breaches
            # (the live monitor skips them the same way): an isolated heat
            # junction reports t_k~0, a gas region cut off from its ExtHydrGrid
            # collapses to pressure_pu~0 (or saturates at the relaxed-Weymouth
            # solver bound ~sqrt(3) — same artefact, high side), and an
            # electricity node cut off from its slack collapses to vm_pu~0. See
            # DEENERGISED_* for why a small floor (not 0) is needed and why
            # genuine out-of-bound readings still gate.
            if (
                (var == "t_k" and val <= 0.0)
                or (
                    var == "pressure_pu"
                    and (
                        val <= DEENERGISED_PRESSURE_PU
                        or val >= DEENERGISED_PRESSURE_HIGH_PU
                    )
                )
                or (var == "vm_pu" and val <= DEENERGISED_VM_PU)
            ):
                continue
            rows.append(_violation_row("node", node.id, sec, var, val, lo, hi))

    el_loading = SECTOR_CONSTRAINTS.get(Sector.ELECTRICITY, {}).get("loading_percent")
    if el_loading is not None:
        lo, hi = el_loading
        for branch in monee_net.branches:
            if not _branch_is_active(branch):
                continue
            if not _branch_carries_sector(branch, monee_net, "electricity"):
                continue
            a, b = branch.id[0], branch.id[1]
            if a in disconnected or b in disconnected:
                continue
            val = _branch_loading_percent(branch, monee_net)
            if val is None or not math.isfinite(val):
                continue
            rows.append(
                _violation_row(
                    "branch",
                    branch.id,
                    Sector.ELECTRICITY,
                    "loading_percent",
                    val,
                    lo,
                    hi,
                )
            )
    return rows


# Constraint variables audited but excluded from the compliance gate. Empty:
# every operating bound in ``SECTOR_CONSTRAINTS`` gates. Heat junction
# temperature (``t_k``) gates on the same footing as voltage, pressure, and line
# loading — a temperature-infeasible heat node is a genuine envelope breach, so a
# run that leaves junctions out of band is non-compliant even though the served-
# fraction metric also debits the cold load. Canonical for both the
# ``constraint_violations_final`` scan (SCARE outcome + oracle claim) and the
# ``_check_constraint_compliance`` CSV claim (which imports this set), so SCARE
# and the oracle gate on identical flags.
NON_GATING_CONSTRAINT_VARIABLES: frozenset[str] = frozenset()

# Canonical display label per raw ``SECTOR_CONSTRAINTS`` variable, for the
# per-variable-type violation tally that accompanies the compliance gate.
# Electricity carries TWO gating variables (voltage + line loading), so the
# per-sector counts can't separate them — this split does. Shared by the
# ``constraint_violations_final`` scan and the ``_check_constraint_compliance``
# CSV claim so both report identical variable buckets. Slack is tallied
# separately (``slack_budget_compliance`` — a different artefact).
CONSTRAINT_VARIABLE_LABEL: dict[str, str] = {
    "vm_pu": "voltage",
    "pressure_pu": "pressure",
    "t_k": "temperature",
    "loading_percent": "line_load",
}


def _variable_tally_entry(
    by_variable: dict[str, dict[str, Any]], var: str
) -> dict[str, Any]:
    """Get-or-create the per-variable-type tally bucket for raw variable
    ``var``, keyed by its :data:`CONSTRAINT_VARIABLE_LABEL` (falls back to the
    raw name for any unmapped variable)."""
    label = CONSTRAINT_VARIABLE_LABEL.get(var, var)
    return by_variable.setdefault(
        label,
        {
            "n_checked": 0,
            "n_violations": 0,
            "worst_overshoot": 0.0,
            "gating": var not in NON_GATING_CONSTRAINT_VARIABLES,
        },
    )


def constraint_violations_final(monee_net: Any) -> dict[str, Any]:
    """End-of-sim hard-bound feasibility summary over the final network state.

    Returns ``{passed, n_checked, n_violations, n_nongating_violations,
    by_sector, violations, nongating_violations}`` where ``passed`` is True iff
    no active, connected node or branch breaches its GATING ``SECTOR_CONSTRAINTS``
    bound (within :data:`_CONSTRAINT_ABS_TOL`). Basis for the
    ``constraint_compliance`` claim: a "compliant" run needs both the operator
    slack budget and in-bounds grid state, so the PWSF gap to the oracle is a
    real allocation gap, not feasibility bought by violations.

    All operating bounds gate, including heat junction temperature (``t_k``): a
    temperature-infeasible heat node breaches its envelope just as an out-of-band
    voltage or pressure does, so it flips ``passed`` (see
    :data:`NON_GATING_CONSTRAINT_VARIABLES`, now empty). De-energised / isolated
    junctions are still filtered out upstream in :func:`constraint_rows` and do
    not count as breaches.
    """
    rows = constraint_rows(monee_net)
    by_sector: dict[str, dict[str, Any]] = {}
    by_variable: dict[str, dict[str, Any]] = {}
    gating: list[dict[str, Any]] = []
    nongating: list[dict[str, Any]] = []
    for r in rows:
        entry = by_sector.setdefault(
            r["sector"],
            {
                "n_checked": 0,
                "n_violations": 0,
                "worst_overshoot": 0.0,
                "n_nongating_violations": 0,
            },
        )
        var_entry = _variable_tally_entry(by_variable, r["variable"])
        entry["n_checked"] += 1
        var_entry["n_checked"] += 1
        if r["violated"]:
            entry["n_violations"] += 1
            entry["worst_overshoot"] = max(entry["worst_overshoot"], r["overshoot"])
            var_entry["n_violations"] += 1
            var_entry["worst_overshoot"] = max(
                var_entry["worst_overshoot"], r["overshoot"]
            )
            if r["variable"] in NON_GATING_CONSTRAINT_VARIABLES:
                entry["n_nongating_violations"] += 1
                nongating.append(r)
            else:
                gating.append(r)
    gating.sort(key=lambda r: r["overshoot"], reverse=True)
    nongating.sort(key=lambda r: r["overshoot"], reverse=True)

    def _fmt(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": r["kind"],
            "id": str(r["id"]),
            "sector": r["sector"],
            "variable": r["variable"],
            "value": round(r["value"], 6),
            "lo": r["lo"],
            "hi": r["hi"],
            "overshoot": round(r["overshoot"], 6),
        }

    return {
        "passed": not gating,
        "n_checked": len(rows),
        "n_violations": len(gating),
        "n_nongating_violations": len(nongating),
        "by_sector": by_sector,
        "by_variable": by_variable,
        "violations": [_fmt(r) for r in gating[:10]],
        "nongating_violations": [_fmt(r) for r in nongating[:10]],
    }


# Time-to-stabilise


def time_to_stabilise_s(world: Any, *, hold_s: float = 1.0) -> float | None:
    """First simulation time after which all sector imbalance series stay near
    steady state for at least ``hold_s`` seconds.

    "Stable" = the absolute first-difference of every sector's balance series
    (``electrical_balance`` / ``gas_balance`` / ``heat_balance``) stays below
    0.5% of its max magnitude for a sustained window. None if never found.
    """
    recordings = getattr(world, "data_collections", {}) or {}
    sector_keys = ("electrical_balance", "gas_balance", "heat_balance")

    series: dict[str, tuple[list[float], list[float]]] = {}
    for k in sector_keys:
        rec = recordings.get(k)
        if not isinstance(rec, WorldRecording):
            continue
        ts = list(rec.timeseries)
        t = list(rec.time)
        if len(ts) >= 2:
            series[k] = (t, ts)

    if not series:
        return None

    # Per-series threshold = 0.5 % of max magnitude (or 1e-6 if zero).
    thresholds = {
        k: max(1e-6, 0.005 * max(abs(v) for v in ts)) for k, (_, ts) in series.items()
    }

    # Series share the recording cadence, but a dropped sample in one series
    # desyncs positional indices — align on the shared timestamps instead. A
    # step is stable only if every series' diff is below threshold.
    common_t = sorted(set.intersection(*(set(t) for t, _ in series.values())))
    if len(common_t) < 2:
        return None
    at_t = {k: dict(zip(t, ts)) for k, (t, ts) in series.items()}

    def step_stable(i: int) -> bool:
        for k in series:
            vals = at_t[k]
            if abs(vals[common_t[i]] - vals[common_t[i - 1]]) > thresholds[k]:
                return False
        return True

    stable_since: float | None = None
    for i in range(1, len(common_t)):
        if step_stable(i):
            if stable_since is None:
                stable_since = common_t[i]
            elif common_t[i] - stable_since >= hold_s:
                return float(stable_since)
        else:
            stable_since = None
    return None


# Optimality gap (vs oracle)


def optimality_gap(scare_pw_served: float, oracle_pw_served: float) -> float:
    """Relative gap ``(oracle - scare) / oracle`` clipped to ``[0, 1]``.
    Negative inputs (oracle worse than scare) clip to 0 — but are logged: since
    the oracle is the constraint-respecting optimum, scare > oracle signals
    either an oracle bug (e.g. a priority-blind LP) or scare credited above
    feasibility, both worth surfacing rather than silently flooring to a
    'perfect' gap of 0."""
    if oracle_pw_served <= 0:
        return 0.0
    gap = (oracle_pw_served - scare_pw_served) / oracle_pw_served
    if not math.isfinite(gap):
        return 0.0
    if gap < 0:
        logger.warning(
            "optimality_gap negative (scare %.4f > oracle %.4f): clipping to 0; "
            "oracle should dominate — check oracle validity / feasibility.",
            scare_pw_served,
            oracle_pw_served,
        )
    return max(0.0, min(1.0, gap))
