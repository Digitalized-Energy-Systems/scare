"""Post-hoc metrics computed from a finished restoration run.

The world / monee net are still in scope at the end of the runner —
this module pulls the final regulation factors and the recorded
timeseries / event ledger out of them and computes the dissertation's
primary outcome metrics.

Outputs are dicts of plain Python scalars / nested dicts so they can be
serialised straight into ``result.json`` and aggregated downstream.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import networkx as nx
from mango.simulation.world import WorldRecording
from monee.model.child import HeatLoad, PowerLoad
from monee.solver.core import find_ignored_nodes

from scare.base.model import SECTOR_CONSTRAINTS, Sector
from scare.base.util import (
    constraint_allowed_fraction,
    constraint_utilization,
    obs_capacity,
    obs_priority,
    obs_setpoint,
    sector_from_grid,
)

logger = logging.getLogger(__name__)


# Priority-tier weight schedule.  Mirrors the chapter's ``w(π) = 2^(P − π)``
# (P = 10 ⇒ tier 1 weighs 512×, tier 10 weighs 1×).  Tier 0 (generators)
# contributes nothing to the served metric — only loads count.
_P = 10


def _tier_weight(tier: int) -> float:
    if tier <= 0:
        return 0.0
    return 2.0 ** max(0, _P - tier)


# ---------------------------------------------------------------------------
# Final served fractions
# ---------------------------------------------------------------------------


def _disconnected_node_ids(monee_net: Any) -> set[int]:
    """Return the set of node IDs that have no path to any grid-forming
    component (ExtPowerGrid / ExtHydrGrid) through the *currently active*
    branch topology.  These nodes' loads are physically un-servable
    regardless of what the LP or the agents claim — they must be counted
    as zero served in the restoration metric.

    Mirrors monee's solver-side ``find_ignored_nodes`` but operates
    purely on the user-visible network so we don't depend on whether
    the solver was invoked with ``exclude_unconnected_nodes=True``.

    The two failure modes we're guarding against:

    1. ``inject_nans`` zeroes ``regulation`` on the LP *copy*; the
       original (which the metric reads) keeps the constructor default
       of 1.0.  A disconnected load then reports ``served = cap``.
    2. The solver was run without ``exclude_unconnected_nodes=True``
       (currently the case for both oracle and scare's per-step solver),
       went infeasible, and left every regulation at 1.0.

    Both routes lead to the metric over-counting served load on
    physically disconnected nodes.  This helper closes both holes.
    """
    try:
        return set(find_ignored_nodes(monee_net))
    except Exception:
        return set()


def _branch_carries_sector(branch: Any, monee_net: Any, sector: str | None) -> bool:
    """Whether this branch should be admitted into the component
    subgraph for ``sector``.

    ``sector is None`` admits every branch (legacy "full-graph" view).
    A named sector admits ONLY same-sector edges — CP couplings
    (``branch.model.is_cp() == True``) are excluded.  This matches the
    L2 component-scope ADMM coordinator-election scope at
    ``src/scare/community/holonic.py:_resolve_component_peer_addrs``,
    which calls ``mirror.reachable_from(my_node, sector=self.sector)``
    and the topology-mirror explicitly rejects CP bridges when a
    sector filter is set (``src/scare/base/topology_mirror.py``
    ``reachable_from`` raises on ``allow_cp_bridges=True`` with a sector).

    Aligning the metric here with the control's scope is the fix for
    the priority-invariant inversions observed in
    ``eval_full_small_20260529-181310/tasks/{52,88,132,133}``: when two
    electricity sub-islands are bridged only by a CP → heat → CP chain,
    the L2 splits into two coordinators (each making independent
    per-tier shed decisions) while the legacy full-graph metric merged
    them and spuriously read an inversion across the coordinator
    boundary.  See ``tests/community/test_component_scope_cp_bridge.py``
    for the minimal reproducer.
    """
    if sector is None:
        return True
    model = getattr(branch, "model", None)
    if model is not None and getattr(model, "is_cp", lambda: False)():
        return False
    # Same-sector check: walk to one endpoint's grid → sector.  Reuse
    # ``sector_from_grid`` (the same resolver the rest of the codebase
    # uses) so a future sector enum addition lands in one place.
    a = branch.id[0]
    try:
        node = monee_net.node_by_id(a)
    except Exception:
        return False
    sec = sector_from_grid(getattr(node, "grid", None))
    return sec is not None and sec.value == sector


def _active_node_components(
    monee_net: Any, *, sector: str | None = None,
) -> dict[Any, int]:
    """Return a mapping ``node_id -> component_index`` over the
    *active*-branch subgraph.  Failed branches (``branch.active is
    False``) and the failed branches' contribution to graph connectivity
    are removed; the result captures the post-failure islands that the
    priority-invariant check needs in order to compare tiers fairly
    within each connected partition.

    ``sector``: when set, restrict the subgraph to same-sector branches
    only (CP couplings excluded).  Default ``None`` preserves the
    legacy full-graph behaviour for callers that don't yet pass a
    sector — but :func:`served_by_load` now stamps each row's
    ``component`` from the **sector-specific** map so the
    ``priority_invariant`` claim aggregates on the same scope L2's
    coordinator election uses.

    Disconnected single nodes get their own component.  Component
    indices are arbitrary but stable within a single call (assigned in
    discovery order via union-find).
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
    """Stringify a component label for the CSV ``component`` column.

    Sector-aware labels are prefixed (``"electricity:0"``) so the
    ``priority_invariant`` aggregator naturally groups per-sector even
    if a future caller mixes legacy (un-prefixed) and per-sector rows.
    The unprefixed form is preserved for ``sector is None`` so existing
    artefacts (and the strict legacy fallback) round-trip unchanged.
    """
    if sector is None:
        return str(comp_idx)
    return f"{sector}:{comp_idx}"


def _branch_is_active(branch: Any) -> bool:
    """``branch.model.active`` when present, falling back to
    ``branch.active``.  Mirrors :meth:`Network._set_active` so we read
    the same flag the simulator writes to."""
    model = getattr(branch, "model", None)
    if model is not None and "active" in getattr(model, "vars", {}):
        return bool(model.active)
    return bool(getattr(branch, "active", True))


# Heat "served" at a node whose temperature is outside the hard heat
# bounds the oracle LP enforces (``SECTOR_CONSTRAINTS[HEAT]["t_k"]``) is
# not physically valid service — the centralised baseline refuses to
# serve it.  The live sim only enforces ``t_k`` softly (reactive
# GridConstraintMonitor), so without this gate it counts heat served at
# infeasible node temperatures and "beats" the temperature-constrained
# oracle (the CP-heavy ``post > baseline`` ratios).  A small tolerance
# keeps marginal numerical wiggle at the bound from over-firing while
# still catching gross violations (cold-day nodes solve to ~240 K).
_HEAT_T_TOL_K: float = 1.0


def _heat_served_feasible(obs: dict, sec: Any) -> bool:
    """False iff this is a heat load whose node temperature is outside the
    hard ``t_k`` bounds the oracle enforces (within ``_HEAT_T_TOL_K``).
    Non-heat loads, missing / non-finite readings, and configs without a
    heat ``t_k`` bound all pass (never gate on absent data)."""
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


def served_by_load(
    monee_net: Any,
    behavior: Any,
    priorities: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Per-load served / demand / fraction with node + component tag.

    Returns a list of dicts ready for CSV-writing or per-component
    aggregation.  Shape:

    ``[{aid, sector, tier, node_id, component, demand, served, fraction,
        disconnected}, ...]``

    Loads on physically disconnected nodes get ``served = 0``,
    matching :func:`served_breakdown`'s contract.  ``component`` is the
    active-subgraph component index from :func:`_active_node_components`
    so the priority-invariant claim can group by it.
    """
    disconnected = _disconnected_node_ids(monee_net)
    # Per-sector component maps so each row's ``component`` label
    # matches the L2 coordinator-election scope (sector subgraph), not
    # the legacy full-graph view that merged electricity sub-islands
    # via CP couplings.  See ``_branch_carries_sector`` for the
    # rationale and the eval_full_small_20260529-181310 failing tasks
    # (52/88/132/133) for the manifestation.
    components_by_sector: dict[str, dict[Any, int]] = {
        sec.value: _active_node_components(monee_net, sector=sec.value)
        for sec in Sector
    }

    _LOAD_CLASSES: tuple[type, ...] = (HeatLoad, PowerLoad)

    rows: list[dict[str, Any]] = []
    for child in monee_net.childs:
        if not isinstance(child.model, _LOAD_CLASSES):
            continue
        aid = f"child-{child.id}"
        obs = behavior.observe(aid) or {}
        cap = obs_capacity(obs)
        if not (cap > 0):
            continue
        is_disconnected = (
            not getattr(child, "active", True)
            or child.node_id in disconnected
        )
        sp = 0.0 if is_disconnected else obs_setpoint(obs)
        served = max(0.0, min(cap, sp))
        node = monee_net.node_by_id(child.node_id)
        sec = sector_from_grid(node.grid)
        if sec is None:
            continue
        # Don't credit heat served at an out-of-bounds node temperature
        # (the oracle won't either) — see ``_heat_served_feasible``.
        temp_infeasible = not _heat_served_feasible(obs, sec)
        if temp_infeasible:
            served = 0.0
        if priorities is not None and aid in priorities:
            tier = int(priorities[aid])
        else:
            tier = obs_priority(obs)
        # Physical serve cap from the load's local constraints (same
        # util→fraction the gossip clamp uses).  A load served at/near
        # this cap is throttled by *physics*, not by a priority decision
        # — the physics-aware priority-invariant check excludes it the
        # way it already excludes disconnected loads.  A temperature-
        # infeasible heat load is hard-capped at 0 here (overriding the
        # tier-dependent deadband, incl. tier-1's immunity) so the
        # priority-invariant check excludes it as constraint-throttled
        # rather than reading its barrier-zeroed served as a priority shed.
        if temp_infeasible:
            allowed = 0.0
        elif is_disconnected:
            allowed = 1.0
        else:
            allowed = constraint_allowed_fraction(obs, sec, tier=tier)
        sec_components = components_by_sector.get(sec.value, {})
        comp_idx = sec_components.get(child.node_id, -1)
        rows.append({
            "aid": aid,
            "sector": sec.value,
            "tier": tier,
            "node_id": child.node_id,
            # Sector-prefixed label.  ``priority_invariant`` groups by
            # ``(sector, component)``, so the prefix is informational —
            # the grouping is correct either way — but it makes the CSV
            # self-describing under inspection.
            "component": _component_label(comp_idx, sec.value),
            "demand": cap,
            "served": served,
            "fraction": served / cap if cap > 0 else 0.0,
            "disconnected": int(is_disconnected),
            "constraint_allowed": allowed,
        })
    return rows


def served_breakdown(
    monee_net: Any,
    behavior: Any,
    priorities: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Walk every load child, compute served / demand per (sector, tier)
    and per-sector aggregates, plus the priority-weighted-served scalar.

    Shape:

    ``{
        "by_tier_sector": {sector: {tier: {demand, served, fraction}}},
        "by_sector":      {sector: {demand, served, fraction}},
        "by_tier":        {tier:   {demand, served, fraction, weight}},
        "priority_weighted_served": float,
        "priority_weighted_demand": float,
        "priority_weighted_fraction": float,
        "n_loads": int,
        "n_loads_served_zero": int
    }``

    ``priority_weighted_*`` divide by total weighted demand so the
    fraction is in [0, 1] and comparable across grids of different
    absolute size.

    Loads on nodes with no active path to a grid-forming source are
    forced to ``served = 0`` (see :func:`_disconnected_node_ids`) — the
    LP / agent state can't be trusted to reflect physical disconnect.
    """
    by_tier_sector: dict[str, dict[int, dict[str, float]]] = {}
    by_sector: dict[str, dict[str, float]] = {}
    by_tier: dict[int, dict[str, float]] = {}
    n_loads = 0
    n_zero = 0
    pw_demand = 0.0
    pw_served = 0.0

    disconnected = _disconnected_node_ids(monee_net)
    # Filter to actual *consumer* load childs.  In monee's hydraulic /
    # multi-energy models, ``Sink`` represents the return-side of a heat
    # exchanger and ``ExtHydrGrid`` / ``ExtPowerGrid`` are slack injectors —
    # none of them are customer demand, yet all three carry mass-flow /
    # p_mw fields that ``obs_capacity`` would otherwise pick up post-LP.
    # Without this filter, scare's metric inflates ``served`` past
    # baseline (validation run: post=6.21 MW vs baseline=5.20 MW on
    # cp_heavy_dependent_45; the 1 MW excess was 170 Sinks + 2 ExtHydrGrids).
    _LOAD_CLASSES: tuple[type, ...] = (HeatLoad, PowerLoad)

    for child in monee_net.childs:
        if not isinstance(child.model, _LOAD_CLASSES):
            continue
        aid = f"child-{child.id}"
        obs = behavior.observe(aid) or {}
        cap = obs_capacity(obs)
        # Skip generators (cap < 0), zero-capacity placeholders, and
        # NaN-capacity entries.  monee's ``persist_solution`` propagates
        # NaN values from ``inject_nans`` back into compound port-childs
        # (e.g.\ SubHG on a disconnected heat node) — those carry no
        # consumer demand of their own and would otherwise pollute the
        # sector totals.
        if not (cap > 0):
            continue
        n_loads += 1
        # ``is_disconnected`` distinguishes the *physical* loss path
        # (priority-blind by definition: no path to a grid-forming
        # source) from the *agent-driven* path (where the QP / ADMM
        # priority weighting actually decides who sheds).  The
        # restoration_breakdown downstream uses this to split per-tier
        # losses into the two contributions.
        is_disconnected = (
            not getattr(child, "active", True)
            or child.node_id in disconnected
        )
        if is_disconnected:
            sp = 0.0
        else:
            sp = obs_setpoint(obs)
        # Clamp to [0, cap] — a load physically can't consume more than its
        # rated demand, and the MAS variants are free to set regulation > 1.0
        # (bound is [0, 2]).  Without clamping, agents over-serving one load
        # mask genuine zero-served loads in the totals: post-restoration
        # ``served`` can exceed pre-failure baseline even when 30+ loads sit
        # disconnected.  Confirmed in the validation run on cp_heavy_45 with
        # generator failures (scare reports 6.05 MW served vs 5.20 MW baseline
        # while 30 loads are at zero).
        served = max(0.0, min(cap, sp))
        demand = cap
        # Demand on physically disconnected nodes — the priority-blind
        # share of the per-tier loss.  Baseline runs (no failures) have
        # this at zero on every tier; the restoration_breakdown uses
        # the difference to attribute losses.
        demand_disc = demand if is_disconnected else 0.0

        node = monee_net.node_by_id(child.node_id)
        sec = sector_from_grid(node.grid)
        if sec is None:
            continue
        # Don't credit heat served at an out-of-bounds node temperature
        # (matches the oracle's hard t_k bound) — see ``_heat_served_feasible``.
        if served > 0.0 and not _heat_served_feasible(obs, sec):
            served = 0.0
        if priorities is not None and aid in priorities:
            tier = int(priorities[aid])
        else:
            tier = obs_priority(obs)
        w = _tier_weight(tier)

        sec_key = sec.value
        by_sector.setdefault(sec_key, {"demand": 0.0, "served": 0.0, "demand_disconnected": 0.0})
        by_sector[sec_key]["demand"] += demand
        by_sector[sec_key]["served"] += served
        by_sector[sec_key]["demand_disconnected"] += demand_disc

        by_tier.setdefault(
            tier,
            {"demand": 0.0, "served": 0.0, "weight": w, "demand_disconnected": 0.0},
        )
        by_tier[tier]["demand"] += demand
        by_tier[tier]["served"] += served
        by_tier[tier]["demand_disconnected"] += demand_disc

        by_tier_sector.setdefault(sec_key, {})
        by_tier_sector[sec_key].setdefault(
            tier, {"demand": 0.0, "served": 0.0, "demand_disconnected": 0.0}
        )
        by_tier_sector[sec_key][tier]["demand"] += demand
        by_tier_sector[sec_key][tier]["served"] += served
        by_tier_sector[sec_key][tier]["demand_disconnected"] += demand_disc

        pw_demand += w * demand
        pw_served += w * served
        if served < 1e-9 and demand > 1e-9:
            n_zero += 1

    # Fill in fractions.
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
        "priority_weighted_demand": pw_demand,
        "priority_weighted_served": pw_served,
        "priority_weighted_fraction": (
            pw_served / pw_demand if pw_demand > 0 else 1.0
        ),
        "n_loads": n_loads,
        "n_loads_served_zero": n_zero,
    }


# ---------------------------------------------------------------------------
# Restoration breakdown — relative to the pre-failure baseline
# ---------------------------------------------------------------------------


def restoration_breakdown(
    post: dict[str, Any], baseline: dict[str, Any] | None
) -> dict[str, Any]:
    """Compare post-restoration served breakdown against the pre-failure
    baseline, producing the metrics that show "how much got lost
    despite restoration".

    ``post`` and ``baseline`` are both ``served_breakdown`` dicts
    (same shape).  When ``baseline`` is None or has zero demand the
    derived ratios collapse to ``1.0`` (no failure → no degradation).

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
                                served_post_mw, ratio (=post/baseline)},
        "by_sector": per-sector {demand_baseline_mw, served_baseline_mw,
                                  served_post_mw, ratio},
    }``

    The ``raw_*`` fields are unweighted MW so the chapter can show
    "absolute load lost despite restoration" alongside the
    priority-weighted fraction.  ``by_tier.ratio`` is what surfaces
    "did tier-1 critical loads actually get fully restored?".
    """
    if not baseline:
        return {"baseline_available": False}

    pw_demand = baseline.get("priority_weighted_demand", 0.0)
    pw_served_baseline = baseline.get("priority_weighted_served", 0.0)
    pw_served_post = post.get("priority_weighted_served", 0.0)

    sector_baseline = baseline.get("by_sector", {})
    sector_post = post.get("by_sector", {})
    total_demand = sum(s.get("demand", 0.0) for s in sector_baseline.values())
    total_served_baseline = sum(
        s.get("served", 0.0) for s in sector_baseline.values()
    )
    total_served_post = sum(s.get("served", 0.0) for s in sector_post.values())

    by_tier_b = baseline.get("by_tier", {})
    by_tier_p = post.get("by_tier", {})
    by_tier_out: dict[str, dict[str, float]] = {}
    for tier in set(by_tier_b) | set(by_tier_p):
        tb = by_tier_b.get(tier, {})
        tp = by_tier_p.get(tier, {})
        d_b = float(tb.get("demand", 0.0))
        s_b = float(tb.get("served", 0.0))
        s_p = float(tp.get("served", 0.0))
        # ``demand_disconnected`` is the priority-blind share of the
        # per-tier loss: load that physically lost its path to a
        # grid-forming source and is therefore irrecoverable regardless
        # of what the agents decide.  The rest of the per-tier loss is
        # the *agent-shed* portion — load the QP / ADMM chose to drop.
        # Splitting the two makes the chapter's priority-waterfall
        # claim verifiable: priority should drive ``agent_shed``, not
        # ``disconnect_lost``.  See P0/P1 audit + restoration
        # validation pass for context.
        disc_p = float(tp.get("demand_disconnected", 0.0))
        total_loss = max(0.0, s_b - s_p)
        disconnect_lost = max(0.0, min(disc_p, total_loss))
        # ``disconnect_lost > total_loss`` would imply we lost more to
        # physical disconnect than the total drop — arithmetically
        # impossible.  Log + clamp so the inversion is visible instead
        # of silently masked by the outer min(1, …).
        if disc_p > total_loss + 1e-12:
            logger.warning(
                "Tier %s: demand_disconnected=%.4g > total_loss=%.4g; "
                "clamping disconnect_lost to total_loss.",
                tier, disc_p, total_loss,
            )
        agent_shed = max(0.0, total_loss - disconnect_lost)
        ratio = s_p / s_b if s_b > 1e-12 else (1.0 if d_b < 1e-12 else 0.0)
        # Agent-only ratio: how would tier i look if disconnection
        # hadn't happened?  Used by the priority-awareness plot.
        s_b_recoverable = max(1e-12, s_b - disconnect_lost)
        agent_ratio = (s_b - total_loss + disconnect_lost) / s_b_recoverable
        by_tier_out[str(tier)] = {
            "demand_baseline_mw":   d_b,
            "served_baseline_mw":   s_b,
            "served_post_mw":       s_p,
            "ratio":                max(0.0, min(1.0, ratio)),
            "disconnect_lost_mw":   disconnect_lost,
            "agent_shed_mw":        agent_shed,
            "agent_only_ratio":     max(0.0, min(1.0, agent_ratio)),
            "weight":               float(tb.get("weight", 0.0)),
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
        total_loss = max(0.0, s_b - s_p)
        disconnect_lost = max(0.0, min(disc_p, total_loss))
        agent_shed = max(0.0, total_loss - disconnect_lost)
        by_sector_out[sec] = {
            "demand_baseline_mw":   d_b,
            "served_baseline_mw":   s_b,
            "served_post_mw":       s_p,
            "ratio":                max(0.0, min(1.0, ratio)),
            "disconnect_lost_mw":   disconnect_lost,
            "agent_shed_mw":        agent_shed,
        }

    # Campaign-level disconnect vs agent split — sum the per-sector
    # contributions so the aggregator + restoration plot can quote the
    # priority-blind share at a glance.
    total_disconnect_lost = sum(
        s.get("disconnect_lost_mw", 0.0) for s in by_sector_out.values()
    )
    total_agent_shed = sum(
        s.get("agent_shed_mw", 0.0) for s in by_sector_out.values()
    )

    return {
        "baseline_available":          True,
        "absolute_load_lost_mw":       max(0.0, total_demand - total_served_post),
        "absolute_load_dropped_mw":    max(0.0, total_served_baseline - total_served_post),
        "disconnect_lost_mw":          total_disconnect_lost,
        "agent_shed_mw":               total_agent_shed,
        "total_demand_mw":             total_demand,
        "total_served_baseline_mw":    total_served_baseline,
        "total_served_post_mw":        total_served_post,
        "raw_restoration_ratio":       (
            total_served_post / total_served_baseline
            if total_served_baseline > 1e-12 else 1.0
        ),
        "pwsf_baseline":               (
            pw_served_baseline / pw_demand if pw_demand > 0 else 1.0
        ),
        "pwsf_post":                   (
            pw_served_post / pw_demand if pw_demand > 0 else 1.0
        ),
        "pwsf_restoration_ratio":      (
            pw_served_post / pw_served_baseline
            if pw_served_baseline > 1e-12 else 1.0
        ),
        "by_tier":                     by_tier_out,
        "by_sector":                   by_sector_out,
    }


# ---------------------------------------------------------------------------
# Constraint violation integral
# ---------------------------------------------------------------------------


def constraint_violation_integral(world: Any) -> dict[str, float]:
    """Per-sector ``∫ max(0, util(t) − 1) dt`` proxied by the recorded
    average utilization timeseries (avg_vm_pu / avg_pressure_pu / avg_t_k).

    Recorded series store the *average* across each sector; we compute
    util from those vs the sector bounds and integrate the overshoot
    above 1.0.  The result is dimensionless and comparable across runs
    on the same grid.
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
        # Trapezoidal integration of max(0, util − 1) over time.
        integral = 0.0
        for i in range(1, len(ts)):
            u_a = constraint_utilization(float(ts[i - 1]), lo, hi)
            u_b = constraint_utilization(float(ts[i]), lo, hi)
            ov_a = max(0.0, u_a - 1.0)
            ov_b = max(0.0, u_b - 1.0)
            dt = float(ts_t[i]) - float(ts_t[i - 1])
            integral += 0.5 * (ov_a + ov_b) * dt
        out[sec.value] = integral
    return out


# ---------------------------------------------------------------------------
# Time-to-stabilise
# ---------------------------------------------------------------------------


def time_to_stabilise_s(world: Any, *, hold_s: float = 1.0) -> float | None:
    """First simulation time after which all sector imbalance series stay
    near steady state for at least ``hold_s`` seconds.

    Heuristic: the recorded ``electrical_balance / gas_balance /
    heat_balance`` series are the per-sector regulation sums.  A run is
    "stable" when the absolute first-difference of every sector's
    series is below 0.5 % of its maximum magnitude for a sustained
    window.  Returns None if no such window is found.
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
        k: max(1e-6, 0.005 * max(abs(v) for v in ts))
        for k, (_, ts) in series.items()
    }

    # Walk a common timestep grid (from the first series — they share
    # the same recording cadence).  Mark t as "stable" only if every
    # series' diff at this step is below threshold.
    ref_t, _ = next(iter(series.values()))

    def step_stable(i: int) -> bool:
        for k, (t, ts) in series.items():
            if i >= len(ts) or i < 1:
                return False
            if abs(ts[i] - ts[i - 1]) > thresholds[k]:
                return False
        return True

    stable_since: float | None = None
    for i in range(1, len(ref_t)):
        if step_stable(i):
            if stable_since is None:
                stable_since = ref_t[i]
            elif ref_t[i] - stable_since >= hold_s:
                return float(stable_since)
        else:
            stable_since = None
    return None


# ---------------------------------------------------------------------------
# Optimality gap (vs oracle)
# ---------------------------------------------------------------------------


def optimality_gap(scare_pw_served: float, oracle_pw_served: float) -> float:
    """Relative gap ``(oracle − scare) / oracle`` clipped to ``[0, 1]``.

    Computed in the aggregator after both result.jsons exist.  Negative
    inputs (oracle worse than scare — shouldn't happen but guard
    against numerical edge cases) are clipped to 0.
    """
    if oracle_pw_served <= 0:
        return 0.0
    gap = (oracle_pw_served - scare_pw_served) / oracle_pw_served
    if not math.isfinite(gap):
        return 0.0
    return max(0.0, min(1.0, gap))
