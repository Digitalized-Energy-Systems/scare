"""Post-hoc metrics computed from a finished restoration run.

The world / monee net are still in scope at the end of the runner —
this module pulls the final regulation factors and the recorded
timeseries / event ledger out of them and computes the dissertation's
primary outcome metrics.

Outputs are dicts of plain Python scalars / nested dicts so they can be
serialised straight into ``result.json`` and aggregated downstream.
"""

from __future__ import annotations

import math
from typing import Any

from scare.base.model import SECTOR_CONSTRAINTS, Sector
from scare.base.util import (
    constraint_utilization,
    obs_capacity,
    obs_priority,
    obs_setpoint,
    sector_from_grid,
)


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
        from monee.solver.core import find_ignored_nodes
    except Exception:  # pragma: no cover - monee always available in this project
        return set()
    try:
        return set(find_ignored_nodes(monee_net))
    except Exception:
        return set()


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
    _LOAD_CLASSES: tuple[type, ...] | None = None
    try:
        from monee.model.child import HeatLoad, PowerLoad
        _LOAD_CLASSES = (HeatLoad, PowerLoad)
    except Exception:  # pragma: no cover
        _LOAD_CLASSES = None

    for child in monee_net.childs:
        if _LOAD_CLASSES is not None and not isinstance(child.model, _LOAD_CLASSES):
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
    from mango.simulation.world import WorldRecording

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
    from mango.simulation.world import WorldRecording

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
