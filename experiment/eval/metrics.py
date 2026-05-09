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
    """
    by_tier_sector: dict[str, dict[int, dict[str, float]]] = {}
    by_sector: dict[str, dict[str, float]] = {}
    by_tier: dict[int, dict[str, float]] = {}
    n_loads = 0
    n_zero = 0
    pw_demand = 0.0
    pw_served = 0.0

    for child in monee_net.childs:
        aid = f"child-{child.id}"
        obs = behavior.observe(aid) or {}
        cap = obs_capacity(obs)
        if cap <= 0:
            # Generators (cap < 0) and unknown agents don't contribute
            # to load served.  Restoration metric is load-side only.
            continue
        n_loads += 1
        sp = obs_setpoint(obs)
        served = max(0.0, sp)
        demand = cap

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
        by_sector.setdefault(sec_key, {"demand": 0.0, "served": 0.0})
        by_sector[sec_key]["demand"] += demand
        by_sector[sec_key]["served"] += served

        by_tier.setdefault(tier, {"demand": 0.0, "served": 0.0, "weight": w})
        by_tier[tier]["demand"] += demand
        by_tier[tier]["served"] += served

        by_tier_sector.setdefault(sec_key, {})
        by_tier_sector[sec_key].setdefault(
            tier, {"demand": 0.0, "served": 0.0}
        )
        by_tier_sector[sec_key][tier]["demand"] += demand
        by_tier_sector[sec_key][tier]["served"] += served

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
