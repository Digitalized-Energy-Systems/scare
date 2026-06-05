"""Supply-priority ADMM allocator shared by L2 (chunk-mate) and L2.5
(coalition) roles.

Mechanism: for each (sector, tier) cell the coordinator pulls Σ_g x_g
toward total demand ``T[s, t]`` with an L1 distance penalty weighted by
priority. Each actor's per-cell upper bound caps its contribution, and a
per-actor coupling ``Σ x_g ≤ supply_g`` prevents committing more than it
holds. Priority weighting plus the share-scaled S coefficient bias toward
high-priority cells, so under scarcity high-priority tiers serve first.

Deliverability: ``actor_ub_overrides`` caps ``ub[(sec, tier)]`` below the
raw supply-vs-demand minimum per actor/cell, modelling that supply only
reaches loads at physically-reachable nodes after failures. ``None``/
missing is a no-op.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    create_admm_sharing_data,
    create_admm_start,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.flex_actor import (
    ADMMFlexActor,
)

from scare.base.util import tier_priority_weight_strict

logger = logging.getLogger(__name__)


def _waterfall_target(
    demand_per_cell: np.ndarray,
    priorities: np.ndarray,
    supply_pool: float,
) -> np.ndarray:
    """Priority-waterfall allocation of ``supply_pool`` across cells.

    Visits cells in descending-priority order, assigning
    ``min(demand_cell, remaining_pool)`` until the pool is exhausted.
    Used as the ADMM target when ``sum(demand) > supply_pool`` so the
    primal residual can reach zero. Sorts on ``priorities[]`` directly,
    so any monotone-in-priority weighting works.
    """
    out = np.zeros_like(demand_per_cell, dtype=float)
    remaining = float(supply_pool)
    if remaining <= 0.0:
        return out
    # Sort by priority weight DESC; stable so ties break by cell index.
    order = np.argsort(-priorities, kind="stable")
    for j in order:
        if remaining <= 1e-12:
            break
        take = min(float(demand_per_cell[j]), remaining)
        if take <= 0.0:
            continue
        out[j] = take
        remaining -= take
    return out


async def allocate_supply_priority(
    *,
    sectors: list[str],
    tiers: list[int],
    actor_supplies: list[dict[str, float]],
    actor_demands: list[dict[str, dict[int, float]]],
    actor_ub_overrides: list[dict[tuple[str, int], float] | None] | None = None,
    priority_tiers: int = 4,
    max_iters: int = 50,
    abs_tol: float = 1e-3,
    enable_priority_weighting: bool = True,
    cp_coupling: list[tuple[int, str, str, float]] | None = None,
) -> tuple[
    dict[str, dict[int, float]],
    list[list[float]],
    dict[str, Any],
]:
    """Run the supply-priority ADMM on the given actor flex slices.

    Parameters
    ----------
    sectors
        Sector value strings (e.g. ``"electricity"``). Non-empty.
    tiers
        Priority tier integers (1 = highest). Non-empty; demand at
        tiers outside this list is ignored.
    actor_supplies
        Per actor: ``{sector_value: total_supply_mw}``. Sector-scalar —
        supply is fungible across tiers within a sector.
    actor_demands
        Per actor (positional with actor_supplies):
        ``{sector_value: {tier: demand_mw}}``. Used to compute target ``T``.
    actor_ub_overrides
        Optional per-actor ``{(sector_value, tier): cap_mw}``; sets
        ``ub[(sec, tier)] = min(raw_ub, cap_mw)`` to model deliverability.
        Use cap 0 (→ 1e-6 for conditioning) when actor cannot reach a cell.
    priority_tiers
        ``P`` in the strict-monotone schedule ``P − tier + 1`` (see
        ``base.util.tier_priority_weight_strict``).
    max_iters, abs_tol
        Coordinator convergence knobs.
    cp_coupling
        Optional ``(actor_index, sector_in, sector_out, ratio)`` tuples.
        Each adds two inequality rows pinning ``Σ_t x[out] = ratio · Σ_t
        x[in]`` for that actor: ``sector_in`` is consumed, ``sector_out``
        produced. A P2H with η=0.9 passes ``(cp_idx, "electricity",
        "heat", 0.9)``. None/empty ⇒ no coupling rows.

    Returns
    -------
    service_fraction
        ``{sector: {tier: fraction_in_[0,1]}}`` — per-cell global service
        fraction.
    per_actor_x
        Per-actor flat allocation vectors (length ``n_sectors * n_tiers``).
    meta
        Diagnostics (``T_per_cell``, ``sum_x_per_cell``,
        ``holon_supply_total``, ``priorities``, ``degenerate``, ...).

    Raises
    ------
    ValueError
        On empty sectors/tiers or mismatched actor list lengths.
    """
    if not sectors:
        raise ValueError("supply-priority ADMM: sectors must be non-empty")
    if not tiers:
        raise ValueError("supply-priority ADMM: tiers must be non-empty")
    if len(actor_supplies) != len(actor_demands):
        raise ValueError(
            "supply-priority ADMM: actor_supplies and actor_demands lengths differ"
        )
    if (
        actor_ub_overrides is not None
        and len(actor_ub_overrides) != len(actor_supplies)
    ):
        raise ValueError(
            "supply-priority ADMM: actor_ub_overrides length must match actor count"
        )

    n_sec = len(sectors)
    n_tier = len(tiers)
    n_dims = n_sec * n_tier
    sec_idx = {s: i for i, s in enumerate(sectors)}
    tier_idx = {t: j for j, t in enumerate(tiers)}

    def _flat_idx(s: str, t: int) -> int:
        return sec_idx[s] * n_tier + tier_idx[t]

    # Total demand per (sector, tier) — the semantic target. The final
    # service fraction is committed/demand against this.
    total_demand_per_cell = np.zeros(n_dims)
    for demand in actor_demands:
        for sec, tier_to_dem in demand.items():
            if sec not in sec_idx:
                continue
            for tier, dem in tier_to_dem.items():
                if tier in tier_idx:
                    total_demand_per_cell[_flat_idx(sec, tier)] += float(dem)
    # ADMM target; rewritten below to a feasible (waterfall) version when
    # supply < demand. Kept separate from total_demand_per_cell so the
    # service-fraction semantics stay against real demand.
    total_T = total_demand_per_cell.copy()

    if not np.any(np.abs(total_T) >= 1e-6):
        # No demand anywhere: return all-1.0 fractions (no-op).
        return (
            {sec: {tier: 1.0 for tier in tiers} for sec in sectors},
            [[0.0] * n_dims for _ in actor_supplies],
            {
                "T_per_cell": total_T.tolist(),
                "sum_x_per_cell": [0.0] * n_dims,
                "actor_supply_total": [
                    sum(float(v) for v in s.values()) for s in actor_supplies
                ],
                "holon_supply_total": 0.0,
                "priorities": [0.0] * n_dims,
                "sectors": list(sectors),
                "tiers": list(tiers),
                "degenerate": True,
            },
        )

    priorities = np.zeros(n_dims)
    for tier in tiers:
        # Strictly-monotone tier weight (tier 1 → P, tier P → 1) for the
        # waterfall sort and S-coefficient. Not the L1 QP schedule: that
        # returns 0 for tier 1 and 1e8 for tier 2, which would mis-sort the
        # waterfall and destabilise the sharing-distance dual. Range [1, P]
        # keeps the sort correct (tier 1 first) and well-conditioned.
        if enable_priority_weighting:
            weight = tier_priority_weight_strict(
                tier, priority_tiers=priority_tiers,
            )
        else:
            weight = 1.0
        for sec in sectors:
            priorities[_flat_idx(sec, tier)] = weight

    holon_supply_total = sum(
        sum(float(v) for v in s.values()) for s in actor_supplies
    )

    # No-supply branch (orphan island split off from grid-forming
    # sources): shed every load. Returning fraction 1.0 here would let
    # the slack backstopping the island stay over-budget. After shedding,
    # the SlackBudgetMonitor feedback re-opens the effective budget and
    # restores whatever service the backstop allows. Slack-budget-as-
    # supply is intentionally not modelled here (larger refactor).
    if holon_supply_total <= 0.0:
        logger.debug(
            "supply-priority ADMM degenerate no-supply: sheds everything "
            "(n_actors=%d, demand_sum=%.4f)",
            len(actor_supplies), float(total_demand_per_cell.sum()),
        )
        return (
            {sec: {tier: 0.0 for tier in tiers} for sec in sectors},
            [[0.0] * n_dims for _ in actor_supplies],
            {
                "T_per_cell": [0.0] * n_dims,
                "demand_per_cell": total_demand_per_cell.tolist(),
                "sum_x_per_cell": [0.0] * n_dims,
                "actor_supply_total": [0.0] * len(actor_supplies),
                "holon_supply_total": 0.0,
                "priorities": [0.0] * n_dims,
                "sectors": list(sectors),
                "tiers": list(tiers),
                "degenerate_no_supply": True,
            },
        )

    # Feasibility cap on the ADMM target. When demand > supply, the
    # sharing-distance term ``||Σ_g x_g − T||`` can never close (the box
    # ub + coupling Σ x_g ≤ supply_g clamp Σ x_g at the pool size), so the
    # residual plateaus on the structural gap and the solver spuriously
    # hits max-iterations. Replacing T with the priority-waterfall target
    # gives ``sum(T) == holon_supply_total`` and the priority-correct
    # per-cell values, so ADMM converges with zero structural residual.
    # Per-actor split remains ADMM's job under the box + coupling.
    total_demand_sum = float(total_demand_per_cell.sum())
    if total_demand_sum > holon_supply_total > 0:
        total_T = _waterfall_target(
            total_demand_per_cell, priorities, holon_supply_total,
        )
        logger.debug(
            "supply-priority ADMM target reset to priority waterfall "
            "(demand_sum=%.4f, holon_supply=%.4f, T_sum=%.4f)",
            total_demand_sum, holon_supply_total, float(total_T.sum()),
        )

    # Waterfall short-circuit: with no per-actor reach restrictions and no
    # CP coupling, ``total_T`` already IS the priority-optimal per-cell
    # allocation and the dispatch layer only reads ``service_fraction =
    # T / demand``, never per-actor ``x``. Running the ADMM here only
    # solves the unneeded per-actor split, which its share-weighted S term
    # solves badly (small actors chronically under-commit, so Σ x < T).
    if (
        actor_ub_overrides is None
        and (cp_coupling is None or len(cp_coupling) == 0)
        and holon_supply_total > 0
    ):
        service_fraction: dict[str, dict[int, float]] = {}
        for sec in sectors:
            service_fraction.setdefault(sec, {})
            for tier in tiers:
                j = _flat_idx(sec, tier)
                demand_cell = float(total_demand_per_cell[j])
                target_cell = float(total_T[j])
                frac = (
                    1.0 if demand_cell <= 1e-9
                    else min(1.0, max(0.0, target_cell / demand_cell))
                )
                service_fraction[sec][tier] = frac
        logger.debug(
            "supply-priority ADMM waterfall short-circuit: "
            "supply=%.4f, demand=%.4f, T_sum=%.4f",
            holon_supply_total, total_demand_sum, float(total_T.sum()),
        )
        return (
            service_fraction,
            [[0.0] * n_dims for _ in actor_supplies],
            {
                "T_per_cell": total_T.tolist(),
                "demand_per_cell": total_demand_per_cell.tolist(),
                "sum_x_per_cell": total_T.tolist(),
                "actor_supply_total": [
                    sum(float(v) for v in s.values()) for s in actor_supplies
                ],
                "holon_supply_total": holon_supply_total,
                "priorities": priorities.tolist(),
                "sectors": list(sectors),
                "tiers": list(tiers),
                "degenerate": False,
                "short_circuit": "waterfall",
            },
        )

    # Index cp_coupling entries by actor to extend each actor's (C, d).
    cp_couplings_by_actor: dict[int, list[tuple[str, str, float]]] = {}
    for entry in (cp_coupling or []):
        try:
            g_idx, sec_in, sec_out, ratio = entry
        except (TypeError, ValueError):
            continue
        if sec_in not in sec_idx or sec_out not in sec_idx:
            # Skip couplings referencing sectors outside this ADMM scope
            # (e.g. a P2G when gas is absent).
            continue
        cp_couplings_by_actor.setdefault(int(g_idx), []).append(
            (str(sec_in), str(sec_out), float(ratio))
        )

    actors: list[ADMMFlexActor] = []
    actor_supply_total: list[float] = []
    for g, supply_map in enumerate(actor_supplies):
        lb = np.zeros(n_dims)
        ub = np.full(n_dims, 1e-6)
        overrides = (
            actor_ub_overrides[g] if actor_ub_overrides is not None else None
        )
        for sec in sectors:
            supply_s = float(supply_map.get(sec, 0.0))
            for tier in tiers:
                j = _flat_idx(sec, tier)
                raw_cap = min(supply_s, total_demand_per_cell[j])
                if overrides is not None:
                    cap = overrides.get((sec, tier))
                    if cap is not None:
                        raw_cap = min(raw_cap, float(cap))
                # Waterfall-shed cell (T[j] = 0): force ub to exact 0 so
                # the local QP cannot leak supply into a dropped cell.
                # Otherwise round-off lets actors deposit tiny amounts that
                # anchor z[j] > 0 and pass the convergence check on a leaked
                # allocation. Exact 0 (bypassing the 1e-6 conditioning floor)
                # is safe: 0 ≤ x ≤ 0 is feasible for OSQP and pins x exactly.
                if total_T[j] <= 1e-12:
                    ub[j] = 0.0
                else:
                    ub[j] = max(raw_cap, 1e-6)

        # Per-actor coupling Σ commitments ≤ total supply; creates the
        # scarcity binding when supply < demand.
        total_supply = sum(float(v) for v in supply_map.values())
        actor_supply_total.append(total_supply)
        C_rows = [np.ones(n_dims)]
        d_rows = [max(total_supply, 0.0)]

        # CP coupling: pin ``Σ_t x[out] = ratio · Σ_t x[in]`` as a pair of
        # inequalities so output commit tracks input draw under conversion
        # efficiency. No-op for non-CP actors.
        for sec_in, sec_out, ratio in cp_couplings_by_actor.get(g, []):
            row_out_minus_r_in = np.zeros(n_dims)
            for tier in tiers:
                row_out_minus_r_in[_flat_idx(sec_out, tier)] += 1.0
                row_out_minus_r_in[_flat_idx(sec_in, tier)] -= ratio
            # out − ratio · in ≤ 0
            C_rows.append(row_out_minus_r_in.copy())
            d_rows.append(0.0)
            # ratio · in − out ≤ 0  (same row, negated)
            C_rows.append(-row_out_minus_r_in)
            d_rows.append(0.0)

        C = np.vstack(C_rows)
        d = np.array(d_rows, dtype=float)

        # S biases each actor toward high-priority cells; scaled by pool
        # share so larger pools have a stronger preference signal.
        share = total_supply / holon_supply_total
        S = -share * priorities

        lb = np.nan_to_num(lb, nan=0.0, posinf=0.0, neginf=0.0)
        ub = np.nan_to_num(ub, nan=1e-6, posinf=1e6, neginf=1e-6)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        actors.append(ADMMFlexActor(lb=lb, u=ub, C=C, d=d, S=S))

    coordinator = create_sharing_target_distance_admm_coordinator()
    coordinator.max_iters = int(max_iters)
    coordinator.abs_tol = float(abs_tol)
    start_msg = create_admm_start(
        create_admm_sharing_data(
            total_T.tolist(), priorities=priorities.tolist()
        )
    )

    await start_coordinated_optimization(actors, coordinator, start_msg)

    results = [a.x.tolist() for a in actors]
    sum_x_per_cell = (
        np.sum(np.array(results), axis=0) if results else np.zeros(n_dims)
    )

    # Service fraction = committed / original demand (not the scaled
    # feasibility target ``total_T``).
    service_fraction: dict[str, dict[int, float]] = {}
    for sec in sectors:
        service_fraction.setdefault(sec, {})
        for tier in tiers:
            j = _flat_idx(sec, tier)
            demand_cell = float(total_demand_per_cell[j])
            committed = float(sum_x_per_cell[j])
            frac = (
                1.0 if demand_cell <= 1e-9
                else min(1.0, max(0.0, committed / demand_cell))
            )
            service_fraction[sec][tier] = frac

    meta = {
        "T_per_cell": total_T.tolist(),
        "demand_per_cell": total_demand_per_cell.tolist(),
        "sum_x_per_cell": sum_x_per_cell.tolist(),
        "actor_supply_total": actor_supply_total,
        "holon_supply_total": holon_supply_total,
        "priorities": priorities.tolist(),
        "sectors": list(sectors),
        "tiers": list(tiers),
        "degenerate": False,
    }

    # Per-cell input/output breakdown at DEBUG, tagged PRIPROBE for grep.
    if logger.isEnabledFor(logging.DEBUG):
        n_actors = len(actor_supplies)
        ub_per_actor: list[list[float]] = []
        if actor_ub_overrides is not None:
            for g, ov in enumerate(actor_ub_overrides):
                row = []
                for sec in sectors:
                    for tier in tiers:
                        row.append(
                            float(ov.get((sec, tier), float("inf"))) if ov else float("inf")
                        )
                ub_per_actor.append(row)
        for sec in sectors:
            for tier in tiers:
                j = _flat_idx(sec, tier)
                d = float(total_demand_per_cell[j])
                t = float(total_T[j])
                x = float(sum_x_per_cell[j])
                ov_row = ",".join(
                    f"{ub_per_actor[g][j]:.4g}" if ub_per_actor else "inf"
                    for g in range(n_actors)
                ) if ub_per_actor else ""
                logger.debug(
                    "PRIPROBE sec=%s tier=%d demand=%.6f T=%.6f x=%.6f frac=%.4f ov_per_actor=[%s]",
                    sec, tier, d, t, x,
                    service_fraction[sec][tier], ov_row,
                )
        logger.debug(
            "PRIPROBE_SUMMARY n_actors=%d holon_supply=%.4f demand_sum=%.4f T_sum=%.4f x_sum=%.4f sectors=%s tiers=%s",
            n_actors, holon_supply_total,
            float(total_demand_per_cell.sum()),
            float(total_T.sum()),
            float(sum_x_per_cell.sum()),
            sectors, tiers,
        )

    return service_fraction, results, meta
