"""Pure-compute helper for the supply-priority ADMM allocation.

Lifted from :meth:`HolonicCommunityRole._run_supply_priority_admm` so
both L2 (chunk-mate ADMM in ``HolonicCommunityRole``) and L2.5 (coalition
ADMM in ``HolonSummaryRole``) can call the same allocator.  Keeping it
as a free coroutine makes it directly unit-testable and isolates the
arithmetic from the per-role flex-collection / dispatch wiring.

Mechanism
---------

For each (sector, tier) cell the coordinator pulls Σ_g x_g toward the
total demand ``T[s, t]`` with the L1 distance penalty weighted by
priority (``2^(P − tier + 1)``).  Each actor's per-cell upper bound
caps its contribution to that cell, and a single per-actor coupling
``Σ_{s, t} x_g[s, t] ≤ supply_g`` enforces that an actor cannot
commit more than it physically holds.  Priority weighting plus the
share-scaled S coefficient bias the per-actor solution toward
high-priority cells, so under scarcity the high-priority tiers get
served first.

Deliverability hook
-------------------

The coalition layer needs to model that an actor's supply can only
reach demand cells whose loads sit at physically-reachable nodes after
failures.  The ``actor_ub_overrides`` parameter lets the caller cap
``ub[(sec, tier)]`` below the raw supply-vs-demand minimum on a
per-actor, per-cell basis.  When ``None`` or missing for a cell the
override is a no-op — preserving the original L2 semantics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _waterfall_target(
    demand_per_cell: np.ndarray,
    priorities: np.ndarray,
    supply_pool: float,
) -> np.ndarray:
    """Priority-waterfall allocation of ``supply_pool`` across cells.

    Visits cells in descending-priority order, assigning
    ``min(demand_cell, remaining_pool)`` to each, stopping once the
    pool is exhausted.  Returns the per-cell target the ADMM should
    track when ``sum(demand) > supply_pool`` so the primal residual
    can drop to zero (the structural gap that produces the misleading
    "ADMM reached max iterations" warnings on deficit-bearing holons).

    Lower tier number = higher priority in the SCARE schedule, but
    the helper sorts on ``priorities[]`` directly so the caller can
    pass any monotone-in-priority weighting.
    """
    out = np.zeros_like(demand_per_cell, dtype=float)
    remaining = float(supply_pool)
    if remaining <= 0.0:
        return out
    # Sort cells by priority weight DESC.  np.argsort is ascending, so
    # negate to flip; stable so ties break by cell index for
    # reproducibility.
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
    priority_tiers: int = 10,
    max_iters: int = 50,
    abs_tol: float = 1e-3,
    enable_priority_weighting: bool = True,
) -> tuple[
    dict[str, dict[int, float]],
    list[list[float]],
    dict[str, Any],
]:
    """Run the supply-priority ADMM on the given actor flex slices.

    The mango role event loop is already async, so callers inside an
    ``async def`` handler ``await`` this coroutine directly.

    Parameters
    ----------
    sectors
        Sector value strings (e.g. ``"electricity"``) the allocation
        spans.  Must be non-empty.
    tiers
        Priority tier integers (1 = highest priority).  Must be non-
        empty.  Demand at tiers outside this list is ignored.
    actor_supplies
        One entry per actor: ``{sector_value: total_supply_mw}``.
        Sector-scalar — within a sector, supply is fungible across
        tiers (the per-cell ub cap models this).
    actor_demands
        One entry per actor (positionally matched to actor_supplies):
        ``{sector_value: {tier: demand_mw}}``.  Used to compute the
        target ``T``.
    actor_ub_overrides
        Optional per-actor ``{(sector_value, tier): cap_mw}`` map.
        When present, ``ub[(sec, tier)]`` for that actor is set to
        ``min(raw_ub, cap_mw)``.  Use this to model deliverability:
        if actor ``g`` cannot route to a tier-t demand region after a
        failure, set the cap to 0 (modelled as 1e-6 to keep the
        solver well-conditioned).
    priority_tiers
        ``P`` in the ``2^(P − tier + 1)`` priority weight, mirroring
        ``base.util.obs_priority``'s tier range.
    max_iters, abs_tol
        Coordinator convergence knobs.

    Returns
    -------
    service_fraction
        ``{sector: {tier: fraction_in_[0,1]}}`` — the global service
        fraction the ADMM converged on per cell.
    per_actor_x
        Per-actor flat allocation vectors (length ``len(sectors) *
        len(tiers)``).
    meta
        Diagnostics: ``T_per_cell``, ``sum_x_per_cell``,
        ``actor_supply_total``, ``holon_supply_total``,
        ``priorities``, ``sectors``, ``tiers``, ``degenerate``.

    Raises
    ------
    ValueError
        On empty sectors or tiers, or mismatched actor list lengths.
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

    from distributed_resource_optimization import (
        create_admm_sharing_data,
        create_admm_start,
        create_sharing_target_distance_admm_coordinator,
        start_coordinated_optimization,
    )
    from distributed_resource_optimization.algorithm.admm.flex_actor import (
        ADMMFlexActor,
    )

    n_sec = len(sectors)
    n_tier = len(tiers)
    n_dims = n_sec * n_tier
    sec_idx = {s: i for i, s in enumerate(sectors)}
    tier_idx = {t: j for j, t in enumerate(tiers)}

    def _flat_idx(s: str, t: int) -> int:
        return sec_idx[s] * n_tier + tier_idx[t]

    # Total demand per (sector, tier) across all actors — the
    # *semantic* target.  Used at the bottom of this function to
    # compute the service fraction (committed / demand) which is what
    # the dispatch layer needs.
    total_demand_per_cell = np.zeros(n_dims)
    for demand in actor_demands:
        for sec, tier_to_dem in demand.items():
            if sec not in sec_idx:
                continue
            for tier, dem in tier_to_dem.items():
                if tier in tier_idx:
                    total_demand_per_cell[_flat_idx(sec, tier)] += float(dem)
    # ADMM target ``total_T`` defaults to total demand.  It is rewritten
    # below to a *feasible* version when supply < demand, so the
    # sharing-distance term has a reachable target.  Keeping it
    # separate from ``total_demand_per_cell`` preserves the per-cell
    # service-fraction semantics for the dispatch layer.
    total_T = total_demand_per_cell.copy()

    if not np.any(np.abs(total_T) >= 1e-6):
        # Degenerate: no demand anywhere in the requested cells.
        # Return all-1.0 fractions (a no-op) and let the caller skip
        # the dispatch.
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

    from scare.base.util import tier_priority_weight

    priorities = np.zeros(n_dims)
    for tier in tiers:
        # Restoration-regime weight (high-priority tier → high weight)
        # via the shared helper — keeps the supply-priority ADMM, the
        # tier-stratified ADMM, the L1 QP, and the coalition allocator
        # on a single per-tier schedule.  When priority weighting is
        # disabled (ablation knob), every tier gets weight 1.0 so the
        # ADMM redistributes supply uniformly across cells; the
        # waterfall feasibility cap below then becomes a flat
        # demand-proportional shave.
        if enable_priority_weighting:
            weight = tier_priority_weight(
                tier, regime=1, priority_tiers=priority_tiers,
            )
        else:
            weight = 1.0
        for sec in sectors:
            priorities[_flat_idx(sec, tier)] = weight

    holon_supply_total = (
        sum(sum(float(v) for v in s.values()) for s in actor_supplies)
        or 1.0
    )

    # --- Feasibility cap on the ADMM target ---
    # When demand exceeds available supply (the common case on a
    # deficit-bearing holon), the sharing-distance term
    # ``|| Σ_g x_g − T ||`` can never close because the per-actor
    # box (ub) and the per-actor coupling (Σ x_g ≤ supply_g) clamp
    # ``Σ x_g`` at the supply pool size.  The primal residual then
    # plateaus at the structural gap, the dual residual collapses to
    # zero (z stops moving), and the library logs a spurious "ADMM
    # reached max iterations" warning — the optimum is correctly
    # found but the convergence test never accepts it.
    #
    # The fix replaces ``T`` (demand everywhere) with the
    # priority-waterfall target: starting from the highest-priority
    # cell, allocate ``min(demand_cell, remaining_supply)`` until the
    # pool is exhausted.  The resulting target has
    # ``sum(T) == holon_supply_total`` and per-cell values that match
    # the priority-correct answer exactly.  ADMM then converges with
    # zero structural residual; the per-actor allocation (which actor
    # contributes how much to each cell) is still ADMM's job under
    # the box + coupling constraints.
    #
    # The original demand is retained in ``total_demand_per_cell`` so
    # the service-fraction calculation at the bottom is
    # committed-divided-by-demand, which is what the dispatch layer
    # expects.
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
                ub[j] = max(raw_cap, 1e-6)

        # Per-actor coupling: Σ commitments ≤ total supply.  Binding
        # constraint that creates scarcity when supply < demand.
        total_supply = sum(float(v) for v in supply_map.values())
        actor_supply_total.append(total_supply)
        C = np.ones((1, n_dims))
        d = np.array([max(total_supply, 0.0)])

        # S biases each actor toward high-priority cells.  Magnitude
        # scaled by share of pool supply so larger pools have a
        # stronger preference signal.
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

    # Service fraction = committed / *original* demand per cell.  The
    # ADMM target ``total_T`` may have been scaled down for
    # feasibility, but the dispatch layer needs the fraction of real
    # demand actually served, not the fraction of the scaled target.
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

    # Per-cell breakdown of the allocator's inputs and outputs.  Kept
    # at DEBUG so it stays grep-able from run.log when investigating a
    # priority-invariant regression but doesn't dominate the file in
    # normal runs.  Lines are tagged with a recognisable prefix.
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
