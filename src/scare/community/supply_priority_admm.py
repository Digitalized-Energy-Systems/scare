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
        ``P`` in the strict-monotone schedule ``P − tier + 1`` (see
        ``base.util.tier_priority_weight_strict``).  4 by default,
        matching the new tier model.
    max_iters, abs_tol
        Coordinator convergence knobs.
    cp_coupling
        Optional list of ``(actor_index, sector_in, sector_out, ratio)``
        tuples expressing cross-sector coupling at coupling-point (CP)
        actors.  For each entry the actor's per-cell vector gets two
        extra inequality rows in its ``(C, d)`` constraint matrix:
        ``Σ_t x[sector_out, t] − ratio · Σ_t x[sector_in, t] ≤ 0`` and
        the reversed sign — together expressing the equality ``output
        = ratio · input``.  Convention: ``sector_in`` is the CP's
        input (consumes from this sector), ``sector_out`` is the
        output (produces into this sector).  A P2H bridging
        electricity → heat with η = 0.9 passes
        ``(cp_idx, "electricity", "heat", 0.9)``.  None or empty
        list ⇒ no coupling rows added (legacy behaviour).

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

    priorities = np.zeros(n_dims)
    for tier in tiers:
        # Strictly-monotone tier weight (tier 1 → P, tier P → 1) for the
        # waterfall sort and ADMM S-coefficient.  We deliberately do NOT
        # use the L1 QP schedule here: that schedule returns weight 0
        # for tier 1 (because tier 1 is hard-locked off-QP at L1) and
        # 1e8 for tier 2 (because L1 wants effectively-strict precedence
        # on a single per-cell variable).  Bringing those magnitudes
        # into the L2 ADMM would (a) sort tier-1 cells AFTER tier-2
        # cells in the waterfall — wrong; (b) destabilise the sharing-
        # distance objective whose dual scales with Σ a_j.  The strict-
        # monotone schedule keeps the sort correct (tier 1 first) and
        # the magnitudes well-conditioned (range ``[1, P]``).
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

    # Degenerate no-supply branch.  When every actor reports zero
    # supply (the orphan-island case after a failure splits a sub-
    # component off the grid-forming sources), the legacy code fell
    # back to ``or 1.0`` — silently substituting a 1 MW phantom pool
    # that produced ``service_fraction == 1.0`` across every tier
    # via the waterfall short-circuit below.  In a real orphan island
    # this means the L2 dispatches "serve everything" → no shed →
    # the slack backstopping the island stays over-budget.
    #
    # Evidence (eval_full_small_20260529-181310/tasks/000088):
    # child-12's orphan sub-coord reports ``supply=0.0000`` every
    # round; the phantom pool produced ``T2=T3=T4=1.0``; child-12's
    # 7 leaders held all 13 island electricity loads at fraction 1.0;
    # the slack ``child-39`` (the LV ext-grid feeding the whole
    # island) settled at +10.6% over budget — the breach the
    # remainder of the eval picked up after Bug 1/2/3/4 closed the
    # primary inversions.
    #
    # Correct behaviour: a truly no-supply sub-component sheds every
    # load.  The ``SlackBudgetMonitor`` feedback path then sees the
    # restored balance and re-opens the effective budget within
    # ``_FEEDBACK_GAIN`` rounds, restoring whatever service the
    # backstop budget allows.  We deliberately do NOT fall back to
    # "include the slack budget as supply" here — that requires
    # plumbing ``HolonSummary.slack_budget_by_sector`` into this
    # allocator (a larger refactor) and is the architecturally
    # cleaner follow-up.  Shedding-then-recovery is the safe minimum.
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

    # Waterfall short-circuit: when there are no per-actor reach
    # restrictions and no CP coupling, the ``total_T`` vector IS the
    # priority-optimal allocation per cell, regardless of how it
    # distributes across actors.  The dispatch layer only reads
    # ``service_fraction = T / demand``, never the per-actor ``x``.
    # The ADMM would otherwise be solving "which actor commits how
    # much to each cell" — a degree of freedom we don't actually
    # need, and one its share-weighted S term solves badly (small
    # actors with weak priority bias chronically under-commit, so
    # ``Σ x`` converges well below ``T``; 2026-05-24 deficit audit
    # showed waterfall T = 0.268 but ADMM committed only 0.115 on
    # simbench_lv with 6 failures and slack 0.15).  Covers both
    # supply ≥ demand (T = demand → frac = 1) and supply < demand
    # (T = waterfall → frac matches the priority-correct schedule).
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

    # Index the cp_coupling entries by actor so we can extend that
    # actor's (C, d) below.  Each entry is consumed once.
    cp_couplings_by_actor: dict[int, list[tuple[str, str, float]]] = {}
    for entry in (cp_coupling or []):
        try:
            g_idx, sec_in, sec_out, ratio = entry
        except (TypeError, ValueError):
            continue
        if sec_in not in sec_idx or sec_out not in sec_idx:
            # Skip couplings that reference sectors not in the current
            # multi-sector ADMM scope (e.g. a P2G when gas is absent).
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
                # Waterfall-shed cells: when the priority waterfall set
                # ``T[j] = 0`` (cell is below the supply cutoff in the
                # priority order), force the per-actor ub to *exact* 0
                # so the local QP cannot leak supply into a cell the
                # waterfall decided to drop.  Without this, the
                # sharing-ADMM's weak per-actor S preference + the local
                # QP's quadratic round-off let small actors deposit
                # tiny positive amounts on shed cells; ``x_avg[j] > 0``
                # then anchors ``z[j] > 0`` via the z-update's
                # quadratic term, and the convergence check passes on
                # a leaked allocation.  Using exact 0 (bypassing the
                # ``max(raw_cap, 1e-6)`` well-conditioning floor) is
                # safe because the constraint ``0 ≤ x ≤ 0`` is feasible
                # for OSQP and pins the variable exactly; the floor
                # only matters when ``raw_cap > 0`` and the solver
                # needs slack to handle near-degenerate cells.
                if total_T[j] <= 1e-12:
                    ub[j] = 0.0
                else:
                    ub[j] = max(raw_cap, 1e-6)

        # Per-actor coupling: Σ commitments ≤ total supply.  Binding
        # constraint that creates scarcity when supply < demand.
        total_supply = sum(float(v) for v in supply_map.values())
        actor_supply_total.append(total_supply)
        C_rows = [np.ones(n_dims)]
        d_rows = [max(total_supply, 0.0)]

        # CP coupling: for each (sector_in, sector_out, ratio) attached
        # to this actor, add the equality ``Σ_t x[out, t] = ratio · Σ_t
        # x[in, t]`` as a pair of inequalities.  Together they pin the
        # CP actor's per-sector flow ratio so its output supply commit
        # stays consistent with its input draw under the conversion
        # efficiency.  No-op for non-CP actors (the list is empty).
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
