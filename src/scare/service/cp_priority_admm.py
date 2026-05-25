"""Priority-cascaded sharing ADMM for cross-sector coupling-point coordination.

Pure-compute kernel for the L3 design (see design.tex §sec:design:cp-admm):
each coupling point holds a single regulation knob ``r_i ∈ [0,1]`` over an
``H``-step rolling horizon; the per-sector commitment is

    x_i[s, k] = r_i[k] · c_i[s]

with ``c_i[s]`` the CP's effective signed capacity in sector ``s`` (load
convention: positive = the CP consumes from ``s``, negative = the CP
produces into ``s``).  The coupling ratio is baked into ``c_i`` at spec
time, so the single-knob constraint and the physical input/output ratio
are both honoured by the variable definition itself.

Mechanism
---------

A standard scaled sharing ADMM is augmented with a priority-marginal
linear penalty that is recomputed each iteration from the current
aggregate via a per-sector waterfall.  The marginal ``λ_s[k]`` is the
priority weight of the highest-priority tier in sector ``s`` that is
currently under-served by the aggregate CP-and-base supply, and zero
otherwise.  Each CP's local subproblem then minimises

    (ρ/2) ‖x_i − (z − u_i)‖²  +  Σ_s λ_s · x_i[s]

over ``r_i ∈ [0,1]``.  Because the linear term is signed by the
direction of ``c_i[s]`` (positive for consumption, negative for
production), it penalises consumption from a scarce sector and rewards
production into one — the price-elastic local response that drives
near-strict priority at convergence.  The full waterfall projection
described in design.tex §priority-cascade is the dual mechanism: as
iterations proceed, every cell that is served drops out of the
marginal, and only the next-highest unserved tier carries weight, so
the system waterfalls down the priority schedule.

Decision invariants
-------------------

- Single-knob: ``x_i[:, k] = r_i[k] · c_i`` is enforced as a hard
  equality by the variable substitution above; the local update is a
  scalar QP in ``r_i[k]`` rather than an ``m``-vector QP in ``x_i[:, k]``.
- Coupling: the per-sector signs and magnitudes of ``c_i`` already
  encode ``x_i[out] = −η_i · x_i[in]``; no separate constraint matrix is
  needed.  CHP and other multi-output couplings are expressed by
  populating ``capacity_by_sector`` on all coupled sectors with the
  right signs and per-sector effective capacities.
- Horizon: ``H = 1`` is the MVP and is exercised by the test suite.
  The data layout reserves the ``H`` axis on every array so the storage
  / receding-horizon extension can drop in inter-step coupling
  constraints in the local update without changing the consensus or
  dual update shapes.

Returns
-------

The kernel emits, per CP, the converged regulation factor over the
horizon plus the resulting per-(sector, tier, step) served amounts (the
waterfall output on the converged aggregate).  Callers — typically a
``CPRole`` running this kernel locally on its replicated peer view —
read ``factor_by_cp[self.cp_id]`` and commit ``r[0]`` as the next
regulation factor on the underlying CP device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPSpec:
    """Per-CP specification consumed by :func:`solve_cp_priority_admm`.

    ``capacity_by_sector`` carries the CP's effective signed capacity in
    each sector it touches, in MW under the load convention: positive =
    the CP consumes from that sector, negative = the CP produces into
    it.  The coupling ratio ``η`` is baked in at spec time, so a P2H
    with 10 MW input and ``η = 0.95`` is described as
    ``{"electricity": 10.0, "heat": -9.5}``.  A CHP with 10 MW gas
    input, 3.5 MW electricity output, 4.5 MW heat output is
    ``{"gas": 10.0, "electricity": -3.5, "heat": -4.5}``.
    """

    cp_id: str
    capacity_by_sector: dict[str, float]
    # Per-step ramp limit on the regulation knob.  H = 1 ignores this;
    # the H > 1 extension consumes it as a chained box constraint
    # |r[k] − r[k−1]| ≤ max_ramp_per_step in the local update.
    max_ramp_per_step: float = 1.0


@dataclass(frozen=True)
class SectorDemand:
    """Per-sector demand profile over the horizon."""

    sector: str
    # tier -> length-H array of MW.
    demand_by_tier: dict[int, np.ndarray]
    # length-H array of MW (positive = base generator supply available
    # to this sector before any CP contribution).
    base_supply: np.ndarray


@dataclass(frozen=True)
class CPAdmmResult:
    # cp_id -> length-H regulation factor in [0, 1].
    factor_by_cp: dict[str, np.ndarray]
    # sector -> tier -> length-H served amount (MW).
    served_by_sector_tier: dict[str, dict[int, np.ndarray]]
    iterations: int
    primal_residual: float
    dual_residual: float
    converged: bool
    # Diagnostics: per-iteration residuals, final marginal priorities.
    history: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tier_priority_weight(
    tier: int, *, priority_tiers: int = 4, base: float = 1.0e4
) -> float:
    """Strictly-monotone weight: tier 1 → ``base^P``, tier P → ``base``.

    Mirrors the four-orders-of-magnitude separation of the design's
    schedule (``base = 1e4`` gives ``[1e16, 1e12, 1e8, 1e4]`` for
    ``P = 4``), but exposed as a parameter so the test suite can use a
    tame base (e.g. 10) for human-readable assertions while production
    callers keep the stiff default.
    """
    if tier < 1:
        return 0.0
    return base ** max(0, priority_tiers - tier + 1)


def waterfall_serve(
    supply: np.ndarray,
    demand: np.ndarray,
) -> np.ndarray:
    """Per-(sector, tier, step) priority-waterfall served amount.

    ``supply`` has shape ``(n_sec, H)``; ``demand`` has shape
    ``(n_sec, n_tier, H)`` with tiers sorted ascending (tier index 0 =
    highest priority).  Iterates cells in ascending tier order per
    sector per step and assigns ``min(demand_cell, remaining_pool)``
    until the pool is exhausted.  Returns the served array with the
    same shape as ``demand``.
    """
    n_sec, n_tier, H = demand.shape
    served = np.zeros_like(demand)
    for s in range(n_sec):
        for k in range(H):
            remaining = max(float(supply[s, k]), 0.0)
            if remaining <= 0.0:
                continue
            for t in range(n_tier):
                dem = float(demand[s, t, k])
                if dem <= 0.0:
                    continue
                take = min(dem, remaining)
                served[s, t, k] = take
                remaining -= take
                if remaining <= 1e-12:
                    break
    return served


def marginal_priority(
    served: np.ndarray,
    demand: np.ndarray,
    priorities: np.ndarray,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    """Per-sector marginal priority value ``λ[s, k]``.

    Equal to the priority weight of the highest-priority tier in sector
    ``s`` at step ``k`` that is not fully served (``served < demand``).
    Zero when every tier with positive demand is fully served — meaning
    no scarcity, the CPs face no priority pressure on that sector.

    ``priorities`` has length ``n_tier`` and is sorted in the same
    ascending tier order as the ``demand`` array's tier axis.
    """
    n_sec, n_tier, H = demand.shape
    lam = np.zeros((n_sec, H))
    for s in range(n_sec):
        for k in range(H):
            for t in range(n_tier):
                dem = float(demand[s, t, k])
                if dem > tol and served[s, t, k] < dem - tol:
                    lam[s, k] = float(priorities[t])
                    break
    return lam


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve_cp_priority_admm(
    cps: list[CPSpec],
    demands: list[SectorDemand],
    *,
    horizon: int = 1,
    rho: float = 1.0,
    max_iters: int = 500,
    abs_tol: float = 1e-3,
    priority_tiers: int = 4,
    priority_weight_base: float = 1.0e4,
    r_damping: float = 0.3,
    record_history: bool = False,
) -> CPAdmmResult:
    """Solve the priority-cascaded sharing ADMM for a CP group.

    The list ``cps`` must contain every CP in the cross-sector
    component over which coordination is being run; the list
    ``demands`` carries the per-sector profile aggregated from the
    relevant leaders' published flex summaries (one entry per sector
    that any CP in ``cps`` touches; sectors absent from ``demands``
    are treated as having zero demand and zero base supply).

    The kernel is fully synchronous and deterministic: given identical
    inputs every caller produces the same output, which is the
    property the replicated-coordinator pattern in the L3 role
    relies on.

    Returns ``CPAdmmResult``.  When ``record_history=True``, the
    ``history`` field carries per-iteration primal/dual residuals and
    the final marginal-priority vector — useful for unit tests and
    for the eval-harness convergence reports.
    """
    H = int(horizon)
    if H < 1:
        raise ValueError("horizon must be >= 1")
    N = len(cps)
    if N == 0:
        return CPAdmmResult(
            factor_by_cp={},
            served_by_sector_tier={
                d.sector: {t: np.asarray(a, dtype=float).copy()
                           for t, a in d.demand_by_tier.items()}
                for d in demands
            },
            iterations=0,
            primal_residual=0.0,
            dual_residual=0.0,
            converged=True,
        )

    # ----- index spaces ------------------------------------------------------
    all_sectors = sorted(
        {s for c in cps for s in c.capacity_by_sector}
        | {d.sector for d in demands}
    )
    if not all_sectors:
        raise ValueError("no sectors found in CPs or demands")
    n_sec = len(all_sectors)
    sec_idx = {s: i for i, s in enumerate(all_sectors)}

    all_tiers = sorted({t for d in demands for t in d.demand_by_tier})
    if not all_tiers:
        # No demand anywhere -> no scarcity, no priority pressure.
        # Each CP defaults to r = 0 (minimum-effort baseline).
        return CPAdmmResult(
            factor_by_cp={c.cp_id: np.zeros(H) for c in cps},
            served_by_sector_tier={d.sector: {} for d in demands},
            iterations=0,
            primal_residual=0.0,
            dual_residual=0.0,
            converged=True,
        )
    n_tier = len(all_tiers)
    tier_idx = {t: i for i, t in enumerate(all_tiers)}

    # ----- pack inputs into arrays ------------------------------------------
    # Per-CP signed capacity vector.
    cap = np.zeros((N, n_sec), dtype=float)
    for i, c in enumerate(cps):
        for s, c_val in c.capacity_by_sector.items():
            cap[i, sec_idx[s]] = float(c_val)
    cap_norm_sq = (cap ** 2).sum(axis=1)

    # Per-sector demand (n_sec, n_tier, H) and base supply (n_sec, H).
    D = np.zeros((n_sec, n_tier, H), dtype=float)
    base_supply = np.zeros((n_sec, H), dtype=float)
    for d in demands:
        s = sec_idx[d.sector]
        bs = np.asarray(d.base_supply, dtype=float)
        if bs.shape != (H,):
            raise ValueError(
                f"base_supply for sector {d.sector!r} must have shape ({H},), "
                f"got {bs.shape}"
            )
        base_supply[s, :] = bs
        for tier, arr in d.demand_by_tier.items():
            if tier not in tier_idx:
                continue
            a = np.asarray(arr, dtype=float)
            if a.shape != (H,):
                raise ValueError(
                    f"demand_by_tier[{tier}] for sector {d.sector!r} must have "
                    f"shape ({H},), got {a.shape}"
                )
            D[s, tier_idx[tier], :] = a

    priorities = np.array(
        [tier_priority_weight(t, priority_tiers=priority_tiers,
                              base=priority_weight_base)
         for t in all_tiers],
        dtype=float,
    )

    # ----- ADMM state -------------------------------------------------------
    # x[i, s, k] = r_i[k] · cap[i, s] is enforced after each x-update.
    x = np.zeros((N, n_sec, H), dtype=float)
    z = np.zeros((n_sec, H), dtype=float)
    u = np.zeros((N, n_sec, H), dtype=float)
    r_curr = np.zeros((N, H), dtype=float)

    # ---- Pre-compute initial λ from the all-zero baseline -----------------
    # Without this, the first x-update sees λ = 0 (no priority signal),
    # commits r = 0, the convergence test sees primal = dual = 0 and
    # declares done — a spurious "converged at no allocation" fixed
    # point.  Computing λ once from the r = 0 supply state gives the
    # first x-update the correct gradient.
    served_init = waterfall_serve(base_supply, D)
    lam = marginal_priority(served_init, D, priorities)

    history_primal: list[float] = []
    history_dual: list[float] = []
    history_r_change: list[float] = []

    primal_res = float("inf")
    dual_res = float("inf")
    converged = False
    iteration = 0

    for iteration in range(max_iters):
        z_prev = z.copy()
        max_r_change = 0.0

        # ---- Local x-update ------------------------------------------------
        # Per (CP, step), solve a scalar QP in r ∈ [0, 1]:
        #     min  (ρ/2) ‖r · cap_i − target_k‖² + (λ_k · cap_i) · r
        # where target_k = z[:, k] − u_i[:, k].  Closed form:
        #     r* = clamp((ρ · cap_i · target_k − λ_k · cap_i) / (ρ · ‖cap_i‖²),
        #                0, 1)
        # then take a damped trust-region step toward r* to stabilise
        # the iteration against the stiff λ feedback (see docstring).
        for i in range(N):
            if cap_norm_sq[i] < 1e-12:
                continue
            for k in range(H):
                target = z[:, k] - u[i, :, k]
                num = float(rho * cap[i] @ target - cap[i] @ lam[:, k])
                den = float(rho * cap_norm_sq[i])
                r_star = num / den
                if r_star < 0.0:
                    r_star = 0.0
                elif r_star > 1.0:
                    r_star = 1.0
                r_new = (1.0 - r_damping) * r_curr[i, k] + r_damping * r_star
                if abs(r_new - r_curr[i, k]) > max_r_change:
                    max_r_change = abs(r_new - r_curr[i, k])
                r_curr[i, k] = r_new
                x[i, :, k] = r_new * cap[i]

        # ---- Global z-update -----------------------------------------------
        # Sharing-ADMM scaled-form mean update.
        z = (x + u).mean(axis=0)

        # Compute aggregate net supply pool per sector per step and the
        # priority waterfall on it.  Load convention: CP's positive x
        # consumes supply (subtract); CP's negative x produces supply
        # (adds — by subtracting a negative).
        x_agg = x.sum(axis=0)
        supply_net = base_supply - x_agg
        served = waterfall_serve(supply_net, D)
        lam = marginal_priority(served, D, priorities)

        # ---- Dual update ---------------------------------------------------
        u = u + x - z[np.newaxis, :, :]

        # ---- Convergence ---------------------------------------------------
        # Classical residuals are recorded for diagnostics, but the
        # gating criterion is the per-step damped factor move: once no
        # CP wants to move in any step, the kernel is at a fixed point
        # of the priority-marginal feedback regardless of how the
        # symmetric-CP primal residual degenerates.
        primal_res = float(np.linalg.norm(x - z[np.newaxis, :, :]))
        dual_res = float(rho * np.linalg.norm(z - z_prev))
        if record_history:
            history_primal.append(primal_res)
            history_dual.append(dual_res)
            history_r_change.append(max_r_change)
        if max_r_change < abs_tol:
            converged = True
            break

    factor_by_cp: dict[str, np.ndarray] = {
        c.cp_id: r_curr[i].copy() for i, c in enumerate(cps)
    }

    served_by_sector_tier = {
        d.sector: {
            t: served[sec_idx[d.sector], tier_idx[t], :].copy()
            for t in all_tiers
            if t in tier_idx
        }
        for d in demands
    }

    history: dict[str, Any] = {}
    if record_history:
        history = {
            "primal_residuals": history_primal,
            "dual_residuals": history_dual,
            "r_changes": history_r_change,
            "marginal_priority": lam.copy(),
        }

    return CPAdmmResult(
        factor_by_cp=factor_by_cp,
        served_by_sector_tier=served_by_sector_tier,
        iterations=iteration + 1,
        primal_residual=primal_res,
        dual_residual=dual_res,
        converged=converged,
        history=history,
    )
