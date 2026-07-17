"""Priority-cascaded sharing ADMM for cross-sector coupling-point coordination.

Pure-compute L3 kernel. Each CP holds one regulation knob ``r_i ∈ [0,1]`` over
an ``H``-step horizon; the per-sector commitment is ``x_i[s,k] = r_i[k]·c_i[s]``,
with ``c_i[s]`` the signed capacity (load convention: + consumes, − produces).
The coupling ratio is baked into ``c_i`` at spec time.

Mechanism: a scaled sharing ADMM with a priority-marginal linear penalty
recomputed each iteration via a per-sector waterfall. The marginal ``λ_s[k]`` is
the priority weight of the highest unserved tier in sector ``s``; each CP's local
subproblem minimises ``(ρ/2)‖x_i − (z − u_i)‖² + Σ_s λ_s·x_i[s]``. The signed
linear term penalises consuming from a scarce sector and rewards producing into
one. As served cells drop out, the marginal waterfalls down the priority schedule.

Invariants:
- Single-knob: the local update is a scalar QP in ``r_i[k]``, not an m-vector QP.
- Coupling: signs/magnitudes of ``c_i`` encode ``x_i[out] = −η_i·x_i[in]``; CHP /
  multi-output couplings populate ``capacity_by_sector`` on all coupled sectors.
- Horizon: ``H = 1`` is the MVP; the ``H`` axis is reserved for a receding-horizon
  extension without changing consensus/dual update shapes.

Returns per CP the converged regulation factor plus per-(sector,tier,step) served
amounts. Callers read ``factor_by_cp[self.cp_id]`` and commit ``r[0]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPSpec:
    """Per-CP spec for the L3 sharing-ADMM kernel.

    ``capacity_by_sector`` is signed MW per sector (load convention: + consumes,
    − produces) with coupling ratio ``η`` baked in: P2H 10 MW @ η=0.95 is
    ``{"electricity": 10.0, "heat": -9.5}``; a CHP is
    ``{"gas": 10.0, "electricity": -3.5, "heat": -4.5}``.
    """

    cp_id: str
    capacity_by_sector: dict[str, float]
    # Per-step ramp limit; H = 1 ignores it. H > 1 uses |r[k] − r[k−1]| ≤ limit.
    max_ramp_per_step: float = 1.0


@dataclass(frozen=True)
class SectorDemand:
    """Per-sector demand profile over the horizon."""

    sector: str
    # tier -> length-H array of MW.
    demand_by_tier: dict[int, np.ndarray]
    # length-H MW: base generator supply before any CP contribution.
    base_supply: np.ndarray
