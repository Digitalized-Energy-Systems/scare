"""Priority-cascaded sharing ADMM for cross-sector coupling-point coordination.

Pure-compute kernel for the L3 design. Each CP holds a single regulation knob
``r_i ∈ [0,1]`` over an ``H``-step horizon; the per-sector commitment is

    x_i[s, k] = r_i[k] · c_i[s]

with ``c_i[s]`` the CP's effective signed capacity in sector ``s`` (load
convention: positive = consumes from ``s``, negative = produces into ``s``). The
coupling ratio is baked into ``c_i`` at spec time, so the single-knob constraint
and the physical input/output ratio are both honoured by the variable itself.

Mechanism: a scaled sharing ADMM augmented with a priority-marginal linear
penalty recomputed each iteration via a per-sector waterfall. The marginal
``λ_s[k]`` is the priority weight of the highest unserved tier in sector ``s``
(zero otherwise); each CP's local subproblem minimises

    (ρ/2) ‖x_i − (z − u_i)‖²  +  Σ_s λ_s · x_i[s]

over ``r_i ∈ [0,1]``. The linear term is signed by ``c_i[s]``, so it penalises
consumption from a scarce sector and rewards production into one — the response
that drives near-strict priority at convergence. As iterations proceed, served
cells drop out of the marginal and only the next-highest unserved tier carries
weight, waterfalling down the priority schedule.

Invariants:
- Single-knob: ``x_i[:, k] = r_i[k] · c_i`` is a hard equality, so the local
  update is a scalar QP in ``r_i[k]`` not an m-vector QP in ``x_i[:, k]``.
- Coupling: signs/magnitudes of ``c_i`` encode ``x_i[out] = −η_i · x_i[in]``; no
  separate constraint matrix. CHP / multi-output couplings are expressed by
  populating ``capacity_by_sector`` on all coupled sectors with the right signs.
- Horizon: ``H = 1`` is the MVP. The ``H`` axis is reserved on every array so a
  receding-horizon extension can add inter-step constraints in the local update
  without changing the consensus/dual update shapes.

Returns, per CP, the converged regulation factor over the horizon plus the
per-(sector, tier, step) served amounts. Callers (a ``CPRole`` running this
locally on its replicated peer view) read ``factor_by_cp[self.cp_id]`` and commit
``r[0]`` as the next regulation factor.
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
    """Per-CP specification consumed by the L3 CP sharing-ADMM kernel.

    ``capacity_by_sector`` is the CP's effective signed capacity per sector, in
    MW under the load convention (positive = consumes, negative = produces). The
    coupling ratio ``η`` is baked in at spec time: a P2H with 10 MW input and
    η = 0.95 is ``{"electricity": 10.0, "heat": -9.5}``; a CHP with 10 MW gas
    input is ``{"gas": 10.0, "electricity": -3.5, "heat": -4.5}``.
    """

    cp_id: str
    capacity_by_sector: dict[str, float]
    # Per-step ramp limit on the knob. H = 1 ignores it; the H > 1 extension uses
    # it as a chained box constraint |r[k] − r[k−1]| ≤ max_ramp_per_step.
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
