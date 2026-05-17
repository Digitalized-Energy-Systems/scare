"""Local Q-V droop control for inverter-coupled PV generators.

Implements the VDE-AR-N 4105 (German LV grid code) characteristic
Q(U) curve at every PowerGenerator.  Each inverter measures its local
bus voltage every electricity poll period and dispatches reactive
power within its apparent-power capability circle to support voltage.

Composes orthogonally with the rest of \\textsc{Scare}:
- Local, autonomous, no communication.
- Faster timescale than the gossip / holonic / CP layers.
- Acts on a separate decision variable (Q), so it does not interfere
  with the active-power restoration the MAS handles.

The role is intentionally communication-free; it does not subscribe
to messages and does not emit events that other agents listen for.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from mango import Role

from scare.base.diagnostics import record_event
from scare.base.model import SECTOR_TIMESCALE, Sector

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VDE-AR-N 4105 (Erzeugungsanlagen am Niederspannungsnetz) — Q(U) curve
# ---------------------------------------------------------------------------
#
# Standard piecewise-linear droop:
#   U / U_n   →   Q
#   ≤ 0.95    →   +Q_max      (overexcited, full injection — under-voltage)
#   = 0.97    →    0
#   = 1.03    →    0           (deadband from 0.97 to 1.03)
#   = 1.05    →   −Q_max      (underexcited, full absorption — over-voltage)
#   ≥ 1.05    →   −Q_max
#
# Outside the linear bands the curve saturates at ±Q_max.
#
# Sign convention follows monee's load convention: ``q_mvar > 0``
# represents *consumption* at the node (i.e. the inverter absorbs
# reactive — used during over-voltage).  ``q_mvar < 0`` represents
# *injection* (overexcited — used during under-voltage).
#
# Apparent-power capability circle:
#   |q_mvar| ≤ √(S_n² − p²)
# bounded further by the cos φ_min the standard prescribes (0.95 for
# ``S_n ≤ 13.8 kVA``, 0.90 for larger inverters).  The capability
# limit ``Q_max`` used in the curve above is the smaller of the two.
VDE_V_LOW: float = 0.95            # full Q+ injection at and below
VDE_V_DEADBAND_LOW: float = 0.97   # Q = 0 from here…
VDE_V_DEADBAND_HIGH: float = 1.03  # …to here
VDE_V_HIGH: float = 1.05           # full Q- absorption at and above

# Per VDE-AR-N 4105 §5.7.2 the displacement-factor envelope is:
#   S_n ≤ 13.8 kVA   →   cos φ ∈ [0.95(under), 0.95(over)]
#   S_n  > 13.8 kVA  →   cos φ ∈ [0.90(under), 0.90(over)]
COS_PHI_THRESHOLD_MVA: float = 0.0138        # 13.8 kVA in MVA
COS_PHI_SMALL: float = 0.95                  # for S_n ≤ 13.8 kVA
COS_PHI_LARGE: float = 0.90                  # for S_n  > 13.8 kVA


def vde_cos_phi_min(s_nom_mva: float) -> float:
    """Return the VDE-AR-N 4105 minimum displacement factor for an
    inverter of rated apparent power ``s_nom_mva``.
    """
    return COS_PHI_SMALL if s_nom_mva <= COS_PHI_THRESHOLD_MVA else COS_PHI_LARGE


def vde_q_curve(v_pu: float, q_max: float) -> float:
    """Evaluate the VDE-AR-N 4105 Q(U) characteristic.

    Parameters
    ----------
    v_pu:
        Per-unit bus voltage magnitude.
    q_max:
        Magnitude of the capability-circle limit (MVar).  The returned
        Q is bounded to ``[-q_max, +q_max]``.

    Returns
    -------
    Q setpoint in monee's load convention:
    positive = absorbs reactive (over-voltage response),
    negative = injects reactive (under-voltage response).
    """
    if v_pu <= VDE_V_LOW:
        return -q_max
    if v_pu >= VDE_V_HIGH:
        return +q_max
    if v_pu <= VDE_V_DEADBAND_LOW:
        # Linear ramp from -q_max at V_LOW to 0 at V_DEADBAND_LOW.
        frac = (VDE_V_DEADBAND_LOW - v_pu) / (VDE_V_DEADBAND_LOW - VDE_V_LOW)
        return -q_max * frac
    if v_pu >= VDE_V_DEADBAND_HIGH:
        # Linear ramp from 0 at V_DEADBAND_HIGH to +q_max at V_HIGH.
        frac = (v_pu - VDE_V_DEADBAND_HIGH) / (VDE_V_HIGH - VDE_V_DEADBAND_HIGH)
        return +q_max * frac
    return 0.0


class ReactivePowerDroopRole(Role):
    """Per-inverter Q(U) droop following VDE-AR-N 4105.

    Installed alongside the energy balance / generation-controller /
    constraint-monitor roles on every PV-coupled :class:`PowerGenerator`
    child agent.  Reads the parent node's ``vm_pu`` each poll period,
    computes the standard Q(U) setpoint, clips to the capability
    circle, and dispatches via ``behavior.act(aid, "set_q", q)``.

    The role is self-contained: no message subscriptions, no events
    emitted that other roles depend on.  All it does is mutate
    ``child.model.q_mvar`` at the rated electrical timescale.  Every
    commanded change is recorded as a ``qv_droop`` event in the
    diagnostics ledger so the post-run analysis can plot Q activity
    against ``vm_pu`` trajectories.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        *,
        s_nom_mva: float,
        cos_phi_min: float | None = None,
        voltage_ref_pu: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        if s_nom_mva <= 0.0 or not math.isfinite(s_nom_mva):
            raise ValueError(
                f"s_nom_mva must be positive and finite; got {s_nom_mva!r}"
            )
        self.s_nom_mva = float(s_nom_mva)
        self.cos_phi_min = (
            float(cos_phi_min) if cos_phi_min is not None
            else vde_cos_phi_min(self.s_nom_mva)
        )
        self.voltage_ref_pu = float(voltage_ref_pu)
        # Last commanded Q (monee convention).  Used by ``_step`` to
        # avoid recording trivial duplicates when V hasn't moved.
        self._last_q: float | None = None

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(Sector.ELECTRICITY, {}).get(
            "poll_period_s", 0.5
        )
        self.context.schedule_periodic_task(self._step, delay=poll)

    async def _step(self) -> None:
        if not self.behavior.has_action(self.context.aid, "set_q"):
            return
        try:
            obs = self.behavior.observe(self.context.aid) or {}
        except AttributeError:
            return
        v = obs.get("vm_pu")
        if v is None:
            return
        try:
            v_f = float(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v_f) or v_f <= 0.0:
            return
        # Active power dispatch (load convention: generator's p_mw is
        # negative in monee).  Magnitude is what enters the capability
        # circle.
        p_raw = obs.get("p_mw", 0.0)
        try:
            p_mag = abs(float(p_raw))
        except (TypeError, ValueError):
            p_mag = 0.0
        if not math.isfinite(p_mag):
            p_mag = 0.0
        # Apply the current ``regulation`` to get the *actually dispatched*
        # real power, then build the capability circle around it.
        regulation = float(obs.get("regulation", 1.0))
        if not math.isfinite(regulation):
            regulation = 1.0
        regulation = max(0.0, min(2.0, regulation))
        p_dispatched = p_mag * regulation
        s_sq = self.s_nom_mva * self.s_nom_mva
        # ``p_dispatched > s_nom_mva`` means the active power exceeds the
        # inverter's apparent-power rating — the capability circle
        # collapses to 0 reactive power.  Should never happen on a
        # well-sized inverter; if it does, surface it so the operator
        # can investigate the rating mismatch instead of silently
        # disabling Q support on the affected inverter.
        if p_dispatched > self.s_nom_mva + 1e-9:
            logger.warning(
                "[%s] qv_droop: p_dispatched=%.4g > s_nom=%.4g (over-rated "
                "real power); Q capability collapses to 0 this tick.",
                self.context.aid, p_dispatched, self.s_nom_mva,
            )
        circle_q = max(0.0, math.sqrt(max(0.0, s_sq - p_dispatched * p_dispatched)))
        # Apply the displacement-factor envelope (VDE-AR-N 4105 §5.7.2).
        # |tan(φ_min)| = √(1 − cos²) / cos.
        cos = max(1e-6, min(1.0, self.cos_phi_min))
        cos_phi_q = max(0.0, p_dispatched) * math.sqrt(1.0 - cos * cos) / cos
        q_max = min(circle_q, cos_phi_q) if cos_phi_q > 0.0 else circle_q
        if q_max <= 0.0:
            # Inverter is at full real-power output with no headroom
            # (or off — p_dispatched == 0 yields cos_phi_q == 0 and
            # circle_q == s_nom; the cos-φ rule kicks in).
            q_cmd = 0.0
        else:
            q_cmd = vde_q_curve(v_f / self.voltage_ref_pu, q_max)
        # Idempotency: skip the act() write if the new command equals
        # the previously dispatched one to within a small tolerance.
        # Records every distinct command (per user spec).
        tol = max(1e-6, 1e-4 * q_max)
        if self._last_q is not None and abs(q_cmd - self._last_q) < tol:
            return
        self.behavior.act(self.context.aid, "set_q", q_cmd)
        self._last_q = q_cmd
        record_event(
            t=self.context.current_timestamp,
            kind="qv_droop",
            aid=self.context.aid,
            sector=Sector.ELECTRICITY.value,
            detail=f"v={v_f:.4f} q={q_cmd:.6f} q_max={q_max:.6f} p={p_dispatched:.4f}",
        )
