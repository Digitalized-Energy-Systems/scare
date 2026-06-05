"""Local Q(U) droop control for inverter-coupled PV generators.

Implements the VDE-AR-N 4105 (German LV grid code) Q(U) curve at every
PowerGenerator. Each inverter measures its local bus voltage every
electricity poll period and dispatches reactive power within its
apparent-power capability circle to support voltage.

Local, autonomous, communication-free, and faster than the gossip /
holonic / CP layers. Acts only on Q, so it does not interfere with the
active-power restoration the MAS handles.
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


# Cache-gate tolerance for ``_step``: when every observed input
# (vm_pu, |p_mw|, regulation) is within this of the previous tick, the
# Q(U) output is unchanged and the step is skipped. Tighter than typical
# voltage fluctuation so the droop still tracks meaningful movement.
_OBS_DELTA_TOL: float = 1e-4


# ---------------------------------------------------------------------------
# VDE-AR-N 4105 Q(U) curve
# ---------------------------------------------------------------------------
#
# Piecewise-linear droop (saturating at ±Q_max outside the bands):
#   U/U_n ≤ 0.95 → +Q_max | 0.97..1.03 → 0 (deadband) | ≥ 1.05 → −Q_max
#
# Sign convention is monee load convention: q_mvar > 0 = absorption
# (over-voltage response), q_mvar < 0 = injection (under-voltage).
#
# Q is bounded by the capability circle |q| ≤ √(S_n² − p²), further
# bounded by cos φ_min; Q_max is the smaller of the two.
VDE_V_LOW: float = 0.95            # full Q+ injection at and below
VDE_V_DEADBAND_LOW: float = 0.97   # Q = 0 from here…
VDE_V_DEADBAND_HIGH: float = 1.03  # …to here
VDE_V_HIGH: float = 1.05           # full Q- absorption at and above

# VDE-AR-N 4105 §5.7.2 displacement-factor envelope by rating.
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
        # Ramp -q_max (V_LOW) → 0 (V_DEADBAND_LOW).
        frac = (VDE_V_DEADBAND_LOW - v_pu) / (VDE_V_DEADBAND_LOW - VDE_V_LOW)
        return -q_max * frac
    if v_pu >= VDE_V_DEADBAND_HIGH:
        # Ramp 0 (V_DEADBAND_HIGH) → +q_max (V_HIGH).
        frac = (v_pu - VDE_V_DEADBAND_HIGH) / (VDE_V_HIGH - VDE_V_DEADBAND_HIGH)
        return +q_max * frac
    return 0.0


class ReactivePowerDroopRole(Role):
    """Per-inverter Q(U) droop following VDE-AR-N 4105.

    Installed alongside the balance / generation-controller /
    constraint-monitor roles on every PV-coupled :class:`PowerGenerator`
    child. Each poll period reads the parent node's ``vm_pu``, computes
    the Q(U) setpoint, clips to the capability circle, and dispatches via
    ``behavior.act(aid, "set_q", q)``.

    Self-contained: no message subscriptions, no events others depend on.
    Each distinct command is recorded as a ``qv_droop`` diagnostics event.
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
        # Last commanded Q (monee convention); gates duplicate records.
        self._last_q: float | None = None
        # Last observed droop inputs (vm_pu, |p|, regulation). Q is a
        # function of these; if none moved beyond ``_OBS_DELTA_TOL``,
        # ``_step`` returns early.
        self._last_obs_v: float | None = None
        self._last_obs_p: float | None = None
        self._last_obs_reg: float | None = None

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
        # Generator p_mw is negative in monee's load convention; the
        # magnitude is what enters the capability circle.
        p_raw = obs.get("p_mw", 0.0)
        try:
            p_mag = abs(float(p_raw))
        except (TypeError, ValueError):
            p_mag = 0.0
        if not math.isfinite(p_mag):
            p_mag = 0.0
        # Scale by ``regulation`` for the actually dispatched real power.
        regulation = float(obs.get("regulation", 1.0))
        if not math.isfinite(regulation):
            regulation = 1.0
        regulation = max(0.0, min(2.0, regulation))
        # Cache gate: skip the pass when every droop input is within
        # tolerance of the previous tick (first call passes, caches None).
        if (
            self._last_obs_v is not None
            and self._last_obs_p is not None
            and self._last_obs_reg is not None
            and abs(v_f - self._last_obs_v) < _OBS_DELTA_TOL
            and abs(p_mag - self._last_obs_p) < _OBS_DELTA_TOL
            and abs(regulation - self._last_obs_reg) < _OBS_DELTA_TOL
        ):
            return
        self._last_obs_v = v_f
        self._last_obs_p = p_mag
        self._last_obs_reg = regulation
        p_dispatched = p_mag * regulation
        s_sq = self.s_nom_mva * self.s_nom_mva
        # p_dispatched > s_nom means real power exceeds the apparent-power
        # rating, collapsing the capability circle to 0 reactive. Should
        # not happen on a well-sized inverter; warn rather than silently
        # disable Q so the rating mismatch is visible.
        if p_dispatched > self.s_nom_mva + 1e-9:
            logger.warning(
                "[%s] qv_droop: p_dispatched=%.4g > s_nom=%.4g (over-rated "
                "real power); Q capability collapses to 0 this tick.",
                self.context.aid, p_dispatched, self.s_nom_mva,
            )
        circle_q = max(0.0, math.sqrt(max(0.0, s_sq - p_dispatched * p_dispatched)))
        # Displacement-factor envelope (VDE-AR-N 4105 §5.7.2):
        # |tan φ_min| = √(1 − cos²) / cos.
        cos = max(1e-6, min(1.0, self.cos_phi_min))
        cos_phi_q = max(0.0, p_dispatched) * math.sqrt(1.0 - cos * cos) / cos
        q_max = min(circle_q, cos_phi_q) if cos_phi_q > 0.0 else circle_q
        if q_max <= 0.0:
            # No reactive headroom (full output, or off: p_dispatched == 0
            # makes cos_phi_q == 0).
            q_cmd = 0.0
        else:
            q_cmd = vde_q_curve(v_f / self.voltage_ref_pu, q_max)
        # Idempotency: skip the act() write when the command is within
        # tolerance of the previous one.
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
