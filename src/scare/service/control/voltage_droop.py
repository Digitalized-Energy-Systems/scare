"""Local Q(U) droop control (VDE-AR-N 4105) for inverter-coupled PV.

Per-inverter, autonomous, communication-free. Acts only on Q, so it does
not interfere with the MAS active-power restoration.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from mango import Role

from scare.base.model import SECTOR_TIMESCALE, Sector
from scare.base.runtime.diagnostics import record_event
from scare.base.util import _QV_RELIEF_TTL_S, publish_qv_relief

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# Cache-gate tolerance for ``_step``: skip when every input (vm_pu, |p_mw|,
# regulation) is within this of the previous tick (Q(U) output unchanged).
_OBS_DELTA_TOL: float = 1e-4


# VDE-AR-N 4105 Q(U) curve. Piecewise-linear droop saturating at ±Q_max.
# Monee load convention: q > 0 = absorption (over-voltage), q < 0 = injection.
# Q bounded by min(capability circle, cos φ_min).
VDE_V_LOW: float = 0.95  # full Q+ injection at and below
VDE_V_DEADBAND_LOW: float = 0.97  # Q = 0 from here…
VDE_V_DEADBAND_HIGH: float = 1.03  # …to here
VDE_V_HIGH: float = 1.05  # full Q- absorption at and above

# VDE-AR-N 4105 §5.7.2 displacement-factor envelope by rating.
COS_PHI_THRESHOLD_MVA: float = 0.0138  # 13.8 kVA in MVA
COS_PHI_SMALL: float = 0.95  # for S_n ≤ 13.8 kVA
COS_PHI_LARGE: float = 0.90  # for S_n  > 13.8 kVA


def _vvw_enabled(behavior) -> bool:
    """True iff coordinated Volt-VAR-Watt support is configured.

    Read from the behavior-attached config; defaults False when absent."""
    cfg = getattr(behavior, "_scare_config", None)
    return bool(getattr(cfg, "enable_vvw_coordination", False))


def _qv_coordination_enabled(behavior) -> bool:
    """True iff the Q(U)-droop / curtailment-auction hand-off is configured.

    When on, the droop publishes its remaining reactive relief so the auction
    sheds active power only for the residual."""
    cfg = getattr(behavior, "_scare_config", None)
    return bool(getattr(cfg, "enable_qv_auction_coordination", False))


# dV/dQ sensitivity for the published reactive relief. The prior is a
# cold-start placeholder, demoted by two guards so it never decides a shed:
#   * confidence gate — advertise ZERO relief until the EMA has
#     ``_DVDQ_MIN_SAMPLES`` real samples (matches default scare baseline);
#   * safety asymmetry — advertise only ``_DVDQ_SAFETY`` of the measured relief,
#     so the estimate under-credits reactive (costs energy, never stability).
_DVDQ_PRIOR: float = 0.03
# Min |Δq| (MVar) before a (Δv, Δq) pair is trusted (below this ΔV is noise).
_DVDQ_MIN_DQ: float = 1e-3
_DVDQ_EMA_ALPHA: float = 0.3
# Real samples the EMA must accumulate before any relief is advertised.
_DVDQ_MIN_SAMPLES: int = 3
# Fraction of measured relief advertised (<1 ⇒ under-credit). Any value in
# (0, 1] is stable; a speed/energy knob, not load-bearing.
_DVDQ_SAFETY: float = 0.7


def vde_cos_phi_min(s_nom_mva: float) -> float:
    """VDE-AR-N 4105 minimum displacement factor for rating ``s_nom_mva``."""
    return COS_PHI_SMALL if s_nom_mva <= COS_PHI_THRESHOLD_MVA else COS_PHI_LARGE


def vde_q_curve(v_pu: float, q_max: float) -> float:
    """Evaluate the VDE-AR-N 4105 Q(U) characteristic.

    ``q_max`` is the capability-circle limit (MVar); Q is bounded to
    ``[-q_max, +q_max]``. Returns Q in monee load convention (positive =
    absorbs / over-voltage response, negative = injects / under-voltage).
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

    Installed on every PV-coupled :class:`PowerGenerator`. Each poll reads the
    parent node ``vm_pu``, computes the Q(U) setpoint clipped to the capability
    circle, and dispatches via ``behavior.act(aid, "set_q", q)``. Self-contained
    (no subscriptions); each distinct command records a ``qv_droop`` event.
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
            float(cos_phi_min)
            if cos_phi_min is not None
            else vde_cos_phi_min(self.s_nom_mva)
        )
        self.voltage_ref_pu = float(voltage_ref_pu)
        # Last commanded Q (monee convention); gates duplicate records.
        self._last_q: float | None = None
        # Last observed droop inputs; ``_step`` returns early if none moved
        # beyond ``_OBS_DELTA_TOL``.
        self._last_obs_v: float | None = None
        self._last_obs_p: float | None = None
        self._last_obs_reg: float | None = None
        # Last published reactive relief; re-stamped on cache-gated ticks (at
        # half the ledger TTL) so the relief stays fresh through static plateaus.
        self._last_relief: float | None = None
        self._last_relief_stamp_t: float = float("-inf")
        # Config flags are static per run; resolve once.
        self._qv_coordination: bool = _qv_coordination_enabled(behavior)
        # |dV/dQ| EMA (p.u. voltage per MVar) for the published reactive relief,
        # seeded from the ``qv_dvdq_prior`` config.
        cfg = getattr(behavior, "_scare_config", None)
        self._dvdq: float = float(getattr(cfg, "qv_dvdq_prior", _DVDQ_PRIOR))
        self._dvdq_n: int = 0  # real (Δv, Δq) samples folded into the EMA
        self._sens_prev_v: float | None = None
        self._sens_prev_q: float | None = None

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(Sector.ELECTRICITY, {}).get("poll_period_s", 0.5)
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
        # Generator p_mw is negative in monee load convention; the magnitude
        # enters the capability circle.
        p_raw = obs.get("p_mw", 0.0)
        try:
            p_mag = abs(float(p_raw))
        except (TypeError, ValueError):
            p_mag = 0.0
        if not math.isfinite(p_mag):
            p_mag = 0.0
        # Scale by ``regulation`` for the dispatched real power.
        regulation = float(obs.get("regulation", 1.0))
        if not math.isfinite(regulation):
            regulation = 1.0
        regulation = max(0.0, min(2.0, regulation))
        # Cache gate: skip when every droop input is within tolerance of the
        # previous tick (first call passes, caches None).
        if (
            self._last_obs_v is not None
            and self._last_obs_p is not None
            and self._last_obs_reg is not None
            and abs(v_f - self._last_obs_v) < _OBS_DELTA_TOL
            and abs(p_mag - self._last_obs_p) < _OBS_DELTA_TOL
            and abs(regulation - self._last_obs_reg) < _OBS_DELTA_TOL
        ):
            # Unchanged inputs => unchanged relief; still re-stamp the ledger
            # so the auction hand-off doesn't lose it during static plateaus.
            # Half-TTL pacing keeps it fresh without a write every poll.
            if self._qv_coordination and self._last_relief is not None:
                now = self.context.current_timestamp
                if now - self._last_relief_stamp_t >= _QV_RELIEF_TTL_S / 2.0:
                    publish_qv_relief(
                        self.behavior,
                        self.context.aid,
                        self._last_relief,
                        now,
                        v_pu=v_f,
                    )
                    self._last_relief_stamp_t = now
            return
        self._last_obs_v = v_f
        self._last_obs_p = p_mag
        self._last_obs_reg = regulation
        p_dispatched = p_mag * regulation
        s_sq = self.s_nom_mva * self.s_nom_mva
        # p_dispatched > s_nom collapses the capability circle to 0 reactive.
        # Warn rather than silently disable Q so the rating mismatch is visible.
        if p_dispatched > self.s_nom_mva + 1e-9:
            logger.warning(
                "[%s] qv_droop: p_dispatched=%.4g > s_nom=%.4g (over-rated "
                "real power); Q capability collapses to 0 this tick.",
                self.context.aid,
                p_dispatched,
                self.s_nom_mva,
            )
        circle_q = max(0.0, math.sqrt(max(0.0, s_sq - p_dispatched * p_dispatched)))
        # Displacement-factor envelope (VDE-AR-N 4105 §5.7.2):
        # |tan φ_min| = √(1 − cos²) / cos.
        cos = max(1e-6, min(1.0, self.cos_phi_min))
        cos_phi_q = max(0.0, p_dispatched) * math.sqrt(1.0 - cos * cos) / cos
        # Coordinated VVW (IEEE 1547-2018): use the FULL capability circle, not
        # the tighter cos-φ cap. Only matters once the auction trims ``p``: freed
        # apparent capacity goes to reactive, clearing over-voltage with less curtailment.
        if _vvw_enabled(self.behavior):
            q_max = circle_q
        else:
            q_max = min(circle_q, cos_phi_q) if cos_phi_q > 0.0 else circle_q
        if q_max <= 0.0:
            # No reactive headroom (full output, or off).
            q_cmd = 0.0
        else:
            q_cmd = vde_q_curve(v_f / self.voltage_ref_pu, q_max)
        # Coordinated hand-off: publish remaining reactive voltage-relief so the
        # auction sheds active only for the residual. ``headroom = q_max-|q_cmd|``
        # is the unused circle (~0 at saturation ⇒ no discount). Published before
        # the idempotency skip so it stays fresh on a no-op tick.
        if self._qv_coordination:
            if self._sens_prev_v is not None and self._sens_prev_q is not None:
                dq = q_cmd - self._sens_prev_q
                dv = v_f - self._sens_prev_v
                if abs(dq) >= _DVDQ_MIN_DQ and math.isfinite(dv):
                    sample = abs(dv / dq)
                    # Clamp absurd jumps (post-failure snapshots) before the EMA.
                    sample = min(sample, 10.0 * self._dvdq + 1.0)
                    self._dvdq = (
                        1.0 - _DVDQ_EMA_ALPHA
                    ) * self._dvdq + _DVDQ_EMA_ALPHA * sample
                    self._dvdq_n += 1
            self._sens_prev_v = v_f
            self._sens_prev_q = q_cmd
            # Confidence gate + safety asymmetry: nothing until calibrated, then
            # only a safe fraction.
            if self._dvdq_n >= _DVDQ_MIN_SAMPLES:
                headroom = max(0.0, q_max - abs(q_cmd))
                relief = headroom * self._dvdq * _DVDQ_SAFETY
            else:
                relief = 0.0
            self._last_relief = relief
            self._last_relief_stamp_t = self.context.current_timestamp
            publish_qv_relief(
                self.behavior,
                self.context.aid,
                relief,
                self.context.current_timestamp,
                v_pu=v_f,
            )
        # Idempotency: skip the act() write when within tolerance of the last.
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
