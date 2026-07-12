"""Outlet-temperature guard for heat-producing coupling points.

A heat-producing CP (P2H / G2H / CHP) injects ``q_mw_heat`` into its outlet
junction as pure energy; on a low-flow junction the resulting temperature rise
``ΔT = Q / (ṁ·c_p)`` drives ``t_k`` far past the operating ceiling. Nothing in
the stack pushes back: the L3 kernel sizes CP regulation off ``demand −
delivered`` (delivered is measured at load setpoints, which injection can never
raise), the heat frontier owns only the LOW side of the band, the curtailment
auction skips ``t_k`` (and shedding heat *loads* on a hot junction removes
cooling draw — wrong direction), and CP models are BORN at ``regulation=1.0``,
so even a run with zero L3 commits injects at rated power.

This role is the missing high-side controller. Per heat-producing CP it polls
the outlet junction's ``t_k`` and maintains a regulation CEILING via AIMD:
multiplicative decrease while the outlet is (nearly) over-temperature,
slow additive recovery once it has cooled well clear, hysteresis hold in
between. The ceiling is published to a freshness-stamped behavior store and
enforced centrally in ``apply_regulate`` for every ``sector="cp"`` write, so
the kernel's deficit-driven re-commits cannot overwrite the wind-down. The
guard also self-actuates when the CP's standing regulation exceeds the
ceiling — this, not the commit clamp, is what covers the born state.

Deliberately flow-free: the junction's ``mass_flow_kgs`` intermediate sums
child (sink/source) flows only, not through-flow, so a physics-informed step
would mis-size on pass-through junctions. Geometric decrease clears any hot
state within a few polls regardless of the local flow regime.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from mango import Role

from scare.base.model import SECTOR_CONSTRAINTS, Sector
from scare.base.runtime.diagnostics import record_event
from scare.base.util import apply_regulate, publish_cp_heat_ceiling

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Poll period (sim-s). Faster than the heat SCADA poll (5 s) on purpose: a
# born-hot outlet must be cleared well before end-of-sim grading, and the
# store's freshness TTL (5 s) needs regular republish.
_GUARD_PERIOD_S: float = 1.0

# Wind-down engages at ``hi - _HI_MARGIN_K`` — before the bound, so the loop
# settles below it instead of oscillating across it (the compliance scan
# tolerates only 1 K past the bound).
_HI_MARGIN_K: float = 5.0

# Recovery engages only below ``hi - _RECOVER_BAND_K``; the 10 K gap between
# the two bands is the hysteresis that prevents a shed/restore limit cycle.
_RECOVER_BAND_K: float = 15.0

# Multiplicative wind-down per poll while hot. Geometric: clears any born-hot
# outlet (regulation 1.0) to a sub-headroom ceiling within ~3-4 polls without
# needing a flow model.
_DECREASE_FACTOR: float = 0.5

# Additive recovery per poll once cooled clear (AIMD) — slow on purpose; the
# kernel re-commits its full deficit-driven factor the moment the cap allows.
_RECOVER_STEP: float = 0.05

# Below this the geometric decrease crawls; snap to a full cut instead.
_CEILING_FLOOR: float = 0.02

# Regulation slack above the ceiling that triggers a self-actuated wind-down
# (matches apply_regulate's dedup tolerance scale).
_ACTUATE_TOL: float = 1e-3


class CPHeatOutletGuard(Role):
    """Hold a heat-producing CP's outlet junction inside the ``t_k`` envelope
    by capping the CP's regulation (see module docstring).

    Args:
        behavior: restoration environment behavior (observe / act).
        outlet_aid: aid of the agent whose observation carries the outlet
            junction's ``t_k`` — the heat-side node agent for branch-hosted
            CPs (P2H/G2H inject at the to-node), the ``SubHG`` child's node
            for CHP-HG compounds, the CP's own aid for HX control nodes
            (their model carries its own ``t_k``).
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        *,
        outlet_aid: str,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.outlet_aid = str(outlet_aid)
        _lo, hi = SECTOR_CONSTRAINTS.get(Sector.HEAT, {}).get(
            "t_k", (313.15, 403.15)
        )
        self._hi = float(hi)
        self._ceiling: float = 1.0

    def setup(self) -> None:
        self.context.schedule_periodic_task(self._control, delay=_GUARD_PERIOD_S)

    def _safe_observe(self, aid: str) -> dict[str, Any] | None:
        try:
            obs = self.behavior.observe(aid)
        except Exception:  # noqa: BLE001
            return None
        return obs or None

    def _outlet_t_k(self) -> float | None:
        """Outlet junction temperature, or None when unavailable or the
        junction is de-energised (isolated junctions read t_k ~0 — an
        artifact, not an operating state; same convention as the constraint
        monitor and the eval scan)."""
        obs = self._safe_observe(self.outlet_aid)
        if obs is None:
            return None
        t = obs.get("t_k")
        try:
            t = float(t)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(t) or t <= 0.0:
            return None
        return t

    async def _control(self) -> None:
        now = float(self.context.current_timestamp)
        t = self._outlet_t_k()
        if t is None:
            # No usable reading: hold and refresh a held ceiling (a transient
            # obs gap must not TTL-release the cap), never tighten or recover.
            if self._ceiling < 1.0:
                publish_cp_heat_ceiling(
                    self.behavior, self.context.aid, self._ceiling, now
                )
            return

        prev = self._ceiling
        if t > self._hi - _HI_MARGIN_K:
            ceiling = prev * _DECREASE_FACTOR
            if ceiling < _CEILING_FLOOR:
                ceiling = 0.0
        elif t < self._hi - _RECOVER_BAND_K:
            ceiling = min(1.0, prev + _RECOVER_STEP)
        else:
            ceiling = prev

        self._ceiling = ceiling
        # Publish every poll a cap is held (freshness stamp); >= 1.0 clears.
        publish_cp_heat_ceiling(self.behavior, self.context.aid, ceiling, now)

        if abs(ceiling - prev) > 1e-9:
            record_event(
                t=now,
                kind=(
                    "cp_heat_outlet_relief"
                    if ceiling < prev
                    else "cp_heat_outlet_recover"
                ),
                aid=str(self.context.aid),
                sector="cp",
                detail=(
                    f"t_k={t:.2f} hi={self._hi:.2f} "
                    f"ceiling {prev:.4f}->{ceiling:.4f}"
                ),
            )

        # Self-actuation: wind the STANDING regulation down to the ceiling.
        # Kernel commits are clamped in apply_regulate, but a CP that never
        # receives a commit (born r=1.0, empty demand sets) has no write to
        # clamp — the guard is its only actuator.
        if ceiling < 1.0:
            own = self._safe_observe(self.context.aid) or {}
            reg = own.get("regulation")
            try:
                reg = float(reg)
            except (TypeError, ValueError):
                reg = None
            if reg is not None and reg > ceiling + _ACTUATE_TOL:
                apply_regulate(
                    self.behavior,
                    self.context.aid,
                    ceiling,
                    sector="cp",
                    reason="cp_heat_outlet_relief",
                    timestamp=now,
                )
