"""Layer-0 gas pressure regulator on the external-grid slack.

A real gas distribution grid has no in-network compressors or pumps; the only
local pressure lever is the **regulator (pressure-reduction) station** at the
feed-in, which holds an outlet setpoint and lifts/lowers the whole fed profile.
This role models exactly that: an autonomous primary-control loop on each gas
``ExtHydrGrid`` slack that senses downstream junction pressure (reported up the
grid via the existing ``ConstraintStateMessage`` mesh), then nudges the slack
``pressure_pu`` setpoint to hold the fed subtree inside the operating band.

It runs *before* any negotiated shedding — hence "layer 0". Raising the source
datum is flow-neutral in this model (loads fix their withdrawals, so mass
balance and the slack draw are unchanged), so it is a near-free lever that
should be tried first. It does **not** drive shedding itself: when the profile
spread exceeds the band width, the setpoint saturates and the residual
under-pressure is left to the existing ``GridConstraintMonitor`` pressure path
(reducing load cuts flow, which shrinks the Weymouth drops and compresses the
spread). See ``project_gas_pressure_violations`` / ``reference_env_pkg_editable``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from mango import Role

from scare.base.model import (
    DEENERGISED_PRESSURE_HIGH_PU,
    DEENERGISED_PRESSURE_PU,
    SECTOR_CONSTRAINTS,
    SECTOR_TIMESCALE,
    ConstraintStateMessage,
    Sector,
)
from scare.base.runtime.diagnostics import record_event
from scare.base.util import lookup_slack_pressure, set_slack_pressure

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Below any monitored-pressure precision: a setpoint move smaller than this is a
# no-op (avoids churning monee state / the dirty flag on rounding wiggle).
_DEADBAND_PU: float = 1e-3

# Comfortable in-band margin (fraction of band width) required on BOTH extremes
# before the regulator relaxes its setpoint back toward nominal. Keeps the
# relaxation from nudging a node back across a bound and limit-cycling against
# the shedding path.
_RELAX_MARGIN_FRAC: float = 0.15

# Freshness window for a downstream pressure report, as a multiple of the gas
# poll period. Reports older than this are dropped (post-failure topology may
# have orphaned the origin).
_FRESHNESS_POLLS: float = 4.0


class GasPressureRegulator(Role):
    """Hold downstream gas pressure inside the band by driving the slack
    ``pressure_pu`` setpoint (the regulator-station lever).

    Args:
        behavior: the restoration environment behavior (observe / act).
        sector: must be ``Sector.GAS``; the role is a no-op otherwise.
        gain: feedback step fraction toward the target setpoint each poll.
            <1 because the Weymouth map is nonlinear (a one-shot linear delta
            overshoots); the loop converges over a few gas ticks.
        nominal_pu: setpoint the regulator relaxes back toward when the profile
            sits comfortably in band, recovering headroom for the next event.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        gain: float = 0.5,
        nominal_pu: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.gain = float(gain)
        self.nominal = float(nominal_pu)
        lo, hi = SECTOR_CONSTRAINTS.get(sector, {}).get(
            "pressure_pu", (0.85, 1.25)
        )
        self._lo = float(lo)
        self._hi = float(hi)
        self._relax_margin = _RELAX_MARGIN_FRAC * (self._hi - self._lo)
        # origin_addr_str -> (pressure_pu, timestamp) from the constraint mesh.
        self._reports: dict[str, tuple[float, float]] = {}
        self._freshness_s: float = 1.0

    def setup(self) -> None:
        if self.sector is not Sector.GAS:
            return
        poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 2.0)
        self._freshness_s = _FRESHNESS_POLLS * float(poll)
        self.context.schedule_periodic_task(self._control, delay=poll)
        self.context.subscribe_message(
            self,
            self._on_state,
            lambda msg, meta: (
                isinstance(msg, ConstraintStateMessage)
                and msg.sector is Sector.GAS
                and msg.variable == "pressure_pu"
            ),
        )

    # -- sensing -------------------------------------------------------------

    def _on_state(self, message: ConstraintStateMessage, meta: dict) -> None:
        """Record a downstream junction pressure report (reverse-flow telemetry
        toward the slack, carried by the existing multi-hop mesh)."""
        val = message.value
        if val is None or not math.isfinite(val):
            return
        key = str(message.origin_addr)
        self._reports[key] = (float(val), float(self.context.current_timestamp))

    def _safe_observe(self) -> dict[str, Any] | None:
        try:
            obs = self.behavior.observe(self.context.aid)
        except Exception:  # noqa: BLE001
            return None
        return obs or None

    @staticmethod
    def _is_energised_pressure(val: float) -> bool:
        """True for a physically meaningful junction pressure. Excludes BOTH
        de-energised artifacts: a source-isolated region collapses to ~0, and a
        zero-flow / P2G junction can saturate monee's relaxed-Weymouth
        ``pressure_squared_pu`` box at its upper bound, reading pressure_pu~sqrt(3).
        Acting on the high artifact made the regulator chase a phantom
        over-pressure and walk the whole profile down to the floor. The
        scan/monitor drop both the same way — see DEENERGISED_PRESSURE_*."""
        return DEENERGISED_PRESSURE_PU < val < DEENERGISED_PRESSURE_HIGH_PU

    def _fed_pressures(self, own_p: float | None, now: float) -> list[float]:
        """Energised pressures over the fed subtree: fresh mesh reports plus the
        slack's own node. De-energised (source-isolated, ~0) and solver-saturated
        (~sqrt(3)) readings are dropped — no setpoint move re-pressurises a region
        cut off from its source, and the high artifact is not a real breach."""
        out: list[float] = []
        for val, t in self._reports.values():
            if (now - t) <= self._freshness_s and self._is_energised_pressure(val):
                out.append(val)
        if (
            own_p is not None
            and math.isfinite(own_p)
            and self._is_energised_pressure(float(own_p))
        ):
            out.append(float(own_p))
        return out

    # -- control -------------------------------------------------------------

    async def _control(self) -> None:
        if self.sector is not Sector.GAS:
            return
        obs = self._safe_observe()
        if obs is None:
            return
        # Current setpoint: last commanded value, else the pinned node pressure.
        p_s = lookup_slack_pressure(self.behavior, self.context.aid)
        if p_s is None:
            p_s = obs.get("pressure_pu")
        if p_s is None or not math.isfinite(p_s):
            return
        p_s = float(p_s)

        now = float(self.context.current_timestamp)
        fed = self._fed_pressures(obs.get("pressure_pu"), now)
        if not fed:
            return
        p_min = min(fed)
        p_max = max(fed)
        lo, hi = self._lo, self._hi

        # Raising the source datum lifts the whole fed profile ~uniformly, so the
        # binding cap is the TOP node (``p_max``), not the slack's own ceiling.
        # ``saturated`` marks a spread wider than the band: the setpoint alone
        # cannot satisfy both extremes (only cutting flow via shedding can).
        target_delta = 0.0
        reason = ""
        saturated_kind = ""
        if p_min < lo:  # under-pressure somewhere -> raise within top headroom
            headroom = max(0.0, hi - p_max)
            target_delta = min(lo - p_min, headroom)
            reason = "underpressure"
            if (lo - p_min) > headroom + _DEADBAND_PU:
                saturated_kind = "gas_pressure_setpoint_saturated"
        elif p_max > hi:  # over-pressure -> lower within bottom headroom
            headroom = max(0.0, p_min - lo)
            target_delta = -min(p_max - hi, headroom)
            reason = "overpressure"
            if (p_max - hi) > headroom + _DEADBAND_PU:
                saturated_kind = "gas_pressure_overpressure_trap"
        else:
            # In band: relax the setpoint toward nominal to recover headroom for
            # the next event. Move toward nominal on whichever side it lies,
            # bounded by the headroom on THAT side minus a buffer, so we keep a
            # margin off the bound and never limit-cycle with the shedding path.
            if p_s > self.nominal + _DEADBAND_PU:  # elevated -> relax down
                low_room = (p_min - lo) - self._relax_margin
                if low_room <= _DEADBAND_PU:
                    return
                target_delta = -min(p_s - self.nominal, low_room)
                reason = "relax"
            elif p_s < self.nominal - _DEADBAND_PU:  # depressed -> relax up
                high_room = (hi - p_max) - self._relax_margin
                if high_room <= _DEADBAND_PU:
                    return
                target_delta = min(self.nominal - p_s, high_room)
                reason = "relax"
            else:
                return

        # Feedback step (not one-shot): converge over a few gas ticks. A move may
        # be below the deadband (e.g. headroom exhausted) yet still saturated —
        # surface saturation regardless so the residual is shed downstream.
        if abs(target_delta) >= _DEADBAND_PU:
            new_setpoint = max(lo, min(hi, p_s + self.gain * target_delta))
            if abs(new_setpoint - p_s) >= _DEADBAND_PU:
                set_slack_pressure(self.behavior, self.context.aid, new_setpoint)
                record_event(
                    t=now,
                    kind=f"gas_pressure_setpoint_{reason}",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=(
                        f"setpoint {p_s:.4f}->{new_setpoint:.4f} "
                        f"p_min={p_min:.4f} p_max={p_max:.4f} "
                        f"band=({lo:.2f},{hi:.2f})"
                    ),
                )

        # Setpoint saturated: it cannot clear the breach alone (profile spread
        # exceeds the band). The two sides differ in what can mop up the residual:
        #   - UNDER-pressure: cutting flow shrinks the Weymouth drops and lifts
        #     downstream pressure, so the existing GridConstraintMonitor
        #     pressure_pu -> curtailment path genuinely helps. Leave it to that.
        #   - OVER-pressure: cutting flow shrinks the drops and RAISES pressure,
        #     making it WORSE. Shedding cannot relieve over-pressure; the
        #     real-world resolution is pressure relief / slam-shut isolation.
        #     Surface it as un-actionable rather than implying shedding clears it.
        if saturated_kind:
            if saturated_kind == "gas_pressure_setpoint_saturated":
                residual = "residual under-pressure left to curtailment path"
            else:
                residual = "over-pressure not relievable by shedding (relief/isolation)"
            record_event(
                t=now,
                kind=saturated_kind,
                aid=self.context.aid,
                sector=self.sector.value,
                detail=(
                    f"spread={p_max - p_min:.4f} > band={hi - lo:.4f}; "
                    f"p_min={p_min:.4f} p_max={p_max:.4f}; {residual}"
                ),
            )
