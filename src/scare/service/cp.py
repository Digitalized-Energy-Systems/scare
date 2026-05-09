from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
)

from scare.base.model import (
    AskEnergyMessage,
    AskForAvailableFlex,
    AvailableFlexAnswer,
    NegotiationFinishedEvent,
    OptimizationFinishedLocalEvent,
    ResponseEnergyMessage,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.util import clamp_to_constraints, kgps_to_mw, mw_to_kgps, obs_setpoint

if TYPE_CHECKING:
    from distributed_resource_optimization import ADMMFlexActor
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Maps sector to the obs key that holds the current setpoint for that sector
_ACCESS_KEYS: dict[Sector, str] = {
    Sector.ELECTRICITY: "el_mw",
    Sector.GAS: "gas_kgps",
    Sector.HEAT: "heat_mw",
}

# ADMM result index for each sector
_RESULT_INDEX: dict[Sector, int] = {
    Sector.ELECTRICITY: 0,
    Sector.HEAT: 1,
    Sector.GAS: 2,
}

# --- CP ↔ sector fixed-point tolerance ---
# A NegotiationFinishedEvent re-triggers CP ADMM only if the sector's
# new setpoint differs from the last one observed here by more than
# this tolerance.  Below the tolerance the loop has reached a
# fixed-point on the CP side and re-triggering would just ping-pong
# back to the group and waste messages.  Units match _ACCESS_KEYS
# (MW for electricity, kg/s for gas, W for heat).
_CP_SETPOINT_TOLERANCE: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,   # MW
    Sector.GAS: 1e-4,           # kg/s
    Sector.HEAT: 1e-4,          # MW (~100 W)
}
_CP_DEFAULT_TOLERANCE = 0.01


class EnergyConverterRole(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        flex_actor: ADMMFlexActor,
        sectors: list[Sector],
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.flex_actor = flex_actor
        self.sectors = sectors

        self._active: bool = False
        self._flex_answers: list[AvailableFlexAnswer] = []
        self._flex_expected: int = 0

        # Per-sector last-observed group setpoint; used to suppress
        # re-triggering CP ADMM when the balance negotiation converged
        # to essentially the same point as before (fixed-point gate).
        self._last_sector_setpoint: dict[Sector, float] = {}

        # Skip-count throttle: ADMM may decline to run because the
        # imbalance vector has the same sign across all sectors (no
        # cross-sector trade improves the objective).  Log every Nth
        # such skip at INFO so it is visible without flooding.
        self._same_sign_skip_count: int = 0

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_energy),
            lambda msg, meta: isinstance(msg, AskEnergyMessage),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_finished),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_flex_answer),
            lambda msg, meta: isinstance(msg, AvailableFlexAnswer),
        )

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        key = _ACCESS_KEYS.get(message.sector)
        if key and key in obs:
            value = float(obs[key]) * float(obs.get("regulation", 1.0))
        else:
            value = obs_setpoint(obs)
        if math.isnan(value):
            value = 0.0
        # CP agents report available=0: they have no spare flex of their own
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=value,
            available=0.0,
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_negotiation_finished(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        if topology_characteristic(self, tid="cps") != "leader":
            return
        if self._active:
            return
        # Fixed-point gate: skip re-trigger if this sector's group
        # setpoint has not moved enough to change the ADMM answer.
        sector = message.sector
        tol = _CP_SETPOINT_TOLERANCE.get(sector, _CP_DEFAULT_TOLERANCE)
        prev = self._last_sector_setpoint.get(sector)
        new = message.new_setpoint
        if prev is not None and abs(new - prev) < tol:
            logger.debug(
                "[%s] CP re-trigger suppressed (sector=%s, |Δ|=%.6g < %.6g)",
                self.context.aid,
                sector.value,
                abs(new - prev),
                tol,
            )
            return
        self._last_sector_setpoint[sector] = new
        self.context.schedule_instant_task(self.trigger_cp_negotiation())

    async def trigger_cp_negotiation(self) -> None:
        if topology_characteristic(self, tid="cps") != "leader":
            return
        if self._active:
            return
        self._active = True

        group_leaders = topology_connectors(self, tid="cps")
        if not group_leaders:
            self._active = False
            return

        self._flex_answers = []
        self._flex_expected = len(group_leaders)

        msg = AskForAvailableFlex(include_connectors=False)
        for addr in group_leaders:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_flex_answer(
        self, message: AvailableFlexAnswer, meta: dict
    ) -> None:
        if not self._active:
            return

        self._flex_answers.append(message)

        if len(self._flex_answers) >= self._flex_expected:
            await self._run_admm()

    async def _run_admm(self) -> None:
        import numpy as np
        from distributed_resource_optimization import (
            create_admm_sharing_data,
            create_admm_start,
            create_sharing_target_distance_admm_coordinator,
            start_coordinated_optimization,
        )

        from scare.base.util import aggregate_priority_weight

        answers = self._flex_answers[:]
        self._flex_answers = []
        self._flex_expected = 0

        imbalance_by_sector: dict[Sector, float] = {}
        # Aggregate priority urgency per sector from all responding groups.
        sector_priority_weight: dict[Sector, float] = {}
        for answer in answers:
            # Use balance (net setpoint = generation + load) as the sector
            # imbalance.  Negative balance = excess generation, positive =
            # unmet demand.  The CP should shift its operating point to
            # compensate for the imbalance, not the total capacity.
            imbalance_by_sector[answer.sector] = (
                imbalance_by_sector.get(answer.sector, 0.0) + answer.balance
            )
            w = aggregate_priority_weight(
                answer.demand_by_priority, answer.served_by_priority
            )
            sector_priority_weight[answer.sector] = (
                sector_priority_weight.get(answer.sector, 0.0) + w
            )

        imb_el = imbalance_by_sector.get(Sector.ELECTRICITY, 0.0)
        imb_heat = imbalance_by_sector.get(Sector.HEAT, 0.0)
        imb_gas = imbalance_by_sector.get(Sector.GAS, 0.0)

        # Imbalances are now reported in their natural sector units —
        # electricity in MW, heat in MW (was W), gas in kg/s.  ADMM
        # lives in MW across all dimensions, so only gas needs unit
        # conversion; the historical /1e6 on heat is no longer correct
        # since monee switched ``q_w_heat`` → ``q_mw_heat``.
        T = np.array([imb_el, imb_heat, kgps_to_mw(imb_gas)])

        if np.all(T >= 0) or np.all(T <= 0):
            self._same_sign_skip_count += 1
            # First skip per leader and every 10th thereafter at INFO so
            # the suppressed coordination is visible without log flooding.
            if (
                self._same_sign_skip_count == 1
                or self._same_sign_skip_count % 10 == 0
            ):
                logger.info(
                    "[%s] CP ADMM skipped (same-sign T=%s, n=%d)",
                    self.context.aid,
                    T.tolist(),
                    self._same_sign_skip_count,
                )
            self._active = False
            return

        # Per-sector priority weights for the ADMM sharing problem.
        # Sectors with higher-priority unserved demand get stronger
        # weight, biasing the CP operating point toward those sectors.
        w_el = sector_priority_weight.get(Sector.ELECTRICITY, 1.0)
        w_heat = sector_priority_weight.get(Sector.HEAT, 1.0)
        w_gas = sector_priority_weight.get(Sector.GAS, 1.0)
        w_max = max(w_el, w_heat, w_gas, 1e-9)
        priorities = np.array([w_el, w_heat, w_gas]) / w_max  # normalise to [0, 1]
        priorities = np.maximum(priorities, 0.01)  # floor to avoid zero-weight

        coordinator = create_sharing_target_distance_admm_coordinator()
        start_msg = create_admm_start(
            create_admm_sharing_data(T.tolist(), priorities=priorities.tolist())
        )

        try:
            await start_coordinated_optimization(
                [self.flex_actor], coordinator, start_msg
            )
            result = list(self.flex_actor.x)
            logger.info("[%s] ADMM result: %s", self.context.aid, result)
            self.context.emit_event(OptimizationFinishedLocalEvent(result=result))
            self._apply_result(result)
            for addr in topology_connectors(self, tid="cps"):
                await self.context.send_message(StartBalanceNegotiation(), receiver_addr=addr)
        except Exception as exc:
            logger.error("[%s] ADMM failed: %s", self.context.aid, exc)
            # Fallback: trigger intra-group gossip so groups can still
            # rebalance locally even though cross-sector optimisation failed.
            for addr in topology_connectors(self, tid="cps"):
                await self.context.send_message(
                    StartBalanceNegotiation(), receiver_addr=addr
                )

        self._active = False

    def _apply_result(self, result: list[float]) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        # result layout: [0=EL, 1=HEAT, 2=GAS]
        # A CP has a single regulation knob.  Compute a factor per sector
        # and apply the one with the strongest signal (largest |value/cap|
        # ratio after clamping).  This ensures the most constrained or
        # most demanded sector drives the operating point.
        best_factor: float | None = None
        best_weight = -1.0
        for sector, idx in _RESULT_INDEX.items():
            key = _ACCESS_KEYS[sector]
            if key not in obs or idx >= len(result):
                continue
            value = result[idx]
            if sector == Sector.GAS:
                value = mw_to_kgps(value)
            value = clamp_to_constraints(value, obs, sector)
            cap = float(obs.get(key, 0.0))
            if cap == 0.0:
                continue
            factor = max(0.0, min(1.0, abs(value / cap)))
            weight = abs(value)
            if weight > best_weight:
                best_weight = weight
                best_factor = factor

        if best_factor is not None:
            from scare.base.util import apply_regulate

            apply_regulate(
                self.behavior,
                self.context.aid,
                best_factor,
                sector="cp",
                reason="cp_admm",
                timestamp=self.context.current_timestamp,
            )
