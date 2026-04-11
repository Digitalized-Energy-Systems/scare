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
from scare.base.util import kgps_to_mw, mw_to_kgps, obs_setpoint

if TYPE_CHECKING:
    from distributed_resource_optimization import ADMMFlexActor
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Maps sector to the obs key that holds the current setpoint for that sector
_ACCESS_KEYS: dict[Sector, str] = {
    Sector.ELECTRICITY: "el_mw",
    Sector.GAS: "gas_kgps",
    Sector.HEAT: "heat_w",
}

# ADMM result index for each sector
_RESULT_INDEX: dict[Sector, int] = {
    Sector.ELECTRICITY: 0,
    Sector.HEAT: 1,
    Sector.GAS: 2,
}


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

    def setup(self) -> None:
        self.context.subscribe_message(
            self,
            self._handle_ask_energy,
            lambda msg, meta: isinstance(msg, AskEnergyMessage),
        )
        self.context.subscribe_message(
            self,
            self._handle_negotiation_finished,
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        self.context.subscribe_message(
            self,
            self._handle_flex_answer,
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
        if topology_characteristic(self, tid="cps") == "leader" and not self._active:
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

        answers = self._flex_answers[:]
        self._flex_answers = []
        self._flex_expected = 0

        flex_by_sector: dict[Sector, float] = {}
        for answer in answers:
            flex_by_sector[answer.sector] = (
                flex_by_sector.get(answer.sector, 0.0) + answer.flex + answer.balance
            )

        flex_el = flex_by_sector.get(Sector.ELECTRICITY, 0.0)
        flex_heat = flex_by_sector.get(Sector.HEAT, 0.0)
        flex_gas = flex_by_sector.get(Sector.GAS, 0.0)

        T = np.array([flex_el, flex_heat / 1e6, kgps_to_mw(flex_gas)])

        if np.all(T >= 0) or np.all(T <= 0):
            logger.debug("[%s] ADMM skipped: T=%s (same sign)", self.context.aid, T)
            self._active = False
            return

        coordinator = create_sharing_target_distance_admm_coordinator()
        start_msg = create_admm_start(create_admm_sharing_data(T.tolist()))

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

        self._active = False

    def _apply_result(self, result: list[float]) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        # result layout: [0=EL, 1=HEAT, 2=GAS]
        for sector, idx in _RESULT_INDEX.items():
            key = _ACCESS_KEYS[sector]
            if key not in obs or idx >= len(result):
                continue
            value = result[idx]
            if sector == Sector.GAS:
                value = mw_to_kgps(value)
            cap = float(obs.get(key, 0.0))
            if cap == 0.0:
                continue
            factor = max(0.0, min(1.0, abs(value / cap)))
            if self.behavior.has_action(self.context.aid, "regulate"):
                self.behavior.act(self.context.aid, "regulate", factor)
            break
