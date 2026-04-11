from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
    topology_neighbors,
)

from scare.base.model import (
    AskEnergyMessage,
    AskForAvailableFlex,
    AvailableFlexAnswer,
    BalanceProblem,
    EnergyNegotiationMessage,
    NegotiationFinishedEvent,
    ResponseEnergyMessage,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.util import obs_capacity, obs_min_max, obs_sector, obs_setpoint

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

_START_THRESHOLD: dict[Sector, float] = {Sector.HEAT: 10.0}
_DEFAULT_START_THRESHOLD = 1e-4

_MAX_HOPS = 100


def _start_threshold(sector: Sector) -> float:
    return _START_THRESHOLD.get(sector, _DEFAULT_START_THRESHOLD)


@dataclass
class _GossipState:
    negotiation_id: str
    target: float
    counter: int
    current_delta: float
    starting_setpoint: float
    # addr_str -> (delta, counter_when_set)
    memory: dict[str, tuple[float, int]] = field(default_factory=dict)


class EnergyBalanceNegotiator(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        priority: int = 0,
        convergence_rate: float = 0.5,
        impact_weight: float = 1.0,
        termination_tolerance: float = 1e-5,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.priority = priority
        self.convergence_rate = convergence_rate
        self.impact_weight = impact_weight
        self.termination_tolerance = termination_tolerance

        self._active: bool = False
        self._gossip: _GossipState | None = None
        # State for the setpoint-gathering phase before gossip starts
        self._trigger_nid: str | None = None
        self._trigger_responses: dict[str, float] = {}
        self._trigger_expected: int = 0

    def setup(self) -> None:
        self.context.subscribe_message(
            self,
            self._handle_ask_energy,
            lambda msg, meta: (
                isinstance(msg, AskEnergyMessage) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            self._handle_response_energy,
            lambda msg, meta: isinstance(msg, ResponseEnergyMessage),
        )
        self.context.subscribe_message(
            self,
            self._handle_negotiation_message,
            lambda msg, meta: (
                isinstance(msg, EnergyNegotiationMessage) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            self._handle_negotiation_finished_msg,
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        self.context.subscribe_message(
            self,
            self._handle_ask_flex,
            lambda msg, meta: isinstance(msg, AskForAvailableFlex),
        )
        self.context.subscribe_message(
            self,
            self._handle_start_balance,
            lambda msg, meta: isinstance(msg, StartBalanceNegotiation),
        )
        self.context.subscribe_event(self, BalanceProblem, self._on_balance_problem)

    async def trigger_balance_negotiation(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        if self._active:
            return
        self._active = True

        neighbours = topology_neighbors(self, tid="groups")
        if not neighbours:
            obs = self.behavior.observe(self.context.aid) or {}
            await self._start_gossip(-obs_setpoint(obs))
            return

        nid = str(uuid4())
        self._trigger_nid = nid
        self._trigger_responses = {}
        self._trigger_expected = len(neighbours)

        msg = AskEnergyMessage(negotiation_id=nid, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs)
        sp = obs_setpoint(obs)
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=sp,
            available=cap - sp,  # headroom, not total capacity
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_response_energy(
        self, message: ResponseEnergyMessage, meta: dict
    ) -> None:
        if message.negotiation_id != self._trigger_nid:
            return

        sender_key = str(mango_sender_addr(meta))
        self._trigger_responses[sender_key] = message.setpoint

        if len(self._trigger_responses) >= self._trigger_expected:
            own_obs = self.behavior.observe(self.context.aid) or {}
            total_sp = obs_setpoint(own_obs) + sum(self._trigger_responses.values())
            self._trigger_nid = None
            self._trigger_responses = {}
            await self._start_gossip(-total_sp)

    async def _start_gossip(self, target: float) -> None:
        if abs(target) < _start_threshold(self.sector):
            self._active = False
            return

        neighbours = topology_neighbors(self, tid="groups")
        nid = str(uuid4())
        self_key = str(self.context.addr)

        obs = self.behavior.observe(self.context.aid) or {}
        starting_sp = obs_setpoint(obs)

        self._gossip = _GossipState(
            negotiation_id=nid,
            target=target,
            counter=0,
            current_delta=0.0,
            starting_setpoint=starting_sp,
            memory={self_key: (0.0, 0)},
        )

        if not neighbours:
            await self._finish_negotiation()
            return

        msg = EnergyNegotiationMessage(
            negotiation_id=nid,
            sector=self.sector,
            negotiation_target=target,
            current_delta=0.0,
            counter=0,
            negotiation_memory=dict(self._gossip.memory),
        )
        for addr in neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_negotiation_message(
        self, message: EnergyNegotiationMessage, meta: dict
    ) -> None:
        nid = message.negotiation_id
        counter = message.counter + 1

        if counter > _MAX_HOPS + 1:
            return

        if self._gossip is None or self._gossip.negotiation_id != nid:
            obs = self.behavior.observe(self.context.aid) or {}
            self._gossip = _GossipState(
                negotiation_id=nid,
                target=message.negotiation_target,
                counter=counter,
                current_delta=0.0,
                starting_setpoint=obs_setpoint(obs),
                memory=dict(message.negotiation_memory),
            )
        else:
            self._gossip.counter = counter
            for k, (delta, cnt) in message.negotiation_memory.items():
                existing = self._gossip.memory.get(k)
                if existing is None or cnt > existing[1]:
                    self._gossip.memory[k] = (delta, cnt)

        target = self._gossip.target
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs)
        dmin, dmax = obs_min_max(obs)
        self_key = str(self.context.addr)

        total_delta = sum(d for d, _ in self._gossip.memory.values())
        open_gap = target - total_delta
        own_change = (
            open_gap * (abs(cap) / 20.0) * self.impact_weight * self.convergence_rate
        )

        # Dynamic priority: direction of target determines which type participates first.
        # target < 0 → need reduction → generators (priority=0) first, loads (priority>0) wait.
        # target > 0 → need increase → loads (priority>0) first, generators (priority=0) wait.
        if target < 0:
            actual_prio = 1000 if self.priority > 0 else 0
        elif target > 0:
            actual_prio = 0 if self.priority > 0 else 1000
        else:
            actual_prio = self.priority

        if actual_prio <= counter:
            current_own = self._gossip.memory.get(self_key, (0.0, 0))[0]
            new_delta = max(dmin, min(dmax, current_own + own_change))
            self._gossip.memory[self_key] = (new_delta, counter)
            self._gossip.current_delta = new_delta
            if cap != 0.0:
                self._apply_setpoint(self._gossip.starting_setpoint + new_delta)

        total_delta = sum(d for d, _ in self._gossip.memory.values())
        open_gap = target - total_delta

        neighbours = topology_neighbors(self, tid="groups")

        if abs(open_gap) <= self.termination_tolerance or counter >= _MAX_HOPS:
            await self._finish_negotiation()
        elif neighbours:
            next_addr = random.choice(neighbours)
            fwd = EnergyNegotiationMessage(
                negotiation_id=nid,
                sector=self.sector,
                negotiation_target=target,
                current_delta=self._gossip.current_delta,
                counter=counter,
                negotiation_memory=dict(self._gossip.memory),
            )
            await self.context.send_message(fwd, receiver_addr=next_addr)

    async def _finish_negotiation(self) -> None:
        starting_sp = (
            self._gossip.starting_setpoint
            if self._gossip
            else obs_setpoint(self.behavior.observe(self.context.aid) or {})
        )
        delta = self._gossip.current_delta if self._gossip else 0.0
        new_sp = starting_sp + delta

        self.context.emit_event(
            NegotiationFinishedEvent(new_setpoint=new_sp, sector=self.sector)
        )

        neighbours = topology_neighbors(self, tid="groups")

        # Broadcast convergence to all group neighbours so each can emit its own local event
        finished_msg = NegotiationFinishedEvent(new_setpoint=0, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(finished_msg, receiver_addr=addr)

        # Leader also notifies CP connectors
        if topology_characteristic(self, tid="groups") == "leader":
            for addr in topology_connectors(self, tid="groups"):
                await self.context.send_message(finished_msg, receiver_addr=addr)

        self._gossip = None
        self._active = False

    async def _handle_negotiation_finished_msg(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        """Convergence broadcast from a gossip peer — emit own local NegotiationFinishedEvent."""
        starting_sp = (
            self._gossip.starting_setpoint
            if self._gossip
            else obs_setpoint(self.behavior.observe(self.context.aid) or {})
        )
        delta = self._gossip.current_delta if self._gossip else 0.0
        self.context.emit_event(
            NegotiationFinishedEvent(
                new_setpoint=starting_sp + delta, sector=self.sector
            )
        )

    async def _handle_ask_flex(self, message: AskForAvailableFlex, meta: dict) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return

        member_aids = [self.context.aid] + [
            addr.aid for addr in topology_neighbors(self, tid="groups")
        ]

        if message.include_connectors:
            for addr in topology_connectors(self, tid="groups"):
                member_aids.append(addr.aid)

        total_flex = 0.0
        total_balance = 0.0
        total_shedded = 0.0
        for aid in member_aids:
            obs = self.behavior.observe(aid) or {}
            if obs_sector(obs) != self.sector:
                continue
            cap = obs_capacity(obs)
            sp = obs_setpoint(obs)
            available = cap - sp  # headroom
            total_flex += available
            total_balance += sp
            # shedded: headroom of agents with positive setpoint and positive headroom (active generators)
            if sp > 0 and available > 0:
                total_shedded += available

        reply = AvailableFlexAnswer(
            flex=total_flex,
            balance=total_balance,
            shedded=total_shedded,
            sector=self.sector,
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_start_balance(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        if topology_characteristic(self, tid="groups") == "leader":
            self.context.schedule_instant_task(self.trigger_balance_negotiation())

    def _on_balance_problem(self, event: BalanceProblem, _src: Any) -> None:
        if event.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") == "leader":
            self.context.schedule_instant_task(self.trigger_balance_negotiation())

    def _apply_setpoint(self, new_setpoint: float) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs)
        if cap == 0.0:
            return
        factor = max(0.0, min(1.0, abs(new_setpoint / cap)))
        if self.behavior.has_action(self.context.aid, "regulate"):
            self.behavior.act(self.context.aid, "regulate", factor)


def create_energy_balance_role(
    behavior: RestorationEnvironmentBehavior,
    sector: Sector,
    obs: dict,
    *,
    priority: int | None = None,
) -> EnergyBalanceNegotiator:
    if priority is None:
        # Generators: delta_min < 0 (can reduce below zero) → acts early (priority=0).
        # Loads: delta_min == 0 (bottoms out at zero) → waits (priority=1000).
        dmin, _ = obs_min_max(obs)
        priority = 1000 if dmin == 0.0 else 0
    return EnergyBalanceNegotiator(
        behavior=behavior,
        sector=sector,
        priority=priority,
    )
