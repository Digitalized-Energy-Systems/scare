from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from mango import Role
from mango.express.topology import topology_neighbors

from scare.base.model import (
    BalanceNegotiationStart,
    CommunityAssignment,
    NegotiationFinishedEvent,
    NegotiationResult,
    Sector,
)

logger = logging.getLogger(__name__)


class PreAssignedCommunityRole(Role):
    """Writes a pre-computed ``CommunityAssignment`` into the agent's
    context at startup.

    Groups are formed offline (label-propagation in the scenario
    builder) and primed into the ``CommunityAssignment`` model at
    setup so downstream roles like :class:`HolonicCommunityRole` can
    find a non-empty community immediately.
    """

    def __init__(self, community_id: UUID) -> None:
        super().__init__()
        self._community_id = community_id

    def setup(self) -> None:
        assignment = self.context.get_or_create_model(CommunityAssignment)
        assignment.community_id = self._community_id
        assignment.neighbors = list(topology_neighbors(self, tid="groups"))
        self.context.update(assignment)


class Community(Role):
    def __init__(self, sector: Sector) -> None:
        super().__init__()
        self.sector = sector
        # negotiation_id → list[NegotiationResult]
        self._collected: dict[str, list[NegotiationResult]] = {}

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_result),
            lambda msg, meta: isinstance(msg, NegotiationResult),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_balance_start),
            lambda msg, meta: isinstance(msg, BalanceNegotiationStart),
        )

    async def _handle_negotiation_result(
        self, message: NegotiationResult, meta: dict
    ) -> None:
        expected = len(topology_neighbors(self, tid="groups"))

        nid = "current"  # simplified: track single active negotiation
        self._collected.setdefault(nid, []).append(message)

        if len(self._collected[nid]) >= expected:
            aggregated = self._aggregate(self._collected.pop(nid))
            logger.info(
                "[%s] community negotiation complete: flex=%.4f sp=%.4f",
                self.context.aid,
                aggregated.flexibility,
                aggregated.control_setpoint,
            )
            self.context.emit_event(
                NegotiationFinishedEvent(
                    new_setpoint=aggregated.control_setpoint,
                    sector=self.sector,
                )
            )

    async def _handle_balance_start(
        self, message: BalanceNegotiationStart, meta: dict
    ) -> None:
        for addr in topology_neighbors(self, tid="groups"):
            await self.context.send_message(message, receiver_addr=addr)

    def _aggregate(self, results: list[NegotiationResult]) -> NegotiationResult:
        if not results:
            return NegotiationResult(flexibility=0.0, control_setpoint=0.0)
        total_flex = sum(r.flexibility for r in results)
        avg_sp = sum(r.control_setpoint for r in results) / len(results)
        return NegotiationResult(flexibility=total_flex, control_setpoint=avg_sp)


class CommunityParticipant(Role):
    def setup(self) -> None:
        pass

    async def send_to_community(self, message: Any) -> None:
        assignment = self.context.get_or_create_model(CommunityAssignment)
        for addr in assignment.neighbors:
            await self.context.send_message(message, receiver_addr=addr)

    def on_change_model(self, model: Any) -> None:
        if isinstance(model, CommunityAssignment):
            logger.debug(
                "[%s] community assignment updated: id=%s neighbours=%d",
                self.context.aid,
                model.community_id,
                len(model.neighbors),
            )
