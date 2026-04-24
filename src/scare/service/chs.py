from __future__ import annotations

import logging
from uuid import UUID, uuid4

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors

from scare.base.model import (
    CHSJoinRequest,
    CHSJoinRequestAnswer,
    CommunityAssignment,
)

logger = logging.getLogger(__name__)


class CHSRole(Role):
    def __init__(
        self,
        *,
        pooling_period_s: float = 5.0,
        max_size: int = 10,
    ) -> None:
        super().__init__()
        self.pooling_period_s = pooling_period_s
        self.max_size = max_size
        # group_id → {aid: accepted bool}
        self._pending_joins: dict[UUID, dict[str, bool | None]] = {}

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        self.context.subscribe_message(
            self,
            _wrap(self._handle_join_request),
            lambda msg, meta: isinstance(msg, CHSJoinRequest),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_join_answer),
            lambda msg, meta: isinstance(msg, CHSJoinRequestAnswer),
        )
        self.context.schedule_periodic_task(
            self._pool_for_community, delay=self.pooling_period_s
        )

    async def _pool_for_community(self) -> None:
        assignment = self.context.get_or_create_model(CommunityAssignment)
        if assignment.community_id is not None:
            return

        neighbours = topology_neighbors(self, tid="groups")

        if not neighbours:
            return

        group_id = uuid4()
        self._pending_joins[group_id] = {str(a): None for a in neighbours}

        req = CHSJoinRequest(group_id=group_id, group_size=len(neighbours) + 1)
        for addr in neighbours[: self.max_size - 1]:
            await self.context.send_message(req, receiver_addr=addr)

    async def _handle_join_request(self, message: CHSJoinRequest, meta: dict) -> None:
        assignment = self.context.get_or_create_model(CommunityAssignment)
        accept = assignment.community_id is None
        if accept:
            assignment.community_id = message.group_id
            assignment.neighbors.append(mango_sender_addr(meta))
            self.context.update(assignment)

        reply = CHSJoinRequestAnswer(group_id=message.group_id, accept=accept)
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_join_answer(
        self, message: CHSJoinRequestAnswer, meta: dict
    ) -> None:
        gid = message.group_id
        if gid not in self._pending_joins:
            return

        sender_key = meta.get("sender_id", str(mango_sender_addr(meta)))
        self._pending_joins[gid][sender_key] = message.accept

        responses = self._pending_joins[gid]
        if all(v is not None for v in responses.values()):
            accepted_addrs = [addr for addr, accepted in responses.items() if accepted]
            if accepted_addrs:
                assignment = self.context.get_or_create_model(CommunityAssignment)
                assignment.community_id = gid
                assignment.neighbors = list(topology_neighbors(self, tid="groups"))
                self.context.update(assignment)
                logger.info(
                    "[%s] community %s formed with %d members",
                    self.context.aid,
                    gid,
                    len(accepted_addrs) + 1,
                )
            del self._pending_joins[gid]
