"""Holon formation: the join proposal/answer handshake and the resolved
member set it produces.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_characteristic, topology_neighbors

from scare.base.model import (
    CommunityAssignment,
    HolonicAssignment,
    HolonicJoinAnswer,
    HolonicJoinRequest,
)
from scare.base.util.ids import deterministic_uuid

logger = logging.getLogger(__name__)


class HolonFormation:
    """Runs the holon join handshake (legacy ``holon``/``sector`` scopes) and
    holds the resolved membership; reads scope/topology through its owning
    role.
    """

    def __init__(self, role: Any) -> None:
        self._role = role
        # holon_id -> {sender_key: (addr, accept_or_None)}. Address kept to
        # build the resolved member list without a topology re-lookup.
        self.pending_proposals: dict[UUID, dict[str, tuple[Any, bool | None]]] = {}
        # Monotonic counter backing reproducible holon ids -- see base.util.ids.
        self.holon_seq = 0
        # Resolved holon membership (leader side), set by ``handle_join_answer``;
        # lets ``flex_expected`` scale with chunk size, not clique size.
        self.member_addrs: list[Any] = []
        self.member_keys: set[str] = set()

    async def try_form(self) -> None:
        role = self._role
        # Component scope elects a coordinator per connected component and never
        # uses the holon clique; skip formation entirely (the ``holons`` topology
        # is not even built in that scope).
        if role.admm_scope == "component":
            return
        if topology_characteristic(role, tid="groups") != "leader":
            return

        assignment = role.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is not None:
            return

        community = role.context.get_or_create_model(CommunityAssignment)
        if community.community_id is None:
            return

        try:
            neighbours = topology_neighbors(role, tid="holons")
        except Exception:
            return
        if not neighbours:
            return

        # Symmetry-breaking: only the lex-smallest aid initiates, else a clique
        # accepts competing requests and ends up leaderless. Rest wait.
        if any(addr.aid < role.context.aid for addr in neighbours):
            return

        candidates = neighbours[: role.max_holon_size - 1]
        self.holon_seq += 1
        holon_id = deterministic_uuid(role.context.aid, self.holon_seq)
        self.pending_proposals[holon_id] = {str(a): (a, None) for a in candidates}

        req = HolonicJoinRequest(
            holon_id=holon_id,
            member_communities=[community.community_id],
            level=1,
        )
        for addr in candidates:
            await role.context.send_message(req, receiver_addr=addr)

    async def handle_join_request(
        self, message: HolonicJoinRequest, meta: dict
    ) -> None:
        role = self._role
        assignment = role.context.get_or_create_model(HolonicAssignment)
        community = role.context.get_or_create_model(CommunityAssignment)
        accept = assignment.holon_id is None and community.community_id is not None

        if accept:
            assignment.holon_id = message.holon_id
            assignment.level = message.level
            assignment.parent_addr = mango_sender_addr(meta)
            role.context.update(assignment)
            logger.info(
                "[%s] joined holon %s at level %d",
                role.context.aid,
                message.holon_id,
                message.level,
            )

        reply = HolonicJoinAnswer(
            holon_id=message.holon_id,
            accept=accept,
            # Fallback for an unregistered community. Stable per agent: a fresh
            # id per reply would make the same community answer under a
            # different identity each round.
            community_id=community.community_id
            or deterministic_uuid(role.context.aid, "community"),
        )
        await role.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def handle_join_answer(self, message: HolonicJoinAnswer, meta: dict) -> None:
        role = self._role
        hid = message.holon_id
        if hid not in self.pending_proposals:
            return

        sender_key = str(mango_sender_addr(meta))
        existing = self.pending_proposals[hid].get(sender_key)
        addr = existing[0] if existing else mango_sender_addr(meta)
        self.pending_proposals[hid][sender_key] = (addr, message.accept)

        responses = self.pending_proposals[hid]
        if not all(ok is not None for _, ok in responses.values()):
            return

        accepted_addrs = [a for _, (a, ok) in responses.items() if ok]
        del self.pending_proposals[hid]

        if not accepted_addrs:
            # All rejected; retry now (a busy rejecter may have finished).
            role.context.schedule_instant_task(role._try_form_holon())
            return

        # Resolved acceptor set only (self/initiator tracked separately, +1 downstream).
        self.member_addrs = list(accepted_addrs)
        self.member_keys = {str(a) for a in accepted_addrs}

        community = role.context.get_or_create_model(CommunityAssignment)
        assignment = role.context.get_or_create_model(HolonicAssignment)
        assignment.holon_id = hid
        assignment.level = 1
        assignment.child_community_ids = (
            [community.community_id] if community.community_id else []
        )
        role.context.update(assignment)

        logger.info(
            "[%s] holon %s formed with %d member groups",
            role.context.aid,
            hid,
            len(accepted_addrs) + 1,
        )
        role._record_event("holon_formed", f"members={len(accepted_addrs) + 1}")
        # Rebalance now so ADMM gets a shot while the deficit is still present.
        role.context.schedule_instant_task(role._try_rebalance())

    def resolve_members(self) -> list[Any]:
        """Holon member addresses: formation-time list, else ``"holons"``
        topology neighbours ([] if unwired).
        """
        if self.member_addrs:
            return list(self.member_addrs)
        try:
            return topology_neighbors(self._role, tid="holons")
        except Exception:
            return []
