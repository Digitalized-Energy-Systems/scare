"""Holonic (multi-level) community formation and coordination.

Implements the "two-layer local-information architecture" and
"hierarchical clustering" recommendations from improvements.txt §7:

  Layer 1 — Sector agents solve local restoration within their own
            energy network using only local measurements and neighbour
            communication (gossip-based balance negotiation).
  Layer 2 — Holon leaders aggregate flex information from member groups
            and use ADMM (via distributed-resource-optimization) for
            inter-group resource sharing.

This module adds a *super-community* (holon) layer on top of the
existing ``CHSRole`` group formation.  After base groups have formed,
group leaders negotiate with neighbouring group leaders to merge into
holons.  The holon leader runs a DRO-based ADMM optimisation to
distribute surplus / deficit across member groups, then instructs each
group to rebalance internally.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_characteristic, topology_neighbors

from scare.base.model import (
    AvailableFlexAnswer,
    CommunityAssignment,
    HolonicAssignment,
    HolonicJoinAnswer,
    HolonicJoinRequest,
    Sector,
    StartBalanceNegotiation,
)

logger = logging.getLogger(__name__)

# Base flex-collection timeout per sector.  Heat networks respond slowly
# (thermal inertia), so group leaders need more time to assemble their
# flex reports.  Electricity is fast; gas sits in between.
_FLEX_TIMEOUT_BASE_S: dict[Sector, float] = {
    Sector.ELECTRICITY: 3.0,
    Sector.GAS: 8.0,
    Sector.HEAT: 15.0,
}
_FLEX_TIMEOUT_DEFAULT_S = 5.0
_FLEX_TIMEOUT_PER_MEMBER_S = 0.5  # added per expected member group


class HolonicCommunityRole(Role):
    """Manages holonic (super-community) formation and inter-group
    coordination using DRO-based ADMM optimisation.

    After base groups have been established (via ``CHSRole``), group
    leaders periodically attempt to merge with neighbouring leaders into
    a *holon*.  The holon leader:

    1. Collects ``AvailableFlexAnswer`` from member group leaders.
    2. Runs an ADMM sharing optimisation (via ``distributed-resource-
       optimization``) to compute optimal resource redistribution
       across groups.
    3. Triggers intra-group rebalancing in each member group.

    This role is attached to every agent that participates in the group
    topology, but only group leaders actively initiate holon formation
    and inter-group coordination.
    """

    def __init__(
        self,
        sector: Sector,
        *,
        formation_period_s: float = 8.0,
        max_holon_size: int = 4,
        rebalance_period_s: float = 10.0,
        flex_timeout_s: float = 5.0,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.formation_period_s = formation_period_s
        self.max_holon_size = max_holon_size
        self.rebalance_period_s = rebalance_period_s
        self.flex_timeout_s = flex_timeout_s

        self._pending_proposals: dict[UUID, dict[str, bool | None]] = {}
        # Collected flex answers from member groups for inter-holon ADMM
        self._flex_answers: list[AvailableFlexAnswer] = []
        self._flex_expected: int = 0
        self._rebalance_active: bool = False

    def setup(self) -> None:
        self.context.schedule_periodic_task(
            self._try_form_holon, delay=self.formation_period_s
        )
        self.context.schedule_periodic_task(
            self._try_rebalance, delay=self.rebalance_period_s
        )

        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        self.context.subscribe_message(
            self,
            _wrap(self._handle_join_request),
            lambda msg, meta: isinstance(msg, HolonicJoinRequest),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_join_answer),
            lambda msg, meta: isinstance(msg, HolonicJoinAnswer),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_flex_answer),
            lambda msg, meta: isinstance(msg, AvailableFlexAnswer)
            and msg.sector == self.sector,
        )

    # ------------------------------------------------------------------
    # Holon formation
    # ------------------------------------------------------------------

    async def _try_form_holon(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return

        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is not None:
            return

        community = self.context.get_or_create_model(CommunityAssignment)
        if community.community_id is None:
            return

        try:
            neighbours = topology_neighbors(self, tid="holons")
        except Exception:
            return
        if not neighbours:
            return

        candidates = neighbours[: self.max_holon_size - 1]
        holon_id = uuid4()
        self._pending_proposals[holon_id] = {str(a): None for a in candidates}

        req = HolonicJoinRequest(
            holon_id=holon_id,
            member_communities=[community.community_id],
            level=1,
        )
        for addr in candidates:
            await self.context.send_message(req, receiver_addr=addr)

    async def _handle_join_request(
        self, message: HolonicJoinRequest, meta: dict
    ) -> None:
        assignment = self.context.get_or_create_model(HolonicAssignment)
        community = self.context.get_or_create_model(CommunityAssignment)
        accept = (
            assignment.holon_id is None and community.community_id is not None
        )

        if accept:
            assignment.holon_id = message.holon_id
            assignment.level = message.level
            assignment.parent_addr = mango_sender_addr(meta)
            self.context.update(assignment)
            logger.info(
                "[%s] joined holon %s at level %d",
                self.context.aid,
                message.holon_id,
                message.level,
            )

        reply = HolonicJoinAnswer(
            holon_id=message.holon_id,
            accept=accept,
            community_id=community.community_id or uuid4(),
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_join_answer(
        self, message: HolonicJoinAnswer, meta: dict
    ) -> None:
        hid = message.holon_id
        if hid not in self._pending_proposals:
            return

        sender_key = str(mango_sender_addr(meta))
        self._pending_proposals[hid][sender_key] = message.accept

        responses = self._pending_proposals[hid]
        if not all(v is not None for v in responses.values()):
            return

        accepted = [addr for addr, ok in responses.items() if ok]
        del self._pending_proposals[hid]

        if not accepted:
            return

        community = self.context.get_or_create_model(CommunityAssignment)
        assignment = self.context.get_or_create_model(HolonicAssignment)
        assignment.holon_id = hid
        assignment.level = 1
        assignment.child_community_ids = (
            [community.community_id] if community.community_id else []
        )
        self.context.update(assignment)

        logger.info(
            "[%s] holon %s formed with %d member groups",
            self.context.aid,
            hid,
            len(accepted) + 1,
        )

    # ------------------------------------------------------------------
    # Inter-group coordination via DRO ADMM
    # ------------------------------------------------------------------

    async def _try_rebalance(self) -> None:
        """Holon leader periodically collects flex from member groups
        and runs ADMM to optimally redistribute resources."""
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None:
            return
        if assignment.parent_addr is not None:
            return  # not the holon leader
        if self._rebalance_active:
            return

        try:
            neighbours = topology_neighbors(self, tid="holons")
        except Exception:
            return
        if not neighbours:
            return

        self._rebalance_active = True
        self._flex_answers = []
        self._flex_expected = len(neighbours)

        from scare.base.model import AskForAvailableFlex

        msg = AskForAvailableFlex(include_connectors=False)
        for addr in neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

        # Schedule a timeout: if not all answers arrive within the
        # deadline, run ADMM with whatever we have (≥2 answers) or
        # release the lock so the next cycle can retry.
        # Adaptive: base per sector + per-member scaling.
        base = _FLEX_TIMEOUT_BASE_S.get(self.sector, _FLEX_TIMEOUT_DEFAULT_S)
        timeout = base + len(neighbours) * _FLEX_TIMEOUT_PER_MEMBER_S
        deadline = self.context.current_timestamp + timeout
        self.context.schedule_timestamp_task(
            self._flex_collection_timeout(), timestamp=deadline
        )

    async def _flex_collection_timeout(self) -> None:
        if not self._rebalance_active:
            return
        received = len(self._flex_answers)
        logger.warning(
            "[%s] holon flex timeout: received %d/%d answers",
            self.context.aid,
            received,
            self._flex_expected,
        )
        if received >= 2:
            await self._run_inter_group_admm()
        else:
            self._rebalance_active = False

    async def _handle_flex_answer(
        self, message: AvailableFlexAnswer, meta: dict
    ) -> None:
        if not self._rebalance_active:
            return
        self._flex_answers.append(message)

        if len(self._flex_answers) >= self._flex_expected:
            await self._run_inter_group_admm()

    async def _run_inter_group_admm(self) -> None:
        """Run ADMM sharing optimisation across member groups.

        Each group's flex is modelled as a multi-dimensional ADMM actor
        with one dimension per energy sector present across all member
        groups.  This allows the optimisation to balance resources across
        sectors simultaneously (e.g. shifting gas surplus to cover heat
        deficit via CHP coupling).

        The sharing target ``T`` is a vector with one entry per sector,
        equal to the total balance (imbalance) in that sector across all
        member groups.

        **Priority-aware allocation:** Each group reports its demand
        broken down by priority tier.  A waterfall allocation computes
        each group's ideal share of the available resources, serving
        high-priority tiers across all groups before any low-priority
        tier.  The ADMM ``S`` parameter (linear cost) is set to pull
        each actor's allocation toward its priority-weighted share.
        """
        from distributed_resource_optimization import (
            create_admm_sharing_data,
            create_admm_start,
            create_sharing_target_distance_admm_coordinator,
            start_coordinated_optimization,
        )
        from distributed_resource_optimization.algorithm.admm.flex_actor import (
            ADMMFlexActor,
        )

        from scare.base.util import compute_priority_weighted_shares

        if not self._rebalance_active:
            return
        answers = self._flex_answers[:]
        self._flex_answers = []
        self._flex_expected = 0
        self._rebalance_active = False  # release lock early to prevent timeout re-entry

        if len(answers) < 2:
            return

        # Determine the union of sectors across all member groups.
        all_sectors: list[str] = sorted(
            {s for a in answers for s in a.balance_by_sector}
        )
        n_dims = len(all_sectors) if all_sectors else 1

        # Fall back to 1D if no per-sector data is available.
        if n_dims <= 1 and not any(a.balance_by_sector for a in answers):
            all_sectors = [self.sector.value]
            n_dims = 1

        sector_idx = {s: i for i, s in enumerate(all_sectors)}

        actors: list[ADMMFlexActor] = []
        total_T = np.zeros(n_dims)

        # Collect per-group bounds first, then compute priority shares.
        group_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for answer in answers:
            lb = np.zeros(n_dims)
            ub = np.full(n_dims, 1e-6)

            if answer.balance_by_sector:
                for sec, bal in answer.balance_by_sector.items():
                    if sec in sector_idx:
                        i = sector_idx[sec]
                        lb[i] = -abs(bal)
                        flex_val = answer.flex_by_sector.get(sec, 0.0)
                        ub[i] = max(abs(flex_val), 1e-6)
                        total_T[i] += bal
            else:
                lb[0] = -abs(answer.balance)
                ub[0] = max(abs(answer.flex), 1e-6)
                total_T[0] += answer.balance

            group_bounds.append((lb, ub))

        if np.all(np.abs(total_T) < 1e-6):
            logger.debug("[%s] holon ADMM skipped: balanced", self.context.aid)
            self._rebalance_active = False
            return

        # --- Priority-weighted S computation ---
        # Compute waterfall shares: how much each group *should* receive
        # if resources were allocated strictly by priority ordering.
        total_generation = sum(
            max(0.0, -a.balance) for a in answers
        )
        priority_shares = compute_priority_weighted_shares(
            [a.demand_by_priority for a in answers],
            [a.served_by_priority for a in answers],
            total_generation,
        )

        for idx, answer in enumerate(answers):
            lb, ub = group_bounds[idx]
            C = np.zeros((0, n_dims))
            d = np.zeros(0)
            # S is a linear cost in the local QP: negative values attract
            # allocation.  Set S proportional to the group's priority-
            # weighted share so ADMM steers resources toward groups with
            # high-priority unserved demand.
            S = np.zeros(n_dims)
            if priority_shares[idx] > 1e-9:
                # Scale S so groups with larger priority shares get
                # stronger negative pull.  The magnitude is normalised
                # relative to the total target to keep the penalty
                # numerically stable alongside the ADMM rho term.
                # Distribute the pull across all dimensions proportional
                # to the group's balance in each dimension.
                t_mag = max(np.max(np.abs(total_T)), 1e-6)
                pull = priority_shares[idx] / t_mag
                if answer.balance_by_sector and n_dims > 1:
                    bal_vec = np.zeros(n_dims)
                    for sec, bal in answer.balance_by_sector.items():
                        if sec in sector_idx:
                            bal_vec[sector_idx[sec]] = abs(bal)
                    bal_sum = np.sum(bal_vec)
                    if bal_sum > 1e-9:
                        S = -(pull * bal_vec / bal_sum)
                    else:
                        S = np.full(n_dims, -pull / n_dims)
                else:
                    S[0] = -pull
            actors.append(ADMMFlexActor(lb=lb, u=ub, C=C, d=d, S=S))

        coordinator = create_sharing_target_distance_admm_coordinator()
        start_msg = create_admm_start(create_admm_sharing_data(total_T.tolist()))

        try:
            await start_coordinated_optimization(actors, coordinator, start_msg)
            results = [a.x.tolist() for a in actors]
            logger.info(
                "[%s] holon ADMM result (sectors=%s): %s (T=%s)",
                self.context.aid,
                all_sectors,
                results,
                total_T.tolist(),
            )
        except Exception as exc:
            logger.error("[%s] holon ADMM failed: %s", self.context.aid, exc)
            # Fallback: still trigger intra-group gossip so member groups
            # can rebalance locally even without inter-group redistribution.

        # Trigger intra-group rebalancing in all member groups (both on
        # success and on failure — groups should always get a chance to
        # re-negotiate after a holon-level event).
        try:
            neighbours = topology_neighbors(self, tid="holons")
        except Exception:
            neighbours = []
        for addr in neighbours:
            await self.context.send_message(StartBalanceNegotiation(), receiver_addr=addr)
