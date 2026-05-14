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
base group formation (``PreAssignedCommunityRole``).  After base
groups have formed, group leaders negotiate with neighbouring group
leaders to merge into holons.  The holon leader runs a DRO-based ADMM
optimisation to distribute surplus / deficit across member groups,
then instructs each group to rebalance internally using the
per-actor allocation as the gossip target.
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
    HebbianFlexBeacon,
    HolonicAssignment,
    HolonicJoinAnswer,
    HolonicJoinRequest,
    NegotiationFinishedEvent,
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


DEFAULT_MAX_HOLON_SIZE: int = 4


class HolonicCommunityRole(Role):
    """Manages holonic (super-community) formation and inter-group
    coordination using DRO-based ADMM optimisation.

    After base groups have been established (via
    ``PreAssignedCommunityRole``), group leaders periodically attempt
    to merge with neighbouring leaders into
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
        formation_period_s: float = 4.0,
        max_holon_size: int = DEFAULT_MAX_HOLON_SIZE,
        rebalance_period_s: float = 60.0,
        rebalance_min_gap_s: float = 2.0,
        flex_timeout_s: float = 5.0,
        enable_hebbian_formation: bool = True,
        hebbian_beacon_period_s: float = 4.0,
        hebbian_eta: float = 0.25,
        hebbian_threshold: float = 0.35,
        hebbian_warmup_s: float = 12.0,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.formation_period_s = formation_period_s
        self.max_holon_size = max_holon_size
        # ``rebalance_period_s`` is the *slow* background heartbeat — only
        # there to catch drift from timeseries inputs (load profiles,
        # supply temperatures) that change without firing any
        # NegotiationFinishedEvent.  Was 2 s while we were validating
        # the reactive path; now relaxed to 60 s because:
        #
        # * For the smoke / discrete-failure scenarios there is no
        #   driving input drift between failures, so the periodic
        #   contributes only "ADMM skipped: balanced" overhead.
        # * For long timeseries runs the slow heartbeat still picks up
        #   accumulated demand shifts that the reactive path (which
        #   fires only on Layer-1 NegotiationFinishedEvent / holon
        #   formation) would otherwise miss.
        #
        # ``rebalance_min_gap_s`` is the *fast* feedback-loop fuse —
        # holon ADMM sends ``StartBalanceNegotiation(override_target=…)``
        # to each member, which produces gossip → finished events →
        # would re-fire this role's ``_on_member_finished``.  Without
        # the gap the loop runs flat out.  2 s is enough to let one
        # full round of member gossip resolve before another ADMM
        # cycle can start, and decoupled from the slow heartbeat so
        # reactive triggers respond promptly to real events.
        self.rebalance_period_s = rebalance_period_s
        self.rebalance_min_gap_s = rebalance_min_gap_s
        self.flex_timeout_s = flex_timeout_s

        # B.2: Hebbian-emergent holon formation parameters.
        # The Hebbian path co-exists with the propose-accept path (above):
        # propose-accept gives a fast-bootstrap membership; Hebbian
        # asymptotically refines it based on observed flex co-variance.
        self.enable_hebbian_formation = enable_hebbian_formation
        self.hebbian_beacon_period_s = hebbian_beacon_period_s
        self.hebbian_eta = hebbian_eta              # learning rate
        self.hebbian_threshold = hebbian_threshold  # H_gh > τ → same cluster
        self.hebbian_warmup_s = hebbian_warmup_s    # cold-start grace

        # holon_id -> {sender_key: (addr, accept_or_None)}.  Storing the
        # address alongside the response so we can build the resolved
        # member list for flex collection without re-resolving from a
        # topology lookup.
        self._pending_proposals: dict[UUID, dict[str, tuple[Any, bool | None]]] = {}
        # Collected flex answers from member groups for inter-holon ADMM.
        # ``_flex_answer_senders`` stores the sender address per answer so
        # the holon leader can route the ADMM per-actor allocation back to
        # each member as an override balance target.
        self._flex_answers: list[AvailableFlexAnswer] = []
        self._flex_answer_senders: list[Any] = []
        self._flex_expected: int = 0
        self._rebalance_active: bool = False
        # Throttle for reactive ``_on_member_finished`` triggers so the
        # holon→member→finished→holon feedback loop is bounded.  Last
        # rebalance start time; new reactive triggers below this gap
        # are dropped.
        self._last_rebalance_t: float = float("-inf")

        # Resolved holon membership on the leader side.  Populated when
        # ``_handle_join_answer`` confirms acceptances; consulted by
        # ``_try_rebalance`` so flex requests target *only* holon members
        # (not every leader in the holons-topology neighbourhood).  This
        # makes ``_flex_expected`` scale with chunk size, not clique size.
        self._holon_member_addrs: list[Any] = []
        self._holon_member_keys: set[str] = set()

        # Throttling flags so the early-return paths in ``_try_rebalance``
        # (no holon yet, not leader, no neighbours) surface once at INFO
        # per leader instead of either (a) being silent at DEBUG or (b)
        # spamming on every periodic tick.
        self._logged_no_holon: bool = False
        self._logged_not_leader: bool = False
        self._logged_no_neighbours: bool = False

        # --- B.2: Hebbian co-variance state ---
        # Each leader keeps a per-peer running mean of (delta_g · delta_h)
        # and a sample count.  Once warmed up, holon membership is the
        # connected component of {peers : H_{gh} > threshold}.
        # addr_str -> (H_gh, samples)
        self._hebbian_H: dict[str, tuple[float, int]] = {}
        # Last own delta_g surrogate broadcast (for diagnostics + own-pair).
        self._last_own_delta_g: float = 0.0
        # Most recent flex aggregate per peer; used to gauge own delta_g
        # by comparison after each rebalance cycle.
        self._peer_last_delta: dict[str, float] = {}
        # Wallclock at which the warmup ended (set in ``setup``).
        self._hebbian_warmup_until: float = 0.0
        # Snapshot of the last Hebbian-emergent member set, used to
        # detect drift relative to the propose/accept membership.
        self._hebbian_members: set[str] = set()

    def setup(self) -> None:
        self.context.schedule_periodic_task(
            self._try_form_holon, delay=self.formation_period_s
        )
        self.context.schedule_periodic_task(
            self._try_rebalance, delay=self.rebalance_period_s
        )
        if self.enable_hebbian_formation:
            self.context.schedule_periodic_task(
                self._hebbian_beacon, delay=self.hebbian_beacon_period_s
            )
            self.context.schedule_periodic_task(
                self._hebbian_recluster,
                delay=max(2.0 * self.hebbian_beacon_period_s, self.formation_period_s),
            )
            self._hebbian_warmup_until = (
                self.context.current_timestamp + self.hebbian_warmup_s
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
        # Reactive Layer-2 trigger: when a member group finishes its
        # balance gossip (NegotiationFinishedEvent broadcast), kick
        # off an inter-group rebalance.  This is the natural moment
        # to redistribute: each member has just resolved its
        # intra-group imbalance, leaving a residual that ADMM can
        # spread across the holon.
        self.context.subscribe_message(
            self,
            _wrap(self._on_member_finished),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent)
            and msg.sector == self.sector,
        )
        if self.enable_hebbian_formation:
            self.context.subscribe_message(
                self,
                _wrap(self._handle_hebbian_beacon),
                lambda msg, meta: isinstance(msg, HebbianFlexBeacon)
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

        # Symmetry-breaking: in a clique of same-sector leaders, every
        # member would otherwise simultaneously initiate and end up
        # accepting each other's competing requests, leaving everyone
        # with parent_addr set and no leader to drive ADMM rebalancing.
        # Only the lexicographically smallest aid initiates; the rest
        # wait passively for the join request.
        if any(addr.aid < self.context.aid for addr in neighbours):
            return

        candidates = neighbours[: self.max_holon_size - 1]
        holon_id = uuid4()
        self._pending_proposals[holon_id] = {
            str(a): (a, None) for a in candidates
        }

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
        existing = self._pending_proposals[hid].get(sender_key)
        addr = existing[0] if existing else mango_sender_addr(meta)
        self._pending_proposals[hid][sender_key] = (addr, message.accept)

        responses = self._pending_proposals[hid]
        if not all(ok is not None for _, ok in responses.values()):
            return

        accepted_addrs = [a for _, (a, ok) in responses.items() if ok]
        del self._pending_proposals[hid]

        if not accepted_addrs:
            return

        # Record the resolved member set (initiator + acceptors) so
        # ``_try_rebalance`` can target only actual holon peers.
        self._holon_member_addrs = list(accepted_addrs)
        self._holon_member_keys = {str(a) for a in accepted_addrs}

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
            len(accepted_addrs) + 1,
        )
        from scare.base.diagnostics import record_event

        record_event(
            t=self.context.current_timestamp,
            kind="holon_formed",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=f"members={len(accepted_addrs) + 1}",
        )
        # Reactive Layer-2 trigger: rebalance immediately after the
        # holon has finished forming.  The periodic ``_try_rebalance``
        # loop alone would otherwise wait up to ``rebalance_period_s``
        # before firing, and at typical ``simulation_duration_s`` only
        # a few cycles fit anyway.  Starting eagerly here ensures the
        # ADMM gets at least one shot while the post-failure deficit
        # is still present.
        self.context.schedule_instant_task(self._try_rebalance())

    # ------------------------------------------------------------------
    # Inter-group coordination via DRO ADMM
    # ------------------------------------------------------------------

    async def _try_rebalance(self) -> None:
        """Holon leader collects flex from member groups and runs ADMM
        to redistribute resources.  Fired periodically (slow heartbeat
        for input drift) AND reactively (on holon formation, on member
        ``NegotiationFinishedEvent``).  See class docstring on
        ``rebalance_period_s`` vs ``rebalance_min_gap_s``.
        """
        # Fast-loop fuse: at most one rebalance start every
        # ``rebalance_min_gap_s`` (decoupled from the slow heartbeat).
        # Prevents the holon→member→finished→holon feedback loop where
        # each ADMM round triggers per-member balance which fires
        # ``NegotiationFinishedEvent``, re-arming the reactive path.
        now = self.context.current_timestamp
        if (now - self._last_rebalance_t) < self.rebalance_min_gap_s:
            return
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None:
            if not self._logged_no_holon:
                logger.info(
                    "[%s] holon rebalance idle: no holon assigned (sector=%s)",
                    self.context.aid,
                    self.sector.value,
                )
                self._logged_no_holon = True
            return
        if assignment.parent_addr is not None:
            if not self._logged_not_leader:
                logger.info(
                    "[%s] holon rebalance idle: not leader (sector=%s)",
                    self.context.aid,
                    self.sector.value,
                )
                self._logged_not_leader = True
            return
        if self._rebalance_active:
            logger.debug("[%s] rebalance skipped: active", self.context.aid)
            return

        # Prefer the resolved member list (populated when the holon was
        # formed); fall back to topology neighbours only if formation
        # didn't track member addresses (legacy path).  Targeting members
        # directly keeps ``_flex_expected`` proportional to chunk size,
        # not clique size, so the timeout fits inside the simulation.
        if self._holon_member_addrs:
            members = list(self._holon_member_addrs)
        else:
            try:
                members = topology_neighbors(self, tid="holons")
            except Exception:
                return
        if not members:
            if not self._logged_no_neighbours:
                logger.info(
                    "[%s] holon rebalance idle: no neighbours (sector=%s)",
                    self.context.aid,
                    self.sector.value,
                )
                self._logged_no_neighbours = True
            return

        self._rebalance_active = True
        self._last_rebalance_t = self.context.current_timestamp
        self._flex_answers = []
        self._flex_answer_senders = []
        # The leader contributes its own group's flex too — without that
        # ADMM would be a single-actor problem and bail out early.
        self._flex_expected = len(members) + 1

        from scare.base.model import AskForAvailableFlex

        logger.debug(
            "[%s] holon rebalance: asking %d members (+self) for flex",
            self.context.aid,
            len(members),
        )
        msg = AskForAvailableFlex(include_connectors=False)
        await self.context.send_message(msg, receiver_addr=self.context.addr)
        for addr in members:
            await self.context.send_message(msg, receiver_addr=addr)

        # Schedule a timeout: if not all answers arrive within the
        # deadline, run ADMM with whatever we have (≥2 answers) or
        # release the lock so the next cycle can retry.
        # Adaptive: base per sector + per-member scaling.
        base = _FLEX_TIMEOUT_BASE_S.get(self.sector, _FLEX_TIMEOUT_DEFAULT_S)
        timeout = base + len(members) * _FLEX_TIMEOUT_PER_MEMBER_S
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

    async def _on_member_finished(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        """A member group finished its balance gossip — try an
        inter-group rebalance now to spread the post-gossip residual.

        Throttled by ``rebalance_min_gap_s`` (the fast feedback-loop
        fuse) — not by the slow periodic heartbeat — so reactive
        triggers respond promptly to genuine events while still
        breaking the holon→member→finished→holon feedback loop.
        ``_try_rebalance`` also self-gates on holon membership /
        leader status / not-currently-rebalancing.
        """
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return
        if self._rebalance_active:
            return
        now = self.context.current_timestamp
        if now - self._last_rebalance_t < self.rebalance_min_gap_s:
            return
        self.context.schedule_instant_task(self._try_rebalance())

    async def _handle_flex_answer(
        self, message: AvailableFlexAnswer, meta: dict
    ) -> None:
        # B.2: regardless of rebalance state, record the peer's signed
        # delta surrogate for the Hebbian co-variance matrix.
        if self.enable_hebbian_formation:
            sender_key = str(mango_sender_addr(meta))
            self._peer_last_delta[sender_key] = self._delta_g_from_flex(message)

        if not self._rebalance_active:
            return
        # Defensive: ignore answers from non-members.  ``_handle_ask_flex``
        # answers any sender, so if a stray flex request reaches a leader
        # outside the holon its reply could otherwise inflate the count.
        if self._holon_member_keys:
            sender_key = str(mango_sender_addr(meta))
            if (
                sender_key != str(self.context.addr)
                and sender_key not in self._holon_member_keys
            ):
                return
        sender = mango_sender_addr(meta)
        self._flex_answers.append(message)
        self._flex_answer_senders.append(sender)

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
        senders = self._flex_answer_senders[:]
        self._flex_answers = []
        self._flex_answer_senders = []
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

        if not np.all(np.isfinite(total_T)):
            logger.warning(
                "[%s] holon ADMM skipped: non-finite target T=%s",
                self.context.aid,
                total_T.tolist(),
            )
            self._rebalance_active = False
            return

        if np.all(np.abs(total_T) < 1e-6):
            logger.info(
                "[%s] holon ADMM skipped: balanced (sectors=%s)",
                self.context.aid,
                all_sectors,
            )
            self._rebalance_active = False
            return

        # --- Priority-weighted S computation ---
        # Compute waterfall shares: how much each group *should* receive
        # if resources were allocated strictly by priority ordering.
        #
        # The waterfall budget is the total amount of resources that the
        # holon can redistribute: excess generation (groups with negative
        # balance, i.e. surplus) plus flex headroom across all groups.
        # The earlier formulation used excess generation alone, which is
        # zero in a pure-deficit holon (every group positive imbalance)
        # — exactly the regime where priority discrimination matters
        # most.  Including flex headroom keeps the budget positive and
        # the relative per-tier shares meaningful even when no group has
        # surplus to donate.
        total_surplus = sum(max(0.0, -a.balance) for a in answers)
        total_flex = sum(max(0.0, a.flex) for a in answers)
        total_available = total_surplus + total_flex
        priority_shares = compute_priority_weighted_shares(
            [a.demand_by_priority for a in answers],
            [a.served_by_priority for a in answers],
            total_available,
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
            lb = np.nan_to_num(lb, nan=0.0, posinf=0.0, neginf=0.0)
            ub = np.nan_to_num(ub, nan=1e-6, posinf=1e6, neginf=1e-6)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            actors.append(ADMMFlexActor(lb=lb, u=ub, C=C, d=d, S=S))

        coordinator = create_sharing_target_distance_admm_coordinator()
        # Tight iter cap so concurrent holon ADMMs across sectors don't
        # block discrete-time progress: 1000 is the package default and
        # is wall-time prohibitive when several leaders rebalance per
        # simulation step.
        coordinator.max_iters = 50
        start_msg = create_admm_start(create_admm_sharing_data(total_T.tolist()))

        from scare.base.diagnostics import record_event

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
            record_event(
                t=self.context.current_timestamp,
                kind="holon_admm_result",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=f"sectors={all_sectors} T={total_T.tolist()}",
            )
        except Exception as exc:
            logger.error("[%s] holon ADMM failed: %s", self.context.aid, exc)
            record_event(
                t=self.context.current_timestamp,
                kind="holon_admm_failed",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=str(exc),
            )
            # Fallback: still trigger intra-group gossip so member groups
            # can rebalance locally even without inter-group redistribution.

        # Trigger intra-group rebalancing in all member groups, routing
        # the ADMM per-actor allocation as the override target so the
        # gossip target reflects the cross-sector optimisation instead
        # of being recomputed locally (which would discard the ADMM
        # result entirely).  Each ``actors[idx].x`` is a vector across
        # ``all_sectors``; we pick the entry matching the member's own
        # sector and pass it as the *target* for that member's gossip
        # (negation: target = -allocation, since the member must absorb
        # ``allocation`` worth of imbalance from the holon's pool).
        #
        # Members whose addresses we couldn't map to an ADMM actor fall
        # back to the historical empty-trigger path (recompute locally).
        sender_to_actor: dict[str, tuple[Any, AvailableFlexAnswer]] = {}
        for sender, answer, actor in zip(senders, answers, actors):
            sender_to_actor[str(sender)] = (actor, answer)

        if self._holon_member_addrs:
            triggers = list(self._holon_member_addrs)
        else:
            try:
                triggers = topology_neighbors(self, tid="holons")
            except Exception:
                triggers = []
        for addr in triggers:
            entry = sender_to_actor.get(str(addr))
            override: float | None = None
            if entry is not None:
                actor_obj, answer = entry
                # ADMM solved successfully → actor_obj.x is populated.
                # Index by the member's natural sector.
                try:
                    x_vec = list(actor_obj.x)
                    if answer.sector.value in sector_idx:
                        override = -float(x_vec[sector_idx[answer.sector.value]])
                    elif x_vec:
                        override = -float(x_vec[0])
                except Exception:  # pragma: no cover - defensive
                    override = None
            await self.context.send_message(
                StartBalanceNegotiation(override_target=override),
                receiver_addr=addr,
            )

    # ------------------------------------------------------------------
    # B.2: Hebbian-emergent holon formation
    # ------------------------------------------------------------------
    #
    # Each leader broadcasts a periodic ``HebbianFlexBeacon`` carrying a
    # scalar delta_g surrogate (signed sector imbalance / capacity), and
    # updates a per-peer Hebbian co-variance estimate from incoming
    # beacons.  After warmup, holon membership is the connected
    # component of {peers : H_{gh} > τ}.  This is *additive* to the
    # propose-accept path: the latter establishes a fast bootstrap
    # membership, the former asymptotically refines it based on
    # observed dynamics.
    #
    # The update rule is the classical exponentially-weighted Hebbian
    # rule (Aoki & Aoyagi 2009):
    #     H_{gh}(t+1) = (1 - eta) * H_{gh}(t) + eta * delta_g * delta_h
    # The threshold τ is in the same scaled units (signed share of
    # capacity), so cross-pair comparison is meaningful.
    # ------------------------------------------------------------------

    def _delta_g_from_flex(self, answer: AvailableFlexAnswer) -> float:
        """Extract a signed scalar surrogate of the leader's group state.

        Uses balance / max(|flex| + |balance|, ε) so that the value
        sits in roughly [-1, 1]: positive ⇒ unmet demand, negative ⇒
        surplus generation.  Group capacity normalises out of the
        product H = δ_g · δ_h, so two leaders that are both
        proportionally stressed pair up regardless of their absolute
        size.
        """
        bal = float(answer.balance)
        flex = float(answer.flex)
        denom = max(abs(bal) + abs(flex), 1e-9)
        return bal / denom

    async def _hebbian_beacon(self) -> None:
        """Periodic broadcast of own δ_g to same-sector neighbours."""
        if not self.enable_hebbian_formation:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Compute own delta_g from the latest known flex (if any) or
        # simply 0 during warmup before any rebalance has run.
        own_key = str(self.context.addr)
        delta_g = self._peer_last_delta.get(own_key, self._last_own_delta_g)
        self._last_own_delta_g = delta_g

        try:
            neighbours = topology_neighbors(self, tid="holons")
        except Exception:
            return
        if not neighbours:
            return
        msg = HebbianFlexBeacon(
            sector=self.sector,
            delta_g=float(delta_g),
            timestamp=float(self.context.current_timestamp),
        )
        for addr in neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_hebbian_beacon(
        self, message: HebbianFlexBeacon, meta: dict
    ) -> None:
        """Update the per-peer Hebbian co-variance estimate."""
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        sender_key = str(sender)
        # Pair the sender's delta with our most recent own delta_g so
        # H_{gh} reflects co-variance, not just the sender's signal.
        own = self._last_own_delta_g
        product = float(message.delta_g) * float(own)
        prev_h, prev_n = self._hebbian_H.get(sender_key, (0.0, 0))
        new_h = (1.0 - self.hebbian_eta) * prev_h + self.hebbian_eta * product
        self._hebbian_H[sender_key] = (new_h, prev_n + 1)

    def hebbian_membership_candidates(self) -> set[str]:
        """Return the set of peer-keys whose H_{gh} exceeds the threshold.

        Public so the diagnostics layer (and the ``C.5`` cluster-sync
        analysis script) can compare static topology membership against
        the dynamically-emergent partition.
        """
        return {
            k for k, (h, _n) in self._hebbian_H.items()
            if h > self.hebbian_threshold
        }

    def _hebbian_recluster(self) -> None:
        """Apply the thresholded Hebbian co-variance to refine the
        leader's holon membership after warmup.

        Only the leader (no parent_addr) reclusters; non-leader peers
        accept whatever membership their leader announces via the
        existing propose/accept channel.  Reclustering modifies
        ``_holon_member_addrs`` / ``_holon_member_keys`` in place so
        the next ``_try_rebalance`` cycle picks up the refined set.
        """
        if not self.enable_hebbian_formation:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        if self.context.current_timestamp < self._hebbian_warmup_until:
            return
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return  # not a leader of a formed holon

        candidates = self.hebbian_membership_candidates()
        if not candidates:
            return

        # Resolve candidate keys back to addresses via the topology.
        try:
            holon_neighbours = topology_neighbors(self, tid="holons")
        except Exception:
            return
        addr_by_key = {str(a): a for a in holon_neighbours}

        new_addrs: list[Any] = []
        new_keys: set[str] = set()
        for key in candidates:
            if key in addr_by_key and len(new_addrs) < self.max_holon_size - 1:
                new_addrs.append(addr_by_key[key])
                new_keys.add(key)

        if not new_addrs:
            return

        prev_keys = self._holon_member_keys
        if new_keys == prev_keys:
            return  # no drift

        self._holon_member_addrs = new_addrs
        self._holon_member_keys = new_keys
        self._hebbian_members = new_keys
        logger.info(
            "[%s] Hebbian re-cluster: holon membership updated to %d peers (sector=%s)",
            self.context.aid,
            len(new_keys),
            self.sector.value,
        )
        from scare.base.diagnostics import record_event

        record_event(
            t=self.context.current_timestamp,
            kind="hebbian_recluster",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=f"members={len(new_keys)} drift={len(new_keys ^ prev_keys)}",
        )
