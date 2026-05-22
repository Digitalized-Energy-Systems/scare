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
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
    topology_neighbors,
)

from scare.base.channel import (
    CPSetpoint,
    HolonAllocation,
    MonotonicVersion,
    SeenVersions,
)
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
from scare.base.topology_mirror import LivePeerFilter

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
        admm_max_iters: int = 50,
        admm_abs_tol: float = 1e-3,
        enable_tier_stratified_admm: bool = True,
        priority_tiers: int = 10,
        admm_mode: str = "demand",
        enable_priority_allocation: bool = True,
        live_member_filter: LivePeerFilter | None = None,
        coalition_constraint_store: Any = None,
        my_node_id: Any = None,
        leader_node_ids: dict[str, Any] | None = None,
        topology_mirror: Any = None,
    ) -> None:
        super().__init__()
        self.sector = sector
        # Shared constraint store the sibling :class:`HolonSummaryRole`
        # writes into when a coalition has been formed.  Read here on
        # every supply-priority dispatch so coalition fractions
        # override L2's per-tier ADMM result for the TTL window.  None
        # ⇒ no merge — L2 keeps its pre-M2 last-write-wins behaviour.
        self._coalition_constraint_store = coalition_constraint_store
        # Optional sibling role (``DynamicHolonRole``) that classifies
        # which holon-member addresses are still physically reachable
        # via live grid edges.  When ``None`` the role operates in
        # static-topology mode — every member listed in
        # ``_holon_member_addrs`` is treated as reachable, matching the
        # pre-Concept-C behaviour.  Consulted by ``_live_members``.
        self._live_member_filter = live_member_filter
        self.formation_period_s = formation_period_s
        self.max_holon_size = max_holon_size
        # ADMM convergence knobs (configurable via RestorationConfiguration
        # so campaigns can trade quality against wallclock cost).
        self.admm_max_iters = admm_max_iters
        self.admm_abs_tol = admm_abs_tol
        # Tier-stratified ADMM (Package C) — see _run_tier_stratified_admm.
        self.enable_tier_stratified_admm = enable_tier_stratified_admm
        self.priority_tiers = priority_tiers
        # Holon ADMM mode: "demand" (Package C) or "supply" (Route A).
        if admm_mode not in {"demand", "supply"}:
            raise ValueError(
                f"holon admm_mode must be 'demand' or 'supply', got {admm_mode!r}"
            )
        self.admm_mode = admm_mode
        # Priority-weighted allocation switch.  When False, both the
        # legacy per-sector ADMM and the supply-priority ADMM use
        # uniform per-tier weights — equivalent to the no-priority
        # ablation.  Wired from ``RestorationConfiguration``'s
        # ``enable_priority_holon_allocation`` so the eval-campaign
        # ablation toggle actually changes behaviour (previously the
        # flag was unread and the ablation column was a no-op).
        self.enable_priority_allocation = bool(enable_priority_allocation)
        # Deliverability wiring (F6).  When all three are present, the
        # supply-priority ADMM passes ``actor_ub_overrides`` that cap
        # each member's per-tier supply commitment at the sum of tier-t
        # demand reachable from that member's home node — preventing
        # the L2 ADMM from allocating supply that the LP cannot route
        # after a partition.  Mirrors what ``HolonSummaryRole`` already
        # does for the L2.5 coalition path; without this wiring an L2
        # round can shed lower-tier loads to "fund" an undeliverable
        # higher-tier commitment that the LP then rounds to zero, with
        # the shed staying in effect for ≥ rebalance_period_s seconds.
        self._my_node_id = my_node_id
        self._leader_node_ids: dict[str, Any] = dict(leader_node_ids or {})
        self._topology_mirror = topology_mirror
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

        # --- Channel-pattern state (L2 <-> L3 direct link) ---
        # ``_version`` advances on every ``HolonAllocation`` we publish;
        # ``_seen_cps`` tracks the latest ``CPSetpoint`` version we have
        # consumed per publisher so the predicate doesn't re-fire on
        # stale data.
        self._version = MonotonicVersion()
        self._seen_cps = SeenVersions()
        self._cp_setpoint_state: dict[tuple[str, str], float] = {}
        self._last_cp_predicate_fire_t: float = -1e9

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
        #
        # Two subscriptions are needed.  ``subscribe_message`` catches
        # broadcasts from *other* group leaders arriving over the
        # holons topology (see balance.py ``_finish_negotiation``).
        # ``subscribe_event`` catches the local-emit case where this
        # agent's own ``EnergyBalanceNegotiator`` just finished its
        # gossip (the same-agent emit_event/send_message buses are
        # disjoint, so the message subscription alone misses the case
        # where the holon leader IS the gossip originator).
        self.context.subscribe_message(
            self,
            _wrap(self._on_member_finished),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent)
            and msg.sector == self.sector,
        )
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_member_finished_local
        )
        if self.enable_hebbian_formation:
            self.context.subscribe_message(
                self,
                _wrap(self._handle_hebbian_beacon),
                lambda msg, meta: isinstance(msg, HebbianFlexBeacon)
                and msg.sector == self.sector,
            )
        # Direct L3 -> L2 trigger (channel/decision pattern).  When a CP
        # plant commits a cross-sector setpoint, the affected holons
        # should re-evaluate without waiting for the L1 gossip chain to
        # propagate the shift.  Predicate inside the handler decides
        # whether the change is large enough to merit a rebalance.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_cp_setpoint),
            lambda msg, meta: isinstance(msg, CPSetpoint),
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

    def _live_members(self, members: list[Any]) -> list[Any]:
        """Return the subset of ``members`` that the
        :class:`LivePeerFilter` considers reachable.

        Pure passthrough when no filter is wired (legacy / static-
        topology mode).  Centralised here so every callsite that
        iterates holon peers picks up Concept-C dynamics uniformly.
        """
        if self._live_member_filter is None:
            return members
        kept: list[Any] = []
        dropped: list[Any] = []
        for m in members:
            if self._live_member_filter.is_live(m):
                kept.append(m)
            else:
                dropped.append(m)
        if dropped and logger.isEnabledFor(logging.DEBUG):
            # ``self.context`` may be ``None`` in unit-test construction;
            # only resolve aid for logging when the role is attached.
            ctx = getattr(self, "context", None)
            logger.debug(
                "[%s] holon filter dropped %d unreachable members (kept=%d)",
                getattr(ctx, "aid", "<detached>"), len(dropped), len(kept),
            )
        return kept

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
        # Concept C — Layer 2 dynamic topology.  Filter out members that
        # the sibling ``DynamicHolonRole`` has classified as physically
        # unreachable.  When no filter is installed, ``_live_members``
        # returns the input list unchanged so legacy behaviour stands.
        members = self._live_members(members)
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

    # Predicate dead-band on the magnitude of cross-sector flow shift
    # in a single CP commit before the holon bothers to rebalance.
    # Smaller than the holon's intrinsic ``admm_abs_tol`` so we don't
    # mask real opportunities, larger than CP regulation noise.
    _CP_PREDICATE_DEAD_BAND_MW: float = 1e-3
    _CP_PREDICATE_MIN_GAP_S: float = 1.0

    async def _handle_cp_setpoint(
        self, message: CPSetpoint, meta: dict
    ) -> None:
        """Direct L3 -> L2 trigger via the channel/decision pattern.

        Updates the per-publisher CP-setpoint memory, version-tracks
        to skip stale repeats, and (if the predicate accepts) calls
        ``_maybe_schedule_rebalance`` to re-run holon ADMM with the
        new cross-sector reality baked into the next round's flex
        collection.
        """
        # Only the holon leader bothers; other members would just
        # double-trigger.
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return
        # Echo guard: a CP commit caused by our own most-recent
        # HolonAllocation isn't fresh news.
        if (
            message.caused_by.get(str(self.context.aid), -1)
            == self._version.current
            and self._version.current > 0
        ):
            return
        if not self._seen_cps.is_fresh(message.publisher, message.version):
            return

        # Track the CP's per-sector flow for our sector specifically.
        flow = float(message.sector_flows_mw.get(self.sector.value, 0.0))
        key = (message.publisher, self.sector.value)
        prev_flow = self._cp_setpoint_state.get(key, 0.0)
        self._cp_setpoint_state[key] = flow
        self._seen_cps.mark(message.publisher, message.version)

        delta = abs(flow - prev_flow)
        if delta < self._CP_PREDICATE_DEAD_BAND_MW:
            return

        now = float(self.context.current_timestamp)
        if now - self._last_cp_predicate_fire_t < self._CP_PREDICATE_MIN_GAP_S:
            return
        self._last_cp_predicate_fire_t = now

        logger.info(
            "[%s] holon predicate fired: sector=%s cp=%s Δflow=%.4f (cause)",
            self.context.aid, self.sector.value, message.publisher, delta,
        )
        self._maybe_schedule_rebalance()

    def _maybe_schedule_rebalance(self) -> None:
        """Shared throttle + gate logic for both the message- and
        event-driven reactive paths.  Returns silently if any gate
        rejects (not in a holon, not the holon leader, rebalance
        already running, or within the ``rebalance_min_gap_s`` fuse
        window).  ``_try_rebalance`` itself does its own holon-
        membership and leader checks too, so this method is a fast
        pre-filter that avoids scheduling a no-op task.
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

    async def _on_member_finished(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        """A holon-peer's group finished its balance gossip — try an
        inter-group rebalance now to spread the post-gossip residual.

        This is the *message* path: the finishing group's leader
        broadcasts ``NegotiationFinishedEvent`` over the ``holons``
        topology (see balance.py ``_finish_negotiation``), reaching
        every same-sector holon peer.  The peer's holon leader (i.e.
        ``parent_addr is None``) then schedules ADMM.

        Throttled by ``rebalance_min_gap_s`` (the fast feedback-loop
        fuse) — not by the slow periodic heartbeat — so reactive
        triggers respond promptly to genuine events while still
        breaking the holon→member→finished→holon feedback loop.
        """
        self._maybe_schedule_rebalance()

    def _on_member_finished_local(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        """The *local* path: when the holon leader is also the gossip
        originator (very common — the lex-smallest leader in the chunk
        is often the one that ends up running gossip), the leader's
        own ``EnergyBalanceNegotiator`` emits ``NegotiationFinishedEvent``
        via ``context.emit_event``.  Local emits travel on a different
        bus than ``send_message``, so the message-path subscription
        misses this case.  Subscribing via ``subscribe_event`` here
        covers it.  Same throttle as ``_on_member_finished``.
        """
        if event.sector != self.sector:
            return
        self._maybe_schedule_rebalance()

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
        """Dispatch to either the legacy per-sector ADMM or the
        tier-stratified per-(sector, priority) ADMM (Package C) based
        on the ``enable_tier_stratified_admm`` flag.

        The tier-stratified path replaces the single scalar override
        per (member, sector) — which loses the holon's priority
        intent in the L2→L1 handoff — with a 2-D allocation
        ``targets[sector][tier]`` that the L1 honour path dispatches
        directly to per-tier sub-populations of each group.

        When the flag is set but the gathered answers show *no
        meaningful per-tier deficit* (e.g. a holon of fully-served
        communities with non-zero flow but no unmet demand), fall
        back to the legacy path.  Without that fallback the
        tier-stratified path would skip on "balanced" and the
        holon would miss the flow-redistribution that the legacy
        provided — observed empirically as a regression on
        label-propagation partitions where the legacy fired ~32
        rounds but the strict tier-stratified fired only 1.
        """
        if not self.enable_tier_stratified_admm:
            await self._run_legacy_per_sector_admm()
            return
        if self.admm_mode == "supply":
            # Route A: supply-priority ADMM.  Fires whenever the
            # holon has any supply at all to allocate; the priority
            # weighting takes over once supply < demand.
            if self._supply_priority_has_anything_to_do():
                await self._run_supply_priority_admm()
                return
            await self._run_legacy_per_sector_admm()
            return
        # Default demand-side path (Package C).
        if self._tier_stratified_has_meaningful_deficit():
            await self._run_tier_stratified_admm()
            return
        await self._run_legacy_per_sector_admm()

    def _supply_priority_has_anything_to_do(self) -> bool:
        """Cheap check: do queued answers contain *any* per-tier demand
        AND *any* supply?  Route A's value is in allocating supply
        across priorities; with one side empty there's nothing to do.
        """
        any_demand = any(
            a.demand_by_sector_priority for a in self._flex_answers
        )
        any_supply = any(
            float(s) > 1e-9
            for a in self._flex_answers
            for s in (a.supply_by_sector or {}).values()
        )
        return any_demand and any_supply

    def _tier_stratified_has_meaningful_deficit(
        self, *, threshold_mw: float = 1e-3
    ) -> bool:
        """Cheap check: do the queued flex answers carry any
        per-(sector, tier) deficit above the noise floor?

        Returns True if at least one (sector, tier) cell has
        |demand − served| > threshold AND tier ≥ 1 (i.e. not a
        generator / slack pseudo-tier).  False otherwise — caller
        should delegate to the legacy flow-redistribution path.
        """
        for answer in self._flex_answers:
            dem_map = answer.demand_by_sector_priority or {}
            ser_map = answer.served_by_sector_priority or {}
            for sec, tier_to_dem in dem_map.items():
                sec_ser = ser_map.get(sec, {})
                for tier, dem in tier_to_dem.items():
                    if tier < 1:
                        continue
                    ser = float(sec_ser.get(tier, 0.0))
                    if abs(float(dem) - ser) > threshold_mw:
                        return True
        return False

    async def _run_legacy_per_sector_admm(self) -> None:
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

        # --- Feasibility cap on T per dimension ---
        # Same logic as the supply-priority path: when |T[i]| exceeds
        # the sum of available actor budgets in dimension i, the L1
        # sharing-distance term plateaus at the structural gap and
        # the library spuriously logs "ADMM reached max iterations".
        # Bound T per dim by the actors' absolute budget envelope so
        # the target is reachable.  Sign is preserved — only magnitude
        # is reduced when over-budget.
        budget_pos = np.zeros(n_dims)
        budget_neg = np.zeros(n_dims)
        for lb_g, ub_g in group_bounds:
            budget_pos += np.maximum(ub_g, 0.0)
            budget_neg += np.maximum(-lb_g, 0.0)
        for i in range(n_dims):
            if total_T[i] > 0 and budget_pos[i] < total_T[i]:
                total_T[i] = budget_pos[i]
            elif total_T[i] < 0 and budget_neg[i] < -total_T[i]:
                total_T[i] = -budget_neg[i]

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
        if self.enable_priority_allocation:
            priority_shares = compute_priority_weighted_shares(
                [a.demand_by_priority for a in answers],
                [a.served_by_priority for a in answers],
                total_available,
            )
        else:
            # Priority allocation disabled — distribute the available
            # budget uniformly across answering groups so every group's
            # share is non-zero (preserves the legacy "balance-only"
            # behaviour the ablation intended).
            even = total_available / max(1, len(answers))
            priority_shares = [even for _ in answers]

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
        # Iter cap and tolerance configured via RestorationConfiguration
        # (plumbed through the role constructor in
        # scenario/restoration.py).  Defaults: 50 iters @ abs_tol=1e-3 —
        # relaxed from the package default 1000 / 1e-4 so concurrent
        # holon ADMMs across sectors don't block discrete-time progress
        # but loose enough that simbench_lv smoke runs converge.
        coordinator.max_iters = int(self.admm_max_iters)
        coordinator.abs_tol = float(self.admm_abs_tol)
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
        # Collect per-member overrides into the targets dict the
        # ``HolonAllocation`` Decision will carry, *and* push the
        # legacy ``StartBalanceNegotiation`` to keep L1's existing
        # override path working.  Two-line cost; one cleanly typed
        # decision on the new channel, one legacy message — both go
        # to the same members.
        allocation_targets: dict[str, float] = {}
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
            if override is not None:
                allocation_targets[str(addr)] = override
            await self.context.send_message(
                StartBalanceNegotiation(override_target=override),
                receiver_addr=addr,
            )

        # Publish HolonAllocation to CP connectors so L3 (CP ADMM)
        # can react to the cross-sector setpoint shift directly,
        # without the three-hop (L2 -> L1 -> gossip-finished -> L3)
        # detour.  Reuses the same ``tid="groups"`` cross-topology
        # link the SectorImbalanceBeacon publishes on.
        if allocation_targets:
            try:
                cp_connectors = list(topology_connectors(self, tid="groups"))
            except Exception:
                cp_connectors = []
            if cp_connectors:
                assignment = self.context.get_or_create_model(HolonicAssignment)
                holon_id = str(assignment.holon_id) if assignment.holon_id else ""
                decision = HolonAllocation(
                    publisher=str(self.context.aid),
                    version=self._version.next(),
                    caused_by={},
                    timestamp_s=float(self.context.current_timestamp),
                    sector=self.sector,
                    targets_mw=allocation_targets,
                    holon_id=holon_id,
                    residual=0.0,
                )
                logger.debug(
                    "[%s] holon publish: sector=%s n_targets=%d v=%d to %d cps",
                    self.context.aid, self.sector.value,
                    len(allocation_targets), decision.version, len(cp_connectors),
                )
                for addr in cp_connectors:
                    await self.context.send_message(decision, receiver_addr=addr)

    async def _run_tier_stratified_admm(self) -> None:
        """Package C — priority-aware holon ADMM.

        Builds a 2-D allocation problem ``T[sector][tier]`` and runs
        the standard ADMM sharing coordinator on a flattened
        ``(n_sectors · n_tiers)``-dimensional vector.  Each member
        actor contributes per-(sector, tier) bounds reflecting its
        local unserved demand / served headroom in that cell.  The
        coordinator's per-dimension ``priorities`` argument weights
        the L1 distance-to-target term: high-priority tiers get
        proportionally more pull toward their target deficit, so the
        solve naturally prioritises tier-1/2 deficits over tier-8/9
        deficits even when both are positive.

        The result ``actor.x[i]`` is a 2-D allocation that the L1
        honour path dispatches per-tier instead of per-group-sector,
        preserving the priority decision through the L2 → L1 handoff.

        Coalition / L2.5 interaction
        ----------------------------

        When a same-sector L2.5 coalition is active, the coalition has
        committed an absolute per-tier service fraction for the cells
        it claimed.  Its ``_reassert_active_coalitions`` re-fires every
        tick to hold those fractions against background drift, using
        ``service_fraction_by_sector_priority`` (absolute regulation
        factor) on the L1 dispatch path.

        This method operates on the same regulation knob but via
        ``override_targets_by_sector_priority`` (incremental delta on
        served setpoint).  If both fired on the same cell, the absolute
        and incremental control views would compound — every coalition
        tick rewrites the factor, every L2 delta drags it back, and
        the setpoint oscillates between the two.

        Resolution: cells with an active coalition record are filtered
        out of the per-member ``override_strat`` payload before
        dispatch.  L2 retains exclusive ownership of cells the coalition
        did not touch; the coalition retains exclusive ownership of the
        cells it claimed until ``ttl_s`` expiry or a
        ``BranchFailureEvent`` invalidation drops the record.  This
        mirrors the merge semantics already in
        ``_run_supply_priority_admm`` (which calls
        :meth:`CoalitionConstraintStore.merge_into`) — both demand and
        supply paths now defer to the coalition for the same cells.
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

        if not self._rebalance_active:
            return
        answers = self._flex_answers[:]
        senders = self._flex_answer_senders[:]
        self._flex_answers = []
        self._flex_answer_senders = []
        self._flex_expected = 0
        self._rebalance_active = False

        if len(answers) < 2:
            return

        # Diagnostic — dump the per-answer demand/served per (sector,
        # tier) so post-run analysis can see what data the ADMM is
        # actually working with.  Only fires when the method is
        # invoked (i.e. when the upstream deficit-check passed), so
        # it doesn't bloat balanced/healthy runs.
        for i, a in enumerate(answers):
            logger.info(
                "[%s] tier-strat answer[%d] demand_by_sec_prio=%s served=%s",
                self.context.aid, i,
                {s: dict(t) for s, t in (a.demand_by_sector_priority or {}).items()},
                {s: dict(t) for s, t in (a.served_by_sector_priority or {}).items()},
            )

        # Discover the (sector, tier) cells that have any demand
        # across the holon.  Empty cells are skipped — adding them to
        # the ADMM would just inflate the dimension without
        # contributing to the solve.
        sectors: list[str] = sorted({
            s for a in answers
            for s in (a.demand_by_sector_priority or a.balance_by_sector or {})
        })
        # Fall back: if no per-sector data, the per-tier ADMM has no
        # 2-D structure to exploit — re-route to the legacy path.
        if not sectors:
            await self._run_legacy_per_sector_admm()
            return

        # Find tiers actually present in any group's demand.  Including
        # tier 0 (slack / generator-class) is harmless — slacks
        # contribute zero demand so their cells stay zero.
        tiers_present: set[int] = set()
        for a in answers:
            for sec_tier_map in (a.demand_by_sector_priority or {}).values():
                tiers_present.update(sec_tier_map.keys())
        tiers = sorted(tiers_present)
        if not tiers:
            # No per-tier info — degenerate to legacy path.
            await self._run_legacy_per_sector_admm()
            return

        n_sec = len(sectors)
        n_tier = len(tiers)
        n_dims = n_sec * n_tier
        sec_idx = {s: i for i, s in enumerate(sectors)}
        tier_idx = {t: j for j, t in enumerate(tiers)}

        def _flat_idx(s: str, t: int) -> int:
            return sec_idx[s] * n_tier + tier_idx[t]

        # Build T per cell: signed deficit = served - demand (positive
        # = group's tier is under-served, in load convention so the
        # ADMM "pull" sign matches ``balance`` from the legacy path).
        total_T = np.zeros(n_dims)
        group_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for answer in answers:
            lb = np.zeros(n_dims)
            ub = np.full(n_dims, 1e-6)
            demand_map = answer.demand_by_sector_priority or {}
            served_map = answer.served_by_sector_priority or {}
            for sec in sectors:
                dem_t = demand_map.get(sec, {})
                ser_t = served_map.get(sec, {})
                for tier in tiers:
                    idx = _flat_idx(sec, tier)
                    dem = float(dem_t.get(tier, 0.0))
                    ser = float(ser_t.get(tier, 0.0))
                    deficit = dem - ser
                    # Sign convention matches legacy: served setpoint
                    # accumulates as ``balance``, so positive
                    # contribution = unmet demand the holon should
                    # pull this group up to absorb.
                    total_T[idx] += deficit
                    # Bounds: the group can absorb at most its unmet
                    # demand (ub > 0) or give back at most its served
                    # amount (lb < 0 if other groups need to take
                    # over).  Numerical floor 1e-6 prevents zero-width
                    # box that the QP solver dislikes.
                    lb[idx] = -max(ser, 0.0)
                    ub[idx] = max(dem - ser, 1e-6)
            group_bounds.append((lb, ub))

        if not np.all(np.isfinite(total_T)):
            logger.warning(
                "[%s] tier-stratified holon ADMM skipped: non-finite T",
                self.context.aid,
            )
            return
        if np.all(np.abs(total_T) < 1e-6):
            logger.info(
                "[%s] tier-stratified holon ADMM skipped: balanced "
                "(sectors=%s tiers=%s)",
                self.context.aid, sectors, tiers,
            )
            return

        # Per-dimension priority weight passed to the coordinator.
        # Larger weight = stronger pull toward target for that
        # dimension.  Shared with the L1 QP gossip weight
        # (``_qp_priority_weight``) and the supply-priority ADMM via
        # :func:`scare.base.util.tier_priority_weight` so every layer
        # agrees on tier ordering.
        from scare.base.util import tier_priority_weight

        P = self.priority_tiers
        priorities = np.zeros(n_dims)
        for sec in sectors:
            for tier in tiers:
                weight = tier_priority_weight(
                    tier, regime=1, priority_tiers=P,
                )
                priorities[_flat_idx(sec, tier)] = weight

        # Per-actor S coefficient.  Encourages each group to absorb
        # the share of T proportional to its own *deficit* share in
        # each cell (not its demand share — a group with demand but
        # no unmet demand has nothing to absorb).  Negative pull =
        # "I want this assignment."  Scaling by per-cell priority
        # weight then biases high-priority deficits to attract
        # stronger absorption pull than low-priority ones.
        actors: list[ADMMFlexActor] = []
        total_deficit_per_cell = np.zeros(n_dims)
        for answer in answers:
            demand_map = answer.demand_by_sector_priority or {}
            served_map = answer.served_by_sector_priority or {}
            for sec, tier_map in demand_map.items():
                if sec not in sec_idx:
                    continue
                for tier, dem in tier_map.items():
                    if tier in tier_idx:
                        ser = float(served_map.get(sec, {}).get(tier, 0.0))
                        deficit = max(0.0, float(dem) - ser)
                        total_deficit_per_cell[_flat_idx(sec, tier)] += deficit
        for idx_a, answer in enumerate(answers):
            lb, ub = group_bounds[idx_a]
            S = np.zeros(n_dims)
            demand_map = answer.demand_by_sector_priority or {}
            served_map = answer.served_by_sector_priority or {}
            for sec, tier_map in demand_map.items():
                if sec not in sec_idx:
                    continue
                for tier, dem in tier_map.items():
                    if tier not in tier_idx:
                        continue
                    idx = _flat_idx(sec, tier)
                    ser = float(served_map.get(sec, {}).get(tier, 0.0))
                    my_deficit = max(0.0, float(dem) - ser)
                    pie = total_deficit_per_cell[idx]
                    if pie > 1e-9 and my_deficit > 1e-9:
                        share = my_deficit / pie
                        # Scale by priority weight so high-priority
                        # cells dominate the local cost balance.
                        S[idx] = -share * priorities[idx]
            lb = np.nan_to_num(lb, nan=0.0, posinf=0.0, neginf=0.0)
            ub = np.nan_to_num(ub, nan=1e-6, posinf=1e6, neginf=1e-6)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

            # Per-actor coupling constraint (Option 2): cap the
            # group's TOTAL absorption across all (sector, tier) cells
            # at its physical flex headroom.  Without this, the per-
            # cell bounds make the problem trivially feasible
            # (Σ_g ub_g[cell] = T[cell]) and the priority weights in
            # the coordinator's L1 distance penalty never actually
            # arbitrate.  Adding the coupling creates scarcity
            # whenever ``Σ_cell deficit_g > flex_g`` (the group has
            # more local deficit than headroom to absorb) — in that
            # regime the ADMM has to choose, and the priority
            # weighting decides which tiers get covered first.
            #
            # ``flex_g`` is the same quantity ``_handle_ask_flex``
            # already reports as ``answer.flex`` (sum of cap - sp for
            # same-sector loads).  For pure-load groups where cap ≈
            # demand, this equals the sum-of-deficits, so the
            # constraint binds tightly and priority arbitrates only
            # marginally; for groups with physical / disconnect-
            # induced cap reduction it binds more sharply.
            budget_g = float(answer.flex)
            if budget_g <= 0.0:
                # No headroom at all → this actor must not absorb in
                # any cell.  Tighten the cap to zero so ADMM doesn't
                # produce a positive allocation that the L1 dispatch
                # then can't deliver.
                C = np.ones((1, n_dims))
                d = np.array([0.0])
            else:
                C = np.ones((1, n_dims))
                d = np.array([budget_g])
            actors.append(ADMMFlexActor(lb=lb, u=ub, C=C, d=d, S=S))

        coordinator = create_sharing_target_distance_admm_coordinator()
        coordinator.max_iters = int(self.admm_max_iters)
        coordinator.abs_tol = float(self.admm_abs_tol)
        start_msg = create_admm_start(
            create_admm_sharing_data(
                total_T.tolist(), priorities=priorities.tolist()
            )
        )

        from scare.base.diagnostics import record_event

        try:
            await start_coordinated_optimization(actors, coordinator, start_msg)
        except Exception as exc:
            logger.error(
                "[%s] tier-stratified holon ADMM failed: %s",
                self.context.aid, exc,
            )
            record_event(
                t=self.context.current_timestamp,
                kind="holon_admm_failed",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=f"tier_stratified: {exc}",
            )
            return

        results = [a.x.tolist() for a in actors]
        # Detailed diagnostic block: log T, priorities, and per-actor x
        # per (sector, tier) cell.  Aggregated allocation per tier
        # across actors lets post-run analysis verify that
        # high-priority tiers attract proportionally larger
        # absorption (the priority-awareness claim).
        per_cell_summary: list[dict[str, float | str | int]] = []
        sum_x_per_cell = np.sum(results, axis=0) if results else np.zeros(n_dims)
        for sec in sectors:
            for tier in tiers:
                idx = _flat_idx(sec, tier)
                per_cell_summary.append({
                    "sector": sec,
                    "tier": tier,
                    "T": round(float(total_T[idx]), 6),
                    "priority_weight": float(priorities[idx]),
                    "sum_x": round(float(sum_x_per_cell[idx]), 6),
                })
        logger.info(
            "[%s] tier-stratified holon ADMM result: sectors=%s tiers=%s "
            "T=%s sum_x=%s",
            self.context.aid, sectors, tiers, total_T.tolist(),
            sum_x_per_cell.tolist(),
        )
        record_event(
            t=self.context.current_timestamp,
            kind="holon_admm_result",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=(
                f"tier_stratified sectors={sectors} tiers={tiers} "
                f"per_cell={per_cell_summary}"
            ),
        )
        # New diagnostic event specifically for priority-awareness
        # verification: aggregate absorption per tier, weighted by
        # priority, lets the post-run analysis check that the
        # allocation respects the priority ordering.
        record_event(
            t=self.context.current_timestamp,
            kind="holon_priority_allocation",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=str({
                f"{sec}:tier{tier}": {
                    "T": round(float(total_T[_flat_idx(sec, tier)]), 6),
                    "weight": float(priorities[_flat_idx(sec, tier)]),
                    "sum_x": round(float(sum_x_per_cell[_flat_idx(sec, tier)]), 6),
                }
                for sec in sectors for tier in tiers
            }),
        )

        # Build the per-member 2-D allocation map and send it back
        # via the new ``StartBalanceNegotiation.override_targets_by_sector_priority``
        # field.  Each member's map is keyed by sector → tier →
        # target_mw, derived from their flat ADMM ``x`` vector.
        sender_to_x: dict[str, list[float]] = {}
        for sender, x_vec in zip(senders, results):
            sender_to_x[str(sender)] = x_vec

        if self._holon_member_addrs:
            triggers = list(self._holon_member_addrs)
        else:
            try:
                triggers = topology_neighbors(self, tid="holons")
            except Exception:
                triggers = []

        # Coalition deferral: when L2.5 has an active coalition
        # constraint for this leader's sector, the coalition owns the
        # per-tier regulation for the claimed tiers via absolute service
        # fractions re-asserted every tick (default 1 s).  L2's per-
        # tier dispatch is incremental on the *same* regulation knob:
        # ``new_sp = sp_curr + delta`` in ``_dispatch_per_tier_targets``.
        # Mixing absolute (coalition) and relative (L2) actuation on
        # the same knob caused the setpoint to oscillate as the two
        # paths fought over it.  We resolve it by partitioning: cells
        # the coalition has claimed are coalition-only until TTL/failure
        # invalidation; cells the coalition has not claimed remain L2's.
        # The supply-priority path performs the same merge through
        # ``CoalitionConstraintStore.merge_into``; this is the
        # equivalent for the demand-side tier-stratified path.
        coalition_tiers: set[int] = set()
        if self._coalition_constraint_store is not None:
            coalition_tiers = self._coalition_constraint_store.active_tiers(
                self.sector, float(self.context.current_timestamp),
            )
        own_sec = self.sector.value
        if coalition_tiers:
            logger.info(
                "[%s] tier-strat deferring tiers=%s to active coalition (sector=%s)",
                self.context.aid, sorted(coalition_tiers), own_sec,
            )

        allocation_targets_scalar: dict[str, float] = {}
        for addr in triggers:
            x_vec = sender_to_x.get(str(addr))
            override_legacy: float | None = None
            override_strat: dict[str, dict[int, float]] | None = None
            if x_vec is not None:
                # Build the per-(sector, tier) override map.
                # ``T`` in this ADMM is the *deficit* (= demand −
                # served, positive when load is unserved) and ``x``
                # is each actor's share of closing that deficit
                # (positive ⇒ "I should serve more").  The L1
                # dispatch (``_dispatch_per_tier_targets``) reads
                # the override as the desired *change in served
                # setpoint*: positive ⇒ raise served (restore),
                # negative ⇒ lower served (shed).  Sign convention
                # therefore matches ``x`` directly — no negation,
                # unlike the legacy per-sector path which uses the
                # gossip's negated target convention.
                strat: dict[str, dict[int, float]] = {}
                for sec in sectors:
                    per_tier: dict[int, float] = {}
                    for tier in tiers:
                        # Skip tiers an active same-sector coalition
                        # has claimed (only for our own sector; other
                        # sectors stay under L2's control).
                        if sec == own_sec and tier in coalition_tiers:
                            continue
                        v = float(x_vec[_flat_idx(sec, tier)])
                        per_tier[tier] = v
                    if per_tier:
                        strat[sec] = per_tier
                if strat:
                    override_strat = strat
                # Also compute the per-sector scalar (sum across
                # tiers) as a backwards-compatible scalar override —
                # so any L1 path that hasn't been migrated to the
                # tier-aware dispatch still receives a usable target.
                # The scalar takes the *negated* sum to match the
                # legacy gossip-target convention (target<0 ⇒
                # absorb in QP gossip's framing).
                if strat:
                    if own_sec in strat:
                        override_legacy = -sum(strat[own_sec].values())
                    else:
                        first_sec = next(iter(strat))
                        override_legacy = -sum(strat[first_sec].values())
            # If every cell for this member was deferred to the
            # coalition (override_strat is None and override_legacy is
            # None), skip the send entirely.  Falling through with a
            # bare ``StartBalanceNegotiation()`` would otherwise drop
            # into ``trigger_balance_negotiation`` on the receiver and
            # fight the coalition's re-asserted fraction.
            if override_strat is None and override_legacy is None:
                continue
            if override_legacy is not None:
                allocation_targets_scalar[str(addr)] = override_legacy
            await self.context.send_message(
                StartBalanceNegotiation(
                    override_target=override_legacy,
                    override_targets_by_sector_priority=override_strat,
                ),
                receiver_addr=addr,
            )

        # Publish HolonAllocation to CP connectors (unchanged from
        # legacy: the CP layer treats the holon's intent as a
        # per-member scalar, which the scalar override above provides).
        if allocation_targets_scalar:
            try:
                cp_connectors = list(topology_connectors(self, tid="groups"))
            except Exception:
                cp_connectors = []
            if cp_connectors:
                assignment = self.context.get_or_create_model(HolonicAssignment)
                holon_id = str(assignment.holon_id) if assignment.holon_id else ""
                decision = HolonAllocation(
                    publisher=str(self.context.aid),
                    version=self._version.next(),
                    caused_by={},
                    timestamp_s=float(self.context.current_timestamp),
                    sector=self.sector,
                    targets_mw=allocation_targets_scalar,
                    holon_id=holon_id,
                    residual=0.0,
                )
                for addr in cp_connectors:
                    await self.context.send_message(decision, receiver_addr=addr)

    async def _run_supply_priority_admm(self) -> None:
        """Route A — supply-priority ADMM.

        Differs from the demand-side path in two key ways:

        1. ``T`` is the *total demand* per (sector, tier) cell across
           the whole holon — i.e. the ideal-served target — not the
           per-cell deficit.  The ADMM tries to match this target,
           with the L1 distance penalty weighted by tier priority.
        2. Each actor's per-cell ``ub`` and per-actor coupling
           ``Σ x ≤ supply_g`` reflect that actor's *generator
           capacity*, not its local demand.  This lets a group with
           supply but no local tier-X demand contribute to the
           holon-wide tier-X service (the LP downstream routes the
           freed power via the grid).

        Output: per-(sector, tier) service fractions sent to each
        leader.  Each leader applies the fraction uniformly to its
        local loads at that priority tier, so a high-priority tier
        served at 100 % and a low-priority tier served at 0 %
        produces a consistent shed-the-low-tier / serve-the-high-
        tier pattern across the holon, regardless of which group
        physically holds the supply.
        """
        if not self._rebalance_active:
            return
        answers = self._flex_answers[:]
        senders = self._flex_answer_senders[:]
        self._flex_answers = []
        self._flex_answer_senders = []
        self._flex_expected = 0
        self._rebalance_active = False

        if len(answers) < 2:
            return

        # Discover (sector, tier) cells with demand anywhere in the
        # holon.  Sectors with supply but no demand are still useful
        # — supply at sector X bypasses the ADMM (sector X has no
        # cells), but the leader's reply will already account for it.
        sectors: list[str] = sorted({
            s for a in answers
            for s in (a.demand_by_sector_priority or {})
        })
        if not sectors:
            await self._run_legacy_per_sector_admm()
            return

        tiers_present: set[int] = set()
        for a in answers:
            for sec_tier_map in (a.demand_by_sector_priority or {}).values():
                tiers_present.update(sec_tier_map.keys())
        tiers = sorted(t for t in tiers_present if t >= 1)
        if not tiers:
            await self._run_legacy_per_sector_admm()
            return

        # Pre-check: if there's no demand anywhere the helper would
        # return a no-op, but the legacy path historically falls back
        # in this corner case.  Preserve that.
        total_demand = sum(
            float(d)
            for a in answers
            for tier_to_dem in (a.demand_by_sector_priority or {}).values()
            for d in tier_to_dem.values()
        )
        if total_demand < 1e-6:
            await self._run_legacy_per_sector_admm()
            return

        actor_supplies = [a.supply_by_sector or {} for a in answers]
        actor_demands = [a.demand_by_sector_priority or {} for a in answers]

        from scare.base.diagnostics import record_event
        from scare.community.supply_priority_admm import (
            allocate_supply_priority,
        )

        # F6: deliverability caps.  When we know each member leader's
        # home node id and have a topology mirror, compute per-actor
        # ``{(sector, tier): cap}`` overrides so the ADMM does not
        # commit supply at an actor that cannot physically route it
        # under the current branch-active mask.  Conservative-by-node
        # variant: cap each cell at the sum of demand co-located at
        # nodes reachable from this actor's home node — collapsing
        # demand to the *leader's* node when we don't have per-load
        # node information (the L2.5 coalition path tracks per-load
        # node ids; L2 currently only knows leader nodes).  An
        # entirely unreachable leader gets all caps at 0; a reachable
        # leader gets uncapped (None entry) so the per-actor coupling
        # is the only binding constraint.
        actor_ub_overrides: list[dict[tuple[str, int], float] | None] | None = None
        if (
            self._topology_mirror is not None
            and self._leader_node_ids
        ):
            try:
                from scare.community.deliverability import (
                    per_actor_deliverable_caps,
                )

                actor_node_ids: list[Any | None] = []
                actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]] = []
                for sender, answer in zip(senders, answers):
                    leader_aid = getattr(sender, "aid", str(sender))
                    node_id = self._leader_node_ids.get(leader_aid)
                    actor_node_ids.append(node_id)
                    # Map this leader's tier-aggregated demand onto its
                    # own home node — coarse but correct: a leader that
                    # is reachable will see its own demand contribute
                    # to every other leader's reachable-cap; a leader
                    # that is unreachable contributes nothing.
                    per_tier: dict[int, dict[Any, float]] = {}
                    if node_id is not None:
                        for sec, tier_map in (
                            answer.demand_by_sector_priority or {}
                        ).items():
                            if sec not in sectors:
                                continue
                            for tier, dem in tier_map.items():
                                per_tier.setdefault(int(tier), {})[node_id] = (
                                    per_tier[int(tier)].get(node_id, 0.0)
                                    + float(dem)
                                )
                    actor_demand_nodes_by_tier.append(per_tier)

                actor_ub_overrides = per_actor_deliverable_caps(
                    actor_node_ids=actor_node_ids,
                    actor_demand_nodes_by_tier=actor_demand_nodes_by_tier,
                    sector=self.sector,
                    mirror=self._topology_mirror,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] supply-priority holon: deliverability caps "
                    "failed (%s) — falling back to raw supply",
                    self.context.aid, exc,
                )
                actor_ub_overrides = None

        try:
            service_fraction, _per_actor_x, meta = await allocate_supply_priority(
                sectors=sectors,
                tiers=tiers,
                actor_supplies=actor_supplies,
                actor_demands=actor_demands,
                actor_ub_overrides=actor_ub_overrides,
                priority_tiers=self.priority_tiers,
                max_iters=int(self.admm_max_iters),
                abs_tol=float(self.admm_abs_tol),
                enable_priority_weighting=self.enable_priority_allocation,
            )
        except Exception as exc:
            logger.error(
                "[%s] supply-priority holon ADMM failed: %s",
                self.context.aid, exc,
            )
            record_event(
                t=self.context.current_timestamp,
                kind="holon_admm_failed",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=f"supply_priority: {exc}",
            )
            return

        total_T = meta["T_per_cell"]
        sum_x_per_cell = meta["sum_x_per_cell"]
        priorities = meta["priorities"]
        n_tier = len(tiers)
        sec_idx = {s: i for i, s in enumerate(sectors)}
        tier_idx = {t: j for j, t in enumerate(tiers)}

        def _flat_idx(s: str, t: int) -> int:
            return sec_idx[s] * n_tier + tier_idx[t]

        logger.info(
            "[%s] supply-priority holon ADMM result: sectors=%s tiers=%s "
            "T=%s sum_x=%s holon_supply=%.4f",
            self.context.aid, sectors, tiers, total_T,
            sum_x_per_cell,
            sum(meta["actor_supply_total"]),
        )
        record_event(
            t=self.context.current_timestamp,
            kind="holon_admm_result",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=f"supply_priority sectors={sectors} tiers={tiers} fractions={service_fraction}",
        )
        record_event(
            t=self.context.current_timestamp,
            kind="holon_priority_allocation",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=str({
                f"{sec}:tier{tier}": {
                    "T": round(float(total_T[_flat_idx(sec, tier)]), 6),
                    "weight": float(priorities[_flat_idx(sec, tier)]),
                    "sum_x": round(float(sum_x_per_cell[_flat_idx(sec, tier)]), 6),
                    "service_frac": round(service_fraction[sec][tier], 4),
                }
                for sec in sectors for tier in tiers
            }),
        )

        # Coalition constraint binding: if the sibling
        # HolonSummaryRole has an active coalition fraction for any
        # (sector, tier) cell, that fraction overrides L2's per-cell
        # result.  L2 retains ownership of cells the coalition didn't
        # touch.  Without this merge, last-write-wins between L2 and
        # L2.5 lets each subsequent L2 round reset the per-tier
        # regulation back to its own (typically more pessimistic)
        # allocation, undoing the coalition's redistribution.
        if self._coalition_constraint_store is not None:
            now = float(self.context.current_timestamp)
            service_fraction = self._coalition_constraint_store.merge_into(
                service_fraction, self.sector, now,
            )

        # Send the SAME service fraction map to every member leader
        # — the fractions are holon-global, so each leader applies
        # them locally.  This is the L1 honour path for Route A.
        if self._holon_member_addrs:
            triggers = list(self._holon_member_addrs)
        else:
            try:
                triggers = topology_neighbors(self, tid="holons")
            except Exception:
                triggers = []
        for addr in triggers:
            await self.context.send_message(
                StartBalanceNegotiation(
                    service_fraction_by_sector_priority=service_fraction,
                ),
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
        """Periodic broadcast of own δ_g to same-sector neighbours.

        The leader ALSO asks itself for fresh flex on each tick.
        ``_peer_last_delta[own_key]`` is only populated by
        ``_handle_flex_answer``, which fires on incoming
        ``AvailableFlexAnswer`` messages.  Non-holon-leader leaders
        never trigger a rebalance round (no ``_try_rebalance`` call),
        so without an explicit self-ask their own δ_g stays at 0
        forever — making every pairwise product δ_g · δ_h zero and
        H_{gh} unable to cross any threshold.  Pre-2026-05 this
        latent bug rendered the Hebbian co-variance estimate
        identically zero for every peer; the recluster's
        ``no_candidates`` early-return was the symptom.
        """
        if not self.enable_hebbian_formation:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        from scare.base.model import AskForAvailableFlex

        # Self-ask so this tick (or the next, if the reply lands after
        # the send below) sees a non-zero own δ_g in
        # ``_peer_last_delta``.  The handler is already subscribed
        # because the leader handles its own AskForAvailableFlex
        # exactly like a member's.
        await self.context.send_message(
            AskForAvailableFlex(include_connectors=False),
            receiver_addr=self.context.addr,
        )

        # Compute own delta_g from the latest known flex (if any) or
        # simply 0 during warmup before the first self-ask reply
        # arrives.
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
        """Return the set of peer-keys whose H_{gh} crosses the threshold
        in the *anti-correlation* direction.

        ADMM benefits when paired groups have *complementary* stress —
        one's surplus matches the other's deficit, and the sharing
        solve has real residual to push around.  Pairing groups whose
        δ moves *together* (high positive H) just stacks similar
        deficits on top of each other and gives ADMM nothing to do.
        So the rule is "admit a peer when ``H_{gh} < -threshold``" —
        same magnitude criterion as the classical Hebbian threshold,
        but mirrored: we want consistent anti-coupling.

        The threshold value retains the same meaning (a magnitude on
        the running EWMA of δ_g · δ_h), so existing tunings of
        ``hebbian_threshold`` carry over with their semantic intact.

        Public so the diagnostics layer (and the ``C.5`` cluster-sync
        analysis script) can compare static topology membership against
        the dynamically-emergent partition.
        """
        return {
            k for k, (h, _n) in self._hebbian_H.items()
            if -h > self.hebbian_threshold
        }

    async def _hebbian_recluster(self) -> None:
        """Apply the thresholded Hebbian co-variance to refine the
        leader's holon membership after warmup.

        Only the leader (no parent_addr) reclusters; non-leader peers
        accept whatever membership their leader announces via the
        existing propose/accept channel.  Reclustering modifies
        ``_holon_member_addrs`` / ``_holon_member_keys`` in place so
        the next ``_try_rebalance`` cycle picks up the refined set.

        ``async`` even though the body has no awaits — mango's
        ``PeriodicScheduledTask.run`` does ``await self._coroutine_func()``
        which requires the callable to return an awaitable; a sync
        function returns None and the periodic loop dies on the first
        invocation with ``TypeError: object NoneType can't be used in
        'await' expression``.  Pre-2026-05 this was an undiagnosed
        latent bug — Hebbian reclustering never actually fired.
        """
        from scare.base.diagnostics import record_event

        if not self.enable_hebbian_formation:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        if self.context.current_timestamp < self._hebbian_warmup_until:
            record_event(
                t=self.context.current_timestamp,
                kind="hebbian_recluster_attempted",
                aid=self.context.aid,
                sector=self.sector.value,
                detail="warmup",
            )
            return
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return  # not a leader of a formed holon

        candidates = self.hebbian_membership_candidates()
        if not candidates:
            record_event(
                t=self.context.current_timestamp,
                kind="hebbian_recluster_attempted",
                aid=self.context.aid,
                sector=self.sector.value,
                detail="no_candidates",
            )
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
            record_event(
                t=self.context.current_timestamp,
                kind="hebbian_recluster_attempted",
                aid=self.context.aid,
                sector=self.sector.value,
                detail="no_drift",
            )
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

        record_event(
            t=self.context.current_timestamp,
            kind="hebbian_recluster",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=f"members={len(new_keys)} drift={len(new_keys ^ prev_keys)}",
        )
