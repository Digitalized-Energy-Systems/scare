"""Holonic (multi-level) community formation and coordination.

Two-layer architecture:
  Layer 1 — sector agents solve local restoration via gossip-based
            balance negotiation using only local/neighbour information.
  Layer 2 — holon leaders aggregate member-group flex and run DRO ADMM
            for inter-group resource sharing.

A super-community (holon) layer sits on top of base group formation
(``PreAssignedCommunityRole``): group leaders merge with neighbouring
leaders into holons; the holon leader runs ADMM to distribute
surplus/deficit across members, then drives each group to rebalance
internally toward the per-actor allocation as gossip target.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from distributed_resource_optimization import (
    create_admm_sharing_data,
    create_admm_start,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.flex_actor import (
    ADMMFlexActor,
)
from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
    topology_neighbors,
)

from scare.base.channel import (
    CoalitionConstraint,
    ComponentAdmmReport,
    ComponentAllocation,
    CPSetpoint,
    HolonAllocation,
    L3RebalanceWakeup,
    MonotonicVersion,
    SeenVersions,
)
from scare.base.diagnostics import record_event
from scare.base.model import (
    AskForAvailableFlex,
    AvailableFlexAnswer,
    CommunityAssignment,
    CommunityReassignedEvent,
    FailureNotice,
    HolonicAssignment,
    HolonicJoinAnswer,
    HolonicJoinRequest,
    L2RecycleEscalation,
    LeaderEmerged,
    LocalGenerationApproval,
    LocalGenerationRequest,
    NegotiationFinishedEvent,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.topology_mirror import LivePeerFilter
from scare.base.util import compute_priority_weighted_shares, tier_priority_weight_strict
from scare.community.deliverability import per_actor_deliverable_caps
from scare.community.holon_flex import (
    aggregate_holon_flex,
    extract_demand_sectors_tiers,
)
from scare.community.supply_priority_admm import allocate_supply_priority

logger = logging.getLogger(__name__)

# Base flex-collection timeout per sector.  Heat is slow (thermal
# inertia) so leaders need more time; electricity is fast; gas between.
_FLEX_TIMEOUT_BASE_S: dict[Sector, float] = {
    Sector.ELECTRICITY: 3.0,
    Sector.GAS: 8.0,
    Sector.HEAT: 15.0,
}
_FLEX_TIMEOUT_DEFAULT_S = 5.0
_FLEX_TIMEOUT_PER_MEMBER_S = 0.5  # added per expected member group


DEFAULT_MAX_HOLON_SIZE: int = 4


class HolonicCommunityRole(Role):
    """Holonic (super-community) formation and inter-group coordination
    via DRO ADMM.

    A holon leader: (1) collects ``AvailableFlexAnswer`` from member
    leaders, (2) runs an ADMM sharing optimisation to redistribute
    resources across groups, (3) triggers intra-group rebalancing.

    Attached to every agent in the group topology, but only group
    leaders initiate holon formation and inter-group coordination.
    """

    def __init__(
        self,
        sector: Sector,
        *,
        formation_period_s: float = 4.0,
        max_holon_size: int = DEFAULT_MAX_HOLON_SIZE,
        rebalance_period_s: float = 1.0,
        rebalance_min_gap_s: float = 0.5,
        flex_timeout_s: float = 5.0,
        watchdog_s: float = 30.0,
        admm_max_iters: int = 50,
        admm_abs_tol: float = 1e-3,
        enable_tier_stratified_admm: bool = True,
        priority_tiers: int = 4,
        admm_scope: str = "sector",
        enable_priority_allocation: bool = True,
        live_member_filter: LivePeerFilter | None = None,
        coalition_constraint_store: Any = None,
        my_node_id: Any = None,
        leader_node_ids: dict[str, Any] | None = None,
        topology_mirror: Any = None,
        cp_node_ids: set[Any] | None = None,
    ) -> None:
        super().__init__()
        self.sector = sector
        # Shared store the sibling ``HolonSummaryRole`` writes coalition
        # fractions into; read on every supply-priority dispatch so they
        # override L2's per-tier ADMM result for the TTL window.  None ⇒
        # no merge (last-write-wins between L2 and L2.5).
        self._coalition_constraint_store = coalition_constraint_store
        # Optional ``DynamicHolonRole`` classifying which holon members
        # are physically reachable via live grid edges.  None ⇒ static-
        # topology mode (every member treated reachable).  Used by
        # ``_live_members``.
        self._live_member_filter = live_member_filter
        self.formation_period_s = formation_period_s
        self.max_holon_size = max_holon_size
        # ADMM convergence knobs (trade quality vs wallclock).
        self.admm_max_iters = admm_max_iters
        self.admm_abs_tol = admm_abs_tol
        # Tier-stratified ADMM: per-(sector, tier) supply-priority
        # allocation.  See ``_run_supply_priority_admm`` /
        # ``_run_component_scoped_admm``.
        self.enable_tier_stratified_admm = enable_tier_stratified_admm
        self.priority_tiers = priority_tiers
        # ADMM scope: "holon" (each leader solves over its holon members),
        # "sector" (deprecated — every holon leader is one actor), or
        # "component" (default — every group leader on the same active
        # subgraph is one actor, coordinator elected per active component).
        if admm_scope not in {"holon", "sector", "component"}:
            raise ValueError(
                "holon admm_scope must be 'holon', 'sector', or 'component', "
                f"got {admm_scope!r}"
            )
        self.admm_scope = admm_scope
        # Priority-weighted allocation switch.  False ⇒ uniform per-tier
        # weights (no-priority ablation).
        self.enable_priority_allocation = bool(enable_priority_allocation)
        # Deliverability wiring (F6).  When all three are present the
        # supply-priority ADMM passes ``actor_ub_overrides`` capping each
        # member's per-tier supply commitment at the tier-t demand
        # reachable from its home node, so L2 never allocates supply the
        # LP cannot route after a partition (mirrors the L2.5 path).
        self._my_node_id = my_node_id
        self._leader_node_ids: dict[str, Any] = dict(leader_node_ids or {})
        self._topology_mirror = topology_mirror
        # Node ids hosting a CP agent.  The per-component L2 path uses
        # this to detect a CP in the leader's multi-sector component;
        # empty set ⇒ no CP to defer to.
        self._cp_node_ids: set[Any] = set(cp_node_ids or set())
        # ``rebalance_period_s`` is the slow background heartbeat that
        # catches drift from timeseries inputs (load profiles, supply
        # temperatures) changing without firing a NegotiationFinishedEvent.
        # ``rebalance_min_gap_s`` is the fast feedback-loop fuse: holon
        # ADMM → member gossip → finished events → re-fires
        # ``_on_member_finished``; the gap lets one round of member gossip
        # resolve before another ADMM cycle, breaking the loop.
        self.rebalance_period_s = rebalance_period_s
        self.rebalance_min_gap_s = rebalance_min_gap_s
        self.flex_timeout_s = flex_timeout_s

        # Slow periodic safety net so a leader that missed every trigger
        # event still retries holon formation / rebalances eventually.
        self.watchdog_s = watchdog_s

        # holon_id -> {sender_key: (addr, accept_or_None)}.  Address kept
        # alongside the response to build the resolved member list without
        # a topology re-lookup.
        self._pending_proposals: dict[UUID, dict[str, tuple[Any, bool | None]]] = {}
        # Collected member flex answers for inter-holon ADMM;
        # ``_flex_answer_senders`` holds the sender per answer so the
        # leader can route the per-actor allocation back as an override
        # balance target.
        self._flex_answers: list[AvailableFlexAnswer] = []
        self._flex_answer_senders: list[Any] = []
        self._flex_expected: int = 0
        self._rebalance_active: bool = False
        # Last rebalance start time; reactive triggers within
        # ``rebalance_min_gap_s`` are dropped (bounds the feedback loop).
        self._last_rebalance_t: float = float("-inf")
        # No-change skip for the watchdog tick: a reactive trigger sets
        # this True, a successful rebalance clears it.  Watchdog skips
        # when still False (nothing moved since last run).  True initially
        # so the first tick runs.
        self._rebalance_dirty: bool = True
        # Guards a single deferred retry scheduled at gap-expiry: a
        # reactive trigger throttled inside ``rebalance_min_gap_s`` would
        # otherwise sit dirty until the slow watchdog tick (never fires in
        # a short sim).  Runs the throttled work as soon as the fuse
        # clears without a periodic heartbeat.
        self._rebalance_retry_pending: bool = False

        # Resolved holon membership (leader side).  Populated by
        # ``_handle_join_answer``; ``_try_rebalance`` targets flex
        # requests at only these members so ``_flex_expected`` scales with
        # chunk size, not clique size.
        self._holon_member_addrs: list[Any] = []
        self._holon_member_keys: set[str] = set()

        # Per-reason set so each ``_try_rebalance`` idle early-return
        # surfaces once at INFO rather than spamming every periodic tick.
        self._idle_logged: set[str] = set()

        # --- Channel-pattern state (L2 <-> L3 direct link) ---
        # ``_version`` advances on every published ``HolonAllocation``;
        # ``_seen_cps`` tracks the latest consumed ``CPSetpoint`` version
        # per publisher so the predicate doesn't re-fire on stale data.
        self._version = MonotonicVersion()
        self._seen_cps = SeenVersions()
        self._cp_setpoint_state: dict[tuple[str, str], float] = {}
        self._last_cp_predicate_fire_t: float = -1e9

        # --- Component-scoped L2 ADMM state (admm_scope="component") ---
        # The component coordinator (lex-smallest leader aid mutually
        # reachable on the active subgraph) buffers ``ComponentAdmmReport``
        # from every reachable leader and runs one ADMM with N actors
        # (= N communities).  Buffer keyed by ``leader_aid``; latest report
        # per leader wins.  ``_component_dispatch_pending`` debounces a
        # burst of reports into one solve.
        self._component_round_counter: int = 0
        self._component_report_buffer: dict[str, tuple[str, Any]] = {}
        self._component_dispatch_pending: bool = False
        # Coordinator-side dispatch throttle (same semantics as
        # ``rebalance_min_gap_s``, tracked separately since reports may
        # arrive between this community's own rebalances).
        self._last_component_dispatch_t: float = float("-inf")
        # Latest dispatched service_fraction — newly-arriving reports
        # merge against it and the heartbeat can skip "no material change".
        self._last_component_fraction: dict[str, dict[int, float]] | None = None

        # --- ComponentAllocation versioning (packet-loss recovery) ---
        # Strictly-monotone counter stamped on each outgoing
        # ``ComponentAllocation``.  Receivers echo the last version they
        # applied (``last_applied_allocation_version``) on their next
        # report; the coordinator detects loss and re-sends to the stale
        # receiver, making the broadcast reliable under packet loss.
        self._allocation_version_counter: int = 0
        # Latest dispatched allocation, re-sent to stale leaders on
        # report-receipt.  None until the first dispatch.
        self._last_dispatched_allocation: Any = None  # ComponentAllocation
        # Latest ``version`` applied as an L2 leaf; echoed in every
        # outgoing ``ComponentAdmmReport``.  -1 = none applied yet.
        self._last_applied_allocation_version: int = -1

    def setup(self) -> None:
        # Holon formation is event-driven (member-finished / join-request
        # / repartition); this slow watchdog covers a leader that missed
        # every trigger event.
        self.context.schedule_periodic_task(
            self._try_form_holon, delay=self.watchdog_s
        )
        # Inter-group rebalance is purely event-driven via the reactive
        # handlers below (L1 NFE, L3 CP setpoint, L2.5 coalition, L1
        # fallback escalation, L3 wakeup) plus this slow watchdog.  No
        # drift-probe timer: every cause of drift is itself an event.
        self.context.schedule_periodic_task(
            self._try_rebalance, delay=self.watchdog_s
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
        # Reactive L2 trigger: when a member group finishes its balance
        # gossip, kick off an inter-group rebalance to spread the residual
        # across the holon.  Two subscriptions: ``subscribe_message``
        # catches broadcasts from other leaders over the holons topology;
        # ``subscribe_event`` catches the local-emit case where this
        # agent's own EnergyBalanceNegotiator finished (emit_event and
        # send_message buses are disjoint).
        self.context.subscribe_message(
            self,
            _wrap(self._on_member_finished),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent)
            and msg.sector == self.sector,
        )
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_member_finished_local
        )
        # Locality-respecting prompt L2 re-cycle: react to the propagated,
        # TTL-bounded ``FailureNotice`` so a topology change re-allocates
        # the affected component promptly under the re-elected coordinator
        # instead of waiting for a downstream L1 negotiation.  Sector-
        # filtered: only a same-sector failure changes this sector's
        # component connectivity.
        self.context.subscribe_message(
            self,
            _wrap(self._on_failure_notice),
            lambda msg, meta: isinstance(msg, FailureNotice)
            and msg.sector == self.sector,
        )
        # L2 recycle escalation: a member relays a locally-detected
        # failure to its leader (from_member=True), or a peer leader
        # fans the escalation across the component (from_member=False).
        self.context.subscribe_message(
            self,
            _wrap(self._handle_l2_recycle),
            lambda msg, meta: isinstance(msg, L2RecycleEscalation)
            and msg.sector == self.sector,
        )
        # Direct L3 -> L2 trigger (channel/decision pattern).  When a CP
        # plant commits a cross-sector setpoint, affected holons
        # re-evaluate without waiting for the L1 gossip chain.  The
        # handler's predicate decides whether the change merits a
        # rebalance.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_cp_setpoint),
            lambda msg, meta: isinstance(msg, CPSetpoint),
        )
        # Direct L2.5 -> L2 trigger.  A fresh ``CoalitionConstraint``
        # kicks an immediate L2 rebalance so the merged service fractions
        # (L2's ADMM result ⊕ coalition's per-tier overrides via the
        # store) propagate to L1 in the same tick rather than waiting for
        # the slow heartbeat.  L2 owns the holon-wide allocation across
        # all tiers, so it merges the cross-holon update.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_coalition_constraint),
            lambda msg, meta: isinstance(msg, CoalitionConstraint)
            and msg.sector == self.sector,
        )
        # L1 stall escalation: a member group's gossip converged with
        # an unresolved deficit and is asking L2 to arbitrate before
        # the local-generation fallback fires.  Routing this through
        # L2 prevents L1 from silently ramping local DGs in parallel
        # to a holon allocation it doesn't know about.  The handler
        # triggers an early rebalance attempt and then approves the
        # fallback for whatever the rebalance cannot absorb.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_local_gen_request),
            lambda msg, meta: isinstance(msg, LocalGenerationRequest)
            and msg.sector == self.sector,
        )
        # S2 — L3 multi-sector coord wake-up.  When the L3 solve dispatches
        # new CP setpoints, the commit changes per-sector supply/demand;
        # L2 must re-evaluate on the post-commit state but otherwise has
        # no reactive trigger for the L3 path.  ``L3RebalanceWakeup`` is a
        # no-payload nudge that marks dirty via
        # ``_maybe_schedule_rebalance`` (still throttled there).
        self.context.subscribe_message(
            self,
            _wrap(self._handle_l3_wakeup),
            lambda msg, meta: isinstance(msg, L3RebalanceWakeup)
            and msg.sector == self.sector,
        )
        # Component-scoped L2 ADMM: every group leader subscribes to both
        # message types.  ``ComponentAdmmReport`` is acted on only if this
        # leader is the elected coordinator (handler self-gates).
        # ``ComponentAllocation`` is acted on by every leader — the
        # coordinator's dispatch envelope, applied directly to the
        # addressee's own community members (no holon hop).
        if self.admm_scope == "component":
            self.context.subscribe_message(
                self,
                _wrap(self._handle_component_admm_report),
                lambda msg, meta: isinstance(msg, ComponentAdmmReport)
                and msg.sector == self.sector,
            )
            self.context.subscribe_message(
                self,
                _wrap(self._handle_component_allocation),
                lambda msg, meta: isinstance(msg, ComponentAllocation)
                and msg.sector == self.sector,
            )

        # LeaderEmerged: a previously-non-leader agent promoted to lead an
        # orphan sub-community after a failure-driven repartition.
        # Updating ``_leader_node_ids`` keeps
        # ``_resolve_component_peer_addrs`` from filtering the new leader
        # out of the component peer set.  Synchronous (just mutates a
        # dict); sector-filtered.
        def _on_leader_emerged_msg(msg: Any, meta: dict) -> None:
            try:
                self._on_leader_emerged(msg)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[%s] _on_leader_emerged failed for %r: %s",
                    self.context.aid, msg, exc,
                )

        self.context.subscribe_message(
            self,
            _on_leader_emerged_msg,
            lambda msg, meta: isinstance(msg, LeaderEmerged)
            and msg.sector == self.sector,
        )

        # Holon-formation event trigger: retry when the eligible-neighbour
        # set could have changed, rather than waiting a full
        # ``watchdog_s`` for the periodic fallback.
        self.context.subscribe_event(
            self, CommunityReassignedEvent, self._on_community_reassigned
        )

    # ------------------------------------------------------------------
    # Holon formation
    # ------------------------------------------------------------------

    def _on_leader_emerged(self, message: LeaderEmerged) -> None:
        """Register a newly-promoted orphan-community leader in
        ``_leader_node_ids`` so ``_resolve_component_peer_addrs`` admits
        it into the component peer set when reachable.  Idempotent.
        """
        aid = str(message.leader_aid)
        if not aid:
            return
        prior = self._leader_node_ids.get(aid)
        self._leader_node_ids[aid] = message.node_id
        if prior is None:
            record_event(
                t=float(self.context.current_timestamp),
                kind="leader_emerged_registered",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=(
                    f"new_leader={aid} node_id={message.node_id} "
                    f"known_leaders={len(self._leader_node_ids)}"
                ),
            )

    def _on_community_reassigned(
        self, _event: CommunityReassignedEvent, _src: Any
    ) -> None:
        """Repartition changed our community membership, so our holon-
        eligibility set may have moved.  Kick a formation attempt and a
        full L2 rebalance without waiting for the watchdog.

        The rebalance is needed because a failure that islands a leader
        re-elects a fresh component coordinator; re-running the cycle on
        the topology-change event makes every reassigned leader re-collect
        and re-report so the new component is allocated consistently
        (otherwise it lingers on the predecessor's stale per-tier
        allocation, risking a priority inversion).
        """
        self.context.schedule_instant_task(self._try_form_holon())
        self.context.schedule_instant_task(self._broadcast_recycle())

    async def _on_failure_notice(self, message: FailureNotice, _meta: dict) -> None:
        """Locality-respecting L2 re-cycle trigger.  Kick a prompt L2
        rebalance so the post-failure component re-allocates under the
        re-elected coordinator.

        The notice is TTL-bounded, sector-tagged gossip originated by
        ``ProblemDetector`` at the failed branch's endpoints, so only
        communities physically reached by the propagation react (we do
        NOT subscribe to the global ``BranchFailureEvent``).  Without
        this, the per-component ADMM re-runs only indirectly via a later
        ``_on_member_finished``, risking a stale allocation / priority
        inversion.

        Fires for heat too: although the heat L1 negotiator ignores the
        notice (heat setpoints are temperature-driven), L2 component
        membership and coordinator are topology concerns that must react
        regardless of sector.  ``_maybe_schedule_rebalance``'s leader gate
        + min-gap throttle keep it cheap and collapse failure bursts.
        """
        await self._broadcast_recycle()

    async def _broadcast_recycle(self) -> None:
        """L2 escalation: tell every peer in the active component to run a
        fresh waterfall, then kick our own.

        The ``FailureNotice`` propagation is TTL-bounded and may reach
        only a few agents near the failure; fanning out across the
        component-peer mesh ensures the re-elected coordinator (possibly
        many hops away) and every leader owning part of the component
        re-collect and re-report, so the allocation covers a complete
        actor set.

        Single-hop: peers receive ``from_member=False`` and only rebalance
        (no re-broadcast), bounding fan-out.  Only group leaders broadcast.
        """
        if topology_characteristic(self, tid="groups") != "leader":
            return
        peers = self._resolve_component_peer_addrs()
        msg = L2RecycleEscalation(sector=self.sector, from_member=False)
        for aid, addr in peers.items():
            if aid == self.context.aid:
                continue
            await self.context.send_message(msg, receiver_addr=addr)
        self._maybe_schedule_rebalance()

    async def _handle_l2_recycle(
        self, message: L2RecycleEscalation, _meta: dict
    ) -> None:
        """Inbound L2 recycle escalation.

        From a member (``from_member=True``): re-broadcast to the whole
        component so all peers re-waterfall.  From a peer leader
        (``from_member=False``): just re-collect and re-report (single-hop,
        no flooding).
        """
        if topology_characteristic(self, tid="groups") != "leader":
            return
        if message.from_member:
            await self._broadcast_recycle()
        else:
            self._maybe_schedule_rebalance()

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

        # Symmetry-breaking: only the lex-smallest aid initiates, so a
        # clique of same-sector leaders doesn't all accept each other's
        # competing requests and end up leaderless.  The rest wait for
        # the join request.
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
            # Every candidate rejected; retry now (a busy rejecter may
            # have since finished its own formation) rather than waiting
            # for the watchdog.
            self.context.schedule_instant_task(self._try_form_holon())
            return

        # Record the resolved member set (initiator + acceptors) so
        # ``_try_rebalance`` targets only actual holon peers.
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
        self._record_event("holon_formed", f"members={len(accepted_addrs) + 1}")
        # Rebalance immediately after the holon forms so the ADMM gets at
        # least one shot while the post-failure deficit is still present,
        # rather than waiting for the periodic loop.
        self.context.schedule_instant_task(self._try_rebalance())

    # ------------------------------------------------------------------
    # Inter-group coordination via DRO ADMM
    # ------------------------------------------------------------------

    def _live_members(self, members: list[Any]) -> list[Any]:
        """Return the subset of ``members`` the :class:`LivePeerFilter`
        considers reachable.  Passthrough when no filter is wired
        (static-topology mode).
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

    def _log_idle_once(self, reason: str) -> None:
        """One-shot info log for a recurring idle condition (per-reason)."""
        if reason in self._idle_logged:
            return
        self._idle_logged.add(reason)
        logger.info(
            "[%s] holon rebalance idle: %s (sector=%s)",
            self.context.aid, reason, self.sector.value,
        )

    def _record_event(self, kind: str, detail: str) -> None:
        """Record a diagnostic event with the standard
        ``(t, kind, aid, sector, detail)`` shape.
        """
        record_event(
            t=self.context.current_timestamp,
            kind=kind,
            aid=self.context.aid,
            sector=self.sector.value,
            detail=detail,
        )

    def _resolve_holon_members(self) -> list[Any]:
        """Return holon member addresses, preferring the formation-time
        list and falling back to the ``"holons"`` topology neighbours
        ([] if the topology isn't wired).
        """
        if self._holon_member_addrs:
            return list(self._holon_member_addrs)
        try:
            return topology_neighbors(self, tid="holons")
        except Exception:
            return []

    async def _try_rebalance(self) -> None:
        """Collect flex for the next L2 ADMM round.

        * ``"component"`` (default): every group leader collects its OWN
          community's flex via a single ``AskForAvailableFlex`` to self;
          the aggregated answer drives ``_run_component_scoped_admm``,
          which pushes a ``ComponentAdmmReport`` to the coordinator.
          Holon membership is irrelevant — leaders outside any holon
          still participate.
        * ``"holon"``/``"sector"`` (legacy): only holon-leaders collect
          flex from their members and run the per-holon ADMM.

        Fired periodically (slow heartbeat) and reactively; throttled by
        ``rebalance_min_gap_s``.
        """
        now = self.context.current_timestamp
        if (now - self._last_rebalance_t) < self.rebalance_min_gap_s:
            return
        if self._rebalance_active:
            logger.debug("[%s] rebalance skipped: active", self.context.aid)
            return
        # Watchdog skip: no reactive trigger since the last rebalance ⇒
        # input unchanged ⇒ round would re-derive the same allocation.
        # Reactive callers set ``_rebalance_dirty`` first, so this only
        # short-circuits the slow periodic invocation.
        if not self._rebalance_dirty:
            logger.debug(
                "[%s] rebalance skipped: no trigger since last run",
                self.context.aid,
            )
            return

        if self.admm_scope == "component":
            # Component-scope: every group leader participates (no holon-
            # membership gate).  Each leader's community = one
            # ComponentAdmmReport = one ADMM actor, so coverage matches
            # the active subgraph.
            if topology_characteristic(self, tid="groups") != "leader":
                return
            # L2 runs in parallel with L3 (no defer): L3 decides cross-
            # sector flows via CP setpoints; the leader's next flex
            # collection reflects the post-CP state and L2 refines per-
            # sector per-tier service fractions.
            self._rebalance_active = True
            self._last_rebalance_t = now
            # Clear dirty now that we're committed; a trigger arriving
            # during execution sets it back so the next watchdog tick
            # fires.
            self._rebalance_dirty = False
            self._flex_answers = []
            self._flex_answer_senders = []
            # Single self-ask: this agent's EnergyBalanceNegotiator
            # aggregates the whole community's flex into one answer.
            self._flex_expected = 1
            await self.context.send_message(
                AskForAvailableFlex(include_connectors=False),
                receiver_addr=self.context.addr,
            )
            base = _FLEX_TIMEOUT_BASE_S.get(
                self.sector, _FLEX_TIMEOUT_DEFAULT_S
            )
            timeout = base + _FLEX_TIMEOUT_PER_MEMBER_S
            deadline = now + timeout
            self.context.schedule_timestamp_task(
                self._flex_collection_timeout(), timestamp=deadline
            )
            return

        # Legacy per-holon (and deprecated sector) path: only holon
        # leaders gather flex from their holon members.
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None:
            self._log_idle_once("no holon assigned")
            return
        if assignment.parent_addr is not None:
            self._log_idle_once("not leader")
            return

        # Prefer the resolved member list; fall back to topology
        # neighbours only if formation didn't track addresses.  Targeting
        # members directly keeps ``_flex_expected`` proportional to chunk
        # size, not clique size.
        if self._holon_member_addrs:
            members = list(self._holon_member_addrs)
        else:
            try:
                members = topology_neighbors(self, tid="holons")
            except Exception:
                return
        # Drop members the sibling ``DynamicHolonRole`` classifies as
        # physically unreachable (no-op when no filter is wired).
        members = self._live_members(members)
        if not members:
            self._log_idle_once("no neighbours")
            return

        self._rebalance_active = True
        self._last_rebalance_t = self.context.current_timestamp
        # Clear dirty now that a run is committed; triggers during
        # execution set it again.
        self._rebalance_dirty = False
        self._flex_answers = []
        self._flex_answer_senders = []
        # The leader contributes its own group's flex too, else ADMM is a
        # single-actor problem and bails out early.
        self._flex_expected = len(members) + 1

        logger.debug(
            "[%s] holon rebalance: asking %d members (+self) for flex",
            self.context.aid,
            len(members),
        )
        msg = AskForAvailableFlex(include_connectors=False)
        await self.context.send_message(msg, receiver_addr=self.context.addr)
        for addr in members:
            await self.context.send_message(msg, receiver_addr=addr)

        # Timeout: if not all answers arrive, run ADMM with whatever we
        # have (≥2) or release the lock for the next cycle.  Adaptive:
        # per-sector base + per-member scaling.
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

    # Min cross-sector flow shift in a single CP commit before the holon
    # rebalances.  Above CP regulation noise, below ``admm_abs_tol``.
    _CP_PREDICATE_DEAD_BAND_MW: float = 1e-3
    _CP_PREDICATE_MIN_GAP_S: float = 1.0

    async def _handle_cp_setpoint(
        self, message: CPSetpoint, meta: dict
    ) -> None:
        """Direct L3 -> L2 trigger (channel/decision pattern).

        Updates the per-publisher CP-setpoint memory, skips stale
        repeats by version, and (if the predicate accepts) schedules a
        rebalance so the next round's flex collection reflects the new
        cross-sector reality.
        """
        # Only the holon leader acts; members would double-trigger.
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

    async def _handle_coalition_constraint(
        self, message: CoalitionConstraint, meta: dict
    ) -> None:
        """Direct L2.5 -> L2 trigger.

        A fresh ``CoalitionConstraint`` carries per-tier service
        fractions the L2.5 coalition decided we should honour.  Re-run
        L2 now to re-merge the local ADMM result with the constraint
        store and re-dispatch to members, instead of waiting for the
        slow heartbeat to consult the store.

        Only the holon leader acts; the ``_maybe_schedule_rebalance``
        throttle prevents a flood of rapid coalitions.
        """
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return
        self._maybe_schedule_rebalance()

    async def _handle_l3_wakeup(
        self, message: L3RebalanceWakeup, meta: dict
    ) -> None:
        """S2 — L3 dispatched new CP setpoints for this sector.  Mark the
        L2 path dirty and let the throttle/scheduler decide when to
        re-fire.  Gates on group-leader membership in
        ``_maybe_schedule_rebalance``.
        """
        self._maybe_schedule_rebalance()

    async def _handle_local_gen_request(
        self, message: LocalGenerationRequest, meta: dict
    ) -> None:
        """L1 stall escalation handler.

        A member leader broadcast ``LocalGenerationRequest`` because its
        gossip converged with an unresolved residual.  L2 responds by
        (1) triggering an early rebalance (the holon ADMM may absorb the
        residual before any local DG ramps) and (2) approving the
        fallback for whatever L2 cannot cover.

        Every holon neighbour runs this handler; to avoid N approvals,
        only the lex-smallest co-recipient replies.  All peers trigger
        the rebalance (throttled by ``_maybe_schedule_rebalance``).
        """
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        try:
            co_recipients = [
                a for a in topology_neighbors(self, tid="holons")
                if a.aid != sender.aid
            ]
        except KeyError:
            co_recipients = []

        self._maybe_schedule_rebalance()

        if any(a.aid < self.context.aid for a in co_recipients):
            return

        approval = LocalGenerationApproval(
            sector=message.sector,
            residual_deficit=float(message.residual_deficit),
        )
        await self.context.send_message(approval, receiver_addr=sender)

    def _maybe_schedule_rebalance(self) -> None:
        """Shared throttle + gate for the reactive paths; a fast pre-filter
        that avoids scheduling a no-op task (``_try_rebalance`` re-checks
        the gates anyway).

        Gates by ``admm_scope``: ``"component"`` ⇒ any group-topology
        leader; ``"holon"``/``"sector"`` ⇒ only holon-leaders.
        """
        if self.admm_scope == "component":
            if topology_characteristic(self, tid="groups") != "leader":
                return
        else:
            assignment = self.context.get_or_create_model(HolonicAssignment)
            if assignment.holon_id is None or assignment.parent_addr is not None:
                return
        # Mark dirty before the throttle below, so a throttled trigger
        # still counts as dirty and the watchdog picks it up next tick.
        self._rebalance_dirty = True
        if self._rebalance_active:
            return
        now = self.context.current_timestamp
        gap_left = (self._last_rebalance_t + self.rebalance_min_gap_s) - now
        if gap_left > 0:
            # Throttled: schedule a single deferred retry at gap-expiry so
            # the work runs as soon as the fuse clears, not at the slow
            # watchdog tick.
            if not self._rebalance_retry_pending:
                self._rebalance_retry_pending = True
                self.context.schedule_timestamp_task(
                    self._deferred_rebalance(), timestamp=now + gap_left
                )
            return
        self.context.schedule_instant_task(self._try_rebalance())

    async def _deferred_rebalance(self) -> None:
        """Fire a throttled rebalance once its ``rebalance_min_gap_s``
        fuse has cleared.  ``_try_rebalance`` re-checks the gap, the
        ``_rebalance_active`` guard and the dirty flag, so this is a
        no-op if the state was already resolved by an intervening round.
        """
        self._rebalance_retry_pending = False
        await self._try_rebalance()

    async def _on_member_finished(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        """Message path: a holon-peer's group finished its balance gossip
        — try an inter-group rebalance to spread the post-gossip residual.

        The finishing leader broadcasts ``NegotiationFinishedEvent`` over
        the ``holons`` topology; the receiving holon leader schedules
        ADMM.  Throttled by ``rebalance_min_gap_s`` to break the
        holon→member→finished→holon feedback loop.
        """
        self._maybe_schedule_rebalance()

    def _on_member_finished_local(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        """Local path: when the holon leader is itself the gossip
        originator, its own ``EnergyBalanceNegotiator`` emits the event
        via ``context.emit_event``, which travels on a different bus than
        ``send_message`` (so the message subscription misses it).  Same
        throttle as ``_on_member_finished``.
        """
        if event.sector != self.sector:
            return
        self._maybe_schedule_rebalance()


    async def _handle_flex_answer(
        self, message: AvailableFlexAnswer, meta: dict
    ) -> None:
        if not self._rebalance_active:
            return
        # Ignore answers from non-members: ``_handle_ask_flex`` answers
        # any sender, so a stray reply could otherwise inflate the count.
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
        """Dispatch to the legacy per-sector ADMM, the tier-stratified
        supply-priority ADMM, or the component-scoped path by scope/mode.

        The tier-stratified path replaces the scalar per-(member, sector)
        override — which loses priority intent in the L2→L1 handoff —
        with a 2-D ``targets[sector][tier]`` allocation.  It falls back to
        the legacy path when the answers show no meaningful per-tier
        deficit, so the holon still gets flow-redistribution.

        ``"component"``: the leader aggregates its flex into one
        ``ComponentAdmmReport`` and pushes it to the elected coordinator
        (possibly itself), which runs the per-component ADMM and
        dispatches a uniform ``ComponentAllocation`` to every leader on
        the active subgraph.
        """
        if self.admm_scope == "component":
            await self._run_component_scoped_admm()
            return
        if not self.enable_tier_stratified_admm:
            await self._run_legacy_per_sector_admm()
            return
        # Supply-priority ADMM (per-holon).  Fires whenever the holon has
        # any supply to allocate; priority weighting binds once
        # supply < demand.
        if self._supply_priority_has_anything_to_do():
            await self._run_supply_priority_admm()
            return
        await self._run_legacy_per_sector_admm()

    def _supply_priority_has_anything_to_do(self) -> bool:
        """True iff queued answers contain both per-tier demand and
        supply — nothing to allocate across priorities otherwise.
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
    async def _run_legacy_per_sector_admm(self) -> None:
        """Run ADMM sharing optimisation across member groups.

        Each group's flex is a multi-dimensional ADMM actor, one
        dimension per sector, so resources balance across sectors at once
        (e.g. gas surplus covering heat deficit via CHP).  The sharing
        target ``T`` is per-sector total imbalance across member groups.

        Priority-aware: each group reports demand by priority tier; a
        waterfall computes each group's ideal share (high tiers across
        all groups before any low tier), and the ADMM ``S`` (linear cost)
        pulls each actor toward its priority-weighted share.
        """

        if not self._rebalance_active:
            return
        answers = self._flex_answers[:]
        senders = self._flex_answer_senders[:]
        self._flex_answers = []
        self._flex_answer_senders = []
        self._flex_expected = 0
        self._rebalance_active = False  # release lock early; prevents timeout re-entry

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

        # Feasibility cap on T per dimension: when |T[i]| exceeds the sum
        # of available actor budgets, the sharing-distance term plateaus
        # at the structural gap and the library spuriously logs "max
        # iterations".  Bound |T| by the budget envelope (sign preserved).
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

        # Priority-weighted S: waterfall shares = how much each group
        # should receive under strict priority ordering.  Budget =
        # surplus (negative-balance groups) + flex headroom across all
        # groups; the flex term keeps the budget positive (and per-tier
        # shares meaningful) in a pure-deficit holon with no surplus.
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
            # Priority allocation disabled — distribute the budget
            # uniformly across answering groups (balance-only ablation).
            even = total_available / max(1, len(answers))
            priority_shares = [even for _ in answers]

        for idx, answer in enumerate(answers):
            lb, ub = group_bounds[idx]
            C = np.zeros((0, n_dims))
            d = np.zeros(0)
            # S is the local-QP linear cost (negative attracts).  Set it
            # proportional to the group's priority-weighted share so ADMM
            # steers resources toward high-priority unserved demand.
            S = np.zeros(n_dims)
            if priority_shares[idx] > 1e-9:
                # Normalise by the total target for stability alongside
                # the rho term, then split the pull across dimensions in
                # proportion to the group's per-dimension balance.
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
        # Iter cap / tolerance from config.  Defaults (50 @ 1e-3) relaxed
        # from the package's 1000 / 1e-4 so concurrent holon ADMMs don't
        # block discrete-time progress, while still converging.
        coordinator.max_iters = int(self.admm_max_iters)
        coordinator.abs_tol = float(self.admm_abs_tol)
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
            self._record_event(
                "holon_admm_result",
                f"sectors={all_sectors} T={total_T.tolist()}",
            )
        except Exception as exc:
            logger.error("[%s] holon ADMM failed: %s", self.context.aid, exc)
            self._record_event("holon_admm_failed", str(exc))
            # On failure still trigger intra-group gossip so members can
            # rebalance locally without inter-group redistribution.

        # Trigger intra-group rebalancing, routing each ADMM allocation as
        # the gossip override target so the result isn't recomputed away.
        # ``actors[idx].x`` is a vector over ``all_sectors``; pick the
        # member's own sector and negate (target = -allocation, since the
        # member must absorb that imbalance from the holon's pool).
        # Members not mapped to an actor fall back to local recompute.
        sender_to_actor: dict[str, tuple[Any, AvailableFlexAnswer]] = {}
        for sender, answer, actor in zip(senders, answers, actors):
            sender_to_actor[str(sender)] = (actor, answer)

        triggers = self._resolve_holon_members()
        # Carry per-member overrides on the ``HolonAllocation`` decision
        # AND push the legacy ``StartBalanceNegotiation`` so L1's existing
        # override path keeps working; both go to the same members.
        allocation_targets: dict[str, float] = {}
        for addr in triggers:
            entry = sender_to_actor.get(str(addr))
            override: float | None = None
            if entry is not None:
                actor_obj, answer = entry
                # Index actor_obj.x by the member's natural sector.
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

        # Publish HolonAllocation to CP connectors so L3 (CP ADMM) reacts
        # to the cross-sector setpoint shift directly, skipping the
        # L2->L1->gossip-finished->L3 detour.  Uses the ``tid="groups"``
        # cross-topology link.
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
    async def _run_supply_priority_admm(self) -> None:
        """Supply-priority ADMM.

        Differs from the demand-side path:
        1. ``T`` is total demand per (sector, tier) cell across the holon
           (the ideal-served target), not the per-cell deficit; the L1
           distance penalty is weighted by tier priority.
        2. Each actor's per-cell ``ub`` and coupling ``Σ x ≤ supply_g``
           reflect generator capacity, not local demand — so a group
           with supply but no local tier-X demand can contribute to
           holon-wide tier-X service (the LP routes the freed power).

        Output: per-(sector, tier) service fractions sent to each leader,
        applied uniformly to local loads at that tier — yielding a
        consistent shed-low / serve-high pattern holon-wide regardless of
        which group physically holds the supply.
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

        actor_supplies = [a.supply_by_sector or {} for a in answers]
        actor_demands = [a.demand_by_sector_priority or {} for a in answers]

        # Active cells across the holon.  A sector with supply but no demand
        # has no cells and bypasses the ADMM (already accounted for in the
        # leader's reply); no demand anywhere ⇒ fall back to the legacy path.
        sectors, tiers, total_demand = extract_demand_sectors_tiers(actor_demands)
        if not sectors or not tiers or total_demand < 1e-6:
            await self._run_legacy_per_sector_admm()
            return

        # F6 deliverability caps.  When member home node ids and a
        # topology mirror are available, compute per-actor
        # ``{(sector, tier): cap}`` overrides so the ADMM doesn't commit
        # supply an actor cannot route under the current branch-active
        # mask.  Conservative-by-node: cap each cell at demand co-located
        # at nodes reachable from the actor's home node (demand collapsed
        # to leader nodes, since L2 only knows leader nodes).  Unreachable
        # leader ⇒ all caps 0; reachable ⇒ uncapped (None), so the
        # per-actor coupling is the only binding constraint.
        actor_ub_overrides: list[dict[tuple[str, int], float] | None] | None = None
        if (
            self._topology_mirror is not None
            and self._leader_node_ids
        ):
            try:
                actor_node_ids: list[Any | None] = []
                actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]] = []
                for sender, answer in zip(senders, answers):
                    leader_aid = getattr(sender, "aid", str(sender))
                    node_id = self._leader_node_ids.get(leader_aid)
                    actor_node_ids.append(node_id)
                    # Map this leader's tier-aggregated demand onto its
                    # home node: a reachable leader's demand contributes
                    # to every other leader's reachable-cap; an
                    # unreachable one contributes nothing.
                    per_tier: dict[int, dict[Any, float]] = {}
                    if node_id is not None:
                        for sec, tier_map in (
                            answer.demand_by_sector_priority or {}
                        ).items():
                            if sec not in sectors:
                                continue
                            for tier, dem in tier_map.items():
                                # Bind the inner dict to a local first:
                                # Python evaluates the RHS before the LHS
                                # subscript, so a combined
                                # ``setdefault(k,{})[n] = ...[k].get(...)``
                                # would KeyError on the first ``k``.
                                inner = per_tier.setdefault(int(tier), {})
                                inner[node_id] = (
                                    inner.get(node_id, 0.0) + float(dem)
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
                    "failed (%s: %s) — falling back to raw supply",
                    self.context.aid, type(exc).__name__, exc,
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
            self._record_event("holon_admm_failed", f"supply_priority: {exc}")
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
        self._record_event(
            "holon_admm_result",
            f"supply_priority sectors={sectors} tiers={tiers} fractions={service_fraction}",
        )
        self._record_event(
            "holon_priority_allocation",
            str({
                f"{sec}:tier{tier}": {
                    "T": round(float(total_T[_flat_idx(sec, tier)]), 6),
                    "weight": float(priorities[_flat_idx(sec, tier)]),
                    "sum_x": round(float(sum_x_per_cell[_flat_idx(sec, tier)]), 6),
                    "service_frac": round(service_fraction[sec][tier], 4),
                }
                for sec in sectors for tier in tiers
            }),
        )

        # Coalition merge: an active coalition fraction for a (sector,
        # tier) cell overrides L2's per-cell result; L2 keeps cells the
        # coalition didn't touch.  Without it, last-write-wins would let
        # each L2 round reset the regulation and undo the redistribution.
        if self._coalition_constraint_store is not None:
            now = float(self.context.current_timestamp)
            service_fraction = self._coalition_constraint_store.merge_into(
                service_fraction, self.sector, now,
            )

        # Send the same (holon-global) service fraction map to every
        # member leader, which applies it locally (L1 honour path).
        triggers = self._resolve_holon_members()
        for addr in triggers:
            await self.context.send_message(
                StartBalanceNegotiation(
                    service_fraction_by_sector_priority=service_fraction,
                ),
                receiver_addr=addr,
            )

    # ------------------------------------------------------------------
    # Per-(sector, active-component) L2 ADMM (admm_scope="component")
    # ------------------------------------------------------------------
    # One ADMM scoped to each active connected component of the sector
    # graph; each community leader is one ADMM actor (the holon
    # abstraction is unused here).
    #
    # Per round:
    #   1. Each leader collects its community's flex (self-sent
    #      ``AskForAvailableFlex`` → one aggregated answer).
    #   2. Pushes a ``ComponentAdmmReport`` to the component coordinator
    #      (lex-smallest leader aid mutually reachable on the active
    #      subgraph).
    #   3. The coordinator buffers reports keyed by leader_aid, debounces
    #      a burst into one supply-priority ADMM solve over N actors, and
    #      dispatches a ``ComponentAllocation`` to every component leader.
    #   4. Each leader applies the per-tier fractions to its own community
    #      members (self-sent ``StartBalanceNegotiation``).
    #
    # Invariants:
    #   * Every load at the same tier in the same (sector, component) is
    #     served at the same fraction — the per-component solve produces
    #     one per-tier fraction, so cross-leader inversions cannot arise.
    #   * Every component leader participates in both input and output.
    #   * A failure splitting a sector re-elects two coordinators that
    #     decide independently for their halves.

    def _resolve_sector_peer_addrs(self) -> dict[str, Any]:
        """Return ``{leader_aid: leader_addr}`` for every same-sector
        leader on the ``holon_summary_<sector>`` topology, including self.
        Unfiltered baseline for ``_resolve_component_peer_addrs``.
        """
        addrs: dict[str, Any] = {self.context.aid: self.context.addr}
        try:
            peers = list(
                topology_neighbors(self, tid=f"holon_summary_{self.sector.value}")
            )
        except Exception:
            return addrs
        for addr in peers:
            aid = getattr(addr, "aid", None)
            if aid is None:
                aid = str(addr)
            addrs[str(aid)] = addr
        return addrs

    def _resolve_component_peer_addrs(self) -> dict[str, Any]:
        """Return ``{leader_aid: leader_addr}`` for every same-sector
        leader in this leader's connected component (mutually reachable
        on the active subgraph).  Falls back to the unfiltered sector
        peer set when the topology mirror or own node id is unavailable.

        The ``holon_summary_<sector>`` mesh also carries CP/branch agents
        (L3 readers) that host no ``HolonicCommunityRole``; a report
        routed to one as lex-smallest "peer" is dropped and the solve
        never runs.  So filter the peer set to known leader aids (self
        always included) to keep CPs out of the coordinator election;
        empty ``_leader_node_ids`` ⇒ unfiltered fallback.
        """
        sector_peers = self._resolve_sector_peer_addrs()
        leader_aids = set(self._leader_node_ids)
        if leader_aids:
            sector_peers = {
                aid: addr for aid, addr in sector_peers.items()
                if aid == self.context.aid or aid in leader_aids
            }
        mirror = self._topology_mirror
        my_node = self._my_node_id
        if mirror is None or my_node is None:
            return sector_peers
        try:
            reachable = mirror.reachable_from(my_node, sector=self.sector)
        except Exception:
            return sector_peers
        out: dict[str, Any] = {}
        for aid, addr in sector_peers.items():
            if aid == self.context.aid:
                # Always include self — a leader is in its own component
                # even if ``_leader_node_ids`` lacks an entry.
                out[aid] = addr
                continue
            node_id = self._leader_node_ids.get(aid)
            if node_id is None or node_id in reachable:
                out[aid] = addr
        return out

    def _component_coordinator_aid(self) -> str | None:
        """Lex-smallest aid among current component peers — this
        component's coordinator.  None only when even self is missing
        (caller falls back to the per-holon path).
        """
        peers = self._resolve_component_peer_addrs()
        if not peers:
            return None
        return min(peers.keys())

    def _multi_sector_l3_active(self) -> bool:
        """True iff this leader sits in a multi-sector component
        containing at least one CP agent (so an L3 coordinator owns the
        decision and L2 defers).

        Traverses the joint multi-sector subgraph via
        ``reachable_from(..., allow_cp_bridges=True)`` and intersects
        with the known CP host node ids; empty ⇒ no CP ⇒ L2 runs locally.
        """
        if not self._cp_node_ids:
            return False
        mirror = self._topology_mirror
        my_node = self._my_node_id
        if mirror is None or my_node is None:
            return False
        try:
            reachable = mirror.reachable_from(
                my_node, sector=None, allow_cp_bridges=True,
            )
        except Exception:
            return False
        return any(n in reachable for n in self._cp_node_ids)

    async def _run_component_scoped_admm(self) -> None:
        """Component-scoped variant of ``_run_supply_priority_admm``.

        The leader has already collected its community's flex.  Branch:
        * coordinator → buffer own report + schedule a debounced solve.
        * non-coordinator → push a ``ComponentAdmmReport`` and wait for
          the returning ``ComponentAllocation``.

        Either way the per-holon ADMM does not run; the coordinator's
        result arrives via ``_handle_component_allocation``.
        ``_flex_answers`` is drained so a follow-up trigger doesn't
        re-fire on the stale buffer.
        """
        if not self._rebalance_active:
            return
        answers = self._flex_answers[:]
        # Drain.
        self._flex_answers = []
        self._flex_answer_senders = []
        self._flex_expected = 0
        self._rebalance_active = False

        if not answers:
            return

        supply, demand, served = aggregate_holon_flex(answers)
        # Report if there's any supply OR demand: a demand-only community
        # must contribute to the sector-wide T vector, a supply-only one
        # to the supply pool.  (Using ``and`` would silently exclude
        # load-only communities, opening a dispatch coverage gap.)
        any_supply = any(v > 1e-9 for v in supply.values())
        any_demand = any(
            mw > 1e-9 for tmap in demand.values() for mw in tmap.values()
        )
        if not (any_supply or any_demand):
            return

        coord_aid = self._component_coordinator_aid()
        if coord_aid is None:
            # No peer topology; degenerate to the per-holon path.
            self._flex_answers = answers  # restore for the fallback
            self._rebalance_active = True
            await self._run_supply_priority_admm()
            return

        round_id = f"r{self._component_round_counter}"
        self._component_round_counter += 1
        now = float(self.context.current_timestamp)
        leader_aid = self.context.aid

        report = ComponentAdmmReport(
            publisher=leader_aid,
            version=self._version.next(),
            timestamp_s=now,
            round_id=round_id,
            sector=self.sector,
            leader_aid=leader_aid,
            supply_by_sector=supply,
            demand_by_sector_priority=demand,
            served_by_sector_priority=served,
            # Implicit ACK: echo the latest applied ComponentAllocation
            # version so the coordinator can detect a missed dispatch.
            last_applied_allocation_version=self._last_applied_allocation_version,
        )

        if coord_aid == leader_aid:
            # I'm the coordinator — buffer my report, trigger the solve.
            self._component_report_buffer[leader_aid] = (round_id, report)
            await self._maybe_run_component_admm(reason="self_report")
            return

        # Push to the coordinator.  No reply timeout: a silent drop is
        # retried by the next trigger (idempotent — buffer keyed by
        # leader_aid).
        peers = self._resolve_component_peer_addrs()
        coord_addr = peers.get(coord_aid)
        if coord_addr is None:
            # Aid known but address unresolved — fall back to the
            # per-holon path so the round still produces an allocation.
            self._flex_answers = answers
            self._rebalance_active = True
            await self._run_supply_priority_admm()
            return
        await self.context.send_message(report, receiver_addr=coord_addr)
        self._record_event(
            "component_report_sent",
            f"coord={coord_aid} leader={leader_aid} round={round_id} "
            f"supply={sum(supply.values()):.4f} demand_tiers={len(demand)}",
        )

    async def _handle_component_admm_report(
        self, message: ComponentAdmmReport, meta: dict
    ) -> None:
        """Coordinator-side: buffer a peer leader's report and schedule
        the debounced ADMM solve.  Non-coordinators drop the message.

        Buffer keyed by ``leader_aid`` — a later report overwrites
        (freshest flex wins).  Drops reports from peers no longer in our
        component (e.g. cut off after the sender pushed) to keep the
        actor set consistent with the active subgraph.
        """
        if self.admm_scope != "component":
            return
        if self._component_coordinator_aid() != self.context.aid:
            return
        # Only buffer reports from leaders still in our active component.
        component_peers = self._resolve_component_peer_addrs()
        if message.leader_aid not in component_peers:
            return
        self._component_report_buffer[message.leader_aid] = (
            message.round_id, message
        )
        # Packet-loss recovery: if the sender's echoed applied-version is
        # behind our latest dispatch, re-send the stashed allocation to
        # just this peer (idempotent at apply time).  getattr-guarded for
        # legacy reports lacking the field.
        try:
            await self._resend_allocation_if_stale(
                message.leader_aid,
                component_peers.get(message.leader_aid),
                int(getattr(message, "last_applied_allocation_version", -1)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[%s] resend-if-stale failed for leader=%s: %s",
                self.context.aid, message.leader_aid, exc,
            )
        await self._maybe_run_component_admm(reason="peer_report")

    async def _resend_allocation_if_stale(
        self,
        leader_aid: str,
        leader_addr: Any | None,
        applied_version: int,
    ) -> None:
        """Re-send the latest ``ComponentAllocation`` to ``leader_addr``
        when its echoed applied-version is behind
        ``self._allocation_version_counter`` (i.e. the original dispatch
        was lost — packet loss).

        Idempotent: the leaf ignores an allocation whose ``version`` ≤
        already-applied, so a benign duplicate is a no-op.  Self-skip:
        the coordinator's own seat was already served by the dispatch
        loop.  Records a ``component_alloc_resent`` diagnostic event.
        """
        if leader_addr is None:
            return
        if leader_aid == self.context.aid:
            return
        if self._last_dispatched_allocation is None:
            return
        if applied_version >= self._allocation_version_counter:
            return
        await self.context.send_message(
            self._last_dispatched_allocation, receiver_addr=leader_addr,
        )
        self._record_event(
            "component_alloc_resent",
            f"target={leader_aid} applied_version={applied_version} "
            f"latest_version={self._allocation_version_counter}",
        )

    async def _maybe_run_component_admm(self, *, reason: str) -> None:
        """Debounce + dispatch: collapse a burst of reports into one ADMM
        solve.  ``_component_dispatch_pending`` lets the first arrival
        own the solve, giving one solve per burst with bounded latency.
        """
        if self._component_dispatch_pending:
            return
        self._component_dispatch_pending = True
        self.context.schedule_instant_task(
            self._run_component_admm_now(reason=reason),
        )

    async def _run_component_admm_now(self, *, reason: str) -> None:
        """Run the per-component ADMM over the current report buffer and
        dispatch the per-tier service fractions to every component leader
        (including self).  Clears ``_component_dispatch_pending`` on
        return.  The buffer is not drained — late reports overwrite and
        the next solve picks up the freshest entries.
        """
        try:
            await self._run_component_admm_now_inner(reason=reason)
        finally:
            self._component_dispatch_pending = False
            self._last_component_dispatch_t = float(self.context.current_timestamp)

    async def _run_component_admm_now_inner(self, *, reason: str) -> None:
        if not self._component_report_buffer:
            return
        # Only include leaders still in this coordinator's component view;
        # a disconnected leader's report stays buffered but the round
        # skips it.
        component_peers = self._resolve_component_peer_addrs()
        leader_aids = sorted(
            aid for aid in self._component_report_buffer
            if aid in component_peers
        )
        if not leader_aids:
            return
        reports = [self._component_report_buffer[a][1] for a in leader_aids]

        actor_supplies = [r.supply_by_sector for r in reports]
        actor_demands = [r.demand_by_sector_priority for r in reports]

        sectors, tiers, total_demand = extract_demand_sectors_tiers(actor_demands)
        if not sectors or not tiers or total_demand < 1e-6:
            return

        try:
            service_fraction, _per_actor_x, meta = await allocate_supply_priority(
                sectors=sectors,
                tiers=tiers,
                actor_supplies=actor_supplies,
                actor_demands=actor_demands,
                actor_ub_overrides=None,
                priority_tiers=self.priority_tiers,
                max_iters=int(self.admm_max_iters),
                abs_tol=float(self.admm_abs_tol),
                enable_priority_weighting=self.enable_priority_allocation,
            )
        except Exception as exc:
            logger.error(
                "[%s] component-scope ADMM failed: %s", self.context.aid, exc,
            )
            self._record_event("holon_admm_failed", f"component_scope: {exc}")
            return

        # No sub-tolerance noise scrub: clamping near-zero fractions to
        # exact 0 locks them in via the per-load cooldown gate (the next
        # complete round's correct value lands inside the cooldown and is
        # suppressed).  The PI claim's 1e-3 tolerance absorbs the noise.

        # Complete + monotone per-tier vector.  A round may solve over only
        # a subset of tiers, so dispatching just those lets a lower tier
        # keep a stale higher fraction from an earlier round (a priority
        # inversion).  Fold the previous dispatch's tiers in (fresh values
        # win), then clamp the vector non-increasing in tier number so
        # tier 1 ≥ tier 2 ≥ … holds by construction over the tiers this
        # coordinator allocated.  Bounded to this coordinator's
        # solve+history (NOT all P tiers): forcing unknown tiers down
        # over-sheds; the coordinator-handoff inversion needs the
        # sector-wide L2.5 reconciliation instead.
        sec_val = self.sector.value
        prev_own = (self._last_component_fraction or {}).get(sec_val, {})
        merged_own = dict(prev_own)
        merged_own.update(service_fraction.get(sec_val, {}))
        cap = 1.0
        for tier in sorted(t for t in merged_own if t >= 1):
            merged_own[tier] = min(merged_own[tier], cap)
            cap = merged_own[tier]
        service_fraction = {**service_fraction, sec_val: merged_own}

        self._last_component_fraction = service_fraction
        logger.info(
            "[%s] component-scope ADMM result (reason=%s): sectors=%s "
            "tiers=%s n_communities=%d fractions=%s",
            self.context.aid, reason, sectors, tiers, len(reports),
            service_fraction,
        )
        self._record_event(
            "holon_admm_result",
            f"component_scope reason={reason} sectors={sectors} tiers={tiers} "
            f"n_communities={len(reports)} fractions={service_fraction}",
        )

        # Dispatch the same ComponentAllocation to every component leader
        # (including self); each applies the fractions to its own
        # community members via ``_handle_component_allocation``.
        round_id = max(
            (self._component_report_buffer[a][0] for a in leader_aids),
            default="",
        )
        now = float(self.context.current_timestamp)
        # Bump the allocation version before building the message so leaf
        # ACK echoes line up with this dispatch.  Stash the message for
        # re-sends to stale leaders (``_resend_allocation_if_stale``).
        self._allocation_version_counter += 1
        allocation = ComponentAllocation(
            publisher=self.context.aid,
            version=self._allocation_version_counter,
            timestamp_s=now,
            round_id=round_id,
            sector=self.sector,
            service_fraction_by_tier=service_fraction.get(self.sector.value, {}),
        )
        self._last_dispatched_allocation = allocation
        for addr in component_peers.values():
            await self.context.send_message(allocation, receiver_addr=addr)

    async def _handle_component_allocation(
        self, message: ComponentAllocation, meta: dict
    ) -> None:
        """Leaf-side: apply the coordinator's per-tier service fraction
        to this leader's OWN community members (no holon hop) via the L1
        honour path.

        Every group leader for this sector handles it (not gated on
        holon membership), so communities outside any holon are covered.
        Coalition merge: an active store fraction wins per-tier.
        """
        if self.admm_scope != "component":
            return
        # Act only as a current group-topology leader (cheap drift guard).
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Rebuild a {sector: {tier: frac}} envelope so the L1 honour path
        # consumes it unchanged.
        service_fraction: dict[str, dict[int, float]] = {
            self.sector.value: dict(message.service_fraction_by_tier),
        }
        if self._coalition_constraint_store is not None:
            now = float(self.context.current_timestamp)
            service_fraction = self._coalition_constraint_store.merge_into(
                service_fraction, self.sector, now,
            )
        # Send to SELF: this agent's EnergyBalanceNegotiator applies the
        # per-tier fraction to every community member (live group
        # neighbours), covering loads outside any holon.
        await self.context.send_message(
            StartBalanceNegotiation(
                service_fraction_by_sector_priority=service_fraction,
            ),
            receiver_addr=self.context.addr,
        )
        # ACK: record the applied version so the next outgoing report
        # echoes it; the coordinator uses the echo to detect drops and
        # re-send.  A stale retransmit (version ≤ applied) is ignored.
        try:
            if int(message.version) > self._last_applied_allocation_version:
                self._last_applied_allocation_version = int(message.version)
        except (TypeError, ValueError):
            # Legacy allocation (no version): ack stays -1 so the
            # coordinator reads the report as "stale or first".
            pass
        self._record_event(
            "holon_priority_allocation",
            f"component_scope round={message.round_id} "
            f"version={getattr(message, 'version', 0)} "
            f"fractions={service_fraction}",
        )
