"""Holonic (multi-level) community formation and coordination.

L1 sector agents solve local restoration via gossip; L2 holon leaders
aggregate member-group flex and run DRO ADMM for inter-group sharing,
then drive each group to rebalance toward the per-actor allocation.
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
from scare.base.runtime.diagnostics import record_event
from scare.base.runtime.trace import optimization
from scare.base.topology.topology_mirror import LivePeerFilter
from scare.base.util import (
    clamp_tier_monotonic,
    compute_priority_weighted_shares,
)
from scare.community.deliverability import per_actor_deliverable_caps
from scare.community.holon_flex import (
    aggregate_holon_flex,
    extract_demand_sectors_tiers,
)
from scare.community.supply_priority_admm import allocate_supply_priority

logger = logging.getLogger(__name__)

# Base flex-collection timeout per sector. Heat is slow (thermal inertia),
# electricity fast, gas between.
_FLEX_TIMEOUT_BASE_S: dict[Sector, float] = {
    Sector.ELECTRICITY: 3.0,
    Sector.GAS: 8.0,
    Sector.HEAT: 15.0,
}
_FLEX_TIMEOUT_DEFAULT_S = 5.0
_FLEX_TIMEOUT_PER_MEMBER_S = 0.5  # per expected member group


DEFAULT_MAX_HOLON_SIZE: int = 4

# Two fraction maps within this per-tier tolerance count as the same
# allocation (matches the actuator dedup tolerance, so a skipped re-dispatch
# would have been all no-ops anyway).
_FRACTION_EQUAL_TOL: float = 1e-3


def _fraction_maps_equal(
    a: dict[str, dict[int, float]],
    b: dict[str, dict[int, float]],
    *,
    tol: float = _FRACTION_EQUAL_TOL,
) -> bool:
    if set(a) != set(b):
        return False
    for sec in a:
        ta, tb = a[sec], b[sec]
        if set(ta) != set(tb):
            return False
        if any(abs(ta[t] - tb[t]) > tol for t in ta):
            return False
    return True


class HolonicCommunityRole(Role):
    """Holonic formation and inter-group coordination via DRO ADMM.

    A holon leader collects ``AvailableFlexAnswer`` from members, runs an
    ADMM sharing optimisation, then triggers intra-group rebalancing.
    Attached to every agent, but only group leaders act.
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
        enable_change_only_dispatch: bool = True,
        live_member_filter: LivePeerFilter | None = None,
        coalition_constraint_store: Any = None,
        my_node_id: Any = None,
        leader_node_ids: dict[str, Any] | None = None,
        topology_mirror: Any = None,
        cp_node_ids: set[Any] | None = None,
    ) -> None:
        super().__init__()
        self.sector = sector
        # Coalition fractions (written by sibling ``HolonSummaryRole``)
        # override L2's per-tier result for the TTL window. None ⇒ no merge.
        self._coalition_constraint_store = coalition_constraint_store
        # Optional ``DynamicHolonRole`` filtering members reachable via live
        # grid edges. None ⇒ static-topology mode (all reachable).
        self._live_member_filter = live_member_filter
        self.formation_period_s = formation_period_s
        self.max_holon_size = max_holon_size
        # ADMM convergence knobs (quality vs wallclock).
        self.admm_max_iters = admm_max_iters
        self.admm_abs_tol = admm_abs_tol
        # Tier-stratified ADMM: per-(sector, tier) supply-priority allocation.
        self.enable_tier_stratified_admm = enable_tier_stratified_admm
        self.priority_tiers = priority_tiers
        # ADMM scope: "holon" (per holon members), "sector" (deprecated), or
        # "component" (default — group leaders on the same active subgraph,
        # coordinator elected per component).
        if admm_scope not in {"holon", "sector", "component"}:
            raise ValueError(
                "holon admm_scope must be 'holon', 'sector', or 'component', "
                f"got {admm_scope!r}"
            )
        self.admm_scope = admm_scope
        # False ⇒ uniform per-tier weights (no-priority ablation).
        self.enable_priority_allocation = bool(enable_priority_allocation)
        # Coordination overhaul: reactive cascade (see
        # config.enable_change_only_dispatch). On the L2 side this only bypasses
        # the rebalance_min_gap_s time-throttle; the change-detection that
        # bounds the cascade lives on the UPWARD L1→L2 edge (balance.py).
        self.enable_change_only_dispatch = bool(enable_change_only_dispatch)
        # Deliverability wiring (F6). When all three are present the supply-
        # priority ADMM caps each member's per-tier commitment at the demand
        # reachable from its home node, so L2 never allocates unroutable supply.
        self._my_node_id = my_node_id
        self._leader_node_ids: dict[str, Any] = dict(leader_node_ids or {})
        self._topology_mirror = topology_mirror
        # Node ids hosting a CP agent; the per-component path uses this to
        # detect a CP in the leader's multi-sector component. Empty ⇒ none.
        self._cp_node_ids: set[Any] = set(cp_node_ids or set())
        # ``rebalance_period_s``: slow heartbeat catching drift from timeseries
        # inputs that change without a NegotiationFinishedEvent.
        # ``rebalance_min_gap_s``: feedback-loop fuse — lets one round of member
        # gossip resolve before the next ADMM cycle (holon→gossip→finished→ADMM).
        self.rebalance_period_s = rebalance_period_s
        self.rebalance_min_gap_s = rebalance_min_gap_s
        self.flex_timeout_s = flex_timeout_s

        # Safety net for a leader that missed every trigger event.
        self.watchdog_s = watchdog_s

        # holon_id -> {sender_key: (addr, accept_or_None)}. Address kept to
        # build the resolved member list without a topology re-lookup.
        self._pending_proposals: dict[UUID, dict[str, tuple[Any, bool | None]]] = {}
        # Collected member flex answers; ``_flex_answer_senders`` holds the
        # sender per answer to route the allocation back as override target.
        self._flex_answers: list[AvailableFlexAnswer] = []
        self._flex_answer_senders: list[Any] = []
        self._flex_expected: int = 0
        # Round tag for ``_flex_collection_timeout``: a stale timeout from a
        # completed round must not release/fire a later round.
        self._flex_round_token: int = 0
        # Stamped on AskForAvailableFlex and echoed by responders; a straggler
        # from round N must not count into round N+1.
        self._flex_round_id: str = ""
        self._rebalance_active: bool = False
        # Reactive triggers within ``rebalance_min_gap_s`` of this are dropped.
        self._last_rebalance_t: float = float("-inf")
        # Watchdog no-change skip: reactive trigger sets True, a successful
        # rebalance clears it. True initially so the first tick runs.
        self._rebalance_dirty: bool = True
        # Guards a single deferred retry at gap-expiry so a throttled trigger
        # runs when the fuse clears, not at the slow watchdog tick.
        self._rebalance_retry_pending: bool = False

        # Resolved holon membership (leader side), set by ``_handle_join_answer``;
        # lets ``_flex_expected`` scale with chunk size, not clique size.
        self._holon_member_addrs: list[Any] = []
        self._holon_member_keys: set[str] = set()

        # Per-reason so each idle early-return surfaces once, not every tick.
        self._idle_logged: set[str] = set()

        # --- Channel-pattern state (L2 <-> L3 direct link) ---
        # ``_version`` advances per published ``HolonAllocation``; ``_seen_cps``
        # tracks the latest consumed ``CPSetpoint`` version per publisher.
        self._version = MonotonicVersion()
        self._seen_cps = SeenVersions()
        self._cp_setpoint_state: dict[tuple[str, str], float] = {}
        self._last_cp_predicate_fire_t: float = -1e9

        # --- Component-scoped L2 ADMM state (admm_scope="component") ---
        # The coordinator (lex-smallest reachable leader aid) buffers
        # ``ComponentAdmmReport`` per leader and runs one ADMM over N actors.
        # ``_component_dispatch_pending`` debounces a report burst into one solve.
        self._component_round_counter: int = 0
        self._component_report_buffer: dict[str, tuple[str, Any]] = {}
        self._component_dispatch_pending: bool = False
        # Latest dispatched fraction; new reports merge against it.
        self._last_component_fraction: dict[str, dict[int, float]] | None = None

        # --- ComponentAllocation versioning (packet-loss recovery) ---
        # Monotone counter per outgoing ``ComponentAllocation``. Receivers echo
        # the last version applied; the coordinator re-sends to stale receivers.
        self._allocation_version_counter: int = 0
        # Latest dispatched allocation, re-sent to stale leaders. None until first.
        self._last_dispatched_allocation: Any = None  # ComponentAllocation
        # Latest (coordinator aid, version) applied as an L2 leaf. Versions are
        # per-publisher, so a re-elected coordinator's fresh low versions must
        # not be judged against the old coordinator's high counter. Echoed in
        # each report only when the publisher matches the current coordinator.
        self._last_applied_allocation_version: int = -1
        self._last_applied_allocation_publisher: str | None = None

    def setup(self) -> None:
        # Formation is event-driven; this slow watchdog covers a leader that
        # missed every trigger.
        self.context.schedule_periodic_task(self._try_form_holon, delay=self.watchdog_s)
        # Rebalance is event-driven via the reactive handlers below plus this
        # watchdog; every cause of drift is itself an event (no drift probe).
        self.context.schedule_periodic_task(self._try_rebalance, delay=self.watchdog_s)

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
            lambda msg, meta: (
                isinstance(msg, AvailableFlexAnswer) and msg.sector == self.sector
            ),
        )
        # Reactive L2 trigger when a member group finishes gossip. Two subs:
        # ``subscribe_message`` for broadcasts from other leaders;
        # ``subscribe_event`` for this agent's own negotiator (disjoint buses).
        self.context.subscribe_message(
            self,
            _wrap(self._on_member_finished),
            lambda msg, meta: (
                isinstance(msg, NegotiationFinishedEvent) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_member_finished_local
        )
        # React to the TTL-bounded ``FailureNotice`` so a topology change re-
        # allocates the affected component under the re-elected coordinator.
        # Sector-filtered: only a same-sector failure changes connectivity.
        self.context.subscribe_message(
            self,
            _wrap(self._on_failure_notice),
            lambda msg, meta: (
                isinstance(msg, FailureNotice) and msg.sector == self.sector
            ),
        )
        # L2 recycle escalation: member relays a local failure to its leader
        # (from_member=True), or a peer leader fans it out (from_member=False).
        self.context.subscribe_message(
            self,
            _wrap(self._handle_l2_recycle),
            lambda msg, meta: (
                isinstance(msg, L2RecycleEscalation) and msg.sector == self.sector
            ),
        )
        # Direct L3 -> L2 trigger: on a CP cross-sector setpoint commit,
        # affected holons re-evaluate without the L1 gossip chain. The
        # handler's predicate decides whether to rebalance.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_cp_setpoint),
            lambda msg, meta: isinstance(msg, CPSetpoint),
        )
        # Direct L2.5 -> L2 trigger: a fresh ``CoalitionConstraint`` kicks an
        # immediate rebalance so merged fractions (ADMM ⊕ coalition overrides)
        # reach L1 this tick instead of waiting for the heartbeat.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_coalition_constraint),
            lambda msg, meta: (
                isinstance(msg, CoalitionConstraint) and msg.sector == self.sector
            ),
        )
        # L1 stall escalation: a member converged with an unresolved deficit
        # and asks L2 to arbitrate before the local-generation fallback fires,
        # preventing L1 from ramping DGs in parallel to an unknown allocation.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_local_gen_request),
            lambda msg, meta: (
                isinstance(msg, LocalGenerationRequest) and msg.sector == self.sector
            ),
        )
        # S2 — L3 multi-sector wake-up: new CP setpoints change per-sector
        # supply/demand. ``L3RebalanceWakeup`` is a no-payload nudge that marks
        # dirty via ``_maybe_schedule_rebalance`` (throttled there).
        self.context.subscribe_message(
            self,
            _wrap(self._handle_l3_wakeup),
            lambda msg, meta: (
                isinstance(msg, L3RebalanceWakeup) and msg.sector == self.sector
            ),
        )
        # Component-scoped L2 ADMM: every leader subscribes to both messages.
        # ``ComponentAdmmReport`` acted on only by the coordinator (self-gated);
        # ``ComponentAllocation`` applied by every leader to its own members.
        if self.admm_scope == "component":
            self.context.subscribe_message(
                self,
                _wrap(self._handle_component_admm_report),
                lambda msg, meta: (
                    isinstance(msg, ComponentAdmmReport) and msg.sector == self.sector
                ),
            )
            self.context.subscribe_message(
                self,
                _wrap(self._handle_component_allocation),
                lambda msg, meta: (
                    isinstance(msg, ComponentAllocation) and msg.sector == self.sector
                ),
            )

        # LeaderEmerged: a promoted orphan-community leader. Updating
        # ``_leader_node_ids`` keeps ``_resolve_component_peer_addrs`` from
        # filtering it out. Synchronous (dict mutate); sector-filtered.
        def _on_leader_emerged_msg(msg: Any, meta: dict) -> None:
            try:
                self._on_leader_emerged(msg)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[%s] _on_leader_emerged failed for %r: %s",
                    self.context.aid,
                    msg,
                    exc,
                )

        self.context.subscribe_message(
            self,
            _on_leader_emerged_msg,
            lambda msg, meta: (
                isinstance(msg, LeaderEmerged) and msg.sector == self.sector
            ),
        )

        # Retry formation when the eligible-neighbour set may have changed,
        # rather than waiting a full ``watchdog_s``.
        self.context.subscribe_event(
            self, CommunityReassignedEvent, self._on_community_reassigned
        )

    # ------------------------------------------------------------------
    # Holon formation
    # ------------------------------------------------------------------

    def _on_leader_emerged(self, message: LeaderEmerged) -> None:
        """Register a promoted orphan-community leader in ``_leader_node_ids``
        so it's admitted to the component peer set when reachable. Idempotent.
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
        """Membership changed, so kick formation + a full rebalance now.

        Needed because islanding re-elects a fresh coordinator; re-running on
        the topology-change event makes every reassigned leader re-report so
        the new component is allocated consistently (else a priority inversion).
        """
        self.context.schedule_instant_task(self._try_form_holon())
        self.context.schedule_instant_task(self._broadcast_recycle())

    async def _on_failure_notice(self, message: FailureNotice, _meta: dict) -> None:
        """Kick a prompt rebalance so the post-failure component re-allocates
        under the re-elected coordinator.

        The notice is TTL-bounded sector-tagged gossip, so only reached
        communities react (not the global ``BranchFailureEvent``). Fires for
        heat too: L2 membership/coordinator are topology concerns. The leader
        gate + min-gap throttle keep it cheap and collapse failure bursts.
        """
        await self._broadcast_recycle()

    async def _broadcast_recycle(self) -> None:
        """Tell every active-component peer to re-waterfall, then kick our own.

        The TTL-bounded ``FailureNotice`` may reach only a few agents; fanning
        out across the component-peer mesh ensures the re-elected coordinator
        and every part-owning leader re-report for a complete actor set.
        Single-hop (peers get ``from_member=False``, no re-broadcast); leaders only.
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

        From a member (``from_member=True``): re-broadcast to the component.
        From a peer leader (``from_member=False``): just re-report (single-hop).
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

        # Symmetry-breaking: only the lex-smallest aid initiates, else a clique
        # accepts competing requests and ends up leaderless. Rest wait.
        if any(addr.aid < self.context.aid for addr in neighbours):
            return

        candidates = neighbours[: self.max_holon_size - 1]
        holon_id = uuid4()
        self._pending_proposals[holon_id] = {str(a): (a, None) for a in candidates}

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
        accept = assignment.holon_id is None and community.community_id is not None

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

    async def _handle_join_answer(self, message: HolonicJoinAnswer, meta: dict) -> None:
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
            # All rejected; retry now (a busy rejecter may have finished).
            self.context.schedule_instant_task(self._try_form_holon())
            return

        # Record the resolved member set (initiator + acceptors).
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
        # Rebalance now so ADMM gets a shot while the deficit is still present.
        self.context.schedule_instant_task(self._try_rebalance())

    # ------------------------------------------------------------------
    # Inter-group coordination via DRO ADMM
    # ------------------------------------------------------------------

    def _live_members(self, members: list[Any]) -> list[Any]:
        """Subset of ``members`` the :class:`LivePeerFilter` deems reachable;
        passthrough when no filter is wired.
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
            # ``self.context`` may be None in unit-test construction.
            ctx = getattr(self, "context", None)
            logger.debug(
                "[%s] holon filter dropped %d unreachable members (kept=%d)",
                getattr(ctx, "aid", "<detached>"),
                len(dropped),
                len(kept),
            )
        return kept

    def _log_idle_once(self, reason: str) -> None:
        """One-shot info log for a recurring idle condition (per-reason)."""
        if reason in self._idle_logged:
            return
        self._idle_logged.add(reason)
        logger.info(
            "[%s] holon rebalance idle: %s (sector=%s)",
            self.context.aid,
            reason,
            self.sector.value,
        )

    def _record_event(self, kind: str, detail: str) -> None:
        """Record a diagnostic event with the standard event shape."""
        record_event(
            t=self.context.current_timestamp,
            kind=kind,
            aid=self.context.aid,
            sector=self.sector.value,
            detail=detail,
        )

    def _resolve_holon_members(self) -> list[Any]:
        """Holon member addresses: formation-time list, else ``"holons"``
        topology neighbours ([] if unwired).
        """
        if self._holon_member_addrs:
            return list(self._holon_member_addrs)
        try:
            return topology_neighbors(self, tid="holons")
        except Exception:
            return []

    async def _try_rebalance(self) -> None:
        """Collect flex for the next L2 ADMM round.

        ``"component"`` (default): every leader collects its own community's
        flex (self-ask) feeding ``_run_component_scoped_admm``; holon membership
        irrelevant. ``"holon"``/``"sector"`` (legacy): only holon-leaders.
        Fired periodically and reactively; throttled by ``rebalance_min_gap_s``.
        """
        now = self.context.current_timestamp
        # Throttle bypassed under the change-only cascade: the upward change-
        # detection (not this time fuse) bounds the holon→member→finished→holon
        # loop, and the _maybe_schedule_rebalance deferred-retry path is skipped
        # when the flag is on, so throttling here would silently drop a within-
        # gap reactive trigger until the slow watchdog.
        if (
            not self.enable_change_only_dispatch
            and (now - self._last_rebalance_t) < self.rebalance_min_gap_s
        ):
            return
        if self._rebalance_active:
            logger.debug("[%s] rebalance skipped: active", self.context.aid)
            return
        # No trigger since last rebalance ⇒ input unchanged ⇒ same allocation.
        # Only short-circuits the slow periodic invocation.
        if not self._rebalance_dirty:
            logger.debug(
                "[%s] rebalance skipped: no trigger since last run",
                self.context.aid,
            )
            return

        if self.admm_scope == "component":
            # Component-scope: every leader participates (no holon gate); each
            # community = one ADMM actor, matching the active subgraph.
            if topology_characteristic(self, tid="groups") != "leader":
                return
            # Runs in parallel with L3 (no defer): the next flex collection
            # reflects the post-CP state.
            self._rebalance_active = True
            self._last_rebalance_t = now
            # Clear dirty now we're committed; a trigger during execution
            # re-sets it so the next watchdog tick fires.
            self._rebalance_dirty = False
            self._flex_answers = []
            self._flex_answer_senders = []
            # Single self-ask aggregates the whole community's flex.
            self._flex_expected = 1
            self._flex_round_token += 1
            round_token = self._flex_round_token
            self._flex_round_id = f"{self.context.aid}/{round_token}"
            await self.context.send_message(
                AskForAvailableFlex(
                    include_connectors=False, round_id=self._flex_round_id
                ),
                receiver_addr=self.context.addr,
            )
            base = _FLEX_TIMEOUT_BASE_S.get(self.sector, _FLEX_TIMEOUT_DEFAULT_S)
            timeout = base + _FLEX_TIMEOUT_PER_MEMBER_S
            deadline = now + timeout
            self.context.schedule_timestamp_task(
                self._flex_collection_timeout(round_token), timestamp=deadline
            )
            return

        # Legacy per-holon / sector path: only holon leaders gather flex.
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None:
            self._log_idle_once("no holon assigned")
            return
        if assignment.parent_addr is not None:
            self._log_idle_once("not leader")
            return

        # Prefer the resolved member list; fall back to topology neighbours.
        if self._holon_member_addrs:
            members = list(self._holon_member_addrs)
        else:
            try:
                members = topology_neighbors(self, tid="holons")
            except Exception:
                return
        # Drop members the sibling ``DynamicHolonRole`` deems unreachable.
        members = self._live_members(members)
        if not members:
            self._log_idle_once("no neighbours")
            return

        self._rebalance_active = True
        self._last_rebalance_t = self.context.current_timestamp
        # Clear dirty now committed; triggers during execution re-set it.
        self._rebalance_dirty = False
        self._flex_answers = []
        self._flex_answer_senders = []
        # Leader contributes its own flex, else ADMM is single-actor and bails.
        self._flex_expected = len(members) + 1
        self._flex_round_token += 1
        round_token = self._flex_round_token
        self._flex_round_id = f"{self.context.aid}/{round_token}"

        logger.debug(
            "[%s] holon rebalance: asking %d members (+self) for flex",
            self.context.aid,
            len(members),
        )
        msg = AskForAvailableFlex(
            include_connectors=False, round_id=self._flex_round_id
        )
        await self.context.send_message(msg, receiver_addr=self.context.addr)
        for addr in members:
            await self.context.send_message(msg, receiver_addr=addr)

        # Timeout: run ADMM with whatever arrived (≥2) or release the lock.
        # Adaptive: per-sector base + per-member scaling.
        base = _FLEX_TIMEOUT_BASE_S.get(self.sector, _FLEX_TIMEOUT_DEFAULT_S)
        timeout = base + len(members) * _FLEX_TIMEOUT_PER_MEMBER_S
        deadline = self.context.current_timestamp + timeout
        self.context.schedule_timestamp_task(
            self._flex_collection_timeout(round_token), timestamp=deadline
        )

    async def _flex_collection_timeout(self, round_token: int) -> None:
        # Round-tagged: a timeout scheduled for an earlier, already-resolved
        # round must not release the lock or fire the ADMM for a later one.
        if round_token != self._flex_round_token:
            return
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

    # Min CP-commit flow shift before rebalancing. Above regulation noise,
    # below ``admm_abs_tol``.
    _CP_PREDICATE_DEAD_BAND_MW: float = 1e-3
    _CP_PREDICATE_MIN_GAP_S: float = 1.0

    async def _handle_cp_setpoint(self, message: CPSetpoint, meta: dict) -> None:
        """Direct L3 -> L2 trigger.

        Update per-publisher CP-setpoint memory, skip stale repeats, and (if
        the predicate accepts) schedule a rebalance.
        """
        # Only the holon leader acts; members would double-trigger.
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return
        # Echo guard: a CP commit caused by our own allocation isn't news.
        if (
            message.caused_by.get(str(self.context.aid), -1) == self._version.current
            and self._version.current > 0
        ):
            return
        if not self._seen_cps.is_fresh(message.publisher, message.version):
            return

        # Track the CP flow for our sector.
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
            self.context.aid,
            self.sector.value,
            message.publisher,
            delta,
        )
        self._maybe_schedule_rebalance()

    async def _handle_coalition_constraint(
        self, message: CoalitionConstraint, meta: dict
    ) -> None:
        """Direct L2.5 -> L2 trigger.

        A fresh ``CoalitionConstraint`` carries per-tier fractions to honour;
        re-run L2 now to re-merge with the store and re-dispatch. Only the
        holon leader acts; the throttle bounds rapid coalitions.
        """
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return
        self._maybe_schedule_rebalance()

    async def _handle_l3_wakeup(self, message: L3RebalanceWakeup, meta: dict) -> None:
        """S2 — L3 dispatched new CP setpoints. Mark dirty and let the
        throttle/scheduler decide; gated on group-leader membership.
        """
        self._maybe_schedule_rebalance()

    async def _handle_local_gen_request(
        self, message: LocalGenerationRequest, meta: dict
    ) -> None:
        """L1 stall escalation handler.

        A member converged with an unresolved residual. L2 (1) triggers an
        early rebalance (ADMM may absorb it before DGs ramp) and (2) approves
        the fallback for the rest. Only the lex-smallest co-recipient replies
        to avoid N approvals; all peers trigger the rebalance.
        """
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        try:
            co_recipients = [
                a for a in topology_neighbors(self, tid="holons") if a.aid != sender.aid
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
        """Shared throttle + gate for reactive paths; pre-filters no-op tasks
        (``_try_rebalance`` re-checks the gates). Gates by ``admm_scope``:
        ``"component"`` ⇒ any group leader; else only holon-leaders.
        """
        if self.admm_scope == "component":
            if topology_characteristic(self, tid="groups") != "leader":
                return
        else:
            assignment = self.context.get_or_create_model(HolonicAssignment)
            if assignment.holon_id is None or assignment.parent_addr is not None:
                return
        # Mark dirty before the throttle so a throttled trigger is picked up
        # by the watchdog next tick.
        self._rebalance_dirty = True
        if self._rebalance_active:
            return
        now = self.context.current_timestamp
        if not self.enable_change_only_dispatch:
            gap_left = (self._last_rebalance_t + self.rebalance_min_gap_s) - now
            if gap_left > 0:
                # Throttled: schedule one deferred retry at gap-expiry.
                if not self._rebalance_retry_pending:
                    self._rebalance_retry_pending = True
                    self.context.schedule_timestamp_task(
                        self._deferred_rebalance(), timestamp=now + gap_left
                    )
                return
        self.context.schedule_instant_task(self._try_rebalance())

    async def _deferred_rebalance(self) -> None:
        """Fire a throttled rebalance once the ``rebalance_min_gap_s`` fuse
        clears. No-op if an intervening round already resolved the state
        (``_try_rebalance`` re-checks the gates).
        """
        self._rebalance_retry_pending = False
        await self._try_rebalance()

    async def _on_member_finished(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        """Message path: a holon-peer finished gossip — rebalance to spread
        the residual. Throttled by ``rebalance_min_gap_s`` to break the
        holon→member→finished→holon feedback loop.
        """
        self._maybe_schedule_rebalance()

    def _on_member_finished_local(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        """Local path: when the leader is itself the gossip originator, its
        negotiator emits via ``emit_event`` (a different bus, so the message
        sub misses it). Same throttle as ``_on_member_finished``.
        """
        if event.sector != self.sector:
            return
        self._maybe_schedule_rebalance()

    async def _handle_flex_answer(
        self, message: AvailableFlexAnswer, meta: dict
    ) -> None:
        if not self._rebalance_active:
            return
        # Strict round identity: asks stamp a fresh id per round and the sole
        # responder (balance._handle_ask_flex) always echoes it, so a round-N
        # straggler can't double-count a member into round N+1.
        if getattr(message, "round_id", "") != self._flex_round_id:
            return
        # Ignore non-member answers: a stray reply would inflate the count.
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
        """Dispatch to the legacy, tier-stratified, or component-scoped path
        by scope/mode.

        The tier-stratified path uses a 2-D ``targets[sector][tier]`` allocation
        (preserving priority intent), falling back to legacy on no per-tier
        deficit. ``"component"`` routes via the elected coordinator.
        """
        if self.admm_scope == "component":
            await self._run_component_scoped_admm()
            return
        if not self.enable_tier_stratified_admm:
            await self._run_legacy_per_sector_admm()
            return
        # Supply-priority ADMM: fires when there's supply to allocate;
        # priority weighting binds once supply < demand.
        if self._supply_priority_has_anything_to_do():
            await self._run_supply_priority_admm()
            return
        await self._run_legacy_per_sector_admm()

    def _supply_priority_has_anything_to_do(self) -> bool:
        """True iff queued answers have both per-tier demand and supply."""
        any_demand = any(a.demand_by_sector_priority for a in self._flex_answers)
        any_supply = any(
            float(s) > 1e-9
            for a in self._flex_answers
            for s in (a.supply_by_sector or {}).values()
        )
        return any_demand and any_supply

    async def _run_legacy_per_sector_admm(self) -> None:
        """Run ADMM sharing across member groups.

        Each group is a multi-dimensional actor (one dim per sector), so
        resources balance across sectors at once (e.g. gas surplus covering
        heat deficit via CHP). ``T`` is per-sector total imbalance. Priority-
        aware: ``S`` pulls each actor toward its priority-weighted share.
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

        # Union of sectors across member groups.
        all_sectors: list[str] = sorted(
            {s for a in answers for s in a.balance_by_sector}
        )
        n_dims = len(all_sectors) if all_sectors else 1

        # Fall back to 1D if no per-sector data.
        if n_dims <= 1 and not any(a.balance_by_sector for a in answers):
            all_sectors = [self.sector.value]
            n_dims = 1

        sector_idx = {s: i for i, s in enumerate(all_sectors)}

        actors: list[ADMMFlexActor] = []
        total_T = np.zeros(n_dims)

        # Per-group bounds first, then priority shares.
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

        # Feasibility cap: when |T[i]| exceeds available budgets, the distance
        # term plateaus and the library spuriously logs "max iterations".
        # Bound |T| by the budget envelope (sign preserved).
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

        # Priority-weighted S: waterfall shares under strict priority ordering.
        # Budget = surplus + flex headroom; the flex term keeps the budget
        # positive in a pure-deficit holon with no surplus.
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
            # Priority disabled — distribute uniformly (balance-only ablation).
            even = total_available / max(1, len(answers))
            priority_shares = [even for _ in answers]

        for idx, answer in enumerate(answers):
            lb, ub = group_bounds[idx]
            C = np.zeros((0, n_dims))
            d = np.zeros(0)
            # S = local-QP linear cost (negative attracts), proportional to the
            # priority-weighted share so ADMM steers toward high-priority demand.
            S = np.zeros(n_dims)
            if priority_shares[idx] > 1e-9:
                # Normalise by total target (stability vs rho), then split the
                # pull across dims by per-dimension balance.
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
        # Iter cap / tol from config. Defaults (50 @ 1e-3) relaxed from 1000 /
        # 1e-4 so concurrent ADMMs don't block discrete-time progress.
        coordinator.max_iters = int(self.admm_max_iters)
        coordinator.abs_tol = float(self.admm_abs_tol)
        start_msg = create_admm_start(create_admm_sharing_data(total_T.tolist()))

        try:
            with optimization(
                "admm_holon",
                logger=logger,
                aid=self.context.aid,
                n_actors=len(actors),
                sectors=all_sectors,
            ):
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
            # On failure still trigger intra-group gossip for local rebalance.

        # Trigger intra-group rebalancing, routing each allocation as the gossip
        # override target. ``actors[idx].x`` is a vector over ``all_sectors``;
        # pick the member's sector and negate (target = -allocation). Unmapped
        # members fall back to local recompute.
        sender_to_actor: dict[str, tuple[Any, AvailableFlexAnswer]] = {}
        for sender, answer, actor in zip(senders, answers, actors):
            sender_to_actor[str(sender)] = (actor, answer)

        triggers = self._resolve_holon_members()
        # Carry per-member overrides on ``HolonAllocation`` AND push the legacy
        # ``StartBalanceNegotiation`` so L1's override path keeps working.
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

        # Publish HolonAllocation to CP connectors so L3 reacts directly,
        # skipping the L2->L1->gossip->L3 detour (``tid="groups"`` link).
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
                    self.context.aid,
                    self.sector.value,
                    len(allocation_targets),
                    decision.version,
                    len(cp_connectors),
                )
                for addr in cp_connectors:
                    await self.context.send_message(decision, receiver_addr=addr)

    async def _run_supply_priority_admm(self) -> None:
        """Supply-priority ADMM.

        Differs from the demand-side path: ``T`` is total demand per
        (sector, tier) cell (priority-weighted), and each actor's ``ub`` /
        coupling reflect generator capacity, so a supply-rich group can serve
        holon-wide tier-X demand. Output: per-(sector, tier) service fractions
        sent to each leader, giving a consistent shed-low/serve-high pattern.
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

        # Active cells. A sector with supply but no demand bypasses the ADMM;
        # no demand anywhere ⇒ fall back to the legacy path.
        sectors, tiers, total_demand = extract_demand_sectors_tiers(actor_demands)
        if not sectors or not tiers or total_demand < 1e-6:
            await self._run_legacy_per_sector_admm()
            return

        # F6 deliverability caps: when member nodes + mirror are available,
        # cap each cell at demand reachable from the actor's home node so the
        # ADMM never commits unroutable supply. Unreachable leader ⇒ caps 0;
        # reachable ⇒ uncapped (None), so coupling is the only binding constraint.
        actor_ub_overrides: list[dict[tuple[str, int], float] | None] | None = None
        if self._topology_mirror is not None and self._leader_node_ids:
            try:
                actor_node_ids: list[Any | None] = []
                actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]] = []
                for sender, answer in zip(senders, answers):
                    leader_aid = getattr(sender, "aid", str(sender))
                    node_id = self._leader_node_ids.get(leader_aid)
                    actor_node_ids.append(node_id)
                    # Map this leader's tier-aggregated demand onto its home
                    # node; an unreachable leader contributes nothing.
                    per_tier: dict[int, dict[Any, float]] = {}
                    if node_id is not None:
                        for sec, tier_map in (
                            answer.demand_by_sector_priority or {}
                        ).items():
                            if sec not in sectors:
                                continue
                            for tier, dem in tier_map.items():
                                # Bind inner dict first: a combined
                                # ``setdefault(k,{})[n] = ...[k].get(...)`` would
                                # KeyError (RHS evaluated before LHS subscript).
                                inner = per_tier.setdefault(int(tier), {})
                                inner[node_id] = inner.get(node_id, 0.0) + float(dem)
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
                    self.context.aid,
                    type(exc).__name__,
                    exc,
                )
                actor_ub_overrides = None

        try:
            with optimization(
                "admm_supply_priority",
                logger=logger,
                scope="holon",
                aid=self.context.aid,
                n_actors=len(actor_supplies),
            ):
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
                self.context.aid,
                exc,
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
            self.context.aid,
            sectors,
            tiers,
            total_T,
            sum_x_per_cell,
            sum(meta["actor_supply_total"]),
        )
        self._record_event(
            "holon_admm_result",
            f"supply_priority sectors={sectors} tiers={tiers} fractions={service_fraction}",
        )
        self._record_event(
            "holon_priority_allocation",
            str(
                {
                    f"{sec}:tier{tier}": {
                        "T": round(float(total_T[_flat_idx(sec, tier)]), 6),
                        "weight": float(priorities[_flat_idx(sec, tier)]),
                        "sum_x": round(float(sum_x_per_cell[_flat_idx(sec, tier)]), 6),
                        "service_frac": round(service_fraction[sec][tier], 4),
                    }
                    for sec in sectors
                    for tier in tiers
                }
            ),
        )

        # Coalition merge: an active coalition fraction overrides L2's per-cell
        # result; untouched cells keep L2's. Without it, last-write-wins would
        # reset the regulation each round.
        if self._coalition_constraint_store is not None:
            now = float(self.context.current_timestamp)
            service_fraction = self._coalition_constraint_store.merge_into(
                service_fraction,
                self.sector,
                now,
            )
            # Post-merge tier-monotonic clamp: same rationale as the
            # component-allocation path — a per-tier coalition override must
            # not lift a lower tier above a higher one.
            for tier_map in service_fraction.values():
                clamp_tier_monotonic(tier_map)

        # Send the holon-global fraction map to every member leader (L1 honour).
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
    # One ADMM per active connected component; each leader is one actor
    # (holon abstraction unused). Per round: each leader self-collects flex,
    # pushes a ``ComponentAdmmReport`` to the coordinator (lex-smallest
    # reachable leader), which debounces a burst into one solve and dispatches
    # a ``ComponentAllocation`` to every leader, who applies it to its members.
    # Invariant: every load at the same tier in a (sector, component) is served
    # at the same fraction, so cross-leader inversions cannot arise.

    def _resolve_sector_peer_addrs(self) -> dict[str, Any]:
        """``{leader_aid: leader_addr}`` for every same-sector leader on the
        ``holon_summary_<sector>`` topology (incl. self). Unfiltered baseline.
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
        """``{leader_aid: leader_addr}`` for same-sector leaders in this
        leader's connected component. Falls back to the unfiltered sector peer
        set when the mirror or own node id is unavailable.

        Filter to known leader aids (self always included) to keep CP/branch
        agents out of the coordinator election (a report routed to one is
        dropped); empty ``_leader_node_ids`` ⇒ unfiltered fallback.
        """
        sector_peers = self._resolve_sector_peer_addrs()
        leader_aids = set(self._leader_node_ids)
        if leader_aids:
            sector_peers = {
                aid: addr
                for aid, addr in sector_peers.items()
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
                # Always include self even if ``_leader_node_ids`` lacks it.
                out[aid] = addr
                continue
            node_id = self._leader_node_ids.get(aid)
            if node_id is None or node_id in reachable:
                out[aid] = addr
        return out

    def _component_coordinator_aid(self) -> str | None:
        """Lex-smallest component-peer aid — the coordinator. None only when
        even self is missing (caller falls back to the per-holon path).
        """
        peers = self._resolve_component_peer_addrs()
        if not peers:
            return None
        return min(peers.keys())

    def _multi_sector_l3_active(self) -> bool:
        """True iff this leader sits in a multi-sector component with a CP
        agent (so an L3 coordinator owns the decision and L2 defers).

        Traverses the joint subgraph via ``reachable_from(allow_cp_bridges=True)``
        and intersects with known CP nodes; empty ⇒ L2 runs locally.
        """
        if not self._cp_node_ids:
            return False
        mirror = self._topology_mirror
        my_node = self._my_node_id
        if mirror is None or my_node is None:
            return False
        try:
            reachable = mirror.reachable_from(
                my_node,
                sector=None,
                allow_cp_bridges=True,
            )
        except Exception:
            return False
        return any(n in reachable for n in self._cp_node_ids)

    async def _run_component_scoped_admm(self) -> None:
        """Component-scoped variant of ``_run_supply_priority_admm``.

        Flex already collected. Coordinator buffers its own report + schedules
        a debounced solve; non-coordinator pushes a ``ComponentAdmmReport`` and
        awaits the ``ComponentAllocation``. ``_flex_answers`` is drained so a
        follow-up trigger doesn't re-fire on the stale buffer.
        """
        if not self._rebalance_active:
            return
        answers = self._flex_answers[:]
        senders = self._flex_answer_senders[:]
        # Drain.
        self._flex_answers = []
        self._flex_answer_senders = []
        self._flex_expected = 0
        self._rebalance_active = False

        if not answers:
            return

        supply, demand, served = aggregate_holon_flex(answers)
        # Report on any supply OR demand: demand-only feeds the T vector,
        # supply-only the pool. ``and`` would drop load-only communities.
        any_supply = any(v > 1e-9 for v in supply.values())
        any_demand = any(mw > 1e-9 for tmap in demand.values() for mw in tmap.values())
        if not (any_supply or any_demand):
            return

        coord_aid = self._component_coordinator_aid()
        if coord_aid is None:
            # No peer topology; degenerate to the per-holon path. Restore both
            # buffers: the fallback zips senders with answers for the
            # deliverability caps, so answers alone leave it inert.
            self._flex_answers = answers
            self._flex_answer_senders = senders
            self._rebalance_active = True
            await self._run_supply_priority_admm()
            return

        round_id = f"r{self._component_round_counter}"
        self._component_round_counter += 1
        now = float(self.context.current_timestamp)
        leader_aid = self.context.aid

        # Implicit ACK: echo the latest applied version so the coordinator can
        # detect a missed dispatch — but only when it was published by the
        # CURRENT coordinator. After re-election the old coordinator's high
        # version would otherwise wedge the new one's ``_resend_allocation_
        # if_stale`` (its counter restarts low).
        echo_version = (
            self._last_applied_allocation_version
            if self._last_applied_allocation_publisher == coord_aid
            else -1
        )
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
            last_applied_allocation_version=echo_version,
        )

        if coord_aid == leader_aid:
            # I'm the coordinator — buffer my report, trigger the solve.
            self._component_report_buffer[leader_aid] = (round_id, report)
            await self._maybe_run_component_admm(reason="self_report")
            return

        # Push to the coordinator. No reply timeout: a drop is retried by the
        # next trigger (idempotent — buffer keyed by leader_aid).
        peers = self._resolve_component_peer_addrs()
        coord_addr = peers.get(coord_aid)
        if coord_addr is None:
            # Aid known but address unresolved — fall back to the per-holon path.
            self._flex_answers = answers
            self._flex_answer_senders = senders
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
        """Coordinator-side: buffer a peer's report (keyed by ``leader_aid``,
        freshest wins) and schedule the debounced solve. Non-coordinators drop;
        reports from peers no longer in our component are dropped too.
        """
        if self.admm_scope != "component":
            return
        if self._component_coordinator_aid() != self.context.aid:
            return
        # Only buffer reports from leaders still in our active component.
        component_peers = self._resolve_component_peer_addrs()
        if message.leader_aid not in component_peers:
            return
        # Per-publisher staleness guard (mirrors ``_on_summary``): under
        # latency/loss a reordered older report must not overwrite a fresher
        # one (the buffer is last-arrival-wins otherwise).
        prior = self._component_report_buffer.get(message.leader_aid)
        if prior is not None and int(
            getattr(message, "version", 0)
        ) <= int(getattr(prior[1], "version", -1)):
            return
        self._component_report_buffer[message.leader_aid] = (message.round_id, message)
        # Packet-loss recovery: if the sender's echoed version trails our latest
        # dispatch, re-send to just this peer. getattr-guarded for legacy reports.
        try:
            await self._resend_allocation_if_stale(
                message.leader_aid,
                component_peers.get(message.leader_aid),
                int(getattr(message, "last_applied_allocation_version", -1)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[%s] resend-if-stale failed for leader=%s: %s",
                self.context.aid,
                message.leader_aid,
                exc,
            )
        await self._maybe_run_component_admm(reason="peer_report")

    async def _resend_allocation_if_stale(
        self,
        leader_aid: str,
        leader_addr: Any | None,
        applied_version: int,
    ) -> None:
        """Re-send the latest ``ComponentAllocation`` to ``leader_addr`` when
        its echoed version trails ``_allocation_version_counter`` (dispatch
        lost). Idempotent (leaf ignores version ≤ applied); self-skipped.
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
            self._last_dispatched_allocation,
            receiver_addr=leader_addr,
        )
        self._record_event(
            "component_alloc_resent",
            f"target={leader_aid} applied_version={applied_version} "
            f"latest_version={self._allocation_version_counter}",
        )

    async def _maybe_run_component_admm(self, *, reason: str) -> None:
        """Debounce: collapse a report burst into one solve. The first arrival
        owns the solve (``_component_dispatch_pending``), one per burst.
        """
        if self._component_dispatch_pending:
            return
        self._component_dispatch_pending = True
        self.context.schedule_instant_task(
            self._run_component_admm_now(reason=reason),
        )

    async def _run_component_admm_now(self, *, reason: str) -> None:
        """Run the per-component ADMM over the buffer and dispatch fractions to
        every leader (incl. self). Clears ``_component_dispatch_pending``; the
        buffer is not drained so late reports overwrite for the next solve.
        """
        try:
            await self._run_component_admm_now_inner(reason=reason)
        finally:
            self._component_dispatch_pending = False

    async def _run_component_admm_now_inner(self, *, reason: str) -> None:
        if not self._component_report_buffer:
            return
        # Only leaders still in this coordinator's component view; a
        # disconnected leader's report stays buffered but is skipped.
        component_peers = self._resolve_component_peer_addrs()
        leader_aids = sorted(
            aid for aid in self._component_report_buffer if aid in component_peers
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
            with optimization(
                "admm_supply_priority",
                logger=logger,
                scope="component",
                aid=self.context.aid,
                n_actors=len(actor_supplies),
            ):
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
                "[%s] component-scope ADMM failed: %s",
                self.context.aid,
                exc,
            )
            self._record_event("holon_admm_failed", f"component_scope: {exc}")
            return

        # No sub-tolerance noise scrub: clamping near-zero to exact 0 would lock
        # it in via the cooldown gate. The PI claim's 1e-3 tolerance absorbs it.

        # Complete + monotone per-tier vector. A round may solve a tier subset,
        # so fold in the previous dispatch's tiers (fresh wins) then clamp
        # non-increasing in tier number, avoiding a stale-higher inversion.
        # Bounded to this coordinator's solve+history (not all P tiers): forcing
        # unknown tiers down over-sheds; handoff inversion needs L2.5 instead.
        sec_val = self.sector.value
        prev_fraction = self._last_component_fraction
        prev_own = (prev_fraction or {}).get(sec_val, {})
        merged_own = dict(prev_own)
        merged_own.update(service_fraction.get(sec_val, {}))
        clamp_tier_monotonic(merged_own)
        service_fraction = {**service_fraction, sec_val: merged_own}

        # Allocation-unchanged gate: when the merged solve equals the last
        # dispatched fractions (within the actuator dedup tolerance), skip the
        # re-dispatch entirely. This (with the leaf-side "anything actually
        # changed" nudge gate) bounds the dispatch→rebalance→report→solve loop
        # under enable_change_only_dispatch, where no time throttle applies.
        # Leaders that missed the previous dispatch still recover via
        # ``_resend_allocation_if_stale`` (version echo), so skipping is safe.
        if (
            self._allocation_version_counter > 0
            and prev_fraction is not None
            and _fraction_maps_equal(prev_fraction, service_fraction)
        ):
            logger.debug(
                "[%s] component-scope ADMM: allocation unchanged — dispatch skipped",
                self.context.aid,
            )
            return
        logger.info(
            "[%s] component-scope ADMM result (reason=%s): sectors=%s "
            "tiers=%s n_communities=%d fractions=%s",
            self.context.aid,
            reason,
            sectors,
            tiers,
            len(reports),
            service_fraction,
        )
        self._record_event(
            "holon_admm_result",
            f"component_scope reason={reason} sectors={sectors} tiers={tiers} "
            f"n_communities={len(reports)} fractions={service_fraction}",
        )

        # Dispatch the same ComponentAllocation to every leader (incl. self);
        # each applies it via ``_handle_component_allocation``. The cascade is
        # bounded by the UPWARD change-detection (L1 only notifies L2 when its
        # converged setpoint moved), not by skipping this authoritative dispatch
        # — skipping it would stale the per-load L2 priority floor (set inside
        # ``apply_regulate``) and let a fresh L1 gossip invert priority.
        round_id = max(
            (self._component_report_buffer[a][0] for a in leader_aids),
            default="",
        )
        now = float(self.context.current_timestamp)
        # Bump version before building the message so leaf ACKs line up. Stash
        # for re-sends to stale leaders (``_resend_allocation_if_stale``).
        self._allocation_version_counter += 1
        allocation = ComponentAllocation(
            publisher=self.context.aid,
            version=self._allocation_version_counter,
            timestamp_s=now,
            round_id=round_id,
            sector=self.sector,
            service_fraction_by_tier=service_fraction.get(self.sector.value, {}),
        )
        # Rebase only on actual dispatch: both the skip gate's prev_fraction
        # and the monotonic-merge anchor must reference the last DISPATCHED
        # map, or skipped rounds accumulate unbounded drift.
        self._last_component_fraction = service_fraction
        self._last_dispatched_allocation = allocation
        for addr in component_peers.values():
            await self.context.send_message(allocation, receiver_addr=addr)

    async def _handle_component_allocation(
        self, message: ComponentAllocation, meta: dict
    ) -> None:
        """Leaf-side: apply the coordinator's per-tier fraction to this leader's
        own community members (no holon hop) via the L1 honour path. Every
        sector leader handles it, covering communities outside any holon.
        Coalition merge: an active store fraction wins per-tier.
        """
        if self.admm_scope != "component":
            return
        # Cheap drift guard: act only as a current group leader.
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Version gate BEFORE applying: a delayed/duplicated older allocation
        # from the same coordinator must not overwrite a fresher one. Versions
        # are per-publisher; a different publisher (coordinator re-election)
        # always passes and resets the counter below.
        try:
            msg_version = int(message.version)
        except (TypeError, ValueError):
            msg_version = None  # legacy allocation: apply, ack stays -1
        msg_publisher = str(getattr(message, "publisher", "") or "")
        if (
            msg_version is not None
            and msg_publisher == self._last_applied_allocation_publisher
            and msg_version <= self._last_applied_allocation_version
        ):
            return
        # Rebuild a {sector: {tier: frac}} envelope for the L1 honour path.
        service_fraction: dict[str, dict[int, float]] = {
            self.sector.value: dict(message.service_fraction_by_tier),
        }
        if self._coalition_constraint_store is not None:
            now = float(self.context.current_timestamp)
            service_fraction = self._coalition_constraint_store.merge_into(
                service_fraction,
                self.sector,
                now,
            )
            # Re-assert non-increasing-in-tier monotonicity AFTER the coalition
            # merge. ``merge_into`` lets a per-tier store fraction win, which can
            # lift a lower-priority tier above a higher one and reintroduce the
            # exact inversion the coordinator's clamp removed. Without this, a
            # competing coalition override silently breaks component priority
            # order (the observed gas tier-1-victim inversions).
            merged = service_fraction.get(self.sector.value)
            if merged:
                clamp_tier_monotonic(merged)
        # Send to SELF: the negotiator applies the fraction to every community
        # member, covering loads outside any holon. This authoritative dispatch
        # also refreshes the per-load L2 priority floor (in ``apply_regulate``);
        # the cascade is bounded upstream by the L1→L2 change-detection, not by
        # skipping it here.
        await self.context.send_message(
            StartBalanceNegotiation(
                service_fraction_by_sector_priority=service_fraction,
            ),
            receiver_addr=self.context.addr,
        )
        # ACK: record (publisher, version) so the next report echoes it (the
        # coordinator detects drops via the echo). A new publisher resets the
        # counter — versions are not comparable across coordinators.
        if msg_version is not None:
            self._last_applied_allocation_publisher = msg_publisher
            self._last_applied_allocation_version = msg_version
        self._record_event(
            "holon_priority_allocation",
            f"component_scope round={message.round_id} "
            f"version={getattr(message, 'version', 0)} "
            f"fractions={service_fraction}",
        )
