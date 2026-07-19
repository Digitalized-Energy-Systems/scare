"""Holonic (multi-level) community formation and coordination.

L1 sector agents solve local restoration via gossip; L2 holon leaders
aggregate member-group flex and run DRO ADMM for inter-group sharing,
then drive each group to rebalance toward the per-actor allocation.

The role itself is the wiring: subscriptions, the leader gate and the
rebalance trigger/throttle. The work lives in owned helpers — formation
(:mod:`holon_formation`), peer resolution (:mod:`holon_peers`), round state
(:mod:`holon_round`), the holon-scoped solvers (:mod:`holon_admm`) and the
component-scoped protocol (:mod:`holon_component`).
"""

from __future__ import annotations

import logging
from typing import Any

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_characteristic, topology_neighbors

from scare.base.channel import (
    CoalitionConstraint,
    ComponentAdmmReport,
    ComponentAllocation,
    CPSetpoint,
    L3RebalanceWakeup,
    MonotonicVersion,
    SeenVersions,
)
from scare.base.model import (
    AskForAvailableFlex,
    AvailableFlexAnswer,
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
)
from scare.base.runtime.diagnostics import record_event
from scare.base.topology.topology_mirror import LivePeerFilter
from scare.base.util import async_dispatch
from scare.community.holon_admm import SectorAdmmRunner
from scare.community.holon_component import ComponentCoordinator
from scare.community.holon_formation import HolonFormation
from scare.community.holon_peers import PeerResolver
from scare.community.holon_round import RebalanceRound

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
        heat_rebalance_period_s: float | None = None,
        live_member_filter: LivePeerFilter | None = None,
        coalition_constraint_store: Any = None,
        my_node_id: Any = None,
        leader_node_ids: dict[str, Any] | None = None,
        topology_mirror: Any = None,
    ) -> None:
        super().__init__()
        self.sector = sector
        # Coalition fractions (written by sibling ``HolonSummaryRole``)
        # override L2's per-tier result for the TTL window. None ⇒ no merge.
        self._coalition_constraint_store = coalition_constraint_store
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
        # L2 side only bypasses the rebalance_min_gap_s throttle; the
        # cascade-bounding change-detection lives on the upward L1→L2 edge
        # (balance.py).
        self.enable_change_only_dispatch = bool(enable_change_only_dispatch)
        # ``rebalance_period_s``: slow drift heartbeat (timeseries changes with
        # no NegotiationFinishedEvent). ``rebalance_min_gap_s``: fuse letting one
        # member-gossip round resolve before the next ADMM cycle.
        self.rebalance_period_s = rebalance_period_s
        self.rebalance_min_gap_s = rebalance_min_gap_s
        self.flex_timeout_s = flex_timeout_s
        # Heat has no gossip finished-event and its ConstraintViolations
        # dead-end before L2, so only this poll tracks the evolving thermal
        # deficit. None ⇒ off (non-heat / flag off).
        self.heat_rebalance_period_s = heat_rebalance_period_s

        # Safety net for a leader that missed every trigger event.
        self.watchdog_s = watchdog_s

        # ``live_member_filter``: optional ``DynamicHolonRole`` filtering members
        # reachable via live grid edges (None ⇒ static-topology mode). The node
        # id / mirror / leader table are the deliverability wiring (F6).
        self._peers = PeerResolver(
            self,
            live_member_filter=live_member_filter,
            my_node_id=my_node_id,
            leader_node_ids=leader_node_ids,
            topology_mirror=topology_mirror,
        )
        self._formation = HolonFormation(self)
        self._round = RebalanceRound(self)
        self._admm = SectorAdmmRunner(self)
        self._component = ComponentCoordinator(self)

        # Per-reason so each idle early-return surfaces once, not every tick.
        self._idle_logged: set[str] = set()

        # --- Channel-pattern state (L2 <-> L3 direct link) ---
        # ``_version`` advances per published ``HolonAllocation``; ``_seen_cps``
        # tracks the latest consumed ``CPSetpoint`` version per publisher.
        self._version = MonotonicVersion()
        self._seen_cps = SeenVersions()
        self._cp_setpoint_state: dict[tuple[str, str], float] = {}
        self._last_cp_predicate_fire_t: float = -1e9

    @property
    def _leader_node_ids(self) -> dict[str, Any]:
        """Live view of the peer resolver's leader registry."""
        return self._peers.leader_node_ids

    def request_rebalance(self) -> None:
        """Public: ask this leader to re-run its L2 round (throttled)."""
        self._maybe_schedule_rebalance()

    def setup(self) -> None:
        # Formation is event-driven; this slow watchdog covers a leader that
        # missed every trigger.
        self.context.schedule_periodic_task(self._try_form_holon, delay=self.watchdog_s)
        # Rebalance is event-driven via the reactive handlers below plus this
        # watchdog; every cause of drift is itself an event (no drift probe).
        self.context.schedule_periodic_task(self._try_rebalance, delay=self.watchdog_s)
        # Heat deficit poll (see __init__): drives the per-tier allocation as
        # temperatures/delivered heat evolve. The min-gap fuse and the
        # unchanged-dispatch guard on the balance side bound the churn.
        if self.sector == Sector.HEAT and self.heat_rebalance_period_s:
            self.context.schedule_periodic_task(
                self._heat_periodic_rebalance, delay=self.heat_rebalance_period_s
            )

        _wrap = async_dispatch(self)

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
        if self._peers.register_leader(message):
            record_event(
                t=float(self.context.current_timestamp),
                kind="leader_emerged_registered",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=(
                    f"new_leader={message.leader_aid} node_id={message.node_id} "
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

    def _acts_as_leader(self) -> bool:
        """True iff this agent should run L2 for the active scope.

        Component scope: any group leader (holon formation is not run, so
        ``HolonicAssignment`` is never set). Legacy holon/sector scope: a
        formation-assigned holon leader (holon_id set, no parent).
        """
        if self.admm_scope == "component":
            try:
                return topology_characteristic(self, tid="groups") == "leader"
            except Exception:
                return False
        assignment = self.context.get_or_create_model(HolonicAssignment)
        return assignment.holon_id is not None and assignment.parent_addr is None

    async def _try_form_holon(self) -> None:
        await self._formation.try_form()

    async def _handle_join_request(
        self, message: HolonicJoinRequest, meta: dict
    ) -> None:
        await self._formation.handle_join_request(message, meta)

    async def _handle_join_answer(self, message: HolonicJoinAnswer, meta: dict) -> None:
        await self._formation.handle_join_answer(message, meta)

    # ------------------------------------------------------------------
    # Inter-group coordination via DRO ADMM
    # ------------------------------------------------------------------

    def _live_members(self, members: list[Any]) -> list[Any]:
        return self._peers.live_members(members)

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
        return self._formation.resolve_members()

    async def _try_rebalance(self) -> None:
        """Collect flex for the next L2 ADMM round.

        ``"component"`` (default): every leader collects its own community's
        flex (self-ask) feeding the component protocol; holon membership
        irrelevant. ``"holon"``/``"sector"`` (legacy): only holon-leaders.
        Fired periodically and reactively; throttled by ``rebalance_min_gap_s``.
        """
        rnd = self._round
        now = self.context.current_timestamp
        # Under change-only dispatch the upward change-detection (not this fuse)
        # bounds the loop and the deferred-retry path is off, so throttling here
        # would drop a within-gap reactive trigger until the slow watchdog.
        if (
            not self.enable_change_only_dispatch
            and (now - rnd.last_t) < self.rebalance_min_gap_s
        ):
            return
        if rnd.active:
            logger.debug("[%s] rebalance skipped: active", self.context.aid)
            return
        # No trigger since last rebalance ⇒ input unchanged ⇒ same allocation.
        # Only short-circuits the slow periodic invocation.
        if not rnd.dirty:
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
            # reflects the post-CP state. Single self-ask aggregates the whole
            # community's flex.
            round_token = rnd.open(expected=1, now=now)
            await self.context.send_message(
                AskForAvailableFlex(include_connectors=False, round_id=rnd.round_id),
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
        if self._formation.member_addrs:
            members = list(self._formation.member_addrs)
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

        # Leader contributes its own flex, else ADMM is single-actor and bails.
        round_token = rnd.open(
            expected=len(members) + 1, now=self.context.current_timestamp
        )

        logger.debug(
            "[%s] holon rebalance: asking %d members (+self) for flex",
            self.context.aid,
            len(members),
        )
        msg = AskForAvailableFlex(include_connectors=False, round_id=rnd.round_id)
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
        rnd = self._round
        if round_token != rnd.token:
            return
        if not rnd.active:
            return
        received = len(rnd.answers)
        logger.warning(
            "[%s] holon flex timeout: received %d/%d answers",
            self.context.aid,
            received,
            rnd.expected,
        )
        if received >= 2:
            await self._run_inter_group_admm()
        else:
            rnd.active = False

    # Min CP-commit flow shift before rebalancing. Above regulation noise,
    # below ``admm_abs_tol``.
    _CP_PREDICATE_DEAD_BAND_MW: float = 1e-3
    _CP_PREDICATE_MIN_GAP_S: float = 1.0

    async def _handle_cp_setpoint(self, message: CPSetpoint, meta: dict) -> None:
        """Direct L3 -> L2 trigger.

        Update per-publisher CP-setpoint memory, skip stale repeats, and (if
        the predicate accepts) schedule a rebalance.
        """
        # Only the leader acts; members would double-trigger.
        if not self._acts_as_leader():
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
        leader acts; the throttle bounds rapid coalitions.
        """
        if not self._acts_as_leader():
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
        if not self._acts_as_leader():
            return
        rnd = self._round
        # Mark dirty before the throttle so a throttled trigger is picked up
        # by the watchdog next tick.
        rnd.dirty = True
        if rnd.active:
            return
        now = self.context.current_timestamp
        if not self.enable_change_only_dispatch:
            gap_left = (rnd.last_t + self.rebalance_min_gap_s) - now
            if gap_left > 0:
                # Throttled: schedule one deferred retry at gap-expiry.
                if not rnd.retry_pending:
                    rnd.retry_pending = True
                    self.context.schedule_timestamp_task(
                        self._deferred_rebalance(), timestamp=now + gap_left
                    )
                return
        self.context.schedule_instant_task(self._try_rebalance())

    async def _heat_periodic_rebalance(self) -> None:
        """Heat deficit poll: re-run the component allocation on a cadence
        (heat has no reactive trigger of its own — see ``__init__``)."""
        self._maybe_schedule_rebalance()

    async def _deferred_rebalance(self) -> None:
        """Fire a throttled rebalance once the ``rebalance_min_gap_s`` fuse
        clears. No-op if an intervening round already resolved the state
        (``_try_rebalance`` re-checks the gates).
        """
        self._round.retry_pending = False
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
        if self._round.add(message, meta, self._formation.member_keys):
            await self._run_inter_group_admm()

    async def _run_inter_group_admm(self) -> None:
        """Dispatch to the legacy, tier-stratified, or component-scoped path
        by scope/mode.

        The tier-stratified path uses a 2-D ``targets[sector][tier]`` allocation
        (preserving priority intent), falling back to legacy on no per-tier
        deficit. ``"component"`` routes via the elected coordinator.
        """
        if self.admm_scope == "component":
            await self._component.run_scoped()
            return
        if not self.enable_tier_stratified_admm:
            await self._admm.run_legacy()
            return
        # Supply-priority ADMM: fires when there's supply to allocate;
        # priority weighting binds once supply < demand.
        if self._admm.has_supply_priority_work():
            await self._admm.run_supply_priority()
            return
        await self._admm.run_legacy()

    # ------------------------------------------------------------------
    # Peer resolution / component-scoped protocol (see the helper modules)
    # ------------------------------------------------------------------

    def _resolve_sector_peer_addrs(self) -> dict[str, Any]:
        return self._peers.sector_peers()

    def _resolve_component_peer_addrs(self) -> dict[str, Any]:
        return self._peers.component_peers()

    def _component_coordinator_aid(self) -> str | None:
        return self._peers.coordinator_aid()

    async def _handle_component_admm_report(
        self, message: ComponentAdmmReport, meta: dict
    ) -> None:
        await self._component.handle_report(message, meta)

    async def _handle_component_allocation(
        self, message: ComponentAllocation, meta: dict
    ) -> None:
        await self._component.handle_allocation(message, meta)
