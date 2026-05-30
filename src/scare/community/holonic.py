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
    HebbianFlexBeacon,
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
from scare.community.supply_priority_admm import allocate_supply_priority

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
        # ``rebalance_period_s`` was 60.0 originally — designed as a
        # slow "drift catching" heartbeat assuming the reactive path
        # (L1 NegotiationFinishedEvent → L2) would handle real
        # failures.  But the 2026-05-23 cascade work showed L1 gossip
        # often falls dormant after the initial flurry, leaving the
        # first (incomplete, peer-reports-not-in) L2 dispatch as the
        # final allocation for the whole sim.  Lowered to 1.0 s so L2
        # fires every sim-second regardless of L1 activity; the
        # ``rebalance_min_gap_s`` fuse below still bounds reactive
        # bursts.  Each L2 round re-converges to the current correct
        # answer, so any wrong first-round dispatch corrects within
        # one second.
        rebalance_period_s: float = 1.0,
        rebalance_min_gap_s: float = 0.5,
        flex_timeout_s: float = 5.0,
        enable_hebbian_formation: bool = True,
        hebbian_beacon_period_s: float = 4.0,
        hebbian_eta: float = 0.25,
        hebbian_threshold: float = 0.35,
        hebbian_warmup_s: float = 12.0,
        hebbian_delta_tol: float = 0.05,
        hebbian_recluster_tol: float = 0.05,
        watchdog_s: float = 30.0,
        admm_max_iters: int = 50,
        admm_abs_tol: float = 1e-3,
        enable_tier_stratified_admm: bool = True,
        priority_tiers: int = 4,
        admm_mode: str = "demand",
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
        # ADMM scope: "holon" (legacy per-holon, each leader solves over
        # its 4 holon-members), "sector" (deprecated, every holon leader
        # in the sector is one actor), or "component" (current default —
        # every group leader on the same active subgraph is one actor,
        # coordinator elected per (sector, active-component)).  See
        # ``RestorationConfiguration.holon_admm_scope`` for rationale.
        if admm_scope not in {"holon", "sector", "component"}:
            raise ValueError(
                "holon admm_scope must be 'holon', 'sector', or 'component', "
                f"got {admm_scope!r}"
            )
        self.admm_scope = admm_scope
        # Priority-weighted allocation switch.  When False, legacy and
        # supply-priority ADMM use uniform per-tier weights (no-priority
        # ablation).
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
        # Set of node ids hosting a CP agent.  Used by the per-component
        # L2 path to detect whether a CP exists in the leader's multi-
        # sector component — when one does, L2 defers to that CP's L3
        # coordinator (Option B): the L3 coord will ask this leader for
        # flex anyway, and running L2's per-sector solve in parallel
        # would race the L3 dispatch.  Empty set ⇒ L2 always runs (no
        # CPs to defer to; pre-Option-B behaviour).
        self._cp_node_ids: set[Any] = set(cp_node_ids or set())
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
        # Delta gates for event-driven Hebbian publishing/reclustering.
        # ``hebbian_delta_tol`` skips a beacon send when our own δ_g
        # hasn't moved since the last broadcast; ``hebbian_recluster_tol``
        # skips a recluster when no per-peer H entry has moved enough
        # to change membership.  Both are in the same scaled units as
        # δ_g (roughly [-1, 1]).
        self.hebbian_delta_tol = hebbian_delta_tol
        self.hebbian_recluster_tol = hebbian_recluster_tol
        # Long-cadence watchdog for the event-driven schedulers: the
        # role still installs slow periodic safety nets so a leader
        # that missed every trigger event still publishes / reclusters
        # / retries holon formation eventually.
        self.watchdog_s = watchdog_s

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
        # No-change skip for the slow watchdog tick: each reactive
        # trigger (NegotiationFinishedEvent, CPSetpoint,
        # CoalitionConstraint, LocalGenerationRequest) sets this to
        # True; a successful rebalance clears it.  When the watchdog
        # fires with the flag still False we know nothing has moved
        # since the last successful rebalance and can skip the round.
        # ``True`` initially so the first watchdog tick still runs
        # (covers the no-events-since-boot edge case).
        self._rebalance_dirty: bool = True
        # A reactive trigger that arrives inside ``rebalance_min_gap_s``
        # is throttled; without recovery it would sit dirty until the
        # *slow* ``watchdog_s`` tick (30 s) — which never fires inside a
        # short eval sim, stranding the work for seconds (eval task-62:
        # the post-failure component re-cycle was throttled at t=0.10 and
        # not recovered until an unrelated trigger at t≈5.3).  This flag
        # guards a single deferred retry scheduled at gap-expiry so the
        # throttled work runs as soon as the fuse clears, without adding a
        # periodic heartbeat that would fire on stable inputs.
        self._rebalance_retry_pending: bool = False

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
        self._idle_logged: set[str] = set()

        # --- B.2: Hebbian co-variance state ---
        # Each leader keeps a per-peer running mean of (delta_g · delta_h)
        # and a sample count.  Once warmed up, holon membership is the
        # connected component of {peers : H_{gh} > threshold}.
        # addr_str -> (H_gh, samples)
        self._hebbian_H: dict[str, tuple[float, int]] = {}
        # Last own delta_g surrogate broadcast (for diagnostics + own-pair).
        self._last_own_delta_g: float = 0.0
        # Last broadcast δ_g — used by the event-driven beacon to
        # short-circuit a send when our own δ_g hasn't moved.  ``None``
        # forces the first beacon to send unconditionally.
        self._last_sent_delta_g: float | None = None
        # Snapshot of the Hebbian H matrix at the time of the last
        # recluster — used by the event-driven recluster to skip when
        # no entry has moved beyond ``hebbian_recluster_tol``.
        self._last_recluster_H: dict[str, float] = {}
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

        # --- Component-scoped L2 ADMM state (admm_scope="component") ---
        # The component coordinator (lex-smallest group-leader aid
        # among leaders mutually reachable on the active branch
        # subgraph in this sector) buffers ``ComponentAdmmReport``
        # from every reachable group leader and runs ONE ADMM with N
        # actors (= N communities in the component).  Buffer keyed by
        # ``leader_aid`` — latest report per leader wins, so a peer
        # that re-reports its flex in a later round overwrites the
        # older entry.  ``_component_round_counter`` is the
        # coordinator's monotonic round id.  The debounce timer
        # ``_component_dispatch_pending`` collapses a burst of reports
        # into one ADMM solve.
        self._component_round_counter: int = 0
        self._component_report_buffer: dict[str, tuple[str, Any]] = {}
        self._component_dispatch_pending: bool = False
        # Throttle on coordinator-side dispatches so a reactive cascade
        # of reports (e.g. every leader pushing on the same
        # NegotiationFinishedEvent) collapses to one solve.  Same
        # semantics as ``rebalance_min_gap_s`` but tracked separately
        # because the coordinator may receive reports for an idle
        # round between its own community's rebalances.
        self._last_component_dispatch_t: float = float("-inf")
        # Latest service_fraction the coordinator dispatched — kept so
        # newly-arriving reports can be merged against the last result
        # and the coordinator can detect "no material change since last
        # dispatch, skip" cases on the periodic heartbeat.
        self._last_component_fraction: dict[str, dict[int, float]] | None = None

        # --- ComponentAllocation versioning (packet-loss recovery) ---
        # Strictly-monotone counter the coordinator stamps onto each
        # outgoing ``ComponentAllocation``.  Receivers echo the last
        # version they applied on their next ``ComponentAdmmReport``
        # (``last_applied_allocation_version``), so the coordinator
        # detects message loss and re-sends the latest allocation just
        # to the stale receiver — turning the fire-and-forget broadcast
        # into a reliable dispatch under non-zero packet loss.  See the
        # docstring on ``channel.ComponentAllocation.version`` for the
        # eval evidence (task 52, 50% packet loss).
        self._allocation_version_counter: int = 0
        # The latest dispatched allocation, used to re-send to stale
        # leaders on report-receipt.  None until the first dispatch.
        self._last_dispatched_allocation: Any = None  # ComponentAllocation
        # The latest ``version`` this leader has applied as an L2
        # leaf.  Echoed back in every outgoing ``ComponentAdmmReport``.
        # ``-1`` = no allocation applied yet.
        self._last_applied_allocation_version: int = -1

    def setup(self) -> None:
        # Holon formation: event-driven via _on_member_finished /
        # _handle_join_request / repartition events (see below).  A
        # slow watchdog covers the case where every trigger event
        # is missed — e.g. a leader that came up after its peers
        # had already finished initial gossip.
        self.context.schedule_periodic_task(
            self._try_form_holon, delay=self.watchdog_s
        )
        # Inter-group rebalance: PURELY event-driven via the seven
        # reactive handlers
        #   - ``_on_member_finished`` / ``_on_member_finished_local``
        #     (L1 NegotiationFinishedEvent)
        #   - ``_handle_cp_setpoint`` (L3 channel-pattern)
        #   - ``_handle_coalition_constraint`` (L2.5 store)
        #   - ``_handle_local_gen_request`` (L1 fallback escalation)
        #   - ``_handle_l3_wakeup`` (L3 multi-sector dispatch)
        #   - ``_dispatch_service_fractions``'s S1 hook on this leader's
        #     own EBN, which mirrors L1's NFE for the dispatch path
        # plus the slow watchdog below.  No 1 s "drift probe" timer:
        # every cause of drift is itself an event; the heartbeat
        # added redundant ADMM rounds on stable inputs without
        # adding information.  See the 2026-05-24 timing audit.
        self.context.schedule_periodic_task(
            self._try_rebalance, delay=self.watchdog_s
        )
        if self.enable_hebbian_formation:
            # Hebbian beacon: event-driven on local gossip
            # convergence (_on_local_gossip_finished) — the only
            # time our own δ_g actually shifts.  Watchdog keeps the
            # baseline fresh during long silent windows.
            self.context.subscribe_event(
                self,
                NegotiationFinishedEvent,
                self._on_local_gossip_finished,
            )
            self.context.schedule_periodic_task(
                self._hebbian_beacon, delay=self.watchdog_s
            )
            # Reclustering is now triggered inside _handle_hebbian_beacon
            # whenever an incoming beacon shifts any H entry by more
            # than hebbian_recluster_tol.  Watchdog catches drift
            # that no individual update tripped.
            self.context.schedule_periodic_task(
                self._hebbian_recluster, delay=self.watchdog_s,
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
        # b1 (data-refresh only — direct ADMM trigger removed after the
        # first attempt over-restored gas demand by running the holon
        # kernel against still-stale peer summaries; the cascade's
        # existing NegotiationFinishedEvent path picks up the next
        # natural round once every leader's HolonSummaryRole has
        # refreshed its slack_budget_by_sector).
        # No subscription needed here.
        # Locality-respecting prompt L2 re-cycle: react to the propagated,
        # TTL-bounded ``FailureNotice`` (same signal the ProblemDetector
        # gossips from the failed branch's endpoints) so a topology change
        # re-allocates the affected component promptly under the
        # re-elected coordinator — instead of waiting for a downstream L1
        # negotiation to finish.  Sector-filtered: only a same-sector
        # branch failure changes this sector's component connectivity.
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
        # Direct L2.5 -> L2 trigger.  When a cross-holon coalition lands
        # a fresh ``CoalitionConstraint`` for this sector, kick off an
        # L2 rebalance immediately so the merged service fractions
        # (L2's own ADMM result ⊕ coalition's per-tier overrides via
        # the ``CoalitionConstraintStore``) propagate to L1 in the
        # same tick.  Without this, the coalition's constraint sat in
        # the store until the next slow heartbeat (``rebalance_period_s``,
        # default 60 s) — far too late for the 10 s smoke and the
        # dominant reason per-component priority_invariant didn't
        # converge.  ``L2.5`` already broadcasts ``StartBalanceNegotiation``
        # to L1 in parallel for the coalition's pair of tiers, but L2
        # owns the holon-wide allocation across *all* tiers and is the
        # natural place to merge the cross-holon update.
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
        # S2 — L3 multi-sector coord wake-up.  When the L3 ADMM solve
        # dispatches new CP setpoints, the CP-side commit changes per-
        # sector supply/demand for the next LP step.  L2 needs to
        # re-evaluate on the post-CP-commit state, but otherwise has
        # no reactive trigger for the L3 path (L1 may stay quiet
        # between cascades).  ``L3RebalanceWakeup`` is a no-payload
        # nudge that marks ``_rebalance_dirty=True`` via
        # ``_maybe_schedule_rebalance``; the throttle there still
        # bounds firing frequency.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_l3_wakeup),
            lambda msg, meta: isinstance(msg, L3RebalanceWakeup)
            and msg.sector == self.sector,
        )
        # Component-scoped L2 ADMM (admm_scope == "component"): every
        # group leader subscribes to both message types.
        # ``ComponentAdmmReport`` is only acted on if this leader is
        # the elected coordinator for its component (the handler does
        # its own coordinator gate).  ``ComponentAllocation`` is acted
        # on by every group leader — it's the coordinator's dispatch
        # envelope and the addressee applies the per-tier service
        # fractions directly to its own community members (no holon
        # hop).
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

        # LeaderEmerged: a previously-non-leader agent was promoted to
        # lead an orphan sub-community after a failure-driven
        # re-partition (see ``DynamicRepartitionRole``).  Updating
        # ``_leader_node_ids`` keeps ``_resolve_component_peer_addrs``
        # from filtering the new leader out of the component peer set
        # when its ``ComponentAdmmReport`` eventually arrives.  The
        # subscription is synchronous (it just mutates a dict); no
        # task wrap needed.  Filter by sector to keep cross-sector
        # broadcasts out of this role's view.
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

        # Holon-formation event triggers: retry when the
        # eligible-neighbour set could have changed.  The watchdog
        # in setup() is the slow fallback; these triggers cover the
        # cases where the periodic retry would otherwise wait a full
        # ``watchdog_s`` to react to a known-relevant event.
        self.context.subscribe_event(
            self, CommunityReassignedEvent, self._on_community_reassigned
        )

    # ------------------------------------------------------------------
    # Holon formation
    # ------------------------------------------------------------------

    def _on_leader_emerged(self, message: LeaderEmerged) -> None:
        """A previously-non-leader agent was promoted to lead an
        orphan sub-community.  Add it to ``_leader_node_ids`` so
        ``_resolve_component_peer_addrs`` admits its address into the
        component peer set when the topology-mirror filter says it's
        reachable.

        Idempotent.  No-op if the aid is already known (e.g. a
        delayed retransmit).  Records a diagnostic event so the
        promotion shows up in ``events.csv``.
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
        """Repartition just changed our community membership — our
        holon-eligibility set may have moved too.  Kick a formation
        attempt without waiting for the watchdog.

        Also kick a full L2 rebalance.  A failure that islands a leader
        re-elects a fresh component coordinator (the lex-smallest leader
        still reachable); previously the reassignment only re-formed
        holons, so the new coordinator's per-component ADMM did not re-run
        until some *unrelated* L1 negotiation later happened to finish and
        fire ``_on_member_finished`` — leaving the post-failure component
        on the predecessor's stale per-tier allocation for seconds (eval
        task-62: the heat component sat un-reallocated from the t=1.0
        repartition until t≈5.3, and the successor then shed tier-3 while
        the predecessor's stale tier-4 = 1.0 lingered → priority
        inversion).  Re-running the cycle on the topology-change event
        itself makes the re-allocation prompt: every reassigned leader
        re-collects and re-reports to the (re-elected) coordinator, so the
        new component is allocated consistently instead of opportunistically.
        """
        self.context.schedule_instant_task(self._try_form_holon())
        self.context.schedule_instant_task(self._broadcast_recycle())

    async def _on_failure_notice(self, message: FailureNotice, _meta: dict) -> None:
        """A ``FailureNotice`` reached this node — kick a prompt L2
        rebalance so the post-failure component re-allocates immediately
        under the (re-elected) coordinator.

        This is the *locality-respecting* L2 re-cycle trigger.  The notice
        is the same TTL-bounded, sector-tagged gossip that
        ``ProblemDetector`` originates at the failed branch's endpoints
        and forwards through surviving same-sector neighbours, so only
        communities physically reached by the propagation react — no agent
        responds to a failure it could not have detected.  (We do *not*
        subscribe to the global ``BranchFailureEvent``: that would let a
        spatially-distant leader observe a failure it is nowhere near.)

        Without this, the per-component ADMM only re-runs *indirectly* —
        when a downstream L1 negotiation finishes and fires
        ``_on_member_finished`` — so after a failure islands a leader the
        freshly re-elected coordinator can lag seconds behind and then
        allocate over a partial actor set, leaving the new component on
        the predecessor's stale per-tier allocation (eval task-62: the
        heat component sat un-reallocated from the t=1.0 repartition until
        t≈5.3, then the successor shed tier-3 while the predecessor's
        stale tier-4=1.0 lingered → a priority inversion).

        Fires for heat too, even though the heat L1 negotiator
        deliberately ignores the notice (heat *setpoints* are
        constraint/temperature-driven): the L2 *component membership and
        coordinator* are a topology concern, so they must react to a
        topology-change signal regardless of sector.  ``_maybe_schedule_
        rebalance``'s group-leader gate + min-gap throttle keep it cheap
        for non-leaders and collapse failure bursts.
        """
        await self._broadcast_recycle()

    async def _broadcast_recycle(self) -> None:
        """L2 escalation: a topology change reached this leader, so tell
        *every* peer in the current active component to run a fresh
        waterfall, then kick our own.

        This is the "all leaders communicate + re-waterfall" step.  The
        ``FailureNotice`` propagation is TTL-bounded and may reach only a
        few agents near the failure (a single comp leader, or only
        members — eval task-62); fanning the escalation out across the
        component-peer mesh ensures the *re-elected* coordinator — which
        can be many physical hops away — and every leader that owns part
        of the component re-collect and re-report, so the post-failure
        allocation is computed over a complete actor set rather than
        whatever partial reports happened to trickle in.

        Single-hop: peers receive ``from_member=False`` and only rebalance
        (they do not re-broadcast), bounding fan-out.  Only group leaders
        broadcast; non-leaders no-op via the membership gate below.
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

        From a *member* (``from_member=True``): this leader owns that
        member's community — re-broadcast to the whole component so all
        peers re-waterfall.  From a *peer leader* (``from_member=False``):
        just re-collect and re-report; do not re-broadcast (single-hop
        fan-out, no flooding).
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
            # Every candidate rejected.  Schedule another formation
            # attempt now (with whatever neighbour set is current)
            # rather than waiting for the watchdog — a peer that
            # rejected because they were busy may have completed
            # their own formation and is now available again.
            self.context.schedule_instant_task(self._try_form_holon())
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
        self._record_event("holon_formed", f"members={len(accepted_addrs) + 1}")
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
        ``(t, kind, aid, sector, detail)`` shape used throughout this
        role.
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

        Two code paths depending on ``admm_scope``:

        * ``"component"`` (default): every group-topology leader
          collects its OWN community's flex via a single
          ``AskForAvailableFlex`` to self.  The reply (one
          aggregated ``AvailableFlexAnswer`` covering this whole
          community) then drives ``_run_inter_group_admm`` →
          ``_run_component_scoped_admm`` which pushes a
          ``ComponentAdmmReport`` to the component coordinator.
          Holon membership is irrelevant: leaders outside any
          holon still participate, closing the coverage gap the
          sector-scope path had.

        * ``"holon"``/``"sector"`` (legacy): only holon-leaders
          collect flex from their holon-members and run the
          per-holon ADMM (or push to the sector coord in the
          deprecated sector mode).

        Fired periodically (slow heartbeat for input drift) AND
        reactively (on holon formation, on member
        ``NegotiationFinishedEvent``).  Throttled by
        ``rebalance_min_gap_s``.
        """
        now = self.context.current_timestamp
        if (now - self._last_rebalance_t) < self.rebalance_min_gap_s:
            return
        if self._rebalance_active:
            logger.debug("[%s] rebalance skipped: active", self.context.aid)
            return
        # Watchdog skip: when no reactive trigger has fired since the
        # last successful rebalance, the input state hasn't materially
        # moved and the round would re-derive the same allocation.
        # Reactive callers go through ``_maybe_schedule_rebalance``
        # which sets ``_rebalance_dirty`` first, so this check only
        # short-circuits the slow periodic invocation.
        if not self._rebalance_dirty:
            logger.debug(
                "[%s] rebalance skipped: no trigger since last run",
                self.context.aid,
            )
            return

        if self.admm_scope == "component":
            # Component-scope: every group leader participates; no
            # holon-membership gate.  Each leader's own community is
            # represented by one ComponentAdmmReport (= one ADMM
            # actor), so coverage matches the active subgraph.
            if topology_characteristic(self, tid="groups") != "leader":
                return
            # L2 runs in parallel with L3 (no defer).  Per the
            # 2026-05-23 architectural reset: L3 decides cross-sector
            # flows (CP setpoints) and applies them via the LP; the
            # leader's NEXT flex collection then reflects the
            # post-CP state and L2 refines per-sector per-tier
            # service fractions.  The defer-to-L3 path I shipped
            # earlier broke this cascade — L1 gossip then won the
            # equilibrium and PI regressed to 15 % vs 51 % under
            # pure per-sector L2.
            self._rebalance_active = True
            self._last_rebalance_t = now
            # Clear the dirty flag now that we are committed to a run.
            # A new reactive trigger arriving during execution will set
            # it back to True, so the next watchdog tick will still
            # fire; absent any new trigger the watchdog skips.
            self._rebalance_dirty = False
            self._flex_answers = []
            self._flex_answer_senders = []
            # Single self-ask: the EnergyBalanceNegotiator on this
            # agent aggregates the whole community's flex into one
            # AvailableFlexAnswer.
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
            self._log_idle_once("no neighbours")
            return

        self._rebalance_active = True
        self._last_rebalance_t = self.context.current_timestamp
        # Clear the watchdog dirty flag now that a run is committed;
        # new triggers during execution will set it again.
        self._rebalance_dirty = False
        self._flex_answers = []
        self._flex_answer_senders = []
        # The leader contributes its own group's flex too — without that
        # ADMM would be a single-actor problem and bail out early.
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

    async def _handle_coalition_constraint(
        self, message: CoalitionConstraint, meta: dict
    ) -> None:
        """Direct L2.5 -> L2 trigger.

        A fresh cross-holon ``CoalitionConstraint`` carries per-tier
        service fractions the L2.5 coalition has decided we should
        honour.  Without this trigger the constraint sits in the
        shared store and is only consulted on L2's slow heartbeat,
        which means a 60 s round-trip before it reaches L1 — far
        longer than the inversion's lifetime in practice.  Re-running
        L2 now re-merges the holon's local ADMM result with the
        constraint store and re-dispatches the merged service
        fractions to every member group leader.

        Only the holon leader acts; other members would double-trigger.
        The shared ``_maybe_schedule_rebalance`` throttle prevents
        a flood when many coalitions land in rapid succession.
        """
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return
        self._maybe_schedule_rebalance()

    async def _handle_l3_wakeup(
        self, message: L3RebalanceWakeup, meta: dict
    ) -> None:
        """S2 — L3 multi-sector coord just dispatched new CP setpoints
        for this leader's sector.  Mark the L2 path dirty and let the
        existing throttle / scheduler decide when to re-fire.  Gates
        on group-leader membership inside ``_maybe_schedule_rebalance``;
        non-leaders no-op.
        """
        self._maybe_schedule_rebalance()

    async def _handle_local_gen_request(
        self, message: LocalGenerationRequest, meta: dict
    ) -> None:
        """L1 stall escalation handler.

        A member's group leader broadcast ``LocalGenerationRequest``
        over the holons topology because its gossip converged with an
        unresolved residual.  L2's response has two parts:

        1. Trigger an early rebalance — the holon ADMM may absorb the
           residual cross-group before any local DG ramps.
        2. Approve the fallback for the residual, so the originator's
           ``LocalGenerationFallbackRole`` can still fire for whatever
           L2 cannot cover.

        The originator addresses every holon neighbour, so every peer
        runs this handler.  To avoid N approvals for one request, only
        the lex-smallest co-recipient sends the reply; every peer
        triggers the rebalance, which is throttled by
        ``_maybe_schedule_rebalance``.
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
        """Shared throttle + gate logic for both the message- and
        event-driven reactive paths.  Returns silently if any gate
        rejects.  ``_try_rebalance`` itself does its own group-
        leadership / holon-membership checks too, so this method is
        a fast pre-filter that avoids scheduling a no-op task.

        Gates depend on ``admm_scope``:

        * ``"component"``: any group-topology leader can rebalance.
          The component-scope path covers every community on the
          active subgraph; holon membership is irrelevant.
        * ``"holon"``/``"sector"``: only holon-leaders (legacy).
        """
        if self.admm_scope == "component":
            if topology_characteristic(self, tid="groups") != "leader":
                return
        else:
            assignment = self.context.get_or_create_model(HolonicAssignment)
            if assignment.holon_id is None or assignment.parent_addr is not None:
                return
        # Mark that *something* triggered a rebalance attempt.  The
        # watchdog tick reads this flag to decide whether to skip a
        # round when nothing has happened since the last successful
        # rebalance.  Setting it here (before the throttle below)
        # means a throttled reactive trigger still counts as "dirty"
        # — the watchdog will pick up the work next tick instead of
        # losing it.
        self._rebalance_dirty = True
        if self._rebalance_active:
            return
        now = self.context.current_timestamp
        gap_left = (self._last_rebalance_t + self.rebalance_min_gap_s) - now
        if gap_left > 0:
            # Throttled.  Don't strand the dirty work until the slow
            # ``watchdog_s`` tick — schedule a single deferred retry at
            # gap-expiry so it runs as soon as the fuse clears.
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
        """Dispatch to the legacy per-sector ADMM, the tier-stratified
        per-(sector, priority) ADMM (Package C), the sector-wide ADMM
        (admm_scope="sector"), or back to per-holon based on the
        configured scope + mode.

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

        admm_scope == "component": the per-community flex collection
        ran, but instead of solving locally this leader aggregates the
        flex into one community-scale ``ComponentAdmmReport`` and
        pushes it to the elected component coordinator.  The
        coordinator (which may be this leader) runs the per-component
        ADMM once enough reports have arrived and dispatches a
        component-uniform ``ComponentAllocation`` to every leader on
        the same active subgraph.
        """
        if self.admm_scope == "component" and self.admm_mode == "supply":
            await self._run_component_scoped_admm()
            return
        if not self.enable_tier_stratified_admm:
            await self._run_legacy_per_sector_admm()
            return
        if self.admm_mode == "supply":
            # Route A: supply-priority ADMM (per-holon scope).  Fires
            # whenever the holon has any supply at all to allocate;
            # the priority weighting takes over once supply < demand.
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

        triggers = self._resolve_holon_members()
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
        # dimension.  Uses the strictly-monotone schedule (tier 1 → P,
        # tier P → 1) rather than the L1 QP schedule: the L2 ADMM's
        # sharing-distance objective wants a well-conditioned tier
        # ordering, and the L1 QP's tier-1 weight of 0 (hard-locked
        # off-QP) would zero out tier-1 here too — wrong, because L2
        # still needs to rank tier-1 cells first.
        P = self.priority_tiers
        priorities = np.zeros(n_dims)
        for sec in sectors:
            for tier in tiers:
                weight = tier_priority_weight_strict(
                    tier, priority_tiers=P,
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

        try:
            await start_coordinated_optimization(actors, coordinator, start_msg)
        except Exception as exc:
            logger.error(
                "[%s] tier-stratified holon ADMM failed: %s",
                self.context.aid, exc,
            )
            self._record_event("holon_admm_failed", f"tier_stratified: {exc}")
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
        self._record_event(
            "holon_admm_result",
            f"tier_stratified sectors={sectors} tiers={tiers} per_cell={per_cell_summary}",
        )
        # Priority-awareness diagnostic: aggregate absorption per
        # (sector, tier) cell with weight + T so post-run analysis can
        # verify the allocation respects the priority ordering.
        self._record_event(
            "holon_priority_allocation",
            str({
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

        triggers = self._resolve_holon_members()

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
                                # NB: Python evaluates the RHS *before*
                                # the LHS subscript target, so
                                # ``per_tier.setdefault(k, {})[n] =
                                # per_tier[k].get(...)`` raises
                                # ``KeyError`` on the very first
                                # encounter of each ``k`` — the
                                # setdefault on the LHS hasn't run
                                # yet when ``per_tier[k]`` on the RHS
                                # is dereferenced.  Bind the inner
                                # dict to a local first.
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
    #
    # Replaces both the per-holon supply-priority ADMM and the
    # intermediate sector-wide variant with one ADMM scoped to each
    # active connected component of the sector graph.  Each *community
    # leader* (one per group-topology leader) is an ADMM actor — the
    # holon abstraction is not used for optimisation here.
    #
    # Pipeline per round:
    #   1. Every group leader collects its own community's flex (via
    #      the existing ``AskForAvailableFlex`` machinery — sent to
    #      self; the ``EnergyBalanceNegotiator._handle_ask_flex``
    #      aggregates the community's per-(sector, tier) demand and
    #      per-sector supply into one ``AvailableFlexAnswer``).
    #   2. The leader pushes a ``ComponentAdmmReport`` to the
    #      coordinator for its component (lex-smallest leader aid
    #      among leaders mutually reachable on the active branch
    #      subgraph).
    #   3. The coordinator buffers reports keyed by ``leader_aid``,
    #      debounces a burst into one solve, runs the supply-priority
    #      ADMM with N_community actors, and dispatches a
    #      ``ComponentAllocation`` to every leader in the component.
    #   4. Each leader applies the per-tier service fractions to its
    #      OWN community members (sends ``StartBalanceNegotiation``
    #      to self; the local EnergyBalanceNegotiator dispatches the
    #      fractions to every community member via apply_regulate).
    #
    # Key invariants delivered:
    #   * Every load at the same tier in the same (sector, active-
    #     component) is served at the same fraction.  Cross-leader
    #     inversions cannot arise — they would require two leaders
    #     in the same component to land different per-tier fractions,
    #     which the per-component solve produces only one of.
    #   * Every group leader in the component participates in both
    #     the input (flex report) and the output (allocation dispatch)
    #     — closing the coverage gap the holon-leader-only path had.
    #   * A failure that splits a sector into two components re-elects
    #     two coordinators that decide independently for their halves.

    def _resolve_sector_peer_addrs(self) -> dict[str, Any]:
        """Return ``{leader_aid: leader_addr}`` for every same-sector
        leader on the ``holon_summary_<sector>`` topology, INCLUDING
        self.  Built from ``topology_neighbors`` plus the local
        context address.  Used by ``_resolve_component_peer_addrs``
        as the unfiltered baseline; not consulted directly by the
        component-scoped path.
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
        leader that is mutually reachable on the active branch
        subgraph — i.e. lives in the same connected component of the
        sector graph as this leader.  Falls back to the unfiltered
        sector peer set when topology mirror or my own node id are
        unavailable (defensive — keeps the path runnable in
        configurations that don't wire the mirror in).

        Coordinator-election eligibility: the ``holon_summary_<sector>``
        mesh also carries CP / branch agents (they subscribe as L3
        readers — see ``CPPriorityAdmmRole``).  Those agents do NOT
        host a ``HolonicCommunityRole`` for this sector, so a
        ``ComponentAdmmReport`` routed to one of them as the
        lex-smallest "peer" is silently dropped, and the per-component
        L2 solve never runs.  Filter the sector-peer set down to known
        leader aids (with self always included) to keep CPs out of
        the election.  When ``_leader_node_ids`` is empty (degenerate /
        not-wired configurations), fall back to the unfiltered set.
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
                # Always include self — the "component a leader is in"
                # contains the leader, even if leader_node_ids has no
                # entry for some reason.
                out[aid] = addr
                continue
            node_id = self._leader_node_ids.get(aid)
            if node_id is None or node_id in reachable:
                out[aid] = addr
        return out

    def _component_coordinator_aid(self) -> str | None:
        """Lex-smallest aid among current component peers — the
        coordinator for this leader's component.  Returns None only
        when even self is missing from the peer set (defensive — the
        caller falls back to the per-holon path in that case).
        """
        peers = self._resolve_component_peer_addrs()
        if not peers:
            return None
        return min(peers.keys())

    def _multi_sector_l3_active(self) -> bool:
        """True iff this leader sits in a multi-sector connected
        component containing at least one CP agent — meaning an L3
        coordinator is responsible for the L2 decision here and this
        leader's per-sector L2 should defer (Option B).

        Uses ``topology_mirror.reachable_from(..., allow_cp_bridges=
        True)`` to traverse the joint multi-sector subgraph (active
        branches AND active CP edges), then intersects with the
        statically-known set of CP host node ids.  Empty intersection
        ⇒ no CP in this component ⇒ L2 runs locally as today.
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

    def _aggregate_holon_flex(self) -> tuple[
        dict[str, float],
        dict[str, dict[int, float]],
        dict[str, dict[int, float]],
    ]:
        """Roll up the holon's collected ``AvailableFlexAnswer`` list
        into one (supply_by_sector, demand_by_sector_priority,
        served_by_sector_priority) triple — the inputs the supply-
        priority ADMM consumes per actor.

        Called once per sector-scoped round, just before pushing a
        ``SectorAdmmReport`` upstream.  The same answers are kept on
        ``self._flex_answers`` so the per-holon legacy fallback
        remains callable.
        """
        supply: dict[str, float] = {}
        demand: dict[str, dict[int, float]] = {}
        served: dict[str, dict[int, float]] = {}
        for a in self._flex_answers:
            for sec, val in (a.supply_by_sector or {}).items():
                supply[sec] = supply.get(sec, 0.0) + float(val)
            for sec, tmap in (a.demand_by_sector_priority or {}).items():
                bucket = demand.setdefault(sec, {})
                for tier, mw in tmap.items():
                    bucket[int(tier)] = bucket.get(int(tier), 0.0) + float(mw)
            for sec, tmap in (a.served_by_sector_priority or {}).items():
                bucket = served.setdefault(sec, {})
                for tier, mw in tmap.items():
                    bucket[int(tier)] = bucket.get(int(tier), 0.0) + float(mw)
        return supply, demand, served

    async def _run_component_scoped_admm(self) -> None:
        """Component-scoped variant of ``_run_supply_priority_admm``.

        The leader has already collected its community's flex answers
        via the same path the per-holon route uses (one
        ``AskForAvailableFlex`` self-message → one
        ``AvailableFlexAnswer`` aggregating the whole community).
        Branch:

        * coordinator (lex-smallest aid in this leader's active
          component) → stash own report in the buffer + schedule a
          debounced component ADMM solve (so a burst of own/peer
          reports collapses to a single ADMM round).
        * non-coordinator → push a ``ComponentAdmmReport`` to the
          coordinator and wait for the ``ComponentAllocation`` that
          comes back.

        Either way, this leader's per-holon ADMM does *not* run —
        the component coordinator's result will arrive via
        ``_handle_component_allocation`` and be applied to community
        members on receipt.  ``_flex_answers`` is drained here so a
        follow-up reactive trigger doesn't re-fire on the stale
        buffer.
        """
        if not self._rebalance_active:
            return
        answers = self._flex_answers[:]
        # Drain — same lifecycle as the per-holon ADMM's snapshot.
        self._flex_answers = []
        self._flex_answer_senders = []
        self._flex_expected = 0
        self._rebalance_active = False

        if not answers:
            return

        supply, demand, served = self._aggregate_holon_flex_from(answers)
        # Skip only when there's literally nothing to contribute
        # (neither supply nor demand).  A demand-only community must
        # still report so the coordinator sees its demand in the
        # sector-wide T vector; a supply-only community must still
        # report so its supply enters the holon-wide supply pool.
        # Earlier sector-scope code used ``and`` here, silently
        # excluding load-only communities — the headline cause of
        # the dispatch coverage gap that produced apparent
        # priority inversions in the 2026-05-22 smoke.
        any_supply = any(v > 1e-9 for v in supply.values())
        any_demand = any(
            mw > 1e-9 for tmap in demand.values() for mw in tmap.values()
        )
        if not (any_supply or any_demand):
            return

        coord_aid = self._component_coordinator_aid()
        if coord_aid is None:
            # No peer topology available; degenerate to the per-holon
            # legacy path so we don't drop the work entirely.
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
            # Implicit-ACK: echo the latest ComponentAllocation version
            # we have applied so the coordinator can detect this
            # leader missed the previous dispatch and re-send.
            last_applied_allocation_version=self._last_applied_allocation_version,
        )

        if coord_aid == leader_aid:
            # I'm the coordinator — buffer my own report and trigger
            # the debounced solve.
            self._component_report_buffer[leader_aid] = (round_id, report)
            await self._maybe_run_component_admm(reason="self_report")
            return

        # Push to the coordinator.  No reply timeout: if the
        # coordinator silently drops, the next reactive trigger will
        # push again (idempotent on the coordinator side since the
        # buffer is keyed by leader_aid).
        peers = self._resolve_component_peer_addrs()
        coord_addr = peers.get(coord_aid)
        if coord_addr is None:
            # Coordinator aid present but address not resolved — fall
            # back to the per-holon path so this round still produces
            # some allocation.
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

    def _aggregate_holon_flex_from(
        self, answers: list[AvailableFlexAnswer]
    ) -> tuple[
        dict[str, float],
        dict[str, dict[int, float]],
        dict[str, dict[int, float]],
    ]:
        """Stateless variant of :meth:`_aggregate_holon_flex` that
        operates on a passed-in answer list.  Used by the sector-scope
        path which has already drained ``self._flex_answers`` into a
        local snapshot.
        """
        prev = self._flex_answers
        self._flex_answers = answers
        try:
            return self._aggregate_holon_flex()
        finally:
            self._flex_answers = prev

    async def _handle_component_admm_report(
        self, message: ComponentAdmmReport, meta: dict
    ) -> None:
        """Coordinator-side: buffer the report from a peer leader and
        schedule the debounced ADMM solve.  Non-coordinator leaders
        silently drop the message.

        Buffer is keyed by ``leader_aid`` — a later report from the
        same leader overwrites the earlier one (the freshest flex
        view wins).  The coordinator's own report is buffered the
        same way via ``_run_component_scoped_admm``.

        Defensive: drop reports from peers we no longer consider in
        our component (e.g. a branch failure cut them off after the
        sender pushed but before we received).  Keeps the actor set
        consistent with the current active subgraph.
        """
        if self.admm_scope != "component":
            return
        if self._component_coordinator_aid() != self.context.aid:
            return
        # L2 runs in parallel with L3; do NOT defer here.  See the
        # comment in ``_try_rebalance``'s component branch.
        # Filter: only buffer reports from leaders still in our active
        # component view.  A peer that's now disconnected shouldn't
        # contribute to OUR sub-component's allocation.
        component_peers = self._resolve_component_peer_addrs()
        if message.leader_aid not in component_peers:
            return
        self._component_report_buffer[message.leader_aid] = (
            message.round_id, message
        )
        # Packet-loss recovery: if the sender's echoed
        # ``last_applied_allocation_version`` is behind the latest
        # version we dispatched, re-send the stashed allocation to
        # JUST this peer.  Idempotent on the receiver (a stale or
        # equal-version retransmit is ignored at apply time).  Wrapped
        # in try/except because the legacy ComponentAdmmReport had no
        # such field — getattr keeps the path safe.
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
        when its echoed ``last_applied_allocation_version`` is behind
        ``self._allocation_version_counter``.

        Trigger: a peer's ``ComponentAdmmReport`` echoes the version
        it last applied; if it is strictly less than our latest
        dispatch, the original ComponentAllocation was lost in
        transit (packet loss is the dominant cause — see
        ``channel.ComponentAllocation`` docstring + task 52 evidence).

        Idempotency: the leaf's apply path ignores a
        ``ComponentAllocation`` whose ``version`` is ≤ the already-
        applied version, so a benign duplicate after a real drop is
        a no-op.

        Self-skip: the coordinator's own seat doesn't send to itself
        — the dispatch loop in ``_run_component_admm_now`` already
        included it.

        Records a ``component_alloc_resent`` diagnostic event so the
        recovery is visible in the events ledger.
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
        """Debounce + dispatch.  Collapses a burst of incoming reports
        into one ADMM solve when they arrive within
        ``rebalance_min_gap_s`` of each other, so the
        post-NegotiationFinishedEvent cascade doesn't fan out into
        many solves.

        Coalesces by re-scheduling: each report sets
        ``_component_dispatch_pending`` and the *first* arrival owns
        the actual solve invocation.  This produces "one solve per
        burst" with bounded latency.
        """
        if self._component_dispatch_pending:
            return
        self._component_dispatch_pending = True
        self.context.schedule_instant_task(
            self._run_component_admm_now(reason=reason),
        )

    async def _run_component_admm_now(self, *, reason: str) -> None:
        """Run the per-component ADMM against the current report
        buffer and dispatch the resulting per-tier service fractions
        to every same-component leader (including self).

        Idempotent: callers debounce via
        ``_component_dispatch_pending``; this method clears the flag
        when it returns.  The buffer is NOT drained — late reports
        for the same leader overwrite, and the next solve picks up
        the freshest entries.
        """
        try:
            await self._run_component_admm_now_inner(reason=reason)
        finally:
            self._component_dispatch_pending = False
            self._last_component_dispatch_t = float(self.context.current_timestamp)

    async def _run_component_admm_now_inner(self, *, reason: str) -> None:
        if not self._component_report_buffer:
            return
        # Only include leaders still on this coordinator's component
        # view.  A leader that's been disconnected since pushing its
        # last report shouldn't participate in this round's
        # allocation; the report stays in the buffer (overwriting is
        # cheap) but the round skips it.
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

        sectors = sorted({
            s for r in reports for s in (r.demand_by_sector_priority or {})
        })
        if not sectors:
            return
        tiers_present: set[int] = set()
        for r in reports:
            for tmap in (r.demand_by_sector_priority or {}).values():
                tiers_present.update(tmap.keys())
        tiers = sorted(t for t in tiers_present if t >= 1)
        if not tiers:
            return
        total_demand = sum(
            float(d) for r in reports
            for tmap in (r.demand_by_sector_priority or {}).values()
            for d in tmap.values()
        )
        if total_demand < 1e-6:
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

        # NB: deliberately no sub-tolerance noise scrub here.  Tried
        # mirroring L3's clamp-below-1e-3 in 2026-05-23 and it
        # regressed PI rather than helping — the first L2 round
        # often runs before all peer reports have arrived, producing
        # near-zero fractions on tiers whose demand was missing from
        # this round's actor set.  Scrubbing those to exact 0 then
        # locked them in via the per-load ``cooldown_s`` gate, since
        # the next (complete) L2 round's correct fraction landed
        # inside the cooldown window and got suppressed.  The PI
        # claim's 1e-3 tolerance already absorbs unscrubbed
        # per-sector noise; we don't gain anything by clamping.

        # G1a — complete + monotone per-tier vector.  A component round
        # may solve over only a subset of tiers (the actors that reported
        # and are currently reachable).  Dispatching just those tiers lets
        # a lower-priority tier keep a stale, higher fraction from an
        # earlier round: the eval task-88 / task-51 inversions, where a
        # later {1,2,3} round shed tier-3 without re-touching a tier-4 an
        # earlier round had set to 1.0, leaving tier-4 served *above* the
        # shed tier-3 in the same component.  Fold the previous dispatch's
        # tiers in to fill the gaps (fresh values win), then clamp the
        # whole vector non-increasing in tier number so the priority
        # ordering (tier 1 ≥ tier 2 ≥ …) holds by construction over the
        # tiers this coordinator has actually allocated.  The waterfall
        # allocator is already monotone over the tiers it solves; this
        # only corrects the cross-round carry-forward.
        #
        # NB: deliberately bounded to tiers present in this coordinator's
        # own solve+history — NOT a force over all P tiers.  The
        # coordinator-handoff inversion (task-62: a fresh successor
        # coordinator that never inherits the predecessor's tier-4 state)
        # is out of reach here and needs the sector-wide L2.5
        # reconciliation; forcing unsolved/unknown tiers down at dispatch
        # over-sheds and regressed the single-coordinator cases this fix
        # targets (eval sweep 2026-05-28).
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

        # Dispatch the same ComponentAllocation to every leader in the
        # active component (including self).  Each leader handles it
        # via ``_handle_component_allocation`` and applies the
        # fractions directly to its own community members.
        round_id = max(
            (self._component_report_buffer[a][0] for a in leader_aids),
            default="",
        )
        now = float(self.context.current_timestamp)
        # Bump the per-coordinator allocation version BEFORE building
        # the message so leaf-side ACK echoes line up with the dispatch
        # we're about to send.  Stash the message for re-sends to
        # stale leaders (see ``_resend_allocation_if_stale``).
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
        """Leaf-side handler: a ComponentAllocation arrived from the
        coordinator.  Apply the per-tier service fraction to this
        leader's OWN community members (no holon hop) via the L1
        honour path.

        Coverage: every group leader on the holon_summary topology
        for this sector handles this message — *not* gated on holon
        membership.  Communities outside any holon used to fall
        through the sector-scope dispatch; this is the closes-the-
        coverage-gap fix from 2026-05-23.

        Coalition merge: if the local coalition store carries an
        active fraction for any tier, that fraction wins per-tier
        (mirrors the legacy ``_run_supply_priority_admm`` behaviour).
        """
        if self.admm_scope != "component":
            return
        # Defensive: leaders only act on this if they are still
        # group-topology leaders.  The subscription is at agent
        # level; non-leader agents shouldn't have one but the gate
        # is cheap and protects against drift.
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Rebuild a {sector_value: {tier: frac}} envelope from the
        # single-sector message so the existing L1 honour path can
        # consume it unchanged.
        service_fraction: dict[str, dict[int, float]] = {
            self.sector.value: dict(message.service_fraction_by_tier),
        }
        if self._coalition_constraint_store is not None:
            now = float(self.context.current_timestamp)
            service_fraction = self._coalition_constraint_store.merge_into(
                service_fraction, self.sector, now,
            )
        # Send StartBalanceNegotiation to SELF — this agent's own
        # EnergyBalanceNegotiator handles it via _dispatch_service_
        # fractions, applying factor=service_fraction[sec][tier] to
        # every community member (the live group-topology neighbours).
        # That covers the whole community, including loads outside
        # any holon.
        await self.context.send_message(
            StartBalanceNegotiation(
                service_fraction_by_sector_priority=service_fraction,
            ),
            receiver_addr=self.context.addr,
        )
        # ACK channel: record the version we just applied so the next
        # outgoing ``ComponentAdmmReport`` echoes it.  The coordinator
        # uses the echo to detect ComponentAllocation drops under
        # packet loss and re-sends to stale leaders.  A retransmit
        # with a stale ``version`` (≤ already-applied) is ignored.
        try:
            if int(message.version) > self._last_applied_allocation_version:
                self._last_applied_allocation_version = int(message.version)
        except (TypeError, ValueError):
            # Defensive — legacy ComponentAllocation (no version field)
            # still applies normally; ack stays at -1 so the
            # coordinator interprets the report as "stale or first".
            pass
        self._record_event(
            "holon_priority_allocation",
            f"component_scope round={message.round_id} "
            f"version={getattr(message, 'version', 0)} "
            f"fractions={service_fraction}",
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

    def _on_local_gossip_finished(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        """Local L1 gossip just converged — δ_g has shifted, so kick
        the (delta-gated) Hebbian beacon.  Same-agent local-event
        bus; ``_hebbian_beacon`` itself checks role/topology gates.
        """
        if event.sector != self.sector:
            return
        if not self.enable_hebbian_formation:
            return
        self.context.schedule_instant_task(self._hebbian_beacon())

    async def _hebbian_beacon(self) -> None:
        """Broadcast own δ_g to same-sector neighbours, but only when
        it has moved beyond ``hebbian_delta_tol`` since the last send.

        Fires on the local ``NegotiationFinishedEvent`` (the only time
        the leader's own balance can actually change) and on a slow
        watchdog so a leader whose triggers were all missed still
        publishes eventually.  The self-ask refreshes
        ``_peer_last_delta[own_key]`` for the *next* call — the
        current call uses whatever was cached from the previous
        self-ask reply (or 0 on first invocation).
        """
        if not self.enable_hebbian_formation:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Self-ask so the *next* beacon (or this one's δ gate
        # comparison) reads a fresh δ_g.  Reply lands asynchronously
        # via ``_handle_flex_answer``.
        await self.context.send_message(
            AskForAvailableFlex(include_connectors=False),
            receiver_addr=self.context.addr,
        )

        own_key = str(self.context.addr)
        delta_g = self._peer_last_delta.get(own_key, self._last_own_delta_g)
        self._last_own_delta_g = delta_g

        # Delta gate: skip the broadcast if our own δ_g hasn't moved
        # enough.  ``None`` sentinel forces the first send through so
        # peers always see at least one beacon to learn our address.
        if (
            self._last_sent_delta_g is not None
            and abs(delta_g - self._last_sent_delta_g) < self.hebbian_delta_tol
        ):
            return

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
        self._last_sent_delta_g = float(delta_g)

    async def _handle_hebbian_beacon(
        self, message: HebbianFlexBeacon, meta: dict
    ) -> None:
        """Update the per-peer Hebbian co-variance estimate, then
        trigger a recluster only when the new H entry has moved
        beyond ``hebbian_recluster_tol`` since the last recluster.
        """
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

        # H-delta gate: defer reclustering to the watchdog unless this
        # update could plausibly change membership.  Comparison is
        # against the snapshot taken at the last recluster, so a
        # sequence of small updates that sum to a meaningful drift
        # still trips the gate.
        last_h = self._last_recluster_H.get(sender_key, 0.0)
        if abs(new_h - last_h) >= self.hebbian_recluster_tol:
            self.context.schedule_instant_task(self._hebbian_recluster())

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
        if not self.enable_hebbian_formation:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        if self.context.current_timestamp < self._hebbian_warmup_until:
            self._record_event("hebbian_recluster_attempted", "warmup")
            return
        assignment = self.context.get_or_create_model(HolonicAssignment)
        if assignment.holon_id is None or assignment.parent_addr is not None:
            return  # not a leader of a formed holon

        candidates = self.hebbian_membership_candidates()
        if not candidates:
            self._record_event("hebbian_recluster_attempted", "no_candidates")
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
        # Snapshot H after every recluster attempt (even when membership
        # didn't change) so the H-delta gate in _handle_hebbian_beacon
        # measures drift since *this* point, not since the last
        # successful membership change.  Without this, a leader whose
        # membership stays stable would keep tripping the gate on
        # every incoming beacon.
        self._last_recluster_H = {
            k: h for k, (h, _n) in self._hebbian_H.items()
        }
        if new_keys == prev_keys:
            self._record_event("hebbian_recluster_attempted", "no_drift")
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

        self._record_event(
            "hebbian_recluster",
            f"members={len(new_keys)} drift={len(new_keys ^ prev_keys)}",
        )
