"""Layer 2.5 — holon-summary mesh + cross-holon coalition formation.

Detection: every group leader periodically publishes a
:class:`scare.base.channel.HolonSummary` on a sector-wide full-mesh
topology (``holon_summary_<sector>``) carrying its community's per-tier
served-MW and demand-MW, subscribes to peers' summaries, and runs a
local cross-holon priority-inversion check.

Coalition: on detection, a deterministically-elected initiator forms an
ad-hoc coalition with the contributing leaders, runs a scoped
supply-priority allocation over the union of their flex, and broadcasts
per-tier service-fraction constraints the members apply as
``StartBalanceNegotiation(service_fraction_by_sector_priority=...)`` on
their own L1 dispatch.

Separate from :class:`HolonicCommunityRole` because this layer is
observability + scoped cooperation only: L2's chunked-clique ADMM keeps
running on its slow heartbeat and the coalition is additive (TTL-bounded
constraints honoured by L1). The coalition also observes every
same-sector leader, a wider scope than L2's chunk-mates.

Detection rule: aggregate all summaries (peers + self), per tier sum
served/demand, compute ``frac[t] = served / demand`` (1.0 when demand
is 0). Inversion when a higher-priority tier ``t_h`` has strictly
smaller ``frac`` than a lower-priority tier ``t_l > t_h`` beyond
``inversion_tol``. Mirrors the priority-invariant claim in
``experiment/eval/claims.py`` so detector and claim agree.

Initiator election: the lex-smallest publisher with non-empty summary
state is the unique initiator per round. Stable under eventual
consistency since membership doesn't change at sub-second rates;
non-initiators run detection but suppress the event + coalition
broadcast, collapsing N duplicate detections into one per inversion.

Coalition lifecycle:
1. Invitation — initiator sends ``CoalitionInvitation`` to peers whose
   summary contributed to the inverted tier pair, plus itself.
2. Acceptance — each invited leader replies with ``CoalitionAcceptance``
   carrying its per-tier supply/demand slice.
3. Allocation — after ``accept_window_s`` the initiator runs a
   supply-priority allocation over the collected acceptances. Supply is
   fungible within a sector, so the initiator holds the aggregate and
   the per-actor agreement step L2's ADMM existed for is moot.
4. Constraint dispatch — initiator sends ``StartBalanceNegotiation`` to
   each accepting member and records the constraint active locally,
   re-broadcasting every L2.5 tick while the TTL holds.

Constraint invalidation:
- ``now > issued_at + ttl_s``: expiry; control returns to L2's ADMM.
- ``BranchFailureEvent`` in-sector: topology changed enough that the
  computed fractions no longer match the live grid; dropping lets the
  failure-retriggered L2 ADMM redecide.

Eventual consistency: no freeze flag / two-phase commit. Last-write-wins
on ``StartBalanceNegotiation`` at L1 means an L2 rebalance between ticks
may briefly override a coalition constraint; the next tick re-asserts.
On TTL expiry L2's allocation takes over cleanly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_characteristic, topology_neighbors

from scare.base.channel import (
    CoalitionAcceptance,
    CoalitionConstraint,
    CoalitionInvitation,
    CPCommitment,
    HolonSummary,
    MonotonicVersion,
)
from scare.base.diagnostics import record_event
from scare.base.model import NegotiationFinishedEvent, Sector, StartBalanceNegotiation
from scare.community.coalition_store import CoalitionConstraintStore
from scare.community.deliverability import per_actor_deliverable_caps
from scare.community.supply_priority_admm import allocate_supply_priority
from scare.base.util import (
    lookup_slack,
    lookup_slack_eff_budget,
    obs_capacity,
    obs_priority,
    obs_sector,
    obs_setpoint,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


@dataclass
class _PendingCoalition:
    """Initiator-side state during the invitation/acceptance window.

    ``acceptances[aid]`` collects each peer's reply; the initiator's own
    entry is filled synchronously since it skips the round-trip.
    ``addr_by_aid`` keeps a sendable address per aid for constraint
    dispatch after allocation. ``run`` flips True once allocated so a
    late acceptance doesn't re-trigger it.
    """

    coalition_id: str
    sector: Sector
    target_tiers: tuple[int, ...]
    member_aids: tuple[str, ...]
    started_at: float
    addr_by_aid: dict[str, Any] = field(default_factory=dict)
    acceptances: dict[str, CoalitionAcceptance] = field(default_factory=dict)
    run: bool = False


@dataclass
class _ActiveCoalition:
    """Initiator-side TTL record of an allocated coalition.

    Re-asserted every ``_tick`` until ``issued_at + ttl_s`` passes or a
    same-sector ``BranchFailureEvent`` invalidates it early.
    ``member_addrs`` holds only accepting members, so a declining peer
    is never overwritten by a fraction it didn't opt into.
    """

    coalition_id: str
    sector: Sector
    service_fraction_by_tier: dict[int, float]
    member_addrs: list[Any]
    issued_at: float
    ttl_s: float


def _xs_registry(
    behavior: "RestorationEnvironmentBehavior",
) -> dict[Sector, dict[str, HolonSummary]]:
    """Per-behavior shared registry of latest HolonSummary by sector.

    Lazy-init, single-process; a distributed deployment would replace
    this with a real cross-sector publish path.
    """
    store = getattr(behavior, "_scare_xs_summaries", None)
    if store is None:
        store = {}
        behavior._scare_xs_summaries = store
    return store


@dataclass
class _ActiveCrossSectorCoalition:
    """Initiator-side TTL record of an allocated cross-sector coalition.

    Distinct from :class:`_ActiveCoalition` because the dispatch spans
    multiple sectors AND includes CP commitments, each fired on its own
    channel per re-assert: sector-keyed service fractions to matching
    leaders; directional flows to each CP's address for the TTL window.
    """

    coalition_id: str
    service_fraction_by_sector_tier: dict[str, dict[int, float]]
    leader_addrs_by_sector: dict[str, list[Any]]
    cp_targets_mw: dict[str, dict[str, float]]  # cp_aid -> sector_v -> mw
    cp_addrs: dict[str, Any]
    sectors: tuple[Sector, ...]
    issued_at: float
    ttl_s: float


class _CoalitionAggregate(NamedTuple):
    """Per-sector aggregation across a coalition's accepting members."""
    total_supply: float
    total_observed_served: float
    demand_by_tier: dict[int, float]
    served_by_tier: dict[int, float]
    actor_supplies: list[dict[str, float]]
    actor_demands: list[dict[str, dict[int, float]]]
    actor_node_ids: list[Any]
    actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]]


class HolonSummaryRole(Role):
    """Periodic publisher + subscriber for cross-holon priority
    observability, plus coalition formation.

    Installed on every group leader (next to
    :class:`HolonicCommunityRole`). Non-leaders stay quiescent: ``setup``
    runs but the leader-check at the top of ``_tick`` returns early, so
    no publish fires.
    """

    def __init__(
        self,
        behavior: "RestorationEnvironmentBehavior",
        sector: Sector,
        *,
        period_s: float = 1.0,
        watchdog_s: float = 30.0,
        inversion_tol: float = 1e-3,
        enable_coalition: bool = True,
        coalition_accept_window_s: float = 1.0,
        coalition_constraint_ttl_s: float = 8.0,
        priority_tiers: int = 4,
        admm_max_iters: int = 50,
        admm_abs_tol: float = 1e-3,
        my_node_id: Any = None,
        member_node_ids: dict[str, Any] | None = None,
        mirror: Any = None,
        constraint_store: CoalitionConstraintStore | None = None,
        enable_cross_sector_coalitions: bool = False,
        cp_meta: dict[str, dict[str, Any]] | None = None,
        peer_leader_addrs: dict[Sector, dict[str, Any]] | None = None,
        enable_heat_cp_supply: bool = False,
        heat_refresh_s: float = 2.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.period_s = period_s
        # Heat→L3 link: heat's normal summary triggers (L1 gossip finish /
        # L2 dispatch) are off (MW-balance deactivated), so a heat leader's
        # served/demand vector would freeze until the watchdog. Refresh on
        # this faster cadence so the delivered-heat deficit reaches the
        # CP-ADMM. Heat-scoped.
        self.enable_heat_cp_supply = bool(enable_heat_cp_supply)
        self.heat_refresh_s = float(heat_refresh_s)
        # Slow safety-net cadence to re-run publish + invariant check +
        # coalition re-assert even when nothing moved, so a late-joining
        # peer sees the current version frontier and an active coalition
        # stays renewed. The dominant trigger is event-driven (see setup).
        self.watchdog_s = watchdog_s
        # Cached last-published vectors so the event-driven publisher can
        # skip when nothing moved by more than ``inversion_tol`` (the same
        # "material change" threshold detection uses).
        self._last_published_served: dict[int, float] = {}
        self._last_published_demand: dict[int, float] = {}
        self.inversion_tol = inversion_tol
        self.enable_coalition = enable_coalition
        self.coalition_accept_window_s = coalition_accept_window_s
        self.coalition_constraint_ttl_s = coalition_constraint_ttl_s
        self.priority_tiers = priority_tiers
        self.admm_max_iters = admm_max_iters
        self.admm_abs_tol = admm_abs_tol
        # Spatial wiring for deliverability-aware coalition allocation.
        # ``my_node_id``: this leader's monee node. ``member_node_ids``:
        # owned-member aid → monee node (for the per-tier demand-location
        # map in the acceptance payload). ``mirror``: shared
        # :class:`GridTopologyMirror` whose ``reachable_from`` drives the
        # per-actor cap. Any being None degrades to raw-supply ADMM
        # (no deliverability caps).
        self._my_node_id = my_node_id
        self._member_node_ids: dict[str, Any] = dict(member_node_ids or {})
        self._mirror = mirror
        # Shared store between L2.5 (writer) and L2 (reader) on the same
        # leader. None ⇒ no binding: the coalition still dispatches its
        # StartBalanceNegotiation, but L2's later rounds overwrite per-tier
        # without checking for active coalitions.
        self._constraint_store = constraint_store
        self._version = MonotonicVersion()
        # Most-recent ``HolonSummary`` per publisher; addr-book populated
        # from incoming summary metadata for direct dispatch.
        self._peer_summaries: dict[str, HolonSummary] = {}
        self._peer_addrs: dict[str, Any] = {}
        # Inversion cooldown (one emit per window, prevents event spam).
        # ``period_s`` so a persistent inversion is re-detected (fresh
        # coalition) on the next tick — needed for the per-component
        # priority-invariant to converge while L2's rebalance is on its
        # slow heartbeat.
        self._last_inversion_emit_t: float = -1e9
        self._inversion_cooldown_s: float = period_s
        # Coalitions keyed by id so parallel coalitions can coexist.
        self._pending_coalitions: dict[str, _PendingCoalition] = {}
        self._active_coalitions: dict[str, _ActiveCoalition] = {}
        self._coalition_counter: int = 0

        # ---- Cross-sector coalition state ----
        # When enabled, cross-sector invariants run after intra-sector
        # checks and may open coalitions spanning a CP.
        self.enable_cross_sector_coalitions = enable_cross_sector_coalitions
        # cp_aid -> {sectors, coupling_ratios, rated_capacity_mw, addr}.
        self._cp_meta: dict[str, dict[str, Any]] = dict(cp_meta or {})
        # sector -> {aid -> addr} for cross-sector invitations.
        self._peer_leader_addrs: dict[Sector, dict[str, Any]] = dict(
            peer_leader_addrs or {}
        )
        self._last_xs_inversion_emit_t: float = -1e9
        self._active_xs_coalitions: dict[str, _ActiveCrossSectorCoalition] = {}

    @property
    def _topology_tid(self) -> str:
        return f"holon_summary_{self.sector.value}"

    def setup(self) -> None:
        logger.debug(
            "[%s] HolonSummaryRole setup: sector=%s period_s=%.2f tid=%s",
            self.context.aid, self.sector.value,
            self.period_s, self._topology_tid,
        )

        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        # Subscribe to summaries from same-sector peers.
        self.context.subscribe_message(
            self,
            _wrap(self._on_summary),
            lambda msg, meta: isinstance(msg, HolonSummary)
            and msg.sector == self.sector,
        )
        # Coalition control-plane subscriptions, sector-filtered so a
        # leader is never pulled into another sector's coalition.
        self.context.subscribe_message(
            self,
            _wrap(self._on_invitation),
            lambda msg, meta: isinstance(msg, CoalitionInvitation)
            and msg.sector == self.sector,
        )
        self.context.subscribe_message(
            self,
            _wrap(self._on_acceptance),
            lambda msg, meta: isinstance(msg, CoalitionAcceptance)
            and msg.sector == self.sector,
        )
        # Inbound coalition constraints from other initiators, stored so
        # this leader's L2 ADMM consults them before dispatching its own
        # fractions (coalition wins per (sector, tier) cell while TTL valid).
        self.context.subscribe_message(
            self,
            _wrap(self._on_constraint),
            lambda msg, meta: isinstance(msg, CoalitionConstraint)
            and msg.sector == self.sector,
        )
        # Event-driven publish: the per-tier vector only moves on L1
        # gossip convergence (NegotiationFinishedEvent) or L2 dispatch
        # (StartBalanceNegotiation), so subscribe to both. ``_publish``
        # short-circuits on unchanged state, so event bursts don't
        # republish identical vectors.
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_local_state_change
        )
        self.context.subscribe_message(
            self,
            _wrap(self._on_l2_dispatch),
            lambda msg, meta: isinstance(msg, StartBalanceNegotiation),
        )
        # Immediate first publish so peer summaries are in flight before
        # the L2 holon ADMM lands its initial allocation.
        self.context.schedule_instant_task(self._tick())
        # Watchdog: low-cadence safety net for missed events (late peer,
        # coalition TTL renewal during a silent window).
        self.context.schedule_periodic_task(self._tick, delay=self.watchdog_s)
        # Heat→L3 refresh: heat has no event-driven publish trigger
        # (MW-balance deactivated), so drive a faster delta-gated
        # republish to keep the delivered-heat vector the CP-ADMM reads
        # current.
        if self.sector == Sector.HEAT and self.enable_heat_cp_supply:
            self.context.schedule_periodic_task(
                self._publish_and_check, delay=self.heat_refresh_s
            )

    async def _tick(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Watchdog path: bypass the delta gate so the version frontier
        # advances even when nothing moved.
        await self._publish(force=True)
        self._check_invariants()
        if self.enable_cross_sector_coalitions:
            self._check_cross_sector_invariants()
        await self._reassert_active_coalitions()

    def _on_local_state_change(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        """L1 gossip converged — the per-tier vector may have moved, so
        attempt a delta-gated publish and re-check invariants.
        """
        if event.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        self.context.schedule_instant_task(self._publish_and_check())

    async def _on_l2_dispatch(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        """L2 dispatched a fresh allocation — the per-tier vector may
        shift once members apply it. Trigger a delta-gated publish + check.
        """
        if topology_characteristic(self, tid="groups") != "leader":
            return
        await self._publish_and_check()


    async def _publish_and_check(self) -> None:
        await self._publish()
        self._check_invariants()
        if self.enable_cross_sector_coalitions:
            self._check_cross_sector_invariants()

    def _summary_changed(
        self,
        served: dict[int, float],
        demand: dict[int, float],
    ) -> bool:
        """True iff the new per-tier vectors differ from the cached
        last-published ones on any tier by more than ``inversion_tol``.

        Compares over the union of tiers so a tier dropping to 0 counts
        as change.
        """
        if not self._last_published_served and not self._last_published_demand:
            return True  # first publish
        tiers = set(served) | set(demand) | set(
            self._last_published_served
        ) | set(self._last_published_demand)
        tol = self.inversion_tol
        for t in tiers:
            if abs(served.get(t, 0.0) - self._last_published_served.get(t, 0.0)) > tol:
                return True
            if abs(demand.get(t, 0.0) - self._last_published_demand.get(t, 0.0)) > tol:
                return True
        return False

    async def _publish(self, *, force: bool = False) -> None:
        """Aggregate this leader's community state per priority tier,
        then broadcast the summary to all same-sector peers via the
        ``holon_summary_<sector>`` topology.
        """
        try:
            peers = list(topology_neighbors(self, tid=self._topology_tid))
        except Exception:
            return
        if not peers:
            return

        per_tier_served: dict[int, float] = {}
        per_tier_demand: dict[int, float] = {}
        supply_total: float = 0.0
        slack_budget_total: float = 0.0  # slack-only; caps CP input draw
        try:
            member_aids = [self.context.aid] + [
                addr.aid for addr in topology_neighbors(self, tid="groups")
            ]
        except Exception:
            member_aids = [self.context.aid]

        for aid in member_aids:
            try:
                obs = self.behavior.observe(aid) or {}
            except (AttributeError, KeyError):
                return
            sector = obs_sector(obs, behavior=self.behavior, aid=aid)
            if sector != self.sector:
                continue
            cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
            if cap < 0:
                # Generator / slack injector — contributes to the supply
                # pool L3's CP-ADMM reads as per-sector ``base_supply``.
                # A slack advertises its operator budget (effective,
                # loss-compensated, when set, else nominal) not raw
                # ``|cap|`` — mirrors
                # ``EnergyBalanceNegotiator._handle_ask_flex`` so the
                # L2→L3 supply matches the L1/L2 pool and the CP draw is
                # capped at the budget.
                if lookup_slack(self.behavior, aid) is not None:
                    eff = lookup_slack_eff_budget(self.behavior, aid)
                    v = float(eff) if eff is not None else abs(cap)
                    supply_total += v
                    slack_budget_total += v
                else:
                    supply_total += abs(cap)
                continue
            if cap == 0:
                continue
            sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
            tier = obs_priority(obs, behavior=self.behavior, aid=aid)
            per_tier_demand[tier] = per_tier_demand.get(tier, 0.0) + abs(cap)
            per_tier_served[tier] = per_tier_served.get(tier, 0.0) + abs(sp)

        # Delta gate: skip publish + version bump when no tier moved by
        # more than ``inversion_tol``. The watchdog passes ``force=True``
        # to keep the version frontier advancing for late-joining peers.
        if not force and not self._summary_changed(
            per_tier_served, per_tier_demand
        ):
            return

        # Cache before ``send_message`` so a re-entrant publish from a
        # downstream event sees the most recent baseline.
        self._last_published_served = dict(per_tier_served)
        self._last_published_demand = dict(per_tier_demand)

        sec_key = self.sector.value
        supply_by_sector = (
            {sec_key: supply_total} if supply_total > 0.0 else {}
        )
        slack_budget_by_sector = (
            {sec_key: slack_budget_total} if slack_budget_total > 0.0 else {}
        )
        demand_by_sector_priority = (
            {sec_key: dict(per_tier_demand)} if per_tier_demand else {}
        )
        served_by_sector_priority = (
            {sec_key: dict(per_tier_served)} if per_tier_served else {}
        )

        summary = HolonSummary(
            publisher=str(self.context.aid),
            version=self._version.next(),
            caused_by={},
            timestamp_s=float(self.context.current_timestamp),
            sector=self.sector,
            per_tier_served_mw=per_tier_served,
            per_tier_demand_mw=per_tier_demand,
            supply_by_sector=supply_by_sector,
            demand_by_sector_priority=demand_by_sector_priority,
            served_by_sector_priority=served_by_sector_priority,
            slack_budget_by_sector=slack_budget_by_sector,
            home_node_id=self._my_node_id,
        )
        # Record our own summary too — the invariant check treats self
        # as just another publisher.
        self._peer_summaries[str(self.context.aid)] = summary
        # Cross-sector visibility (additive): mirror into a shared
        # per-sector registry that other-sector roles read during
        # cross-sector detection, avoiding a new topology mesh.
        # Single-process only; needs a real publish path across hosts.
        _xs_registry(self.behavior).setdefault(self.sector, {})[
            str(self.context.aid)
        ] = summary
        for addr in peers:
            await self.context.send_message(summary, receiver_addr=addr)

    async def _on_summary(
        self, message: HolonSummary, meta: dict
    ) -> None:
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        # Normalise to the bare aid string to match the
        # ``str(self.context.aid)`` self-key from ``_publish``. Mixed keys
        # would make the lex-smallest election always pick the
        # "AgentAddress(..." prefix, disabling the initiator path on the
        # actual lex-smallest aid.
        key = getattr(sender, "aid", None) or str(sender)
        prior = self._peer_summaries.get(key)
        if prior is not None and message.version <= prior.version:
            return  # stale
        self._peer_summaries[key] = message
        # Full address for sending coalition messages back to this peer.
        self._peer_addrs[key] = sender
        # Mirror into the shared cross-sector registry (as in _publish).
        _xs_registry(self.behavior).setdefault(message.sector, {})[key] = message
        # Peer view shifted — re-run detection now rather than lag it by
        # the full watchdog interval.
        if topology_characteristic(self, tid="groups") == "leader":
            self._check_invariants()
            if self.enable_cross_sector_coalitions:
                self._check_cross_sector_invariants()

    # ------------------------------------------------------------------
    # Detection + initiator election
    # ------------------------------------------------------------------

    def _is_elected_initiator(self) -> bool:
        """True when this leader is the lex-smallest publisher with
        non-empty summary state.

        Deterministic across leaders sharing a ``_peer_summaries``
        snapshot. Under eventual consistency the election can briefly
        flip when a previously-silent leader first publishes; the
        double-fire is absorbed by last-write-wins at L1 dispatch (worst
        case one extra message exchange, no deadlock).
        """
        if not self._peer_summaries:
            return False
        publishers = sorted(self._peer_summaries.keys())
        return publishers[0] == str(self.context.aid)

    def _check_invariants(self) -> None:
        """Aggregate peer summaries by tier, detect inversions, and
        (on the elected initiator) open a coalition.

        Emit and coalition formation are both gated on initiator election
        to yield one event + one coalition per inversion cohort, not N.
        """
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Need a peer summary besides our own to call an inversion
        # "cross-holon". The first tick fires before peer summaries
        # arrive; deferring lets the next tick reason on real state.
        if len(self._peer_summaries) < 2:
            return
        if not self._is_elected_initiator():
            return

        served_at_tier: dict[int, float] = {}
        demand_at_tier: dict[int, float] = {}
        for s in self._peer_summaries.values():
            for tier, served in s.per_tier_served_mw.items():
                served_at_tier[tier] = served_at_tier.get(tier, 0.0) + float(served)
            for tier, demand in s.per_tier_demand_mw.items():
                demand_at_tier[tier] = demand_at_tier.get(tier, 0.0) + float(demand)

        if not demand_at_tier:
            return

        tiers_sorted = sorted(
            t for t, d in demand_at_tier.items() if d > 1e-9 and t >= 1
        )
        if len(tiers_sorted) < 2:
            return
        fracs: dict[int, float] = {
            t: served_at_tier.get(t, 0.0) / demand_at_tier[t]
            for t in tiers_sorted
        }

        total_served = sum(served_at_tier.get(t, 0.0) for t in tiers_sorted)
        total_demand = sum(demand_at_tier[t] for t in tiers_sorted)
        if total_demand <= 1e-9 or total_served >= total_demand - 1e-6:
            return

        now = float(self.context.current_timestamp)
        if now - self._last_inversion_emit_t < self._inversion_cooldown_s:
            return

        emitted = False
        # Open a coalition for the worst inversion (largest fraction gap)
        # per tick. Bundling every inverted pair into one multi-tier
        # coalition is too aggressive: the supply-priority ADMM waterfalls
        # the whole tier set in one broadcast, dropping mid tiers in
        # lockstep and producing jarring shed cascades. With the
        # per-tick cooldown, successive worst-gap targets clear every
        # persistent inversion within seconds while drops stay small.
        worst_pair: tuple[int, int] | None = None
        worst_gap: float = 0.0
        for i in range(1, len(tiers_sorted)):
            t_prev, t_cur = tiers_sorted[i - 1], tiers_sorted[i]
            f_prev, f_cur = fracs[t_prev], fracs[t_cur]
            gap = f_cur - f_prev
            if gap > self.inversion_tol:
                record_event(
                    t=now,
                    kind="priority_inversion_detected",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=(
                        f"tier_high={t_prev} (frac={f_prev:.3f}) "
                        f"tier_low={t_cur} (frac={f_cur:.3f}) "
                        f"n_publishers={len(self._peer_summaries)}"
                    ),
                )
                emitted = True
                if gap > worst_gap:
                    worst_gap = gap
                    worst_pair = (t_prev, t_cur)
        if emitted:
            self._last_inversion_emit_t = now
            logger.info(
                "[%s] cross-holon priority inversion detected (sector=%s, "
                "n_publishers=%d, n_tiers=%d, fracs=%s)",
                self.context.aid, self.sector.value,
                len(self._peer_summaries), len(tiers_sorted),
                {t: round(f, 3) for t, f in fracs.items()},
            )
            if self.enable_coalition and worst_pair is not None:
                # Open as an instant task so the check stays synchronous
                # and the rest of the tick (re-assert) still runs.
                self.context.schedule_instant_task(
                    self._open_coalition(worst_pair, dict(demand_at_tier))
                )

    # ------------------------------------------------------------------
    # Cross-sector coalition detection + allocation
    # ------------------------------------------------------------------

    def _check_cross_sector_invariants(self) -> None:
        """Detect cross-sector priority inversions across CP-bridged
        sector pairs.

        For each CP in ``_cp_meta`` bridging ``self.sector``, compare the
        other sector's latest summaries (from the shared registry) and
        flag where a higher-priority tier on one side is served at a
        lower fraction than a lower-priority tier on the other. If found
        and this role is the elected initiator (lex-smallest aid across
        both sectors' publishers), open a cross-sector coalition.
        """
        if not self._cp_meta:
            return
        registry = _xs_registry(self.behavior)
        own_sec = self.sector
        own_aid = str(self.context.aid)
        own_summaries = registry.get(own_sec, {})
        if not own_summaries:
            return
        now = float(self.context.current_timestamp)
        if now - self._last_xs_inversion_emit_t < self._inversion_cooldown_s:
            return

        for cp_aid, meta in self._cp_meta.items():
            sectors_bridged = meta.get("sectors", [])
            if own_sec not in sectors_bridged:
                continue
            for peer_sec in sectors_bridged:
                if peer_sec == own_sec:
                    continue
                peer_summaries = registry.get(peer_sec, {})
                if not peer_summaries:
                    continue
                pair = self._find_inversion_pair(
                    own_summaries, peer_summaries
                )
                if pair is None:
                    continue
                t_own_high, t_peer_low, frac_own, frac_peer = pair
                # Initiator election: lex-smallest aid across the union
                # of both sides' publishers. Skip if we're not it.
                union_aids = sorted(
                    set(own_summaries.keys()) | set(peer_summaries.keys())
                )
                if not union_aids or union_aids[0] != own_aid:
                    continue
                self._last_xs_inversion_emit_t = now
                record_event(
                    t=now,
                    kind="cross_sector_inversion_detected",
                    aid=self.context.aid,
                    sector=own_sec.value,
                    detail=(
                        f"cp={cp_aid} own_sec={own_sec.value} "
                        f"tier_high={t_own_high} frac_high={frac_own:.3f} "
                        f"peer_sec={peer_sec.value} tier_low={t_peer_low} "
                        f"frac_low={frac_peer:.3f}"
                    ),
                )
                logger.info(
                    "[%s] cross-sector inversion: cp=%s %s.t%d=%.3f vs "
                    "%s.t%d=%.3f — opening coalition",
                    self.context.aid, cp_aid,
                    own_sec.value, t_own_high, frac_own,
                    peer_sec.value, t_peer_low, frac_peer,
                )
                self.context.schedule_instant_task(
                    self._open_cross_sector_coalition(
                        cp_aid=cp_aid,
                        own_sec=own_sec,
                        peer_sec=peer_sec,
                        t_own_high=t_own_high,
                        t_peer_low=t_peer_low,
                    )
                )
                return  # one coalition per tick

    def _find_inversion_pair(
        self,
        own_summaries: dict[str, HolonSummary],
        peer_summaries: dict[str, HolonSummary],
    ) -> tuple[int, int, float, float] | None:
        """Return ``(t_own_high, t_peer_low, frac_own, frac_peer)`` if a
        cross-sector inversion exists, else None.

        Inversion: some own-side tier with strict priority over a peer
        tier (lower tier number = higher priority) is served at a
        fraction at least ``inversion_tol`` below the peer's.
        """
        own_dem, own_ser = self._aggregate_tier(own_summaries)
        peer_dem, peer_ser = self._aggregate_tier(peer_summaries)
        if not own_dem or not peer_dem:
            return None
        for t_own in sorted(own_dem.keys()):
            if t_own < 1 or own_dem[t_own] <= 1e-9:
                continue
            f_own = own_ser.get(t_own, 0.0) / own_dem[t_own]
            for t_peer in sorted(peer_dem.keys()):
                if t_peer <= t_own:
                    continue  # peer not lower-priority
                if peer_dem[t_peer] <= 1e-9:
                    continue
                f_peer = peer_ser.get(t_peer, 0.0) / peer_dem[t_peer]
                if f_peer > f_own + self.inversion_tol:
                    return t_own, t_peer, f_own, f_peer
        return None

    @staticmethod
    def _aggregate_tier(
        summaries: dict[str, HolonSummary],
    ) -> tuple[dict[int, float], dict[int, float]]:
        dem: dict[int, float] = {}
        ser: dict[int, float] = {}
        for s in summaries.values():
            for tier, v in s.per_tier_demand_mw.items():
                dem[tier] = dem.get(tier, 0.0) + float(v)
            for tier, v in s.per_tier_served_mw.items():
                ser[tier] = ser.get(tier, 0.0) + float(v)
        return dem, ser

    async def _open_cross_sector_coalition(
        self,
        *,
        cp_aid: str,
        own_sec: Sector,
        peer_sec: Sector,
        t_own_high: int,
        t_peer_low: int,
    ) -> None:
        """Build a cross-sector allocation and dispatch it immediately.

        No invitation round: relies on build-time ``_cp_meta`` for CP
        rated capacity + coupling and on the cross-sector registry for
        leaders' current per-tier state. A handshake could be added for
        runtime-varying CP capacity without changing the dispatch path.
        """
        meta = self._cp_meta.get(cp_aid)
        if meta is None:
            return
        coupling = meta.get("coupling_ratios", {}) or {}
        rated = meta.get("rated_capacity_mw", {}) or {}
        cp_addr = meta.get("addr")
        if cp_addr is None:
            return

        # Direction: CP pushes into own_sec (raising its deficit-side
        # service) and draws from peer_sec. Coupling keyed (in, out).
        key = (peer_sec.value, own_sec.value)
        eta = float(coupling.get(key, 0.0))
        cp_cap_out = float(rated.get(own_sec.value, 0.0))
        if eta <= 0.0 or cp_cap_out <= 0.0:
            return  # CP can't push into own_sec this direction

        registry = _xs_registry(self.behavior)
        own_summaries = registry.get(own_sec, {})
        peer_summaries = registry.get(peer_sec, {})

        own_dem, own_ser = self._aggregate_tier(own_summaries)
        peer_dem, peer_ser = self._aggregate_tier(peer_summaries)

        deficit_own_high = max(0.0, own_dem.get(t_own_high, 0.0) - own_ser.get(t_own_high, 0.0))
        served_peer_low = peer_ser.get(t_peer_low, 0.0)
        # Own-sec MW the CP can produce (after η) by drawing peer's
        # tier_low served.
        peer_freeable_own = served_peer_low * eta

        # Bounded by deficit, peer's available served, and CP rated.
        transfer_out = min(deficit_own_high, peer_freeable_own, cp_cap_out)
        if transfer_out <= 1e-6:
            logger.info(
                "[%s] cross-sector coalition skipped: nothing to transfer "
                "(deficit=%.4f, peer_freeable=%.4f, cp_cap=%.4f)",
                self.context.aid, deficit_own_high, peer_freeable_own, cp_cap_out,
            )
            return
        transfer_in = transfer_out / eta

        new_ser_own = own_ser.get(t_own_high, 0.0) + transfer_out
        own_total = own_dem.get(t_own_high, 0.0)
        new_frac_own = min(1.0, new_ser_own / own_total) if own_total > 1e-9 else 1.0

        new_ser_peer = max(0.0, served_peer_low - transfer_in)
        peer_total = peer_dem.get(t_peer_low, 0.0)
        new_frac_peer = max(0.0, new_ser_peer / peer_total) if peer_total > 1e-9 else 1.0

        service_fraction_by_sector_tier: dict[str, dict[int, float]] = {
            own_sec.value: {t_own_high: new_frac_own},
            peer_sec.value: {t_peer_low: new_frac_peer},
        }
        cp_targets: dict[str, dict[str, float]] = {
            cp_aid: {own_sec.value: +transfer_out, peer_sec.value: -transfer_in}
        }

        # Resolve leader addresses: ``own_sec`` from our peer book
        # (intra-sector summaries → _peer_addrs); ``peer_sec`` from the
        # scenario-supplied ``_peer_leader_addrs`` map.
        own_addrs: list[Any] = []
        for aid in own_summaries:
            if aid == str(self.context.aid):
                continue
            addr = self._peer_addrs.get(aid)
            if addr is not None:
                own_addrs.append(addr)
        peer_addrs: list[Any] = []
        peer_book = self._peer_leader_addrs.get(peer_sec, {})
        for aid in peer_summaries:
            addr = peer_book.get(aid)
            if addr is not None:
                peer_addrs.append(addr)

        self._coalition_counter += 1
        coalition_id = f"xs:{self.context.aid}#{self._coalition_counter}"
        now = float(self.context.current_timestamp)
        active = _ActiveCrossSectorCoalition(
            coalition_id=coalition_id,
            service_fraction_by_sector_tier=service_fraction_by_sector_tier,
            leader_addrs_by_sector={
                own_sec.value: own_addrs,
                peer_sec.value: peer_addrs,
            },
            cp_targets_mw=cp_targets,
            cp_addrs={cp_aid: cp_addr},
            sectors=(own_sec, peer_sec),
            issued_at=now,
            ttl_s=float(self.coalition_constraint_ttl_s),
        )
        self._active_xs_coalitions[coalition_id] = active

        # Persist so L2 (per-sector ADMM) and L3 (CP ADMM) see it.
        if self._constraint_store is not None:
            self._constraint_store.set(
                coalition_id=coalition_id,
                sector=own_sec,
                service_fraction_by_tier={t_own_high: new_frac_own},
                issued_at=now,
                ttl_s=float(self.coalition_constraint_ttl_s),
            )
            # Distinct id so the peer-side record isn't overwritten by
            # the own-side set above.
            self._constraint_store.set(
                coalition_id=f"{coalition_id}/peer",
                sector=peer_sec,
                service_fraction_by_tier={t_peer_low: new_frac_peer},
                issued_at=now,
                ttl_s=float(self.coalition_constraint_ttl_s),
            )
            self._constraint_store.set_cp_envelope(
                coalition_id=coalition_id,
                cp_id=cp_aid,
                target_flows_mw=cp_targets[cp_aid],
                issued_at=now,
                ttl_s=float(self.coalition_constraint_ttl_s),
            )

        record_event(
            t=now,
            kind="cross_sector_coalition_allocation",
            aid=self.context.aid,
            sector=own_sec.value,
            detail=(
                f"id={coalition_id} cp={cp_aid} "
                f"transfer_out={transfer_out:.4f} transfer_in={transfer_in:.4f} "
                f"own_frac={{{t_own_high}: {new_frac_own:.3f}}} "
                f"peer_frac={{{t_peer_low}: {new_frac_peer:.3f}}}"
            ),
        )

        await self._dispatch_active_xs_coalition(active)

    async def _dispatch_active_xs_coalition(
        self, active: _ActiveCrossSectorCoalition
    ) -> None:
        """Send one ``StartBalanceNegotiation`` per sector to its leaders
        + one ``CPCommitment`` per CP. Idempotent: shared by the initial
        fire and every re-assert tick.
        """
        for sec_v, addrs in active.leader_addrs_by_sector.items():
            tier_map = active.service_fraction_by_sector_tier.get(sec_v, {})
            if not tier_map:
                continue
            payload = StartBalanceNegotiation(
                service_fraction_by_sector_priority={sec_v: dict(tier_map)},
            )
            for addr in addrs:
                await self.context.send_message(payload, receiver_addr=addr)
        # Self-dispatch so own L1 applies the fraction too.
        own_v = self.sector.value
        own_tier_map = active.service_fraction_by_sector_tier.get(own_v, {})
        if own_tier_map:
            own_addr = getattr(self.context, "addr", None)
            if own_addr is not None:
                await self.context.send_message(
                    StartBalanceNegotiation(
                        service_fraction_by_sector_priority={own_v: dict(own_tier_map)},
                    ),
                    receiver_addr=own_addr,
                )
        for cp_aid, flows in active.cp_targets_mw.items():
            cp_addr = active.cp_addrs.get(cp_aid)
            if cp_addr is None:
                continue
            commit = CPCommitment(
                publisher=str(self.context.aid),
                version=self._version.next(),
                caused_by={},
                timestamp_s=float(self.context.current_timestamp),
                coalition_id=active.coalition_id,
                cp_id=cp_aid,
                target_flows_mw=dict(flows),
                ttl_s=float(active.ttl_s),
            )
            await self.context.send_message(commit, receiver_addr=cp_addr)

    # ------------------------------------------------------------------
    # Coalition initiator path
    # ------------------------------------------------------------------

    async def _open_coalition(
        self,
        target_tiers: tuple[int, ...],
        demand_at_tier: dict[int, float],
    ) -> None:
        """Build the coalition member list and broadcast invitations.

        Members are publishers with non-zero demand in any target tier
        (the only leaders whose dispatch a fraction change there
        affects); zero-demand peers would just inflate message volume.
        Self is always a member. ``target_tiers`` carries every flagged
        tier so one ADMM pass redistributes all of them.
        """
        if not target_tiers:
            return
        member_aids: list[str] = [str(self.context.aid)]
        for aid, summary in self._peer_summaries.items():
            if aid == str(self.context.aid):
                continue
            if any(
                float(summary.per_tier_demand_mw.get(t, 0.0)) > 1e-9
                for t in target_tiers
            ):
                member_aids.append(aid)
        if len(member_aids) < 2:
            return

        self._coalition_counter += 1
        coalition_id = f"{self.context.aid}#{self._coalition_counter}"
        now = float(self.context.current_timestamp)
        pending = _PendingCoalition(
            coalition_id=coalition_id,
            sector=self.sector,
            target_tiers=tuple(target_tiers),
            member_aids=tuple(member_aids),
            started_at=now,
        )
        # Pre-seed the initiator's own acceptance from a local observe
        # (we already know our own state; skip the round-trip).
        own_acc = self._local_acceptance(coalition_id, target_tiers)
        if own_acc is not None:
            pending.acceptances[str(self.context.aid)] = own_acc
        pending.addr_by_aid[str(self.context.aid)] = None  # sentinel

        invitation = CoalitionInvitation(
            publisher=str(self.context.aid),
            version=self._version.next(),
            caused_by={},
            timestamp_s=now,
            coalition_id=coalition_id,
            sector=self.sector,
            target_tiers=tuple(target_tiers),
            member_aids=tuple(member_aids),
            ttl_s=float(self.coalition_constraint_ttl_s),
        )

        n_sent = 0
        for aid in member_aids:
            if aid == str(self.context.aid):
                continue
            addr = self._peer_addrs.get(aid)
            if addr is None:
                continue
            pending.addr_by_aid[aid] = addr
            await self.context.send_message(invitation, receiver_addr=addr)
            n_sent += 1

        if n_sent == 0:
            return

        self._pending_coalitions[coalition_id] = pending
        logger.info(
            "[%s] coalition %s opened: tiers=%s members=%d (invitations=%d)",
            self.context.aid, coalition_id, target_tiers,
            len(member_aids), n_sent,
        )
        # Allocate after the acceptance window.
        try:
            self.context.schedule_timestamp_task(
                self._close_and_allocate(coalition_id),
                timestamp=now + float(self.coalition_accept_window_s),
            )
        except Exception:
            # Scheduler lacks absolute timestamps: fall back to instant,
            # using whatever acceptances have arrived (absent peers just
            # don't contribute).
            self.context.schedule_instant_task(
                self._close_and_allocate(coalition_id)
            )

    async def _on_invitation(
        self, message: CoalitionInvitation, meta: dict
    ) -> None:
        """Reply with an acceptance when included in the member list.

        No flex-at-target gate: the initiator already pre-filtered, and a
        no-flex member just contributes zeros the allocator handles.
        """
        own_aid = str(self.context.aid)
        if own_aid not in message.member_aids:
            return
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        target_tiers = tuple(message.target_tiers) if message.target_tiers else ()
        acceptance = self._local_acceptance(message.coalition_id, target_tiers)
        if acceptance is None:
            return
        await self.context.send_message(acceptance, receiver_addr=sender)

    def _local_acceptance(
        self,
        coalition_id: str,
        target_tiers_in: tuple[int, ...],
    ) -> CoalitionAcceptance | None:
        """Build this leader's acceptance payload from a fresh observe
        over its own community.

        Like :class:`HolonicCommunityRole._build_flex_answer`: per-(sector,
        tier) demand + per-sector supply for the initiator's arbitration.
        Demand is restricted to ``target_tiers_in`` to keep the payload
        small; supply spans all sectors since the LP may route freed
        supply through CPs.
        """
        try:
            member_aids = [self.context.aid] + [
                addr.aid for addr in topology_neighbors(self, tid="groups")
            ]
        except Exception:
            member_aids = [self.context.aid]

        target_tiers = {int(t) for t in target_tiers_in if t is not None}
        sector_str = self.sector.value
        supply_by_sector: dict[str, float] = {}
        demand_by_sector_priority: dict[str, dict[int, float]] = {}
        served_by_sector_priority: dict[str, dict[int, float]] = {}
        # ``demand_nodes_by_tier[tier][node_id] = mw``: spatial demand
        # footprint for per-actor reachability via the mirror. Keyed by
        # node (not aid) so the initiator needs no global aid→node map;
        # each acceptance carries its own leader's slice.
        demand_nodes_by_tier: dict[int, dict[Any, float]] = {}
        for aid in member_aids:
            try:
                obs = self.behavior.observe(aid) or {}
            except (AttributeError, KeyError):
                return None
            sec = obs_sector(obs, behavior=self.behavior, aid=aid)
            if sec is None:
                continue
            cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
            sec_v = sec.value
            if cap < 0:  # generator-class — register supply
                # Slack advertises its (effective) budget, not raw |cap|
                # (see publish path).
                if lookup_slack(self.behavior, aid) is not None:
                    eff = lookup_slack_eff_budget(self.behavior, aid)
                    add = float(eff) if eff is not None else abs(float(cap))
                else:
                    add = abs(float(cap))
                supply_by_sector[sec_v] = supply_by_sector.get(sec_v, 0.0) + add
                continue
            if cap <= 0:  # slack / passive — skip
                continue
            if sec != self.sector:
                continue  # sector-scoped invariant: skip other-sector demand
            tier = obs_priority(obs, behavior=self.behavior, aid=aid)
            if target_tiers and tier not in target_tiers:
                continue
            sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
            demand_by_sector_priority.setdefault(sec_v, {})
            demand_by_sector_priority[sec_v][tier] = (
                demand_by_sector_priority[sec_v].get(tier, 0.0) + abs(float(cap))
            )
            served_by_sector_priority.setdefault(sec_v, {})
            served_by_sector_priority[sec_v][tier] = (
                served_by_sector_priority[sec_v].get(tier, 0.0) + abs(float(sp))
            )
            node = self._member_node_ids.get(str(aid))
            if node is not None:
                bucket = demand_nodes_by_tier.setdefault(tier, {})
                bucket[node] = bucket.get(node, 0.0) + abs(float(cap))

        return CoalitionAcceptance(
            publisher=str(self.context.aid),
            version=self._version.next(),
            caused_by={},
            timestamp_s=float(self.context.current_timestamp),
            coalition_id=coalition_id,
            sector=self.sector,
            accepted=True,
            supply_by_sector=supply_by_sector,
            demand_by_sector_priority=demand_by_sector_priority,
            served_by_sector_priority=served_by_sector_priority,
            home_node_id=self._my_node_id,
            demand_nodes_by_tier=demand_nodes_by_tier,
        )

    async def _on_constraint(
        self, message: CoalitionConstraint, meta: dict
    ) -> None:
        """Persist an incoming coalition constraint so this leader's L2
        ADMM consults it on dispatch.

        Trusts the initiator's TTL. Each ``set`` is authoritative;
        out-of-order arrivals would briefly enforce a stale fraction.
        """
        if self._constraint_store is None:
            return
        now = float(self.context.current_timestamp)
        self._constraint_store.set(
            coalition_id=message.coalition_id,
            sector=message.sector,
            service_fraction_by_tier=message.service_fraction_by_tier,
            issued_at=float(message.timestamp_s) or now,
            ttl_s=float(message.ttl_s),
        )

    async def _on_acceptance(
        self, message: CoalitionAcceptance, meta: dict
    ) -> None:
        pending = self._pending_coalitions.get(message.coalition_id)
        if pending is None or pending.run:
            return  # not our coalition, or already allocated
        sender = mango_sender_addr(meta)
        sender_aid = (
            getattr(sender, "aid", None) or message.publisher or str(sender)
        )
        if sender_aid not in pending.member_aids:
            return
        pending.acceptances[sender_aid] = message
        if sender is not None:
            pending.addr_by_aid[sender_aid] = sender

    @staticmethod
    def _cap_fractions_by_feasibility(
        fractions: dict[int, float],
        demand_by_tier: dict[int, float],
        served_by_tier: dict[int, float],
    ) -> int:
        """No-op shim kept for callers that read its return value.

        Capping each tier's fraction at the observed served fraction
        formed a self-reinforcing loop that locked tiers at their
        degenerate operating point. Supply budgeting now lives in
        ``budget_scale`` and the ADMM box constraints, making this cap
        redundant and harmful.
        """
        return 0

    def _aggregate_coalition_supply_demand(
        self,
        accepting: list[CoalitionAcceptance],
        sector_str: str,
    ) -> _CoalitionAggregate:
        """Sum supply / demand / served across accepting members and
        copy per-actor inputs into fresh structures for the allocator.
        """
        total_supply = 0.0
        total_observed_served = 0.0
        demand_by_tier: dict[int, float] = {}
        served_by_tier: dict[int, float] = {}
        actor_supplies: list[dict[str, float]] = []
        actor_demands: list[dict[str, dict[int, float]]] = []
        actor_node_ids: list[Any] = []
        actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]] = []
        for acc in accepting:
            total_supply += float(acc.supply_by_sector.get(sector_str, 0.0))
            for tier, dem in acc.demand_by_sector_priority.get(sector_str, {}).items():
                demand_by_tier[tier] = demand_by_tier.get(tier, 0.0) + float(dem)
            for tier, srv in acc.served_by_sector_priority.get(sector_str, {}).items():
                served_by_tier[tier] = served_by_tier.get(tier, 0.0) + float(srv)
                total_observed_served += float(srv)
            actor_supplies.append(dict(acc.supply_by_sector))
            actor_demands.append({
                k: dict(v) for k, v in acc.demand_by_sector_priority.items()
            })
            actor_node_ids.append(acc.home_node_id)
            actor_demand_nodes_by_tier.append({
                tier: dict(nodes)
                for tier, nodes in (acc.demand_nodes_by_tier or {}).items()
            })
        return _CoalitionAggregate(
            total_supply=total_supply,
            total_observed_served=total_observed_served,
            demand_by_tier=demand_by_tier,
            served_by_tier=served_by_tier,
            actor_supplies=actor_supplies,
            actor_demands=actor_demands,
            actor_node_ids=actor_node_ids,
            actor_demand_nodes_by_tier=actor_demand_nodes_by_tier,
        )

    async def _close_and_allocate(self, coalition_id: str) -> None:
        pending = self._pending_coalitions.pop(coalition_id, None)
        if pending is None or pending.run:
            return
        pending.run = True

        accepting = [
            a for a in pending.acceptances.values() if a.accepted
        ]
        if len(accepting) < 2:
            logger.info(
                "[%s] coalition %s closed without enough acceptances "
                "(%d/%d), no allocation",
                self.context.aid, coalition_id,
                len(accepting), len(pending.member_aids),
            )
            return

        sector_str = self.sector.value
        agg = self._aggregate_coalition_supply_demand(accepting, sector_str)
        total_supply = agg.total_supply
        total_observed_served = agg.total_observed_served
        demand_by_tier = agg.demand_by_tier
        served_by_tier = agg.served_by_tier
        actor_supplies = agg.actor_supplies
        actor_demands = agg.actor_demands
        actor_node_ids = agg.actor_node_ids
        actor_demand_nodes_by_tier = agg.actor_demand_nodes_by_tier

        if not demand_by_tier:
            return

        tiers_for_admm = sorted(t for t in demand_by_tier.keys() if t >= 1)
        if not tiers_for_admm:
            return

        # Observed-served budget cap: when total_observed_served <
        # total_demand the LP is bottlenecked; scale each actor's supply
        # by the delivery efficiency so the ADMM's per-actor coupling
        # binds at the realistic ceiling. No-op when not bottlenecked.
        #
        # Denominator ``min(total_supply, total_demand_sector)`` is the
        # achievable-delivery ceiling (whichever of nameplate or demand
        # binds). Dividing by raw nameplate alone explodes the scale when
        # supply >> demand (typical for heat) and spirals into curtailment;
        # capping at demand keeps the scale as "fraction of achievable
        # delivery currently realised".
        total_demand_sector = sum(demand_by_tier.values())
        budget_scale = 1.0
        if (
            total_observed_served > 0.0
            and total_supply > 0.0
            and total_demand_sector > 1e-9
            and total_observed_served < total_demand_sector - 1e-9
        ):
            denom = max(min(total_supply, total_demand_sector), 1e-9)
            # Cap at 1.0 so delivery never inflates supply past raw
            # nameplate (the per-cell ub still uses raw).
            budget_scale = min(1.0, total_observed_served / denom)
            for supply_map in actor_supplies:
                if sector_str in supply_map:
                    supply_map[sector_str] = (
                        float(supply_map[sector_str]) * budget_scale
                    )

        # Deliverability caps: narrow each actor's per-(sector, tier) ub
        # to its reachable-demand sum so a stranded actor's supply isn't
        # committed to demand it can't reach. ``mirror=None`` ⇒ all-None
        # entries and the ADMM falls back to raw supply caps.
        try:
            actor_ub_overrides = per_actor_deliverable_caps(
                actor_node_ids=actor_node_ids,
                actor_demand_nodes_by_tier=actor_demand_nodes_by_tier,
                sector=self.sector,
                mirror=self._mirror,
            )
        except Exception as exc:
            logger.warning(
                "[%s] coalition %s: deliverability caps failed (%s) — "
                "falling back to raw supply",
                self.context.aid, coalition_id, exc,
            )
            actor_ub_overrides = None

        try:
            service_fraction_map, _per_actor_x, _meta = await allocate_supply_priority(
                sectors=[sector_str],
                tiers=tiers_for_admm,
                actor_supplies=actor_supplies,
                actor_demands=actor_demands,
                actor_ub_overrides=actor_ub_overrides,
                priority_tiers=self.priority_tiers,
                max_iters=int(self.admm_max_iters),
                abs_tol=float(self.admm_abs_tol),
            )
            fractions = dict(service_fraction_map.get(sector_str, {}))
            alloc_method = "admm"
        except Exception as exc:
            logger.warning(
                "[%s] coalition %s ADMM failed (%s) — falling back to "
                "centralised greedy",
                self.context.aid, coalition_id, exc,
            )
            record_event(
                t=float(self.context.current_timestamp),
                kind="coalition_admm_failed",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=f"id={coalition_id} exc={exc!r}",
            )
            # Safety net: priority-greedy on the aggregate pool so a
            # solver hiccup never loses the coalition entirely.
            remaining_supply = max(total_supply, 0.0)
            fractions = {}
            for tier in tiers_for_admm:
                dem = demand_by_tier[tier]
                if dem <= 1e-9:
                    fractions[tier] = 1.0
                    continue
                served = min(remaining_supply, dem)
                fractions[tier] = max(0.0, min(1.0, served / dem))
                remaining_supply -= served
            alloc_method = "greedy_fallback"

        n_capped = self._cap_fractions_by_feasibility(
            fractions, demand_by_tier, served_by_tier
        )

        record_event(
            t=float(self.context.current_timestamp),
            kind="coalition_allocation",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=(
                f"id={coalition_id} method={alloc_method} "
                f"tiers={pending.target_tiers} "
                f"n_accept={len(accepting)} supply={total_supply:.4f} "
                f"observed_served={total_observed_served:.4f} "
                f"budget_scale={budget_scale:.3f} "
                f"feasibility_caps={n_capped} "
                f"demand_by_tier={{"
                f"{', '.join(f'{t}: {v:.4f}' for t, v in sorted(demand_by_tier.items()))}}} "
                f"served_by_tier={{"
                f"{', '.join(f'{t}: {v:.4f}' for t, v in sorted(served_by_tier.items()))}}} "
                f"fractions={{{', '.join(f'{t}: {v:.3f}' for t, v in sorted(fractions.items()))}}}"
            ),
        )
        logger.info(
            "[%s] coalition %s allocated (%s): supply=%.4f "
            "demand_by_tier=%s fractions=%s",
            self.context.aid, coalition_id, alloc_method, total_supply,
            {t: round(v, 4) for t, v in sorted(demand_by_tier.items())},
            {t: round(v, 3) for t, v in sorted(fractions.items())},
        )

        # Record the active coalition so each ``_tick`` re-asserts the
        # constraint until TTL expiry. The initial dispatch and re-fires
        # share the same helper.
        addrs: list[Any] = []
        for acc in accepting:
            aid = acc.publisher
            if aid == str(self.context.aid):
                continue  # initiator dispatches to itself separately
            addr = pending.addr_by_aid.get(aid)
            if addr is not None:
                addrs.append(addr)
        now = float(self.context.current_timestamp)
        active = _ActiveCoalition(
            coalition_id=coalition_id,
            sector=self.sector,
            service_fraction_by_tier=fractions,
            member_addrs=addrs,
            issued_at=now,
            ttl_s=float(self.coalition_constraint_ttl_s),
        )
        self._active_coalitions[coalition_id] = active

        # Persist locally + broadcast a CoalitionConstraint so every
        # member's L2 ADMM merges the same fractions for the TTL window
        # instead of overwriting them on its next rebalance.
        if self._constraint_store is not None:
            self._constraint_store.set(
                coalition_id=coalition_id,
                sector=self.sector,
                service_fraction_by_tier=fractions,
                issued_at=now,
                ttl_s=float(self.coalition_constraint_ttl_s),
            )
        constraint_msg = CoalitionConstraint(
            publisher=str(self.context.aid),
            version=self._version.next(),
            caused_by={},
            timestamp_s=now,
            coalition_id=coalition_id,
            sector=self.sector,
            service_fraction_by_tier=dict(fractions),
            ttl_s=float(self.coalition_constraint_ttl_s),
        )
        # Broadcast to every same-sector peer leader, not just accepting
        # members: a chunk initiator outside the coalition could
        # otherwise dispatch un-merged L2 fractions to coalition members
        # in its chunk. Filling every store makes any L2 dispatch merge
        # the coalition fractions first. Cheap: one small message per peer.
        broadcast_targets = set()
        for addr in active.member_addrs:
            broadcast_targets.add(addr)
        for aid, addr in self._peer_addrs.items():
            if aid == str(self.context.aid):
                continue
            broadcast_targets.add(addr)
        # Self-loop so the initiator's own ``HolonicCommunityRole`` reacts
        # like peers: the handler triggers an L2 rebalance merging the new
        # fractions. Without it the initiator's L2 only re-runs on its slow
        # heartbeat while peers rebalance within ms.
        own_addr = getattr(self.context, "addr", None)
        if own_addr is not None:
            broadcast_targets.add(own_addr)
        for addr in broadcast_targets:
            await self.context.send_message(constraint_msg, receiver_addr=addr)

        await self._dispatch_active_coalition(active)

    async def _dispatch_active_coalition(
        self, active: _ActiveCoalition
    ) -> None:
        """Send the constraint as a StartBalanceNegotiation to every
        accepting member, plus self so the L1 handler runs its normal path.
        """
        service_fraction_by_sector_priority = {
            active.sector.value: dict(active.service_fraction_by_tier),
        }
        payload = StartBalanceNegotiation(
            service_fraction_by_sector_priority=
                service_fraction_by_sector_priority,
        )
        for addr in active.member_addrs:
            await self.context.send_message(payload, receiver_addr=addr)
        # Self via ``self.context.addr`` so the message lands on this
        # leader's own EnergyBalanceNegotiator through the same handler
        # path L2 messages use.
        own_addr = getattr(self.context, "addr", None)
        if own_addr is not None:
            await self.context.send_message(payload, receiver_addr=own_addr)

    async def _reassert_active_coalitions(self) -> None:
        """Per-tick TTL pruning + re-broadcast.

        Re-broadcasting each tick is what makes a coalition constraint
        hold against L2: if L2 overwrites the L1 fraction between ticks,
        the next tick re-asserts it within ``period_s``.
        """
        now = float(self.context.current_timestamp)
        # Prune expired store records so L2's merge sees only valid
        # fractions. Idempotent.
        if self._constraint_store is not None:
            self._constraint_store.prune(now)
        if self._active_coalitions:
            expired: list[str] = []
            for coalition_id, active in self._active_coalitions.items():
                if now > active.issued_at + active.ttl_s:
                    expired.append(coalition_id)
                    continue
                await self._dispatch_active_coalition(active)
            for cid in expired:
                self._active_coalitions.pop(cid, None)
                logger.debug(
                    "[%s] coalition %s expired (ttl reached)",
                    self.context.aid, cid,
                )
        # Cross-sector coalitions: same shape, separate dispatch
        # (multi-sector + CPCommitment).
        if self._active_xs_coalitions:
            expired_xs: list[str] = []
            for cid, active in self._active_xs_coalitions.items():
                if now > active.issued_at + active.ttl_s:
                    expired_xs.append(cid)
                    continue
                await self._dispatch_active_xs_coalition(active)
            for cid in expired_xs:
                self._active_xs_coalitions.pop(cid, None)
                logger.debug(
                    "[%s] cross-sector coalition %s expired (ttl reached)",
                    self.context.aid, cid,
                )

    # ------------------------------------------------------------------
    # Failure invalidation
    # ------------------------------------------------------------------

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Drop all active coalition constraints for this sector.

        Wired by the scenario builder on ``BranchFailureEvent``. Doesn't
        check whether the failed branch is in-sector — invalidates on any
        failure since cross-sector coupling can make a CP failure relevant
        everywhere. The failure-retriggered L2 ADMM produces a fresh
        allocation grounded in the post-failure topology.
        """
        n_active = len(self._active_coalitions)
        n_pending = len(self._pending_coalitions)
        n_xs = len(self._active_xs_coalitions)
        n_store = (
            self._constraint_store.clear(self.sector)
            if self._constraint_store is not None else 0
        )
        if not n_active and not n_pending and not n_store and not n_xs:
            return
        self._active_coalitions.clear()
        self._pending_coalitions.clear()
        # Cross-sector coalitions invalidate on any branch failure —
        # conservative; can't filter precisely since the store-level
        # clear above already dropped same-sector envelopes.
        self._active_xs_coalitions.clear()
        logger.info(
            "[%s] branch failure invalidated %d active + %d pending "
            "+ %d cross-sector + %d stored coalitions (sector=%s)",
            self.context.aid, n_active, n_pending, n_xs, n_store,
            self.sector.value,
        )
