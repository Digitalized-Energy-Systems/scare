"""Layer 2.5 — holon-summary mesh + cross-holon coalition formation.

Each group leader publishes a :class:`HolonSummary` on a sector-wide
full mesh (``holon_summary_<sector>``) and runs a cross-holon
priority-inversion check. On detection a deterministically-elected
initiator forms an ad-hoc coalition, runs a scoped supply-priority
allocation over the members' flex, and broadcasts per-tier
service-fraction constraints they apply on L1 dispatch.

Additive to :class:`HolonicCommunityRole`: L2's ADMM keeps running and
coalition constraints are TTL-bounded. Inversion = a higher-priority
tier served at a strictly smaller frac than a lower-priority one beyond
``inversion_tol`` (mirrors ``experiment/eval/claims.py``). Initiator =
lex-smallest publisher with non-empty state, collapsing N duplicate
detections into one.

Constraints expire on ``now > issued_at + ttl_s`` (control returns to
L2) or on a ``BranchFailureEvent`` (topology changed). No two-phase
commit: last-write-wins at L1, the next tick re-asserts.
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
from scare.base.model import NegotiationFinishedEvent, Sector, StartBalanceNegotiation
from scare.base.runtime.diagnostics import record_event
from scare.base.util import (
    clamp_tier_monotonic,
    lookup_slack,
    lookup_slack_eff_budget,
    obs_capacity,
    obs_priority,
    obs_sector,
    obs_setpoint,
)
from scare.community.coalition_store import CoalitionConstraintStore
from scare.community.deliverability import per_actor_deliverable_caps
from scare.community.supply_priority_admm import allocate_supply_priority

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


@dataclass
class _PendingCoalition:
    """Initiator-side state during the invitation/acceptance window.

    ``run`` flips True once allocated so a late acceptance can't
    re-trigger it.
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

    Re-asserted every ``_tick`` until TTL expiry or a same-sector
    ``BranchFailureEvent``. ``member_addrs`` holds only accepting members.
    """

    coalition_id: str
    sector: Sector
    service_fraction_by_tier: dict[int, float]
    member_addrs: list[Any]
    issued_at: float
    ttl_s: float


def _xs_registry(
    behavior: RestorationEnvironmentBehavior,
) -> dict[Sector, dict[str, HolonSummary]]:
    """Per-behavior shared registry of latest HolonSummary by sector.

    Lazy-init, single-process; a distributed deployment needs a real
    cross-sector publish path.
    """
    store = getattr(behavior, "_scare_xs_summaries", None)
    if store is None:
        store = {}
        behavior._scare_xs_summaries = store
    return store


@dataclass
class _ActiveCrossSectorCoalition:
    """Initiator-side TTL record of an allocated cross-sector coalition.

    Unlike :class:`_ActiveCoalition` the dispatch spans multiple sectors
    and includes per-CP commitments (directional flows).
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
    """Cross-holon priority observability + coalition formation.

    Installed on every group leader. Non-leaders stay quiescent:
    ``_tick``'s leader-check returns early, so no publish fires.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
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
        cp_budget_nominal: bool = True,
        coalition_delivered_supply: bool = True,
        cp_commitment_actuatable: bool = False,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.period_s = period_s
        # Heat→L3: heat's summary triggers are off, so refresh faster to
        # keep the delivered-heat deficit flowing to the CP-ADMM.
        self.enable_heat_cp_supply = bool(enable_heat_cp_supply)
        self.heat_refresh_s = float(heat_refresh_s)
        self.cp_budget_nominal = bool(cp_budget_nominal)
        # Credit coalition-pool generators at delivered |sp| (not rated |cap|)
        # so a curtailed generator can't fund the pool at nameplate; also gates
        # cross-sector coalitions on the CP transfer being actuatable (see
        # enable_coalition_delivered_supply).
        self.coalition_delivered_supply = bool(coalition_delivered_supply)
        # Whether a CPCommitment consumer exists (legacy EnergyConverterRole L3).
        # Under the default priority-ADMM L3 it does not, so a cross-sector
        # coalition's promised CP transfer never actuates — don't raise
        # own-sector fractions on it (they'd be funded by the slack instead).
        self.cp_commitment_actuatable = bool(cp_commitment_actuatable)
        # Slow safety-net cadence re-running publish + check + re-assert
        # even when idle; the dominant trigger is event-driven (see setup).
        self.watchdog_s = watchdog_s
        # Cached last-published vectors so the publisher can skip when
        # nothing moved by more than ``inversion_tol``.
        self._last_published_served: dict[int, float] = {}
        self._last_published_demand: dict[int, float] = {}
        self.inversion_tol = inversion_tol
        self.enable_coalition = enable_coalition
        self.coalition_accept_window_s = coalition_accept_window_s
        self.coalition_constraint_ttl_s = coalition_constraint_ttl_s
        self.priority_tiers = priority_tiers
        self.admm_max_iters = admm_max_iters
        self.admm_abs_tol = admm_abs_tol
        # Deliverability wiring (leader node, member aid→node, mirror).
        # Any being None degrades to raw-supply ADMM (no caps).
        self._my_node_id = my_node_id
        self._member_node_ids: dict[str, Any] = dict(member_node_ids or {})
        self._mirror = mirror
        # Shared store between L2.5 (writer) and L2 (reader). None ⇒ L2's
        # later rounds overwrite per-tier without checking coalitions.
        self._constraint_store = constraint_store
        self._version = MonotonicVersion()
        # Latest summary + address per publisher, for direct dispatch.
        self._peer_summaries: dict[str, HolonSummary] = {}
        self._peer_addrs: dict[str, Any] = {}
        # Cooldown = ``period_s`` so a persistent inversion re-detects
        # each tick, converging while L2 rebalances on its slow heartbeat.
        self._last_inversion_emit_t: float = -1e9
        self._inversion_cooldown_s: float = period_s
        # Keyed by id so parallel coalitions can coexist.
        self._pending_coalitions: dict[str, _PendingCoalition] = {}
        self._active_coalitions: dict[str, _ActiveCoalition] = {}
        self._coalition_counter: int = 0

        # ---- Cross-sector coalition state ----
        # When enabled, cross-sector invariants run after intra-sector
        # checks and may open coalitions spanning a CP.
        self.enable_cross_sector_coalitions = enable_cross_sector_coalitions
        # cp_aid -> {sectors, coupling_ratios, rated_capacity_mw, addr}.
        self._cp_meta: dict[str, dict[str, Any]] = dict(cp_meta or {})
        # sector -> {aid -> addr}.
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
            self.context.aid,
            self.sector.value,
            self.period_s,
            self._topology_tid,
        )

        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))

            return _sync

        # Subscribe to summaries from same-sector peers.
        self.context.subscribe_message(
            self,
            _wrap(self._on_summary),
            lambda msg, meta: (
                isinstance(msg, HolonSummary) and msg.sector == self.sector
            ),
        )
        # Coalition control-plane subs, sector-filtered so a leader is
        # never pulled into another sector's coalition.
        self.context.subscribe_message(
            self,
            _wrap(self._on_invitation),
            lambda msg, meta: (
                isinstance(msg, CoalitionInvitation) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._on_acceptance),
            lambda msg, meta: (
                isinstance(msg, CoalitionAcceptance) and msg.sector == self.sector
            ),
        )
        # Inbound constraints from other initiators, stored so this
        # leader's L2 ADMM consults them first (coalition wins per cell).
        self.context.subscribe_message(
            self,
            _wrap(self._on_constraint),
            lambda msg, meta: (
                isinstance(msg, CoalitionConstraint) and msg.sector == self.sector
            ),
        )
        # Event-driven publish: the per-tier vector only moves on L1
        # gossip convergence or L2 dispatch, so subscribe to both.
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
        # Watchdog: low-cadence safety net for missed events.
        self.context.schedule_periodic_task(self._tick, delay=self.watchdog_s)
        # Heat→L3 refresh: heat has no event-driven publish trigger, so
        # drive a faster delta-gated republish to keep the CP-ADMM current.
        if self.sector == Sector.HEAT and self.enable_heat_cp_supply:
            self.context.schedule_periodic_task(
                self._publish_and_check, delay=self.heat_refresh_s
            )

    async def _tick(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Watchdog path: bypass the delta gate so the version frontier
        # advances even when idle.
        await self._publish(force=True)
        self._check_invariants()
        if self.enable_cross_sector_coalitions:
            self._check_cross_sector_invariants()
        await self._reassert_active_coalitions()

    def _on_local_state_change(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        """L1 gossip converged — delta-gated publish + re-check."""
        if event.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        self.context.schedule_instant_task(self._publish_and_check())

    async def _on_l2_dispatch(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        """L2 dispatched a fresh allocation — delta-gated publish + check."""
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
        """True iff any tier moved by more than ``inversion_tol`` vs the
        last published vectors (union of tiers, so a drop to 0 counts).
        """
        if not self._last_published_served and not self._last_published_demand:
            return True  # first publish
        tiers = (
            set(served)
            | set(demand)
            | set(self._last_published_served)
            | set(self._last_published_demand)
        )
        tol = self.inversion_tol
        for t in tiers:
            if abs(served.get(t, 0.0) - self._last_published_served.get(t, 0.0)) > tol:
                return True
            if abs(demand.get(t, 0.0) - self._last_published_demand.get(t, 0.0)) > tol:
                return True
        return False

    async def _publish(self, *, force: bool = False) -> None:
        """Aggregate this leader's community state per tier and broadcast
        to all same-sector peers via ``holon_summary_<sector>``.
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
                # Generator / slack injector feeds L3's supply pool. A
                # slack advertises its (effective) budget, not raw |cap|.
                slack_meta = lookup_slack(self.behavior, aid)
                if slack_meta is not None:
                    eff = lookup_slack_eff_budget(self.behavior, aid)
                    v = float(eff) if eff is not None else abs(cap)
                    supply_total += v
                    # The CP input cap must NOT see the wound-down eff
                    # budget: the integral feedback exists to make L1/L2
                    # shed native load toward B, but with it a converter
                    # (Ση<1) has zero input headroom and its converged
                    # optimum is r=0 — starving the one actuator that can
                    # relieve the over-draw. The CP kernel's cascade
                    # arbitrates native serving vs CP input inside the
                    # same B, so give it the nominal operator budget.
                    if self.cp_budget_nominal:
                        slack_budget_total += abs(slack_meta.cap)
                    else:
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

        # Delta gate: skip when no tier moved by more than ``inversion_tol``.
        # The watchdog forces through to advance the version frontier.
        if not force and not self._summary_changed(per_tier_served, per_tier_demand):
            return

        # Cache before ``send_message`` so a re-entrant publish sees the
        # latest baseline.
        self._last_published_served = dict(per_tier_served)
        self._last_published_demand = dict(per_tier_demand)

        sec_key = self.sector.value
        supply_by_sector = {sec_key: supply_total} if supply_total > 0.0 else {}
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
        # Cross-sector visibility: mirror into a shared per-sector registry
        # other-sector roles read. Single-process only; no cross-host path.
        _xs_registry(self.behavior).setdefault(self.sector, {})[
            str(self.context.aid)
        ] = summary
        for addr in peers:
            await self.context.send_message(summary, receiver_addr=addr)

    async def _on_summary(self, message: HolonSummary, meta: dict) -> None:
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        # Normalise to the bare aid string to match the self-key from
        # ``_publish``; mixed keys would break the lex-smallest election.
        key = getattr(sender, "aid", None) or str(sender)
        prior = self._peer_summaries.get(key)
        if prior is not None and message.version <= prior.version:
            return  # stale
        self._peer_summaries[key] = message
        # Full address for sending coalition messages back to this peer.
        self._peer_addrs[key] = sender
        _xs_registry(self.behavior).setdefault(message.sector, {})[key] = message
        # Peer view shifted — re-run detection now, not next watchdog.
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

        A brief election flip (a silent leader first publishing)
        double-fires but is absorbed by last-write-wins at L1.
        """
        if not self._peer_summaries:
            return False
        publishers = sorted(self._peer_summaries.keys())
        return publishers[0] == str(self.context.aid)

    def _check_invariants(self) -> None:
        """Aggregate peer summaries by tier, detect inversions, and
        (on the elected initiator) open a coalition.

        Gated on initiator election to yield one event + one coalition
        per inversion cohort, not N.
        """
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Need a peer summary besides our own to call it "cross-holon";
        # the first tick fires before any peer summary arrives.
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
            t: served_at_tier.get(t, 0.0) / demand_at_tier[t] for t in tiers_sorted
        }

        total_served = sum(served_at_tier.get(t, 0.0) for t in tiers_sorted)
        total_demand = sum(demand_at_tier[t] for t in tiers_sorted)
        if total_demand <= 1e-9 or total_served >= total_demand - 1e-6:
            return

        now = float(self.context.current_timestamp)
        if now - self._last_inversion_emit_t < self._inversion_cooldown_s:
            return

        emitted = False
        # Worst inversion (largest gap) per tick: bundling every pair
        # would drop mid tiers in lockstep; successive ticks clear them.
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
                self.context.aid,
                self.sector.value,
                len(self._peer_summaries),
                len(tiers_sorted),
                {t: round(f, 3) for t, f in fracs.items()},
            )
            if self.enable_coalition and worst_pair is not None:
                # Instant task so the check stays synchronous and the
                # rest of the tick still runs.
                self.context.schedule_instant_task(
                    self._open_coalition(worst_pair, dict(demand_at_tier))
                )

    # ------------------------------------------------------------------
    # Cross-sector coalition detection + allocation
    # ------------------------------------------------------------------

    def _check_cross_sector_invariants(self) -> None:
        """Detect cross-sector priority inversions across CP-bridged
        sector pairs and, if elected initiator (lex-smallest aid across
        both sides' publishers), open a cross-sector coalition.
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
                pair = self._find_inversion_pair(own_summaries, peer_summaries)
                if pair is None:
                    continue
                t_own_high, t_peer_low, frac_own, frac_peer = pair
                # Initiator = lex-smallest aid across both sides; skip
                # otherwise.
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
                    self.context.aid,
                    cp_aid,
                    own_sec.value,
                    t_own_high,
                    frac_own,
                    peer_sec.value,
                    t_peer_low,
                    frac_peer,
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
                return  # one per tick

    def _find_inversion_pair(
        self,
        own_summaries: dict[str, HolonSummary],
        peer_summaries: dict[str, HolonSummary],
    ) -> tuple[int, int, float, float] | None:
        """Return ``(t_own_high, t_peer_low, frac_own, frac_peer)`` if an
        inversion exists, else None.

        Inversion: an own-side tier with strict priority over a peer tier
        (lower number = higher priority) served at least ``inversion_tol``
        below the peer's fraction.
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
                    continue  # not lower-priority
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

        No invitation round: uses build-time ``_cp_meta`` (CP capacity +
        coupling) and the registry's current per-tier state.
        """
        # Don't raise own-sector service fractions on a CP transfer that has no
        # actuator: under the default priority-ADMM L3 there is no CPCommitment
        # consumer, so the promised inflow never materialises and the raised
        # fractions get funded by the slack instead (the child-118 overdraw).
        if self.coalition_delivered_supply and not self.cp_commitment_actuatable:
            record_event(
                t=float(self.context.current_timestamp),
                kind="cross_sector_coalition_skipped_unactuatable",
                aid=self.context.aid,
                sector=own_sec.value,
                detail=f"cp={cp_aid} peer={peer_sec.value} (no CPCommitment consumer)",
            )
            return

        meta = self._cp_meta.get(cp_aid)
        if meta is None:
            return
        coupling = meta.get("coupling_ratios", {}) or {}
        rated = meta.get("rated_capacity_mw", {}) or {}
        cp_addr = meta.get("addr")
        if cp_addr is None:
            return

        # CP pushes into own_sec and draws from peer_sec. Coupling keyed
        # (in, out).
        key = (peer_sec.value, own_sec.value)
        eta = float(coupling.get(key, 0.0))
        cp_cap_out = float(rated.get(own_sec.value, 0.0))
        if eta <= 0.0 or cp_cap_out <= 0.0:
            return  # CP can't push this direction

        registry = _xs_registry(self.behavior)
        own_summaries = registry.get(own_sec, {})
        peer_summaries = registry.get(peer_sec, {})

        own_dem, own_ser = self._aggregate_tier(own_summaries)
        peer_dem, peer_ser = self._aggregate_tier(peer_summaries)

        deficit_own_high = max(
            0.0, own_dem.get(t_own_high, 0.0) - own_ser.get(t_own_high, 0.0)
        )
        served_peer_low = peer_ser.get(t_peer_low, 0.0)
        # Own-sec MW the CP can produce (after η) from peer's tier_low.
        peer_freeable_own = served_peer_low * eta

        # Bounded by deficit, peer's available served, and CP rated.
        transfer_out = min(deficit_own_high, peer_freeable_own, cp_cap_out)
        if transfer_out <= 1e-6:
            logger.info(
                "[%s] cross-sector coalition skipped: nothing to transfer "
                "(deficit=%.4f, peer_freeable=%.4f, cp_cap=%.4f)",
                self.context.aid,
                deficit_own_high,
                peer_freeable_own,
                cp_cap_out,
            )
            return
        transfer_in = transfer_out / eta

        new_ser_own = own_ser.get(t_own_high, 0.0) + transfer_out
        own_total = own_dem.get(t_own_high, 0.0)
        new_frac_own = min(1.0, new_ser_own / own_total) if own_total > 1e-9 else 1.0

        new_ser_peer = max(0.0, served_peer_low - transfer_in)
        peer_total = peer_dem.get(t_peer_low, 0.0)
        new_frac_peer = (
            max(0.0, new_ser_peer / peer_total) if peer_total > 1e-9 else 1.0
        )

        service_fraction_by_sector_tier: dict[str, dict[int, float]] = {
            own_sec.value: {t_own_high: new_frac_own},
            peer_sec.value: {t_peer_low: new_frac_peer},
        }
        cp_targets: dict[str, dict[str, float]] = {
            cp_aid: {own_sec.value: +transfer_out, peer_sec.value: -transfer_in}
        }

        # Leader addresses: own_sec from our peer book, peer_sec from the
        # scenario-supplied ``_peer_leader_addrs``.
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
            # Distinct id so the peer-side record isn't overwritten.
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
        """Send one ``StartBalanceNegotiation`` per sector + one
        ``CPCommitment`` per CP. Idempotent (initial fire + re-asserts).
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

        Members = publishers with non-zero demand in any target tier
        (plus self); zero-demand peers would just add message volume.
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
        # Pre-seed our own acceptance locally; skip the round-trip.
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
            self.context.aid,
            coalition_id,
            target_tiers,
            len(member_aids),
            n_sent,
        )
        # Allocate after the acceptance window.
        try:
            self.context.schedule_timestamp_task(
                self._close_and_allocate(coalition_id),
                timestamp=now + float(self.coalition_accept_window_s),
            )
        except Exception:
            # No absolute-timestamp scheduling: fall back to instant with
            # whatever acceptances have arrived.
            self.context.schedule_instant_task(self._close_and_allocate(coalition_id))

    async def _on_invitation(self, message: CoalitionInvitation, meta: dict) -> None:
        """Reply with an acceptance when in the member list.

        No flex gate: the initiator pre-filtered, and a no-flex member
        just contributes zeros.
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
        """Build this leader's acceptance payload from a fresh observe.

        Per-(sector, tier) demand (restricted to ``target_tiers_in``) +
        per-sector supply (all sectors, since the LP may route freed
        supply through CPs).
        """
        try:
            member_aids = [self.context.aid] + [
                addr.aid for addr in topology_neighbors(self, tid="groups")
            ]
        except Exception:
            member_aids = [self.context.aid]

        target_tiers = {int(t) for t in target_tiers_in if t is not None}
        supply_by_sector: dict[str, float] = {}
        demand_by_sector_priority: dict[str, dict[int, float]] = {}
        served_by_sector_priority: dict[str, dict[int, float]] = {}
        # Spatial demand footprint for per-actor reachability, keyed by
        # node (not aid) so the initiator needs no global aid→node map.
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
                # Slack advertises its (effective) budget (see publish).
                if lookup_slack(self.behavior, aid) is not None:
                    eff = lookup_slack_eff_budget(self.behavior, aid)
                    add = float(eff) if eff is not None else abs(float(cap))
                elif self.coalition_delivered_supply:
                    # Generator: credit DELIVERED |sp|, not RATED |cap| — a
                    # curtailed generator can't fund the pool at its nameplate
                    # (mirrors the L2 supply pool in balance.py).
                    sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
                    add = abs(float(sp))
                else:
                    add = abs(float(cap))
                supply_by_sector[sec_v] = supply_by_sector.get(sec_v, 0.0) + add
                continue
            if cap <= 0:  # zero-capacity / passive — skip
                continue
            if sec != self.sector:
                continue  # sector-scoped invariant: skip other-sector demand
            tier = obs_priority(obs, behavior=self.behavior, aid=aid)
            if target_tiers and tier not in target_tiers:
                continue
            sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
            demand_by_sector_priority.setdefault(sec_v, {})
            demand_by_sector_priority[sec_v][tier] = demand_by_sector_priority[
                sec_v
            ].get(tier, 0.0) + abs(float(cap))
            served_by_sector_priority.setdefault(sec_v, {})
            served_by_sector_priority[sec_v][tier] = served_by_sector_priority[
                sec_v
            ].get(tier, 0.0) + abs(float(sp))
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

    async def _on_constraint(self, message: CoalitionConstraint, meta: dict) -> None:
        """Persist an incoming constraint so this leader's L2 ADMM
        consults it. Trusts the initiator's TTL.
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

    async def _on_acceptance(self, message: CoalitionAcceptance, meta: dict) -> None:
        pending = self._pending_coalitions.get(message.coalition_id)
        if pending is None or pending.run:
            return  # not ours, or already allocated
        sender = mango_sender_addr(meta)
        sender_aid = getattr(sender, "aid", None) or message.publisher or str(sender)
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

        Capping at the observed served fraction self-reinforced into a
        lock at the degenerate point; budgeting now lives in
        ``budget_scale`` + the ADMM box constraints.
        """
        return 0

    def _aggregate_coalition_supply_demand(
        self,
        accepting: list[CoalitionAcceptance],
        sector_str: str,
    ) -> _CoalitionAggregate:
        """Sum supply/demand/served across accepting members; copy
        per-actor inputs into fresh structures for the allocator.
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
            actor_demands.append(
                {k: dict(v) for k, v in acc.demand_by_sector_priority.items()}
            )
            actor_node_ids.append(acc.home_node_id)
            actor_demand_nodes_by_tier.append(
                {
                    tier: dict(nodes)
                    for tier, nodes in (acc.demand_nodes_by_tier or {}).items()
                }
            )
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

        accepting = [a for a in pending.acceptances.values() if a.accepted]
        if len(accepting) < 2:
            logger.info(
                "[%s] coalition %s closed without enough acceptances "
                "(%d/%d), no allocation",
                self.context.aid,
                coalition_id,
                len(accepting),
                len(pending.member_aids),
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

        # Budget cap: when bottlenecked, scale supply by delivery vs
        # ``min(supply, demand)`` (raw nameplate over-scales when supply >> demand).
        total_demand_sector = sum(demand_by_tier.values())
        budget_scale = 1.0
        if (
            total_observed_served > 0.0
            and total_supply > 0.0
            and total_demand_sector > 1e-9
            and total_observed_served < total_demand_sector - 1e-9
        ):
            denom = max(min(total_supply, total_demand_sector), 1e-9)
            # Cap at 1.0 so delivery never inflates supply past nameplate.
            budget_scale = min(1.0, total_observed_served / denom)
            for supply_map in actor_supplies:
                if sector_str in supply_map:
                    supply_map[sector_str] = (
                        float(supply_map[sector_str]) * budget_scale
                    )

        # Deliverability caps: bound each actor's ub to its reachable
        # demand so it isn't committed to unreachable load. None ⇒ raw caps.
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
                self.context.aid,
                coalition_id,
                exc,
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
                self.context.aid,
                coalition_id,
                exc,
            )
            record_event(
                t=float(self.context.current_timestamp),
                kind="coalition_admm_failed",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=f"id={coalition_id} exc={exc!r}",
            )
            # Safety net: priority-greedy on the aggregate pool so a
            # solver hiccup never loses the coalition.
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
            self.context.aid,
            coalition_id,
            alloc_method,
            total_supply,
            {t: round(v, 4) for t, v in sorted(demand_by_tier.items())},
            {t: round(v, 3) for t, v in sorted(fractions.items())},
        )

        # Record the active coalition so each ``_tick`` re-asserts the
        # constraint until TTL expiry.
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

        # Persist locally + broadcast so every member's L2 ADMM merges
        # the same fractions for the TTL window instead of overwriting.
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
        # Broadcast to every same-sector peer leader, not just members, so
        # an outside chunk initiator's L2 dispatch also merges these first.
        broadcast_targets = set()
        for addr in active.member_addrs:
            broadcast_targets.add(addr)
        for aid, addr in self._peer_addrs.items():
            if aid == str(self.context.aid):
                continue
            broadcast_targets.add(addr)
        # Self-loop so the initiator's own L2 rebalance merges the new
        # fractions immediately rather than waiting for its heartbeat.
        own_addr = getattr(self.context, "addr", None)
        if own_addr is not None:
            broadcast_targets.add(own_addr)
        for addr in broadcast_targets:
            await self.context.send_message(constraint_msg, receiver_addr=addr)

        await self._dispatch_active_coalition(active)

    async def _dispatch_active_coalition(self, active: _ActiveCoalition) -> None:
        """Send the constraint as a StartBalanceNegotiation to every
        accepting member, plus self.

        Tier-monotonic clamp mirrors the component-allocation path: a coalition
        fraction map dispatched directly must not serve a lower-priority tier
        above a higher one.
        """
        service_fraction_by_sector_priority = {
            active.sector.value: clamp_tier_monotonic(
                dict(active.service_fraction_by_tier)
            ),
        }
        payload = StartBalanceNegotiation(
            service_fraction_by_sector_priority=service_fraction_by_sector_priority,
        )
        for addr in active.member_addrs:
            await self.context.send_message(payload, receiver_addr=addr)
        # Self so the message lands on this leader's own
        # EnergyBalanceNegotiator via the same handler path L2 uses.
        own_addr = getattr(self.context, "addr", None)
        if own_addr is not None:
            await self.context.send_message(payload, receiver_addr=own_addr)

    async def _reassert_active_coalitions(self) -> None:
        """Per-tick TTL pruning + re-broadcast.

        Re-broadcasting each tick is what holds a constraint against L2:
        if L2 overwrites the L1 fraction, the next tick re-asserts it.
        """
        now = float(self.context.current_timestamp)
        # Prune expired store records so L2's merge sees only valid ones.
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
                    self.context.aid,
                    cid,
                )
        # Cross-sector coalitions: same shape, separate dispatch.
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
                    self.context.aid,
                    cid,
                )

    # ------------------------------------------------------------------
    # Failure invalidation
    # ------------------------------------------------------------------

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Drop all active coalition constraints for this sector.

        Wired on ``BranchFailureEvent``. Invalidates on any failure (not
        just in-sector) since cross-sector coupling can make a CP failure
        relevant everywhere; L2's retrigger then re-allocates.
        """
        n_active = len(self._active_coalitions)
        n_pending = len(self._pending_coalitions)
        n_xs = len(self._active_xs_coalitions)
        n_store = (
            self._constraint_store.clear(self.sector)
            if self._constraint_store is not None
            else 0
        )
        if not n_active and not n_pending and not n_store and not n_xs:
            return
        self._active_coalitions.clear()
        self._pending_coalitions.clear()
        # Cross-sector coalitions invalidate on any failure (conservative).
        self._active_xs_coalitions.clear()
        logger.info(
            "[%s] branch failure invalidated %d active + %d pending "
            "+ %d cross-sector + %d stored coalitions (sector=%s)",
            self.context.aid,
            n_active,
            n_pending,
            n_xs,
            n_store,
            self.sector.value,
        )
