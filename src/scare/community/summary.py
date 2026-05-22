"""Layer 2.5 — holon-summary mesh + cross-holon coalition formation.

Milestone 1 (detection): every group leader periodically publishes a
:class:`scare.base.channel.HolonSummary` on a sector-wide full-mesh
topology (``holon_summary_<sector>``) carrying the leader's own
community per-tier served-MW and demand-MW.  Every leader subscribes
to the same channel, accumulates the latest summary per peer, and
runs a local *cross-holon priority inversion* check.

Milestone 2 (this module): on detection, a deterministically-elected
initiator forms an ad-hoc coalition with the leaders whose holons
contribute to the inversion, runs a scoped supply-priority allocation
over the union of their flex, and broadcasts per-tier service-fraction
constraints that the coalition members apply as
``StartBalanceNegotiation(service_fraction_by_sector_priority=...)``
to their own L1 dispatch.

Why this is its own role and not part of :class:`HolonicCommunityRole`:

- This layer is observability + scoped cooperation only.  The
  underlying chunked-clique ADMM at L2 keeps running on its slow
  heartbeat; the coalition is *additive*, dictating constraints that
  the L1 dispatch path honours for the duration of a TTL.  Keeping
  M2 in a separate role makes the "L2 stays as-is" guarantee
  structural rather than relying on careful avoidance of method
  overlap.
- The coalition operates on *every* same-sector leader's published
  state, which is a wider observation scope than L2's chunk-mates.

Detection rule
--------------

For each (sector, holon-summary-graph) the role aggregates received
summaries plus its own.  For each tier ``t`` it sums received
``per_tier_served_mw`` and ``per_tier_demand_mw`` across publishers,
then computes ``frac[t] = served / demand`` (1.0 when demand is 0).
An inversion is recorded when a higher-priority tier ``t_h`` has
strictly smaller ``frac`` than a lower-priority tier ``t_l > t_h``,
by more than the configured tolerance.  The check mirrors the
priority-invariant claim in ``experiment/eval/claims.py`` so the
detector and the claim agree on what counts.

Initiator election (M2)
-----------------------

When an inversion fires, every leader in the sector observes the
same set of peer summaries (eventual-consistency caveat: some peers
may be one tick behind, but the lex-smallest publisher is stable
across that window because membership doesn't change at sub-second
rates).  The lex-smallest publisher with non-empty summary state is
the unique coalition initiator for this round.  Non-initiators still
run the detection and store summaries, but suppress the
``priority_inversion_detected`` event and the coalition-formation
broadcast — this collapses what was N duplicate M1 events into one
per inversion and gives a single owner for the coalition lifecycle.

Coalition lifecycle
-------------------

1. **Invitation** — initiator builds ``member_aids`` from peers whose
   summary contributed to the inverted tier pair, plus itself, and
   sends ``CoalitionInvitation`` on the same sector-wide mesh.
2. **Acceptance** — every invited leader replies with
   ``CoalitionAcceptance`` carrying its own per-tier supply / demand
   slice (same shape ``HolonicCommunityRole`` already consumes for
   its supply-priority ADMM).
3. **Allocation** — after a short ``accept_window_s`` the initiator
   runs a centralised priority-weighted greedy allocation over the
   collected acceptances.  Centralised greedy gives the optimal
   priority-ordered allocation when supply is fungible within a
   sector — the parent holon's ADMM only existed because each actor
   held distinct supply that had to agree; here the initiator
   already has the aggregate so the agreement step is moot.
4. **Constraint dispatch** — initiator sends
   ``StartBalanceNegotiation(service_fraction_by_sector_priority=...)``
   directly to every accepting member, piggy-backing on the same
   handler L2 uses.  The initiator also records the constraint as
   active locally so it can re-broadcast on every L2.5 tick while
   the TTL is still valid, overriding any out-of-band L2 dispatch
   that fires between coalitions.

Constraint invalidation
-----------------------

A coalition constraint is invalidated by either:

- ``now > issued_at + ttl_s``: natural expiry; control returns to
  the underlying L2 holon ADMM on the next L2 rebalance tick.
- ``BranchFailureEvent`` in the same sector: the post-failure
  topology has changed enough that the recently-computed fractions
  no longer correspond to the live grid.  Dropping the constraint
  lets the L2 holon ADMM (which gets re-triggered by the failure)
  redecide allocations.

Eventual consistency
--------------------

The coalition runs while chunked-clique L2 ADMM rounds are still in
flight.  No freeze flag, no two-phase commit.  Last-write-wins on
``StartBalanceNegotiation`` at the L1 leader means a coalition's
constraint may be briefly overridden by an L2 rebalance that fires
between coalition ticks; the next coalition tick (default 1 s)
re-asserts the constraint.  When the TTL expires, L2's allocation
takes over cleanly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from scare.community.coalition_store import CoalitionConstraintStore
from scare.base.model import Sector, StartBalanceNegotiation
from scare.base.util import (
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

    ``acceptances[aid] = CoalitionAcceptance`` collects each peer's
    reply; the entry for the initiator itself is filled in
    synchronously when the coalition is opened, since the initiator
    skips the round-trip.  ``addr_by_aid`` is the back-channel: we
    need a sendable address to dispatch the constraint after
    allocation, so we keep the address used to send the invitation
    keyed by aid.  ``run`` flips True once the allocation has been
    computed so a late-arriving acceptance doesn't re-trigger it.
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

    Re-asserted on every ``_tick`` until ``issued_at + ttl_s`` has
    passed or a same-sector ``BranchFailureEvent`` invalidates it
    early.  ``member_addrs`` is the dispatch list — only members that
    actually accepted are included so a declining peer is not
    over-written by a fraction it did not opt in to.
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

    Lazy-init.  Reads + writes are single-process; for distributed
    deployments a real cross-sector publish path would replace this.
    """
    store = getattr(behavior, "_scare_xs_summaries", None)
    if store is None:
        store = {}
        behavior._scare_xs_summaries = store
    return store


@dataclass
class _ActiveCrossSectorCoalition:
    """Initiator-side TTL record of an allocated cross-sector coalition.

    Distinct from :class:`_ActiveCoalition` because the dispatch fan-out
    spans multiple sectors AND includes CP commitments — the per-tick
    re-assert has to fire each piece on the right channel.  Sector-keyed
    service fractions go to the matching leaders; CP commitments are
    dispatched to the CPs' own addresses with the directional flows
    each CP must hold for the TTL window.
    """

    coalition_id: str
    service_fraction_by_sector_tier: dict[str, dict[int, float]]
    leader_addrs_by_sector: dict[str, list[Any]]
    cp_targets_mw: dict[str, dict[str, float]]  # cp_aid -> sector_v -> mw
    cp_addrs: dict[str, Any]
    sectors: tuple[Sector, ...]
    issued_at: float
    ttl_s: float


class HolonSummaryRole(Role):
    """Periodic publisher + subscriber for cross-holon priority
    observability, plus M2 coalition formation.

    Installed on every group leader (next to
    :class:`HolonicCommunityRole`).  Non-leaders silently drop into a
    quiescent state — their ``setup`` runs but no publish ever fires
    because the leader-check at the top of ``_tick`` returns early.
    """

    def __init__(
        self,
        behavior: "RestorationEnvironmentBehavior",
        sector: Sector,
        *,
        period_s: float = 1.0,
        inversion_tol: float = 1e-3,
        enable_coalition: bool = True,
        coalition_accept_window_s: float = 1.0,
        coalition_constraint_ttl_s: float = 8.0,
        priority_tiers: int = 10,
        admm_max_iters: int = 50,
        admm_abs_tol: float = 1e-3,
        my_node_id: Any = None,
        member_node_ids: dict[str, Any] | None = None,
        mirror: Any = None,
        constraint_store: CoalitionConstraintStore | None = None,
        enable_cross_sector_coalitions: bool = False,
        cp_meta: dict[str, dict[str, Any]] | None = None,
        peer_leader_addrs: dict[Sector, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.period_s = period_s
        self.inversion_tol = inversion_tol
        self.enable_coalition = enable_coalition
        self.coalition_accept_window_s = coalition_accept_window_s
        self.coalition_constraint_ttl_s = coalition_constraint_ttl_s
        self.priority_tiers = priority_tiers
        self.admm_max_iters = admm_max_iters
        self.admm_abs_tol = admm_abs_tol
        # Spatial wiring for deliverability-aware coalition allocation.
        # ``my_node_id`` is this leader's monee node; ``member_node_ids``
        # maps each owned member's aid to its monee node (used for the
        # per-tier demand-location map in the acceptance payload).
        # ``mirror`` is the shared :class:`GridTopologyMirror` whose
        # ``reachable_from(node, sector=...)`` drives the per-actor cap
        # computation.  Any of these being ``None`` degrades the
        # coalition to raw-supply ADMM (still better than greedy on
        # the per-actor coupling, but without deliverability caps).
        self._my_node_id = my_node_id
        self._member_node_ids: dict[str, Any] = dict(member_node_ids or {})
        self._mirror = mirror
        # Shared store between L2.5 (writer) and L2 (reader) on the
        # same leader.  None ⇒ no constraint binding — the coalition
        # still dispatches its StartBalanceNegotiation, but L2's
        # subsequent rounds will overwrite per-tier without checking
        # for active coalitions (the pre-store M2 behaviour).
        self._constraint_store = constraint_store
        self._version = MonotonicVersion()
        # ``_peer_summaries[publisher_aid] = HolonSummary`` — each
        # entry is the most recent summary received from that peer.
        # Older versions are overwritten silently.
        self._peer_summaries: dict[str, HolonSummary] = {}
        # Address book built from incoming HolonSummary metadata, so
        # the coalition initiator can target invitations + constraint
        # dispatches at specific aids without re-deriving via
        # topology_neighbors (which only gives a flat list).
        self._peer_addrs: dict[str, Any] = {}
        # Cooldown so a single persistent inversion doesn't spam the
        # event ledger.  Reset to current sim-time on every emit.
        self._last_inversion_emit_t: float = -1e9
        self._inversion_cooldown_s: float = max(2.0 * period_s, 5.0)
        # M2 state.  Both keyed by coalition_id so a single role can
        # hold multiple parallel coalitions (in practice rare — one
        # inversion-per-cooldown-window — but the dict is cheap).
        self._pending_coalitions: dict[str, _PendingCoalition] = {}
        self._active_coalitions: dict[str, _ActiveCoalition] = {}
        self._coalition_counter: int = 0

        # ---- Cross-sector coalition state ----
        # When enabled, ``_check_cross_sector_invariants`` runs after the
        # intra-sector check on every tick and may open coalitions
        # spanning sectors connected by a CP.
        self.enable_cross_sector_coalitions = enable_cross_sector_coalitions
        # cp_aid -> {
        #   "sectors": list[Sector],
        #   "coupling_ratios": dict[(in_sec_v, out_sec_v), float],
        #   "rated_capacity_mw": dict[sector_v, float],
        #   "addr": AgentAddress,
        # }
        self._cp_meta: dict[str, dict[str, Any]] = dict(cp_meta or {})
        # sector -> {aid -> addr} for the initiator to reach peer-sector
        # leaders during cross-sector invitations without re-deriving
        # from topology.  Populated at scenario-build time.
        self._peer_leader_addrs: dict[Sector, dict[str, Any]] = dict(
            peer_leader_addrs or {}
        )
        # cooldown shared with intra-sector path — one inversion per
        # cooldown window prevents per-tick re-detection storms.
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
        # Coalition control-plane subscriptions.  Both filter on
        # sector so a leader in one sector never gets pulled into
        # another sector's coalition.
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
        # Inbound coalition constraints from other initiators.  Stored
        # locally so this leader's L2 ADMM consults them before
        # dispatching its own service fractions (coalition wins per
        # (sector, tier) cell while the TTL is still valid).
        self.context.subscribe_message(
            self,
            _wrap(self._on_constraint),
            lambda msg, meta: isinstance(msg, CoalitionConstraint)
            and msg.sector == self.sector,
        )
        # Periodic publish + invariant check + coalition re-assert.
        self.context.schedule_periodic_task(self._tick, delay=self.period_s)

    async def _tick(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        await self._publish()
        self._check_invariants()
        if self.enable_cross_sector_coalitions:
            self._check_cross_sector_invariants()
        await self._reassert_active_coalitions()

    async def _publish(self) -> None:
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
            if cap <= 0:  # generators, slack — not load
                continue
            sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
            tier = obs_priority(obs, behavior=self.behavior, aid=aid)
            per_tier_demand[tier] = per_tier_demand.get(tier, 0.0) + abs(cap)
            per_tier_served[tier] = per_tier_served.get(tier, 0.0) + abs(sp)

        summary = HolonSummary(
            publisher=str(self.context.aid),
            version=self._version.next(),
            caused_by={},
            timestamp_s=float(self.context.current_timestamp),
            sector=self.sector,
            per_tier_served_mw=per_tier_served,
            per_tier_demand_mw=per_tier_demand,
        )
        # Record our own latest summary too — the invariant check
        # treats self as just another publisher.
        self._peer_summaries[str(self.context.aid)] = summary
        # Cross-sector visibility (additive): mirror the summary into
        # a shared per-sector registry on the behavior.  Other-sector
        # roles read from this registry during cross-sector detection
        # without requiring a new topology mesh — fine for the
        # single-process simulation, will need a real publish path if
        # the runtime ever spans hosts.
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
        # Normalise the key to the bare aid string so it matches the
        # ``str(self.context.aid)`` key used in ``_publish`` for the
        # self-entry.  Otherwise the dict ends up with mixed keys
        # ("child-0" for self, "AgentAddress(..., aid='child-1')" for
        # peers) and the lex-smallest election picks the
        # "AgentAddress(..." prefix every time — silently disabling
        # the initiator path on the actual lex-smallest aid.
        key = getattr(sender, "aid", None) or str(sender)
        prior = self._peer_summaries.get(key)
        if prior is not None and message.version <= prior.version:
            return  # stale
        self._peer_summaries[key] = message
        # Remember the full address — used for sending coalition
        # messages back to this peer.
        self._peer_addrs[key] = sender
        # Mirror into the shared cross-sector registry so cross-sector
        # detection sees this peer too (same rationale as _publish).
        _xs_registry(self.behavior).setdefault(message.sector, {})[key] = message

    # ------------------------------------------------------------------
    # Detection + initiator election
    # ------------------------------------------------------------------

    def _is_elected_initiator(self) -> bool:
        """Return True when this leader is the lex-smallest publisher
        with non-empty summary state.

        Election is deterministic across leaders that see the same
        ``_peer_summaries`` snapshot.  Under eventual consistency
        some leaders may be one tick behind; the election can briefly
        flip if a previously-silent leader publishes for the first
        time, but the resulting double-fire is absorbed by the
        ``last-write-wins`` semantics at the L1 dispatch.  Worst case:
        one extra coalition message exchange — no deadlock.
        """
        if not self._peer_summaries:
            return False
        publishers = sorted(self._peer_summaries.keys())
        return publishers[0] == str(self.context.aid)

    def _check_invariants(self) -> None:
        """Aggregate peer summaries by tier, detect inversions, and
        (on the elected initiator) open a coalition.

        Both the diagnostic emit and the coalition formation are
        gated on initiator election so we get one event + one
        coalition per inversion cohort, instead of N of each.
        """
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Need at least one peer summary in addition to our own to
        # call an inversion "cross-holon".  The first L2.5 tick fires
        # before any peer summaries have arrived through the
        # messaging layer; deferring lets the second tick reason on
        # real cross-holon state.
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

        from scare.base.diagnostics import record_event

        emitted = False
        first_pair: tuple[int, int] | None = None
        for i in range(1, len(tiers_sorted)):
            t_prev, t_cur = tiers_sorted[i - 1], tiers_sorted[i]
            f_prev, f_cur = fracs[t_prev], fracs[t_cur]
            if f_cur > f_prev + self.inversion_tol:
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
                if first_pair is None:
                    first_pair = (t_prev, t_cur)
        if emitted:
            self._last_inversion_emit_t = now
            logger.info(
                "[%s] cross-holon priority inversion detected (sector=%s, "
                "n_publishers=%d, n_tiers=%d, fracs=%s)",
                self.context.aid, self.sector.value,
                len(self._peer_summaries), len(tiers_sorted),
                {t: round(f, 3) for t, f in fracs.items()},
            )
            if self.enable_coalition and first_pair is not None:
                # Schedule the coalition open as an instant task so
                # the check itself stays synchronous and the rest of
                # the tick (re-assert) still runs.
                self.context.schedule_instant_task(
                    self._open_coalition(first_pair, dict(demand_at_tier))
                )

    # ------------------------------------------------------------------
    # Cross-sector coalition detection + allocation
    # ------------------------------------------------------------------

    def _check_cross_sector_invariants(self) -> None:
        """Detect cross-sector priority inversions across CP-bridged
        sector pairs.

        For each CP in ``_cp_meta`` whose bridged sector pair contains
        ``self.sector``, look at the *other* sector's most recent
        summaries (read from the shared cross-sector registry) and
        flag the case where a higher-priority tier on one side is
        served at a lower fraction than a lower-priority tier on the
        other side.  When such an inversion is found and the elected
        initiator (lex-smallest aid across both sectors' active
        publishers) is this role, open a cross-sector coalition.
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
                # Initiator election: lex-smallest aid across the
                # union of publishers from both sides.  Skip if we're
                # not it — the elected initiator runs the coalition.
                union_aids = sorted(
                    set(own_summaries.keys()) | set(peer_summaries.keys())
                )
                if not union_aids or union_aids[0] != own_aid:
                    continue
                self._last_xs_inversion_emit_t = now
                from scare.base.diagnostics import record_event

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
                return  # one coalition per tick is enough

    def _find_inversion_pair(
        self,
        own_summaries: dict[str, HolonSummary],
        peer_summaries: dict[str, HolonSummary],
    ) -> tuple[int, int, float, float] | None:
        """Return ``(t_own_high, t_peer_low, frac_own, frac_peer)`` if a
        cross-sector inversion exists, or None.

        Definition: there exists ``t_own_high`` on the own-sector side
        with strict priority over ``t_peer_low`` on the peer side
        (lower tier number is higher priority) AND the own side's
        fraction is at least ``inversion_tol`` below the peer's.
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
                    continue  # not an inversion (peer not lower-priority)
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

        No invitation round in this first cut — we rely on the
        scenario-build-time ``_cp_meta`` for the CP's rated capacity
        and coupling, and on the cross-sector summary registry for
        the leaders' current per-tier state.  Skipping the round-trip
        keeps the latency low and the implementation surface small;
        if future work needs per-CP capacity that varies at runtime
        (e.g. a CP that publishes its own beacon), the invitation/
        acceptance handshake can be added without changing the
        dispatch path.
        """
        meta = self._cp_meta.get(cp_aid)
        if meta is None:
            return
        coupling = meta.get("coupling_ratios", {}) or {}
        rated = meta.get("rated_capacity_mw", {}) or {}
        cp_addr = meta.get("addr")
        if cp_addr is None:
            return

        # Determine direction.  Coupling is keyed by (in_sec_v, out_sec_v).
        # We need the CP to push *into* own_sec (raise own's deficit-side
        # service) and draw *from* peer_sec.
        key = (peer_sec.value, own_sec.value)
        eta = float(coupling.get(key, 0.0))
        cp_cap_out = float(rated.get(own_sec.value, 0.0))
        if eta <= 0.0 or cp_cap_out <= 0.0:
            # CP cannot push into own_sec in this direction — skip.
            return

        registry = _xs_registry(self.behavior)
        own_summaries = registry.get(own_sec, {})
        peer_summaries = registry.get(peer_sec, {})

        own_dem, own_ser = self._aggregate_tier(own_summaries)
        peer_dem, peer_ser = self._aggregate_tier(peer_summaries)

        deficit_own_high = max(0.0, own_dem.get(t_own_high, 0.0) - own_ser.get(t_own_high, 0.0))
        served_peer_low = peer_ser.get(t_peer_low, 0.0)
        # Peer-side freeable supply (in own-sec MW after η): how much
        # can the CP produce in own_sec by drawing peer's tier_low served?
        peer_freeable_own = served_peer_low * eta

        # Transfer is bounded by deficit, peer's available served, and CP rated.
        transfer_out = min(deficit_own_high, peer_freeable_own, cp_cap_out)
        if transfer_out <= 1e-6:
            logger.info(
                "[%s] cross-sector coalition skipped: nothing to transfer "
                "(deficit=%.4f, peer_freeable=%.4f, cp_cap=%.4f)",
                self.context.aid, deficit_own_high, peer_freeable_own, cp_cap_out,
            )
            return
        transfer_in = transfer_out / eta

        # Compute new service fractions.
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

        # Resolve leader addresses for both sectors.  ``own_sec`` uses
        # our own peer book (populated by intra-sector summaries +
        # _peer_addrs); ``peer_sec`` uses the scenario-supplied
        # ``_peer_leader_addrs`` map.
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

        # Issue identifiers + coalition state.
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

        # Persist into the shared constraint store so L2 (per-sector
        # ADMM) and L3 (CP ADMM) both see the commitment.
        if self._constraint_store is not None:
            self._constraint_store.set(
                coalition_id=coalition_id,
                sector=own_sec,
                service_fraction_by_tier={t_own_high: new_frac_own},
                issued_at=now,
                ttl_s=float(self.coalition_constraint_ttl_s),
            )
            # Use a distinct id for the peer-side record so it's not
            # overwritten by the own-side set above.
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

        from scare.base.diagnostics import record_event

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
        """Send one ``StartBalanceNegotiation`` per sector to its
        leaders + one ``CPCommitment`` per CP.  Idempotent — the same
        helper is used for the initial fire and every re-assert tick.
        """
        # Per-sector leader dispatch.
        for sec_v, addrs in active.leader_addrs_by_sector.items():
            tier_map = active.service_fraction_by_sector_tier.get(sec_v, {})
            if not tier_map:
                continue
            payload = StartBalanceNegotiation(
                service_fraction_by_sector_priority={sec_v: dict(tier_map)},
            )
            for addr in addrs:
                await self.context.send_message(payload, receiver_addr=addr)
        # Self-dispatch on own sector so own L1 dispatches the fraction too.
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
        # CP commitments.
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
        target_pair: tuple[int, int],
        demand_at_tier: dict[int, float],
    ) -> None:
        """Build the coalition member list and broadcast invitations.

        Members are publishers whose latest summary has non-zero
        demand in either of the two target tiers — those are the
        leaders whose dispatch will actually be affected by a service-
        fraction change at those tiers.  Adding peers with zero
        demand at the targets would just inflate message volume.

        Self is always a member: the initiator's own holon will also
        be re-allocated by the coalition's fractions.
        """
        tier_high, tier_low = target_pair
        member_aids: list[str] = [str(self.context.aid)]
        for aid, summary in self._peer_summaries.items():
            if aid == str(self.context.aid):
                continue
            d_h = float(summary.per_tier_demand_mw.get(tier_high, 0.0))
            d_l = float(summary.per_tier_demand_mw.get(tier_low, 0.0))
            if d_h > 1e-9 or d_l > 1e-9:
                member_aids.append(aid)
        if len(member_aids) < 2:
            return

        self._coalition_counter += 1
        coalition_id = f"{self.context.aid}#{self._coalition_counter}"
        now = float(self.context.current_timestamp)
        pending = _PendingCoalition(
            coalition_id=coalition_id,
            sector=self.sector,
            target_tiers=(tier_high, tier_low),
            member_aids=tuple(member_aids),
            started_at=now,
        )
        # Pre-seed the initiator's own acceptance from a local
        # observe — it would loop back to us through the messaging
        # layer anyway, and we already know our own state.
        own_acc = self._local_acceptance(coalition_id, target_pair)
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
            target_tiers=tuple(target_pair),
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
            "[%s] coalition %s opened: tiers=(%d,%d) members=%d (invitations=%d)",
            self.context.aid, coalition_id, tier_high, tier_low,
            len(member_aids), n_sent,
        )
        # Schedule the allocation pass after the acceptance window.
        try:
            self.context.schedule_timestamp_task(
                self._close_and_allocate(coalition_id),
                timestamp=now + float(self.coalition_accept_window_s),
            )
        except Exception:
            # Defensive: if the scheduler does not accept absolute
            # timestamps in this build, fall back to an instant task
            # — coalitions will use whatever acceptances have already
            # arrived (which is fine; absent peers simply don't
            # contribute).
            self.context.schedule_instant_task(
                self._close_and_allocate(coalition_id)
            )

    async def _on_invitation(
        self, message: CoalitionInvitation, meta: dict
    ) -> None:
        """Reply with an acceptance when included in the member list.

        We do not gate on "this leader has flex at the target tiers"
        — the initiator already pre-filtered to such leaders, and a
        no-flex member just contributes zeros which the centralised
        allocator handles cleanly.
        """
        own_aid = str(self.context.aid)
        if own_aid not in message.member_aids:
            return
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        target_pair = (
            tuple(message.target_tiers)
            if len(message.target_tiers) >= 2
            else (None, None)
        )
        acceptance = self._local_acceptance(message.coalition_id, target_pair)
        if acceptance is None:
            return
        await self.context.send_message(acceptance, receiver_addr=sender)

    def _local_acceptance(
        self,
        coalition_id: str,
        target_pair: tuple[int, int] | tuple[None, None],
    ) -> CoalitionAcceptance | None:
        """Build this leader's acceptance payload from a fresh observe
        pass over its own community.

        Mirrors :class:`HolonicCommunityRole._build_flex_answer` in
        spirit — we want per-(sector, tier) demand and per-sector
        supply so the initiator can run the same arbitration logic.
        We restrict demand to ``target_pair`` to keep the payload
        small; supply is reported across all sectors because the LP
        downstream may route freed supply through CPs.
        """
        try:
            member_aids = [self.context.aid] + [
                addr.aid for addr in topology_neighbors(self, tid="groups")
            ]
        except Exception:
            member_aids = [self.context.aid]

        target_tiers = {t for t in target_pair if t is not None}
        sector_str = self.sector.value
        supply_by_sector: dict[str, float] = {}
        demand_by_sector_priority: dict[str, dict[int, float]] = {}
        served_by_sector_priority: dict[str, dict[int, float]] = {}
        # ``demand_nodes_by_tier[tier][node_id] = mw``: the spatial
        # demand footprint the initiator uses to compute per-actor
        # reachability via the shared mirror.  Aggregating by node
        # (not aid) means the initiator does not need a global
        # aid → node_id map; each acceptance carries the slice that
        # belongs to the replying leader.
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
                supply_by_sector[sec_v] = (
                    supply_by_sector.get(sec_v, 0.0) + abs(float(cap))
                )
                continue
            if cap <= 0:  # slack / passive — skip
                continue
            if sec != self.sector:
                # Other-sector demand isn't part of this coalition's
                # arbitration (sector-scoped invariant).  Skip it
                # entirely so the payload stays focused.
                continue
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
        """Persist an incoming coalition constraint into the shared
        store so this leader's L2 ADMM consults it on dispatch.

        We trust the initiator's TTL.  Late-arriving messages with a
        ``coalition_id`` already in the store overwrite by latest
        version (Decision.version comparison handled by the caller —
        the store treats each ``set`` as authoritative for now;
        out-of-order would just briefly enforce a stale fraction).
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

        # Aggregate supply / demand summaries for diagnostics; the
        # actual allocation goes through the per-actor ADMM helper so
        # an actor with abundant supply but no live path to any
        # demand cell contributes zero (its per-cell ub gets capped
        # at the reachable-demand sum, not at raw supply).
        sector_str = self.sector.value
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
            sec_map = acc.demand_by_sector_priority.get(sector_str, {})
            for tier, dem in sec_map.items():
                demand_by_tier[tier] = demand_by_tier.get(tier, 0.0) + float(dem)
            served_map = acc.served_by_sector_priority.get(sector_str, {})
            for tier, srv in served_map.items():
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

        if not demand_by_tier:
            return

        tiers_for_admm = sorted(t for t in demand_by_tier.keys() if t >= 1)
        if not tiers_for_admm:
            return

        # Observed-served budget cap.  The M1 summaries reveal what the
        # downstream LP is currently delivering under regulation=1.0;
        # that's the realistic deliverability ceiling for the
        # coalition's region, independent of raw generator nameplate.
        # When ``total_observed_served < total_demand`` the LP is
        # bottlenecked (line / voltage limit somewhere) and the
        # coalition's job is to redistribute that limited delivery by
        # priority — shed some lower-priority tiers so the LP can
        # serve the higher-priority ones first.  We model this by
        # scaling each actor's effective supply down to its share of
        # the observed-served budget, which makes the ADMM's per-actor
        # coupling bind at the realistic ceiling instead of at raw
        # nameplate.  When ``total_observed_served >= total_demand``
        # there's no bottleneck visible and the scaling is a no-op.
        total_demand_sector = sum(demand_by_tier.values())
        budget_scale = 1.0
        if (
            total_observed_served > 0.0
            and total_supply > 0.0
            and total_demand_sector > 1e-9
            and total_observed_served < total_demand_sector - 1e-9
        ):
            # Scale = observed_served / raw_supply.  Capped at 1.0 so a
            # generous LP delivery never inflates supply beyond what
            # the actor actually holds (the ADMM's per-cell ub still
            # uses raw values internally; only the per-actor coupling
            # ``Σ_t x_g ≤ supply_g`` is tightened).
            budget_scale = min(1.0, total_observed_served / total_supply)
            for supply_map in actor_supplies:
                if sector_str in supply_map:
                    supply_map[sector_str] = (
                        float(supply_map[sector_str]) * budget_scale
                    )

        from scare.base.diagnostics import record_event
        from scare.community.deliverability import per_actor_deliverable_caps
        from scare.community.supply_priority_admm import (
            allocate_supply_priority,
        )

        # Deliverability caps: each actor's per-(sector, tier) ub is
        # narrowed to the reachable-demand sum at that tier, so supply
        # at a stranded actor cannot be committed to demand it cannot
        # physically reach.  ``mirror=None`` makes the helper return
        # all-None entries and the ADMM falls back to raw supply caps.
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
            # Safety net: priority-greedy on the aggregate pool.  Same
            # behaviour as the pre-ADMM milestone-2 allocator, so we
            # never lose the coalition entirely on a solver hiccup.
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

        # Feasibility guard: cap each tier's allocated fraction at its
        # observed served fraction.  The M1 summaries reveal what the
        # downstream LP can actually deliver — asking for more
        # triggers an LP infeasibility that monee rolls back to its
        # previous net_results, making the coalition's broadcast a
        # no-op.  Capping at observed keeps every dispatch within the
        # LP's feasibility envelope.  Side effect: the coalition can
        # never *probe* for additional capacity by raising a tier
        # above its observed level.  That's the trade-off for safety
        # without LP-in-the-loop.  When the bottleneck is shared
        # (shedding tier-4 would free routing for tier-2) the guard
        # is too conservative; when it's local (task 13's tier-2
        # disconnect) the guard prevents a failed probe that would
        # roll back to the prior state anyway.
        n_capped = 0
        for tier in list(fractions.keys()):
            dem = demand_by_tier.get(tier, 0.0)
            srv = served_by_tier.get(tier, 0.0)
            if dem <= 1e-9:
                continue
            observed_frac = max(0.0, min(1.0, srv / dem))
            if fractions[tier] > observed_frac + 1e-6:
                fractions[tier] = observed_frac
                n_capped += 1

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

        # Record the active coalition so the next ``_tick`` cycle re-
        # asserts the constraint until TTL expiry.  Dispatching the
        # initial StartBalanceNegotiation also happens through the
        # same helper, so both the initial fire and the re-fires take
        # the identical path.
        addrs: list[Any] = []
        for acc in accepting:
            aid = acc.publisher
            if aid == str(self.context.aid):
                continue  # the initiator dispatches to itself separately
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

        # Persist on the initiator's own store + broadcast a
        # CoalitionConstraint to every accepting member so their L2
        # ADMM consults the same fractions in subsequent rounds.
        # Without this, last-write-wins between L2 and L2.5 lets L2
        # overwrite per-tier whenever it rebalances; with this, L2
        # merges the active coalition fractions into its dispatch
        # for the TTL window.
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
        # Broadcast to every same-sector peer leader, not just
        # accepting members.  Holon chunks span the full leader set,
        # so a chunk initiator that wasn't in the coalition would
        # otherwise dispatch un-merged L2 fractions to coalition
        # members in its chunk — overriding the coalition's
        # per-tier decision.  Filling every leader's store ensures
        # any L2 dispatch in the sector merges the coalition's
        # fractions before sending.  Cheap: one extra small message
        # per same-sector peer.
        broadcast_targets = set()
        for addr in active.member_addrs:
            broadcast_targets.add(addr)
        for aid, addr in self._peer_addrs.items():
            if aid == str(self.context.aid):
                continue
            broadcast_targets.add(addr)
        for addr in broadcast_targets:
            await self.context.send_message(constraint_msg, receiver_addr=addr)

        await self._dispatch_active_coalition(active)

    async def _dispatch_active_coalition(
        self, active: _ActiveCoalition
    ) -> None:
        """Send the constraint as a StartBalanceNegotiation to every
        accepting member, including self via a self-message so the L1
        handler runs through its normal path.
        """
        service_fraction_by_sector_priority = {
            active.sector.value: dict(active.service_fraction_by_tier),
        }
        payload = StartBalanceNegotiation(
            service_fraction_by_sector_priority=
                service_fraction_by_sector_priority,
        )
        # Dispatch to peers.
        for addr in active.member_addrs:
            await self.context.send_message(payload, receiver_addr=addr)
        # Dispatch to self.  Using ``self.context.addr`` ensures the
        # message lands on this leader's own EnergyBalanceNegotiator
        # via the same handler path as messages from L2.
        own_addr = getattr(self.context, "addr", None)
        if own_addr is not None:
            await self.context.send_message(payload, receiver_addr=own_addr)

    async def _reassert_active_coalitions(self) -> None:
        """Per-tick TTL pruning + re-broadcast.

        Re-broadcasting on every tick is what makes the coalition's
        constraint actually "hold" against the underlying L2 ADMM:
        if L2 fires between ticks and overwrites the service-fraction
        at the L1 level, the next tick re-asserts the coalition's
        fraction within ``period_s`` seconds.
        """
        now = float(self.context.current_timestamp)
        # Prune expired records in the shared store so L2's merge
        # only sees still-valid coalition fractions.  Idempotent.
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

        Wired by the scenario builder via
        ``behavior_in(world, _trigger, on_global_event=BranchFailureEvent,
        role_types=HolonSummaryRole)``.  We do not check whether the
        failed branch is in this sector — the conservative choice is
        to invalidate on any failure event the role receives, since
        cross-sector dependencies can make a coupling-point failure
        relevant to every sector's allocation.  The next L2 ADMM
        round (triggered by the same failure through the existing
        path) will produce a fresh allocation that's grounded in the
        post-failure topology.
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
        # Cross-sector coalitions invalidate on any branch failure
        # in any of their bridged sectors — conservative, but cheap.
        # We can't filter precisely here because the store-level
        # clear above already dropped same-sector envelopes.
        self._active_xs_coalitions.clear()
        logger.info(
            "[%s] branch failure invalidated %d active + %d pending "
            "+ %d cross-sector + %d stored coalitions (sector=%s)",
            self.context.aid, n_active, n_pending, n_xs, n_store,
            self.sector.value,
        )
