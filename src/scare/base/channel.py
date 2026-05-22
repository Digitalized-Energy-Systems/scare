"""Typed pub/sub primitives with monotonic versioning for SCARE's
inter-layer communication.

Layered on top of mango — does not introduce a parallel runtime.
Cross-agent publication uses ``send_message``; the simulation clock
drives periodic publishers via ``schedule_periodic_task``.

The primitives here exist to fix a specific pathology: today L3 (CP
ADMM, ``EnergyConverterRole``) only fires reactively when L1 gossip
broadcasts ``NegotiationFinishedEvent``.  When failures don't push L1
past its start threshold (e.g. small-magnitude branches on
``simbench_lv_cp_heavy`` with the holon layer disabled), L1 stays silent
and L3 never engages — even when there is an unmet cross-sector
imbalance that the CP plants could resolve.

The fix is two changes that compose:

1. **Trigger predicate** — each layer evaluates ``should_run(inputs)``
   over inputs it watches directly, not on event-arrival semantics
   from a specific upstream layer.
2. **Monotonic version + caused_by** — every published decision carries
   a ``version`` that increments per publisher and a ``caused_by``
   map of ``{publisher: version_consumed}`` so subscribers auto-damp
   their own echoes.

This module owns the primitives.  L3 is the first layer migrated;
L1/L2 keep their existing event/message wiring for now.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_characteristic, topology_connectors

from scare.base.model import Sector
from scare.base.util import obs_setpoint

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# ---- Decision base type -----------------------------------------------------


@dataclass
class Decision:
    """Versioned, attributable payload published on a typed channel.

    Subclasses add their channel-specific payload fields.  The four
    fields here are channel-protocol fields and must not be reused.

    ``caused_by`` lets subscribers detect when a new decision was
    triggered by their own earlier publication and skip the re-fire
    (echo damping).
    """

    publisher: str
    version: int
    caused_by: dict[str, int] = field(default_factory=dict)
    timestamp_s: float = 0.0


@dataclass
class SectorImbalanceUpdate(Decision):
    """A group leader's local imbalance estimate for one sector.

    Published periodically by ``SectorImbalanceBeacon``; consumed by
    ``EnergyConverterRole`` (L3) to drive the trigger predicate.

    ``local_imbalance_mw`` is *signed*: positive = surplus available to
    export, negative = local deficit needing import.  L3 aggregates by
    sector across all publishers it sees and checks the resulting
    vector for a same-sign-skip pattern (no beneficial cross-sector
    trade exists) before invoking ADMM.

    The value reported is the leader's own contribution, not the full
    group sum — collecting the group sum already happens inside L3's
    ``AskForAvailableFlex`` round once the trigger fires.  This beacon
    is a cheap "should you wake up?" hint, not a substitute for the
    proper flex collection.
    """

    sector: Sector = Sector.ELECTRICITY
    local_imbalance_mw: float = 0.0


@dataclass
class HolonAllocation(Decision):
    """A holon's ADMM-result allocation for one of its member groups.

    Published by ``HolonicCommunityRole`` (L2) immediately after a
    successful inter-group ADMM round, alongside the existing
    ``StartBalanceNegotiation`` overrides to members.  Consumed by
    ``EnergyConverterRole`` (L3) so CPs can react to the cross-sector
    setpoint shift *without* waiting for the downstream L1 gossip to
    finish — closes the direct L2 → L3 link that today is mediated
    through three hops (L2 → L1 → gossip-finished → L3).

    ``targets_mw`` is signed in load convention (positive = consume
    from sector, negative = produce into sector).  The CP-side
    predicate aggregates across publishers and sectors the same way
    it does for ``SectorImbalanceUpdate`` — the two streams compose,
    one carries the LP-reported stress, the other carries the L2
    decision that will eventually create it.
    """

    sector: Sector = Sector.ELECTRICITY
    targets_mw: dict[str, float] = field(default_factory=dict)
    holon_id: str = ""
    residual: float = 0.0


@dataclass
class CPSetpoint(Decision):
    """A CP plant's chosen cross-sector setpoint.

    Published by ``EnergyConverterRole`` (L3) immediately after a
    successful ADMM round commits a new operating point via
    ``_apply_result``.  Consumed by ``HolonicCommunityRole`` (L2) so
    holons whose member groups are affected by this CP can
    re-evaluate their allocation directly — closes the direct L3 →
    L2 link.

    ``sector_flows_mw`` is the per-sector signed flow the CP has
    committed to (load convention).  The applied regulation factor is
    surfaced separately so subscribers can distinguish a "small
    correction" CP setpoint from one that materially redistributes
    cross-sector flow.
    """

    cp_id: str = ""
    sector_flows_mw: dict[str, float] = field(default_factory=dict)
    regulation_factor: float = 1.0


@dataclass
class HolonSummary(Decision):
    """Post-rebalance per-tier served/demand summary for one
    holon-leader's local community.

    Published periodically on the sector-wide ``holon_summary_<sector>``
    full-mesh topology so every leader can see what every other
    leader's holon is doing per priority tier.  Consumed by the L2.5
    coalition-detection logic in
    :class:`HolonSummaryRole` to identify cross-holon priority
    inversions ("leader A serves tier-2 at 37% while leader B serves
    tier-4 at 100% in the same physical component").

    Communication-only — no optimization decisions ride on this
    channel.  When an inversion fires, a separate coalition message
    (milestone 2) carries the actual coordination.

    ``per_tier_served_mw`` and ``per_tier_demand_mw`` are the leader's
    *own community* aggregates per priority tier in the named sector,
    so subscribers can compute the fraction safely on the receiver
    side (no division-by-tiny in the publisher).
    """

    sector: Sector = Sector.ELECTRICITY
    per_tier_served_mw: dict[int, float] = field(default_factory=dict)
    per_tier_demand_mw: dict[int, float] = field(default_factory=dict)


@dataclass
class CoalitionInvitation(Decision):
    """Layer-2.5 milestone-2 invitation to join an ad-hoc rebalance coalition.

    Published by the lex-smallest leader that detected a cross-holon
    priority inversion in its ``_peer_summaries`` (election keeps a
    single initiator per inversion cohort).  Sent on the same
    ``holon_summary_<sector>`` mesh used for the M1 detection signal:
    no new topology is required, and the initiator already has the
    full peer address list from the periodic publish.

    ``target_tiers`` carries the (tier_high, tier_low) pair from the
    inversion so an invited leader can reply with just the demand /
    supply slices that are relevant — keeping the per-coalition
    payload small even if the holon has many other tiers in play.

    ``ttl_s`` is the upper bound on how long the resulting constraint
    is allowed to override the underlying L2 holon-ADMM allocation
    (see :class:`HolonSummaryRole._active_coalitions`).  TTL is also
    invalidated early on any ``BranchFailureEvent`` reaching the
    initiator, per the design directive that failures invalidate
    coalition constraints immediately.
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    target_tiers: tuple[int, ...] = ()
    member_aids: tuple[str, ...] = ()
    ttl_s: float = 10.0


@dataclass
class CoalitionAcceptance(Decision):
    """Reply from an invited leader carrying its scoped flex slice.

    The initiator runs the coalition ADMM on the aggregate of the
    received acceptances (plus its own state), so the acceptance
    payload mirrors :class:`AvailableFlexAnswer`'s per-(sector, tier)
    schema — same shape the existing supply-priority ADMM in
    :class:`HolonicCommunityRole` already consumes.

    A non-accepting leader can set ``accepted=False`` to opt out (e.g.
    if it has no demand at the target tiers and no supply that could
    help).  The initiator simply skips such replies.

    ``home_node_id`` and ``demand_nodes_by_tier`` carry the spatial
    information the initiator needs to compute per-actor
    deliverability caps via the shared :class:`GridTopologyMirror`.
    Without these the coalition ADMM treats supply as fungible across
    the whole pool — fine for an undamaged grid, but on a post-failure
    topology it lets the allocator commit supply that the LP cannot
    physically route to demand on the other side of a broken edge.

    Cross-sector coalition (new)
    ----------------------------

    When a CP joins a coalition, it advertises supply on the *output*
    sectors it can drive, and lists ``coupling_ratios`` of the form
    ``{(in_sector, out_sector): mw_out_per_mw_in}`` (efficiency).  The
    coalition initiator uses this to compute the implied input-side
    draw after the supply-priority ADMM has allocated the CP's output.
    A regular sector leader leaves ``coupling_ratios`` empty.
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    accepted: bool = True
    supply_by_sector: dict[str, float] = field(default_factory=dict)
    demand_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    served_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    home_node_id: Any = None
    demand_nodes_by_tier: dict[int, dict[Any, float]] = field(default_factory=dict)
    # Cross-sector coalition payload — non-empty only for CP members.
    coupling_ratios: dict[tuple[str, str], float] = field(default_factory=dict)
    is_cp: bool = False


@dataclass
class CoalitionConstraint(Decision):
    """Coalition-issued per-(sector, tier) service-fraction constraint
    with TTL.

    Issued by the coalition initiator to every accepting member after
    the scoped ADMM converges.  Recipients write it into their
    :class:`CoalitionConstraintStore` so the underlying
    :class:`HolonicCommunityRole` consults it before dispatching its
    own ADMM result — coalition wins per-tier, L2 covers cells the
    coalition didn't touch.  Constraints expire on ``issued_at +
    ttl_s`` or on a :class:`BranchFailureEvent`.
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    service_fraction_by_tier: dict[int, float] = field(default_factory=dict)
    ttl_s: float = 8.0


@dataclass
class CPCommitment(Decision):
    """Cross-sector coalition commitment dispatched to a CP member.

    Issued by the L2.5 coalition initiator alongside the per-sector
    ``CoalitionConstraint``/``StartBalanceNegotiation`` messages when
    the coalition spans multiple sectors via a CP (the
    ``enable_cross_sector_coalitions`` path).  Carries the directional
    sector flows the CP should commit to for the TTL window:

    * ``target_flows_mw[sector]`` is signed in load convention —
      positive means "consume from this sector", negative means
      "produce into this sector".  A P2H given a coalition commit to
      serve 0.8 MW of heat receives ``{electricity: +X, heat: -0.8}``.

    The CP writes the commitment into its envelope state (see
    :class:`scare.service.cp.EnergyConverterRole`) and clamps its own
    L3 ADMM's per-sector bounds so it cannot drift outside the
    committed range.  Expiry semantics match
    :class:`CoalitionConstraint`: ``issued_at + ttl_s`` or a
    ``BranchFailureEvent``.

    Off-by-default ablation knob: when
    ``RestorationConfiguration.enable_cross_sector_coalitions`` is
    False the coalition initiator never emits this message and the CP
    runs without an envelope (legacy L3-free behaviour).
    """

    coalition_id: str = ""
    cp_id: str = ""
    target_flows_mw: dict[str, float] = field(default_factory=dict)
    ttl_s: float = 8.0


# ---- Publisher / subscriber state -----------------------------------------


class MonotonicVersion:
    """Per-publisher monotonic counter.

    A publisher increments once per *published* decision.  When the
    publisher is silent (e.g. local state unchanged so nothing to
    announce), the counter does not move — subscribers see the same
    version and the predicate returns False.
    """

    def __init__(self) -> None:
        self._v: int = 0

    @property
    def current(self) -> int:
        return self._v

    def next(self) -> int:
        self._v += 1
        return self._v


class SeenVersions:
    """Per-subscriber memory of the latest version consumed per publisher.

    ``mark`` is called *after* a successful consumption, so the
    predicate's ``is_fresh`` check uses the *unconsumed* version frontier.
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def is_fresh(self, publisher: str, version: int) -> bool:
        return version > self._seen.get(publisher, -1)

    def mark(self, publisher: str, version: int) -> None:
        prev = self._seen.get(publisher, -1)
        if version > prev:
            self._seen[publisher] = version

    def latest(self, publisher: str) -> int:
        return self._seen.get(publisher, -1)


# ---- Publisher role: SectorImbalanceBeacon --------------------------------


class SectorImbalanceBeacon(Role):
    """Periodic publisher of ``SectorImbalanceUpdate`` on group leaders.

    Installed alongside ``EnergyBalanceNegotiator`` on the group leader
    of each ``groups`` topology cluster when ``enable_cp_admm`` is True.
    Discovers CP destinations via the same cross-topology link that
    ``EnergyConverterRole`` uses for its own NegotiationFinishedEvent
    fan-out — no new topology required.

    Every tick advances the version and publishes the current local
    imbalance.  An earlier draft gated publishes on a change-vs-last-
    publish dead-band, but that left the per-publisher version stuck
    at 1 for a quiescent grid: when stress arrived later, the
    subscriber's ``SeenVersions`` already had version 1 marked and
    skipped the new (still-version-1) decision.  The dead-band
    correctly belongs on the *subscriber* (``_PREDICATE_DEAD_BAND_MW``
    in cp.py) — the publisher's job is to keep the version frontier
    advancing so the subscriber can detect change at all.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        period_s: float = 0.5,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.period_s = period_s
        self._version = MonotonicVersion()

    def setup(self) -> None:
        logger.debug(
            "[%s] SectorImbalanceBeacon setup: sector=%s period_s=%.2f",
            self.context.aid, self.sector.value, self.period_s,
        )
        self.context.schedule_periodic_task(self._tick, delay=self.period_s)

    async def _tick(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # The group leader is registered in the ``groups`` topology;
        # ``connect_topologies(cps_topo, groups_topo, sector.value)``
        # registers the reverse link on the groups side, so ``tid=
        # "groups"`` is the correct lookup direction for finding CP
        # connectors from a leader (querying with ``tid="cps"`` returns
        # empty since the leader is not in the cps topology itself).
        #
        # Mango's ``Topology._connectors`` deduplicates by agent uid, so
        # an agent marked with multiple connector types only appears
        # under one of them.  For multi-sector CPs (CHP marks
        # electricity+heat+gas) this means heat / gas group leaders see
        # no connectors and the beacon stays silent on those sectors —
        # mirrors the existing balance.py NegotiationFinishedEvent
        # broadcast which has the same latent limitation.  When a real
        # heat/gas channel is needed, fix it at the mango layer rather
        # than papering over it here.
        connectors = topology_connectors(self, tid="groups")
        if not connectors:
            return

        try:
            obs = self.behavior.observe(self.context.aid) or {}
        except (AttributeError, KeyError):
            return
        if not obs:
            return

        # Signed *unmet flow* through this aid post-regulation, same
        # convention as cp.py:_run_admm:
        #   positive = unmet demand (load curtailed, CP-import could
        #              help by bringing energy in from another sector)
        #   negative = unsupplied supply (gen curtailed, CP-export could
        #              place this energy in another sector)
        # ``cap - sp`` is just static headroom and reads 0 for every
        # nominally-operating load — useless for stress signalling.  We
        # need the regulation gap so the predicate fires when the LP
        # has been forced to curtail.
        sp = obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        try:
            reg = float(obs.get("regulation", 1.0))
        except (TypeError, ValueError):
            reg = 1.0
        imbalance = sp * (1.0 - reg)

        # No publisher-side dead-band.  Initial attempt gated on a
        # ``|Δimbalance| < dead_band`` filter, but on a quiescent grid
        # the first publish carries imb≈0 and every subsequent tick
        # is within the dead-band of zero — version never advances, so
        # the subscriber's SeenVersions never sees fresh data even when
        # the underlying regulation shifts later.  Predicate-side dead-
        # banding (``_PREDICATE_DEAD_BAND_MW`` in cp.py) is the right
        # place for that gate; the publisher's job is to keep version
        # advancing so the subscriber can re-evaluate.
        logger.debug(
            "[%s] beacon publish: sector=%s imb=%.4f v=%d to %d cps",
            self.context.aid, self.sector.value, imbalance,
            self._version.current + 1, len(connectors),
        )

        decision = SectorImbalanceUpdate(
            publisher=self.context.aid,
            version=self._version.next(),
            caused_by={},
            timestamp_s=float(self.context.current_timestamp),
            sector=self.sector,
            local_imbalance_mw=imbalance,
        )

        for addr in connectors:
            await self.context.send_message(decision, receiver_addr=addr)
