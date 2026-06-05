"""Typed pub/sub primitives with monotonic versioning for inter-layer
communication.

Layered on top of mango: cross-agent publication uses ``send_message``;
the simulation clock drives periodic publishers via
``schedule_periodic_task``.

Each layer evaluates a ``should_run(inputs)`` predicate over inputs it
watches directly rather than reacting to upstream events. Every
published decision carries a per-publisher monotonic ``version`` and a
``caused_by`` map of ``{publisher: version_consumed}`` so subscribers
damp their own echoes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_characteristic, topology_connectors

from scare.base.model import NegotiationFinishedEvent, Sector
from scare.base.util import obs_capacity, obs_setpoint

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# ---- Decision base type -----------------------------------------------------


@dataclass
class Decision:
    """Versioned, attributable payload published on a typed channel.

    Subclasses add channel-specific payload fields; the four fields here
    are channel-protocol fields and must not be reused. ``caused_by``
    lets a subscriber detect a decision triggered by its own earlier
    publication and skip the re-fire (echo damping).
    """

    publisher: str
    version: int
    caused_by: dict[str, int] = field(default_factory=dict)
    timestamp_s: float = 0.0


@dataclass
class SectorImbalanceUpdate(Decision):
    """A group leader's local imbalance estimate for one sector.

    Published by ``SectorImbalanceBeacon``. ``local_imbalance_mw`` is
    signed: positive = surplus available to export, negative = local
    deficit needing import. It is the leader's own contribution, not the
    group sum — a cheap "should you wake up?" hint; L3 collects the full
    group sum in its ``AskForAvailableFlex`` round once triggered.
    """

    sector: Sector = Sector.ELECTRICITY
    local_imbalance_mw: float = 0.0


@dataclass
class HolonAllocation(Decision):
    """A holon's ADMM-result allocation for one of its member groups.

    Published by ``HolonicCommunityRole`` (L2) after a successful
    inter-group ADMM round; consumed by ``EnergyConverterRole`` (L3) so
    CPs react to the cross-sector setpoint shift without waiting for
    downstream L1 gossip (direct L2 -> L3 link).

    ``targets_mw`` is signed in load convention (positive = consume from
    sector, negative = produce into sector). The CP-side predicate
    aggregates this stream and ``SectorImbalanceUpdate`` the same way.
    """

    sector: Sector = Sector.ELECTRICITY
    targets_mw: dict[str, float] = field(default_factory=dict)
    holon_id: str = ""
    residual: float = 0.0


@dataclass
class CPSetpoint(Decision):
    """A CP plant's chosen cross-sector setpoint.

    Published by ``EnergyConverterRole`` (L3) after an ADMM round commits
    a new operating point; consumed by ``HolonicCommunityRole`` (L2) so
    affected holons re-evaluate their allocation directly (L3 -> L2).

    ``sector_flows_mw`` is the per-sector signed flow committed (load
    convention). ``regulation_factor`` is surfaced separately so
    subscribers distinguish a small correction from a setpoint that
    materially redistributes cross-sector flow.
    """

    cp_id: str = ""
    sector_flows_mw: dict[str, float] = field(default_factory=dict)
    regulation_factor: float = 1.0


@dataclass
class HolonSummary(Decision):
    """Post-rebalance per-tier served/demand summary for one
    holon-leader's local community.

    Published periodically on the sector-wide ``holon_summary_<sector>``
    full mesh so every leader sees every other holon's per-tier service.
    Consumed by :class:`HolonSummaryRole` (L2.5) to detect cross-holon
    priority inversions. Communication-only — no optimization decisions
    ride on this channel; an inversion triggers a separate coalition
    message.

    ``per_tier_served_mw`` / ``per_tier_demand_mw`` are the leader's own
    community aggregates per tier, so subscribers compute the fraction
    receiver-side (no division-by-tiny in the publisher).
    """

    sector: Sector = Sector.ELECTRICITY
    per_tier_served_mw: dict[int, float] = field(default_factory=dict)
    per_tier_demand_mw: dict[int, float] = field(default_factory=dict)
    # Mirrors the ``CoalitionAcceptance`` slice so an L2 leader can run
    # ``allocate_supply_priority`` on the gossiped peer view without a
    # flex-collection round-trip. Dict shape (single key per sector role)
    # lets the kernel ingest summary and acceptance payloads via one path.
    supply_by_sector: dict[str, float] = field(default_factory=dict)
    demand_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    served_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    # Per-sector slack-only operator budget (Σ over slack members'
    # eff_budget), split from ``supply_by_sector`` so the L3 CP-ADMM caps
    # a CP's INPUT-sector draw at the binding slack's operator budget
    # rather than the aggregate generator pool.
    slack_budget_by_sector: dict[str, float] = field(default_factory=dict)
    # Publisher's monee node id on the per-sector subgraph; used for
    # deliverability filtering via ``GridTopologyMirror.reachable_from``.
    home_node_id: Any = None


@dataclass
class CoalitionInvitation(Decision):
    """L2.5 invitation to join an ad-hoc rebalance coalition.

    Published by the lex-smallest leader that detected a cross-holon
    priority inversion (election keeps one initiator per inversion
    cohort). Sent on the ``holon_summary_<sector>`` mesh used for
    detection — no new topology required.

    ``target_tiers`` carries the (tier_high, tier_low) pair so invitees
    reply with only the relevant demand/supply slices. ``ttl_s`` bounds
    how long the resulting constraint may override the L2 holon-ADMM
    allocation; it is also invalidated early on any ``BranchFailureEvent``.
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    target_tiers: tuple[int, ...] = ()
    member_aids: tuple[str, ...] = ()
    ttl_s: float = 10.0


@dataclass
class CoalitionAcceptance(Decision):
    """Reply from an invited leader carrying its scoped flex slice.

    The initiator runs the coalition ADMM on the aggregate of received
    acceptances plus its own state, so the payload mirrors
    :class:`AvailableFlexAnswer`'s per-(sector, tier) schema. Set
    ``accepted=False`` to opt out (skipped by the initiator).

    ``home_node_id`` and ``demand_nodes_by_tier`` give the initiator the
    spatial info to compute per-actor deliverability caps via the shared
    :class:`GridTopologyMirror`; without them the ADMM would treat supply
    as fungible and could commit supply the LP cannot route across a
    broken edge.

    Cross-sector: a CP advertises supply on the output sectors it drives
    and lists ``coupling_ratios`` ``{(in, out): mw_out_per_mw_in}`` so the
    initiator derives the implied input-side draw. Regular leaders leave
    ``coupling_ratios`` empty.
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    accepted: bool = True
    supply_by_sector: dict[str, float] = field(default_factory=dict)
    demand_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    served_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    home_node_id: Any = None
    demand_nodes_by_tier: dict[int, dict[Any, float]] = field(default_factory=dict)
    # Non-empty only for CP members.
    coupling_ratios: dict[tuple[str, str], float] = field(default_factory=dict)
    is_cp: bool = False


@dataclass
class CPSummary(Decision):
    """Per-CP self-description for the L3 priority-cascaded ADMM mesh.

    Published on the CP-only summary overlay by every CP carrying
    :class:`~scare.service.cp_priority_admm_role.CPPriorityAdmmRole`.
    Every CP in the same cross-sector connected component subscribes and
    accumulates the latest summary per publisher, feeding the L3
    sharing-ADMM kernel its full peer view without a coordinator or
    request/reply round. Fields are what the kernel's :class:`CPSpec`
    consumes:

    * ``capacity_by_sector`` — per-sector signed effective capacity (MW,
      load convention: positive = consumes, negative = produces). The
      coupling ratio η is baked in at build time, so the kernel's
      ``x_i = r_i · c_i`` substitution honours the CP's physics.
    * ``home_node_id`` — CP host monee node id, used for cross-sector
      reachability via the shared :class:`GridTopologyMirror`.

    Republished (delta gate + watchdog, as :class:`HolonSummary`) only
    when ``capacity_by_sector`` materially shifts or the watchdog fires.
    """

    capacity_by_sector: dict[str, float] = field(default_factory=dict)
    home_node_id: Any = None


@dataclass
class CoalitionConstraint(Decision):
    """Coalition-issued per-(sector, tier) service-fraction constraint
    with TTL.

    Issued by the coalition initiator to accepting members after the
    scoped ADMM converges. Recipients store it in
    :class:`CoalitionConstraintStore`; the underlying
    :class:`HolonicCommunityRole` consults it before dispatching its own
    ADMM result (coalition wins per-tier, L2 covers untouched cells).
    Expires on ``issued_at + ttl_s`` or a :class:`BranchFailureEvent`.
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    service_fraction_by_tier: dict[int, float] = field(default_factory=dict)
    ttl_s: float = 8.0


@dataclass
class ComponentAdmmReport(Decision):
    """A group leader's flex report for the per-(sector, active-component)
    L2 ADMM.

    Sent by every group leader to the component coordinator (lex-smallest
    aid among leaders mutually reachable on the sector's active branch
    subgraph). The coordinator buffers reports by ``round_id`` +
    ``leader_aid`` (latest-wins), runs the supply-priority ADMM over all
    reports, then dispatches a component-uniform
    :class:`ComponentAllocation` to every leader.

    Active when ``RestorationConfiguration.holon_admm_scope ==
    "component"``: each community leader is one ADMM actor (no holon
    abstraction). When a failure splits a sector, each sub-component
    re-elects its own coordinator and runs independently.

    Fields mirror the subset of :class:`AvailableFlexAnswer` the
    supply-priority ADMM consumes (per-sector supply, per-(sector, tier)
    demand), keeping wire size O(n_tiers · n_sectors).
    """

    round_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    leader_aid: str = ""
    supply_by_sector: dict[str, float] = field(default_factory=dict)
    demand_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    served_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    # Implicit ACK: echoes the last applied ``ComponentAllocation.version``
    # so the coordinator detects leaders that missed a dispatch under
    # packet loss and re-sends to them. ``-1`` = none applied yet.
    last_applied_allocation_version: int = -1


@dataclass
class ComponentAllocation(Decision):
    """Per-(sector, active-component) ADMM result: the per-tier service
    fraction every group leader in the component applies to its members.

    Issued by the component coordinator after collecting a
    :class:`ComponentAdmmReport` from each leader. Recipients rebroadcast
    as ``StartBalanceNegotiation(service_fraction_by_sector_priority=…)``
    to their community members.

    Component-uniform: every leader gets the same
    ``service_fraction_by_tier``, so all loads at the same tier in the
    same (sector, component) are served equally — no cross-community
    priority inversion.

    ``version`` is a monotone-per-coordinator dispatch counter; receivers
    echo it via ``ComponentAdmmReport.last_applied_allocation_version``
    so the coordinator detects message loss and re-sends, making the
    fire-and-forget broadcast reliable under packet loss.
    """

    round_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    service_fraction_by_tier: dict[int, float] = field(default_factory=dict)
    # Monotone-per-coordinator dispatch version; ``0`` = first dispatch
    # since startup, incremented once per ``_run_component_admm_round``
    # send. Default 0 keeps version-less senders as a single repeating
    # round (no spurious stale-version retries).
    version: int = 0


@dataclass
class CPFlexReport(Decision):
    """A CP agent's self-description for the multi-sector L3 coordinator.

    Sent by every CP to the L3 coordinator (lex-smallest CP aid in the
    multi-sector connected component per the topology mirror with
    ``allow_cp_bridges=True``) when asked for state at the start of a
    joint ADMM round. Same role :class:`ComponentAdmmReport` plays for
    community leaders. Fields build the CP's actor row:

    * ``sectors`` — sector values this CP bridges (e.g.
      ``["electricity", "heat"]``), sizing the per-cell vector.
    * ``capacity_mw`` — rated conversion capacity, signed load
      convention on the input side (positive ⇒ consumes from source).
    * ``coupling_ratios`` — keyed by ``(in_sector, out_sector)``; value
      ``r`` means ``output ≤ r · input`` (e.g. 90% P2H = 0.9).
    * ``current_setpoint_by_sector`` — the CP's realised per-sector flow
      under the previous round, used as the ADMM warm start.
    """

    cp_aid: str = ""
    sectors: list[str] = field(default_factory=list)
    capacity_mw: float = 0.0
    coupling_ratios: dict[str, float] = field(default_factory=dict)
    current_setpoint_by_sector: dict[str, float] = field(default_factory=dict)


@dataclass
class CPAllocation(Decision):
    """Per-CP allocation envelope dispatched by the L3 coordinator to
    every CP in the multi-sector component.

    Result of the joint multi-sector ADMM. Carries the CP's per-sector
    flow targets, applied via ``_apply_result`` / ``apply_regulate``
    (same semantics as ``CPSetpoint.sector_flows_mw``), addressed by
    ``cp_aid`` so one broadcast can include multiple CPs. Coupling
    (``output = ratio × input``) is already enforced in the ADMM, so the
    recipient just applies the dispatched flow.
    """

    cp_aid: str = ""
    round_id: str = ""
    sector_flows_mw: dict[str, float] = field(default_factory=dict)


@dataclass
class L3RebalanceWakeup(Decision):
    """Wake-up signal from the L3 multi-sector coordinator to every group
    leader in the active component, sent after the L3 ADMM dispatches CP
    allocations.

    Carries no payload beyond the ``sector`` filter. Recipients call
    :meth:`HolonicCommunityRole._maybe_schedule_rebalance`, marking
    ``_rebalance_dirty=True`` and scheduling a fresh L2 round on the
    post-CP-commit state.

    Dedicated message because the alternatives carry side effects:
    ``NegotiationFinishedEvent`` makes ``GenerationController`` re-apply
    regulation, and ``CoalitionConstraint`` / ``HolonAllocation`` carry
    payload other consumers act on. The CP setpoints ride on
    :class:`CPAllocation`; this only nudges L2 awake.
    """

    sector: Sector = Sector.ELECTRICITY


@dataclass
class CPCommitment(Decision):
    """Cross-sector coalition commitment dispatched to a CP member.

    Issued by the L2.5 coalition initiator (alongside the per-sector
    ``CoalitionConstraint`` / ``StartBalanceNegotiation`` messages) when
    a coalition spans sectors via a CP. Carries the directional sector
    flows the CP commits to for the TTL window:

    * ``target_flows_mw[sector]`` signed load convention — positive =
      consume, negative = produce. A P2H committed to 0.8 MW heat gets
      ``{electricity: +X, heat: -0.8}``.

    The CP writes this into its envelope state
    (:class:`scare.service.cp.EnergyConverterRole`) and clamps its L3
    ADMM per-sector bounds to it. Expiry matches
    :class:`CoalitionConstraint`. Gated on
    ``RestorationConfiguration.enable_cross_sector_coalitions``; when
    False the message is never emitted and CPs run without an envelope.
    """

    coalition_id: str = ""
    cp_id: str = ""
    target_flows_mw: dict[str, float] = field(default_factory=dict)
    ttl_s: float = 8.0


# ---- Publisher / subscriber state -----------------------------------------


class MonotonicVersion:
    """Per-publisher monotonic counter, incremented once per published
    decision. A silent publisher leaves the counter still, so subscribers
    see the same version and the predicate returns False.
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

    ``mark`` is called after a successful consumption, so ``is_fresh``
    checks against the unconsumed version frontier.
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
    """Event-driven publisher of ``SectorImbalanceUpdate`` on group leaders.

    Installed alongside ``EnergyBalanceNegotiator`` on each ``groups``
    cluster leader when ``enable_cp_admm`` is True. Discovers CP
    destinations via the same cross-topology link ``EnergyConverterRole``
    uses — no new topology required.

    Triggered by the co-located ``NegotiationFinishedEvent`` on gossip
    convergence. A slow ``watchdog_s`` republishes current state so a
    late-joining subscriber still sees the version frontier.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        watchdog_s: float = 30.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.watchdog_s = watchdog_s
        self._version = MonotonicVersion()

    def setup(self) -> None:
        logger.debug(
            "[%s] SectorImbalanceBeacon setup: sector=%s watchdog_s=%.2f",
            self.context.aid, self.sector.value, self.watchdog_s,
        )
        # Primary trigger: same-agent emit from
        # EnergyBalanceNegotiator._finish_negotiation on gossip convergence.
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_negotiation_finished
        )
        # Watchdog: advances the version frontier for late-joining subscribers.
        self.context.schedule_periodic_task(
            self._tick, delay=self.watchdog_s
        )

    def _on_negotiation_finished(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        if event.sector != self.sector:
            return
        self.context.schedule_instant_task(self._tick())

    async def _tick(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Look up CP connectors via ``tid="groups"``: the reverse link is
        # registered on the groups side, so the leader is not itself in the
        # cps topology.
        #
        # Gotcha: mango's ``Topology._connectors`` deduplicates by agent uid,
        # so a multi-sector CP (CHP marks electricity+heat+gas) appears under
        # only one connector type — heat/gas leaders see none and the beacon
        # stays silent there. Same latent limitation as the balance.py NFE
        # broadcast; fix at the mango layer if a real heat/gas channel is needed.
        connectors = topology_connectors(self, tid="groups")
        if not connectors:
            return

        try:
            obs = self.behavior.observe(self.context.aid) or {}
        except (AttributeError, KeyError):
            return
        if not obs:
            return

        # Signed unmet flow through this aid post-regulation (same
        # convention as cp.py:_run_admm):
        #   positive = unmet demand (load curtailed; CP-import could help)
        #   negative = unsupplied supply (gen curtailed; CP-export could place it)
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        sp = obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        # Regulation gap (load-convention MW): what LP curtailment removed
        # from this aid's served setpoint, signed like ``cap`` (positive
        # loads, negative generators). Use the gap, not static headroom
        # ``cap``, so the predicate fires only on actual curtailment.
        imbalance = cap - sp

        # No publisher-side dead-band: gating on |Δimbalance| would freeze
        # the version on a quiescent grid (imb≈0) and starve subscribers of
        # later motion. Dead-banding belongs in the predicate
        # (``_PREDICATE_DEAD_BAND_MW`` in cp.py); the publisher just keeps
        # the version advancing.
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
