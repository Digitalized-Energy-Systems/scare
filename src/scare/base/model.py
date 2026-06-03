from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import UUID


class Sector(str, Enum):
    ELECTRICITY = "electricity"
    HEAT = "heat"
    GAS = "gas"


class SystemStrategy(Enum):
    CP_TO_GROUP = auto()
    GROUP_TO_CP = auto()
    SIMULTANEOUSLY = auto()


# ---------------------------------------------------------------------------
# Sector-specific physical constraint bounds
# ---------------------------------------------------------------------------

# Default safety margins per sector.  Agents operate with conservative
# margins so that local decisions don't cause violations elsewhere (see
# improvements.txt §5 "Conservative feasibility margins").
#
# Keys match the observation dicts returned by
# ``RestorationEnvironmentBehavior.observe()``.  Units are those of the
# underlying monee model (p.u. for voltage/pressure, Kelvin for temp).
SECTOR_CONSTRAINTS: dict[Sector, dict[str, tuple[float, float]]] = {
    Sector.ELECTRICITY: {
        "vm_pu": (0.95, 1.05),           # voltage magnitude [p.u.]
        # Line / transformer thermal loading.  loading_percent is a
        # one-sided quantity (0 = idle, 100 = at limit), but the
        # existing ``constraint_utilization`` formula is symmetric
        # around the midpoint.  Setting the lower bound to -100
        # places the midpoint at 0 and keeps the utility formula
        # unchanged; values < 0 cannot occur physically, so the
        # lower half is just unused.  Surfaced as an observation
        # on PowerLine branch agents — see
        # ``GridConstraintMonitor`` branch mode.
        "loading_percent": (-100.0, 100.0),
    },
    Sector.GAS: {
        "pressure_pu": (0.90, 1.10),     # junction pressure [p.u.]
    },
    Sector.HEAT: {
        # DHS operating envelope: supply (~80–130 °C) and return
        # (~40–70 °C); a single bound pair must admit both to avoid
        # false violations at consumer-side junctions.
        "t_k": (313.15, 403.15),         # junction temperature [K] (40–130 °C)
    },
}

# Proactive warning threshold: fraction of the feasible range at which an
# agent starts signalling neighbours that it is approaching a limit.
PROACTIVE_WARNING_FRACTION: float = 0.85


# Per-sector time-scale constants (improvements.txt §5 "Sector-specific
# time-scale awareness").  Electricity propagates nearly instantly, gas
# changes take minutes, heat transport takes hours.
SECTOR_TIMESCALE: dict[Sector, dict[str, float]] = {
    Sector.ELECTRICITY: {
        "poll_period_s": 0.5,
        "convergence_rate": 0.6,
        "decision_delay_s": 0.0,
    },
    Sector.GAS: {
        "poll_period_s": 2.0,
        "convergence_rate": 0.3,
        "decision_delay_s": 1.0,
    },
    Sector.HEAT: {
        "poll_period_s": 5.0,
        "convergence_rate": 0.15,
        "decision_delay_s": 3.0,
    },
}


@dataclass
class EnergyData:
    electricity: dict[str, float] = field(default_factory=dict)
    gas: dict[str, float] = field(default_factory=dict)
    heat: dict[str, float] = field(default_factory=dict)

    def get_sector(self, sector: Sector) -> dict[str, float]:
        match sector:
            case Sector.ELECTRICITY:
                return self.electricity
            case Sector.GAS:
                return self.gas
            case Sector.HEAT:
                return self.heat
        raise ValueError(f"Unknown sector: {sector}")


@dataclass
class CommunityAssignment:
    community_id: UUID | None = None
    neighbors: list[Any] = field(default_factory=list)
    # Leader of the current community.  ``None`` ⇒ leader resolved
    # statically from the ``groups`` topology characteristic.  Set
    # explicitly by ``DynamicRepartitionRole`` after a failure-driven
    # re-election so downstream code can pick the right leader when
    # the orphaned sub-community has chosen a different one than the
    # static partition assigned.
    leader_addr: Any | None = None


@dataclass
class RepartitionAssignment:
    """Sent from the original group leader to each orphaned member
    after a failure-driven re-partition.  Carries the orphan's new
    community membership (a fresh ``community_id`` shared by all
    orphans), the elected new leader, and the full list of fellow
    orphan addresses (so the receiving agent can update its own
    ``CommunityAssignment.neighbors``).
    """

    community_id: UUID
    new_leader_addr: Any
    orphan_addrs: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class CommunityReassignedEvent:
    """Local event emitted by the orphan after its CommunityAssignment
    has been updated post-repartition.  Downstream roles that cache
    community-derived state (e.g. cached leader address) should
    subscribe and refresh.
    """

    new_leader_addr: Any
    n_neighbors: int


@dataclass
class LeaderEmerged:
    """Cross-agent broadcast: a previously-non-leader agent has been
    promoted to lead an orphan sub-community after a failure-driven
    re-partition (``DynamicRepartitionRole``).

    Sent by the orphan that became the new leader to every same-sector
    peer it knows about (via the existing ``holon_summary_<sector>``
    mesh).  Receivers add ``(aid, node_id)`` to their local
    ``_leader_node_ids`` map so the new leader appears in
    ``_resolve_component_peer_addrs`` and can participate in component-
    scope ADMM and the L3 escalation path.

    Without this broadcast the new leader is invisible to the elected
    component coordinator: ``ComponentAdmmReport`` from the orphan sub-
    community is dropped because the coordinator's leader registry
    doesn't include the sender, and ``SlackBudgetMonitor`` overrides
    routed to the orphan leader cannot escalate beyond their own
    island.  See ``eval_full_small_20260529-181310/tasks/000088``
    (child-25 / slack__electricity__child-39 +10.6%).
    """

    leader_aid: str
    leader_addr: Any
    node_id: Any
    sector: Sector


@dataclass
class ResultService:
    aid_to_result: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class NegotiationFinishedEvent:
    new_setpoint: float
    sector: Sector


@dataclass(frozen=True)
class OptimizationFinishedLocalEvent:
    result: list[float]  # [el_mw, heat_mw, gas_kgps]


@dataclass
class ReconfigurationCompletedEvent:
    """Emitted after grid reconfiguration closes tie switches,
    signalling that new resources may be reachable and balance
    negotiation should restart."""
    closed_switches: int = 0


@dataclass
class LineFailure:
    source_node_id: Any
    target_node_id: Any
    branch_id: tuple


@dataclass
class FailureNotice:
    """Distributed branch-failure announcement.

    Originated at the two endpoint nodes of a failed branch (by
    ``ProblemDetector.on_global_event``) and propagated through the
    physical-grid neighbour graph hop-by-hop.  Each ``ProblemDetector``
    along the way deduplicates by ``(origin_addr, branch_id)``,
    decrements ``hops_remaining`` by an edge-type-dependent cost, and
    forwards to grid neighbours whose connecting branch is either
    same-sector or a coupling-point bridge.  Replaces the centralised
    ``BranchFailureEvent → behavior_in`` dispatch that violated the
    distributed-self-organisation invariant of the architecture.

    ``sector`` carries the failing branch's own sector — agents react
    only when their sector matches.  Cross-sector reactions are reached
    through CP-bridge edges *physically*, but the eventual coupling
    response (e.g. heat junction cooling because power was cut to a
    CHP) flows through ``ConstraintViolation`` rather than this notice.
    """

    branch_id: tuple
    sector: Sector
    hops_remaining: int
    origin_addr: Any


@dataclass
class AskEnergyMessage:
    negotiation_id: str
    sector: Sector


@dataclass
class ResponseEnergyMessage:
    negotiation_id: str
    setpoint: float
    available: float
    name: str = ""


@dataclass
class EnergyNegotiationMessage:
    negotiation_id: str
    sector: Sector
    negotiation_target: float
    current_delta: float
    counter: int
    # Per-agent contribution ledger: agent_key -> (delta, counter, priority,
    # saturated).  An aggregate-only digest double-counts in cycles;
    # propagating the whole ledger and merging by latest-counter-per-agent
    # avoids it.  ``saturated`` (P1) flags entries whose delta has hit
    # dmin/dmax so the equal-share denominator counts only free agents.
    memory: dict = field(default_factory=dict)
    # P6: scalar dual variable for the primal-dual QP gossip.  Default
    # 0.0 corresponds to the equal-share / pre-P6 behaviour (no QP
    # primal update).
    dual_lambda: float = 0.0


@dataclass
class GridPathMessage:
    source_addr: Any
    target_addr: Any
    target_node_id: Any
    path: list[Any]
    asked_agents: list[Any]
    uncertain_connections: list[tuple[Any, Any]]
    search_id: str = ""
    # Running maximum of ``loading_percent`` along the path so far,
    # populated by each forwarding agent from its co-located branch
    # observation.  Carried in the message so the originator can rank
    # competing paths by peak thermal stress (6c).  Zero on init.
    max_loading_percent: float = 0.0


@dataclass
class GridPathResult:
    path: list[Any]
    uncertain_connections: list[tuple[Any, Any]]
    search_id: str = ""
    # Peak line loading along the path that produced this result.
    # The reconfigurator collects all results within a short window
    # and picks the one with the lowest peak (6c).
    max_loading_percent: float = 0.0


@dataclass
class L2RecycleEscalation:
    """Escalate a locally-detected topology change to the L2 component layer.

    A community *member* that receives a ``FailureNotice`` cannot itself
    re-run the per-component waterfall — but it can tell its leader.  It
    sends this with ``from_member=True`` to its community leader; the
    leader then re-broadcasts it (``from_member=False``) to every peer in
    its active component and runs a fresh rebalance.  Peers receiving the
    ``from_member=False`` form re-collect and re-report to the (possibly
    re-elected) coordinator but do *not* re-broadcast, bounding the fan-out
    to a single hop.

    This rides the agent organisation (member → leader → component-peer
    mesh), so the re-cycle reaches a coordinator that is many physical
    hops from the failure — beyond the TTL-bounded ``FailureNotice``
    propagation — while still being driven only by agents that locally
    detected the failure (no global event subscription).
    """

    sector: Sector
    from_member: bool = False


@dataclass
class AskForAvailableFlex:
    include_connectors: bool = False


@dataclass
class AvailableFlexAnswer:
    flex: float
    balance: float
    shedded: float
    sector: Sector
    # Per-sector breakdown for multi-dimensional ADMM at holon level.
    # Keys are Sector.value strings; values are (flex, balance) tuples.
    flex_by_sector: dict[str, float] = field(default_factory=dict)
    balance_by_sector: dict[str, float] = field(default_factory=dict)
    # Priority-tier demand breakdown for upper-level priority-aware allocation.
    # Keys are priority tiers (int, 1=highest); values are MW (or equivalent).
    demand_by_priority: dict[int, float] = field(default_factory=dict)
    served_by_priority: dict[int, float] = field(default_factory=dict)
    # Per-(sector, tier) breakdown — populated alongside the tier-only
    # aggregates so the holon ADMM can build a tier-stratified target
    # vector instead of collapsing demand to a single per-sector scalar.
    # ``demand_by_sector_priority["electricity"][2] = 5.0`` means: "this
    # group has 5 MW of nominal tier-2 demand in the electricity sector".
    # ``_handle_ask_flex`` iterates every member regardless of the
    # leader's own sector, so the map is sector-agnostic on the
    # producer side; empty here just means "no positive-cap load in
    # this group had a registered tier and a known sector".
    demand_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    served_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    # Total *supply* (generator-class) capacity per sector.  Used by the
    # Route-A "supply-priority" holon ADMM mode to model the holon's
    # scarce supply pool and let priority weighting arbitrate which
    # tiers' demands get served.  Sum of |cap| across generators /
    # slack-class children in the group, per sector.
    supply_by_sector: dict[str, float] = field(default_factory=dict)
    # Unmet load per sector — captures load that the LP could not
    # deliver (regulation forced to 0 by monee's disconnect handling,
    # or otherwise shed).  Separate from ``balance_by_sector`` because
    # ``balance`` reports actually-flowing setpoints (which collapse to
    # zero on a disconnect, hiding the deficit).  The CP ADMM uses
    # this so cross-sector help can be triggered when one sector has
    # surplus and another has invisible-because-disconnected demand.
    unmet_by_sector: dict[str, float] = field(default_factory=dict)


@dataclass
class StartBalanceNegotiation:
    # When set, the receiving group leader skips the local
    # ask-energy → response-energy round and uses this value directly
    # as the gossip target (= negative of the net setpoint the group is
    # expected to absorb).  Populated by the holonic ADMM coordinator so
    # the per-actor allocation from Layer 2 actually drives the per-group
    # gossip target instead of being discarded.  None preserves the
    # original behaviour: member leader recomputes its own target.
    override_target: float | None = None
    # Per-(sector, tier) override produced by the tier-stratified holon
    # ADMM (Package C, ``enable_tier_stratified_holon_admm``).  When
    # set, the receiving leader bypasses the scalar gossip target and
    # dispatches each agent directly with the slice of its tier's
    # allocation, preserving the holon's global priority decision
    # through the L2 → L1 handoff (the scalar ``override_target`` does
    # not).  Schema: ``{sector_value: {tier: target_mw}}`` where the
    # tier-2 entry in electricity is the new desired setpoint sum for
    # all tier-2 electricity loads in this group.  ``None`` keeps the
    # legacy scalar-only behaviour.
    override_targets_by_sector_priority: dict[str, dict[int, float]] | None = None
    # Route-A supply-priority allocation: per-(sector, tier) service
    # fraction the holon has decided for the system.  Each local load
    # at (sec, tier) gets ``factor = service_fraction[sec][tier]``
    # applied via ``apply_regulate``.  Distinct from
    # ``override_targets_by_sector_priority`` (which carries absolute
    # MW deltas under the demand-side formulation): the supply-side
    # formulation gives a fractional service decision that's
    # independent of the local demand magnitude, so the holon's
    # priority allocation reaches each load uniformly within its tier.
    service_fraction_by_sector_priority: dict[str, dict[int, float]] | None = None


@dataclass
class LocalGenerationRequest:
    """Sent by an L1 group leader to its L2 holon peers when gossip
    converges with an unresolved deficit.  The holon decides whether
    to absorb the residual cross-group via an early rebalance, and
    replies with a ``LocalGenerationApproval`` so the originator's
    fallback role activates only after L2 has had a chance to act.

    See improvements.txt §5 SHOULD "Fallback / islanding capability".
    The §5 wording speaks of "islanding"; in this implementation the
    fallback is a dispatch heuristic that ramps local generator-class
    children to cover the residual — no physical islanding (switch
    opening, grid-forming) is performed by the receiver.  That lives
    in monee's ``enable_islanding`` extension."""

    sector: Sector
    residual_deficit: float


@dataclass
class LocalGenerationApproval:
    """L2's response to a ``LocalGenerationRequest``: green-lights the
    originator's ``LocalGenerationFallbackRole`` for the (possibly
    reduced) residual that L2 could not absorb cross-group.

    Carried back to the originator over the holons topology so L1
    fallback activation is mediated by L2 rather than fired locally
    behind L2's back.  In configurations without a holon layer
    (``enable_holonic=False``) the originating role emits this event
    directly so the fallback path still works."""

    sector: Sector
    residual_deficit: float


@dataclass
class BalanceNegotiationStart:
    community_id: UUID
    target: list[float]


@dataclass
class NegotiationResult:
    flexibility: float
    control_setpoint: float


@dataclass
class BalanceProblem:
    sector: Sector
    imbalance: float


@dataclass
class CHSJoinRequest:
    group_id: UUID
    group_size: int


@dataclass
class CHSJoinRequestAnswer:
    group_id: UUID
    accept: bool


# ---------------------------------------------------------------------------
# Grid constraint violation / warning events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintViolation:
    """Emitted when a local grid measurement exceeds hard bounds."""

    sector: Sector
    variable: str       # e.g. "vm_pu", "pressure_pu", "t_k"
    value: float
    bound_low: float
    bound_high: float
    node_id: Any = None


@dataclass(frozen=True)
class ConstraintWarning:
    """Proactive curtailment signal: agent is approaching a limit
    (improvements.txt §6 "Proactive curtailment signaling")."""

    sector: Sector
    variable: str
    value: float
    bound_low: float
    bound_high: float
    utilization: float   # 0..1  how close to the violated bound
    node_id: Any = None


@dataclass
class ConstraintStateMessage:
    """Exchanged between neighbours so they can build a 2-3 hop picture
    of constraint tightness (improvements.txt §5 "Multi-hop information
    propagation")."""

    sector: Sector
    variable: str
    value: float
    utilization: float
    hops_remaining: int
    origin_addr: Any = None
    # Priority-coordination fields (heat frontier controller).  Carried on
    # the same broadcast so a cold heat load can see whether lower-priority
    # reducible heat load exists in its hydraulic region and defer its own
    # (tier-blind) shed to the priority waterfall.  ``None`` on non-heat /
    # legacy messages.
    priority_tier: int | None = None
    reducible: float | None = None


@dataclass
class CurtailmentRequest:
    """Sent to neighbours when a constraint is violated, asking them to
    reduce injection / consumption."""

    sector: Sector
    amount: float  # MW / kg/s / W to curtail (positive = reduce)


@dataclass
class CurtailmentNeed:
    """Auction-phase message: broadcast to neighbours announcing the
    total curtailment required so they can bid on how much they are
    willing / able to absorb.  Sender collects bids and then issues
    per-neighbour :class:`CurtailmentRequest` messages."""

    sector: Sector
    total_amount: float   # aggregate fractional curtailment needed [0..1]
    auction_id: str
    # Origin of the violation this auction relieves, so a bidder can weight
    # its willingness by its cross-sensitivity (electrical proximity) to the
    # violated node/branch — a load close to the violation moves the
    # constraint more per MW shed than a distant one.  ``origin_addr`` keys
    # into the bidder's cached multi-hop ``ConstraintStateMessage`` distance;
    # ``variable`` selects which violation.  Defaulted for back-compat.
    origin_addr: Any = None
    variable: str = ""


@dataclass
class CurtailmentBid:
    """Reply to a :class:`CurtailmentNeed`.  Carries a scalar willingness
    score — higher = more effective / more able to absorb curtailment —
    that the auctioneer uses to allocate the total."""

    auction_id: str
    willingness: float
    sector: Sector
    # Priority tier and current reducible draw of the bidder.  The default
    # willingness-proportional allocator only consults ``willingness``; the
    # branch-downstream line-relief waterfall
    # (``enable_line_relief_waterfall``) additionally needs the bidder's tier
    # to shed in strict reverse-priority order and its reducible draw to know
    # when a tier is exhausted.  Defaulted for back-compat with bids that
    # don't carry them.
    tier: int = 0
    reducible: float = 0.0


# ---------------------------------------------------------------------------
# Holonic community messages
# ---------------------------------------------------------------------------


@dataclass
class HolonicJoinRequest:
    """A community leader proposes to a neighbouring community leader to
    form a super-community (holon)."""

    holon_id: UUID
    member_communities: list[UUID]
    level: int  # 0 = base group, 1 = super-community, …


@dataclass
class HolonicJoinAnswer:
    holon_id: UUID
    accept: bool
    community_id: UUID  # the answering community's id


@dataclass
class HolonicAssignment:
    """Stored on each agent: which holon (super-community) it belongs to."""

    holon_id: UUID | None = None
    level: int = 0
    parent_addr: Any = None  # address of the holon leader
    child_community_ids: list[UUID] = field(default_factory=list)


@dataclass
class HebbianFlexBeacon:
    """B.2: periodic broadcast from a group leader carrying its latest
    flex-side scalar surrogate $\\delta_g$ (signed imbalance / capacity).

    Receivers update their local Hebbian co-variance estimate
    $H_{gh}(t)$ from the pair $(\\delta_g, \\delta_h)$.  High-correlation
    pairs cluster naturally into the same holon.  No global view is
    required; every leader exchanges only with its same-sector
    topology neighbours.
    """

    sector: Sector
    delta_g: float          # leader's signed surrogate, normalised by capacity
    timestamp: float        # sender's simulation time at emission
