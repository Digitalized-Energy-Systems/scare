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

# Per-sector hard constraint bounds. Keys match observation-dict keys;
# units are the monee model's (p.u. for voltage/pressure, Kelvin for temp).
SECTOR_CONSTRAINTS: dict[Sector, dict[str, tuple[float, float]]] = {
    Sector.ELECTRICITY: {
        "vm_pu": (0.95, 1.05),           # voltage magnitude [p.u.]
        # Thermal loading. loading_percent is one-sided (0=idle, 100=limit)
        # but ``constraint_utilization`` is symmetric around the midpoint;
        # the -100 lower bound puts the midpoint at 0 (the lower half is
        # physically unreachable). On PowerLine branch agents.
        "loading_percent": (-100.0, 100.0),
    },
    Sector.GAS: {
        "pressure_pu": (0.90, 1.10),     # junction pressure [p.u.]
    },
    Sector.HEAT: {
        # DHS envelope: one pair must admit both supply (~80–130 °C) and
        # return (~40–70 °C) to avoid false violations at consumer junctions.
        "t_k": (313.15, 403.15),         # junction temperature [K] (40–130 °C)
    },
}

# Fraction of the feasible range at which an agent starts signalling
# neighbours that it is approaching a limit.
PROACTIVE_WARNING_FRACTION: float = 0.85


# Per-sector time-scale constants: electricity propagates near-instantly,
# gas changes take minutes, heat transport takes hours.
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
    # Current community leader. ``None`` ⇒ resolved statically from the
    # ``groups`` topology. Set explicitly by ``DynamicRepartitionRole``
    # after a failure-driven re-election that picked a different leader
    # than the static partition.
    leader_addr: Any | None = None


@dataclass
class RepartitionAssignment:
    """Sent from the original group leader to each orphaned member after a
    failure-driven re-partition. Carries the orphans' shared new
    ``community_id``, the elected new leader, and all fellow orphan
    addresses (so the receiver can update its ``CommunityAssignment``).
    """

    community_id: UUID
    new_leader_addr: Any
    orphan_addrs: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class CommunityReassignedEvent:
    """Local event emitted by the orphan after its CommunityAssignment is
    updated post-repartition. Roles caching community-derived state (e.g.
    leader address) should subscribe and refresh.
    """

    new_leader_addr: Any
    n_neighbors: int


@dataclass
class LeaderEmerged:
    """Cross-agent broadcast: an agent was promoted to lead an orphan
    sub-community after a failure-driven re-partition.

    Sent by the new leader to every same-sector peer it knows (via the
    ``holon_summary_<sector>`` mesh). Receivers add ``(aid, node_id)`` to
    their ``_leader_node_ids`` map so the new leader appears in
    ``_resolve_component_peer_addrs`` and can join component-scope ADMM
    and the L3 escalation path. Without it the new leader is invisible to
    the component coordinator and its reports/overrides are dropped.
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
    """Emitted after grid reconfiguration closes tie switches: new
    resources may be reachable, so balance negotiation should restart."""
    closed_switches: int = 0


@dataclass
class LineFailure:
    source_node_id: Any
    target_node_id: Any
    branch_id: tuple


@dataclass
class FailureNotice:
    """Distributed branch-failure announcement.

    Originates at the failed branch's two endpoint nodes and propagates
    hop-by-hop through the physical-grid neighbour graph. Each
    ``ProblemDetector`` deduplicates by ``(origin_addr, branch_id)``,
    decrements ``hops_remaining`` by an edge-type-dependent cost, and
    forwards to same-sector or CP-bridge grid neighbours.

    ``sector`` is the failing branch's own sector — agents react only on
    a sector match. Cross-sector coupling responses flow through
    ``ConstraintViolation``, not this notice.
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
    # saturated). Propagating the full ledger (merged by latest counter per
    # agent) avoids the double-counting an aggregate digest hits in cycles.
    # ``saturated`` flags entries at dmin/dmax so the equal-share denominator
    # counts only free agents.
    memory: dict = field(default_factory=dict)
    # Scalar dual variable for the primal-dual QP gossip. 0.0 = equal-share
    # behaviour (no QP primal update).
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
    # Running max ``loading_percent`` along the path, updated by each
    # forwarding agent from its branch obs, so the originator can rank
    # competing paths by peak thermal stress.
    max_loading_percent: float = 0.0


@dataclass
class GridPathResult:
    path: list[Any]
    uncertain_connections: list[tuple[Any, Any]]
    search_id: str = ""
    # Peak line loading along this result's path; the reconfigurator picks
    # the result with the lowest peak among those in a short window.
    max_loading_percent: float = 0.0


@dataclass
class L2RecycleEscalation:
    """Escalate a locally-detected topology change to the L2 component layer.

    A member that gets a ``FailureNotice`` can't re-run the per-component
    waterfall itself, so it sends this with ``from_member=True`` to its
    leader. The leader re-broadcasts (``from_member=False``) to every active
    component peer and runs a fresh rebalance; peers re-collect and re-report
    to the coordinator but do NOT re-broadcast, bounding fan-out to one hop.

    Riding the member → leader → component-peer mesh lets the re-cycle reach
    a coordinator many physical hops from the failure (beyond the TTL-bounded
    ``FailureNotice``) while staying driven only by locally-detecting agents.
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
    # Per-sector (flex, balance) for holon-level multi-dimensional ADMM.
    # Keys are Sector.value strings.
    flex_by_sector: dict[str, float] = field(default_factory=dict)
    balance_by_sector: dict[str, float] = field(default_factory=dict)
    # Per-tier demand/served for priority-aware allocation. Keys are tiers
    # (1=highest); values in MW (or equivalent).
    demand_by_priority: dict[int, float] = field(default_factory=dict)
    served_by_priority: dict[int, float] = field(default_factory=dict)
    # Per-(sector, tier) breakdown so the holon ADMM can build a
    # tier-stratified target instead of a single per-sector scalar.
    # ``demand_by_sector_priority["electricity"][2] = 5.0`` = 5 MW nominal
    # tier-2 electricity demand. Sector-agnostic on the producer side;
    # empty = no positive-cap load with a registered tier and known sector.
    demand_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    served_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    # Total generator-class supply capacity per sector (sum of |cap| across
    # generators / slacks). Used by the supply-priority holon ADMM mode to
    # model the scarce supply pool that priority weighting arbitrates.
    supply_by_sector: dict[str, float] = field(default_factory=dict)
    # Unmet load per sector — load the LP could not deliver (regulation
    # forced to 0 on disconnect, or otherwise shed). Separate from
    # ``balance_by_sector``, which reports flowing setpoints that collapse to
    # zero on disconnect and hide the deficit. Lets the CP ADMM trigger
    # cross-sector help for invisible-because-disconnected demand.
    unmet_by_sector: dict[str, float] = field(default_factory=dict)


@dataclass
class StartBalanceNegotiation:
    # When set, the receiving leader skips the local ask/response round and
    # uses this directly as the gossip target (= negative of the net setpoint
    # the group should absorb), so the L2 per-actor allocation drives the
    # per-group target. ``None`` ⇒ leader recomputes its own target.
    override_target: float | None = None
    # Per-(sector, tier) override from the tier-stratified holon ADMM. When
    # set, the leader bypasses the scalar target and dispatches each agent
    # with its tier's allocation slice, preserving the holon's priority
    # decision through the L2 → L1 handoff. Schema
    # ``{sector_value: {tier: target_mw}}`` (target = desired setpoint sum
    # for that sector/tier's loads). ``None`` keeps scalar-only behaviour.
    override_targets_by_sector_priority: dict[str, dict[int, float]] | None = None
    # Supply-priority allocation: per-(sector, tier) service fraction; each
    # local load at (sec, tier) gets ``factor = service_fraction[sec][tier]``
    # via ``apply_regulate``. Unlike ``override_targets_by_sector_priority``
    # (absolute MW deltas), this is demand-magnitude-independent, so the
    # priority allocation reaches each load uniformly within its tier.
    service_fraction_by_sector_priority: dict[str, dict[int, float]] | None = None


@dataclass
class LocalGenerationRequest:
    """Sent by an L1 group leader to its L2 holon peers when gossip
    converges with an unresolved deficit. The holon may absorb the residual
    cross-group via an early rebalance, then replies with a
    ``LocalGenerationApproval`` so the originator's fallback role activates
    only after L2 has acted.

    The fallback is a dispatch heuristic ramping local generator-class
    children to cover the residual — no physical islanding (switch opening,
    grid-forming); that lives in monee's ``enable_islanding`` extension."""

    sector: Sector
    residual_deficit: float


@dataclass
class LocalGenerationApproval:
    """L2's response to a ``LocalGenerationRequest``: green-lights the
    originator's ``LocalGenerationFallbackRole`` for the (possibly reduced)
    residual L2 could not absorb cross-group.

    Carried back over the holons topology so L1 fallback activation is
    mediated by L2. With ``enable_holonic=False`` the originating role emits
    this directly so the fallback path still works."""

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
    """Proactive curtailment signal: agent is approaching a limit."""

    sector: Sector
    variable: str
    value: float
    bound_low: float
    bound_high: float
    utilization: float   # 0..1, how close to the violated bound
    node_id: Any = None


@dataclass
class ConstraintStateMessage:
    """Exchanged between neighbours to build a 2-3 hop picture of
    constraint tightness."""

    sector: Sector
    variable: str
    value: float
    utilization: float
    hops_remaining: int
    origin_addr: Any = None
    # Priority-coordination fields (heat frontier controller): let a cold
    # heat load see whether lower-priority reducible heat load exists in its
    # hydraulic region and defer its own tier-blind shed to the priority
    # waterfall. ``None`` on non-heat / legacy messages.
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
    """Auction-phase broadcast announcing total curtailment required; bidders
    reply with how much they can absorb. Sender then issues per-neighbour
    :class:`CurtailmentRequest` messages."""

    sector: Sector
    total_amount: float   # aggregate fractional curtailment needed [0..1]
    auction_id: str
    # Origin/variable of the relieved violation, so a bidder can weight its
    # willingness by cross-sensitivity (electrical proximity) — a closer load
    # moves the constraint more per MW shed. ``origin_addr`` keys the bidder's
    # cached multi-hop distance. Defaulted for back-compat.
    origin_addr: Any = None
    variable: str = ""


@dataclass
class CurtailmentBid:
    """Reply to a :class:`CurtailmentNeed`. Carries a scalar willingness
    score (higher = more able to absorb curtailment) used to allocate the
    total."""

    auction_id: str
    willingness: float
    sector: Sector
    # Bidder's tier and current reducible draw. The default allocator uses
    # only ``willingness``; the line-relief waterfall also needs ``tier`` to
    # shed in reverse-priority order and ``reducible`` to know when a tier is
    # exhausted. Defaulted for back-compat.
    tier: int = 0
    reducible: float = 0.0


# ---------------------------------------------------------------------------
# Holonic community messages
# ---------------------------------------------------------------------------


@dataclass
class HolonicJoinRequest:
    """A community leader proposes to a neighbouring leader to form a
    super-community (holon)."""

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
