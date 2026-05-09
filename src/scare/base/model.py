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
    },
    Sector.GAS: {
        "pressure_pu": (0.90, 1.10),     # junction pressure [p.u.]
    },
    Sector.HEAT: {
        # Heat networks carry both hot supply (~60–130 °C) and cold return
        # (~20 °C); a single bound pair must admit both to avoid false
        # violations at consumer-side junctions.
        "t_k": (283.15, 403.15),         # junction temperature [K] (10–130 °C)
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


@dataclass
class GridPathResult:
    path: list[Any]
    uncertain_connections: list[tuple[Any, Any]]
    search_id: str = ""


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


@dataclass
class StartBalanceNegotiation:
    pass


@dataclass
class IslandingRequest:
    """Emitted when gossip negotiation converges with unresolved deficit,
    triggering the islanding fallback (improvements.txt §5 SHOULD
    "Fallback / islanding capability")."""

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


@dataclass
class CurtailmentBid:
    """Reply to a :class:`CurtailmentNeed`.  Carries a scalar willingness
    score — higher = more effective / more able to absorb curtailment —
    that the auctioneer uses to allocate the total."""

    auction_id: str
    willingness: float
    sector: Sector


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
