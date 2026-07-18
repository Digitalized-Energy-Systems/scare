from __future__ import annotations

import math
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


# --- Sector-specific physical constraint bounds ---

# Per-sector hard bounds. Keys match observation-dict keys; monee units (p.u.
# for voltage/pressure, K for temp).
SECTOR_CONSTRAINTS: dict[Sector, dict[str, tuple[float, float]]] = {
    Sector.ELECTRICITY: {
        "vm_pu": (0.95, 1.05),  # voltage magnitude [p.u.]
        # Thermal loading. One-sided (0=idle, 100=limit), but utilization is
        # symmetric around the midpoint, so -100 puts the midpoint at 0.
        "loading_percent": (-100.0, 100.0),
    },
    Sector.GAS: {
        "pressure_pu": (0.85, 1.25),  # junction pressure [p.u.]
    },
    Sector.HEAT: {
        # DHS envelope: one pair must admit both supply (~80–130 °C) and
        # return (~40–70 °C) to avoid false violations.
        "t_k": (313.15, 403.15),  # junction temperature [K] (40–130 °C)
    },
}

# Gas pressure (p.u.) at/below which a node is DE-ENERGISED, not under-pressure: a source-isolated
# region collapses to ~0 yet stays graph-connected (find_ignored_nodes misses it, no re-pressurise lever).
# 0.1 sits between that (~0) and the lowest served junction (~0.54); genuine under-pressure (0.7 vs 0.85 floor) still gates.
DEENERGISED_PRESSURE_PU: float = 0.1

# Upper companion to DEENERGISED_PRESSURE_PU. A de-energised junction can also
# saturate monee's ``pressure_squared_pu`` solver Var at its upper bound (3.0)
# instead of collapsing to 0, giving ``pressure_pu = sqrt(3) ~ 1.732``. Like the
# ~0 collapse this is a solver-bound artefact, not a physical over-pressure
# (genuinely served junctions stay at or below the 1.25 gas ceiling), so the
# same scan/monitor/recording skip it. The threshold sits just below sqrt(3) so
# only the saturated value is dropped and real over-pressure still gates.
DEENERGISED_PRESSURE_HIGH_PU: float = 1.70

# Voltage companion to DEENERGISED_PRESSURE_PU. An electricity node that loses
# its path to a grid-forming source but stays graph-connected (so neither
# monee's ``find_ignored_nodes`` nor ours excludes it) collapses to vm_pu~0
# rather than a physical under-voltage. Empirically de-energised nodes read
# exactly 0.0; the lowest genuinely-served junction stays well above the 0.95
# floor, so a 0.5 threshold drops only the collapsed reading while real
# under-voltage (e.g. 0.9 vs the 0.95 floor) still gates.
DEENERGISED_VM_PU: float = 0.5


def is_energised_reading(variable: str, value: Any) -> bool:
    """Single source of truth: does *value* read a real energised state for
    *variable*, or a de-energisation / solver-bound artefact? Consulted by every
    actuator and every grader so they cannot disagree by copy-drift.

    Bands: ``vm_pu`` > DEENERGISED_VM_PU; DEENERGISED_PRESSURE_PU < ``pressure_pu``
    < DEENERGISED_PRESSURE_HIGH_PU; ``t_k`` > 0. A variable with no artefact (e.g.
    ``loading_percent``) is always energised. None / non-numeric / non-finite is
    NOT energised — the ``float()`` guard rejects None before ``math.isfinite``.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    if variable == "vm_pu":
        return v > DEENERGISED_VM_PU
    if variable == "pressure_pu":
        return DEENERGISED_PRESSURE_PU < v < DEENERGISED_PRESSURE_HIGH_PU
    if variable == "t_k":
        return v > 0.0
    return True


# Fraction of feasible range at which an agent warns neighbours it nears a limit.
PROACTIVE_WARNING_FRACTION: float = 0.85


# Per-sector time-scales: electricity near-instant, gas minutes, heat hours.
# These sim-second periods are physical durations only after multiplying by the
# environment's ``physics_time_scale`` (e.g. at scale=720 the heat 5 s poll maps
# to ~1 physical hour). At ``physics_time_scale=1`` the sim clock IS physical
# time, so they must be set to real physical periods (gas minutes, heat tens of
# minutes) via ``set_sector_timescale`` — the defaults below are the compressed
# small-clock values used by the 1:1 legacy campaigns.
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

# Pristine baseline so ``set_sector_timescale`` can reset before applying a
# per-run override — a worker process runs many tasks in sequence and must not
# leak one task's slow time-scales into the next.
_DEFAULT_SECTOR_TIMESCALE: dict[Sector, dict[str, float]] = {
    sector: dict(vals) for sector, vals in SECTOR_TIMESCALE.items()
}


def set_sector_timescale(
    overrides: "dict[str, dict[str, float]] | None" = None,
) -> None:
    """Reset the per-sector agent time-scales to their defaults, then apply
    *overrides* IN PLACE (so services that imported ``SECTOR_TIMESCALE`` see the
    change). Call once per task before the agents are built.

    *overrides* keys are carrier names (``"electricity"``/``"gas"``/``"heat"``);
    each value updates a subset of ``poll_period_s`` / ``decision_delay_s`` /
    ``convergence_rate`` for that sector. Reset-then-apply is what keeps a
    slow-time-scale task from bleeding into the next task on a reused worker
    process. Intended for ``physics_time_scale=1`` runs, where the poll/decision
    periods are real physical seconds and must be sized up (gas minutes, heat
    tens of minutes) or the discrete-event count explodes over a multi-hour
    horizon. Pair with periodic-only physics solving
    (``energy_flow_cooldown_s_override`` >= ``physics_interval_s``,
    ``energy_flow_max_acts=0``) so the physics settles at converged agent states
    rather than re-solving mid-negotiation on every setpoint write.
    """
    for sector, vals in _DEFAULT_SECTOR_TIMESCALE.items():
        SECTOR_TIMESCALE[sector] = dict(vals)
    for name, vals in (overrides or {}).items():
        try:
            sector = Sector(name)
        except ValueError:
            continue
        SECTOR_TIMESCALE.setdefault(sector, {}).update(
            {k: float(v) for k, v in vals.items()}
        )


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
    # Current leader. None ⇒ resolved statically from ``groups``. Set by
    # DynamicRepartitionRole after a failure-driven re-election.
    leader_addr: Any | None = None


@dataclass
class RepartitionAssignment:
    """New community_id, leader, and fellow-orphan addresses sent to each
    orphan after a failure-driven re-partition."""

    community_id: UUID
    new_leader_addr: Any
    orphan_addrs: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class CommunityReassignedEvent:
    """Local event emitted by the orphan after its CommunityAssignment updates
    post-repartition; roles caching community state should refresh."""

    new_leader_addr: Any
    n_neighbors: int


@dataclass
class LeaderEmerged:
    """Broadcast that an agent was promoted to lead an orphan sub-community.

    Sent to every same-sector peer (holon_summary mesh); receivers add
    (aid, node_id) to ``_leader_node_ids`` so the new leader joins
    component-scope ADMM and L3 escalation. Without it it stays invisible
    to the coordinator and its reports/overrides are dropped.
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
    # Gossip nid this finish belongs to. Non-empty on the peer convergence
    # broadcast so members (incl. the blocked originator) can release matching
    # gossip state instead of waiting for the wallclock timeout. "" on local
    # emits and legacy senders (no release).
    negotiation_id: str = ""


@dataclass(frozen=True)
class OptimizationFinishedLocalEvent:
    result: list[float]  # [el_mw, heat_mw, gas_kgps]


@dataclass
class ReconfigurationCompletedEvent:
    """Emitted after reconfiguration closes tie switches; new resources may be
    reachable, so balance negotiation should restart."""

    closed_switches: int = 0


@dataclass
class LineFailure:
    source_node_id: Any
    target_node_id: Any
    branch_id: tuple


@dataclass
class FailureNotice:
    """Distributed branch-failure announcement.

    Originates at the failed branch's endpoints and propagates hop-by-hop;
    each ProblemDetector dedups by (origin_addr, branch_id), decrements
    hops_remaining by edge cost, and forwards to same-sector / CP-bridge
    neighbours. ``sector`` is the failing branch's own — agents react only on a
    match (cross-sector responses flow through ConstraintViolation).
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
    # Per-agent ledger agent_key -> (delta, counter, priority, saturated).
    # Full ledger (merged by latest counter) avoids the double-counting an
    # aggregate digest hits in cycles; ``saturated`` excludes dmin/dmax entries
    # from the equal-share denominator.
    memory: dict = field(default_factory=dict)
    # Scalar dual for the primal-dual QP gossip. 0.0 = equal-share (no QP update).
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
    # Running max loading_percent along the path (each forwarder updates from
    # its branch obs) so the originator can rank paths by peak thermal stress.
    max_loading_percent: float = 0.0


@dataclass
class GridPathResult:
    path: list[Any]
    uncertain_connections: list[tuple[Any, Any]]
    search_id: str = ""
    # Peak line loading along this path; reconfigurator picks the lowest peak in
    # a short window.
    max_loading_percent: float = 0.0


@dataclass
class L2RecycleEscalation:
    """Escalate a locally-detected topology change to the L2 component layer.

    A member sends this (from_member=True) to its leader, which re-broadcasts
    (from_member=False) to every active component peer and rebalances; peers
    re-report but don't re-broadcast, bounding fan-out to one hop. Reaches a
    coordinator beyond the TTL-bounded FailureNotice's reach.
    """

    sector: Sector
    from_member: bool = False


@dataclass
class AskForAvailableFlex:
    include_connectors: bool = False
    # Flex-collection round identity; answers echo it so the collector can
    # ignore late replies from an earlier round. "" = legacy sender.
    round_id: str = ""


@dataclass
class AvailableFlexAnswer:
    flex: float
    balance: float
    shedded: float
    sector: Sector
    # Per-sector (flex, balance) for holon multi-dimensional ADMM; keys are
    # Sector.value strings.
    flex_by_sector: dict[str, float] = field(default_factory=dict)
    balance_by_sector: dict[str, float] = field(default_factory=dict)
    # Per-tier demand/served for priority-aware allocation. Tiers 1=highest;
    # values in MW (or equivalent).
    demand_by_priority: dict[int, float] = field(default_factory=dict)
    served_by_priority: dict[int, float] = field(default_factory=dict)
    # Per-(sector, tier) breakdown so the holon ADMM builds a tier-stratified
    # target. e.g. ["electricity"][2] = 5 MW tier-2 demand. Empty = no
    # positive-cap load with a registered tier and known sector.
    demand_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    served_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    # Total generator-class supply capacity per sector (Σ|cap|). Models the
    # scarce supply pool the supply-priority holon ADMM arbitrates.
    supply_by_sector: dict[str, float] = field(default_factory=dict)
    # Unmet load per sector (LP couldn't deliver). Separate from
    # ``balance_by_sector``, whose setpoints collapse to 0 on disconnect and
    # hide the deficit; lets the CP ADMM help disconnected demand.
    unmet_by_sector: dict[str, float] = field(default_factory=dict)
    # Echo of AskForAvailableFlex.round_id; "" = legacy responder.
    round_id: str = ""


@dataclass
class StartBalanceNegotiation:
    # When set, the leader uses this directly as the gossip target (= negative
    # net setpoint to absorb) instead of a local ask/response round. None ⇒
    # leader recomputes its own.
    override_target: float | None = None
    # Per-(sector, tier) override from the tier-stratified holon ADMM: dispatch
    # each agent with its tier's slice, preserving the holon's priority through
    # L2→L1. Schema {sector_value: {tier: target_mw}}. None = scalar-only.
    override_targets_by_sector_priority: dict[str, dict[int, float]] | None = None
    # Supply-priority allocation: per-(sector, tier) service fraction applied as
    # the load's factor. Unlike the MW-delta override above, it's demand-
    # magnitude-independent, reaching each load uniformly within its tier.
    service_fraction_by_sector_priority: dict[str, dict[int, float]] | None = None


@dataclass
class LocalGenerationRequest:
    """Sent by an L1 leader to its L2 holon peers when gossip converges with an
    unresolved deficit. L2 may absorb it cross-group, then replies with a
    LocalGenerationApproval so the fallback activates only after L2 acts.

    Fallback ramps local generator children to cover the residual — no physical
    islanding (that lives in monee's ``enable_islanding`` extension)."""

    sector: Sector
    residual_deficit: float


@dataclass
class LocalGenerationApproval:
    """L2's response to a LocalGenerationRequest: green-lights the originator's
    fallback for the residual L2 couldn't absorb cross-group. With
    ``enable_holonic=False`` the originating role emits this directly."""

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


# --- Grid constraint violation / warning events ---


@dataclass(frozen=True)
class ConstraintViolation:
    """Emitted when a local grid measurement exceeds hard bounds."""

    sector: Sector
    variable: str  # e.g. "vm_pu", "pressure_pu", "t_k"
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
    utilization: float  # 0..1, how close to the violated bound
    node_id: Any = None


@dataclass
class ConstraintStateMessage:
    """Exchanged between neighbours to build a 2-3 hop picture of constraint
    tightness."""

    sector: Sector
    variable: str
    value: float
    utilization: float
    hops_remaining: int
    origin_addr: Any = None
    # Priority-coordination fields (heat frontier): let a cold heat load see
    # lower-priority reducible load in its region and defer to the waterfall.
    # None on non-heat / legacy messages.
    priority_tier: int | None = None
    reducible: float | None = None
    # Static water-subnetwork id of the origin — scopes waterfall partners to
    # peers that share hydraulics. None = unknown (admit, legacy behavior).
    component_id: Any = None


@dataclass
class CurtailmentRequest:
    """Sent to neighbours on a violation, asking them to reduce injection /
    consumption."""

    sector: Sector
    amount: float  # MW / kg/s / W to curtail (positive = reduce)


@dataclass
class CurtailmentNeed:
    """Auction-phase broadcast of total curtailment required; bidders reply
    with how much they can absorb, then the sender issues per-neighbour
    :class:`CurtailmentRequest`."""

    sector: Sector
    total_amount: float  # aggregate fractional curtailment needed [0..1]
    auction_id: str
    # Origin/variable of the violation, so a bidder can weight willingness by
    # electrical proximity. ``origin_addr`` keys cached multi-hop distance.
    origin_addr: Any = None
    variable: str = ""
    # Excess-injection violation (over-voltage): bidders curtail generation
    # only and exclude load-shed. See ``curtail_willingness``.
    injection_relief: bool = False


@dataclass
class CurtailmentBid:
    """Reply to :class:`CurtailmentNeed`: scalar willingness (higher = more
    able to absorb) used to allocate the total."""

    auction_id: str
    willingness: float
    sector: Sector
    # Bidder's tier + reducible draw. Default allocator uses only willingness;
    # the line-relief waterfall needs ``tier`` (reverse-priority order) and
    # ``reducible`` (tier-exhaustion). Defaulted for back-compat.
    tier: int = 0
    reducible: float = 0.0


# --- Holonic community messages ---


@dataclass
class HolonicJoinRequest:
    """A leader proposes to a neighbouring leader to form a holon."""

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
    """Which holon (super-community) an agent belongs to."""

    holon_id: UUID | None = None
    level: int = 0
    parent_addr: Any = None  # address of the holon leader
    child_community_ids: list[UUID] = field(default_factory=list)
