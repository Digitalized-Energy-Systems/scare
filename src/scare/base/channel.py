"""Typed pub/sub primitives with monotonic versioning for inter-layer comms.

Layered on mango (``send_message`` + ``schedule_periodic_task``). Each layer
evaluates a ``should_run(inputs)`` predicate over watched inputs. Every decision
carries a per-publisher monotonic ``version`` and a ``caused_by`` map
(``{publisher: version_consumed}``) so subscribers damp their own echoes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from scare.base.model import Sector

logger = logging.getLogger(__name__)


# ---- Decision base type -----------------------------------------------------


@dataclass
class Decision:
    """Versioned, attributable payload published on a typed channel.

    Subclasses add payload fields; these four are protocol fields and must not be
    reused. ``caused_by`` lets a subscriber detect (and skip) its own echo.
    """

    publisher: str
    version: int
    caused_by: dict[str, int] = field(default_factory=dict)
    timestamp_s: float = 0.0


@dataclass
class SectorTierFlex:
    """The per-(sector, tier) flex quad shared by the flex/summary/coalition/
    component payloads. Composed into the Decision subclasses so the shape is
    declared once; ``holon_flex.aggregate_holon_flex`` ingests exactly these."""

    supply_by_sector: dict[str, float] = field(default_factory=dict)
    demand_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)
    served_by_sector_priority: dict[str, dict[int, float]] = field(default_factory=dict)


class SectorTierFlexLike(Protocol):
    """Structural type of anything the supply-priority ADMM can ingest — the four
    flex-quad DTOs (AvailableFlexAnswer + the three Decision subclasses) satisfy
    it, replacing the manual 'same dict shape' docstring invariant."""

    supply_by_sector: dict[str, float]
    demand_by_sector_priority: dict[str, dict[int, float]]
    served_by_sector_priority: dict[str, dict[int, float]]


@dataclass
class HolonAllocation(Decision):
    """A holon's ADMM-result allocation for one of its member groups.

    Published by ``HolonicCommunityRole`` (L2) after an inter-group ADMM round;
    consumed by ``EnergyConverterRole`` (L3) as a direct L2->L3 link so CPs react
    without waiting for L1 gossip. ``targets_mw`` is signed (load convention).
    """

    sector: Sector = Sector.ELECTRICITY
    targets_mw: dict[str, float] = field(default_factory=dict)
    holon_id: str = ""
    residual: float = 0.0


@dataclass
class CPSetpoint(Decision):
    """A CP plant's chosen cross-sector setpoint.

    Published by ``EnergyConverterRole`` (L3) after an ADMM round; consumed by
    ``HolonicCommunityRole`` (L2) as a direct L3->L2 link. ``sector_flows_mw`` is
    signed per-sector flow (load convention); ``regulation_factor`` is surfaced
    separately so subscribers gauge how materially the setpoint shifts flow.
    """

    cp_id: str = ""
    sector_flows_mw: dict[str, float] = field(default_factory=dict)
    regulation_factor: float = 1.0


@dataclass
class HolonSummary(SectorTierFlex, Decision):
    """Post-rebalance per-tier served/demand summary for one leader's community.

    Published periodically on the sector-wide ``holon_summary_<sector>`` mesh;
    consumed by :class:`HolonSummaryRole` (L2.5) to detect cross-holon priority
    inversions. Communication-only — an inversion triggers a separate coalition
    message. ``per_tier_*`` are the leader's own per-tier aggregates, so
    subscribers compute the fraction receiver-side.
    """

    sector: Sector = Sector.ELECTRICITY
    per_tier_served_mw: dict[int, float] = field(default_factory=dict)
    per_tier_demand_mw: dict[int, float] = field(default_factory=dict)
    # supply_by_sector / demand_by_sector_priority / served_by_sector_priority
    # come from the SectorTierFlex mixin (the shared ADMM-ingest slice).
    # Per-sector slack-only operator budget (Σ slack members' eff_budget), split
    # from ``supply_by_sector`` so L3 CP-ADMM caps a CP's input-sector draw at
    # the slack budget rather than the aggregate generator pool.
    slack_budget_by_sector: dict[str, float] = field(default_factory=dict)
    # Publisher's monee node id; used for deliverability filtering via
    # ``GridTopologyMirror.reachable_from``.
    home_node_id: Any = None


@dataclass
class CoalitionInvitation(Decision):
    """L2.5 invitation to join an ad-hoc rebalance coalition.

    Published by the lex-smallest leader detecting a cross-holon priority
    inversion (one initiator per cohort) on the ``holon_summary_<sector>`` mesh.
    ``target_tiers`` is the (high, low) pair scoping invitee replies; ``ttl_s``
    bounds how long the constraint overrides L2 (also voided on
    ``BranchFailureEvent``).
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    target_tiers: tuple[int, ...] = ()
    member_aids: tuple[str, ...] = ()
    ttl_s: float = 10.0


@dataclass
class CoalitionAcceptance(SectorTierFlex, Decision):
    """Reply from an invited leader carrying its scoped flex slice.

    The initiator runs coalition ADMM on the aggregate of acceptances plus its
    own state, mirroring :class:`AvailableFlexAnswer`'s per-(sector, tier) schema;
    ``accepted=False`` opts out. ``home_node_id`` / ``demand_nodes_by_tier`` give
    deliverability caps via the shared :class:`GridTopologyMirror`, so the ADMM
    cannot commit supply the LP can't route across a broken edge. A CP advertises
    output-sector supply with ``coupling_ratios`` ``{(in, out): mw_out_per_mw_in}``
    so the initiator derives the input-side draw; regular leaders leave it empty.
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    accepted: bool = True
    # supply/demand/served-by-sector come from the SectorTierFlex mixin.
    home_node_id: Any = None
    demand_nodes_by_tier: dict[int, dict[Any, float]] = field(default_factory=dict)
    # Non-empty only for CP members.
    coupling_ratios: dict[tuple[str, str], float] = field(default_factory=dict)
    is_cp: bool = False


@dataclass
class CPSummary(Decision):
    """Per-CP self-description for the L3 priority-cascaded ADMM mesh.

    Published on the CP-only overlay by every
    :class:`~scare.service.coupling.cp_priority_admm_role.CPPriorityAdmmRole`;
    every CP in the cross-sector component accumulates the latest per publisher,
    feeding the kernel its peer view with no coordinator. Fields feed
    :class:`CPSpec`:

    * ``capacity_by_sector`` — signed effective capacity (MW, load convention).
      The coupling ratio η is baked in so the kernel's ``x_i = r_i · c_i``
      substitution honours CP physics.
    * ``home_node_id`` — CP host node id, for reachability via the mirror.

    Republished (delta gate + watchdog) only on a material capacity shift.
    """

    capacity_by_sector: dict[str, float] = field(default_factory=dict)
    home_node_id: Any = None


@dataclass
class CoalitionConstraint(Decision):
    """Coalition-issued per-(sector, tier) service-fraction constraint with TTL.

    Issued by the initiator to accepting members after the scoped ADMM converges;
    stored in :class:`CoalitionConstraintStore` and consulted by
    :class:`HolonicCommunityRole` before its own ADMM result (coalition wins
    per-tier, L2 covers untouched cells). Expires on ``issued_at + ttl_s`` or a
    :class:`BranchFailureEvent`.
    """

    coalition_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    service_fraction_by_tier: dict[int, float] = field(default_factory=dict)
    ttl_s: float = 8.0


@dataclass
class ComponentAdmmReport(SectorTierFlex, Decision):
    """A group leader's flex report for the per-(sector, active-component) L2 ADMM.

    Sent by every leader to the component coordinator (lex-smallest aid among
    leaders mutually reachable on the active branch subgraph), which buffers by
    ``round_id`` + ``leader_aid`` (latest-wins), runs the supply-priority ADMM,
    then dispatches a component-uniform :class:`ComponentAllocation`. Active when
    ``holon_admm_scope == "component"`` (each leader is one ADMM actor); a sector
    split re-elects a coordinator per sub-component. Fields mirror the
    :class:`AvailableFlexAnswer` subset the ADMM consumes.
    """

    round_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    leader_aid: str = ""
    # supply/demand/served-by-sector come from the SectorTierFlex mixin.
    # Implicit ACK: echoes the last applied ``ComponentAllocation.version`` so the
    # coordinator detects missed dispatches and re-sends. ``-1`` = none yet.
    last_applied_allocation_version: int = -1


@dataclass
class ComponentAllocation(Decision):
    """Per-(sector, active-component) ADMM result: the per-tier service fraction
    every leader in the component applies to its members.

    Issued by the coordinator after collecting a :class:`ComponentAdmmReport` from
    each leader; recipients rebroadcast as ``StartBalanceNegotiation``.
    Component-uniform (same ``service_fraction_by_tier`` for all leaders), so
    same-tier loads in a (sector, component) are served equally. ``version`` is a
    monotone-per-coordinator dispatch counter receivers echo via
    ``last_applied_allocation_version``, making the broadcast reliable under loss.
    """

    round_id: str = ""
    sector: Sector = Sector.ELECTRICITY
    service_fraction_by_tier: dict[int, float] = field(default_factory=dict)
    # Monotone-per-coordinator dispatch version (0 = first), bumped once per
    # ``_run_component_admm_round`` send; default 0 avoids stale-version retries.
    version: int = 0


@dataclass
class CPFlexReport(Decision):
    """A CP agent's self-description for the multi-sector L3 coordinator.

    Sent by every CP to the L3 coordinator (lex-smallest CP aid in the
    multi-sector component, ``allow_cp_bridges=True``) at the start of a joint
    ADMM round — the CP analogue of :class:`ComponentAdmmReport`. Fields:

    * ``sectors`` — sectors this CP bridges, sizing the per-cell vector.
    * ``capacity_mw`` — rated capacity, signed load convention on the input side.
    * ``coupling_ratios`` — keyed by ``(in, out)``; ``r`` means ``output ≤ r·input``.
    * ``current_setpoint_by_sector`` — previous-round flow, used as warm start.
    """

    cp_aid: str = ""
    sectors: list[str] = field(default_factory=list)
    capacity_mw: float = 0.0
    coupling_ratios: dict[str, float] = field(default_factory=dict)
    current_setpoint_by_sector: dict[str, float] = field(default_factory=dict)


@dataclass
class CPAllocation(Decision):
    """Per-CP allocation envelope from the L3 coordinator to every CP in the
    multi-sector component.

    Result of the joint ADMM: per-sector flow targets applied via
    ``CpActuator.apply`` / ``apply_regulate`` (same as ``CPSetpoint.sector_flows_mw``),
    addressed by ``cp_aid``. Coupling is already enforced in the ADMM, so the
    recipient just applies the dispatched flow.
    """

    cp_aid: str = ""
    round_id: str = ""
    sector_flows_mw: dict[str, float] = field(default_factory=dict)


@dataclass
class L3RebalanceWakeup(Decision):
    """Wake-up from the L3 coordinator to every leader in the active component,
    sent after L3 dispatches CP allocations.

    Carries no payload beyond the ``sector`` filter; recipients call
    :meth:`HolonicCommunityRole._maybe_schedule_rebalance` to run a fresh L2 round
    on the post-CP-commit state. Dedicated message because the alternatives carry
    side effects — this only nudges L2 awake (CP setpoints ride on
    :class:`CPAllocation`).
    """

    sector: Sector = Sector.ELECTRICITY


@dataclass
class CPCommitment(Decision):
    """Cross-sector coalition commitment dispatched to a CP member.

    Issued by the L2.5 initiator (alongside ``CoalitionConstraint`` /
    ``StartBalanceNegotiation``) when a coalition spans sectors via a CP. Carries
    the directional flows the CP commits to for the TTL window: ``target_flows_mw``
    signed load convention (e.g. a P2H at 0.8 MW heat → ``{electricity: +X,
    heat: -0.8}``). The CP writes this into its envelope state and clamps its L3
    ADMM bounds to it. Gated on ``enable_cross_sector_coalitions``; when False CPs
    run without an envelope.
    """

    coalition_id: str = ""
    cp_id: str = ""
    target_flows_mw: dict[str, float] = field(default_factory=dict)
    ttl_s: float = 8.0


# ---- Publisher / subscriber state -----------------------------------------


class MonotonicVersion:
    """Per-publisher monotonic counter, bumped once per decision. A silent
    publisher leaves it still, so subscribers' predicate returns False.
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
    ``mark`` runs post-consumption, so ``is_fresh`` checks the unconsumed frontier.
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
