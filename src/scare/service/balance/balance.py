from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
    topology_neighbors,
)
from monee.model.child import ExtHydrGrid, ExtPowerGrid

from scare.base.addressing import is_child_aid
from scare.base.model import (
    SECTOR_CONSTRAINTS,
    SECTOR_TIMESCALE,
    AskEnergyMessage,
    AskForAvailableFlex,
    AvailableFlexAnswer,
    BalanceProblem,
    ConstraintViolation,
    ConstraintWarning,
    EnergyNegotiationMessage,
    FailureNotice,
    L2RecycleEscalation,
    LocalGenerationApproval,
    LocalGenerationRequest,
    NegotiationFinishedEvent,
    ResponseEnergyMessage,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.runtime.diagnostics import (
    record_event,
    record_negotiation,
    record_regulate,
)
from scare.base.util import (
    apply_regulate,
    async_dispatch,
    clamp_to_constraints,
    constraint_allowed_fraction,
    constraint_utilization,
    has_gen_curtail_lock,
    l2_effective_floor,
    last_actuated_factor,
    line_congestion_ceiling,
    lookup_cp_supply,
    lookup_slack,
    lookup_slack_eff_budget,
    note_actuated_factor,
    obs_capacity,
    obs_min_max,
    obs_priority,
    obs_sector,
    obs_setpoint,
    set_l2_priority_floor,
)
from scare.community.holonic import HolonicCommunityRole
from scare.service.balance.gossip_math import (
    compute_lambda_seed,
    ledger_merge,
    ledger_sum_responsiveness,
    ledger_total_delta,
    qp_primal,
    qp_priority_weight,
    step_size,
)
from scare.service.balance.grid_former import GridFormerPolicy
from scare.service.balance.neighbour_router import NeighbourRouter
from scare.service.balance.trust import TrustLedger, TrustParams

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

_DEFAULT_START_THRESHOLD = 1e-4
# Per-sector overrides (currently all share the default).
_START_THRESHOLD: dict[Sector, float] = {}

# Per-group threshold = max(floor, fraction · Σ|cap|): scales noise tolerance
# with group capacity.
_THRESHOLD_CAPACITY_FRACTION: float = 0.005
_THRESHOLD_ABS_FLOOR: float = 1e-6

# Heat util above which an agent would contribute headroom to the thermal-deficit
# target. Feeds the now-disabled heat MW-balance path.
_HEAT_CLEAR_FRACTION: float = 0.6

# Share of the unserved heat gap offered on top of delivered each flex cycle
# (``_handle_ask_flex``); geometric climb to the frontier, controller trims
# overreach. 0.2 is A/B-validated (ab_heat_priority_v2) but converges slowly over
# ~6 rebalance rounds per 30 s task (tier-4 under-settles) — 0.3-0.5 is the knob,
# though the 0.4 A/B was confounded by concurrent CP-fix tree changes.
_HEAT_L2_PROBE_SHARE: float = 0.2

# Credit ``delivered + share*(rating-delivered)`` so the free-Var former produces
# to meet offered load; a positive share over-credits when it shares an island with
# a budgeted slack (offered load routes through the slack and the L2 floor blocks
# re-shed). recoverable_islanding seed 100000023: share 0.0 -> PWSF 0.42, gas slack
# PASS; 0.5 -> 0.66, FAIL (110% over); rating -> 0.77, FAIL (151%). Default 0 keeps
# the gas slack compliant. Analogue of ``_HEAT_L2_PROBE_SHARE`` (heat has no slack
# budget).
_GRID_FORMER_SUPPLY_PROBE_SHARE: float = 0.0

_MAX_HOPS = 100

# Freshness (sim-s) of a congestion-price ceiling read in ``_apply_setpoint``;
# matches the branch monitor's publish TTL (see constraints.py).
_LINE_CONGESTION_TTL_S: float = 3.0

# Robbins-Monro step decay: gain = gamma_s / (1 + k / k0). k0 ≈ LV group size.
_STEP_DECAY_K0_DEFAULT: int = 20

# P2 stall detection: gap window range below tol while gap still exceeds the
# per-group threshold ⇒ stuck.
_STALL_WINDOW_FACTOR: int = 2
_STALL_TOL_FRACTION: float = 0.005
_STALL_TOL_FLOOR: float = 1e-6

# Base wallclock deadline per sector (heat slowest); scaled by group size at use.
_GOSSIP_TIMEOUT_BASE_S: dict[Sector, float] = {
    Sector.ELECTRICITY: 5.0,
    Sector.GAS: 15.0,
    Sector.HEAT: 30.0,
}
_GOSSIP_TIMEOUT_DEFAULT_S = 15.0
_GOSSIP_TIMEOUT_PER_AGENT_S = 0.5

# Coordination overhaul: a converged setpoint within this of the last upward-
# notified value counts as "unchanged" — the L1→L2/L3 notification is skipped so
# the holon ADMM is not re-triggered for a no-op finish.
_UPWARD_NOTIFY_TOL = 1e-3

# Poll periods of silence before a peer counts as dead.
_HEARTBEAT_MAX_AGE_MULTIPLE: float = 8.0

# Intra-sector priority tiers (lower = higher urgency). Tier 1 = critical
# (hard-locked pre-step); tiers 2-4 = QP-weighted with steep exponents.
_PRIORITY_TIERS = 4

# Byzantine cap: a participant's delta is clipped to this multiple of |target|.
_BYZANTINE_DELTA_CAP_MULTIPLE: float = 5.0


def _start_threshold(sector: Sector) -> float:
    return _START_THRESHOLD.get(sector, _DEFAULT_START_THRESHOLD)


def _is_slack_class_child(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a monee ``ExtPowerGrid``/``ExtHydrGrid`` slack child.

    Suppresses regulation writes on slacks: ``regulation < 1`` clamps the LP's
    slack envelope and the next solve goes infeasible. Covers bounded and unbounded.
    """
    if not is_child_aid(aid):
        return False
    try:
        cid = int(aid[len("child-") :])
    except ValueError:
        return False
    net = getattr(behavior, "_net", None)
    if net is None:
        return False
    try:
        child = net.child_by_id(cid)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(child.model, (ExtPowerGrid, ExtHydrGrid))


def _heat_thermal_deficit_mw(obs: dict) -> float:
    """MW of demand reduction contributed to the group's thermal-deficit target.

    ``max(0, util - ϑ_clear) · |cap|`` over the worst local constraint util.
    Loads only (cap > 0); heat sector only.
    """
    cap = obs_capacity(obs)
    if cap <= 0:
        return 0.0
    bounds = SECTOR_CONSTRAINTS.get(Sector.HEAT, {})
    worst_util = 0.0
    for var, (lo, hi) in bounds.items():
        if var in obs:
            worst_util = max(
                worst_util, constraint_utilization(float(obs[var]), lo, hi)
            )
    if worst_util <= _HEAT_CLEAR_FRACTION:
        return 0.0
    deficit_fraction = worst_util - _HEAT_CLEAR_FRACTION
    return deficit_fraction * abs(cap)


def _is_saturated(delta: float, dmin: float, dmax: float) -> bool:
    """True iff *delta* is within tolerance of either box bound.

    Tolerance scales with box magnitude so large boxes tolerate solver noise.
    """
    sat_tol = 1e-9 + 1e-6 * max(abs(dmin), abs(dmax), 1.0)
    return delta <= dmin + sat_tol or delta >= dmax - sat_tol


def _deterministic_next(neighbours: list, negotiation_id: str, counter: int) -> Any:
    """Deterministic next gossip target via hash of (negotiation_id, counter).

    Gives competing agents a consistent send order for conflict resolution.
    """
    if not neighbours:
        return None
    h = hashlib.sha256(f"{negotiation_id}:{counter}".encode()).digest()
    idx = int.from_bytes(h[:4], "big") % len(neighbours)
    return neighbours[idx]


def _deterministic_sub_round(
    agent_addr: str, negotiation_id: str, tier: int, tier_size: int
) -> int:
    """Deterministic sub-round index in [0, tier_size) for intra-tier serialization."""
    h = hashlib.sha256(f"{negotiation_id}:{tier}:{agent_addr}".encode()).digest()
    return int.from_bytes(h[:4], "big") % tier_size


@dataclass
class _GossipState:
    negotiation_id: str
    target: float
    counter: int
    current_delta: float
    starting_setpoint: float
    # Feasible-δ box anchored to the *starting* setpoint, NOT recomputed per step:
    # the agent's own regulate flips the LP's reported sp, flipping the box sign
    # into a full-shed/full-load oscillation. Anchoring keeps δ cumulative and the
    # box constant.
    dmin_starting: float = 0.0
    dmax_starting: float = 0.0
    # Ledger (delta, counter, priority, saturated); merge keeps the highest-counter
    # entry to avoid cyclic double-counting. Saturated entries are excluded from the
    # equal-share denominator so contraction doesn't collapse as the boundary set
    # grows.
    memory: dict[str, tuple[float, int, int, bool]] = field(default_factory=dict)
    # True only for the originator; peers built from a received message set
    # False. Originator records terminal diary events once per nid, preserving
    # ``started == Σ terminals``.
    is_originator: bool = False
    # P2: rolling window of post-update gaps, sized ``_STALL_WINDOW_FACTOR ·
    # n_active``. Low range + gap above threshold ⇒ originator escalates.
    gap_window: list[float] = field(default_factory=list)
    # P6: scalar dual variable for the primal-dual QP gossip. At the KKT optimum
    # it is the scarcity price λ* with ``Σ clamp(a_i · λ, dmin_i, dmax_i) = T``.
    # Gossiped with the ledger; receiver does a primal then dual update.
    dual_lambda: float = 0.0


@dataclass
class NegotiationSession:
    """In-flight negotiation state shared across the balance negotiator's
    trigger, gossip-engine, actuator, and L2-dispatch collaborators."""

    neg_seq: int = 0  # deterministic negotiation-id counter (aid/seq, not uuid4)
    active: bool = False  # in-flight guard
    gossip: _GossipState | None = None
    # Setpoint-gathering (trigger) phase, before gossip starts.
    trigger_nid: str | None = None
    trigger_responses: dict[str, float] = field(default_factory=dict)
    trigger_expected: int = 0
    group_capacity_abs: float = 0.0  # Σ|cap| over the group; drives the threshold
    restoration_floor: float = 0.0  # monotonic factor floor during restoration
    constraint_violation_active: bool = False
    last_notified_setpoint: float | None = None  # upward-notify change gate
    last_dispatched_service_fraction: dict[str, dict[int, float]] | None = None


def _credit_cp_supply(
    supply_by_sector: dict[str, float],
    cp_aids: list[str] | None,
    behavior: Any,
    now: float,
) -> None:
    """Add this leader's coupling points' committed production to the pool.

    ``supply_by_sector`` is otherwise built by walking node children, but every
    converter is a monee *branch* and has no member aid, so its output is
    invisible here. That is harmless where the carrier has native generation and
    fatal where it does not: ``simbench_lv_gas_dependent`` is built with
    ``gas_gen_share=0`` (gas exists only as P2G output), so the pool reads 0.0000,
    and the holon sheds every gas load — including tier 1 — within 0.2 s.

    Scoped to *this* leader's CP connectors, never the grid-wide fleet: a
    holon may only count converters it is actually coupled to. Credits are
    delivered-and-fresh (see :func:`~scare.base.util.lookup_cp_supply`), so a
    throttled or failed converter stops contributing on its own.
    """
    for aid in cp_aids or ():
        produced = lookup_cp_supply(behavior, aid, now)
        if not produced:
            continue
        for sec_key, mw in produced.items():
            supply_by_sector[sec_key] = supply_by_sector.get(sec_key, 0.0) + float(mw)


def _compute_flex_report(
    *,
    member_aids: list[str],
    behavior: Any,
    grid_former_policy: GridFormerPolicy,
    role_sector: Sector,
    now: float,
    enable_heat_l2_dispatch: bool,
    round_id: str,
    credit_gen_capacity: bool = False,
    cp_supply_aids: list[str] | None = None,
) -> AvailableFlexAnswer:
    """Leader-side flex aggregation over group members (read-only): supply/demand
    pools, per-(sector, tier) splits, unmet demand, and the heat delivered-plus-probe
    frontier.

    ``cp_supply_aids`` are this leader's coupling-point connectors, whose
    committed production is credited into ``supply_by_sector`` — see
    :func:`_credit_cp_supply`."""
    total_flex = 0.0
    total_balance = 0.0
    total_shedded = 0.0
    flex_by_sector: dict[str, float] = {}
    balance_by_sector: dict[str, float] = {}
    unmet_by_sector: dict[str, float] = {}
    demand_by_priority: dict[int, float] = {}
    served_by_priority: dict[int, float] = {}
    demand_by_sector_priority: dict[str, dict[int, float]] = {}
    served_by_sector_priority: dict[str, dict[int, float]] = {}
    supply_by_sector: dict[str, float] = {}
    for aid in member_aids:
        obs = behavior.observe(aid) or {}
        sector = obs_sector(obs, behavior=behavior, aid=aid)
        if sector is None:
            continue
        cap = obs_capacity(obs, behavior=behavior, aid=aid)
        sp = obs_setpoint(obs, behavior=behavior, aid=aid)
        available = cap - sp  # headroom
        sec_key = sector.value
        # A promoted island reference is a generator, never a load: credit its
        # supply and skip flex/demand accounting (its sign-flipping free var
        # otherwise reads as phantom demand).
        if grid_former_policy.is_former(aid):
            supply_by_sector[sec_key] = supply_by_sector.get(
                sec_key, 0.0
            ) + grid_former_policy.supply_credit(aid, sp)
            continue
        flex_by_sector[sec_key] = flex_by_sector.get(sec_key, 0.0) + available
        balance_by_sector[sec_key] = balance_by_sector.get(sec_key, 0.0) + sp
        # Route-A supply pool (cap < 0): generators count *delivered* |sp| (curtailed
        # supply can't inflate the pool); slacks use their budgeted rating.
        if cap < 0:
            if lookup_slack(behavior, aid) is not None:
                eff = lookup_slack_eff_budget(behavior, aid)
                gen_supply = float(eff) if eff is not None else abs(cap)
            else:
                gen_supply = abs(sp)
                # R1 lever (``enable_gen_capacity_supply``): credit available
                # capacity so the pool can ask for a ramp. A fresh over-voltage
                # curtail-lock keeps the delivered-only credit (that supply
                # must not inflate the pool); the congestion ceiling scales it.
                if (
                    credit_gen_capacity
                    and sector is not Sector.HEAT
                    and not has_gen_curtail_lock(behavior, aid, now)
                ):
                    ceiling = line_congestion_ceiling(
                        behavior, aid, now, _LINE_CONGESTION_TTL_S
                    )
                    gen_supply = max(gen_supply, abs(cap) * ceiling)
            supply_by_sector[sec_key] = supply_by_sector.get(sec_key, 0.0) + gen_supply
        # Priority-tier demand aggregation (loads only: cap > 0).
        if cap > 0:
            prio = obs_priority(
                obs, behavior=behavior, aid=aid, record_default_fallback_t=now
            )
            demand_by_priority[prio] = demand_by_priority.get(prio, 0.0) + abs(cap)
            served_by_priority[prio] = served_by_priority.get(prio, 0.0) + abs(sp)
            demand_by_sector_priority.setdefault(sec_key, {})
            demand_by_sector_priority[sec_key][prio] = demand_by_sector_priority[
                sec_key
            ].get(prio, 0.0) + abs(cap)
            served_by_sector_priority.setdefault(sec_key, {})
            served_by_sector_priority[sec_key][prio] = served_by_sector_priority[
                sec_key
            ].get(prio, 0.0) + abs(sp)
            # Unmet (rated cap - actual sp) captures the silent disconnect-loss
            # case the CP layer would otherwise miss.
            unmet = abs(cap) - abs(sp)
            if unmet > 1e-12:
                unmet_by_sector[sec_key] = unmet_by_sector.get(sec_key, 0.0) + unmet
        if sector != role_sector:
            continue
        total_flex += available
        total_balance += sp
        if sp > 0 and available > 0:
            total_shedded += available

    # Before the heat override on purpose: that branch REPLACES the heat entry,
    # so heat keeps its A/B-validated probe and only gas/electricity take the
    # converter credit. Crediting heat here too would be double-counted.
    _credit_cp_supply(supply_by_sector, cp_supply_aids, behavior, now)

    # Heat has no bounded slack pool, so report delivered-to-loads plus an upward
    # probe share of the gap (delivered alone ratchets DOWN toward the frontier).
    if enable_heat_l2_dispatch:
        sec_heat = Sector.HEAT.value
        delivered = sum(served_by_sector_priority.get(sec_heat, {}).values())
        demand = sum(demand_by_sector_priority.get(sec_heat, {}).values())
        supply_by_sector[sec_heat] = delivered + _HEAT_L2_PROBE_SHARE * max(
            0.0, demand - delivered
        )

    return AvailableFlexAnswer(
        flex=total_flex,
        balance=total_balance,
        shedded=total_shedded,
        sector=role_sector,
        flex_by_sector=flex_by_sector,
        balance_by_sector=balance_by_sector,
        demand_by_priority=demand_by_priority,
        served_by_priority=served_by_priority,
        unmet_by_sector=unmet_by_sector,
        demand_by_sector_priority=demand_by_sector_priority,
        served_by_sector_priority=served_by_sector_priority,
        supply_by_sector=supply_by_sector,
        round_id=round_id,
    )


class SetpointActuator:
    """Physical setpoint-write mechanism for the balance negotiator: clamp +
    floors + CLPU rate-limit, then ``apply_regulate``, and the gossip-ledger
    writeback. Owns the CLPU/last-regulate state; reads the session + role
    config through its owning role."""

    def __init__(self, role: EnergyBalanceNegotiator, ramp_per_s: float) -> None:
        self._role = role
        self._last_regulate_timestamp: float | None = None
        self._last_regulate_factor: float = 0.0
        self._clpu_ramp_per_s: float = ramp_per_s

    def grid_former_excluded(self) -> bool:
        """True iff THIS agent is a promoted island reference excluded from the
        MW gossip's curtailable set (``enable_grid_former_curtail_guard``).

        A ``GridForming*`` unit is its island's slack/voltage reference: at
        regulation=1 its ``p_mw`` is a free LP Var that absorbs the island
        residual automatically, so curtailing it (regulation<1) collapses the
        balance and the islanding solve goes infeasible. Excluded like the
        ExtGrid slack — pinned δ-box + no actuation — so the MAS balances load
        AROUND the fixed reference instead of fighting the actuator guard.
        """
        return self._role._grid_former_policy.is_former(self._role.context.aid)

    def apply_setpoint(self, new_setpoint: float) -> float | None:
        """Actuate ``new_setpoint`` (after clamp + floors); return the applied
        signed setpoint ``factor * cap``, or ``None`` when nothing is actuated
        (cap=0, tier-1 hard-lock, slack, grid-former). The caller writes the
        delta back to the gossip ledger.
        """
        obs = self._role.behavior.observe(self._role.context.aid) or {}
        cap = obs_capacity(
            obs, behavior=self._role.behavior, aid=self._role.context.aid
        )
        if cap == 0.0:
            return None
        # Tier-1 hard-lock guard: the pre-step already set tier-1 loads and the
        # QP gives them a_i = 0. Skip the write so the apply-on-first-visit path
        # doesn't drag them back to 0 off a stale ``starting_sp``.
        if int(self._role.priority) == 1:
            return None
        # Slacks have a free LP Var; writing ``regulation = sp/rating`` clamps
        # the envelope and presolves into infeasibility. The slack carries the
        # residual; gossip must not curtail it. Use a class check, not the
        # registry: the unbounded heat-side ExtHydrGrid never registers a rating.
        if _is_slack_class_child(self._role.behavior, self._role.context.aid):
            return None
        # A promoted grid-former is a fixed island reference like the slack:
        # never actuate its regulation (kept at its born 1.0 so its free p_mw
        # anchors the island). Gossip already pins its δ-box to zero, so this is
        # the actuation-side half of the same exclusion.
        if self.grid_former_excluded():
            return None

        # Constraint-aware clamp near/beyond bounds. Pass the tier so critical
        # loads (tier <= 2) get the tighter 0.99 deadband.
        if self._role.constraint_aware:
            new_setpoint = clamp_to_constraints(
                new_setpoint, obs, self._role.sector, tier=self._role.priority
            )

        factor = max(0.0, min(1.0, abs(new_setpoint / cap)))

        # Gen held down by a live curtail-lock: CLAMP to the held level (not
        # defer/None) so the writeback records the true held contribution and marks
        # it saturated, letting the dual reallocate. Deferring leaves the requested
        # delta as phantom gen supply — A/B-validated worse (loads shed against paper
        # supply).
        if cap < 0 and has_gen_curtail_lock(
            self._role.behavior,
            self._role.context.aid,
            self._role.context.current_timestamp,
        ):
            held = last_actuated_factor(self._role.behavior, self._role.context.aid)
            if held is not None and factor > held + 1e-6:
                record_event(
                    t=self._role.context.current_timestamp,
                    kind="gossip_gen_clamped_to_curtail_lock",
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    detail=f"requested_factor={factor:.4f} held={held:.4f}",
                )
                factor = held

        # No-regret floor applies only during restoration (target > 0); shedding
        # (target < 0) legitimately reduces factor.
        target = (
            self._role._sess.gossip.target
            if self._role._sess.gossip is not None
            else 0.0
        )
        is_restoration = target > 0
        if self._role.priority > 0 and is_restoration:
            self._role._listener.check_violation_cleared()

            if self._role.enable_monotonic_floor:
                if factor > self._role._sess.restoration_floor:
                    self._role._sess.restoration_floor = factor
                elif not self._role._sess.constraint_violation_active:
                    factor = self._role._sess.restoration_floor

            # CLPU rate limit: ramp-up only; decreases pass through.
            if self._role.enable_clpu_ramp:
                factor = self._rate_limit_increase(factor)

        # L2 priority-floor: the ADMM set this load's served tier, so a
        # supply-poor group must not shed below ``min(L2 alloc, constraint-
        # allowed)`` just to zero its own imbalance. Tiers 2/3/4 only.
        if self._role.enable_l2_priority_floor:
            floor = l2_effective_floor(
                self._role.behavior,
                self._role.context.aid,
                obs,
                self._role.sector,
                self._role.priority,
            )
            if floor is not None and factor < floor:
                factor = floor

        # Soft congestion-price ceiling for a generator on an overloaded export
        # branch (``enable_line_congestion_price``). Authoritative — placed AFTER
        # the floors: caps this gen's ramp at ``1 - Σ branch prices`` so gossip
        # can still serve LOCAL load up to the export-clearing level but not push
        # the line back over. Reversible: as the line clears the price decays and
        # the ceiling lifts. No lock, so nothing pins the gen at 0. Gens only.
        if cap < 0 and getattr(
            getattr(self._role.behavior, "_scare_config", None),
            "enable_line_congestion_price",
            False,
        ):
            ceiling = line_congestion_ceiling(
                self._role.behavior,
                self._role.context.aid,
                self._role.context.current_timestamp,
                _LINE_CONGESTION_TTL_S,
            )
            if ceiling < factor:
                factor = ceiling

        # Gossip regulates bypass the ``apply_regulate`` dedup on purpose: the
        # ledger advances regardless, so dedupping micro-steps would diverge it
        # from physical state and stall at k_max. Warm-start absorbs the deltas.
        if self._role.behavior.has_action(self._role.context.aid, "regulate"):
            self._role.behavior.act(self._role.context.aid, "regulate", factor)
            # Keep the dedup cache truthful: a stale cache would drop a later L2
            # re-dispatch that restores this load.
            note_actuated_factor(self._role.behavior, self._role.context.aid, factor)
            record_regulate(
                t=self._role.context.current_timestamp,
                aid=self._role.context.aid,
                sector=self._role.sector.value,
                factor=factor,
                reason="balance",
            )
            self._last_regulate_timestamp = self._role.context.current_timestamp
            self._last_regulate_factor = factor

        # Signed actuated setpoint: ``factor * cap`` is the realised power after
        # clamp + floors. The caller reconciles the ledger against it.
        return factor * cap

    def writeback_actuated_delta(
        self,
        self_key: str,
        applied_sp: float | None,
        requested_delta: float,
        counter: int,
        dmin: float,
        dmax: float,
    ) -> None:
        """Reconcile the gossip ledger with the physically-actuated setpoint.

        Records the actually-applied delta so ``_gossip_total_delta`` reflects
        real consumption. A constraint-held load is marked *saturated* so the
        dual denominator excludes it and freed supply flows to unconstrained loads.
        """
        if (
            not self._role.enable_actuated_ledger_writeback
            or applied_sp is None
            or self._role._sess.gossip is None
        ):
            return
        actuated_delta = applied_sp - self._role._sess.gossip.starting_setpoint
        # Held below the requested magnitude ⇒ constraint-bound.
        held_below = abs(actuated_delta) < abs(requested_delta) - 1e-12
        saturated = held_below or _is_saturated(actuated_delta, dmin, dmax)
        self._role._sess.gossip.memory[self_key] = (
            actuated_delta,
            counter,
            self._role.priority,
            saturated,
        )
        self._role._sess.gossip.current_delta = actuated_delta

    def _rate_limit_increase(self, requested: float) -> float:
        prev = self._last_regulate_factor
        if requested <= prev:
            return requested
        last_t = self._last_regulate_timestamp
        if last_t is None:
            return requested
        dt = max(0.0, self._role.context.current_timestamp - last_t)
        max_delta = self._clpu_ramp_per_s * dt
        return min(requested, prev + max_delta)

    def try_self_dispatch(self, deficit: float) -> None:
        """Inline local-gen fallback for isolated agents: if a generator with
        headroom, ramp up to cover as much of the deficit as possible.
        """
        if deficit <= 0:
            return
        obs = self._role.behavior.observe(self._role.context.aid) or {}
        cap = obs_capacity(
            obs, behavior=self._role.behavior, aid=self._role.context.aid
        )
        if cap >= 0:
            return  # not a generator
        # Curtail-vs-ramp interlock is enforced in ``apply_regulate``: a write
        # to a generator the auction holds down for a live violation defers there.
        sp = obs_setpoint(obs, behavior=self._role.behavior, aid=self._role.context.aid)
        headroom = abs(cap) - abs(sp)
        if headroom < 1e-6:
            return
        share = min(headroom, deficit)
        new_factor = min(1.0, (abs(sp) + share) / abs(cap))
        applied = apply_regulate(
            self._role.behavior,
            self._role.context.aid,
            new_factor,
            sector=self._role.sector.value,
            reason="self_local_gen",
            timestamp=self._role.context.current_timestamp,
        )
        if applied:
            # Out-of-band self actuation — invalidate the upward-notify baseline.
            self._role._sess.last_notified_setpoint = None
            logger.info(
                "[%s] self local-gen: ramped to %.1f%% (deficit=%.4f)",
                self._role.context.aid,
                new_factor * 100,
                deficit,
            )


class ConstraintSignalListener:
    """Receives constraint warnings/violations and computes the gossip
    participation throttle. Owns the proactive-utilization map; reads the
    session + neighbour monitor through its owning role."""

    def __init__(self, role: EnergyBalanceNegotiator) -> None:
        self._role = role
        # variable -> (utilization, timestamp recorded)
        self._proactive_util: dict[str, tuple[float, float]] = {}

    def record_warning(self, event: ConstraintWarning) -> None:
        # Record proximity-to-bound util so the gossip step can throttle. Other
        # sectors ignored (coupling handled at holon/CP level).
        if event.sector != self._role.sector:
            return
        self._proactive_util[event.variable] = (
            float(event.utilization),
            self._role.context.current_timestamp,
        )

    def _live_proactive_utils(self) -> list[float]:
        """Recorded utilizations, dropping expired ones.

        ConstraintWarning fires only above the warning threshold and nothing is
        emitted on recovery, so without an age-out the worst entry throttles
        this agent for the rest of the run.
        """
        ttl = self._role.proactive_util_ttl_s
        if ttl <= 0.0:
            return [u for u, _ in self._proactive_util.values()]
        now = self._role.context.current_timestamp
        for var, (_, t) in list(self._proactive_util.items()):
            if now - t > ttl:
                del self._proactive_util[var]
        return [u for u, _ in self._proactive_util.values()]

    def check_violation_cleared(self) -> None:
        """Clear the violation flag if the monitor reports local feasibility again."""
        if not self._role._sess.constraint_violation_active:
            return
        monitor = self.find_constraint_monitor()
        if monitor is not None and monitor.is_locally_feasible():
            self._role._sess.constraint_violation_active = False

    def worst_neighbour_utilization(self) -> float:
        """Worst utilization reported by any neighbour, via the monitor (0 if none)."""
        monitor = self.find_constraint_monitor()
        return monitor.worst_neighbour_utilization() if monitor is not None else 0.0

    def compute_participation_scale(self, obs: dict) -> float:
        """Throttle in [0, 1] blending local, worst-neighbour, and proactive util.

        Heat exempt: thermal violations want stressed loads to shed, not throttle.
        """
        if not self._role.constraint_aware or self._role.sector == Sector.HEAT:
            return 1.0
        scale = 1.0
        for var, (lo, hi) in SECTOR_CONSTRAINTS.get(self._role.sector, {}).items():
            if var in obs:
                util = constraint_utilization(float(obs[var]), lo, hi)
                scale = min(scale, max(0.0, 1.0 - util))
        neigh_util = self.worst_neighbour_utilization()
        if neigh_util > 0.0:
            scale = min(scale, max(0.0, 1.0 - neigh_util))
        live_proactive = self._live_proactive_utils()
        if live_proactive:
            scale = min(scale, max(0.0, 1.0 - max(live_proactive)))
        return scale

    def find_constraint_monitor(self):
        from scare.service.control.constraints import GridConstraintMonitor

        # Use ``get_role(cls)``; RoleContext has no ``.roles`` attribute (it
        # silently returns None). Sector-guard defensively.
        monitor = self._role.context.get_role(GridConstraintMonitor)
        if monitor is not None and monitor.sector == self._role.sector:
            return monitor
        return None


class TriggerCoordinator:
    """Setpoint-gathering trigger phase: gather the group setpoints, run the
    tier-1 hard pre-step, and hand off to the gossip engine. Operates on the
    shared session; reads router/config/gossip through its owning role."""

    def __init__(self, role: EnergyBalanceNegotiator) -> None:
        self._role = role

    async def trigger_balance_negotiation(self) -> None:
        if topology_characteristic(self._role, tid="groups") != "leader":
            return
        # MW balance deactivated for heat: frontier controller + auction own it
        # and the unbounded slack means no MW imbalance to resolve.
        if self._role.sector == Sector.HEAT:
            return
        if self._role._sess.active:
            return
        self._role._sess.active = True

        neighbours = self._role._live_neighbours()
        self._role._touch_neighbours(neighbours)
        # Snapshot group |cap| (loads only, cap > 0) so the threshold scales
        # with present demand — the right denominator for curtailment too.
        members = [self._role.context.aid] + [a.aid for a in neighbours]
        cap_sum = 0.0
        for aid in members:
            cap = obs_capacity(self._role.behavior.observe(aid) or {})
            if cap > 0:
                cap_sum += cap
        self._role._sess.group_capacity_abs = cap_sum
        logger.info(
            "[%s] balance negotiation triggered (sector=%s, group size=%d, Σ|cap_load|=%.4f)",
            self._role.context.aid,
            self._role.sector.value,
            len(neighbours) + 1,
            cap_sum,
        )
        if not neighbours:
            obs = self._role.behavior.observe(self._role.context.aid) or {}
            await self._role._start_gossip(-self._reported_setpoint(obs))
            return

        self._role._sess.neg_seq += 1
        nid = f"{self._role.context.aid}/{self._role._sess.neg_seq}"
        self._role._sess.trigger_nid = nid
        self._role._sess.trigger_responses = {}
        self._role._sess.trigger_expected = len(neighbours)

        msg = AskEnergyMessage(negotiation_id=nid, sector=self._role.sector)
        for addr in neighbours:
            await self._role.context.send_message(msg, receiver_addr=addr)

        # Deadline mirroring the gossip timeout: one dropped reply must not
        # wedge the leader ``_active=True`` forever. On expiry proceed with the
        # responses that did arrive (missing members contribute sp=0).
        base = _GOSSIP_TIMEOUT_BASE_S.get(self._role.sector, _GOSSIP_TIMEOUT_DEFAULT_S)
        timeout = base + len(neighbours) * _GOSSIP_TIMEOUT_PER_AGENT_S
        self._role.context.schedule_timestamp_task(
            self._trigger_timeout(nid),
            timestamp=self._role.context.current_timestamp + timeout,
        )

    async def _trigger_timeout(self, nid: str) -> None:
        if self._role._sess.trigger_nid != nid:
            return
        logger.warning(
            "[%s] trigger phase timed out (%d/%d responses) — proceeding",
            self._role.context.aid,
            len(self._role._sess.trigger_responses),
            self._role._sess.trigger_expected,
        )
        await self._complete_trigger_phase()

    async def _complete_trigger_phase(self) -> None:
        own_obs = self._role.behavior.observe(self._role.context.aid) or {}
        total_sp = self._reported_setpoint(own_obs) + sum(
            self._role._sess.trigger_responses.values()
        )
        responders = set(self._role._sess.trigger_responses)
        self._role._sess.trigger_nid = None
        self._role._sess.trigger_responses = {}

        # Tier-1 hard pre-step: lift tier-1 to regulation=1 if the pool
        # covers it, else pro-rata distribute and shed tiers 2/3/4.
        residual_target, skip_gossip = self._pre_apply_tier1_hard(total_sp, responders)
        if skip_gossip:
            # Residual below threshold.
            self._role._sess.active = False
            return
        await self._role._start_gossip(residual_target)

    async def handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        obs = self._role.behavior.observe(self._role.context.aid) or {}
        cap = obs_capacity(
            obs, behavior=self._role.behavior, aid=self._role.context.aid
        )
        sp = self._reported_setpoint(obs)
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=sp,
            available=cap - sp,  # headroom, not total cap
        )
        await self._role.context.send_message(
            reply, receiver_addr=mango_sender_addr(meta)
        )

    def _reported_setpoint(self, obs: dict) -> float:
        """Setpoint contribution to the group's negotiation target.

        El/gas: the raw setpoint (Σ s_i ≈ 0 ⇒ balanced). Heat: amplified by the
        local thermal deficit so a stressed group surfaces a negative target.

        F2 — slack target: with a positive ``slack_target_fraction`` a registered
        slack reports its *target infeed* ``fraction · rating`` instead of the LP
        draw, reframing the imbalance so the rest of the group balances to that
        target. Only coherent when the slack's community spans the component
        (``component_level``). At the default ``fraction == 0.0`` this branch is
        SKIPPED — reporting ``0.0`` there is an implicit "drive the draw to zero"
        target, strictly tighter than the operator budget B and fighting the
        ``SlackBudgetMonitor``; instead the slack reports its true draw and the
        monitor is the sole budget authority.
        """
        slack = lookup_slack(self._role.behavior, self._role.context.aid)
        cfg = getattr(self._role.behavior, "_scare_config", None)
        fraction = float(
            getattr(cfg, "slack_target_fraction", 0.0) if cfg is not None else 0.0
        )
        if slack is not None and fraction > 0.0:
            # ``slack.cap`` is generator-convention (negative); import target is
            # its magnitude.
            return fraction * abs(slack.cap)
        sp = obs_setpoint(obs, behavior=self._role.behavior, aid=self._role.context.aid)
        if self._role.sector == Sector.HEAT:
            sp += _heat_thermal_deficit_mw(obs)
        return sp

    async def handle_response_energy(
        self, message: ResponseEnergyMessage, meta: dict
    ) -> None:
        if message.negotiation_id != self._role._sess.trigger_nid:
            return

        sender_key = str(mango_sender_addr(meta))
        self._role._sess.trigger_responses[sender_key] = message.setpoint

        if len(self._role._sess.trigger_responses) >= self._role._sess.trigger_expected:
            await self._complete_trigger_phase()

    def _pre_apply_tier1_hard(
        self, total_sp: float, responders: set[str]
    ) -> tuple[float, bool]:
        """Tier-1 hard-constraint pre-step over the leader's group.

        Feasible (``pool >= tier1_unmet``): lift tier-1 to ``regulation = 1``,
        return residual ``(-total_sp) - tier1_unmet`` (tier-1 has ``a_i = 0``).
        Infeasible: pro-rata the pool by per-load unmet, shed tiers 2/3/4 to 0,
        return the post-apply group imbalance so gossip can still ramp
        generator headroom the pool estimate missed.

        ``responders``: sender keys (``str(addr)``) that answered this trigger
        round's AskEnergyMessage.

        Returns (residual target, skip flag — True iff residual ≤ threshold).
        """
        original_target = -float(total_sp)
        threshold = self._role._per_group_threshold()

        members = [(self._role.context.aid, True)]
        for neigh in self._role._live_neighbours():
            members.append((neigh.aid, str(neigh) in responders))

        tier1_records: list[
            tuple[str, float, float, str]
        ] = []  # (aid, cap, sp, sector)
        non_tier1_loads: list[
            tuple[str, float, float, str, int]
        ] = []  # (aid, cap, sp, sec, tier)
        pool = 0.0
        for aid, responded in members:
            obs = self._role.behavior.observe(aid) or {}
            cap = obs_capacity(obs, behavior=self._role.behavior, aid=aid)
            sec_enum = obs_sector(obs, behavior=self._role.behavior, aid=aid)
            if sec_enum is None:
                continue
            sec = sec_enum.value
            # A promoted island reference is a generator, never a load: credit
            # its supply (delivered + headroom probe) to the pool and skip the
            # load/gen bins (its free var otherwise reads as sign-flipping
            # load/ratcheting supply).
            if self._role._grid_former_policy.is_former(aid):
                pool += self._role._grid_former_policy.supply_credit(
                    aid, obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
                )
                continue
            if cap > 0:
                prio = obs_priority(obs, behavior=self._role.behavior, aid=aid)
                sp = obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
                if int(prio) == 1:
                    tier1_records.append((aid, float(cap), float(sp), sec))
                else:
                    non_tier1_loads.append((aid, float(cap), float(sp), sec, int(prio)))
            elif cap < 0:
                # Live (responded) gens credit delivered |sp| + rampable headroom,
                # else a cold start (sp≈0) freezes the pool at slack-only and zeroes
                # tiers 2-4; non-responders keep delivered-only so unreachable
                # capacity can't declare tier-1 feasible; slacks keep budgeted rating.
                if lookup_slack(self._role.behavior, aid) is not None:
                    pool += abs(float(cap))
                else:
                    sp = obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
                    if responded:
                        pool += abs(float(sp)) + max(
                            0.0, abs(float(cap)) - abs(float(sp))
                        )
                    else:
                        pool += abs(float(sp))

        tier1_unmet_per_load = [
            max(0.0, cap - sp) for (_aid, cap, sp, _sec) in tier1_records
        ]
        tier1_unmet = sum(tier1_unmet_per_load)

        # No tier-1 deficit: QP runs over the original imbalance.
        if tier1_unmet <= threshold:
            return original_target, abs(original_target) <= threshold

        if pool + threshold >= tier1_unmet:
            # Feasible: lift every tier-1 load to regulation = 1.
            now = float(self._role.context.current_timestamp)
            applied_tier1 = 0
            for aid, _cap, _sp, sec in tier1_records:
                apply_regulate(
                    self._role.behavior,
                    aid,
                    1.0,
                    sector=sec,
                    reason="tier1_hard",
                    timestamp=now,
                    priority_tier=1,
                )
                applied_tier1 += 1
            residual = original_target - tier1_unmet
            logger.info(
                "[%s] tier-1 hard pre-step (feasible): pool=%.4f "
                "tier1_unmet=%.4f applied=%d residual_target=%.4f",
                self._role.context.aid,
                pool,
                tier1_unmet,
                applied_tier1,
                residual,
            )
            return residual, abs(residual) <= threshold

        # Infeasible: pro-rata pool across tier-1 by unmet; tiers 2-4 -> 0.
        now = float(self._role.context.current_timestamp)
        applied_tier1 = 0
        applied_shed = 0
        for (aid, cap, sp, sec), unmet in zip(tier1_records, tier1_unmet_per_load):
            if unmet <= 0.0 or cap <= 0.0:
                continue
            share = pool * (unmet / tier1_unmet)
            new_sp = sp + share
            factor = max(0.0, min(1.0, new_sp / cap))
            apply_regulate(
                self._role.behavior,
                aid,
                factor,
                sector=sec,
                reason="tier1_infeasible",
                timestamp=now,
                priority_tier=1,
            )
            applied_tier1 += 1
        shed_recovered = 0.0
        for aid, cap, sp, sec, prio in non_tier1_loads:
            if cap <= 0.0:
                continue
            # Real tier passed so the L2 floor clamps this reactive shed up to
            # the component allocation (``tier1_starvation`` is in
            # ``L1_REACTIVE_SHED_REASONS``) — no cross-leader tier inversion.
            apply_regulate(
                self._role.behavior,
                aid,
                0.0,
                sector=sec,
                reason="tier1_starvation",
                timestamp=now,
                priority_tier=prio,
            )
            shed_recovered += sp
            applied_shed += 1
        # Post-apply imbalance: tier-1 lifted by ``pool``, tiers 2-4 shed
        # their ``sp``. Gossip it rather than bailing — the pool is only an
        # estimate and the gossip QP can still engage generator dmin/dmax
        # headroom it missed.
        residual = original_target - pool + shed_recovered
        logger.info(
            "[%s] tier-1 hard pre-step (INFEASIBLE): pool=%.4f "
            "tier1_unmet=%.4f tier1_loads=%d non_tier1_shed=%d "
            "residual_target=%.4f",
            self._role.context.aid,
            pool,
            tier1_unmet,
            applied_tier1,
            applied_shed,
            residual,
        )
        return residual, abs(residual) <= threshold

    # ------------------------------------------------------------------
    # Gossip phase
    # ------------------------------------------------------------------


class L2DispatchHandler:
    """L2/L3 directive handling: apply the holon's per-tier service fractions
    (dispatch-only; ramp generators / refresh the priority floor) and yield any
    in-flight L1 gossip so the authoritative L2 decision lands. Operates on the
    shared session; reads config/router/gossip through its owning role."""

    def __init__(self, role: EnergyBalanceNegotiator) -> None:
        self._role = role

    def _yield_to_l2_authority(self, route: str) -> None:
        """Abandon any in-flight L1 gossip so an arriving L2 directive can land.

        L2 carries the holon's authoritative priority decision; without the
        yield a gossip's ``_active=True`` would swallow it.
        """
        if not self._role._sess.active:
            return
        self._role._close_inflight_originator(
            "abandoned", log_reason=f"yielding to L2 ({route})"
        )
        self._role._sess.gossip = None
        self._role._sess.active = False

    async def handle_start_balance(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        if topology_characteristic(self._role, tid="groups") != "leader":
            return
        # MW balance deactivated for heat: gossip and scalar overrides never
        # run. With the heat L2 reconnect on, Route-A service fractions ARE
        # actuated (dispatch-only) — the tier-graded allocation heat otherwise
        # lacks entirely; the curtail-lock keeps temperature-feasibility
        # authority with the frontier (L2 raises defer, L2 sheds pass).
        if self._role.sector == Sector.HEAT:
            if not self._role.enable_heat_l2_dispatch:
                return
            service_frac = getattr(message, "service_fraction_by_sector_priority", None)
            if not service_frac or not self._heat_fractions_meaningful(service_frac):
                return
            if (
                self._role.enable_change_only_dispatch
                and self._service_fraction_unchanged(service_frac)
            ):
                return
            self._role.context.schedule_instant_task(
                self._dispatch_service_fractions(service_frac)
            )
            return
        # Route A (supply-priority): highest precedence; holon-global service
        # fractions applied per local-load-tier.
        service_frac = getattr(message, "service_fraction_by_sector_priority", None)
        if service_frac:
            # Unchanged allocation + own in-flight originator gossip: re-assert the
            # per-load floor but do NOT abandon the gossip — cuts the dominant
            # "yielding to L2" abandonment without staling the floor. Safe under
            # upward change-detection (any real change moves a setpoint → fresh
            # changed dispatch).
            if (
                self._role.enable_change_only_dispatch
                and self._role._sess.gossip is not None
                and self._role._sess.gossip.is_originator
                and self._service_fraction_unchanged(service_frac)
            ):
                self._refresh_l2_floor(service_frac)
                return
            self._yield_to_l2_authority("service_fraction")
            self._role._sess.active = True
            self._role.context.schedule_instant_task(
                self._dispatch_service_fractions(service_frac)
            )
            return
        # Tier-stratified override beats the scalar one: it preserves the
        # holon's priority decision the scalar would collapse.
        per_tier = getattr(message, "override_targets_by_sector_priority", None)
        if per_tier:
            # A different L2 route drifts loads away from the last service
            # fraction; invalidate so the unchanged-guard can't later match it.
            self._role._sess.last_dispatched_service_fraction = None
            self._yield_to_l2_authority("per_tier")
            self._role._sess.active = True
            self._role.context.schedule_instant_task(
                self._dispatch_per_tier_targets(per_tier)
            )
            return
        override = getattr(message, "override_target", None)
        if override is not None and math.isfinite(override):
            # L2 ADMM computed this leader's share: skip the ask-energy round
            # and use it directly as the gossip target.
            self._role._sess.last_dispatched_service_fraction = None
            self._yield_to_l2_authority("override_target")
            self._role._sess.active = True
            self._role.context.schedule_instant_task(
                self._role._start_gossip(float(override))
            )
            return
        self._role.context.schedule_instant_task(
            self._role._trigger.trigger_balance_negotiation()
        )

    _REASSERT_TOL: float = 1e-3

    async def reassert_standing_allocation(self) -> None:
        """Restore-only re-apply of the last dispatched allocation.

        The dispatch caps each load by local feasibility at write time;
        constraint release fires no event, so a load capped to ~0 stays
        latched below the standing allocation until the next L2 solve (the
        no-trigger gate may never run one — the task-17 latch). Lift such
        loads back toward ``min(allocation, constraint_allowed)``. Never
        pulls a load down; heat excluded (the frontier owns heat restores);
        every interlock still applies via ``apply_regulate``.
        """
        if topology_characteristic(self._role, tid="groups") != "leader":
            return
        sf = self._role._sess.last_dispatched_service_fraction
        if not sf:
            return
        now = self._role.context.current_timestamp
        members = [self._role.context.aid] + [
            n.aid for n in self._role._live_neighbours()
        ]
        for aid in members:
            # A promoted island reference must never enter the load path: its
            # sign-flipping free var can read as positive cap, and a reassert
            # write would seed a generator-keyed L2 floor (see the dispatch
            # loop's identical guard).
            if self._role._grid_former_policy.is_former(aid):
                continue
            obs = self._role.behavior.observe(aid) or {}
            cap = obs_capacity(obs, behavior=self._role.behavior, aid=aid)
            if cap <= 0:
                continue
            sec = obs_sector(obs, behavior=self._role.behavior, aid=aid)
            if sec is None or sec is Sector.HEAT:
                continue
            prio = obs_priority(obs, behavior=self._role.behavior, aid=aid)
            frac = sf.get(sec.value, {}).get(prio)
            if frac is None:
                continue
            target = min(
                max(0.0, min(1.0, float(frac))),
                constraint_allowed_fraction(obs, sec, tier=prio),
            )
            current = float(obs.get("regulation", 1.0))
            if target > current + self._REASSERT_TOL:
                apply_regulate(
                    self._role.behavior,
                    aid,
                    target,
                    sector=sec.value,
                    reason="l2_reassert",
                    timestamp=now,
                    priority_tier=prio,
                )

    async def _dispatch_service_fractions(
        self, service_fraction: dict[str, dict[int, float]]
    ) -> None:
        """Apply a Route-A supply-priority allocation to local agents.

        ``service_fraction[sector][tier] ∈ [0, 1]`` becomes each matching load's
        regulation factor. Generators untouched; the LP routes freed supply.
        """
        # Record what we actually dispatch so an identical later allocation can
        # take the floor-refresh-only path (no gossip preemption).
        self._role._sess.last_dispatched_service_fraction = service_fraction
        # This dispatch may move our own setpoint out-of-band, so invalidate the
        # upward-notify baseline: the next gossip finish must re-report rather
        # than be suppressed for re-converging to a now-stale notified value.
        self._role._sess.last_notified_setpoint = None
        try:
            members = [self._role.context.aid]
            for neigh in self._role._live_neighbours():
                members.append(neigh.aid)

            applied = 0
            shed_count = 0
            served_by_sector: dict[str, float] = {}
            gen_members: list[tuple[str, Sector, float, float]] = []
            for aid in members:
                obs = self._role.behavior.observe(aid) or {}
                # A promoted island reference is a generator, never a load: skip
                # it so a positive free p_mw/mass_flow can't be shed as demand.
                if self._role._grid_former_policy.is_former(aid):
                    continue
                cap = obs_capacity(obs, behavior=self._role.behavior, aid=aid)
                if cap <= 0:  # generator/slack source
                    # R3: collect dispatchable DGs (cap<0, non-slack) to ramp
                    # toward the served demand instead of shedding only. Slacks
                    # are grid-following (no regulation knob) and excluded.
                    if (
                        self._role.enable_l2_generator_ramp
                        and cap < 0
                        and lookup_slack(self._role.behavior, aid) is None
                    ):
                        gsec = obs_sector(obs, behavior=self._role.behavior, aid=aid)
                        if gsec is not None and gsec is not Sector.HEAT:
                            gsp = obs_setpoint(
                                obs, behavior=self._role.behavior, aid=aid
                            )
                            gen_members.append((aid, gsec, cap, gsp))
                    continue
                sec = obs_sector(obs, behavior=self._role.behavior, aid=aid)
                if sec is None:
                    continue
                prio = obs_priority(obs, behavior=self._role.behavior, aid=aid)
                frac = service_fraction.get(sec.value, {}).get(prio)
                if frac is None:
                    # No allocation for this (sec, tier): preserve current state.
                    continue
                factor = max(0.0, min(1.0, float(frac)))
                # El/gas: local feasibility caps the holon allocation. HEAT
                # exempt (frontier controller owns its temperature).
                if sec is not Sector.HEAT:
                    factor = min(
                        factor, constraint_allowed_fraction(obs, sec, tier=prio)
                    )
                if factor < 1.0:
                    shed_count += 1
                served_by_sector[sec.value] = (
                    served_by_sector.get(sec.value, 0.0) + factor * cap
                )
                # Count only writes ``apply_regulate`` actually landed (it
                # returns False on dedup/defer): ``applied`` gates the cascade
                # nudge below, and an all-no-op dispatch must not re-fire it.
                if apply_regulate(
                    self._role.behavior,
                    aid,
                    factor,
                    sector=sec.value,
                    reason="holon_supply_priority",
                    timestamp=self._role.context.current_timestamp,
                    priority_tier=int(prio),
                ):
                    applied += 1

            # R3: ramp dispatchable DGs toward their sector's served demand so
            # enforcement realizes the holon-assumed supply rather than shedding
            # to the un-ramped generation level.
            if self._role.enable_l2_generator_ramp and gen_members:
                applied += self._ramp_member_generators(gen_members, served_by_sector)

            if applied:
                logger.info(
                    "[%s] supply-frac dispatched: %d regulations, %d sheds, fracs=%s",
                    self._role.context.aid,
                    applied,
                    shed_count,
                    {
                        sec: {t: round(v, 3) for t, v in tm.items()}
                        for sec, tm in service_fraction.items()
                    },
                )
                # S1: close the L2->L1->L2 cascade by nudging the local
                # HolonicCommunityRole (apply_regulate emits nothing). Do NOT emit
                # NFE here — its placeholder sp mis-triggers stability and resets the
                # leader factor to 0. The ``applied`` gate keeps a no-op re-dispatch
                # from spinning the cascade.
                holon_role = self._role.context.get_role(HolonicCommunityRole)
                if holon_role is not None:
                    try:
                        holon_role.request_rebalance()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "[%s] dispatch L2 re-fire skipped: %s",
                            self._role.context.aid,
                            exc,
                        )
        finally:
            self._role._sess.active = False

    def _ramp_member_generators(
        self,
        gen_members: list[tuple[str, Sector, float, float]],
        served_by_sector: dict[str, float],
    ) -> int:
        """R3: ramp dispatchable DGs toward covering their sector's served
        demand, distributing the deficit (served demand minus current local
        generation) across DGs in proportion to headroom. Reuses the
        ``_try_self_dispatch`` ramp arithmetic; slacks are excluded by the
        caller. Uses the dedicated ``l2_gen_ramp`` reason (in
        ``GEN_RESTORE_REASONS``) so the gen curtail-ramp interlock defers it
        while the auction holds a generator down; actuator dedup applies via
        ``apply_regulate``. ``priority_tier`` stays None so no L2 floor is
        written for generators. Returns the number of generators actuated.
        """
        by_sector: dict[str, list[tuple[str, float, float]]] = {}
        for aid, sec, cap, sp in gen_members:
            by_sector.setdefault(sec.value, []).append((aid, cap, sp))

        ramped = 0
        for sec_val, gens in by_sector.items():
            served = served_by_sector.get(sec_val, 0.0)
            if served <= 0.0:
                continue
            # Gap between served load and currently-delivered local generation;
            # positive ⇒ the shortfall is being imported via the slack/grid, so
            # ramping local DGs relieves the import (and the constraints it
            # loads) instead of shedding to the un-ramped level.
            current_gen = sum(abs(sp) for _, _, sp in gens)
            deficit = served - current_gen
            if deficit <= 1e-6:
                continue
            headrooms = [(aid, cap, sp, abs(cap) - abs(sp)) for aid, cap, sp in gens]
            total_headroom = sum(h for *_, h in headrooms if h > 1e-9)
            if total_headroom <= 1e-9:
                continue
            for aid, cap, sp, headroom in headrooms:
                if headroom <= 1e-9:
                    continue
                share = min(headroom, deficit * (headroom / total_headroom))
                new_factor = min(1.0, (abs(sp) + share) / abs(cap))
                if apply_regulate(
                    self._role.behavior,
                    aid,
                    new_factor,
                    sector=sec_val,
                    reason="l2_gen_ramp",
                    timestamp=self._role.context.current_timestamp,
                ):
                    ramped += 1
        return ramped

    def _heat_fractions_meaningful(
        self, service_fraction: dict[str, dict[int, float]]
    ) -> bool:
        """Degenerate-allocation guard for the heat dispatch-only path. An
        all-zero heat allocation (the waterfall's no-supply branch, e.g. a
        transient delivered-heat readout of 0) must not zero every heat load:
        a fully dark region would stay dark — the L2 shed passes, sets no
        curtail-lock, and delivered heat (the supply estimate) never recovers.
        """
        tiers = service_fraction.get(Sector.HEAT.value)
        if not tiers:
            return False
        return any(v > 0.0 for v in tiers.values())

    def _service_fraction_unchanged(self, new: dict[str, dict[int, float]]) -> bool:
        """True when *new* matches the last dispatched service fraction within
        ``_UPWARD_NOTIFY_TOL`` (same sectors, same tiers, values within tol)."""
        prev = self._role._sess.last_dispatched_service_fraction
        if prev is None or set(prev) != set(new):
            return False
        for sec in new:
            pt, nt = prev[sec], new[sec]
            if set(pt) != set(nt):
                return False
            if any(abs(pt[t] - nt[t]) > _UPWARD_NOTIFY_TOL for t in nt):
                return False
        return True

    def _refresh_l2_floor(self, service_fraction: dict[str, dict[int, float]]) -> None:
        """Re-assert the per-load L2 priority floor from an UNCHANGED allocation
        without actuating or preempting an in-flight gossip. Mirrors the floor
        write in ``_dispatch_service_fractions``/``apply_regulate`` (cap by the
        local constraint fraction) but writes the floor store directly, so the
        in-flight gossip keeps running and honours the refreshed floor.
        """
        if not self._role.enable_l2_priority_floor:
            return
        members = [self._role.context.aid]
        for neigh in self._role._live_neighbours():
            members.append(neigh.aid)
        for aid in members:
            obs = self._role.behavior.observe(aid) or {}
            cap = obs_capacity(obs, behavior=self._role.behavior, aid=aid)
            if cap <= 0:  # loads only
                continue
            sec = obs_sector(obs, behavior=self._role.behavior, aid=aid)
            if sec is None or sec is Sector.HEAT:
                continue
            prio = obs_priority(obs, behavior=self._role.behavior, aid=aid)
            frac = service_fraction.get(sec.value, {}).get(prio)
            if frac is None:
                continue
            factor = max(0.0, min(1.0, float(frac)))
            factor = min(factor, constraint_allowed_fraction(obs, sec, tier=prio))
            set_l2_priority_floor(self._role.behavior, aid, factor)

    async def _dispatch_per_tier_targets(
        self, per_tier: dict[str, dict[int, float]]
    ) -> None:
        """Apply a tier-stratified holon allocation to local agents.

        ``per_tier[sector][tier]`` is the holon-decided change in served
        setpoint for the cell, split across members by capacity share. Bypasses
        the gossip QP (holon already solved priority globally); CLPU ramp and
        monotonic floor still apply via ``apply_regulate``.
        """
        try:
            members = [self._role.context.aid]
            for neigh in self._role._live_neighbours():
                members.append(neigh.aid)

            # Group members by (sector, tier) to split each tier's target.
            per_cell_aids: dict[tuple[str, int], list[str]] = {}
            for aid in members:
                obs = self._role.behavior.observe(aid) or {}
                cap = obs_capacity(obs, behavior=self._role.behavior, aid=aid)
                if cap <= 0:
                    continue  # generators/slacks contribute via setpoint
                prio = obs_priority(obs, behavior=self._role.behavior, aid=aid)
                sec = obs_sector(obs, behavior=self._role.behavior, aid=aid)
                if sec is None:
                    continue
                per_cell_aids.setdefault((sec.value, prio), []).append(aid)

            applied = 0
            for sec, tier_map in per_tier.items():
                for tier, tgt in tier_map.items():
                    aids = per_cell_aids.get((sec, tier), [])
                    if not aids:
                        continue
                    # ``tgt`` (cell's served-sp change) split by capacity.
                    caps = []
                    for aid in aids:
                        obs = self._role.behavior.observe(aid) or {}
                        caps.append(
                            abs(
                                obs_capacity(obs, behavior=self._role.behavior, aid=aid)
                            )
                        )
                    total_cap = sum(caps) or 1.0
                    # Positive tgt = serve more; negative = shed.
                    for aid, cap in zip(aids, caps):
                        share = cap / total_cap
                        delta_sp = tgt * share
                        obs = self._role.behavior.observe(aid) or {}
                        sp_curr = obs_setpoint(
                            obs, behavior=self._role.behavior, aid=aid
                        )
                        new_sp = sp_curr + delta_sp
                        if cap == 0.0:
                            continue
                        factor = max(0.0, min(1.0, new_sp / cap))
                        # El/gas: cap by local feasibility. HEAT exempt.
                        try:
                            _sec_enum = Sector(sec)
                        except ValueError:
                            _sec_enum = None
                        if _sec_enum is not None and _sec_enum is not Sector.HEAT:
                            factor = min(
                                factor,
                                constraint_allowed_fraction(
                                    obs, _sec_enum, tier=int(tier)
                                ),
                            )
                        apply_regulate(
                            self._role.behavior,
                            aid,
                            factor,
                            sector=sec,
                            reason="holon_tier_alloc",
                            timestamp=self._role.context.current_timestamp,
                            priority_tier=int(tier),
                        )
                        applied += 1

            if applied:
                logger.info(
                    "[%s] tier-alloc dispatched: %d regulations across %d cells",
                    self._role.context.aid,
                    applied,
                    sum(1 for tm in per_tier.values() for _ in tm),
                )
        finally:
            self._role._sess.active = False


class GossipEngine:
    """Gossip/ADMM negotiation engine: primal-dual token gossip, stall
    detection, terminal-diary bookkeeping, and finish/writeback. Operates on the
    shared session; drives the actuator/router and reads config via its role."""

    def __init__(self, role: EnergyBalanceNegotiator) -> None:
        self._role = role

    def _gossip_total_delta(self) -> float:
        """``Σ δ_i`` across the active gossip ledger (0 when no gossip active)."""
        if self._role._sess.gossip is None:
            return 0.0
        return ledger_total_delta(self._role._sess.gossip.memory)

    # ------------------------------------------------------------------
    # Neighbour liveness / heartbeat
    # ------------------------------------------------------------------

    def _update_gap_window_and_check_stall(
        self, open_gap: float, target: float
    ) -> bool:
        """P2: append the post-update gap and decide whether gossip has stalled.

        Stall when past warm-up, window full, its range is below
        ``max(_STALL_TOL_FRACTION · |T|, _STALL_TOL_FLOOR)``, and the gap still
        exceeds the per-group threshold. Warm-up covers the priority-gating
        delay plus a full post-warmup window so it doesn't fire during silence.
        """
        if self._role._sess.gossip is None:
            return False
        active = max(1, len(self._role._sess.gossip.memory))
        window_size = _STALL_WINDOW_FACTOR * active
        win = self._role._sess.gossip.gap_window
        win.append(open_gap)
        if len(win) > window_size:
            del win[0]
        # Warm-up gate: early rounds are silenced by priority/sub-round gating.
        warmup = _PRIORITY_TIERS + 1 + window_size
        if self._role._sess.gossip.counter < warmup:
            return False
        if len(win) < window_size:
            return False
        rng = max(win) - min(win)
        tol = max(_STALL_TOL_FRACTION * abs(target), _STALL_TOL_FLOOR)
        if rng > tol:
            return False
        return abs(open_gap) > self._per_group_threshold()

    async def _finish_negotiation_stalled(self) -> None:
        """P2: terminate a stalled gossip and escalate to local-gen fallback.

        Only the originator records the ``stalled`` terminal; then finishes with
        ``record_finished=False`` to avoid a double ``finished`` terminal.
        """
        if self._role._sess.gossip is None:
            return
        total_delta = self._gossip_total_delta()
        target = self._role._sess.gossip.target
        residual = target - total_delta
        logger.info(
            "[%s] gossip stalled (sector=%s, residual=%.4f, window=%d)",
            self._role.context.aid,
            self._role.sector.value,
            residual,
            len(self._role._sess.gossip.gap_window),
        )
        if self._role._sess.gossip.is_originator:
            record_negotiation(
                t=self._role.context.current_timestamp,
                aid=self._role.context.aid,
                sector=self._role.sector.value,
                nid=self._role._sess.gossip.negotiation_id,
                event="stalled",
                target=target,
                residual=residual,
                group_size=len(self._role._sess.gossip.memory),
            )
        # Suppress the "finished" entry — this terminal is "stalled".
        await self._finish_negotiation(record_finished=False)

    def _per_group_threshold(self) -> float:
        """Threshold scaled by the group's load-capacity snapshot.

        Falls back to the sector default floor before the first trigger.
        """
        if self._role._sess.group_capacity_abs <= 0.0:
            return _start_threshold(self._role.sector)
        return max(
            _THRESHOLD_ABS_FLOOR,
            _THRESHOLD_CAPACITY_FRACTION * self._role._sess.group_capacity_abs,
        )

    async def _start_gossip(self, target: float) -> None:
        # MW balance deactivated for heat; also guards the holon override_target
        # path that calls here directly.
        if self._role.sector == Sector.HEAT:
            return
        threshold = self._per_group_threshold()
        if abs(target) < threshold:
            logger.info(
                "[%s] gossip skipped: already balanced (target=%.4f, threshold=%.4f)",
                self._role.context.aid,
                target,
                threshold,
            )
            record_negotiation(
                t=self._role.context.current_timestamp,
                aid=self._role.context.aid,
                sector=self._role.sector.value,
                nid="",
                event="skipped_balanced",
                target=target,
            )
            self._role._sess.active = False
            return

        # An overlapping trigger can reach here with a live originator gossip;
        # retire it as ``abandoned`` first so its diary terminal isn't dropped.
        self._close_inflight_originator(
            "abandoned", log_reason="superseded by new gossip"
        )

        # Reset the violation flag so the monotonic floor only yields while a
        # violation is actively present.
        self._role._sess.constraint_violation_active = False

        # Gossip-only neighbours: excludes members without a negotiator (branch
        # monitors) that would drop the token.
        neighbours = self._role._gossip_neighbours()
        self._role._touch_neighbours(neighbours)
        self._role._sess.neg_seq += 1
        nid = f"{self._role.context.aid}/{self._role._sess.neg_seq}"
        self_key = str(self._role.context.addr)

        obs = self._role.behavior.observe(self._role.context.aid) or {}
        starting_sp = obs_setpoint(
            obs, behavior=self._role.behavior, aid=self._role.context.aid
        )
        # Anchor the QP δ-box to the starting state (see _GossipState);
        # per-step recompute causes a self-driven sign-flip oscillation.
        dmin_start, dmax_start = obs_min_max(
            obs, behavior=self._role.behavior, aid=self._role.context.aid
        )
        if self._role._actuator.grid_former_excluded():
            # Pin the island reference: a zero δ-box makes the QP compute δ=0
            # (marked saturated → dropped from the dual denominator), so gossip
            # never curtails the former and the MAS balances load AROUND it.
            dmin_start = dmax_start = 0.0

        lambda_seed = compute_lambda_seed(
            target,
            len(neighbours),
            priority=self._role.priority,
            priority_tiers=_PRIORITY_TIERS,
        )

        self._role._sess.gossip = _GossipState(
            negotiation_id=nid,
            target=target,
            counter=0,
            current_delta=0.0,
            starting_setpoint=starting_sp,
            dmin_starting=dmin_start,
            dmax_starting=dmax_start,
            memory={self_key: (0.0, 0, self._role.priority, False)},
            is_originator=True,
            dual_lambda=lambda_seed,
        )

        if not neighbours:
            # Isolated agent: approve the fallback directly with the full
            # deficit (activates local DGs) and self-dispatch inline if that
            # role is absent.
            logger.info(
                "[%s] gossip skipped: singleton (target=%.4f) — escalating to local-gen fallback",
                self._role.context.aid,
                target,
            )
            record_negotiation(
                t=self._role.context.current_timestamp,
                aid=self._role.context.aid,
                sector=self._role.sector.value,
                nid="",
                event="skipped_singleton",
                target=target,
                group_size=1,
            )
            if abs(target) > threshold:
                record_event(
                    t=self._role.context.current_timestamp,
                    kind="local_gen_request",
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    detail=f"residual={target:.4f} (singleton)",
                )
                self._role.context.emit_event(
                    LocalGenerationApproval(
                        sector=self._role.sector, residual_deficit=target
                    )
                )
                self._role._actuator.try_self_dispatch(target)
            self._role._sess.active = False
            self._role._sess.gossip = None
            return

        # Committed to a multi-party gossip; record the start.
        group_size = len(neighbours) + 1
        logger.info(
            "[%s] starting gossip (sector=%s, target=%.4f)",
            self._role.context.aid,
            self._role.sector.value,
            target,
        )
        record_negotiation(
            t=self._role.context.current_timestamp,
            aid=self._role.context.aid,
            sector=self._role.sector.value,
            nid=nid,
            event="started",
            target=target,
            group_size=group_size,
        )

        msg = EnergyNegotiationMessage(
            negotiation_id=nid,
            sector=self._role.sector,
            negotiation_target=target,
            current_delta=0.0,
            counter=0,
            memory=dict(self._role._sess.gossip.memory),
            dual_lambda=self._role._sess.gossip.dual_lambda,
        )
        if self._role.enable_qp_gossip:
            # Single-token: the dual λ can't be averaged across parallel tokens
            # with divergent ledger views. Forward to one K-weighted next-hop.
            next_addr = self._role._next_hop(neighbours, nid, 0)
            await self._role.context.send_message(msg, receiver_addr=next_addr)
        else:
            # Equal-share tolerates multi-token broadcast: the ledger merge
            # composes correctly.
            for addr in neighbours:
                await self._role.context.send_message(msg, receiver_addr=addr)

        # Wallclock timeout: force-finish if not converged. Per-sector base +
        # per-agent scaling.
        base = _GOSSIP_TIMEOUT_BASE_S.get(self._role.sector, _GOSSIP_TIMEOUT_DEFAULT_S)
        timeout = base + len(neighbours) * _GOSSIP_TIMEOUT_PER_AGENT_S
        deadline = self._role.context.current_timestamp + timeout
        self._role.context.schedule_timestamp_task(
            self._gossip_timeout(nid), timestamp=deadline
        )

    async def _gossip_timeout(self, negotiation_id: str) -> None:
        if (
            self._role._sess.gossip is not None
            and self._role._sess.gossip.negotiation_id == negotiation_id
        ):
            logger.warning(
                "[%s] gossip %s timed out — forcing finish",
                self._role.context.aid,
                negotiation_id[:8],
            )
            if self._role._sess.gossip.is_originator:
                total_delta = self._gossip_total_delta()
                residual = self._role._sess.gossip.target - total_delta
                record_negotiation(
                    t=self._role.context.current_timestamp,
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    nid=negotiation_id,
                    event="timed_out",
                    target=self._role._sess.gossip.target,
                    residual=residual,
                    group_size=len(self._role._sess.gossip.memory),
                )
            await self._finish_negotiation(record_finished=False)

    async def _handle_negotiation_message(
        self, message: EnergyNegotiationMessage, meta: dict
    ) -> None:
        nid = message.negotiation_id
        counter = message.counter + 1

        if counter > self._role.max_hops + 1:
            return

        self_key = str(self._role.context.addr)

        if (
            self._role._sess.gossip is None
            or self._role._sess.gossip.negotiation_id != nid
        ):
            # A different nid arriving over our in-flight gossip: record an
            # ``abandoned`` terminal for the old nid before overwriting state
            # (preserves started == Σ terminals).
            if (
                self._role._sess.gossip is not None
                and self._role._sess.gossip.is_originator
                and self._role._sess.gossip.negotiation_id != nid
            ):
                prev_total = self._gossip_total_delta()
                record_negotiation(
                    t=self._role.context.current_timestamp,
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    nid=self._role._sess.gossip.negotiation_id,
                    event="abandoned",
                    target=self._role._sess.gossip.target,
                    residual=self._role._sess.gossip.target - prev_total,
                    group_size=len(self._role._sess.gossip.memory),
                )
            obs = self._role.behavior.observe(self._role.context.aid) or {}
            init_dmin, init_dmax = obs_min_max(
                obs, behavior=self._role.behavior, aid=self._role.context.aid
            )
            if self._role._actuator.grid_former_excluded():
                init_dmin = init_dmax = 0.0  # pin the reference (see _start_gossip)
            # Adopt via ledger_merge so the incoming ledger gets the same
            # Byzantine clip as the merge path below.
            cap_byz = _BYZANTINE_DELTA_CAP_MULTIPLE * max(
                abs(message.negotiation_target), 1.0
            )
            adopted_memory: dict[str, tuple[float, int, int, bool]] = {}
            ledger_merge(adopted_memory, message.memory, byzantine_cap=cap_byz)
            self._role._sess.gossip = _GossipState(
                negotiation_id=nid,
                target=message.negotiation_target,
                counter=counter,
                current_delta=0.0,
                starting_setpoint=obs_setpoint(
                    obs, behavior=self._role.behavior, aid=self._role.context.aid
                ),
                dmin_starting=init_dmin,
                dmax_starting=init_dmax,
                memory=adopted_memory,
                dual_lambda=getattr(message, "dual_lambda", 0.0),
            )
        else:
            self._role._sess.gossip.counter = counter
            # P6: λ travels with the message; adopt the latest (safe under
            # single-token gossip).
            self._role._sess.gossip.dual_lambda = getattr(
                message, "dual_lambda", self._role._sess.gossip.dual_lambda
            )
            # Merge ledger keeping newest-counter entries; Byzantine-clip each
            # delta to a multiple of |target|.
            cap_byz = _BYZANTINE_DELTA_CAP_MULTIPLE * max(
                abs(self._role._sess.gossip.target), 1.0
            )
            ledger_merge(
                self._role._sess.gossip.memory, message.memory, byzantine_cap=cap_byz
            )

        target = self._role._sess.gossip.target
        obs = self._role.behavior.observe(self._role.context.aid) or {}
        cap = obs_capacity(
            obs, behavior=self._role.behavior, aid=self._role.context.aid
        )
        # Gossip-anchored δ-box, not a fresh ``obs_min_max`` (which flips after
        # the agent's own regulate and drives a bang-bang oscillation).
        dmin = self._role._sess.gossip.dmin_starting
        dmax = self._role._sess.gossip.dmax_starting

        prev_own = self._role._sess.gossip.memory.get(
            self_key, (0.0, 0, self._role.priority, False)
        )[0]
        total_delta = self._gossip_total_delta()
        open_gap = target - total_delta

        participation_scale = self._role._listener.compute_participation_scale(obs)

        active_count = max(1, len(self._role._sess.gossip.memory))
        n_free = max(
            1, sum(1 for v in self._role._sess.gossip.memory.values() if not v[3])
        )

        if self._role.enable_qp_gossip:
            # P6: primal-dual QP closed-form update. δ_i = clamp(a_i · λ, dmin,
            # dmax), a_i = priority weight, sign(λ) = sign(T). A continuous
            # priority waterfall; participation_scale folds into a_i.
            target_sign = 1 if target > 0 else (-1 if target < 0 else 0)
            a_i_base = qp_priority_weight(
                self._role.priority,
                target_sign,
                priority_tiers=_PRIORITY_TIERS,
            )
            a_i = a_i_base * self._role.impact_weight * participation_scale
            new_delta = qp_primal(a_i, self._role._sess.gossip.dual_lambda, dmin, dmax)
            saturated = _is_saturated(new_delta, dmin, dmax)
            self._role._sess.gossip.memory[self_key] = (
                new_delta,
                counter,
                self._role.priority,
                saturated,
            )
            self._role._sess.gossip.current_delta = new_delta
            # Dedup gate (QP only): δ is monotonic in λ, so once saturated
            # further visits request the same δ — skip the actuator write to
            # avoid quadratic re-solves. First visit always applies.
            delta_step = abs(new_delta - prev_own)
            apply_threshold = 1e-4 * max(abs(cap), 1.0)
            if cap != 0.0 and (delta_step > apply_threshold or prev_own == 0.0):
                applied_sp = self._role._actuator.apply_setpoint(
                    self._role._sess.gossip.starting_setpoint + new_delta
                )
                # Write back the physically-actuated delta (not the requested):
                # a clamped load shows its true contribution, so the dual raises
                # λ and unconstrained loads absorb the freed supply.
                self._role._actuator.writeback_actuated_delta(
                    self_key,
                    applied_sp,
                    new_delta,
                    counter,
                    dmin,
                    dmax,
                )

            # Dual update: λ ← λ + γ_k · (T − Σ δ) / Σ a_a. The Σ a_a norm makes
            # λ converge in ~one step when nothing is clamped (λ* = T / Σ a_j);
            # γ_k (Robbins-Monro) damps box-noise once saturated.
            total_delta_post = self._gossip_total_delta()
            residual = target - total_delta_post
            # Normalise over *unsaturated* entries only: saturated agents add no
            # δ for further λ, so counting them slows the agents that can move.
            sum_a_est = ledger_sum_responsiveness(
                self._role._sess.gossip.memory,
                target_sign,
                priority_tiers=_PRIORITY_TIERS,
            )
            self._role._sess.gossip.dual_lambda += (
                step_size(
                    self._role.convergence_rate,
                    counter,
                    step_decay_k0=self._role.step_decay_k0,
                )
                * residual
                / sum_a_est
            )
        else:
            # Equal-share step: each active participant aims for 1/n_free of the
            # open gap, scaled by step + participation. Priority/sub-round gated.
            own_change = (
                (open_gap / n_free)
                * self._role.impact_weight
                * step_size(
                    self._role.convergence_rate,
                    counter,
                    step_decay_k0=self._role.step_decay_k0,
                )
                * participation_scale
            )

            actual_prio = _compute_actual_priority(self._role.priority, target)

            if actual_prio <= counter:
                tier_size = max(1, active_count // max(1, _PRIORITY_TIERS))
                if tier_size > 1:
                    sub_idx = _deterministic_sub_round(
                        self_key, nid, actual_prio, tier_size
                    )
                    rounds_in_tier = counter - actual_prio
                    if rounds_in_tier % tier_size != sub_idx:
                        own_change = 0.0

                current_own = prev_own
                new_delta = max(dmin, min(dmax, current_own + own_change))
                saturated = _is_saturated(new_delta, dmin, dmax)
                self._role._sess.gossip.memory[self_key] = (
                    new_delta,
                    counter,
                    self._role.priority,
                    saturated,
                )
                self._role._sess.gossip.current_delta = new_delta
                if cap != 0.0:
                    applied_sp = self._role._actuator.apply_setpoint(
                        self._role._sess.gossip.starting_setpoint + new_delta
                    )
                    self._role._actuator.writeback_actuated_delta(
                        self_key,
                        applied_sp,
                        new_delta,
                        counter,
                        dmin,
                        dmax,
                    )

        # Recompute total after own update.
        total_delta = self._gossip_total_delta()
        open_gap = target - total_delta

        # P2: stall detection — saturated without converging ⇒ escalate now
        # rather than spinning to k_max.
        stalled = self._update_gap_window_and_check_stall(open_gap, target)

        # Next-hop over gossip-capable peers only (branch monitors drop the
        # token and let the gossip time out).
        neighbours = self._role._gossip_neighbours()

        if stalled:
            await self._finish_negotiation_stalled()
            return

        if (
            abs(open_gap) <= self._role.termination_tolerance
            or counter >= self._role.max_hops
        ):
            await self._finish_negotiation()
        elif neighbours:
            next_addr = self._role._next_hop(neighbours, nid, counter)
            fwd = EnergyNegotiationMessage(
                negotiation_id=nid,
                sector=self._role.sector,
                negotiation_target=target,
                current_delta=self._role._sess.gossip.current_delta,
                counter=counter,
                memory=dict(self._role._sess.gossip.memory),
                dual_lambda=self._role._sess.gossip.dual_lambda,
            )
            await self._role.context.send_message(fwd, receiver_addr=next_addr)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    async def _finish_negotiation(self, *, record_finished: bool = True) -> None:
        starting_sp = (
            self._role._sess.gossip.starting_setpoint
            if self._role._sess.gossip
            else obs_setpoint(self._role.behavior.observe(self._role.context.aid) or {})
        )
        delta = (
            self._role._sess.gossip.current_delta if self._role._sess.gossip else 0.0
        )
        new_sp = starting_sp + delta
        # Coordination overhaul: only propagate this finish UPWARD (to the holon
        # ADMM at L2 and CP at L3, plus the local-leader L2 self-trigger) when
        # the converged setpoint actually moved since the last notification — a
        # gossip that re-converges to the same value must not re-trigger the
        # cascade (this is what lets the time-throttle be removed).
        prev_notified = self._role._sess.last_notified_setpoint
        notify_upward = (
            not self._role.enable_change_only_dispatch
            or prev_notified is None
            or abs(new_sp - prev_notified) > _UPWARD_NOTIFY_TOL
        )
        if notify_upward:
            self._role._sess.last_notified_setpoint = new_sp

        if self._role._sess.gossip is not None:
            total_delta = self._gossip_total_delta()
            target = self._role._sess.gossip.target
            residual = target - total_delta
            logger.info(
                "[%s] gossip finished (sector=%s, delta=%.4f, residual=%.4f)",
                self._role.context.aid,
                self._role.sector.value,
                delta,
                residual,
            )
            # ``record_finished=False`` when the caller recorded a more specific
            # terminal. Only the originator records (peers never recorded a
            # "started", so their terminal would inflate the invariant).
            if record_finished and self._role._sess.gossip.is_originator:
                record_negotiation(
                    t=self._role.context.current_timestamp,
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    nid=self._role._sess.gossip.negotiation_id,
                    event="finished",
                    target=target,
                    residual=residual,
                    group_size=len(self._role._sess.gossip.memory),
                )

            # Unresolved deficit escalates to local-gen fallback via L2 first
            # (holon absorbs cross-group before L1 falls back to local DGs).
            # Leader-only; members surface residual via the NFE broadcast.
            if (
                abs(residual) > self._per_group_threshold() * 10
                and topology_characteristic(self._role, tid="groups") == "leader"
            ):
                record_event(
                    t=self._role.context.current_timestamp,
                    kind="local_gen_request",
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    detail=f"residual={residual:.4f}",
                )
                # Component scope: the leader's own component ADMM was already
                # re-triggered by the local NegotiationFinishedEvent above (the
                # coordinator re-solves and floors what it can route); approve
                # the local-DG fallback for the residual it cannot. Legacy holon
                # scope: escalate to holon-clique peers so their L2 arbitrates
                # first, else self-approve when the clique is empty.
                if self._role._component_scope:
                    self._role.context.emit_event(
                        LocalGenerationApproval(
                            sector=self._role.sector,
                            residual_deficit=residual,
                        )
                    )
                else:
                    request = LocalGenerationRequest(
                        sector=self._role.sector, residual_deficit=residual
                    )
                    try:
                        holon_peers = list(topology_neighbors(self._role, tid="holons"))
                    except KeyError:
                        holon_peers = []
                    if holon_peers:
                        for addr in holon_peers:
                            await self._role.context.send_message(
                                request, receiver_addr=addr
                            )
                    else:
                        self._role.context.emit_event(
                            LocalGenerationApproval(
                                sector=self._role.sector,
                                residual_deficit=residual,
                            )
                        )

        # Local event: consumed by this agent's own L2 (_on_member_finished_local)
        # and L3 (CP channel) — the upward self-trigger. Gate on change.
        if notify_upward:
            self._role.context.emit_event(
                NegotiationFinishedEvent(new_setpoint=new_sp, sector=self._role.sector)
            )

        # Broadcast convergence to gossip-capable neighbours so each emits its
        # own local event. ``new_setpoint`` carries the leader's converged sp so
        # the CP fixed-point gate can detect movement (a hard zero would suppress
        # CP re-triggers); neighbours re-derive their own sp. ``negotiation_id``
        # lets members with matching gossip state (incl. a blocked originator)
        # release it instead of timing out.
        finished_nid = (
            self._role._sess.gossip.negotiation_id if self._role._sess.gossip else ""
        )
        neighbours = self._role._gossip_neighbours()
        finished_msg = NegotiationFinishedEvent(
            new_setpoint=new_sp, sector=self._role.sector, negotiation_id=finished_nid
        )
        for addr in neighbours:
            await self._role.context.send_message(finished_msg, receiver_addr=addr)

        # Upward L1→L2 / L1→L3 reactive triggers: notify holon peers (so the
        # holon ADMM redistributes residual) and CP connectors. Gated on change
        # so a no-op finish does not re-trigger the cascade. The priority-aware
        # payload is re-fetched in ``_try_rebalance``.
        if notify_upward:
            # Component scope drives L2 off each leader's own local finish
            # (emitted above) plus the coordinator's per-leader report buffer, so
            # the cross-leader holon-clique nudge is redundant there (and the
            # clique is an arbitrary lex chunk, not the coordination set). Legacy
            # holon scope still fans the finish out to clique peers.
            if not self._role._component_scope:
                try:
                    holon_peers = topology_neighbors(self._role, tid="holons")
                except KeyError:
                    holon_peers = []
                for addr in holon_peers:
                    await self._role.context.send_message(
                        finished_msg, receiver_addr=addr
                    )

            # Leader also notifies CP connectors (both scopes)
            if topology_characteristic(self._role, tid="groups") == "leader":
                cp_connectors = list(topology_connectors(self._role, tid="groups"))
                if cp_connectors:
                    logger.info(
                        "[%s] gossip finished: notifying %d CP connectors (new_sp=%.4f)",
                        self._role.context.aid,
                        len(cp_connectors),
                        new_sp,
                    )
                for addr in cp_connectors:
                    await self._role.context.send_message(
                        finished_msg, receiver_addr=addr
                    )

        self._role._sess.gossip = None
        # ``_active`` is owned by the trigger phase while ``_trigger_nid`` is
        # set (gossip adoption never sets it); a finishing adopted gossip must
        # not release the trigger's re-entry guard. The trigger's own gossip
        # always runs with ``_trigger_nid is None`` (cleared in
        # ``_complete_trigger_phase``), so clearing is correct then.
        if self._role._sess.trigger_nid is None:
            self._role._sess.active = False

    def flush_pending(self) -> None:
        """Record still-active gossip as ``stalled`` (progress) or ``abandoned``.

        Called at world teardown so an in-flight negotiation still counts toward
        the ``started == Σ terminals`` invariant.
        """
        if self._role._sess.gossip is None:
            return
        if self._role._sess.gossip.is_originator:
            total_delta = self._gossip_total_delta()
            target = self._role._sess.gossip.target
            residual = target - total_delta
            # ``stalled`` (closed >= 30% of |target|) is a soft terminal.
            if abs(target) > 1e-12:
                progress = (abs(target) - abs(residual)) / abs(target)
            else:
                progress = 1.0
            event = "stalled" if progress >= 0.3 else "abandoned"
            record_negotiation(
                t=self._role.context.current_timestamp,
                aid=self._role.context.aid,
                sector=self._role.sector.value,
                nid=self._role._sess.gossip.negotiation_id,
                event=event,
                target=target,
                residual=residual,
                group_size=len(self._role._sess.gossip.memory),
            )
        self._role._sess.gossip = None
        self._role._sess.active = False

    async def _handle_negotiation_finished_msg(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        """Convergence broadcast from a peer; release matching gossip state
        and emit own local NFE.

        A token-holder that detects convergence finishes only its own state;
        without the nid-matched release here every other member (incl. the
        originator) stayed blocked until the wallclock timeout and the
        originator logged a spurious ``timed_out`` terminal.
        """
        starting_sp = (
            self._role._sess.gossip.starting_setpoint
            if self._role._sess.gossip
            else obs_setpoint(self._role.behavior.observe(self._role.context.aid) or {})
        )
        delta = (
            self._role._sess.gossip.current_delta if self._role._sess.gossip else 0.0
        )
        nid = getattr(message, "negotiation_id", "")
        if (
            nid
            and self._role._sess.gossip is not None
            and self._role._sess.gossip.negotiation_id == nid
        ):
            if self._role._sess.gossip.is_originator:
                total_delta = self._gossip_total_delta()
                record_negotiation(
                    t=self._role.context.current_timestamp,
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    nid=nid,
                    event="finished",
                    target=self._role._sess.gossip.target,
                    residual=self._role._sess.gossip.target - total_delta,
                    group_size=len(self._role._sess.gossip.memory),
                )
            self._role._sess.gossip = None
            # Same ownership rule as ``_finish_negotiation``: only the trigger
            # phase may hold ``_active`` while ``_trigger_nid`` is set.
            if self._role._sess.trigger_nid is None:
                self._role._sess.active = False
        self._role.context.emit_event(
            NegotiationFinishedEvent(
                new_setpoint=starting_sp + delta, sector=self._role.sector
            )
        )

    # ------------------------------------------------------------------
    # Flex reporting
    # ------------------------------------------------------------------

    def _close_inflight_originator(
        self, event: str, log_reason: str | None = None
    ) -> None:
        """Record a terminal for in-flight gossip this agent originated.

        Preserves ``started == Σ terminals`` before teardown. No-op for relays.
        Shared by the four sites that retire an active gossip.
        """
        if self._role._sess.gossip is None or not self._role._sess.gossip.is_originator:
            return
        total_delta = self._gossip_total_delta()
        record_negotiation(
            t=self._role.context.current_timestamp,
            aid=self._role.context.aid,
            sector=self._role.sector.value,
            nid=self._role._sess.gossip.negotiation_id,
            event=event,
            target=self._role._sess.gossip.target,
            residual=self._role._sess.gossip.target - total_delta,
            group_size=len(self._role._sess.gossip.memory),
        )
        if log_reason is not None:
            logger.info(
                "[%s] gossip %s %s — %s",
                self._role.context.aid,
                self._role._sess.gossip.negotiation_id[:8],
                event,
                log_reason,
            )


class EnergyBalanceNegotiator(Role):
    """Gossip-based energy balance negotiation.

    Features: priority-ordered participation (high-priority loads restore
    first / shed last), monotonic progress during restoration (no-regret
    switching unless a violation forces a drop), deterministic hash-based
    next-hop, constraint-aware setpoint clamping, and sector-timescale-derived
    convergence rate.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        priority: int = 0,
        convergence_rate: float | None = None,
        impact_weight: float = 1.0,
        termination_tolerance: float = 1e-5,
        constraint_aware: bool = True,
        proactive_util_ttl_s: float = 0.0,
        enable_monotonic_floor: bool = True,
        enable_clpu_ramp: bool = True,
        max_hops: int = _MAX_HOPS,
        step_decay_k0: int = _STEP_DECAY_K0_DEFAULT,
        enable_qp_gossip: bool = True,
        enable_l2_generator_ramp: bool = True,
        enable_change_only_dispatch: bool = True,
        enable_l2_priority_floor: bool = True,
        enable_actuated_ledger_writeback: bool = True,
        enable_heat_l2_dispatch: bool = False,
        enable_gen_capacity_supply: bool = False,
        enable_cp_supply_credit: bool = False,
        enable_l2_allocation_reassert: bool = False,
        l2_allocation_reassert_s: float = 2.0,
        component_scope: bool = False,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self._grid_former_policy = GridFormerPolicy(
            behavior, probe_share=_GRID_FORMER_SUPPLY_PROBE_SHARE
        )
        self.sector = sector
        # True when L2 runs per connected component (holon cliques are not
        # built). Upward reactive triggers then go to the leader's own L2 and
        # the component-peer mesh, not the arbitrary ``holons`` chunk topology.
        self._component_scope = bool(component_scope)
        # Monotonic counter for DETERMINISTIC negotiation ids feeding hash-based
        # routing; a uuid4 id (os.urandom, immune to seeding) made routing vary
        # run-to-run; aid/seq is unique + reproducible. See project_restoration_sim_nonreproducible.
        self._sess = NegotiationSession()
        self.priority = priority
        self.impact_weight = impact_weight
        self.termination_tolerance = termination_tolerance
        self.constraint_aware = constraint_aware
        self.proactive_util_ttl_s = proactive_util_ttl_s
        self.enable_monotonic_floor = enable_monotonic_floor
        self.enable_clpu_ramp = enable_clpu_ramp
        # L2 priority-floor: clamp gossip sheds up to the component ADMM's
        # per-load allocation, blocking the L2->L1 tier inversion.
        self.enable_l2_priority_floor = enable_l2_priority_floor
        # Write the physically-actuated (clamped/floored) delta back into the
        # ledger so the dual sees a constrained load's true contribution and
        # reallocates freed supply to unconstrained loads.
        self.enable_actuated_ledger_writeback = enable_actuated_ledger_writeback
        self.max_hops = max_hops
        self.step_decay_k0 = max(1, int(step_decay_k0))
        # Ablation flag: True = primal-dual QP gossip, else equal-share update.
        # Only the per-agent update rule differs.
        self.enable_qp_gossip = enable_qp_gossip
        # R3: ramp dispatchable DGs in the L2 service-fraction dispatch instead
        # of shed-only enforcement (see config.enable_l2_generator_ramp).
        self.enable_l2_generator_ramp = enable_l2_generator_ramp
        # Coordination overhaul: gate the upward L1→L2/L3 notifications on the
        # converged setpoint actually moving, so a gossip that re-converges to
        # the same result does not re-trigger the holon ADMM — the cascade
        # self-terminates at a fixed point (see config.enable_change_only_dispatch).
        self.enable_change_only_dispatch = bool(enable_change_only_dispatch)
        # Heat L2 reconnect (opt-in): actuate holon service fractions for heat
        # (dispatch-only; gossip stays heat-excluded) and report delivered heat
        # as the sector's flex supply pool.
        self.enable_heat_l2_dispatch = bool(enable_heat_l2_dispatch)
        self.enable_gen_capacity_supply = bool(enable_gen_capacity_supply)
        self.enable_cp_supply_credit = bool(enable_cp_supply_credit)
        self.enable_l2_allocation_reassert = bool(enable_l2_allocation_reassert)
        self.l2_allocation_reassert_s = float(l2_allocation_reassert_s)

        # Sector-specific convergence rate unless overridden.
        ts = SECTOR_TIMESCALE.get(sector, {})
        self.convergence_rate = (
            convergence_rate
            if convergence_rate is not None
            else ts.get("convergence_rate", 0.5)
        )

        # CLPU rate limiter (post-outage inrush is 2-6x steady state) lives in
        # the actuator; ramp scales with convergence_rate.
        self._actuator = SetpointActuator(self, self.convergence_rate)

        # ``_heartbeat_max_age_s`` is leftover write-only state (no aging-out).
        poll = ts.get("poll_period_s", 1.0)
        self._heartbeat_max_age_s: float = poll * _HEARTBEAT_MAX_AGE_MULTIPLE

        # B.1: continuous coupling weights K_ij(t) in [0, 1] biasing forwarding
        # and gating liveness (owned by the router). Decay scales with poll period.
        self._router = NeighbourRouter(
            TrustLedger(
                TrustParams(
                    decay_rate_per_s=1.0 / max(poll * _HEARTBEAT_MAX_AGE_MULTIPLE, 1.0),
                    recover_rate=0.6,
                    liveness_threshold=0.5,
                    initial=1.0,
                )
            )
        )

        self._listener = ConstraintSignalListener(self)
        self._trigger = TriggerCoordinator(self)
        self._dispatch = L2DispatchHandler(self)
        self._engine = GossipEngine(self)

    def setup(self) -> None:
        # Register this aid as gossip-capable so peers route
        # EnergyNegotiationMessage only to agents that process it. Branch
        # monitors share the el groups but have no negotiator and would drop
        # the token.
        store = getattr(self.behavior, "_scare_gossip_capable", None)
        if store is None:
            store = {}
            self.behavior._scare_gossip_capable = store
        store.setdefault(self.sector, set()).add(self.context.aid)

        # R2 drift fix: restore-only re-assert of the standing L2 allocation
        # (constraint release fires no event — see the task-17 latch).
        if self.enable_l2_allocation_reassert:
            self.context.schedule_periodic_task(
                self._dispatch.reassert_standing_allocation,
                delay=self.l2_allocation_reassert_s,
            )

        # Mango dispatches synchronously; wrap async handlers to self-schedule
        # (tracked by termination detection) and stamp the sender's heartbeat.
        _wrap = async_dispatch(self, on_receive=self._record_sender)

        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_energy),
            lambda msg, meta: (
                isinstance(msg, AskEnergyMessage) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_response_energy),
            lambda msg, meta: isinstance(msg, ResponseEnergyMessage),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_message),
            lambda msg, meta: (
                isinstance(msg, EnergyNegotiationMessage) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_finished_msg),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_flex),
            lambda msg, meta: isinstance(msg, AskForAvailableFlex),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_start_balance),
            lambda msg, meta: isinstance(msg, StartBalanceNegotiation),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_failure_notice),
            lambda msg, meta: isinstance(msg, FailureNotice),
        )
        self.context.subscribe_event(self, BalanceProblem, self._on_balance_problem)
        self.context.subscribe_event(
            self, ConstraintViolation, self._on_constraint_violation
        )
        self.context.subscribe_event(
            self, ConstraintWarning, self._on_constraint_warning
        )
        # Mango needs >=1 local subscriber per emitted event type. A non-leader
        # hitting the singleton-fallback path would crash emit_event without
        # this no-op (real fallback logic is leader-only).
        self.context.subscribe_event(
            self, LocalGenerationApproval, self._on_local_gen_approval_noop
        )
        # Same no-op for NegotiationFinishedEvent: in minimal compositions the
        # negotiator may be alone with no GenerationController subscriber.
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_finished_noop
        )

    # ------------------------------------------------------------------
    # Constraint violation tracking (for monotonic progress override)
    # ------------------------------------------------------------------

    def _on_local_gen_approval_noop(
        self, _event: LocalGenerationApproval, _src: Any
    ) -> None:
        # No-op; the leader handles the response. See setup().
        return

    def _on_finished_noop(self, _event: NegotiationFinishedEvent, _src: Any) -> None:
        # No-op; keeps dispatch safe in minimal compositions. See setup().
        return

    def _on_constraint_warning(self, event: ConstraintWarning, _src: Any) -> None:
        self._listener.record_warning(event)

    def _on_constraint_violation(self, event: ConstraintViolation, _src: Any) -> None:
        if event.sector == self.sector:
            self._sess.constraint_violation_active = True
            # Cancel active gossip: the constraint landscape changed, so a stale
            # target may push deeper into violation. A fresh round retriggers.
            if self._sess.gossip is not None:
                logger.info(
                    "[%s] cancelling gossip %s due to %s violation",
                    self.context.aid,
                    self._sess.gossip.negotiation_id[:8],
                    event.variable,
                )
                if self._sess.gossip.is_originator:
                    total_delta = self._gossip_total_delta()
                    record_negotiation(
                        t=self.context.current_timestamp,
                        aid=self.context.aid,
                        sector=self.sector.value,
                        nid=self._sess.gossip.negotiation_id,
                        event="cancelled",
                        target=self._sess.gossip.target,
                        residual=self._sess.gossip.target - total_delta,
                        group_size=len(self._sess.gossip.memory),
                    )
                self._sess.gossip = None
                self._sess.active = False

    def _record_sender(self, meta: dict) -> None:
        addr = mango_sender_addr(meta)
        if addr is None:
            return
        self._router.record_sender(str(addr), self.context.current_timestamp)

    def _touch_neighbours(self, addrs: list) -> None:
        self._router.touch(addrs, self.context.current_timestamp)

    def _live_neighbours(self) -> list:
        """Live group neighbours (branch agents included) for the flex-query
        round; token routing uses ``_gossip_neighbours``."""
        return self._router.live(
            topology_neighbors(self, tid="groups"), self.context.current_timestamp
        )

    def _gossip_neighbours(self) -> list:
        """Live neighbours with a same-sector negotiator (token-processing peers).

        Branch agents sit in the el groups (flex/overload) but run no negotiator,
        so a token forwarded to them dies. The gossip-capable registry (see
        ``setup()``) excludes them from routing while keeping them for flex.
        """
        store = getattr(self.behavior, "_scare_gossip_capable", {})
        capable = store.get(self.sector, set())
        return [a for a in self._live_neighbours() if a.aid in capable]

    def _next_hop(self, neighbours: list, nid: str, counter: int):
        return self._router.next_hop(
            neighbours, nid, counter, self.context.current_timestamp
        )

    # ------------------------------------------------------------------
    # Trigger phase
    # ------------------------------------------------------------------

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        await self._trigger.handle_ask_energy(message, meta)

    async def _handle_response_energy(
        self, message: ResponseEnergyMessage, meta: dict
    ) -> None:
        await self._trigger.handle_response_energy(message, meta)

    async def _handle_start_balance(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        await self._dispatch.handle_start_balance(message, meta)

    async def trigger_balance_negotiation(self) -> None:
        # Public: the scenario's failure-event handlers schedule this on the
        # role itself, not on _trigger.
        await self._trigger.trigger_balance_negotiation()

    async def _start_gossip(self, target: float) -> None:
        await self._engine._start_gossip(target)

    def _gossip_total_delta(self) -> float:
        return self._engine._gossip_total_delta()

    def _per_group_threshold(self) -> float:
        return self._engine._per_group_threshold()

    def _close_inflight_originator(
        self, event: str, log_reason: str | None = None
    ) -> None:
        self._engine._close_inflight_originator(event, log_reason)

    def flush_pending(self) -> None:
        self._engine.flush_pending()

    async def _handle_negotiation_message(
        self, message: EnergyNegotiationMessage, meta: dict
    ) -> None:
        await self._engine._handle_negotiation_message(message, meta)

    async def _handle_negotiation_finished_msg(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        await self._engine._handle_negotiation_finished_msg(message, meta)

    async def _handle_ask_flex(self, message: AskForAvailableFlex, meta: dict) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        member_aids = [self.context.aid] + [
            addr.aid for addr in topology_neighbors(self, tid="groups")
        ]
        if message.include_connectors:
            for addr in topology_connectors(self, tid="groups"):
                member_aids.append(addr.aid)
        cp_supply_aids: list[str] = []
        if self.enable_cp_supply_credit:
            try:
                cp_supply_aids = [
                    addr.aid for addr in topology_connectors(self, tid="groups")
                ]
            except Exception:  # noqa: BLE001 - no connector topology on this grid
                cp_supply_aids = []
        reply = _compute_flex_report(
            member_aids=member_aids,
            behavior=self.behavior,
            grid_former_policy=self._grid_former_policy,
            role_sector=self.sector,
            now=self.context.current_timestamp,
            enable_heat_l2_dispatch=self.enable_heat_l2_dispatch,
            round_id=message.round_id,
            credit_gen_capacity=self.enable_gen_capacity_supply,
            cp_supply_aids=cp_supply_aids,
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_failure_notice(self, message: FailureNotice, meta: dict) -> None:
        # Distributed failure trigger (TTL-bounded through the physical
        # neighbour graph). Heat reacts via ConstraintViolation instead; other
        # sectors trigger only on a sector match.
        if message.sector != self.sector:
            return
        # L2 escalation (all sectors): relay to the leader so it re-waterfalls
        # the component, recycling L2 allocation/membership. Only the leader
        # acts (``_handle_l2_recycle`` gates on leadership).
        try:
            group_neighbours = list(topology_neighbors(self, tid="groups"))
        except Exception:  # noqa: BLE001
            group_neighbours = []
        escalation = L2RecycleEscalation(sector=self.sector, from_member=True)
        for addr in group_neighbours:
            await self.context.send_message(escalation, receiver_addr=addr)
        # L1 setpoint trigger: sector-specific, leader-only.
        if self.sector == Sector.HEAT:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        self.context.schedule_instant_task(self._trigger.trigger_balance_negotiation())

    def _on_balance_problem(self, event: BalanceProblem, _src: Any) -> None:
        if event.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") == "leader":
            self.context.schedule_instant_task(
                self._trigger.trigger_balance_negotiation()
            )

    # ------------------------------------------------------------------
    # Setpoint application with monotonic progress guarantee
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Priority mapping helper
# ---------------------------------------------------------------------------


def _compute_actual_priority(priority: int, target: float) -> int:
    """Map a raw priority and imbalance direction to a gossip round.

    Restoration (target > 0): loads by priority (lower = earlier), generators
    last. Reduction (target < 0): generators first, loads shed reverse-priority.
    """
    if target < 0:
        if priority == 0:
            return 0  # generators first
        # Invert: more important = shed later (higher round).
        return max(1, _PRIORITY_TIERS - min(priority, _PRIORITY_TIERS) + 1)
    elif target > 0:
        if priority > 0:
            return min(priority, _PRIORITY_TIERS)
        return _PRIORITY_TIERS + 1  # generators last during restoration
    return priority


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_energy_balance_role(
    behavior: RestorationEnvironmentBehavior,
    sector: Sector,
    obs: dict,
    *,
    priority: int | None = None,
    constraint_aware: bool = True,
    proactive_util_ttl_s: float = 0.0,
    enable_monotonic_floor: bool = True,
    enable_clpu_ramp: bool = True,
    termination_tolerance: float = 1e-5,
    max_hops: int = _MAX_HOPS,
    step_decay_k0: int = _STEP_DECAY_K0_DEFAULT,
    enable_qp_gossip: bool = True,
    enable_l2_generator_ramp: bool = True,
    enable_change_only_dispatch: bool = True,
    enable_l2_priority_floor: bool = True,
    enable_actuated_ledger_writeback: bool = True,
    enable_heat_l2_dispatch: bool = False,
    enable_gen_capacity_supply: bool = False,
    enable_cp_supply_credit: bool = False,
    enable_l2_allocation_reassert: bool = False,
    l2_allocation_reassert_s: float = 2.0,
    component_scope: bool = False,
) -> EnergyBalanceNegotiator:
    if priority is None:
        priority = obs_priority(obs)
    return EnergyBalanceNegotiator(
        behavior=behavior,
        sector=sector,
        priority=priority,
        constraint_aware=constraint_aware,
        proactive_util_ttl_s=proactive_util_ttl_s,
        enable_monotonic_floor=enable_monotonic_floor,
        enable_clpu_ramp=enable_clpu_ramp,
        termination_tolerance=termination_tolerance,
        max_hops=max_hops,
        step_decay_k0=step_decay_k0,
        enable_qp_gossip=enable_qp_gossip,
        enable_l2_generator_ramp=enable_l2_generator_ramp,
        enable_change_only_dispatch=enable_change_only_dispatch,
        enable_l2_priority_floor=enable_l2_priority_floor,
        enable_actuated_ledger_writeback=enable_actuated_ledger_writeback,
        enable_heat_l2_dispatch=enable_heat_l2_dispatch,
        enable_gen_capacity_supply=enable_gen_capacity_supply,
        enable_cp_supply_credit=enable_cp_supply_credit,
        enable_l2_allocation_reassert=enable_l2_allocation_reassert,
        l2_allocation_reassert_s=l2_allocation_reassert_s,
        component_scope=component_scope,
    )
