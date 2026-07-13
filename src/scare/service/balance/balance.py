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
    clamp_to_constraints,
    constraint_allowed_fraction,
    constraint_utilization,
    has_gen_curtail_lock,
    l2_effective_floor,
    last_actuated_factor,
    line_congestion_ceiling,
    set_l2_priority_floor,
    lookup_slack,
    lookup_slack_cp_reserve,
    lookup_slack_eff_budget,
    note_actuated_factor,
    obs_capacity,
    obs_min_max,
    obs_priority,
    obs_sector,
    obs_setpoint,
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
from scare.service.balance.trust import TrustLedger, TrustParams, hash_weighted_choice

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

# Heat util above which an agent contributes headroom to the thermal-deficit
# target. Below the 0.85 warning so gossip triggers pre-violation.
_HEAT_CLEAR_FRACTION: float = 0.6

# Heat L2 supply probe: share of the unserved heat gap offered on top of the
# delivered total each flex cycle (see ``_handle_ask_flex``). Geometric climb
# to the feasibility frontier; the frontier controller trims overreach.
# 0.2 is the A/B-validated value (ab_heat_priority_v2). It converges slowly
# for the ~6 effective rebalance rounds of a 30 s task (tier-4 settles below
# its feasible level on well-supplied grids, costing heat PWSF) — 0.3-0.5 is
# the knob to try, but the 0.4 probe A/B was confounded by concurrent CP-fix
# tree changes and needs a clean campaign.
_HEAT_L2_PROBE_SHARE: float = 0.2

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
    if not aid.startswith("child-"):
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


def _slack_fed_nodes(net: Any) -> set | None:
    """Node ids reachable from an ``ExtPowerGrid`` slack over active, closed,
    non-CP branches. ``None`` when the net has no electric slack — islanding
    can't be judged then and black-start must not fire. CP branches are excluded
    (so reachability stays within the electricity carrier)."""
    try:
        branches = list(net.branches)
        childs = list(net.childs)
    except Exception:  # noqa: BLE001
        return None
    slack_nodes = {c.node_id for c in childs if isinstance(c.model, ExtPowerGrid)}
    if not slack_nodes:
        return None
    adj: dict[Any, list[Any]] = {}
    for b in branches:
        try:
            if b.model.is_cp():
                continue
            if (
                not getattr(b, "active", True)
                or not getattr(b.model, "active", True)
                or not int(getattr(b.model, "on_off", 1) or 0)
            ):
                continue
            a, c = b.id[0], b.id[1]
        except Exception:  # noqa: BLE001
            continue
        adj.setdefault(a, []).append(c)
        adj.setdefault(c, []).append(a)
    seen = set(slack_nodes)
    frontier = list(slack_nodes)
    while frontier:
        nxt: list[Any] = []
        for n in frontier:
            for nb in adj.get(n, ()):
                if nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return seen


def _load_islanded_from_slack(behavior: Any, aid: str, ts: float) -> bool:
    """True iff *aid*'s node is severed from every ``ExtPowerGrid`` slack on the
    current topology (its own island behind opened branches). Reachability is
    cached per timestamp on the behavior — the topology is identical for every
    agent within a step, so the BFS runs once per step, not once per agent."""
    net = getattr(behavior, "_net", None)
    if net is None or not aid.startswith("child-"):
        return False
    cache = getattr(behavior, "_island_fed_cache", None)
    if cache is None or cache[0] != ts:
        fed = _slack_fed_nodes(net)
        behavior._island_fed_cache = (ts, fed)
    else:
        fed = cache[1]
    if fed is None:
        return False
    try:
        node_id = net.child_by_id(int(aid[len("child-") :])).node_id
    except Exception:  # noqa: BLE001
        return False
    return node_id not in fed


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
    # Feasible-δ box anchored to the *starting* setpoint. NOT recomputed per
    # step: the agent's own regulate flips the LP's reported sp, flipping the
    # box sign and driving a full-shed/full-load oscillation. Anchoring keeps δ
    # a true cumulative change and the box constant.
    dmin_starting: float = 0.0
    dmax_starting: float = 0.0
    # Per-agent contribution ledger merged by keeping the highest-counter entry
    # (avoids cyclic double-counting). Value: (delta, counter, priority, saturated).
    # saturated entries are excluded from the equal-share denominator so
    # per-visit contraction doesn't collapse as the boundary set grows.
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
        enable_monotonic_floor: bool = True,
        enable_clpu_ramp: bool = True,
        enable_island_blackstart: bool = False,
        max_hops: int = _MAX_HOPS,
        step_decay_k0: int = _STEP_DECAY_K0_DEFAULT,
        enable_qp_gossip: bool = True,
        enable_l2_generator_ramp: bool = True,
        enable_change_only_dispatch: bool = True,
        enable_l2_priority_floor: bool = True,
        enable_actuated_ledger_writeback: bool = True,
        enable_nominal_slack_supply: bool = False,
        enable_cp_aware_slack_supply: bool = False,
        enable_heat_l2_dispatch: bool = False,
        component_scope: bool = False,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        # True when L2 runs per connected component (holon cliques are not
        # built). Upward reactive triggers then go to the leader's own L2 and
        # the component-peer mesh, not the arbitrary ``holons`` chunk topology.
        self._component_scope = bool(component_scope)
        # Monotonic counter for DETERMINISTIC negotiation IDs. The id feeds
        # hash-based gossip routing; a uuid4 id (os.urandom, immune to seeding)
        # made routing vary run-to-run. ``aid/seq`` is unique and reproducible.
        # See project_restoration_sim_nonreproducible.
        self._neg_seq = 0
        self.priority = priority
        self.impact_weight = impact_weight
        self.termination_tolerance = termination_tolerance
        self.constraint_aware = constraint_aware
        self.enable_monotonic_floor = enable_monotonic_floor
        self.enable_clpu_ramp = enable_clpu_ramp
        # Island black-start: shed a newly-islanded load to zero, then let CLPU
        # ramp it back under the island former's coordination (see config).
        self.enable_island_blackstart = bool(enable_island_blackstart)
        self._island_blackstart_active = False
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
        self._last_notified_setpoint: float | None = None
        # Last service-fraction actually dispatched; an identical incoming
        # allocation re-asserts the floor without preempting an in-flight gossip.
        self._last_dispatched_service_fraction: dict[str, dict[int, float]] | None = (
            None
        )
        # Fix 2 (opt-in): credit a slack's supply contribution at the nominal
        # operator budget B (abs(cap)) rather than the wound-down eff_budget.
        self.enable_nominal_slack_supply = bool(enable_nominal_slack_supply)
        # CP-aware slack supply (opt-in): debit the slack's credited budget by
        # the SlackBudgetMonitor's measured over-draw so the holon balances
        # native load NET of the cross-sector (CP) draw riding the slack.
        self.enable_cp_aware_slack_supply = bool(enable_cp_aware_slack_supply)
        # Heat L2 reconnect (opt-in): actuate holon service fractions for heat
        # (dispatch-only; gossip stays heat-excluded) and report delivered heat
        # as the sector's flex supply pool.
        self.enable_heat_l2_dispatch = bool(enable_heat_l2_dispatch)

        # Sector-specific convergence rate unless overridden.
        ts = SECTOR_TIMESCALE.get(sector, {})
        self.convergence_rate = (
            convergence_rate
            if convergence_rate is not None
            else ts.get("convergence_rate", 0.5)
        )

        self._active: bool = False
        self._gossip: _GossipState | None = None
        # Setpoint-gathering phase state, before gossip starts.
        self._trigger_nid: str | None = None
        self._trigger_responses: dict[str, float] = {}
        self._trigger_expected: int = 0

        # Total |cap| across the group (refreshed each trigger); drives the
        # per-group threshold.
        self._group_capacity_abs: float = 0.0

        # Monotonic floor: highest regulation factor applied during restoration;
        # may only decrease while a violation is active.
        self._restoration_floor: float = 0.0
        self._constraint_violation_active: bool = False

        # CLPU rate limiter: post-outage inrush is 2-6x steady state. Caps factor
        # growth (decreases unrestricted); ramp scales with convergence_rate.
        self._last_regulate_timestamp: float | None = None
        self._last_regulate_factor: float = 0.0
        self._clpu_ramp_per_s: float = self.convergence_rate

        # Neighbour liveness: str(addr) -> last inbound timestamp. Seeded on
        # first contact so unresponsive nodes age out rather than ghost-alive.
        self._neighbour_last_seen: dict[str, float] = {}
        poll = ts.get("poll_period_s", 1.0)
        self._heartbeat_max_age_s: float = poll * _HEARTBEAT_MAX_AGE_MULTIPLE

        # B.1: continuous coupling weights K_ij(t) in [0, 1] biasing forwarding
        # and gating liveness. Decay scales with poll period.
        self._trust = TrustLedger(
            TrustParams(
                decay_rate_per_s=1.0 / max(poll * _HEARTBEAT_MAX_AGE_MULTIPLE, 1.0),
                recover_rate=0.6,
                liveness_threshold=0.5,
                initial=1.0,
            )
        )

        # Proactive constraint util: variable -> last-reported util in [0, 1]
        # from ConstraintWarning events. Throttles the gossip step near a bound.
        self._proactive_util: dict[str, float] = {}

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

        # Mango dispatches synchronously; wrap async handlers to self-schedule
        # (tracked by termination detection) and stamp the sender's heartbeat.
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self._record_sender(meta)
                self.context.schedule_instant_task(coro_fn(msg, meta))

            return _sync

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
        # Record proximity-to-bound util so the gossip step can throttle. Other
        # sectors ignored (coupling handled at holon/CP level).
        if event.sector != self.sector:
            return
        self._proactive_util[event.variable] = float(event.utilization)

    def _on_constraint_violation(self, event: ConstraintViolation, _src: Any) -> None:
        if event.sector == self.sector:
            self._constraint_violation_active = True
            # Cancel active gossip: the constraint landscape changed, so a stale
            # target may push deeper into violation. A fresh round retriggers.
            if self._gossip is not None:
                logger.info(
                    "[%s] cancelling gossip %s due to %s violation",
                    self.context.aid,
                    self._gossip.negotiation_id[:8],
                    event.variable,
                )
                if self._gossip.is_originator:
                    total_delta = self._gossip_total_delta()
                    record_negotiation(
                        t=self.context.current_timestamp,
                        aid=self.context.aid,
                        sector=self.sector.value,
                        nid=self._gossip.negotiation_id,
                        event="cancelled",
                        target=self._gossip.target,
                        residual=self._gossip.target - total_delta,
                        group_size=len(self._gossip.memory),
                    )
                self._gossip = None
                self._active = False

    def _check_violation_cleared(self) -> None:
        """Clear the violation flag if the monitor reports local feasibility again."""
        if not self._constraint_violation_active:
            return
        monitor = self._find_constraint_monitor()
        if monitor is not None and monitor.is_locally_feasible():
            self._constraint_violation_active = False

    def _worst_neighbour_utilization(self) -> float:
        """Worst utilization reported by any neighbour, via the monitor (0 if none)."""
        monitor = self._find_constraint_monitor()
        return monitor.worst_neighbour_utilization() if monitor is not None else 0.0

    def _gossip_total_delta(self) -> float:
        """``Σ δ_i`` across the active gossip ledger (0 when no gossip active)."""
        if self._gossip is None:
            return 0.0
        return ledger_total_delta(self._gossip.memory)

    def _compute_participation_scale(self, obs: dict) -> float:
        """Throttle in [0, 1] blending local, worst-neighbour, and proactive util.

        Heat exempt: thermal violations want stressed loads to shed, not throttle.
        """
        if not self.constraint_aware or self.sector == Sector.HEAT:
            return 1.0
        scale = 1.0
        for var, (lo, hi) in SECTOR_CONSTRAINTS.get(self.sector, {}).items():
            if var in obs:
                util = constraint_utilization(float(obs[var]), lo, hi)
                scale = min(scale, max(0.0, 1.0 - util))
        neigh_util = self._worst_neighbour_utilization()
        if neigh_util > 0.0:
            scale = min(scale, max(0.0, 1.0 - neigh_util))
        if self._proactive_util:
            worst_proactive = max(self._proactive_util.values())
            scale = min(scale, max(0.0, 1.0 - worst_proactive))
        return scale

    def _find_constraint_monitor(self):
        from scare.service.control.constraints import GridConstraintMonitor

        # Use ``get_role(cls)``; RoleContext has no ``.roles`` attribute (it
        # silently returns None). Sector-guard defensively.
        monitor = self.context.get_role(GridConstraintMonitor)
        if monitor is not None and monitor.sector == self.sector:
            return monitor
        return None

    # ------------------------------------------------------------------
    # Neighbour liveness / heartbeat
    # ------------------------------------------------------------------

    def _record_sender(self, meta: dict) -> None:
        addr = mango_sender_addr(meta)
        if addr is None:
            return
        key = str(addr)
        now = self.context.current_timestamp
        self._neighbour_last_seen[key] = now
        # B.1: each received message recovers the K-score.
        self._trust.on_message_received(key, now)

    def _touch_neighbours(self, addrs: list) -> None:
        """Seed the heartbeat clock for just-contacted neighbours.

        Grants a grace period before an unresponsive node ages out.
        """
        now = self.context.current_timestamp
        for addr in addrs:
            key = str(addr)
            if key not in self._neighbour_last_seen:
                self._neighbour_last_seen[key] = now

    def _update_gap_window_and_check_stall(
        self, open_gap: float, target: float
    ) -> bool:
        """P2: append the post-update gap and decide whether gossip has stalled.

        Stall when past warm-up, window full, its range is below
        ``max(_STALL_TOL_FRACTION · |T|, _STALL_TOL_FLOOR)``, and the gap still
        exceeds the per-group threshold. Warm-up covers the priority-gating
        delay plus a full post-warmup window so it doesn't fire during silence.
        """
        if self._gossip is None:
            return False
        active = max(1, len(self._gossip.memory))
        window_size = _STALL_WINDOW_FACTOR * active
        win = self._gossip.gap_window
        win.append(open_gap)
        if len(win) > window_size:
            del win[0]
        # Warm-up gate: early rounds are silenced by priority/sub-round gating.
        warmup = _PRIORITY_TIERS + 1 + window_size
        if self._gossip.counter < warmup:
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
        if self._gossip is None:
            return
        total_delta = self._gossip_total_delta()
        target = self._gossip.target
        residual = target - total_delta
        logger.info(
            "[%s] gossip stalled (sector=%s, residual=%.4f, window=%d)",
            self.context.aid,
            self.sector.value,
            residual,
            len(self._gossip.gap_window),
        )
        if self._gossip.is_originator:
            record_negotiation(
                t=self.context.current_timestamp,
                aid=self.context.aid,
                sector=self.sector.value,
                nid=self._gossip.negotiation_id,
                event="stalled",
                target=target,
                residual=residual,
                group_size=len(self._gossip.memory),
            )
        # Suppress the "finished" entry — this terminal is "stalled".
        await self._finish_negotiation(record_finished=False)

    def _per_group_threshold(self) -> float:
        """Threshold scaled by the group's load-capacity snapshot.

        Falls back to the sector default floor before the first trigger.
        """
        if self._group_capacity_abs <= 0.0:
            return _start_threshold(self.sector)
        return max(
            _THRESHOLD_ABS_FLOOR,
            _THRESHOLD_CAPACITY_FRACTION * self._group_capacity_abs,
        )

    def _live_neighbours(self) -> list:
        """Group neighbours whose trust score K_ij exceeds the liveness threshold.

        Unknown neighbours bootstrap optimistically (K = 1.0). Includes every
        live neighbour (branch agents too) — for the flex-query round. For token
        routing use ``_gossip_neighbours``.
        """
        all_neighbours = topology_neighbors(self, tid="groups")
        now = self.context.current_timestamp
        return [a for a in all_neighbours if self._trust.is_live(str(a), now)]

    def _gossip_neighbours(self) -> list:
        """Live neighbours with a same-sector negotiator (token-processing peers).

        Branch agents sit in the el groups (flex/overload) but run no negotiator,
        so a token forwarded to them dies. The gossip-capable registry (see
        ``setup()``) excludes them from routing while keeping them for flex.
        """
        store = getattr(self.behavior, "_scare_gossip_capable", {})
        capable = store.get(self.sector, set())
        return [a for a in self._live_neighbours() if a.aid in capable]

    def _scored_neighbours(self, neighbours: list) -> list[float]:
        """Return the K-score for each neighbour in ``neighbours`` order."""
        now = self.context.current_timestamp
        return [self._trust.score(str(a), now) for a in neighbours]

    def _next_hop(self, neighbours: list, nid: str, counter: int):
        """B.1: K-weighted deterministic next-hop.

        Picks proportional to trust K_ij; uniform SHA256-modulo when all K equal.
        Low-K neighbours are routed around.
        """
        if not neighbours:
            return None
        weights = self._scored_neighbours(neighbours)
        return hash_weighted_choice(neighbours, weights, f"{nid}:{counter}")

    # ------------------------------------------------------------------
    # Trigger phase
    # ------------------------------------------------------------------

    async def trigger_balance_negotiation(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # MW balance deactivated for heat: frontier controller + auction own it
        # and the unbounded slack means no MW imbalance to resolve.
        if self.sector == Sector.HEAT:
            return
        if self._active:
            return
        self._active = True

        neighbours = self._live_neighbours()
        self._touch_neighbours(neighbours)
        # Snapshot group |cap| (loads only, cap > 0) so the threshold scales
        # with present demand — the right denominator for curtailment too.
        members = [self.context.aid] + [a.aid for a in neighbours]
        cap_sum = 0.0
        for aid in members:
            cap = obs_capacity(self.behavior.observe(aid) or {})
            if cap > 0:
                cap_sum += cap
        self._group_capacity_abs = cap_sum
        logger.info(
            "[%s] balance negotiation triggered (sector=%s, group size=%d, Σ|cap_load|=%.4f)",
            self.context.aid,
            self.sector.value,
            len(neighbours) + 1,
            cap_sum,
        )
        if not neighbours:
            obs = self.behavior.observe(self.context.aid) or {}
            await self._start_gossip(-self._reported_setpoint(obs))
            return

        self._neg_seq += 1
        nid = f"{self.context.aid}/{self._neg_seq}"
        self._trigger_nid = nid
        self._trigger_responses = {}
        self._trigger_expected = len(neighbours)

        msg = AskEnergyMessage(negotiation_id=nid, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

        # Deadline mirroring the gossip timeout: one dropped reply must not
        # wedge the leader ``_active=True`` forever. On expiry proceed with the
        # responses that did arrive (missing members contribute sp=0).
        base = _GOSSIP_TIMEOUT_BASE_S.get(self.sector, _GOSSIP_TIMEOUT_DEFAULT_S)
        timeout = base + len(neighbours) * _GOSSIP_TIMEOUT_PER_AGENT_S
        self.context.schedule_timestamp_task(
            self._trigger_timeout(nid),
            timestamp=self.context.current_timestamp + timeout,
        )

    async def _trigger_timeout(self, nid: str) -> None:
        if self._trigger_nid != nid:
            return
        logger.warning(
            "[%s] trigger phase timed out (%d/%d responses) — proceeding",
            self.context.aid,
            len(self._trigger_responses),
            self._trigger_expected,
        )
        await self._complete_trigger_phase()

    async def _complete_trigger_phase(self) -> None:
        own_obs = self.behavior.observe(self.context.aid) or {}
        total_sp = self._reported_setpoint(own_obs) + sum(
            self._trigger_responses.values()
        )
        responders = set(self._trigger_responses)
        self._trigger_nid = None
        self._trigger_responses = {}

        # Tier-1 hard pre-step: lift tier-1 to regulation=1 if the pool
        # covers it, else pro-rata distribute and shed tiers 2/3/4.
        residual_target, skip_gossip = self._pre_apply_tier1_hard(
            total_sp, responders
        )
        if skip_gossip:
            # Residual below threshold.
            self._active = False
            return
        await self._start_gossip(residual_target)

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        sp = self._reported_setpoint(obs)
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=sp,
            available=cap - sp,  # headroom, not total cap
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    def _reported_setpoint(self, obs: dict) -> float:
        """Setpoint contribution to the group's negotiation target.

        El/gas: the raw setpoint (Σ s_i ≈ 0 ⇒ balanced). Heat: amplified by the
        local thermal deficit so a stressed group surfaces a negative target.

        F2 — slack target: a registered slack reports its *target infeed*
        ``slack_target_fraction · rating`` instead of the LP draw, reframing the
        imbalance so the rest of the group balances to that target. Only
        coherent when the slack's community spans the component
        (``component_level``); else ``SlackBudgetMonitor`` enforces the budget.
        """
        slack = lookup_slack(self.behavior, self.context.aid)
        if slack is not None:
            cfg = getattr(self.behavior, "_scare_config", None)
            fraction = float(
                getattr(cfg, "slack_target_fraction", 0.0) if cfg is not None else 0.0
            )
            # ``slack.cap`` is generator-convention (negative); import target is
            # its magnitude.
            rating = abs(slack.cap)
            sp = fraction * rating
            return sp
        sp = obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        if self.sector == Sector.HEAT:
            sp += _heat_thermal_deficit_mw(obs)
        return sp

    async def _handle_response_energy(
        self, message: ResponseEnergyMessage, meta: dict
    ) -> None:
        if message.negotiation_id != self._trigger_nid:
            return

        sender_key = str(mango_sender_addr(meta))
        self._trigger_responses[sender_key] = message.setpoint

        if len(self._trigger_responses) >= self._trigger_expected:
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
        threshold = self._per_group_threshold()

        members = [(self.context.aid, True)]
        for neigh in self._live_neighbours():
            members.append((neigh.aid, str(neigh) in responders))

        tier1_records: list[
            tuple[str, float, float, str]
        ] = []  # (aid, cap, sp, sector)
        non_tier1_loads: list[
            tuple[str, float, float, str, int]
        ] = []  # (aid, cap, sp, sec, tier)
        pool = 0.0
        for aid, responded in members:
            obs = self.behavior.observe(aid) or {}
            cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
            sec_enum = obs_sector(obs, behavior=self.behavior, aid=aid)
            if sec_enum is None:
                continue
            sec = sec_enum.value
            if cap > 0:
                prio = obs_priority(obs, behavior=self.behavior, aid=aid)
                sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
                if int(prio) == 1:
                    tier1_records.append((aid, float(cap), float(sp), sec))
                else:
                    non_tier1_loads.append(
                        (aid, float(cap), float(sp), sec, int(prio))
                    )
            elif cap < 0:
                # Generators that answered this trigger round are proven live:
                # credit delivered |sp| plus rampable headroom, else a cold
                # start (sp≈0) freezes the pool at slack-only and zeroes tiers
                # 2-4 forever. Non-responders keep delivered-only |sp| so
                # unreachable capacity can't declare tier-1 "feasible". Slacks
                # keep the budgeted rating.
                if lookup_slack(self.behavior, aid) is not None:
                    pool += abs(float(cap))
                else:
                    sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
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
            now = float(self.context.current_timestamp)
            applied_tier1 = 0
            for aid, _cap, _sp, sec in tier1_records:
                apply_regulate(
                    self.behavior,
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
                self.context.aid,
                pool,
                tier1_unmet,
                applied_tier1,
                residual,
            )
            return residual, abs(residual) <= threshold

        # Infeasible: pro-rata pool across tier-1 by unmet; tiers 2-4 -> 0.
        now = float(self.context.current_timestamp)
        applied_tier1 = 0
        applied_shed = 0
        for (aid, cap, sp, sec), unmet in zip(tier1_records, tier1_unmet_per_load):
            if unmet <= 0.0 or cap <= 0.0:
                continue
            share = pool * (unmet / tier1_unmet)
            new_sp = sp + share
            factor = max(0.0, min(1.0, new_sp / cap))
            apply_regulate(
                self.behavior,
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
                self.behavior,
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
            self.context.aid,
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

    async def _start_gossip(self, target: float) -> None:
        # MW balance deactivated for heat; also guards the holon override_target
        # path that calls here directly.
        if self.sector == Sector.HEAT:
            return
        threshold = self._per_group_threshold()
        if abs(target) < threshold:
            logger.info(
                "[%s] gossip skipped: already balanced (target=%.4f, threshold=%.4f)",
                self.context.aid,
                target,
                threshold,
            )
            record_negotiation(
                t=self.context.current_timestamp,
                aid=self.context.aid,
                sector=self.sector.value,
                nid="",
                event="skipped_balanced",
                target=target,
            )
            self._active = False
            return

        # An overlapping trigger can reach here with a live originator gossip;
        # retire it as ``abandoned`` first so its diary terminal isn't dropped.
        self._close_inflight_originator(
            "abandoned", log_reason="superseded by new gossip"
        )

        # Reset the violation flag so the monotonic floor only yields while a
        # violation is actively present.
        self._constraint_violation_active = False

        # Gossip-only neighbours: excludes members without a negotiator (branch
        # monitors) that would drop the token.
        neighbours = self._gossip_neighbours()
        self._touch_neighbours(neighbours)
        self._neg_seq += 1
        nid = f"{self.context.aid}/{self._neg_seq}"
        self_key = str(self.context.addr)

        obs = self.behavior.observe(self.context.aid) or {}
        starting_sp = obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        # Anchor the QP δ-box to the starting state (see _GossipState);
        # per-step recompute causes a self-driven sign-flip oscillation.
        dmin_start, dmax_start = obs_min_max(
            obs, behavior=self.behavior, aid=self.context.aid
        )

        lambda_seed = compute_lambda_seed(
            target,
            len(neighbours),
            priority=self.priority,
            priority_tiers=_PRIORITY_TIERS,
        )

        self._gossip = _GossipState(
            negotiation_id=nid,
            target=target,
            counter=0,
            current_delta=0.0,
            starting_setpoint=starting_sp,
            dmin_starting=dmin_start,
            dmax_starting=dmax_start,
            memory={self_key: (0.0, 0, self.priority, False)},
            is_originator=True,
            dual_lambda=lambda_seed,
        )

        if not neighbours:
            # Isolated agent: approve the fallback directly with the full
            # deficit (activates local DGs) and self-dispatch inline if that
            # role is absent.
            logger.info(
                "[%s] gossip skipped: singleton (target=%.4f) — escalating to local-gen fallback",
                self.context.aid,
                target,
            )
            record_negotiation(
                t=self.context.current_timestamp,
                aid=self.context.aid,
                sector=self.sector.value,
                nid="",
                event="skipped_singleton",
                target=target,
                group_size=1,
            )
            if abs(target) > threshold:
                record_event(
                    t=self.context.current_timestamp,
                    kind="local_gen_request",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=f"residual={target:.4f} (singleton)",
                )
                self.context.emit_event(
                    LocalGenerationApproval(sector=self.sector, residual_deficit=target)
                )
                self._try_self_dispatch(target)
            self._active = False
            self._gossip = None
            return

        # Committed to a multi-party gossip; record the start.
        group_size = len(neighbours) + 1
        logger.info(
            "[%s] starting gossip (sector=%s, target=%.4f)",
            self.context.aid,
            self.sector.value,
            target,
        )
        record_negotiation(
            t=self.context.current_timestamp,
            aid=self.context.aid,
            sector=self.sector.value,
            nid=nid,
            event="started",
            target=target,
            group_size=group_size,
        )

        msg = EnergyNegotiationMessage(
            negotiation_id=nid,
            sector=self.sector,
            negotiation_target=target,
            current_delta=0.0,
            counter=0,
            memory=dict(self._gossip.memory),
            dual_lambda=self._gossip.dual_lambda,
        )
        if self.enable_qp_gossip:
            # Single-token: the dual λ can't be averaged across parallel tokens
            # with divergent ledger views. Forward to one K-weighted next-hop.
            next_addr = self._next_hop(neighbours, nid, 0)
            await self.context.send_message(msg, receiver_addr=next_addr)
        else:
            # Equal-share tolerates multi-token broadcast: the ledger merge
            # composes correctly.
            for addr in neighbours:
                await self.context.send_message(msg, receiver_addr=addr)

        # Wallclock timeout: force-finish if not converged. Per-sector base +
        # per-agent scaling.
        base = _GOSSIP_TIMEOUT_BASE_S.get(self.sector, _GOSSIP_TIMEOUT_DEFAULT_S)
        timeout = base + len(neighbours) * _GOSSIP_TIMEOUT_PER_AGENT_S
        deadline = self.context.current_timestamp + timeout
        self.context.schedule_timestamp_task(
            self._gossip_timeout(nid), timestamp=deadline
        )

    async def _gossip_timeout(self, negotiation_id: str) -> None:
        if self._gossip is not None and self._gossip.negotiation_id == negotiation_id:
            logger.warning(
                "[%s] gossip %s timed out — forcing finish",
                self.context.aid,
                negotiation_id[:8],
            )
            if self._gossip.is_originator:
                total_delta = self._gossip_total_delta()
                residual = self._gossip.target - total_delta
                record_negotiation(
                    t=self.context.current_timestamp,
                    aid=self.context.aid,
                    sector=self.sector.value,
                    nid=negotiation_id,
                    event="timed_out",
                    target=self._gossip.target,
                    residual=residual,
                    group_size=len(self._gossip.memory),
                )
            await self._finish_negotiation(record_finished=False)

    async def _handle_negotiation_message(
        self, message: EnergyNegotiationMessage, meta: dict
    ) -> None:
        nid = message.negotiation_id
        counter = message.counter + 1

        if counter > self.max_hops + 1:
            return

        self_key = str(self.context.addr)

        if self._gossip is None or self._gossip.negotiation_id != nid:
            # A different nid arriving over our in-flight gossip: record an
            # ``abandoned`` terminal for the old nid before overwriting state
            # (preserves started == Σ terminals).
            if (
                self._gossip is not None
                and self._gossip.is_originator
                and self._gossip.negotiation_id != nid
            ):
                prev_total = self._gossip_total_delta()
                record_negotiation(
                    t=self.context.current_timestamp,
                    aid=self.context.aid,
                    sector=self.sector.value,
                    nid=self._gossip.negotiation_id,
                    event="abandoned",
                    target=self._gossip.target,
                    residual=self._gossip.target - prev_total,
                    group_size=len(self._gossip.memory),
                )
            obs = self.behavior.observe(self.context.aid) or {}
            init_dmin, init_dmax = obs_min_max(
                obs, behavior=self.behavior, aid=self.context.aid
            )
            # Adopt via ledger_merge so the incoming ledger gets the same
            # Byzantine clip as the merge path below.
            cap_byz = _BYZANTINE_DELTA_CAP_MULTIPLE * max(
                abs(message.negotiation_target), 1.0
            )
            adopted_memory: dict[str, tuple[float, int, int, bool]] = {}
            ledger_merge(adopted_memory, message.memory, byzantine_cap=cap_byz)
            self._gossip = _GossipState(
                negotiation_id=nid,
                target=message.negotiation_target,
                counter=counter,
                current_delta=0.0,
                starting_setpoint=obs_setpoint(
                    obs, behavior=self.behavior, aid=self.context.aid
                ),
                dmin_starting=init_dmin,
                dmax_starting=init_dmax,
                memory=adopted_memory,
                dual_lambda=getattr(message, "dual_lambda", 0.0),
            )
        else:
            self._gossip.counter = counter
            # P6: λ travels with the message; adopt the latest (safe under
            # single-token gossip).
            self._gossip.dual_lambda = getattr(
                message, "dual_lambda", self._gossip.dual_lambda
            )
            # Merge ledger keeping newest-counter entries; Byzantine-clip each
            # delta to a multiple of |target|.
            cap_byz = _BYZANTINE_DELTA_CAP_MULTIPLE * max(abs(self._gossip.target), 1.0)
            ledger_merge(self._gossip.memory, message.memory, byzantine_cap=cap_byz)

        target = self._gossip.target
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        # Gossip-anchored δ-box, not a fresh ``obs_min_max`` (which flips after
        # the agent's own regulate and drives a bang-bang oscillation).
        dmin = self._gossip.dmin_starting
        dmax = self._gossip.dmax_starting

        prev_own = self._gossip.memory.get(self_key, (0.0, 0, self.priority, False))[0]
        total_delta = self._gossip_total_delta()
        open_gap = target - total_delta

        participation_scale = self._compute_participation_scale(obs)

        active_count = max(1, len(self._gossip.memory))
        n_free = max(1, sum(1 for v in self._gossip.memory.values() if not v[3]))

        if self.enable_qp_gossip:
            # P6: primal-dual QP closed-form update. δ_i = clamp(a_i · λ, dmin,
            # dmax), a_i = priority weight, sign(λ) = sign(T). A continuous
            # priority waterfall; participation_scale folds into a_i.
            target_sign = 1 if target > 0 else (-1 if target < 0 else 0)
            a_i_base = qp_priority_weight(
                self.priority,
                target_sign,
                priority_tiers=_PRIORITY_TIERS,
            )
            a_i = a_i_base * self.impact_weight * participation_scale
            new_delta = qp_primal(a_i, self._gossip.dual_lambda, dmin, dmax)
            saturated = _is_saturated(new_delta, dmin, dmax)
            self._gossip.memory[self_key] = (
                new_delta,
                counter,
                self.priority,
                saturated,
            )
            self._gossip.current_delta = new_delta
            # Dedup gate (QP only): δ is monotonic in λ, so once saturated
            # further visits request the same δ — skip the actuator write to
            # avoid quadratic re-solves. First visit always applies.
            delta_step = abs(new_delta - prev_own)
            apply_threshold = 1e-4 * max(abs(cap), 1.0)
            if cap != 0.0 and (delta_step > apply_threshold or prev_own == 0.0):
                applied_sp = self._apply_setpoint(
                    self._gossip.starting_setpoint + new_delta
                )
                # Write back the physically-actuated delta (not the requested):
                # a clamped load shows its true contribution, so the dual raises
                # λ and unconstrained loads absorb the freed supply.
                self._writeback_actuated_delta(
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
                self._gossip.memory,
                target_sign,
                priority_tiers=_PRIORITY_TIERS,
            )
            self._gossip.dual_lambda += (
                step_size(
                    self.convergence_rate, counter, step_decay_k0=self.step_decay_k0
                )
                * residual
                / sum_a_est
            )
        else:
            # Equal-share step: each active participant aims for 1/n_free of the
            # open gap, scaled by step + participation. Priority/sub-round gated.
            own_change = (
                (open_gap / n_free)
                * self.impact_weight
                * step_size(
                    self.convergence_rate, counter, step_decay_k0=self.step_decay_k0
                )
                * participation_scale
            )

            actual_prio = _compute_actual_priority(self.priority, target)

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
                self._gossip.memory[self_key] = (
                    new_delta,
                    counter,
                    self.priority,
                    saturated,
                )
                self._gossip.current_delta = new_delta
                if cap != 0.0:
                    applied_sp = self._apply_setpoint(
                        self._gossip.starting_setpoint + new_delta
                    )
                    self._writeback_actuated_delta(
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
        neighbours = self._gossip_neighbours()

        if stalled:
            await self._finish_negotiation_stalled()
            return

        if abs(open_gap) <= self.termination_tolerance or counter >= self.max_hops:
            await self._finish_negotiation()
        elif neighbours:
            next_addr = self._next_hop(neighbours, nid, counter)
            fwd = EnergyNegotiationMessage(
                negotiation_id=nid,
                sector=self.sector,
                negotiation_target=target,
                current_delta=self._gossip.current_delta,
                counter=counter,
                memory=dict(self._gossip.memory),
                dual_lambda=self._gossip.dual_lambda,
            )
            await self.context.send_message(fwd, receiver_addr=next_addr)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    async def _finish_negotiation(self, *, record_finished: bool = True) -> None:
        starting_sp = (
            self._gossip.starting_setpoint
            if self._gossip
            else obs_setpoint(self.behavior.observe(self.context.aid) or {})
        )
        delta = self._gossip.current_delta if self._gossip else 0.0
        new_sp = starting_sp + delta
        # Coordination overhaul: only propagate this finish UPWARD (to the holon
        # ADMM at L2 and CP at L3, plus the local-leader L2 self-trigger) when
        # the converged setpoint actually moved since the last notification — a
        # gossip that re-converges to the same value must not re-trigger the
        # cascade (this is what lets the time-throttle be removed).
        prev_notified = self._last_notified_setpoint
        notify_upward = (
            not self.enable_change_only_dispatch
            or prev_notified is None
            or abs(new_sp - prev_notified) > _UPWARD_NOTIFY_TOL
        )
        if notify_upward:
            self._last_notified_setpoint = new_sp

        if self._gossip is not None:
            total_delta = self._gossip_total_delta()
            target = self._gossip.target
            residual = target - total_delta
            logger.info(
                "[%s] gossip finished (sector=%s, delta=%.4f, residual=%.4f)",
                self.context.aid,
                self.sector.value,
                delta,
                residual,
            )
            # ``record_finished=False`` when the caller recorded a more specific
            # terminal. Only the originator records (peers never recorded a
            # "started", so their terminal would inflate the invariant).
            if record_finished and self._gossip.is_originator:
                record_negotiation(
                    t=self.context.current_timestamp,
                    aid=self.context.aid,
                    sector=self.sector.value,
                    nid=self._gossip.negotiation_id,
                    event="finished",
                    target=target,
                    residual=residual,
                    group_size=len(self._gossip.memory),
                )

            # Unresolved deficit escalates to local-gen fallback via L2 first
            # (holon absorbs cross-group before L1 falls back to local DGs).
            # Leader-only; members surface residual via the NFE broadcast.
            if (
                abs(residual) > self._per_group_threshold() * 10
                and topology_characteristic(self, tid="groups") == "leader"
            ):
                record_event(
                    t=self.context.current_timestamp,
                    kind="local_gen_request",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=f"residual={residual:.4f}",
                )
                # Component scope: the leader's own component ADMM was already
                # re-triggered by the local NegotiationFinishedEvent above (the
                # coordinator re-solves and floors what it can route); approve
                # the local-DG fallback for the residual it cannot. Legacy holon
                # scope: escalate to holon-clique peers so their L2 arbitrates
                # first, else self-approve when the clique is empty.
                if self._component_scope:
                    self.context.emit_event(
                        LocalGenerationApproval(
                            sector=self.sector,
                            residual_deficit=residual,
                        )
                    )
                else:
                    request = LocalGenerationRequest(
                        sector=self.sector, residual_deficit=residual
                    )
                    try:
                        holon_peers = list(topology_neighbors(self, tid="holons"))
                    except KeyError:
                        holon_peers = []
                    if holon_peers:
                        for addr in holon_peers:
                            await self.context.send_message(
                                request, receiver_addr=addr
                            )
                    else:
                        self.context.emit_event(
                            LocalGenerationApproval(
                                sector=self.sector,
                                residual_deficit=residual,
                            )
                        )

        # Local event: consumed by this agent's own L2 (_on_member_finished_local)
        # and L3 (CP channel) — the upward self-trigger. Gate on change.
        if notify_upward:
            self.context.emit_event(
                NegotiationFinishedEvent(new_setpoint=new_sp, sector=self.sector)
            )

        # Broadcast convergence to gossip-capable neighbours so each emits its
        # own local event. ``new_setpoint`` carries the leader's converged sp so
        # the CP fixed-point gate can detect movement (a hard zero would suppress
        # CP re-triggers); neighbours re-derive their own sp. ``negotiation_id``
        # lets members with matching gossip state (incl. a blocked originator)
        # release it instead of timing out.
        finished_nid = self._gossip.negotiation_id if self._gossip else ""
        neighbours = self._gossip_neighbours()
        finished_msg = NegotiationFinishedEvent(
            new_setpoint=new_sp, sector=self.sector, negotiation_id=finished_nid
        )
        for addr in neighbours:
            await self.context.send_message(finished_msg, receiver_addr=addr)

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
            if not self._component_scope:
                try:
                    holon_peers = topology_neighbors(self, tid="holons")
                except KeyError:
                    holon_peers = []
                for addr in holon_peers:
                    await self.context.send_message(finished_msg, receiver_addr=addr)

            # Leader also notifies CP connectors (both scopes)
            if topology_characteristic(self, tid="groups") == "leader":
                cp_connectors = list(topology_connectors(self, tid="groups"))
                if cp_connectors:
                    logger.info(
                        "[%s] gossip finished: notifying %d CP connectors (new_sp=%.4f)",
                        self.context.aid,
                        len(cp_connectors),
                        new_sp,
                    )
                for addr in cp_connectors:
                    await self.context.send_message(finished_msg, receiver_addr=addr)

        self._gossip = None
        # ``_active`` is owned by the trigger phase while ``_trigger_nid`` is
        # set (gossip adoption never sets it); a finishing adopted gossip must
        # not release the trigger's re-entry guard. The trigger's own gossip
        # always runs with ``_trigger_nid is None`` (cleared in
        # ``_complete_trigger_phase``), so clearing is correct then.
        if self._trigger_nid is None:
            self._active = False

    def flush_pending(self) -> None:
        """Record still-active gossip as ``stalled`` (progress) or ``abandoned``.

        Called at world teardown so an in-flight negotiation still counts toward
        the ``started == Σ terminals`` invariant.
        """
        if self._gossip is None:
            return
        if self._gossip.is_originator:
            total_delta = self._gossip_total_delta()
            target = self._gossip.target
            residual = target - total_delta
            # ``stalled`` (closed >= 30% of |target|) is a soft terminal.
            if abs(target) > 1e-12:
                progress = (abs(target) - abs(residual)) / abs(target)
            else:
                progress = 1.0
            event = "stalled" if progress >= 0.3 else "abandoned"
            record_negotiation(
                t=self.context.current_timestamp,
                aid=self.context.aid,
                sector=self.sector.value,
                nid=self._gossip.negotiation_id,
                event=event,
                target=target,
                residual=residual,
                group_size=len(self._gossip.memory),
            )
        self._gossip = None
        self._active = False

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
            self._gossip.starting_setpoint
            if self._gossip
            else obs_setpoint(self.behavior.observe(self.context.aid) or {})
        )
        delta = self._gossip.current_delta if self._gossip else 0.0
        nid = getattr(message, "negotiation_id", "")
        if nid and self._gossip is not None and self._gossip.negotiation_id == nid:
            if self._gossip.is_originator:
                total_delta = self._gossip_total_delta()
                record_negotiation(
                    t=self.context.current_timestamp,
                    aid=self.context.aid,
                    sector=self.sector.value,
                    nid=nid,
                    event="finished",
                    target=self._gossip.target,
                    residual=self._gossip.target - total_delta,
                    group_size=len(self._gossip.memory),
                )
            self._gossip = None
            # Same ownership rule as ``_finish_negotiation``: only the trigger
            # phase may hold ``_active`` while ``_trigger_nid`` is set.
            if self._trigger_nid is None:
                self._active = False
        self.context.emit_event(
            NegotiationFinishedEvent(
                new_setpoint=starting_sp + delta, sector=self.sector
            )
        )

    # ------------------------------------------------------------------
    # Flex reporting
    # ------------------------------------------------------------------

    async def _handle_ask_flex(self, message: AskForAvailableFlex, meta: dict) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return

        member_aids = [self.context.aid] + [
            addr.aid for addr in topology_neighbors(self, tid="groups")
        ]

        if message.include_connectors:
            for addr in topology_connectors(self, tid="groups"):
                member_aids.append(addr.aid)

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
            obs = self.behavior.observe(aid) or {}
            sector = obs_sector(obs, behavior=self.behavior, aid=aid)
            if sector is None:
                continue
            cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
            sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
            available = cap - sp  # headroom
            # Per-sector breakdown for multi-dimensional ADMM.
            sec_key = sector.value
            flex_by_sector[sec_key] = flex_by_sector.get(sec_key, 0.0) + available
            balance_by_sector[sec_key] = balance_by_sector.get(sec_key, 0.0) + sp
            # Route-A supply pool (generators, cap < 0). Generators count
            # *delivered* ``|sp|``, not rated ``|cap|``, so curtailed supply
            # can't inflate the pool. Slacks instead use the budgeted rating.
            if cap < 0:
                if lookup_slack(self.behavior, aid) is not None:
                    if self.enable_cp_aware_slack_supply:
                        # CP-aware: single-controller form B - reserve. Base off
                        # the nominal budget B (== abs(cap); see
                        # _maybe_register_slack) and debit the budget already
                        # consumed by the cross-sector (CP) draw + losses riding
                        # this slack (the monitor's measured over-draw). The holon
                        # then balances native load against the budget NET of the
                        # CP draw and sheds native load until the physical slack
                        # lands at B (re-measured each monitor poll => converges).
                        # NB base off B, NOT eff_budget: the eff_budget feedback
                        # ALSO winds down on the same over-draw, so eff_budget -
                        # reserve would double-subtract the excess and over-shed.
                        reserve = lookup_slack_cp_reserve(self.behavior, aid) or 0.0
                        gen_supply = max(0.0, abs(cap) - float(reserve))
                    elif self.enable_nominal_slack_supply:
                        # Fix 2: nominal operator budget B (== abs(cap); see
                        # _maybe_register_slack, which registers the slack at its
                        # _scare_slack_budget_mw). The eff_budget the
                        # SlackBudgetMonitor maintains is wound DOWN below B
                        # whenever physical import exceeds budget (irreducible CP
                        # draw on cp-heavy grids), shrinking the serviceable pool
                        # and over-shedding feasible load; B is the same hard
                        # bound the oracle serves against.
                        gen_supply = abs(cap)
                    else:
                        # Slack: effective budget if loss-compensation set one
                        # (targets ``B - losses``), else nominal.
                        eff = lookup_slack_eff_budget(self.behavior, aid)
                        gen_supply = float(eff) if eff is not None else abs(cap)
                else:
                    gen_supply = abs(sp)  # generator: deliverable, not rated
                supply_by_sector[sec_key] = (
                    supply_by_sector.get(sec_key, 0.0) + gen_supply
                )
            # Priority-tier demand aggregation (loads only: cap > 0).
            if cap > 0:
                prio = obs_priority(
                    obs,
                    behavior=self.behavior,
                    aid=aid,
                    record_default_fallback_t=self.context.current_timestamp,
                )
                demand_by_priority[prio] = demand_by_priority.get(prio, 0.0) + abs(cap)
                served_by_priority[prio] = served_by_priority.get(prio, 0.0) + abs(sp)
                # Per-(sector, tier) split for the tier-stratified holon ADMM.
                demand_by_sector_priority.setdefault(sec_key, {})
                demand_by_sector_priority[sec_key][prio] = demand_by_sector_priority[
                    sec_key
                ].get(prio, 0.0) + abs(cap)
                served_by_sector_priority.setdefault(sec_key, {})
                served_by_sector_priority[sec_key][prio] = served_by_sector_priority[
                    sec_key
                ].get(prio, 0.0) + abs(sp)
                # Unmet demand (rated cap - actual sp): captures the silent
                # disconnect-loss case (regulation=0 on a load with no path to a
                # grid-former) the CP layer would otherwise miss.
                unmet = abs(cap) - abs(sp)
                if unmet > 1e-12:
                    unmet_by_sector[sec_key] = unmet_by_sector.get(sec_key, 0.0) + unmet
            if sector != self.sector:
                continue
            total_flex += available
            total_balance += sp
            if sp > 0 and available > 0:
                total_shedded += available

        # Heat L2 reconnect: heat has no bounded slack pool (the unbounded
        # ExtHydrGrid never registers a rating), so the gen-only ledger reads
        # supply=0 for gen-less groups — and the allocation's no-supply branch
        # would shed every tier. Report heat DELIVERED TO LOADS instead: that
        # total is the pool the per-tier waterfall can reallocate. Replaces
        # (not max) the gen ledger — an in-group CHP's injection is the same
        # MW the consuming groups' loads already report, and the component
        # merge sums supplies across leaders.
        #
        # Plus an upward PROBE: delivered alone ratchets DOWN — the fractions
        # it produces cap the loads, so delivered can never rise above the
        # (transient) level it was sampled at, and L2-shed tiers stay shed
        # forever. Offering a share of the unserved gap each cycle lets the
        # estimate climb to the true feasibility frontier (physics delivers
        # the probe ⇒ next sample is higher; it doesn't ⇒ the frontier sheds
        # the overreach back with a restorable curtail-lock).
        if self.enable_heat_l2_dispatch:
            sec_heat = Sector.HEAT.value
            delivered = sum(served_by_sector_priority.get(sec_heat, {}).values())
            demand = sum(demand_by_sector_priority.get(sec_heat, {}).values())
            supply_by_sector[sec_heat] = delivered + _HEAT_L2_PROBE_SHARE * max(
                0.0, demand - delivered
            )

        reply = AvailableFlexAnswer(
            flex=total_flex,
            balance=total_balance,
            shedded=total_shedded,
            sector=self.sector,
            flex_by_sector=flex_by_sector,
            balance_by_sector=balance_by_sector,
            demand_by_priority=demand_by_priority,
            served_by_priority=served_by_priority,
            unmet_by_sector=unmet_by_sector,
            demand_by_sector_priority=demand_by_sector_priority,
            served_by_sector_priority=served_by_sector_priority,
            supply_by_sector=supply_by_sector,
            round_id=message.round_id,
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    def _close_inflight_originator(
        self, event: str, log_reason: str | None = None
    ) -> None:
        """Record a terminal for in-flight gossip this agent originated.

        Preserves ``started == Σ terminals`` before teardown. No-op for relays.
        Shared by the four sites that retire an active gossip.
        """
        if self._gossip is None or not self._gossip.is_originator:
            return
        total_delta = self._gossip_total_delta()
        record_negotiation(
            t=self.context.current_timestamp,
            aid=self.context.aid,
            sector=self.sector.value,
            nid=self._gossip.negotiation_id,
            event=event,
            target=self._gossip.target,
            residual=self._gossip.target - total_delta,
            group_size=len(self._gossip.memory),
        )
        if log_reason is not None:
            logger.info(
                "[%s] gossip %s %s — %s",
                self.context.aid,
                self._gossip.negotiation_id[:8],
                event,
                log_reason,
            )

    def _yield_to_l2_authority(self, route: str) -> None:
        """Abandon any in-flight L1 gossip so an arriving L2 directive can land.

        L2 carries the holon's authoritative priority decision; without the
        yield a gossip's ``_active=True`` would swallow it.
        """
        if not self._active:
            return
        self._close_inflight_originator(
            "abandoned", log_reason=f"yielding to L2 ({route})"
        )
        self._gossip = None
        self._active = False

    async def _handle_start_balance(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # MW balance deactivated for heat: gossip and scalar overrides never
        # run. With the heat L2 reconnect on, Route-A service fractions ARE
        # actuated (dispatch-only) — the tier-graded allocation heat otherwise
        # lacks entirely; the curtail-lock keeps temperature-feasibility
        # authority with the frontier (L2 raises defer, L2 sheds pass).
        if self.sector == Sector.HEAT:
            if not self.enable_heat_l2_dispatch:
                return
            service_frac = getattr(
                message, "service_fraction_by_sector_priority", None
            )
            if not service_frac or not self._heat_fractions_meaningful(
                service_frac
            ):
                return
            if (
                self.enable_change_only_dispatch
                and self._service_fraction_unchanged(service_frac)
            ):
                return
            self.context.schedule_instant_task(
                self._dispatch_service_fractions(service_frac)
            )
            return
        # Route A (supply-priority): highest precedence; holon-global service
        # fractions applied per local-load-tier.
        service_frac = getattr(message, "service_fraction_by_sector_priority", None)
        if service_frac:
            # Coordination overhaul: when the allocation is UNCHANGED and a
            # gossip this agent originated is in flight, re-assert the per-load
            # priority floor but do NOT abandon the gossip — let it converge.
            # This cuts the dominant "yielding to L2" abandonment without staling
            # the floor (every current member, incl. newly-relevant loads, is
            # re-floored). Safe under the upward change-detection: any real
            # change moves a setpoint → triggers a fresh, changed dispatch.
            if (
                self.enable_change_only_dispatch
                and self._gossip is not None
                and self._gossip.is_originator
                and self._service_fraction_unchanged(service_frac)
            ):
                self._refresh_l2_floor(service_frac)
                return
            self._yield_to_l2_authority("service_fraction")
            self._active = True
            self.context.schedule_instant_task(
                self._dispatch_service_fractions(service_frac)
            )
            return
        # Tier-stratified override beats the scalar one: it preserves the
        # holon's priority decision the scalar would collapse.
        per_tier = getattr(message, "override_targets_by_sector_priority", None)
        if per_tier:
            # A different L2 route drifts loads away from the last service
            # fraction; invalidate so the unchanged-guard can't later match it.
            self._last_dispatched_service_fraction = None
            self._yield_to_l2_authority("per_tier")
            self._active = True
            self.context.schedule_instant_task(
                self._dispatch_per_tier_targets(per_tier)
            )
            return
        override = getattr(message, "override_target", None)
        if override is not None and math.isfinite(override):
            # L2 ADMM computed this leader's share: skip the ask-energy round
            # and use it directly as the gossip target.
            self._last_dispatched_service_fraction = None
            self._yield_to_l2_authority("override_target")
            self._active = True
            self.context.schedule_instant_task(self._start_gossip(float(override)))
            return
        self.context.schedule_instant_task(self.trigger_balance_negotiation())

    async def _dispatch_service_fractions(
        self, service_fraction: dict[str, dict[int, float]]
    ) -> None:
        """Apply a Route-A supply-priority allocation to local agents.

        ``service_fraction[sector][tier] ∈ [0, 1]`` becomes each matching load's
        regulation factor. Generators untouched; the LP routes freed supply.
        """
        # Record what we actually dispatch so an identical later allocation can
        # take the floor-refresh-only path (no gossip preemption).
        self._last_dispatched_service_fraction = service_fraction
        # This dispatch may move our own setpoint out-of-band, so invalidate the
        # upward-notify baseline: the next gossip finish must re-report rather
        # than be suppressed for re-converging to a now-stale notified value.
        self._last_notified_setpoint = None
        try:
            members = [self.context.aid]
            for neigh in self._live_neighbours():
                members.append(neigh.aid)

            applied = 0
            shed_count = 0
            served_by_sector: dict[str, float] = {}
            gen_members: list[tuple[str, Sector, float, float]] = []
            for aid in members:
                obs = self.behavior.observe(aid) or {}
                cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
                if cap <= 0:  # generator/slack source
                    # R3: collect dispatchable DGs (cap<0, non-slack) to ramp
                    # toward the served demand instead of shedding only. Slacks
                    # are grid-following (no regulation knob) and excluded.
                    if (
                        self.enable_l2_generator_ramp
                        and cap < 0
                        and lookup_slack(self.behavior, aid) is None
                    ):
                        gsec = obs_sector(obs, behavior=self.behavior, aid=aid)
                        if gsec is not None and gsec is not Sector.HEAT:
                            gsp = obs_setpoint(
                                obs, behavior=self.behavior, aid=aid
                            )
                            gen_members.append((aid, gsec, cap, gsp))
                    continue
                sec = obs_sector(obs, behavior=self.behavior, aid=aid)
                if sec is None:
                    continue
                prio = obs_priority(obs, behavior=self.behavior, aid=aid)
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
                    self.behavior,
                    aid,
                    factor,
                    sector=sec.value,
                    reason="holon_supply_priority",
                    timestamp=self.context.current_timestamp,
                    priority_tier=int(prio),
                ):
                    applied += 1

            # R3: ramp dispatchable DGs toward their sector's served demand so
            # enforcement realizes the holon-assumed supply rather than shedding
            # to the un-ramped generation level.
            if self.enable_l2_generator_ramp and gen_members:
                applied += self._ramp_member_generators(
                    gen_members, served_by_sector
                )

            if applied:
                logger.info(
                    "[%s] supply-frac dispatched: %d regulations, %d sheds, fracs=%s",
                    self.context.aid,
                    applied,
                    shed_count,
                    {
                        sec: {t: round(v, 3) for t, v in tm.items()}
                        for sec, tm in service_fraction.items()
                    },
                )
                # S1: close the L2->L1->L2 cascade. ``apply_regulate`` emits
                # nothing, so nudge the local HolonicCommunityRole to rebalance.
                # Do NOT emit NFE here: its placeholder sp would mis-trigger
                # stability and reset the leader's factor to 0. Only reached
                # when at least one setpoint actually changed (``applied`` gate)
                # so a no-op re-dispatch can't spin the cascade.
                holon_role = self.context.get_role(HolonicCommunityRole)
                if holon_role is not None:
                    try:
                        holon_role._maybe_schedule_rebalance()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "[%s] dispatch L2 re-fire skipped: %s",
                            self.context.aid,
                            exc,
                        )
        finally:
            self._active = False

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
            headrooms = [
                (aid, cap, sp, abs(cap) - abs(sp)) for aid, cap, sp in gens
            ]
            total_headroom = sum(h for *_, h in headrooms if h > 1e-9)
            if total_headroom <= 1e-9:
                continue
            for aid, cap, sp, headroom in headrooms:
                if headroom <= 1e-9:
                    continue
                share = min(headroom, deficit * (headroom / total_headroom))
                new_factor = min(1.0, (abs(sp) + share) / abs(cap))
                if apply_regulate(
                    self.behavior,
                    aid,
                    new_factor,
                    sector=sec_val,
                    reason="l2_gen_ramp",
                    timestamp=self.context.current_timestamp,
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

    def _service_fraction_unchanged(
        self, new: dict[str, dict[int, float]]
    ) -> bool:
        """True when *new* matches the last dispatched service fraction within
        ``_UPWARD_NOTIFY_TOL`` (same sectors, same tiers, values within tol)."""
        prev = self._last_dispatched_service_fraction
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
        if not self.enable_l2_priority_floor:
            return
        members = [self.context.aid]
        for neigh in self._live_neighbours():
            members.append(neigh.aid)
        for aid in members:
            obs = self.behavior.observe(aid) or {}
            cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
            if cap <= 0:  # loads only
                continue
            sec = obs_sector(obs, behavior=self.behavior, aid=aid)
            if sec is None or sec is Sector.HEAT:
                continue
            prio = obs_priority(obs, behavior=self.behavior, aid=aid)
            frac = service_fraction.get(sec.value, {}).get(prio)
            if frac is None:
                continue
            factor = max(0.0, min(1.0, float(frac)))
            factor = min(factor, constraint_allowed_fraction(obs, sec, tier=prio))
            set_l2_priority_floor(self.behavior, aid, factor)

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
            members = [self.context.aid]
            for neigh in self._live_neighbours():
                members.append(neigh.aid)

            # Group members by (sector, tier) to split each tier's target.
            per_cell_aids: dict[tuple[str, int], list[str]] = {}
            for aid in members:
                obs = self.behavior.observe(aid) or {}
                cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
                if cap <= 0:
                    continue  # generators/slacks contribute via setpoint
                prio = obs_priority(obs, behavior=self.behavior, aid=aid)
                sec = obs_sector(obs, behavior=self.behavior, aid=aid)
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
                        obs = self.behavior.observe(aid) or {}
                        caps.append(
                            abs(obs_capacity(obs, behavior=self.behavior, aid=aid))
                        )
                    total_cap = sum(caps) or 1.0
                    # Positive tgt = serve more; negative = shed.
                    for aid, cap in zip(aids, caps):
                        share = cap / total_cap
                        delta_sp = tgt * share
                        obs = self.behavior.observe(aid) or {}
                        sp_curr = obs_setpoint(obs, behavior=self.behavior, aid=aid)
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
                            self.behavior,
                            aid,
                            factor,
                            sector=sec,
                            reason="holon_tier_alloc",
                            timestamp=self.context.current_timestamp,
                            priority_tier=int(tier),
                        )
                        applied += 1

            if applied:
                logger.info(
                    "[%s] tier-alloc dispatched: %d regulations across %d cells",
                    self.context.aid,
                    applied,
                    sum(1 for tm in per_tier.values() for _ in tm),
                )
        finally:
            self._active = False

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
        self.context.schedule_instant_task(self.trigger_balance_negotiation())

    def _on_balance_problem(self, event: BalanceProblem, _src: Any) -> None:
        if event.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") == "leader":
            self.context.schedule_instant_task(self.trigger_balance_negotiation())

    def island_blackstart_shed(self) -> None:
        """Cold-load-pickup shed for a load severed into its own island.

        Fired on every failure event (see ``_add_system_behaviors``), NOT from
        the gossip path — an islanded load falls out of gossip/holon
        coordination entirely and would otherwise stay frozen at its pre-island
        setpoint, which a lone island former can't supply (so every solve is
        infeasible). When this load's node is severed from every ExtPowerGrid
        slack, drop it to zero and release the no-regret floor; recovery (ramp
        back up) is left to whatever coordination the island re-forms. No-op
        while reconnected. Electricity loads only.
        """
        if not self.enable_island_blackstart or self.sector is not Sector.ELECTRICITY:
            return
        aid = self.context.aid
        obs = self.behavior.observe(aid) or {}
        if obs_capacity(obs, behavior=self.behavior, aid=aid) <= 0:
            return  # loads only (cap > 0)
        if not _load_islanded_from_slack(
            self.behavior, aid, self.context.current_timestamp
        ):
            self._island_blackstart_active = False
            return
        self._island_blackstart_active = True
        self._restoration_floor = 0.0
        self._last_regulate_factor = 0.0
        applied = apply_regulate(
            self.behavior,
            aid,
            0.0,
            sector=self.sector.value,
            reason="island_blackstart",
            timestamp=self.context.current_timestamp,
        )
        if applied:
            record_event(
                t=self.context.current_timestamp,
                kind="island_blackstart_shed",
                aid=aid,
                sector=self.sector.value,
                detail="node severed from slack; shed to 0 for cold-load pickup",
            )

    # ------------------------------------------------------------------
    # Setpoint application with monotonic progress guarantee
    # ------------------------------------------------------------------

    def _apply_setpoint(self, new_setpoint: float) -> float | None:
        """Actuate ``new_setpoint`` (after clamp + floors); return the applied
        signed setpoint ``factor * cap``, or ``None`` when nothing is actuated
        (cap=0, tier-1 hard-lock, slack). The caller writes the delta back to
        the gossip ledger.
        """
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap == 0.0:
            return None
        # Tier-1 hard-lock guard: the pre-step already set tier-1 loads and the
        # QP gives them a_i = 0. Skip the write so the apply-on-first-visit path
        # doesn't drag them back to 0 off a stale ``starting_sp``.
        if int(self.priority) == 1:
            return None
        # Slacks have a free LP Var; writing ``regulation = sp/rating`` clamps
        # the envelope and presolves into infeasibility. The slack carries the
        # residual; gossip must not curtail it. Use a class check, not the
        # registry: the unbounded heat-side ExtHydrGrid never registers a rating.
        if _is_slack_class_child(self.behavior, self.context.aid):
            return None

        # Constraint-aware clamp near/beyond bounds. Pass the tier so critical
        # loads (tier <= 2) get the tighter 0.99 deadband.
        if self.constraint_aware:
            new_setpoint = clamp_to_constraints(
                new_setpoint, obs, self.sector, tier=self.priority
            )

        factor = max(0.0, min(1.0, abs(new_setpoint / cap)))

        # A generator the auction holds down for a live over-voltage (fresh
        # curtail-lock) must not be re-ramped by this direct-act path — it
        # bypasses ``apply_regulate``'s interlock and the gossip anchor
        # predates the curtail. CLAMP to the held level rather than defer:
        # the actuated-ledger writeback below then records the true (held)
        # contribution and marks this member saturated, so the dual
        # reallocates the shortfall. A deferral (returning None) skips the
        # writeback and leaves the requested delta on the books as phantom
        # gen supply — A/B-validated worse (loads shed against paper supply).
        if cap < 0 and has_gen_curtail_lock(
            self.behavior, self.context.aid, self.context.current_timestamp
        ):
            held = last_actuated_factor(self.behavior, self.context.aid)
            if held is not None and factor > held + 1e-6:
                record_event(
                    t=self.context.current_timestamp,
                    kind="gossip_gen_clamped_to_curtail_lock",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=f"requested_factor={factor:.4f} held={held:.4f}",
                )
                factor = held

        # Island black-start (the shed itself is event-driven — see
        # ``island_blackstart_shed`` — because an islanded load drops out of
        # gossip and this path never runs for it). Here we only CLEAR the hold
        # once the load is coordinated again while reconnected, so the no-regret
        # floor re-engages for normal restoration.
        if (
            self.enable_island_blackstart
            and self._island_blackstart_active
            and self.sector is Sector.ELECTRICITY
            and not _load_islanded_from_slack(
                self.behavior, self.context.aid, self.context.current_timestamp
            )
        ):
            self._island_blackstart_active = False

        # No-regret floor applies only during restoration (target > 0); shedding
        # (target < 0) legitimately reduces factor.
        target = self._gossip.target if self._gossip is not None else 0.0
        is_restoration = target > 0
        if self.priority > 0 and is_restoration:
            self._check_violation_cleared()

            if self.enable_monotonic_floor:
                if factor > self._restoration_floor:
                    self._restoration_floor = factor
                elif (
                    not self._constraint_violation_active
                    and not self._island_blackstart_active
                ):
                    factor = self._restoration_floor

            # CLPU rate limit: ramp-up only; decreases pass through.
            if self.enable_clpu_ramp:
                factor = self._rate_limit_increase(factor)

        # L2 priority-floor: the ADMM set this load's served tier, so a
        # supply-poor group must not shed below ``min(L2 alloc, constraint-
        # allowed)`` just to zero its own imbalance. Tiers 2/3/4 only.
        if self.enable_l2_priority_floor:
            floor = l2_effective_floor(
                self.behavior, self.context.aid, obs, self.sector, self.priority
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
            getattr(self.behavior, "_scare_config", None),
            "enable_line_congestion_price",
            False,
        ):
            ceiling = line_congestion_ceiling(
                self.behavior,
                self.context.aid,
                self.context.current_timestamp,
                _LINE_CONGESTION_TTL_S,
            )
            if ceiling < factor:
                factor = ceiling

        # Gossip regulates bypass the ``apply_regulate`` dedup on purpose: the
        # ledger advances regardless, so dedupping micro-steps would diverge it
        # from physical state and stall at k_max. Warm-start absorbs the deltas.
        if self.behavior.has_action(self.context.aid, "regulate"):
            self.behavior.act(self.context.aid, "regulate", factor)
            # Keep the dedup cache truthful: a stale cache would drop a later L2
            # re-dispatch that restores this load.
            note_actuated_factor(self.behavior, self.context.aid, factor)
            record_regulate(
                t=self.context.current_timestamp,
                aid=self.context.aid,
                sector=self.sector.value,
                factor=factor,
                reason="balance",
            )
            self._last_regulate_timestamp = self.context.current_timestamp
            self._last_regulate_factor = factor

        # Signed actuated setpoint: ``factor * cap`` is the realised power after
        # clamp + floors. The caller reconciles the ledger against it.
        return factor * cap

    def _writeback_actuated_delta(
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
            not self.enable_actuated_ledger_writeback
            or applied_sp is None
            or self._gossip is None
        ):
            return
        actuated_delta = applied_sp - self._gossip.starting_setpoint
        # Held below the requested magnitude ⇒ constraint-bound.
        held_below = abs(actuated_delta) < abs(requested_delta) - 1e-12
        saturated = held_below or _is_saturated(actuated_delta, dmin, dmax)
        self._gossip.memory[self_key] = (
            actuated_delta,
            counter,
            self.priority,
            saturated,
        )
        self._gossip.current_delta = actuated_delta

    def _rate_limit_increase(self, requested: float) -> float:
        prev = self._last_regulate_factor
        if requested <= prev:
            return requested
        last_t = self._last_regulate_timestamp
        if last_t is None:
            return requested
        dt = max(0.0, self.context.current_timestamp - last_t)
        max_delta = self._clpu_ramp_per_s * dt
        return min(requested, prev + max_delta)

    def _try_self_dispatch(self, deficit: float) -> None:
        """Inline local-gen fallback for isolated agents: if a generator with
        headroom, ramp up to cover as much of the deficit as possible.
        """
        if deficit <= 0:
            return
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap >= 0:
            return  # not a generator
        # Curtail-vs-ramp interlock is enforced in ``apply_regulate``: a write
        # to a generator the auction holds down for a live violation defers there.
        sp = obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        headroom = abs(cap) - abs(sp)
        if headroom < 1e-6:
            return
        share = min(headroom, deficit)
        new_factor = min(1.0, (abs(sp) + share) / abs(cap))
        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            new_factor,
            sector=self.sector.value,
            reason="self_local_gen",
            timestamp=self.context.current_timestamp,
        )
        if applied:
            # Out-of-band self actuation — invalidate the upward-notify baseline.
            self._last_notified_setpoint = None
            logger.info(
                "[%s] self local-gen: ramped to %.1f%% (deficit=%.4f)",
                self.context.aid,
                new_factor * 100,
                deficit,
            )


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
    enable_monotonic_floor: bool = True,
    enable_clpu_ramp: bool = True,
    enable_island_blackstart: bool = False,
    termination_tolerance: float = 1e-5,
    max_hops: int = _MAX_HOPS,
    step_decay_k0: int = _STEP_DECAY_K0_DEFAULT,
    enable_qp_gossip: bool = True,
    enable_l2_generator_ramp: bool = True,
    enable_change_only_dispatch: bool = True,
    enable_l2_priority_floor: bool = True,
    enable_actuated_ledger_writeback: bool = True,
    enable_nominal_slack_supply: bool = False,
    enable_cp_aware_slack_supply: bool = False,
    enable_heat_l2_dispatch: bool = False,
    component_scope: bool = False,
) -> EnergyBalanceNegotiator:
    if priority is None:
        priority = obs_priority(obs)
    return EnergyBalanceNegotiator(
        behavior=behavior,
        sector=sector,
        priority=priority,
        constraint_aware=constraint_aware,
        enable_monotonic_floor=enable_monotonic_floor,
        enable_clpu_ramp=enable_clpu_ramp,
        enable_island_blackstart=enable_island_blackstart,
        termination_tolerance=termination_tolerance,
        max_hops=max_hops,
        step_decay_k0=step_decay_k0,
        enable_qp_gossip=enable_qp_gossip,
        enable_l2_generator_ramp=enable_l2_generator_ramp,
        enable_change_only_dispatch=enable_change_only_dispatch,
        enable_l2_priority_floor=enable_l2_priority_floor,
        enable_actuated_ledger_writeback=enable_actuated_ledger_writeback,
        enable_nominal_slack_supply=enable_nominal_slack_supply,
        enable_cp_aware_slack_supply=enable_cp_aware_slack_supply,
        enable_heat_l2_dispatch=enable_heat_l2_dispatch,
        component_scope=component_scope,
    )
