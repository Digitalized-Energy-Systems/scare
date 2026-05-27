from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
    topology_neighbors,
)
from monee.model.child import ExtHydrGrid, ExtPowerGrid

from scare.base.diagnostics import record_event, record_negotiation, record_regulate
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
    LocalGenerationApproval,
    LocalGenerationRequest,
    NegotiationFinishedEvent,
    ResponseEnergyMessage,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.trust import TrustLedger, TrustParams, hash_weighted_choice
from scare.base.util import (
    apply_regulate,
    clamp_to_constraints,
    constraint_utilization,
    l2_effective_floor,
    lookup_slack,
    note_actuated_factor,
    obs_capacity,
    obs_min_max,
    obs_priority,
    obs_sector,
    obs_setpoint,
    tier_priority_weight,
)
from scare.community.holonic import HolonicCommunityRole

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

_DEFAULT_START_THRESHOLD = 1e-4
# Sector-specific overrides of ``_DEFAULT_START_THRESHOLD``.  All sectors
# currently share the same threshold (MW magnitudes).
_START_THRESHOLD: dict[Sector, float] = {}

# Per-group threshold = max(_THRESHOLD_ABS_FLOOR,
# _THRESHOLD_CAPACITY_FRACTION · Σ|cap|).  Scales noise tolerance with
# group capacity so we reject sub-half-percent imbalances.
_THRESHOLD_CAPACITY_FRACTION: float = 0.005
_THRESHOLD_ABS_FLOOR: float = 1e-6

# Heat utilization above which an agent contributes headroom to the
# group's thermal-deficit target.  Just below the 0.85 proactive
# warning so warmup/cooldown triggers gossip pre-violation.
_HEAT_CLEAR_FRACTION: float = 0.6

_MAX_HOPS = 100

# Robbins-Monro step decay: gain = gamma_s / (1 + k / k0).  Satisfies
# Σ γ_k = ∞, Σ γ_k² < ∞.  k0 ≈ typical simbench LV group size.
_STEP_DECAY_K0_DEFAULT: int = 20

# P2 stall detection.  When the recent gap window's range falls below
# _STALL_TOL_FRACTION · |T| and the gap still exceeds the per-group
# threshold, declare stuck and emit LocalGenerationRequest.
_STALL_WINDOW_FACTOR: int = 2
_STALL_TOL_FRACTION: float = 0.005
_STALL_TOL_FLOOR: float = 1e-6

# Base wallclock deadline per sector (heat is slowest due to high
# decision delay).  Scaled further by group size at use site.
_GOSSIP_TIMEOUT_BASE_S: dict[Sector, float] = {
    Sector.ELECTRICITY: 5.0,
    Sector.GAS: 15.0,
    Sector.HEAT: 30.0,
}
_GOSSIP_TIMEOUT_DEFAULT_S = 15.0
_GOSSIP_TIMEOUT_PER_AGENT_S = 0.5

# Stale-neighbour pruning: how many poll periods of silence count as
# dead.  Heat tolerates long gaps; electricity does not.
_HEARTBEAT_MAX_AGE_MULTIPLE: float = 8.0

# Intra-sector priority tiers (lower = higher urgency, gossips earlier).
# 4-tier model: tier 1 = critical (hard-locked at the leader pre-step
# before the QP runs), tiers 2–4 = QP-weighted with steep exponents
# (1e8 / 1e4 / 1.0) so the proportional equilibrium is effectively
# strict.  See ``scare.base.util.tier_priority_weight`` for the
# canonical schedule shared across L1 / L2 / L3.
_PRIORITY_TIERS = 4

# Byzantine cap: a single participant's delta is clipped to this
# multiple of the negotiation target magnitude.
_BYZANTINE_DELTA_CAP_MULTIPLE: float = 5.0


def _start_threshold(sector: Sector) -> float:
    return _START_THRESHOLD.get(sector, _DEFAULT_START_THRESHOLD)


def _is_slack_class_child(behavior: Any, aid: str) -> bool:
    """True iff *aid* refers to a monee ``ExtPowerGrid`` or ``ExtHydrGrid``
    child — the network's slack-class boundary.

    Used to suppress regulation writes on slacks; writing
    ``regulation < 1`` clamps the LP's effective slack envelope and the
    next energy-flow solve goes infeasible.  Covers both bounded
    (registered) and unbounded (heat-side) slacks uniformly.
    """
    if not aid.startswith("child-"):
        return False
    try:
        cid = int(aid[len("child-"):])
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
    """Return MW of demand reduction this heat agent should contribute to
    its group's thermal-deficit target.

    ``max(0, util - ϑ_clear) · |cap|`` with the dominant local
    constraint utilization.  Only loads (cap > 0) contribute; heat
    generators are handled by the local-generation fallback.  Caller must
    restrict invocation to heat sector.
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
    """True iff *delta* sits within numerical tolerance of either box bound.

    Tolerance scales with box magnitude (``1e-9 + 1e-6 · max(|dmin|, |dmax|, 1)``)
    so large-box problems don't reject near-bound primal values as
    unsaturated due to ADMM solver noise.
    """
    sat_tol = 1e-9 + 1e-6 * max(abs(dmin), abs(dmax), 1.0)
    return delta <= dmin + sat_tol or delta >= dmax - sat_tol


def _deterministic_next(neighbours: list, negotiation_id: str, counter: int) -> Any:
    """Select the next gossip target deterministically.

    Uses a hash of (negotiation_id, counter) to pick a neighbour index.
    This replaces ``random.choice`` and ensures that two agents competing
    for the same resource always send in the same order, which is a
    prerequisite for deterministic conflict resolution.
    """
    if not neighbours:
        return None
    h = hashlib.sha256(f"{negotiation_id}:{counter}".encode()).digest()
    idx = int.from_bytes(h[:4], "big") % len(neighbours)
    return neighbours[idx]


def _deterministic_sub_round(
    agent_addr: str, negotiation_id: str, tier: int, tier_size: int
) -> int:
    """Compute a deterministic sub-round index for intra-tier ordering.

    Agents within the same priority tier are serialized: each gets a
    unique sub-round index in [0, tier_size).  The index is stable for
    a given (agent, negotiation, tier) triple.
    """
    h = hashlib.sha256(
        f"{negotiation_id}:{tier}:{agent_addr}".encode()
    ).digest()
    return int.from_bytes(h[:4], "big") % tier_size


@dataclass
class _GossipState:
    negotiation_id: str
    target: float
    counter: int
    current_delta: float
    starting_setpoint: float
    # Feasible-δ box, anchored to the *starting* setpoint of the
    # negotiation.  Recomputing dmin/dmax from ``obs_min_max`` on every
    # primal step creates a chicken-and-egg loop: when the agent's own
    # regulate from the previous step flips the LP's reported sp from
    # cap → 0, the box flips from [−cap, 0] → [0, cap]; the next primal
    # step then computes ``δ ≈ 0`` (clamped to the new dmax-from-above
    # side), apply_setpoint(starting + 0) = starting → factor=1.0; sp
    # flips back to cap, box flips back, δ=−cap → factor=0.  Result is
    # a 40 ms-period oscillation between full-shed and full-load, with
    # net gossip progress = 0.  Confirmed root cause of child-194's
    # bang-bang behaviour on gas in task-0 simbench_lv.  Anchoring to
    # the starting state turns δ into a true cumulative change and
    # keeps the per-step box constant across the negotiation.
    dmin_starting: float = 0.0
    dmax_starting: float = 0.0
    # Shared ledger of per-agent contributions, merged across received
    # messages by taking the entry with the highest counter per agent.
    # This replaces the aggregate digest that double-counted in cycles.
    # addr_str -> (delta, counter_when_set, priority, saturated_flag).
    # The saturated flag (P1) is True when the agent's last applied
    # delta hit dmin or dmax within tolerance; ``_n_free()`` excludes
    # such entries from the equal-share denominator so the per-visit
    # contraction does not collapse as the boundary set grows.
    memory: dict[str, tuple[float, int, int, bool]] = field(default_factory=dict)
    # True only for the negotiator that *originated* the gossip via
    # ``_start_gossip``.  Peers that receive a gossip message and create
    # a local state from it set this False.  Used by the diagnostics
    # diary so terminal events ("finished", "timed_out", "cancelled",
    # "abandoned", "stalled") are recorded once per nid (by the
    # originator), preserving the ``started == Σ terminals`` invariant.
    is_originator: bool = False
    # P2: rolling window of post-update gap values for stall detection.
    # Sized at ``_STALL_WINDOW_FACTOR · n_active`` rounds; when the
    # range across the window is below threshold and the gap still
    # exceeds the per-group threshold, the originator emits
    # LocalGenerationRequest and finishes with terminal "stalled".
    gap_window: list[float] = field(default_factory=list)
    # P6: scalar dual variable for the primal-dual QP gossip.  At the
    # KKT optimum, ``dual_lambda`` equals the unique scarcity price
    # such that ``Σ clamp(a_i · λ, dmin_i, dmax_i) = T``.  Gossiped
    # along with the ledger; the receiving agent does both a primal
    # update (its own ``δ_i = clamp(a_i · λ, dmin_i, dmax_i)``) and a
    # dual update (``λ ← λ + γ_k · (T − Σ_a Δ_a)``).
    dual_lambda: float = 0.0


class EnergyBalanceNegotiator(Role):
    """Gossip-based energy balance negotiation with:

    - **Priority-ordered participation**: agents join the gossip at a
      round determined by their priority tier, ensuring high-priority
      loads are restored first (or shed last).
    - **Monotonic progress**: once a load's regulation factor has been
      increased during a restoration (target > 0), it will not be
      decreased below that floor unless a hard constraint violation
      forces it (improvements.txt §5 MUST "no-regret switching").
    - **Deterministic conflict resolution**: gossip forwarding uses a
      hash-based deterministic selector instead of ``random.choice``,
      and when two agents compete for the same headroom the
      higher-priority agent wins (improvements.txt §5 MUST "conflict
      resolution mechanism").
    - **Constraint-aware clamping**: setpoints are clamped using the
      local constraint utilization, so agents near voltage/pressure/
      temperature bounds automatically contribute less.
    - **Sector time-scale awareness**: convergence rate defaults to the
      sector-specific value from ``SECTOR_TIMESCALE``.
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
        max_hops: int = _MAX_HOPS,
        step_decay_k0: int = _STEP_DECAY_K0_DEFAULT,
        enable_qp_gossip: bool = True,
        enable_l2_priority_floor: bool = True,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.priority = priority
        self.impact_weight = impact_weight
        self.termination_tolerance = termination_tolerance
        self.constraint_aware = constraint_aware
        self.enable_monotonic_floor = enable_monotonic_floor
        self.enable_clpu_ramp = enable_clpu_ramp
        # L2 priority-floor: clamp gossip ``balance`` sheds up to the
        # component ADMM's per-load allocation (relaxed by local
        # constraints).  Blocks the L2→L1 override that inverts tiers
        # (eval task-88); see ``_apply_setpoint``.
        self.enable_l2_priority_floor = enable_l2_priority_floor
        self.max_hops = max_hops
        self.step_decay_k0 = max(1, int(step_decay_k0))
        # P6: when True, run the primal-dual QP gossip; when False the
        # historical equal-share update is used.  Both routes share the
        # same ledger, Byzantine cap, heartbeat liveness, deterministic
        # next-hop, P1 saturation flag, P2 stall detection, P3 step
        # decay, and termination logic — only the per-agent update rule
        # differs.  Exposed as an ablation flag so the evaluation
        # harness can run head-to-head comparisons.
        self.enable_qp_gossip = enable_qp_gossip

        # Apply sector-specific convergence rate if not overridden.
        ts = SECTOR_TIMESCALE.get(sector, {})
        self.convergence_rate = (
            convergence_rate if convergence_rate is not None
            else ts.get("convergence_rate", 0.5)
        )

        self._active: bool = False
        self._gossip: _GossipState | None = None
        # State for the setpoint-gathering phase before gossip starts
        self._trigger_nid: str | None = None
        self._trigger_responses: dict[str, float] = {}
        self._trigger_expected: int = 0

        # Total absolute capacity across this leader's group (refreshed
        # each ``trigger_balance_negotiation``).  Drives the per-group
        # threshold so noise scales with group size.
        self._group_capacity_abs: float = 0.0

        # --- Monotonic progress floor ---
        # Tracks the highest regulation factor this agent has applied
        # during restoration.  The factor may only decrease if a hard
        # constraint violation is active.
        self._restoration_floor: float = 0.0
        self._constraint_violation_active: bool = False

        # --- Cold-load pickup rate limiter ---
        # Post-outage inrush is 2-6x steady state for the first minutes.
        # Loads cap how fast their regulation factor can grow; decreases
        # (shedding) are unrestricted.  Ramp rate scales with sector
        # convergence_rate so heat (slow) ramps gentler than electricity.
        self._last_regulate_timestamp: float | None = None
        self._last_regulate_factor: float = 0.0
        self._clpu_ramp_per_s: float = self.convergence_rate

        # --- Neighbour liveness (heartbeat) ---
        # Maps str(neighbour_addr) -> timestamp of last inbound message.
        # A neighbour absent for longer than HEARTBEAT_MAX_AGE is pruned
        # from gossip forwarding.  Bootstrap: address enters the map the
        # first time we attempt contact (touch), so an unresponsive node
        # ages out rather than remaining ghost-alive forever.
        self._neighbour_last_seen: dict[str, float] = {}
        poll = ts.get("poll_period_s", 1.0)
        self._heartbeat_max_age_s: float = poll * _HEARTBEAT_MAX_AGE_MULTIPLE

        # --- B.1: continuous coupling weights K_ij(t) ---
        # Each known neighbour carries a continuous trust score in [0, 1]
        # that biases gossip forwarding and tightens the liveness gate.
        # Refined heartbeat: the binary "have we heard recently?" check
        # is replaced by K >= liveness_threshold.  Decay rate scales with
        # the sector poll period so heat (slow polling) doesn't pessimise
        # K too aggressively.
        self._trust = TrustLedger(
            TrustParams(
                decay_rate_per_s=1.0 / max(poll * _HEARTBEAT_MAX_AGE_MULTIPLE, 1.0),
                recover_rate=0.6,
                liveness_threshold=0.5,
                initial=1.0,
            )
        )

        # --- Local proactive constraint utilization ---
        # Populated by the co-located GridConstraintMonitor's
        # ConstraintWarning events.  Keyed by variable name; value is the
        # last-reported utilization in [0, 1].  Used to throttle the
        # gossip step so an agent close to a hard bound contributes less.
        self._proactive_util: dict[str, float] = {}

    def setup(self) -> None:
        # Register this aid as "gossip-capable" so other group members
        # can route ``EnergyNegotiationMessage`` only to peers that will
        # actually process it.  The registry lives on the shared
        # ``behavior`` (sector → set of aids) and is consulted by
        # ``_gossip_neighbours``.  PowerLine branch agents are joined
        # to the same electricity groups for flex-query and line-
        # overload-relief routing, but they do not have an
        # EnergyBalanceNegotiator — without this filter the gossip
        # token's deterministic next-hop frequently forwarded to a
        # branch monitor that silently dropped the message, killing
        # the gossip after one or two hops (see scenario-trace audit:
        # ~60 % of failed gossips were token deaths at branches).
        store = getattr(self.behavior, "_scare_gossip_capable", None)
        if store is None:
            store = {}
            self.behavior._scare_gossip_capable = store
        store.setdefault(self.sector, set()).add(self.context.aid)

        # Mango's handle_message dispatches synchronously, so async handlers
        # must be wrapped to schedule themselves via the agent scheduler.
        # This ensures the simulation's termination detection can track them.
        # Every inbound message also stamps the sender's heartbeat so
        # neighbour liveness tracking stays up to date without explicit
        # heartbeats.
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
        # Mango requires at least one local subscriber per emitted event
        # type.  LocalGenerationFallbackRole is attached only to group
        # leaders, so non-leader members that hit the singleton-fallback
        # path would crash mango.emit_event without this no-op safety
        # net.  The actual fallback logic still lives on the leader;
        # this handler just satisfies the dispatch path.
        self.context.subscribe_event(
            self, LocalGenerationApproval, self._on_local_gen_approval_noop
        )
        # Same defensive pattern for NegotiationFinishedEvent: in the
        # production scenario every child also hosts ``GenerationController``
        # which subscribes to it, but in unit tests / minimal compositions
        # the negotiator may be the only role on the agent.  P2's early
        # stall termination can fire ``_finish_negotiation`` before any
        # external listener is wired, so a noop here keeps the dispatch
        # path safe without changing production behaviour.
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_finished_noop
        )

    # ------------------------------------------------------------------
    # Constraint violation tracking (for monotonic progress override)
    # ------------------------------------------------------------------

    def _on_local_gen_approval_noop(
        self, _event: LocalGenerationApproval, _src: Any
    ) -> None:
        # Intentionally empty — see setup() for the rationale.  The
        # leader's LocalGenerationFallbackRole handles the actual
        # response.
        return

    def _on_finished_noop(
        self, _event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        # Intentionally empty — production runs always have a
        # GenerationController subscribing to NegotiationFinishedEvent
        # and acting on it; this noop keeps mango's dispatch path safe
        # in minimal compositions where no external listener is wired.
        return

    def _on_constraint_warning(self, event: ConstraintWarning, _src: Any) -> None:
        # Co-located monitor reports proximity to a bound; record the
        # latest utilization so the gossip step can throttle this agent's
        # contribution.  Other-sector warnings are ignored — sector
        # coupling is handled at the holon / CP level, not in gossip.
        if event.sector != self.sector:
            return
        self._proactive_util[event.variable] = float(event.utilization)

    def _on_constraint_violation(self, event: ConstraintViolation, _src: Any) -> None:
        if event.sector == self.sector:
            self._constraint_violation_active = True
            # Cancel any active gossip: the constraint landscape has changed,
            # so continuing to converge on a stale target is wasteful and may
            # push the system further into violation.  The BalanceProblem
            # event emitted by GridConstraintMonitor will trigger a fresh
            # negotiation that incorporates updated constraint utilization.
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
        """Clear the violation flag if the co-located GridConstraintMonitor
        reports all local constraints are satisfied again."""
        if not self._constraint_violation_active:
            return
        monitor = self._find_constraint_monitor()
        if monitor is not None and monitor.is_locally_feasible():
            self._constraint_violation_active = False

    def _worst_neighbour_utilization(self) -> float:
        """Peek at the co-located GridConstraintMonitor (if present) for the
        worst utilization reported by any 1-N hop neighbour."""
        monitor = self._find_constraint_monitor()
        return monitor.worst_neighbour_utilization() if monitor is not None else 0.0

    def _gossip_total_delta(self) -> float:
        """``Σ δ_i`` across all participants in the active gossip ledger
        (``0`` when no gossip is active)."""
        if self._gossip is None:
            return 0.0
        return sum(v[0] for v in self._gossip.memory.values())

    def _compute_participation_scale(self, obs: dict) -> float:
        """Constraint-aware throttle ``∈ [0, 1]`` blending local
        utilization, worst-neighbour utilization, and proactive
        warnings.  Heat is exempt: thermal violations want stressed
        loads to shed (handled by priority + clamp), not throttle.
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
        from scare.service.constraints import GridConstraintMonitor

        for role in getattr(self.context, "roles", []):
            if isinstance(role, GridConstraintMonitor) and role.sector == self.sector:
                return role
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
        # B.1: every received message recovers the K-score multiplicatively.
        self._trust.on_message_received(key, now)

    def _touch_neighbours(self, addrs: list) -> None:
        """Seed the heartbeat clock for neighbours we have just contacted.

        Keeps an unresponsive node from aging out immediately (grace
        period equal to the heartbeat timeout) while ensuring that if it
        never replies, it will still be pruned on the next round.
        """
        now = self.context.current_timestamp
        for addr in addrs:
            key = str(addr)
            if key not in self._neighbour_last_seen:
                self._neighbour_last_seen[key] = now

    def _update_gap_window_and_check_stall(
        self, open_gap: float, target: float
    ) -> bool:
        """P2: append the post-update gap to the rolling window and
        decide whether the protocol has stalled.

        A stall is declared when (a) the run is past warm-up, (b) the
        window is full, (c) the max-min range across the window is
        below ``max(_STALL_TOL_FRACTION · |T|, _STALL_TOL_FLOOR)``,
        and (d) the current gap still exceeds the per-group threshold.

        Warm-up = ``_PRIORITY_TIERS + 1 + window_size`` rounds.  This
        accounts for the priority-gating delay (generators wait until
        counter $\\ge P+1$ during restoration) plus a full window's
        worth of post-warmup gap samples.  Without it the stall
        detector would fire during the silence before the lowest-tier
        agents are eligible to act.
        """
        if self._gossip is None:
            return False
        active = max(1, len(self._gossip.memory))
        window_size = _STALL_WINDOW_FACTOR * active
        win = self._gossip.gap_window
        win.append(open_gap)
        if len(win) > window_size:
            del win[0]
        # Warm-up gate: priority tiers and sub-round serialisation can
        # silence many early rounds; don't decide on stall until the
        # protocol has had a fair chance to converge.
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
        """P2: terminate a stalled gossip and escalate to the local-
        generation fallback.

        Only the originator records the ``stalled`` diary terminal so
        the ``started == Σ terminals`` invariant remains exact.
        Emits LocalGenerationRequest with the residual deficit if this
        agent is the group leader (the same gate as in
        ``_finish_negotiation``); ``_finish_negotiation`` is then
        called with ``record_finished=False`` so it does not double-
        count a ``finished`` terminal.
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
        # Suppress the "finished" diary entry — this terminal is "stalled"
        await self._finish_negotiation(record_finished=False)

    # ------------------------------------------------------------------
    # P6: primal-dual QP helpers
    # ------------------------------------------------------------------

    def _qp_priority_weight(self, target_sign: int) -> float:
        """Priority cost weight for the QP responsiveness ``a_i``.

        Delegates to ``tier_priority_weight`` (the single source of
        truth for the 4-tier schedule).  Tier 1 is hard-locked at the
        leader pre-step, so this returns the defensive weight 1.0 for
        tier-1 entries that might still reach the QP.  Tiers 2–4 get
        the 1e8 / 1e4 / 1.0 schedule.  Generators always return 1.0 in
        either direction — the existing primal-clamp sign handles
        their participation symmetrically.
        """
        return tier_priority_weight(
            self.priority,
            regime=int(target_sign),
            priority_tiers=_PRIORITY_TIERS,
        )

    def _qp_responsiveness(self, _cap: float, target_sign: int) -> float:
        """Per-agent QP coefficient ``a_i = w_i`` (priority weight).

        Capacity does not enter ``a_i`` directly — it enters the
        formulation through the box constraints
        ``[δ_min, δ_max]`` instead.  Two agents with the same priority
        but different capacities will move identical ``δ`` values per
        unit of ``λ``; the smaller-capacity agent saturates earlier
        because its box is smaller, which is the correct waterfall
        behaviour.  Putting capacity-squared in ``a_i`` would make
        the dual scale wildly with grid size and inflate ``λ`` by
        orders of magnitude for no convergence benefit.
        """
        return self._qp_priority_weight(target_sign)

    def _qp_primal(
        self,
        a_i: float,
        lam: float,
        dmin: float,
        dmax: float,
        _target_sign: int,
        _cap: float,
    ) -> float:
        """Closed-form primal update: ``δ_i = clamp(a_i · λ, dmin, dmax)``.

        The sign of ``λ`` matches the sign of the target ``T``, so for
        restoration (``T > 0``) ``λ`` rises from 0 to ``λ* = T/Σa_j``
        and pushes every agent's ``δ`` positive (loads up, generators
        shed); for curtailment (``T < 0``) ``λ`` falls below 0 and
        pushes every agent's ``δ`` negative (loads shed, generators
        ramp up).  Box clamping enforces feasibility unconditionally.
        """
        return max(dmin, min(dmax, a_i * lam))

    def _compute_lambda_seed(self, target: float, n_neighbours: int) -> float:
        """Seed λ so the originator's first primal step makes meaningful
        progress while leaving the dual update room to correct.

        Aims the originator's first-step δ at its fair share
        ``target / n_seed``, giving ``λ₀ = target / (n_seed · a_self)``.
        Clamped to ``|target|`` so pathological tier combinations cannot
        inject an unbounded first step.
        """
        target_sign = 1 if target > 0 else (-1 if target < 0 else 0)
        a_self = max(self._qp_priority_weight(target_sign), 1.0)
        n_seed = max(2, n_neighbours + 1)
        lambda_seed = target / (n_seed * a_self)
        return max(-abs(target), min(abs(target), lambda_seed))

    def _entry_responsiveness(self, prio: int, target_sign: int) -> float:
        """``a_i`` from a ledger entry's stored priority — used by the
        receiver to estimate ``Σ a_j`` for dual-step normalisation.

        Mirrors ``_qp_priority_weight`` exactly (same schedule, same
        source of truth) so the dual update agrees with each agent's
        own primal step.
        """
        return tier_priority_weight(
            int(prio),
            regime=int(target_sign),
            priority_tiers=_PRIORITY_TIERS,
        )

    def _step_size(self, counter: int) -> float:
        """Robbins-Monro diminishing step (P3).

        Returns ``gamma_s / (1 + k / k0)``.  Satisfies
        ``Σ γ_k = ∞`` and ``Σ γ_k² < ∞`` so the gossip dynamics
        converge almost surely under bounded-variance noise.  At
        ``counter = 0`` this exactly matches the historical
        constant-step behaviour, so cold-start dynamics are unchanged.
        """
        return self.convergence_rate / (1.0 + max(0, counter) / self.step_decay_k0)

    def _per_group_threshold(self) -> float:
        """Per-negotiation threshold scaled by the leader's snapshot of
        total group load capacity.  Falls back to the sector default
        floor only when the leader has no capacity information yet
        (e.g. before the first ``trigger_balance_negotiation``).
        """
        if self._group_capacity_abs <= 0.0:
            return _start_threshold(self.sector)
        return max(
            _THRESHOLD_ABS_FLOOR,
            _THRESHOLD_CAPACITY_FRACTION * self._group_capacity_abs,
        )

    def _live_neighbours(self) -> list:
        """Return group neighbours whose continuous trust score K_ij is
        above the liveness threshold (B.1).

        Bootstraps unknown neighbours optimistically (initial K = 1.0 in
        the ``TrustLedger``), so first contact always succeeds; recovery
        is multiplicative on every received message and decay is linear
        in the silence interval scaled by the sector poll period.

        Includes *every* live group neighbour — used for the flex-query
        round (``AskEnergyMessage``) which branch agents reply to via
        their stub.  For gossip token routing, use ``_gossip_neighbours``
        instead so the token only travels among peers that subscribe to
        ``EnergyNegotiationMessage``.
        """
        all_neighbours = topology_neighbors(self, tid="groups")
        now = self.context.current_timestamp
        return [a for a in all_neighbours if self._trust.is_live(str(a), now)]

    def _gossip_neighbours(self) -> list:
        """Live group neighbours that have an ``EnergyBalanceNegotiator``
        of the same sector — i.e. agents that will actually process an
        ``EnergyNegotiationMessage``.

        PowerLine branch agents (line-loading-relief feature) are joined
        to the electricity groups topology for the flex-query and
        overload-relief routing they participate in, but they have a
        ``GridConstraintMonitor`` in branch mode only — no
        ``EnergyBalanceNegotiator``, so the gossip protocol dies on a
        first-hop forward to them.  Filtering on registered gossip-
        capable aids (see ``setup()``) keeps them in the community for
        flex purposes while excluding them from token routing.
        """
        store = getattr(self.behavior, "_scare_gossip_capable", {})
        capable = store.get(self.sector, set())
        return [a for a in self._live_neighbours() if a.aid in capable]

    def _scored_neighbours(self, neighbours: list) -> list[float]:
        """Return the K-score for each neighbour in ``neighbours`` order."""
        now = self.context.current_timestamp
        return [self._trust.score(str(a), now) for a in neighbours]

    def _next_hop(self, neighbours: list, nid: str, counter: int):
        """B.1: K-weighted deterministic next-hop selection.

        Generalises ``_deterministic_next`` by picking the index in
        proportion to the trust score K_ij.  Reduces to the uniform
        SHA256 modulo when all K's are equal, so behaviour matches the
        legacy path under healthy conditions.  When some neighbours
        have low K (recent silence, partial outage), the gossip routes
        around them automatically.
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
        if self._active:
            return
        self._active = True

        neighbours = self._live_neighbours()
        self._touch_neighbours(neighbours)
        # Snapshot the group's absolute capacity so the threshold for
        # this negotiation scales with what is physically there.  Loads
        # only (cap > 0): the threshold gates demand-side imbalance, and
        # large generators in a load-light group shouldn't relax it.
        # During restoration with target<0 (curtailment) this is also
        # the right denominator — we want to ignore noise relative to
        # the demand we are protecting.
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

        nid = str(uuid4())
        self._trigger_nid = nid
        self._trigger_responses = {}
        self._trigger_expected = len(neighbours)

        msg = AskEnergyMessage(negotiation_id=nid, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        sp = self._reported_setpoint(obs)
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=sp,
            available=cap - sp,  # headroom, not total capacity
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    def _reported_setpoint(self, obs: dict) -> float:
        """Setpoint contribution to the group's negotiation target.

        For electricity / gas this is the raw setpoint (Σ s_i ≈ 0
        ⇒ balanced).  For heat the setpoint is amplified by this
        agent's local thermal deficit so a stressed group surfaces a
        non-zero negative target — the gossip then sheds load via the
        existing reverse-priority rules in ``_compute_actual_priority``.

        F2 — Slack target:  if this agent is registered as a slack
        (``register_slack``), report the *target infeed* rather than
        the LP's current operating point.  The target is
        ``slack_target_fraction · rating`` in load convention (positive
        when the operator wants the slack to import).  This shifts the
        gossip's imbalance accounting from "everything balances to
        zero" (which the LP achieves trivially) to "the rest of the
        group should balance such that the slack only draws its
        target" — which is the real operator objective.

        NB: gossip is per-community, but the slack budget is a global
        property of the connected component.  The gossip target
        derived from this setpoint only matches the operator's
        intent when the slack's community spans (most of) the
        component — which is only true for ``component_level``.
        For ``single_level`` and the holonic ``scare`` variant the
        community is small and the gossip target derived here is
        incoherent with the global budget.  Budget enforcement for
        those variants happens via
        :class:`~scare.service.slack_budget.SlackBudgetMonitor` 's
        signed ``override_target`` (the over-budget magnitude in
        gossip-target convention), not via the per-community
        ``-total_sp`` target.
        """
        slack = lookup_slack(self.behavior, self.context.aid)
        if slack is not None:
            cfg = getattr(self.behavior, "_scare_config", None)
            fraction = float(
                getattr(cfg, "slack_target_fraction", 0.0) if cfg is not None else 0.0
            )
            # ``slack.cap`` is generator-convention (negative); the
            # *import* target is the positive of that magnitude.  In
            # load convention an importing slack has positive p_mw.
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
            own_obs = self.behavior.observe(self.context.aid) or {}
            total_sp = (
                self._reported_setpoint(own_obs)
                + sum(self._trigger_responses.values())
            )
            self._trigger_nid = None
            self._trigger_responses = {}

            # Tier-1 hard-constraint pre-step.  Apply ``regulation = 1`` to
            # every tier-1 load if the community's generator pool can
            # cover the total tier-1 demand; otherwise distribute the
            # pool pro-rata across tier-1 and force tiers 2/3/4 to 0
            # (no QP needed — the trivial allocation is exact).
            residual_target, skip_gossip = self._pre_apply_tier1_hard(
                total_sp
            )
            if skip_gossip:
                # Tier-1 infeasible case OR nothing left to negotiate
                # after pre-step (residual below threshold).
                self._active = False
                return
            await self._start_gossip(residual_target)

    def _pre_apply_tier1_hard(self, total_sp: float) -> tuple[float, bool]:
        """Tier-1 hard-constraint pre-step.

        Walks the leader's group members, separates tier-1 loads from
        tier-2/3/4 loads and generators, and decides between two paths:

        **Feasible** (``pool >= tier1_unmet``): lift every tier-1 load
        to ``regulation = 1`` directly via ``apply_regulate``, then
        return the residual imbalance ``T_residual = (-total_sp) -
        tier1_unmet`` so the gossip QP only needs to clear what's left
        after tier-1 is fully served.  The QP runs over tiers 2/3/4 +
        generators; tier-1 loads have ``a_i = 0`` (see
        ``tier_priority_weight``) so they sit out the QP.

        **Infeasible** (``pool < tier1_unmet``): distribute the
        available pool across tier-1 loads pro-rata by per-load unmet
        demand, force tier-2/3/4 loads to ``regulation = 0``, and
        return ``(0.0, skip_gossip=True)``.  The trivial allocation is
        priority-correct by construction; running the QP would only
        introduce noise.

        ``total_sp`` is the leader's current snapshot of the group's
        net setpoint (load convention); the legacy gossip target is
        ``-total_sp``.  The first return value is the residual target
        for ``_start_gossip``; the second is True iff the gossip
        should be skipped entirely (infeasible OR residual ≤
        threshold).
        """
        original_target = -float(total_sp)
        threshold = self._per_group_threshold()

        members = [self.context.aid]
        for neigh in self._live_neighbours():
            members.append(neigh.aid)

        tier1_records: list[tuple[str, float, float, str]] = []  # (aid, cap, sp, sector)
        non_tier1_loads: list[tuple[str, float, str]] = []       # (aid, cap, sector)
        pool = 0.0
        for aid in members:
            obs = self.behavior.observe(aid) or {}
            cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
            sec_enum = obs_sector(obs, behavior=self.behavior, aid=aid)
            if sec_enum is None:
                continue
            sec = sec_enum.value
            if cap > 0:
                prio = obs_priority(obs, behavior=self.behavior, aid=aid)
                if int(prio) == 1:
                    sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
                    tier1_records.append((aid, float(cap), float(sp), sec))
                else:
                    non_tier1_loads.append((aid, float(cap), sec))
            elif cap < 0:
                pool += abs(float(cap))

        tier1_unmet_per_load = [
            max(0.0, cap - sp) for (_aid, cap, sp, _sec) in tier1_records
        ]
        tier1_unmet = sum(tier1_unmet_per_load)

        # No tier-1 loads OR no tier-1 deficit → nothing to pre-apply;
        # the QP runs unchanged over the original imbalance.
        if tier1_unmet <= threshold:
            return original_target, abs(original_target) <= threshold

        if pool + threshold >= tier1_unmet:
            # Feasible: lift every tier-1 load to regulation = 1.
            now = float(self.context.current_timestamp)
            applied_tier1 = 0
            for (aid, _cap, _sp, sec) in tier1_records:
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
                self.context.aid, pool, tier1_unmet, applied_tier1, residual,
            )
            return residual, abs(residual) <= threshold

        # Infeasible: pro-rata pool across tier-1 by unmet; tiers 2-4 → 0.
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
        for (aid, cap, sec) in non_tier1_loads:
            if cap <= 0.0:
                continue
            apply_regulate(
                self.behavior,
                aid,
                0.0,
                sector=sec,
                reason="tier1_starvation",
                timestamp=now,
                priority_tier=None,
            )
            applied_shed += 1
        logger.info(
            "[%s] tier-1 hard pre-step (INFEASIBLE): pool=%.4f "
            "tier1_unmet=%.4f tier1_loads=%d non_tier1_shed=%d",
            self.context.aid, pool, tier1_unmet, applied_tier1, applied_shed,
        )
        return 0.0, True

    # ------------------------------------------------------------------
    # Gossip phase
    # ------------------------------------------------------------------

    async def _start_gossip(self, target: float) -> None:
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

        # An overlapping trigger (e.g. a slack-budget override arriving
        # while an AskEnergy round's response is still in flight) can
        # reach ``_start_gossip`` with a previous originator gossip still
        # live in ``self._gossip``.  Overwriting it below would drop its
        # diary terminal — the task-7/22 ``started != Σ terminals`` leak.
        # Retire it as ``abandoned`` first.
        self._close_inflight_originator(
            "abandoned", log_reason="superseded by new gossip"
        )

        # Clear violation flag at the start of each new negotiation so
        # that the monotonic floor is only breached while a violation is
        # actively present.
        self._constraint_violation_active = False

        # Gossip-only neighbour list: excludes group members without an
        # EnergyBalanceNegotiator (e.g. PowerLine branch monitors added
        # to electricity groups for the overload-relief feature).
        # Including them as gossip targets kills the token on first
        # forward — they have no handler for EnergyNegotiationMessage.
        neighbours = self._gossip_neighbours()
        self._touch_neighbours(neighbours)
        nid = str(uuid4())
        self_key = str(self.context.addr)

        obs = self.behavior.observe(self.context.aid) or {}
        starting_sp = obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        # Anchor the QP δ-box to the starting state for the whole
        # negotiation — see _GossipState comment.  Recomputing this
        # box per step from live obs caused a self-driven oscillation
        # where the box's sign flipped each round the agent regulated
        # itself.
        dmin_start, dmax_start = obs_min_max(
            obs, behavior=self.behavior, aid=self.context.aid
        )

        lambda_seed = self._compute_lambda_seed(target, len(neighbours))

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
            # Isolated agent: gossip can't help and there is no L2 to
            # consult (no group, therefore no holon).  Approve the
            # fallback directly with the full deficit so a co-located
            # LocalGenerationFallbackRole (or the agent itself) can
            # activate local DGs.  We also self-dispatch inline for
            # the common case where the fallback role is absent.
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
                    LocalGenerationApproval(
                        sector=self.sector, residual_deficit=target
                    )
                )
                self._try_self_dispatch(target)
            self._active = False
            self._gossip = None
            return

        # Now we are committed to a multi-party gossip; record the start.
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
            # Single-token semantics: the QP dual variable cannot be
            # consistently averaged across parallel tokens with
            # different ledger views (Σ a_j differs by orders of
            # magnitude between tokens that have seen different
            # peers).  Forward to a single K-weighted next-hop instead,
            # just like every subsequent round (B.1).
            next_addr = self._next_hop(neighbours, nid, 0)
            await self.context.send_message(msg, receiver_addr=next_addr)
        else:
            # Equal-share path is robust to multi-token broadcast at
            # start because the ledger merge composes correctly.
            for addr in neighbours:
                await self.context.send_message(msg, receiver_addr=addr)

        # Wallclock timeout: force-finish if gossip hasn't converged
        # within the deadline.  Adaptive: base per sector + per-agent
        # scaling so large groups get proportionally more time.
        base = _GOSSIP_TIMEOUT_BASE_S.get(self.sector, _GOSSIP_TIMEOUT_DEFAULT_S)
        timeout = base + len(neighbours) * _GOSSIP_TIMEOUT_PER_AGENT_S
        deadline = self.context.current_timestamp + timeout
        self.context.schedule_timestamp_task(
            self._gossip_timeout(nid), timestamp=deadline
        )

    async def _gossip_timeout(self, negotiation_id: str) -> None:
        if (
            self._gossip is not None
            and self._gossip.negotiation_id == negotiation_id
        ):
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
            # Diary closure: if we are the originator of an in-flight
            # gossip and a different nid arrives, the previous gossip
            # would otherwise be silently abandoned (its scheduled
            # wallclock timeout still fires later but sees the new nid
            # and exits).  Record an explicit ``abandoned`` terminal
            # for the previous nid before overwriting state, preserving
            # the started == Σ terminals invariant.
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
                memory=dict(message.memory),
                dual_lambda=getattr(message, "dual_lambda", 0.0),
            )
        else:
            self._gossip.counter = counter
            # P6: λ travels with the message; the receiver always adopts
            # the latest value.  Concurrency-safe under single-token
            # gossip because the counter is monotonically increasing
            # along the token's trajectory.
            self._gossip.dual_lambda = getattr(message, "dual_lambda", self._gossip.dual_lambda)
            # Merge per-agent ledger: for each agent, keep the entry with
            # the newest counter.  This prevents the double-counting that
            # a single aggregate digest suffers in a cyclic gossip graph.
            #
            # Byzantine bound: clip each delta to a fixed multiple of the
            # target magnitude.  A faulty or misbehaving agent reporting
            # an absurd contribution cannot corrupt total_delta for the
            # whole group.
            cap_byz = _BYZANTINE_DELTA_CAP_MULTIPLE * max(abs(self._gossip.target), 1.0)
            for k, v in message.memory.items():
                local = self._gossip.memory.get(k)
                if local is None or local[1] < v[1]:
                    # Tolerate 3-tuple legacy entries (saturated default False)
                    # so a rolling upgrade can read old in-flight messages.
                    if len(v) >= 4:
                        delta, ctr, prio, sat = v[0], v[1], v[2], bool(v[3])
                    else:
                        delta, ctr, prio = v[0], v[1], v[2]
                        sat = False
                    if delta > cap_byz or delta < -cap_byz:
                        delta = max(-cap_byz, min(cap_byz, delta))
                    self._gossip.memory[k] = (delta, ctr, prio, sat)

        target = self._gossip.target
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        # Use the gossip-anchored δ-box (captured at the start of the
        # negotiation), not a fresh per-step ``obs_min_max``.  The latter
        # tracks the LP's current sp, which flips after the agent's own
        # previous regulate — re-reading it caused the self-driven
        # bang-bang oscillation on child-194 (gas, simbench_lv).
        dmin = self._gossip.dmin_starting
        dmax = self._gossip.dmax_starting

        prev_own = self._gossip.memory.get(self_key, (0.0, 0, self.priority, False))[0]
        total_delta = self._gossip_total_delta()
        open_gap = target - total_delta

        participation_scale = self._compute_participation_scale(obs)

        active_count = max(1, len(self._gossip.memory))
        n_free = max(1, sum(
            1 for v in self._gossip.memory.values() if not v[3]
        ))

        if self.enable_qp_gossip:
            # --- P6: primal-dual QP closed-form update ---
            # The receiving agent computes its own primal δ_i directly
            # from the gossiped dual variable λ:
            #     δ_i = clamp(a_i · λ, dmin_i, dmax_i)
            # with a_i = w_i (priority weight from
            # ``_qp_priority_weight``) and the sign of λ tracking the
            # sign of the target T.  Priority ordering becomes a
            # continuous waterfall: as |λ| grows, high-w_i agents
            # saturate first.  The constraint-aware ``participation_scale``
            # is folded into a_i so an agent near a hard bound
            # contributes less per unit λ.  No priority-tier gate or
            # sub-round serialisation — priority is encoded in a_i.
            target_sign = 1 if target > 0 else (-1 if target < 0 else 0)
            a_i_base = self._qp_responsiveness(cap, target_sign)
            a_i = a_i_base * self.impact_weight * participation_scale
            new_delta = self._qp_primal(
                a_i, self._gossip.dual_lambda, dmin, dmax, target_sign, cap
            )
            saturated = _is_saturated(new_delta, dmin, dmax)
            self._gossip.memory[self_key] = (
                new_delta, counter, self.priority, saturated
            )
            self._gossip.current_delta = new_delta
            # Dedup gate (QP only).  Without priority/sub-round gating
            # every visit would call _apply_setpoint and trigger a
            # monee re-solve.  The QP closed-form δ is monotonic in λ,
            # so once an agent saturates further visits ask for the
            # same δ — skip the actuator write to avoid quadratic
            # solver work on large grids.  The first visit (prev_own
            # == 0) always applies so factor moves off the initial
            # state.
            delta_step = abs(new_delta - prev_own)
            apply_threshold = 1e-4 * max(abs(cap), 1.0)
            if cap != 0.0 and (delta_step > apply_threshold or prev_own == 0.0):
                self._apply_setpoint(self._gossip.starting_setpoint + new_delta)

            # --- Dual update (gradient on residual) ---
            # λ ← λ + γ_k · (T − Σ_a memory[a].δ) / Σ_a a_a.
            # The Σ a_a normalisation makes the per-step change in λ
            # match the local contribution to the residual: at the
            # unconstrained KKT optimum λ* = T / Σ a_j, so this update
            # converges in essentially one step when no agent is
            # clamped.  γ_k (Robbins-Monro decay) damps oscillations
            # caused by box-projection noise once agents saturate.
            total_delta_post = self._gossip_total_delta()
            residual = target - total_delta_post
            # Normalise the dual step by Σ a_j over *unsaturated* entries
            # only.  Saturated agents (v[3] == True) sit at a box bound and
            # contribute zero additional δ for any further change in λ, so
            # including their priority weight in the denominator inflates
            # the sum and slows convergence for the agents that can still
            # move.  Empirically (priority_dispatch_probe task 0) ~54 % of
            # ledger entries were saturated, contributing a ~20 % artificial
            # damping on the dual update with no convergence benefit.
            # Fall back to all entries if every agent is saturated (no one
            # can move, so the choice of denominator no longer matters for
            # the algorithm — but a zero denominator would crash).
            sum_a_est = sum(
                self._entry_responsiveness(int(v[2]), target_sign)
                for v in self._gossip.memory.values()
                if not v[3]
            )
            if sum_a_est <= 0.0:
                sum_a_est = sum(
                    self._entry_responsiveness(int(v[2]), target_sign)
                    for v in self._gossip.memory.values()
                ) or 1.0
            self._gossip.dual_lambda += (
                self._step_size(counter) * residual / sum_a_est
            )
        else:
            # --- Legacy equal-share step (P1 / P3 only) ---
            # Each active participant aims for 1/n_free of the remaining
            # open gap, scaled by Robbins-Monro step + constraint
            # participation.  Priority and sub-round gating apply.
            own_change = (
                (open_gap / n_free)
                * self.impact_weight
                * self._step_size(counter)
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
                    new_delta, counter, self.priority, saturated
                )
                self._gossip.current_delta = new_delta
                if cap != 0.0:
                    self._apply_setpoint(self._gossip.starting_setpoint + new_delta)

        # Recompute total after own update
        total_delta = self._gossip_total_delta()
        open_gap = target - total_delta

        # P2: stall detection — append the post-update gap to the window;
        # if the window range is below tolerance and the gap is still
        # above the per-group threshold, the protocol has saturated
        # without converging.  Emit LocalGenerationRequest immediately rather
        # than spinning to k_max.
        stalled = self._update_gap_window_and_check_stall(open_gap, target)

        # Next-hop selection over gossip-capable peers only.  A token
        # forwarded to a member without an EnergyBalanceNegotiator
        # (e.g. a PowerLine branch monitor in an electricity group) is
        # silently dropped by mango — the token vanishes, the gossip
        # times out.  Audit on simbench_lv showed ~60 % of failed
        # gossips were token deaths at branch hops.
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
            # ``record_finished=False`` suppresses the diary entry when
            # the caller has already recorded a more specific terminal
            # event (currently: ``timed_out``).  Only the originator
            # records — peers that joined via incoming gossip messages
            # didn't record a "started" so they shouldn't claim a
            # terminal either; otherwise the ``started == Σ terminals``
            # invariant inflates.
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

            # Unresolved deficit escalates to the local-generation
            # fallback, but only via L2 so the holon can attempt to
            # absorb the residual cross-group before L1 falls back to
            # local DGs.  Only the group leader escalates —
            # LocalGenerationFallbackRole is attached only there.
            # Members converging with residual still surface it via
            # the normal NegotiationFinishedEvent broadcast.
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
                request = LocalGenerationRequest(
                    sector=self.sector, residual_deficit=residual
                )
                # Prefer routing through the L2 holon: the holon role
                # will trigger an early ADMM rebalance attempt and
                # reply with a LocalGenerationApproval whose residual
                # reflects what L2 could not absorb.  If no holon
                # peers exist (non-holonic config) the originator
                # approves the request locally so the fallback still
                # fires.
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

        self.context.emit_event(
            NegotiationFinishedEvent(new_setpoint=new_sp, sector=self.sector)
        )

        # Broadcast to gossip-capable peers only — branch monitors have
        # no NegotiationFinishedEvent handler so sending to them just
        # wastes a scheduled message that the simulation-termination
        # tracker has to wait on.
        neighbours = self._gossip_neighbours()

        # Broadcast convergence to all live group neighbours so each can
        # emit its own local event.  Pruned neighbours are skipped —
        # sending to an unreachable peer just wastes a scheduled message
        # that will stall the simulation-termination tracker.
        # ``new_setpoint`` carries the leader's converged setpoint so the
        # CP fixed-point gate (cp.py:_handle_negotiation_finished) can
        # detect setpoint movement.  Earlier versions hard-coded zero;
        # the gate then compared |0 − 0| < tol on every subsequent
        # broadcast and suppressed every CP re-trigger past the first.
        # Neighbours re-derive their own setpoint locally via
        # ``starting_sp + current_delta`` (see _handle_negotiation_finished_msg)
        # so they are unaffected by what value travels in this field.
        finished_msg = NegotiationFinishedEvent(new_setpoint=new_sp, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(finished_msg, receiver_addr=addr)

        # Layer-2 reactive trigger: also notify holon peers.  Only group
        # leaders are injected into the ``holons`` topology, so this
        # broadcast naturally targets the right audience — the holon
        # leader (and its same-sector siblings) sees that this group
        # has finished its intra-group balance pass and can schedule a
        # holon-level ADMM round to redistribute any residual.  Without
        # this notification, ``HolonicCommunityRole._on_member_finished``
        # never fires reactively (it only listens for send_message
        # arrivals, and the groups-topology broadcast above never
        # reaches another group's leader).  The priority-aware payload
        # the holon needs is re-fetched fresh inside ``_try_rebalance``
        # via ``AskForAvailableFlex`` so each member's post-gossip
        # ``demand_by_priority`` / ``served_by_priority`` drives the
        # cross-group ADMM's S-pull (see holonic._try_rebalance, where
        # priority_shares becomes the negative cost steering allocation
        # toward groups with high-priority unserved demand).
        try:
            holon_peers = topology_neighbors(self, tid="holons")
        except KeyError:
            holon_peers = []
        for addr in holon_peers:
            await self.context.send_message(finished_msg, receiver_addr=addr)

        # Leader also notifies CP connectors
        if topology_characteristic(self, tid="groups") == "leader":
            cp_connectors = list(topology_connectors(self, tid="groups"))
            if cp_connectors:
                logger.info(
                    "[%s] gossip finished: notifying %d CP connectors (new_sp=%.4f)",
                    self.context.aid, len(cp_connectors), new_sp,
                )
            for addr in cp_connectors:
                await self.context.send_message(finished_msg, receiver_addr=addr)

        self._gossip = None
        self._active = False

    def flush_pending(self) -> None:
        """Record any still-active gossip as ``abandoned`` (or ``stalled``
        when meaningful progress was made) in the diary.

        Called from the scenario-level world teardown so a negotiation
        that was in flight when the simulation ended doesn't disappear
        silently from the per-event accounting.  After this call the
        ledger satisfies ``started == finished + timed_out + cancelled
        + abandoned + stalled`` for every nid.

        Distinguishing ``stalled`` (progress made but cut short by
        sim-end) from ``abandoned`` (no movement) is important: the
        previous behaviour lumped both under ``abandoned``, producing
        the 87 %-abandonment headline on 5 s smoke runs even though
        most of those gossips were actively closing the residual when
        the clock ran out.
        """
        if self._gossip is None:
            return
        if self._gossip.is_originator:
            total_delta = self._gossip_total_delta()
            target = self._gossip.target
            residual = target - total_delta
            # Progress threshold: closed ≥ 30 % of the target's magnitude.
            # ``stalled`` is a soft terminal that still counts toward the
            # diary invariant but signals "in-flight at sim-end" rather
            # than "never started moving".
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
        """Convergence broadcast from a gossip peer — emit own local NegotiationFinishedEvent."""
        starting_sp = (
            self._gossip.starting_setpoint
            if self._gossip
            else obs_setpoint(self.behavior.observe(self.context.aid) or {})
        )
        delta = self._gossip.current_delta if self._gossip else 0.0
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
            # Per-sector breakdown for multi-dimensional ADMM
            sec_key = sector.value
            flex_by_sector[sec_key] = flex_by_sector.get(sec_key, 0.0) + available
            balance_by_sector[sec_key] = balance_by_sector.get(sec_key, 0.0) + sp
            # Route-A supply-priority: accumulate generator-class
            # rated supply per sector.  Convention: cap < 0 for
            # generators (load-sign convention), so |cap| is the
            # rated output magnitude.  Slack agents register via
            # ``register_slack`` with their rated capacity surfaced
            # by ``obs_capacity`` — they too count as supply.
            if cap < 0:
                supply_by_sector[sec_key] = supply_by_sector.get(sec_key, 0.0) + abs(cap)
            # Priority-tier demand aggregation (loads only: cap > 0)
            if cap > 0:
                prio = obs_priority(
                    obs,
                    behavior=self.behavior,
                    aid=aid,
                    record_default_fallback_t=self.context.current_timestamp,
                )
                demand_by_priority[prio] = demand_by_priority.get(prio, 0.0) + abs(cap)
                served_by_priority[prio] = served_by_priority.get(prio, 0.0) + abs(sp)
                # Per-(sector, tier) split for the tier-stratified holon
                # ADMM.  Same data, just keyed by sector first so the
                # holon can build a 2D target vector.
                demand_by_sector_priority.setdefault(sec_key, {})
                demand_by_sector_priority[sec_key][prio] = (
                    demand_by_sector_priority[sec_key].get(prio, 0.0) + abs(cap)
                )
                served_by_sector_priority.setdefault(sec_key, {})
                served_by_sector_priority[sec_key][prio] = (
                    served_by_sector_priority[sec_key].get(prio, 0.0) + abs(sp)
                )
                # Unmet demand: rated cap minus actual sp.  Captures the
                # silent disconnect-loss case where monee's ``find_ignored_
                # nodes`` sets regulation=0 on a load that has no path to a
                # grid-former.  Without this the CP layer cannot see the
                # cross-sector deficit and skips ADMM with same-sign T.
                unmet = abs(cap) - abs(sp)
                if unmet > 1e-12:
                    unmet_by_sector[sec_key] = (
                        unmet_by_sector.get(sec_key, 0.0) + unmet
                    )
            if sector != self.sector:
                continue
            total_flex += available
            total_balance += sp
            if sp > 0 and available > 0:
                total_shedded += available

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
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    def _close_inflight_originator(self, event: str, log_reason: str | None = None) -> None:
        """Record a terminal for the in-flight gossip if this agent is
        its originator, preserving the ``started == Σ terminals`` diary
        invariant whenever ``self._gossip`` is about to be overwritten
        or torn down.

        ``event`` is the terminal kind (``"abandoned"`` / ``"cancelled"``).
        No-op when there is no in-flight gossip or this agent is only a
        relay (non-originator gossips never recorded a ``started``).

        This consolidates the four sites that retire an active gossip —
        ``_handle_negotiation_message`` (nid change),
        ``_on_constraint_violation`` (cancel), ``_yield_to_l2_authority``
        (L2 pre-emption), and ``_start_gossip`` (overlapping trigger) —
        so none of them can leak an unterminated ``started``.  The last
        of these was the eval_full_small task-7/22 diary leak: two
        negotiation triggers raced (slack-budget override + a balance
        round) and the second ``_start_gossip`` overwrote the first
        originator gossip with no terminal.
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
        """Abandon any in-flight L1 gossip so an arriving L2 directive
        can land.

        The supply-priority / tier-stratified / override-target paths
        carry the holon's authoritative priority decision; without this
        yield, ``_active=True`` from a curtailment gossip (e.g. one
        triggered by a heat-temperature constraint violation) would
        silently swallow the L2 message and leave high-priority loads
        stuck at the gossip's saturating shed — the eval_full_small
        task-89 heat inversion (child-146 tier-2 shed to 0, never
        restored by the corrective L2 ``{tier 2: 1.0}`` allocation).
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
        # Route A (supply-priority) takes highest precedence: the
        # holon-global service fractions are applied directly per
        # local-load-tier.
        service_frac = getattr(
            message, "service_fraction_by_sector_priority", None
        )
        if service_frac:
            self._yield_to_l2_authority("service_fraction")
            self._active = True
            self.context.schedule_instant_task(
                self._dispatch_service_fractions(service_frac)
            )
            return
        # Package C tier-stratified override takes precedence over the
        # scalar override.  The per-(sector, tier) map preserves the
        # holon's priority decision; the scalar collapses it.
        per_tier = getattr(
            message, "override_targets_by_sector_priority", None
        )
        if per_tier:
            self._yield_to_l2_authority("per_tier")
            self._active = True
            self.context.schedule_instant_task(
                self._dispatch_per_tier_targets(per_tier)
            )
            return
        override = getattr(message, "override_target", None)
        if override is not None and math.isfinite(override):
            # Holonic ADMM (Layer 2) computed this leader's share of the
            # holon-wide imbalance.  Skip the local ask-energy round and
            # use the ADMM result directly as the gossip target so the
            # cross-sector optimisation actually drives the per-group
            # contribution instead of being discarded.
            self._yield_to_l2_authority("override_target")
            self._active = True
            self.context.schedule_instant_task(self._start_gossip(float(override)))
            return
        self.context.schedule_instant_task(self.trigger_balance_negotiation())

    async def _dispatch_service_fractions(
        self, service_fraction: dict[str, dict[int, float]]
    ) -> None:
        """Apply a Route-A supply-priority allocation to local agents.

        ``service_fraction[sector][tier] ∈ [0, 1]`` is the *fraction
        of demand* the holon has decided to serve at (sector, tier),
        decided globally across the holon based on supply scarcity
        and priority weighting.  Each local load at (sec, tier)
        receives the same fraction as its regulation factor — so a
        tier-2 cell served at 1.0 globally produces factor=1.0 on
        every tier-2 load in this group, while a tier-8 cell at 0.0
        sheds every tier-8 load.

        Generators are not touched here — they're already at their
        rated output; the LP solver downstream routes the freed
        supply via the grid to satisfy the served demand.
        """
        try:
            members = [self.context.aid]
            for neigh in self._live_neighbours():
                members.append(neigh.aid)

            applied = 0
            shed_count = 0
            for aid in members:
                obs = self.behavior.observe(aid) or {}
                cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
                if cap <= 0:  # generator-class — leave alone
                    continue
                sec = obs_sector(obs, behavior=self.behavior, aid=aid)
                if sec is None:
                    continue
                prio = obs_priority(obs, behavior=self.behavior, aid=aid)
                frac = service_fraction.get(sec.value, {}).get(prio)
                if frac is None:
                    # No allocation for this (sec, tier) — preserve
                    # current state.  Could happen if this tier had
                    # zero demand at holon collection time but a
                    # load drifted into the tier since.
                    continue
                factor = max(0.0, min(1.0, float(frac)))
                if factor < 1.0:
                    shed_count += 1
                apply_regulate(
                    self.behavior,
                    aid,
                    factor,
                    sector=sec.value,
                    reason="holon_supply_priority",
                    timestamp=self.context.current_timestamp,
                    priority_tier=int(prio),
                )
                applied += 1

            if applied:
                logger.info(
                    "[%s] supply-frac dispatched: %d regulations, %d sheds, fracs=%s",
                    self.context.aid, applied, shed_count,
                    {sec: {t: round(v, 3) for t, v in tm.items()}
                     for sec, tm in service_fraction.items()},
                )
                # S1 — close the L2→L1→L2 cascade.  Applying factors via
                # ``apply_regulate`` mutates community state but emits
                # nothing on its own; without an event the L2 watchdog
                # short-circuits on ``_rebalance_dirty=False`` and L3
                # never re-fires either.  Mark dirty + schedule a
                # rebalance directly on the local HolonicCommunityRole
                # via ``_maybe_schedule_rebalance``, which already
                # encapsulates the gate logic (group-leader check,
                # min-gap throttle).  We deliberately do NOT
                # ``emit_event(NegotiationFinishedEvent(...))`` here:
                # the gossip path's NFE carries the leader's converged
                # setpoint and ``GenerationController`` re-applies
                # that setpoint as a regulation factor.  Emitting NFE
                # from the dispatch path with a placeholder setpoint
                # mis-triggers stability and resets the leader's own
                # factor to 0.  Direct schedule keeps the cascade
                # without that side-effect.
                for role in getattr(self.context, "roles", []):
                    if isinstance(role, HolonicCommunityRole):
                        try:
                            role._maybe_schedule_rebalance()
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "[%s] dispatch L2 re-fire skipped: %s",
                                self.context.aid, exc,
                            )
                        break
        finally:
            self._active = False

    async def _dispatch_per_tier_targets(
        self, per_tier: dict[str, dict[int, float]]
    ) -> None:
        """Apply a tier-stratified holon allocation to local agents.

        ``per_tier[sector][tier]`` is the holon-decided change in
        served setpoint for the (sector, tier) sub-population of
        *this leader's group*.  Each agent in the group with the
        matching sector + tier gets a regulation update proportional
        to its share of the tier's total capacity.

        This is the L1 honour path for Package C.  Bypasses the
        gossip QP — the holon already solved the priority allocation
        globally, and re-running the QP locally would let the local
        ``_qp_priority_weight`` overrule the holon decision, which is
        exactly the bug that motivated Package C.  CLPU ramp and
        monotonic floor still apply: ``apply_regulate`` consults the
        sector config and clamps regulation increases accordingly.
        """
        try:
            members = [self.context.aid]
            for neigh in self._live_neighbours():
                members.append(neigh.aid)

            # Group members by (sector, tier) so we can split each
            # tier's target proportionally across its members.
            per_cell_aids: dict[tuple[str, int], list[str]] = {}
            for aid in members:
                obs = self.behavior.observe(aid) or {}
                cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
                if cap <= 0:
                    continue  # generators / slacks contribute via setpoint, not tier
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
                    # ``tgt`` is the change in served setpoint the holon
                    # has allocated to this (sector, tier) cell.  In
                    # gossip's sign convention this is negative when
                    # we have to import (i.e. raise served sp).  Total
                    # absorbed = -tgt; split across members by capacity.
                    caps = []
                    for aid in aids:
                        obs = self.behavior.observe(aid) or {}
                        caps.append(
                            abs(obs_capacity(obs, behavior=self.behavior, aid=aid))
                        )
                    total_cap = sum(caps) or 1.0
                    # Compute each agent's new factor.  Sign of tgt:
                    # negative = absorb (raise factor toward 1), positive
                    # = shed (lower factor).  The holon ADMM produced
                    # tgt as -allocation (see _run_tier_stratified_admm
                    # override construction); so positive tgt means
                    # "this cell should serve more".
                    for aid, cap in zip(aids, caps):
                        share = cap / total_cap
                        delta_sp = tgt * share  # change in served setpoint
                        obs = self.behavior.observe(aid) or {}
                        sp_curr = obs_setpoint(
                            obs, behavior=self.behavior, aid=aid
                        )
                        new_sp = sp_curr + delta_sp
                        if cap == 0.0:
                            continue
                        factor = max(0.0, min(1.0, new_sp / cap))
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
                    self.context.aid, applied,
                    sum(1 for tm in per_tier.values() for _ in tm),
                )
        finally:
            self._active = False

    async def _handle_failure_notice(
        self, message: FailureNotice, meta: dict
    ) -> None:
        # Distributed failure trigger.  ``ProblemDetector`` propagates
        # this through the physical-grid neighbour graph TTL-bounded;
        # only children at affected nodes receive it.  Heat sector is
        # constraint-driven (chapter §3.1 / Gap 1) — heat negotiators
        # ignore failure notices and react via ConstraintViolation
        # instead.  Other sectors trigger only when the notice's sector
        # matches: cross-sector coupling effects propagate physically
        # but the agent-side response goes through ConstraintViolation,
        # not this notice.
        if self.sector == Sector.HEAT:
            return
        if message.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        self.context.schedule_instant_task(self.trigger_balance_negotiation())

    def _on_balance_problem(self, event: BalanceProblem, _src: Any) -> None:
        if event.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") == "leader":
            self.context.schedule_instant_task(self.trigger_balance_negotiation())

    # ------------------------------------------------------------------
    # Setpoint application with monotonic progress guarantee
    # ------------------------------------------------------------------

    def _apply_setpoint(self, new_setpoint: float) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap == 0.0:
            return
        # Tier-1 hard-lock guard: the leader's pre-step has already
        # applied ``regulation = 1`` (feasible branch) or a pro-rata
        # share (infeasible branch) to every tier-1 load.  The QP that
        # follows assigns these agents ``a_i = 0`` so they don't
        # contribute, but the existing apply-on-first-visit path would
        # still drag their factor back to 0 because the QP-side
        # ``starting_sp`` is read from a not-yet-refreshed obs.
        # Skip every actuator write on tier-1 during the QP — the
        # pre-step's write stands until the next negotiation cycle
        # re-evaluates feasibility.
        if int(self.priority) == 1:
            return
        # Slack agents (ExtPowerGrid / ExtHydrGrid) have a *free* p_mw /
        # mass_flow Var the LP picks within a wide physical envelope.
        # ``_reported_setpoint`` already surfaces a soft slack-target
        # contribution into the gossip's imbalance accounting (F2), so
        # the protocol pushes the *other* agents toward equilibrium
        # against that target.  Writing ``regulation = sp / rating`` on
        # the slack itself clamps the LP's effective slack envelope to
        # an arbitrary mid-gossip fraction — and any subsequent step
        # that needs more slack to balance the network finds the slack
        # capped at that fraction, presolving into infeasibility.  The
        # slack carries the residual; gossip must not curtail it.
        #
        # We have to use a class check (not just the slack registry)
        # because heat-side ExtHydrGrid is intentionally left unbounded
        # by ``apply_slack_budget`` (the heat LP has no slack-budget
        # discipline), which means ``_maybe_register_slack`` cannot
        # derive a rating and skips it — yet it is still structurally
        # a slack and must never be curtailed.
        if _is_slack_class_child(self.behavior, self.context.aid):
            return

        # Constraint-aware clamping: reduce the setpoint when local grid
        # measurements are near or beyond safety bounds.  Pass the role's
        # own priority tier so critical loads (tier ≤ 2) get the tighter
        # 0.99 deadband — without this, the priority-blind clamp
        # truncates tier-1 demand as soon as any local variable drifts
        # past 0.85, silently overruling the priority waterfall.
        if self.constraint_aware:
            new_setpoint = clamp_to_constraints(
                new_setpoint, obs, self.sector, tier=self.priority
            )

        factor = max(0.0, min(1.0, abs(new_setpoint / cap)))

        # --- Monotonic progress guarantee ---
        # The "no-regret switching" floor only applies when the current
        # negotiation is a restoration (target > 0 means the group needs
        # to grow net supply/demand back).  In a shedding negotiation
        # (target < 0), loads legitimately need to reduce factor to
        # rebalance, so the floor must not block them.
        target = self._gossip.target if self._gossip is not None else 0.0
        is_restoration = target > 0
        if self.priority > 0 and is_restoration:
            self._check_violation_cleared()

            if self.enable_monotonic_floor:
                if factor > self._restoration_floor:
                    self._restoration_floor = factor
                elif not self._constraint_violation_active:
                    factor = self._restoration_floor

            # --- Cold-load pickup rate limit ---
            # Only apply to ramp-up (increase).  Decreases pass through
            # immediately so shedding and violation-driven reductions
            # are not throttled.
            if self.enable_clpu_ramp:
                factor = self._rate_limit_increase(factor)

        # --- L2 priority-floor (gossip path) ---
        # The component ADMM decided this load's served tier; a
        # supply-poor local gossip group must not shed it below that
        # decision purely to zero its own imbalance.  Clamp the gossip
        # factor up to ``min(L2 allocation, constraint-allowed)`` — the
        # constraint term (computed from the same util as the clamp
        # above) lets curtailment/physics still shed it during a real
        # violation, per-load and continuously, so the floor and the
        # constraint clamp never fight.  Tier 1 already returned above;
        # this governs tiers 2/3/4 only.
        if self.enable_l2_priority_floor:
            floor = l2_effective_floor(
                self.behavior, self.context.aid, obs, self.sector, self.priority
            )
            if floor is not None and factor < floor:
                factor = floor

        # NB: gossip-driven regulates intentionally bypass the
        # ``apply_regulate`` no-op dedup.  Each gossip round computes a
        # small per-step delta (order of convergence_rate × cap / n) and
        # the protocol's correctness depends on the requested delta
        # being applied — the per-agent ledger in ``_GossipState.memory``
        # advances regardless of whether behavior.act is called, so
        # dedupping micro-steps here causes the ledger to diverge from
        # physical state and gossip stalls at k_max instead of
        # converging.  monee's warm-start absorbs consecutive small
        # deltas efficiently, so they are not the bottleneck here.
        if self.behavior.has_action(self.context.aid, "regulate"):
            self.behavior.act(self.context.aid, "regulate", factor)
            # Keep the ``apply_regulate`` dedup cache truthful — this
            # direct write bypasses it, and a stale cache silently drops
            # a later L2 re-dispatch that would restore this load (the
            # gossip-shed-never-restored cause; see note_actuated_factor).
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
        """Inline local-gen fallback for isolated agents without a
        co-located LocalGenerationFallbackRole.

        If this agent is a generator with available headroom, ramp up to
        cover as much of the deficit as possible.
        """
        if deficit <= 0:
            return
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap >= 0:
            return  # not a generator (generators have negative capacity)
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
    """Map a raw priority value and imbalance direction to a gossip round.

    The mapping guarantees:
    - **Restoration (target > 0)**: loads participate ordered by priority
      (lower number = higher urgency = earlier round).  Generators wait
      until all load tiers have had a chance.
    - **Reduction (target < 0)**: generators (priority 0) go first.
      Loads are shed in *reverse* priority order — high-priority loads
      (low number) are shed last.
    """
    if target < 0:
        if priority == 0:
            return 0  # generators first
        # Invert: lower number → more important → shed later (higher round)
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
    termination_tolerance: float = 1e-5,
    max_hops: int = _MAX_HOPS,
    step_decay_k0: int = _STEP_DECAY_K0_DEFAULT,
    enable_qp_gossip: bool = True,
    enable_l2_priority_floor: bool = True,
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
        termination_tolerance=termination_tolerance,
        max_hops=max_hops,
        step_decay_k0=step_decay_k0,
        enable_qp_gossip=enable_qp_gossip,
        enable_l2_priority_floor=enable_l2_priority_floor,
    )
