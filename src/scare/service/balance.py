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
    L2RecycleEscalation,
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
    constraint_allowed_fraction,
    constraint_utilization,
    l2_effective_floor,
    lookup_slack,
    lookup_slack_eff_budget,
    note_actuated_factor,
    obs_capacity,
    obs_min_max,
    obs_priority,
    obs_sector,
    obs_setpoint,
)
from scare.community.holonic import HolonicCommunityRole
from scare.service.gossip_math import (
    compute_lambda_seed,
    ledger_merge,
    ledger_sum_responsiveness,
    ledger_total_delta,
    qp_primal,
    qp_priority_weight,
    step_size,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

_DEFAULT_START_THRESHOLD = 1e-4
# Per-sector threshold overrides (currently all sectors share the default).
_START_THRESHOLD: dict[Sector, float] = {}

# Per-group threshold = max(_THRESHOLD_ABS_FLOOR,
# _THRESHOLD_CAPACITY_FRACTION · Σ|cap|): scales noise tolerance with group
# capacity to reject sub-half-percent imbalances.
_THRESHOLD_CAPACITY_FRACTION: float = 0.005
_THRESHOLD_ABS_FLOOR: float = 1e-6

# Heat utilization above which an agent contributes headroom to the group's
# thermal-deficit target. Below the 0.85 warning so gossip triggers pre-violation.
_HEAT_CLEAR_FRACTION: float = 0.6

_MAX_HOPS = 100

# Robbins-Monro step decay: gain = gamma_s / (1 + k / k0); satisfies
# Σ γ_k = ∞, Σ γ_k² < ∞. k0 ≈ typical LV group size.
_STEP_DECAY_K0_DEFAULT: int = 20

# P2 stall detection: when the recent gap window's range falls below
# _STALL_TOL_FRACTION · |T| and the gap still exceeds the per-group
# threshold, declare stuck and emit LocalGenerationRequest.
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

# Stale-neighbour pruning: poll periods of silence before a peer counts as dead.
_HEARTBEAT_MAX_AGE_MULTIPLE: float = 8.0

# Intra-sector priority tiers (lower = higher urgency, gossips earlier).
# Tier 1 = critical (hard-locked at the leader pre-step before the QP);
# tiers 2-4 = QP-weighted with steep exponents (1e8 / 1e4 / 1.0) so the
# proportional equilibrium is effectively strict. Schedule lives in
# ``scare.base.util.tier_priority_weight`` (shared across L1/L2/L3).
_PRIORITY_TIERS = 4

# Byzantine cap: a participant's delta is clipped to this multiple of |target|.
_BYZANTINE_DELTA_CAP_MULTIPLE: float = 5.0


def _start_threshold(sector: Sector) -> float:
    return _START_THRESHOLD.get(sector, _DEFAULT_START_THRESHOLD)


def _is_slack_class_child(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a monee ``ExtPowerGrid``/``ExtHydrGrid`` child (the
    network's slack-class boundary).

    Used to suppress regulation writes on slacks: writing ``regulation < 1``
    clamps the LP's slack envelope and the next solve goes infeasible. Covers
    both bounded (registered) and unbounded (heat-side) slacks.
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
    """MW of demand reduction this heat agent contributes to its group's
    thermal-deficit target: ``max(0, util - ϑ_clear) · |cap|`` over the
    dominant local constraint utilization. Loads only (cap > 0); heat
    generators go via the local-generation fallback. Heat sector only.
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

    Tolerance scales with box magnitude (``1e-9 + 1e-6 · max(|dmin|, |dmax|, 1)``)
    so large boxes don't reject near-bound values due to solver noise.
    """
    sat_tol = 1e-9 + 1e-6 * max(abs(dmin), abs(dmax), 1.0)
    return delta <= dmin + sat_tol or delta >= dmax - sat_tol


def _deterministic_next(neighbours: list, negotiation_id: str, counter: int) -> Any:
    """Pick the next gossip target deterministically via hash of
    (negotiation_id, counter). Ensures agents competing for the same
    resource send in the same order (needed for deterministic conflict
    resolution).
    """
    if not neighbours:
        return None
    h = hashlib.sha256(f"{negotiation_id}:{counter}".encode()).digest()
    idx = int.from_bytes(h[:4], "big") % len(neighbours)
    return neighbours[idx]


def _deterministic_sub_round(
    agent_addr: str, negotiation_id: str, tier: int, tier_size: int
) -> int:
    """Deterministic sub-round index in [0, tier_size) for intra-tier
    serialization. Stable for a given (agent, negotiation, tier) triple.
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
    # Feasible-δ box, anchored to the negotiation's *starting* setpoint.
    # Must NOT be recomputed per step from ``obs_min_max``: the agent's own
    # regulate flips the LP's reported sp, flipping the box sign each round
    # and driving a full-shed/full-load oscillation with zero net progress.
    # Anchoring keeps δ a true cumulative change and the box constant.
    dmin_starting: float = 0.0
    dmax_starting: float = 0.0
    # Per-agent contribution ledger, merged across messages by keeping the
    # highest-counter entry per agent (avoids cyclic double-counting).
    # addr_str -> (delta, counter_when_set, priority, saturated_flag).
    # saturated_flag (P1) is True when the last delta hit a box bound;
    # saturated entries are excluded from the equal-share denominator so
    # per-visit contraction doesn't collapse as the boundary set grows.
    memory: dict[str, tuple[float, int, int, bool]] = field(default_factory=dict)
    # True only for the negotiator that originated the gossip; peers that
    # build state from a received message set False. Terminal diary events
    # are recorded once per nid (by the originator), preserving the
    # ``started == Σ terminals`` invariant.
    is_originator: bool = False
    # P2: rolling window of post-update gaps, sized
    # ``_STALL_WINDOW_FACTOR · n_active``. When its range is below threshold
    # and the gap still exceeds the per-group threshold, the originator
    # emits LocalGenerationRequest and finishes with terminal "stalled".
    gap_window: list[float] = field(default_factory=list)
    # P6: scalar dual variable for the primal-dual QP gossip. At the KKT
    # optimum it is the scarcity price λ* with
    # ``Σ clamp(a_i · λ, dmin_i, dmax_i) = T``. Gossiped with the ledger;
    # the receiver does a primal update (``δ_i = clamp(a_i · λ, dmin_i, dmax_i)``)
    # and a dual update (``λ ← λ + γ_k · (T − Σ_a Δ_a)``).
    dual_lambda: float = 0.0


class EnergyBalanceNegotiator(Role):
    """Gossip-based energy balance negotiation with:

    - Priority-ordered participation: agents join at a round set by their
      priority tier, so high-priority loads restore first / shed last.
    - Monotonic progress: during restoration (target > 0) a load's
      regulation factor never drops below its established floor unless a
      hard constraint violation forces it ("no-regret switching").
    - Deterministic conflict resolution: hash-based next-hop selection;
      on contention for headroom the higher-priority agent wins.
    - Constraint-aware clamping: setpoints clamped by local constraint
      utilization, so agents near voltage/pressure/temperature bounds
      contribute less.
    - Sector time-scale awareness: convergence rate defaults to the
      sector value from ``SECTOR_TIMESCALE``.
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
        enable_actuated_ledger_writeback: bool = True,
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
        # L2 priority-floor: clamp gossip sheds up to the component ADMM's
        # per-load allocation (relaxed by local constraints), blocking the
        # L2->L1 tier inversion. See ``_apply_setpoint``.
        self.enable_l2_priority_floor = enable_l2_priority_floor
        # Write the PHYSICALLY-actuated (constraint-clamped / floored) delta
        # back into the gossip ledger after ``_apply_setpoint`` so the dual sees
        # a constrained load's true (smaller) contribution and reallocates the
        # freed supply to unconstrained loads. Without it the ledger keeps the
        # unclamped requested delta and the freed supply is never re-served.
        self.enable_actuated_ledger_writeback = enable_actuated_ledger_writeback
        self.max_hops = max_hops
        self.step_decay_k0 = max(1, int(step_decay_k0))
        # When True run the primal-dual QP gossip, else the equal-share
        # update. Both share ledger, Byzantine cap, heartbeat liveness,
        # next-hop, saturation flag, stall detection, step decay, and
        # termination — only the per-agent update rule differs. Ablation flag.
        self.enable_qp_gossip = enable_qp_gossip

        # Sector-specific convergence rate unless overridden.
        ts = SECTOR_TIMESCALE.get(sector, {})
        self.convergence_rate = (
            convergence_rate if convergence_rate is not None
            else ts.get("convergence_rate", 0.5)
        )

        self._active: bool = False
        self._gossip: _GossipState | None = None
        # Setpoint-gathering phase state, before gossip starts.
        self._trigger_nid: str | None = None
        self._trigger_responses: dict[str, float] = {}
        self._trigger_expected: int = 0

        # Total |cap| across this leader's group (refreshed each
        # ``trigger_balance_negotiation``); drives the per-group threshold.
        self._group_capacity_abs: float = 0.0

        # Monotonic progress floor: highest regulation factor applied during
        # restoration; may only decrease while a hard constraint violation is
        # active.
        self._restoration_floor: float = 0.0
        self._constraint_violation_active: bool = False

        # Cold-load pickup rate limiter: post-outage inrush is 2-6x steady
        # state. Caps how fast the regulation factor can grow (decreases /
        # sheds unrestricted); ramp scales with sector convergence_rate.
        self._last_regulate_timestamp: float | None = None
        self._last_regulate_factor: float = 0.0
        self._clpu_ramp_per_s: float = self.convergence_rate

        # Neighbour liveness: str(addr) -> timestamp of last inbound message.
        # A peer silent longer than HEARTBEAT_MAX_AGE is pruned from gossip.
        # An address enters the map on first contact attempt so unresponsive
        # nodes age out rather than staying ghost-alive forever.
        self._neighbour_last_seen: dict[str, float] = {}
        poll = ts.get("poll_period_s", 1.0)
        self._heartbeat_max_age_s: float = poll * _HEARTBEAT_MAX_AGE_MULTIPLE

        # B.1: continuous coupling weights K_ij(t) in [0, 1] per neighbour,
        # biasing forwarding and gating liveness (K >= liveness_threshold
        # replaces the binary heartbeat). Decay scales with poll period so
        # slow-polling heat doesn't pessimise K too fast.
        self._trust = TrustLedger(
            TrustParams(
                decay_rate_per_s=1.0 / max(poll * _HEARTBEAT_MAX_AGE_MULTIPLE, 1.0),
                recover_rate=0.6,
                liveness_threshold=0.5,
                initial=1.0,
            )
        )

        # Local proactive constraint utilization: variable -> last-reported
        # utilization in [0, 1], from the co-located GridConstraintMonitor's
        # ConstraintWarning events. Throttles the gossip step near a bound.
        self._proactive_util: dict[str, float] = {}

    def setup(self) -> None:
        # Register this aid as gossip-capable (shared ``behavior`` registry,
        # sector -> set of aids; read by ``_gossip_neighbours``) so peers
        # route ``EnergyNegotiationMessage`` only to agents that process it.
        # PowerLine branch agents share the electricity groups but have no
        # EnergyBalanceNegotiator; without this filter the token's next-hop
        # forwards to a branch monitor that drops it, killing the gossip.
        store = getattr(self.behavior, "_scare_gossip_capable", None)
        if store is None:
            store = {}
            self.behavior._scare_gossip_capable = store
        store.setdefault(self.sector, set()).add(self.context.aid)

        # Mango dispatches handle_message synchronously, so async handlers are
        # wrapped to schedule themselves via the agent scheduler (keeps them
        # tracked by termination detection). Each inbound message also stamps
        # the sender's heartbeat for liveness tracking.
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
        # Mango needs >=1 local subscriber per emitted event type.
        # LocalGenerationFallbackRole is leader-only, so a non-leader hitting
        # the singleton-fallback path would crash emit_event without this
        # no-op. Real fallback logic stays on the leader.
        self.context.subscribe_event(
            self, LocalGenerationApproval, self._on_local_gen_approval_noop
        )
        # Same no-op pattern for NegotiationFinishedEvent: production agents
        # also host ``GenerationController`` (a subscriber), but in minimal
        # compositions the negotiator may be alone and stall termination can
        # fire ``_finish_negotiation`` with no external listener wired.
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_finished_noop
        )

    # ------------------------------------------------------------------
    # Constraint violation tracking (for monotonic progress override)
    # ------------------------------------------------------------------

    def _on_local_gen_approval_noop(
        self, _event: LocalGenerationApproval, _src: Any
    ) -> None:
        # No-op; the leader's LocalGenerationFallbackRole handles the
        # response. See setup() for why this subscriber exists.
        return

    def _on_finished_noop(
        self, _event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        # No-op; production agents have a GenerationController subscribing to
        # NegotiationFinishedEvent. Keeps dispatch safe in minimal compositions.
        return

    def _on_constraint_warning(self, event: ConstraintWarning, _src: Any) -> None:
        # Record proximity-to-bound utilization so the gossip step can
        # throttle this agent. Other-sector warnings ignored (sector coupling
        # is handled at the holon/CP level, not in gossip).
        if event.sector != self.sector:
            return
        self._proactive_util[event.variable] = float(event.utilization)

    def _on_constraint_violation(self, event: ConstraintViolation, _src: Any) -> None:
        if event.sector == self.sector:
            self._constraint_violation_active = True
            # Cancel any active gossip: the constraint landscape changed, so
            # converging on a stale target may push further into violation.
            # GridConstraintMonitor's BalanceProblem triggers a fresh round.
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
        return ledger_total_delta(self._gossip.memory)

    def _compute_participation_scale(self, obs: dict) -> float:
        """Constraint-aware throttle in [0, 1] blending local, worst-neighbour,
        and proactive-warning utilization. Heat exempt: thermal violations want
        stressed loads to shed (via priority + clamp), not throttle.
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
        # B.1: each received message recovers the K-score multiplicatively.
        self._trust.on_message_received(key, now)

    def _touch_neighbours(self, addrs: list) -> None:
        """Seed the heartbeat clock for just-contacted neighbours: grants a
        grace period (the heartbeat timeout) before an unresponsive node ages
        out, but still prunes it next round if it never replies.
        """
        now = self.context.current_timestamp
        for addr in addrs:
            key = str(addr)
            if key not in self._neighbour_last_seen:
                self._neighbour_last_seen[key] = now

    def _update_gap_window_and_check_stall(
        self, open_gap: float, target: float
    ) -> bool:
        """P2: append the post-update gap to the rolling window and decide
        whether the protocol has stalled.

        Stall when: past warm-up, window full, its max-min range is below
        ``max(_STALL_TOL_FRACTION · |T|, _STALL_TOL_FLOOR)``, and the current
        gap still exceeds the per-group threshold.

        Warm-up = ``_PRIORITY_TIERS + 1 + window_size`` rounds: covers the
        priority-gating delay (generators wait until counter >= P+1 during
        restoration) plus a full window of post-warmup samples, so the
        detector doesn't fire during the pre-eligibility silence.
        """
        if self._gossip is None:
            return False
        active = max(1, len(self._gossip.memory))
        window_size = _STALL_WINDOW_FACTOR * active
        win = self._gossip.gap_window
        win.append(open_gap)
        if len(win) > window_size:
            del win[0]
        # Warm-up gate: priority tiers and sub-round serialisation silence
        # many early rounds; don't decide stall before a fair chance to converge.
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

        Only the originator records the ``stalled`` terminal (keeps
        ``started == Σ terminals`` exact). ``_finish_negotiation`` is then
        called with ``record_finished=False`` to avoid double-counting a
        ``finished`` terminal; it emits LocalGenerationRequest if leader.
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
        """Per-negotiation threshold scaled by the leader's snapshot of total
        group load capacity. Falls back to the sector default floor when no
        capacity info exists yet (before the first trigger).
        """
        if self._group_capacity_abs <= 0.0:
            return _start_threshold(self.sector)
        return max(
            _THRESHOLD_ABS_FLOOR,
            _THRESHOLD_CAPACITY_FRACTION * self._group_capacity_abs,
        )

    def _live_neighbours(self) -> list:
        """Group neighbours whose trust score K_ij exceeds the liveness
        threshold (B.1). Unknown neighbours bootstrap optimistically (initial
        K = 1.0) so first contact succeeds.

        Includes every live group neighbour — for the flex-query round
        (``AskEnergyMessage``), which branch agents answer. For gossip token
        routing use ``_gossip_neighbours`` (only peers that subscribe to
        ``EnergyNegotiationMessage``).
        """
        all_neighbours = topology_neighbors(self, tid="groups")
        now = self.context.current_timestamp
        return [a for a in all_neighbours if self._trust.is_live(str(a), now)]

    def _gossip_neighbours(self) -> list:
        """Live group neighbours with a same-sector ``EnergyBalanceNegotiator``
        — agents that actually process an ``EnergyNegotiationMessage``.

        PowerLine branch agents sit in the electricity groups (for flex-query
        and overload-relief) but run only a branch-mode ``GridConstraintMonitor``,
        no negotiator, so the token dies on a forward to them. Filtering on the
        gossip-capable registry (see ``setup()``) keeps them in the community
        for flex while excluding them from token routing.
        """
        store = getattr(self.behavior, "_scare_gossip_capable", {})
        capable = store.get(self.sector, set())
        return [a for a in self._live_neighbours() if a.aid in capable]

    def _scored_neighbours(self, neighbours: list) -> list[float]:
        """Return the K-score for each neighbour in ``neighbours`` order."""
        now = self.context.current_timestamp
        return [self._trust.score(str(a), now) for a in neighbours]

    def _next_hop(self, neighbours: list, nid: str, counter: int):
        """B.1: K-weighted deterministic next-hop. Picks the index in
        proportion to trust score K_ij; reduces to uniform SHA256-modulo when
        all K are equal. Low-K neighbours (silence, partial outage) are routed
        around automatically.
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
        # MW balance is deactivated for heat: the frontier controller +
        # curtailment auction own it and the unbounded heat slack means no MW
        # imbalance to resolve. El/gas unaffected.
        if self.sector == Sector.HEAT:
            return
        if self._active:
            return
        self._active = True

        neighbours = self._live_neighbours()
        self._touch_neighbours(neighbours)
        # Snapshot group |cap| so the threshold scales with what's physically
        # present. Loads only (cap > 0): the threshold gates demand-side
        # imbalance, and is also the right denominator for curtailment (T<0) —
        # ignore noise relative to the demand being protected.
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

        El/gas: the raw setpoint (Σ s_i ≈ 0 ⇒ balanced). Heat: amplified by
        the local thermal deficit so a stressed group surfaces a negative
        target, shedding load via the reverse-priority rules in
        ``_compute_actual_priority``.

        F2 — slack target: if this agent is a registered slack, report the
        *target infeed* ``slack_target_fraction · rating`` (load convention,
        positive = import) instead of the LP's current draw. This reframes the
        imbalance from "balance to zero" (trivial for the LP) to "the rest of
        the group balances so the slack draws only its target".

        Gossip is per-community but the slack budget is a property of the whole
        connected component, so this target only matches operator intent when
        the slack's community spans most of the component (``component_level``).
        For ``single_level`` / holonic ``scare`` the community is small and
        this target is incoherent with the global budget; budget enforcement
        there uses ``SlackBudgetMonitor``'s signed ``override_target`` instead.
        """
        slack = lookup_slack(self.behavior, self.context.aid)
        if slack is not None:
            cfg = getattr(self.behavior, "_scare_config", None)
            fraction = float(
                getattr(cfg, "slack_target_fraction", 0.0) if cfg is not None else 0.0
            )
            # ``slack.cap`` is generator-convention (negative); the import
            # target is its magnitude (load convention: importing slack > 0).
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

            # Tier-1 hard pre-step: lift every tier-1 load to regulation=1 if
            # the generator pool covers tier-1 demand; else distribute the pool
            # pro-rata and force tiers 2/3/4 to 0 (trivial allocation, no QP).
            residual_target, skip_gossip = self._pre_apply_tier1_hard(
                total_sp
            )
            if skip_gossip:
                # Tier-1 infeasible, or residual below threshold.
                self._active = False
                return
            await self._start_gossip(residual_target)

    def _pre_apply_tier1_hard(self, total_sp: float) -> tuple[float, bool]:
        """Tier-1 hard-constraint pre-step over the leader's group.

        Feasible (``pool >= tier1_unmet``): lift every tier-1 load to
        ``regulation = 1`` and return residual ``(-total_sp) - tier1_unmet`` so
        the QP clears only what's left. Tier-1 loads have ``a_i = 0`` so they
        sit out the QP.

        Infeasible (``pool < tier1_unmet``): distribute the pool pro-rata by
        per-load unmet demand, force tiers 2/3/4 to ``regulation = 0``, and
        return ``(0.0, skip_gossip=True)`` (priority-correct by construction).

        ``total_sp`` is the group's net setpoint (load convention); gossip
        target is ``-total_sp``. Returns (residual target for ``_start_gossip``,
        skip flag — True iff infeasible OR residual ≤ threshold).
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

        # No tier-1 loads / no tier-1 deficit: QP runs over the original imbalance.
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
        # MW balance deactivated for heat (see ``trigger_balance_negotiation``);
        # also guards the holon ``override_target`` path that calls here directly.
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

        # An overlapping trigger (e.g. a slack-budget override arriving while
        # an AskEnergy response is still in flight) can reach here with a live
        # originator gossip; overwriting it would drop its diary terminal.
        # Retire it as ``abandoned`` first.
        self._close_inflight_originator(
            "abandoned", log_reason="superseded by new gossip"
        )

        # Reset the violation flag so the monotonic floor only yields while a
        # violation is actively present.
        self._constraint_violation_active = False

        # Gossip-only neighbours: excludes members without an
        # EnergyBalanceNegotiator (e.g. branch monitors); forwarding the token
        # to them kills it (no EnergyNegotiationMessage handler).
        neighbours = self._gossip_neighbours()
        self._touch_neighbours(neighbours)
        nid = str(uuid4())
        self_key = str(self.context.addr)

        obs = self.behavior.observe(self.context.aid) or {}
        starting_sp = obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        # Anchor the QP δ-box to the starting state for the whole negotiation
        # (see _GossipState); recomputing it per step causes a self-driven
        # sign-flip oscillation.
        dmin_start, dmax_start = obs_min_max(
            obs, behavior=self.behavior, aid=self.context.aid
        )

        lambda_seed = compute_lambda_seed(
            target, len(neighbours),
            priority=self.priority, priority_tiers=_PRIORITY_TIERS,
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
            # Isolated agent: gossip can't help and there's no L2/holon to
            # consult. Approve the fallback directly with the full deficit so a
            # co-located LocalGenerationFallbackRole activates local DGs, and
            # self-dispatch inline in case that role is absent.
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
            # Single-token: the QP dual variable can't be consistently averaged
            # across parallel tokens with different ledger views (Σ a_j varies
            # by orders of magnitude). Forward to one K-weighted next-hop.
            next_addr = self._next_hop(neighbours, nid, 0)
            await self.context.send_message(msg, receiver_addr=next_addr)
        else:
            # Equal-share path tolerates multi-token broadcast: the ledger
            # merge composes correctly.
            for addr in neighbours:
                await self.context.send_message(msg, receiver_addr=addr)

        # Wallclock timeout: force-finish if gossip hasn't converged. Adaptive:
        # per-sector base + per-agent scaling for larger groups.
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
            # If we originate an in-flight gossip and a different nid arrives,
            # record an explicit ``abandoned`` terminal for the old nid before
            # overwriting state (preserves started == Σ terminals; its
            # scheduled timeout would otherwise see the new nid and exit).
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
            # P6: λ travels with the message; the receiver adopts the latest.
            # Safe under single-token gossip (counter increases monotonically
            # along the trajectory).
            self._gossip.dual_lambda = getattr(message, "dual_lambda", self._gossip.dual_lambda)
            # Merge per-agent ledger keeping the newest-counter entry (avoids
            # the double-counting an aggregate digest suffers in cyclic graphs).
            # Byzantine bound: clip each delta to a multiple of |target| so one
            # misbehaving agent can't corrupt group total_delta.
            cap_byz = _BYZANTINE_DELTA_CAP_MULTIPLE * max(abs(self._gossip.target), 1.0)
            ledger_merge(self._gossip.memory, message.memory, byzantine_cap=cap_byz)

        target = self._gossip.target
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        # Use the gossip-anchored δ-box, not a fresh per-step ``obs_min_max``:
        # the latter tracks the LP's current sp, which flips after the agent's
        # own regulate and drives a self-induced bang-bang oscillation.
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
            # P6: primal-dual QP closed-form update.
            # δ_i = clamp(a_i · λ, dmin_i, dmax_i) with a_i = priority weight
            # and sign(λ) = sign(T). Priority becomes a continuous waterfall:
            # as |λ| grows, high-a_i agents saturate first. ``participation_scale``
            # folds into a_i so agents near a bound contribute less per unit λ.
            # No tier gate / sub-round serialisation — priority is in a_i.
            target_sign = 1 if target > 0 else (-1 if target < 0 else 0)
            a_i_base = qp_priority_weight(
                self.priority, target_sign, priority_tiers=_PRIORITY_TIERS,
            )
            a_i = a_i_base * self.impact_weight * participation_scale
            new_delta = qp_primal(a_i, self._gossip.dual_lambda, dmin, dmax)
            saturated = _is_saturated(new_delta, dmin, dmax)
            self._gossip.memory[self_key] = (
                new_delta, counter, self.priority, saturated
            )
            self._gossip.current_delta = new_delta
            # Dedup gate (QP only): the closed-form δ is monotonic in λ, so
            # once saturated further visits request the same δ — skip the
            # actuator write to avoid quadratic re-solves on large grids. First
            # visit (prev_own == 0) always applies to move off the initial state.
            delta_step = abs(new_delta - prev_own)
            apply_threshold = 1e-4 * max(abs(cap), 1.0)
            if cap != 0.0 and (delta_step > apply_threshold or prev_own == 0.0):
                applied_sp = self._apply_setpoint(
                    self._gossip.starting_setpoint + new_delta
                )
                # Ledger write-back (before the dual update): record the delta
                # PHYSICALLY actuated, not the requested one. A constraint-
                # clamped load thus shows its true (smaller) contribution, the
                # residual persists, and the dual below raises λ so unconstrained
                # loads absorb the freed supply — the constraint stays solved
                # (the clamp at actuation is untouched).
                self._writeback_actuated_delta(
                    self_key, applied_sp, new_delta, counter, dmin, dmax,
                )

            # Dual update (gradient on residual):
            # λ ← λ + γ_k · (T − Σ δ) / Σ a_a. The Σ a_a normalisation makes
            # the λ step match the local residual contribution: at the
            # unconstrained KKT optimum λ* = T / Σ a_j, so it converges in ~one
            # step when nothing is clamped. γ_k (Robbins-Monro) damps box-noise
            # oscillation once agents saturate.
            total_delta_post = self._gossip_total_delta()
            residual = target - total_delta_post
            # Normalise by Σ a_j over *unsaturated* entries only: saturated
            # agents sit at a box bound and add no δ for further λ, so counting
            # them inflates the denominator and slows the agents that can still
            # move (see ledger_sum_responsiveness for the fallback).
            sum_a_est = ledger_sum_responsiveness(
                self._gossip.memory, target_sign, priority_tiers=_PRIORITY_TIERS,
            )
            self._gossip.dual_lambda += (
                step_size(self.convergence_rate, counter,
                          step_decay_k0=self.step_decay_k0)
                * residual / sum_a_est
            )
        else:
            # Equal-share step (P1/P3 only): each active participant aims for
            # 1/n_free of the open gap, scaled by Robbins-Monro step +
            # constraint participation. Priority and sub-round gating apply.
            own_change = (
                (open_gap / n_free)
                * self.impact_weight
                * step_size(self.convergence_rate, counter,
                            step_decay_k0=self.step_decay_k0)
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
                    applied_sp = self._apply_setpoint(
                        self._gossip.starting_setpoint + new_delta
                    )
                    self._writeback_actuated_delta(
                        self_key, applied_sp, new_delta, counter, dmin, dmax,
                    )

        # Recompute total after own update
        total_delta = self._gossip_total_delta()
        open_gap = target - total_delta

        # P2: stall detection — if the window range is below tolerance and the
        # gap still exceeds the per-group threshold, the protocol saturated
        # without converging; escalate now rather than spinning to k_max.
        stalled = self._update_gap_window_and_check_stall(open_gap, target)

        # Next-hop over gossip-capable peers only: a token forwarded to a
        # member without an EnergyBalanceNegotiator (e.g. a branch monitor) is
        # silently dropped and the gossip times out.
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
            # ``record_finished=False`` when the caller already recorded a more
            # specific terminal (e.g. ``timed_out``). Only the originator
            # records: peers never recorded a "started", so a terminal from
            # them would inflate the ``started == Σ terminals`` invariant.
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

            # Unresolved deficit escalates to local-gen fallback via L2 first,
            # so the holon can absorb the residual cross-group before L1 falls
            # back to local DGs. Leader-only (LocalGenerationFallbackRole lives
            # there); members surface residual via the NFE broadcast.
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
                # Route through the L2 holon: it triggers an early ADMM
                # rebalance and replies with a LocalGenerationApproval whose
                # residual is what L2 couldn't absorb. With no holon peers
                # (non-holonic config) approve locally so the fallback fires.
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

        # Broadcast convergence to gossip-capable group neighbours so each can
        # emit its own local event (branch monitors lack an NFE handler and
        # would just stall the termination tracker). ``new_setpoint`` carries
        # the leader's converged setpoint so the CP fixed-point gate
        # (cp.py:_handle_negotiation_finished) can detect setpoint movement; a
        # hard-coded zero would suppress every CP re-trigger past the first.
        # Neighbours re-derive their own setpoint via ``starting_sp +
        # current_delta`` so the value carried here doesn't affect them.
        neighbours = self._gossip_neighbours()
        finished_msg = NegotiationFinishedEvent(new_setpoint=new_sp, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(finished_msg, receiver_addr=addr)

        # Layer-2 reactive trigger: notify holon peers (only leaders are in the
        # ``holons`` topology). This lets the holon leader schedule a
        # holon-level ADMM round to redistribute residual; without it
        # ``HolonicCommunityRole._on_member_finished`` never fires reactively.
        # The priority-aware payload is re-fetched in ``_try_rebalance`` via
        # ``AskForAvailableFlex`` so post-gossip demand/served-by-priority
        # drives the cross-group ADMM's S-pull toward high-priority unserved demand.
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
        """Record any still-active gossip as ``stalled`` (progress made) or
        ``abandoned`` (no movement) in the diary.

        Called from world teardown so an in-flight negotiation at sim-end
        still counts in per-event accounting; afterward
        ``started == finished + timed_out + cancelled + abandoned + stalled``
        holds for every nid.
        """
        if self._gossip is None:
            return
        if self._gossip.is_originator:
            total_delta = self._gossip_total_delta()
            target = self._gossip.target
            residual = target - total_delta
            # ``stalled`` (closed >= 30% of |target|) is a soft terminal: counts
            # toward the invariant but signals "in-flight at sim-end".
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
        """Convergence broadcast from a peer; emit own local NegotiationFinishedEvent."""
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
            # Per-sector breakdown for multi-dimensional ADMM.
            sec_key = sector.value
            flex_by_sector[sec_key] = flex_by_sector.get(sec_key, 0.0) + available
            balance_by_sector[sec_key] = balance_by_sector.get(sec_key, 0.0) + sp
            # Route-A supply-priority: accumulate generator-class deliverable
            # supply per sector (generators have cap < 0).
            #
            # Non-slack generators count *delivered* ``|sp|`` (= |cap|·regulation),
            # not rated ``|cap|``: failed (regulation 0) and constraint-curtailed
            # (line/voltage hold regulation < 1) generators report full |cap| but
            # can't push it, so counting rated would inflate the pool with
            # undeliverable supply, over-serve the waterfall, and blow the slack
            # budget. A healthy generator sits at regulation 1 (|sp| == |cap|),
            # so |sp| self-adapts to failure/curtailment without duals.
            #
            # Slacks are the exception: ``obs_capacity`` returns the registered
            # budget (cap = -budget) but ``obs_setpoint`` is the LP's actual
            # (possibly over-budget) draw, which must NOT enter the pool. Keep
            # slacks at the budgeted rating so the pool reflects the allowance.
            if cap < 0:
                if lookup_slack(self.behavior, aid) is not None:
                    # Slack: advertise the *effective* budget when
                    # SlackBudgetMonitor's loss-compensation set one (pool
                    # targets ``B - losses``, draw lands at ``B``), else nominal.
                    eff = lookup_slack_eff_budget(self.behavior, aid)
                    gen_supply = float(eff) if eff is not None else abs(cap)
                else:
                    gen_supply = abs(sp)    # generator: deliverable, not rated
                supply_by_sector[sec_key] = supply_by_sector.get(sec_key, 0.0) + gen_supply
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
                # Per-(sector, tier) split for the tier-stratified holon ADMM
                # (same data keyed by sector first for a 2D target vector).
                demand_by_sector_priority.setdefault(sec_key, {})
                demand_by_sector_priority[sec_key][prio] = (
                    demand_by_sector_priority[sec_key].get(prio, 0.0) + abs(cap)
                )
                served_by_sector_priority.setdefault(sec_key, {})
                served_by_sector_priority[sec_key][prio] = (
                    served_by_sector_priority[sec_key].get(prio, 0.0) + abs(sp)
                )
                # Unmet demand: rated cap minus actual sp. Captures the silent
                # disconnect-loss case where monee's ``find_ignored_nodes`` sets
                # regulation=0 on a load with no path to a grid-former; without
                # it the CP layer misses the cross-sector deficit.
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
        """Record a terminal (``event``: ``"abandoned"``/``"cancelled"``) for
        the in-flight gossip if this agent originated it, preserving the
        ``started == Σ terminals`` invariant before ``self._gossip`` is
        overwritten or torn down. No-op for relays (no ``started`` recorded).

        Shared by the four sites that retire an active gossip
        (``_handle_negotiation_message`` nid change, ``_on_constraint_violation``,
        ``_yield_to_l2_authority``, ``_start_gossip`` overlapping trigger) so
        none can leak an unterminated ``started``.
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

        The supply-priority / tier-stratified / override-target paths carry the
        holon's authoritative priority decision; without this yield, a
        curtailment gossip's ``_active=True`` would swallow the L2 message and
        leave high-priority loads stuck at the gossip's saturating shed.
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
        # MW balance deactivated for heat: ignore L2/L3 holon supply-priority
        # overrides (heat owned by the frontier controller + auction).
        if self.sector == Sector.HEAT:
            return
        # Route A (supply-priority) has highest precedence: holon-global service
        # fractions applied directly per local-load-tier.
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
        # Tier-stratified override takes precedence over the scalar one: the
        # per-(sector, tier) map preserves the holon's priority decision that
        # the scalar would collapse.
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
            # L2 holonic ADMM computed this leader's share of the holon-wide
            # imbalance: skip the local ask-energy round and use the ADMM result
            # directly as the gossip target so it drives the per-group contribution.
            self._yield_to_l2_authority("override_target")
            self._active = True
            self.context.schedule_instant_task(self._start_gossip(float(override)))
            return
        self.context.schedule_instant_task(self.trigger_balance_negotiation())

    async def _dispatch_service_fractions(
        self, service_fraction: dict[str, dict[int, float]]
    ) -> None:
        """Apply a Route-A supply-priority allocation to local agents.

        ``service_fraction[sector][tier] ∈ [0, 1]`` is the holon-global
        fraction of demand to serve at (sector, tier); each local load at
        (sec, tier) gets that fraction as its regulation factor.

        Generators are untouched (already at rated output); the LP routes the
        freed supply to satisfy served demand.
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
                if cap <= 0:  # generator-class: leave alone
                    continue
                sec = obs_sector(obs, behavior=self.behavior, aid=aid)
                if sec is None:
                    continue
                prio = obs_priority(obs, behavior=self.behavior, aid=aid)
                frac = service_fraction.get(sec.value, {}).get(prio)
                if frac is None:
                    # No allocation for this (sec, tier): preserve current
                    # state (tier may have had zero demand at collection time).
                    continue
                factor = max(0.0, min(1.0, float(frac)))
                # El/gas: local physical feasibility is a hard ceiling on the
                # holon allocation (no other temperature-aware lever guards
                # them). HEAT exempt: the frontier controller owns its
                # temperature; capping here would let a transient t_k dip
                # re-shed a feasible heat load (and the heat slack is unbounded).
                if sec is not Sector.HEAT:
                    factor = min(
                        factor, constraint_allowed_fraction(obs, sec, tier=prio)
                    )
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
                # S1: close the L2->L1->L2 cascade. ``apply_regulate`` mutates
                # state but emits nothing, so without a nudge the L2 watchdog
                # short-circuits on ``_rebalance_dirty=False``. Schedule a
                # rebalance directly on the local HolonicCommunityRole via
                # ``_maybe_schedule_rebalance`` (encapsulates the gate logic).
                # Do NOT emit NegotiationFinishedEvent here: its placeholder
                # setpoint would mis-trigger stability and reset the leader's
                # own factor to 0 (the gossip path's NFE carries the real sp).
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

        ``per_tier[sector][tier]`` is the holon-decided change in served
        setpoint for the (sector, tier) sub-population of this leader's group;
        each matching agent gets a regulation update proportional to its share
        of the tier's total capacity.

        Bypasses the gossip QP: the holon already solved the priority
        allocation globally, and re-running the QP would let local
        ``_qp_priority_weight`` overrule it. CLPU ramp and monotonic floor
        still apply via ``apply_regulate``.
        """
        try:
            members = [self.context.aid]
            for neigh in self._live_neighbours():
                members.append(neigh.aid)

            # Group members by (sector, tier) to split each tier's target
            # proportionally across its members.
            per_cell_aids: dict[tuple[str, int], list[str]] = {}
            for aid in members:
                obs = self.behavior.observe(aid) or {}
                cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
                if cap <= 0:
                    continue  # generators/slacks contribute via setpoint, not tier
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
                    # ``tgt`` is the holon-allocated change in served setpoint
                    # for this cell; split across members by capacity.
                    caps = []
                    for aid in aids:
                        obs = self.behavior.observe(aid) or {}
                        caps.append(
                            abs(obs_capacity(obs, behavior=self.behavior, aid=aid))
                        )
                    total_cap = sum(caps) or 1.0
                    # New factor per agent. Positive tgt = serve more (raise
                    # factor); negative = shed.
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
                        # Cap by local feasibility for el/gas (see the matching
                        # clamp in _dispatch_service_fractions). HEAT exempt:
                        # frontier controller owns its temperature.
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
                    self.context.aid, applied,
                    sum(1 for tm in per_tier.values() for _ in tm),
                )
        finally:
            self._active = False

    async def _handle_failure_notice(
        self, message: FailureNotice, meta: dict
    ) -> None:
        # Distributed failure trigger (``ProblemDetector`` propagates it
        # TTL-bounded through the physical neighbour graph). Heat is
        # constraint-driven, so heat negotiators ignore failure notices and
        # react via ConstraintViolation; other sectors trigger only on a
        # sector match (cross-sector coupling propagates physically, the
        # agent-side response goes via ConstraintViolation).
        if message.sector != self.sector:
            return
        # L2 escalation (all sectors, members included): relay the topology
        # change to the community leader so it can re-waterfall the component.
        # The component coordinator may be beyond this notice's TTL, so a member
        # that detected the failure tells its leader, which fans the escalation
        # across the component mesh. This recycles L2 allocation/membership,
        # distinct from the heat L1 setpoint trigger below. Route over the
        # ``groups`` topology (leader reachable there); only the leader acts
        # (``_handle_l2_recycle`` gates on leadership), so non-leaders ignore it.
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

    # ------------------------------------------------------------------
    # Setpoint application with monotonic progress guarantee
    # ------------------------------------------------------------------

    def _apply_setpoint(self, new_setpoint: float) -> float | None:
        """Actuate ``new_setpoint`` (after constraint clamp + floors) and return
        the PHYSICALLY-applied signed setpoint (``factor * cap``), or ``None``
        when nothing is actuated (cap=0, tier-1 hard-lock, slack, no action).
        The caller uses the return value to write the actuated delta back into
        the gossip ledger.
        """
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap == 0.0:
            return None
        # Tier-1 hard-lock guard: the leader's pre-step already set tier-1
        # loads (regulation=1 feasible, or pro-rata infeasible) and the QP
        # gives them a_i = 0. Skip the actuator write so the apply-on-first-
        # visit path doesn't drag them back to 0 off a stale ``starting_sp``;
        # the pre-step's write stands until the next negotiation cycle.
        if int(self.priority) == 1:
            return None
        # Slacks (ExtPowerGrid/ExtHydrGrid) have a free p_mw/mass_flow Var the
        # LP picks within a wide envelope; ``_reported_setpoint`` already
        # surfaces a soft slack target (F2). Writing ``regulation = sp/rating``
        # clamps the LP's slack envelope to an arbitrary fraction, presolving
        # into infeasibility when more slack is later needed. The slack carries
        # the residual; gossip must not curtail it.
        #
        # Use a class check, not the slack registry: heat-side ExtHydrGrid is
        # left unbounded by ``apply_slack_budget`` so it never registers a
        # rating, yet is structurally a slack and must never be curtailed.
        if _is_slack_class_child(self.behavior, self.context.aid):
            return None

        # Constraint-aware clamping: reduce the setpoint near/beyond safety
        # bounds. Pass the priority tier so critical loads (tier <= 2) get the
        # tighter 0.99 deadband; a priority-blind clamp would truncate tier-1
        # demand as soon as any local variable drifts past 0.85.
        if self.constraint_aware:
            new_setpoint = clamp_to_constraints(
                new_setpoint, obs, self.sector, tier=self.priority
            )

        factor = max(0.0, min(1.0, abs(new_setpoint / cap)))

        # Monotonic progress: the "no-regret switching" floor applies only
        # during restoration (target > 0). In a shedding negotiation (target <
        # 0) loads legitimately reduce factor, so the floor must not block them.
        target = self._gossip.target if self._gossip is not None else 0.0
        is_restoration = target > 0
        if self.priority > 0 and is_restoration:
            self._check_violation_cleared()

            if self.enable_monotonic_floor:
                if factor > self._restoration_floor:
                    self._restoration_floor = factor
                elif not self._constraint_violation_active:
                    factor = self._restoration_floor

            # Cold-load pickup rate limit: ramp-up only; decreases (sheds,
            # violation-driven reductions) pass through unthrottled.
            if self.enable_clpu_ramp:
                factor = self._rate_limit_increase(factor)

        # L2 priority-floor (gossip path): the component ADMM set this load's
        # served tier, so a supply-poor local group must not shed it below that
        # just to zero its own imbalance. Clamp up to ``min(L2 allocation,
        # constraint-allowed)`` — the constraint term still lets physics shed
        # it during a real violation, so floor and clamp never fight. Tiers
        # 2/3/4 only (tier 1 returned above).
        if self.enable_l2_priority_floor:
            floor = l2_effective_floor(
                self.behavior, self.context.aid, obs, self.sector, self.priority
            )
            if floor is not None and factor < floor:
                factor = floor

        # Gossip-driven regulates bypass the ``apply_regulate`` dedup on
        # purpose: the ledger in ``_GossipState.memory`` advances whether or
        # not ``behavior.act`` is called, so dedupping micro-steps would let
        # the ledger diverge from physical state and stall gossip at k_max.
        # monee's warm-start absorbs the small consecutive deltas efficiently.
        if self.behavior.has_action(self.context.aid, "regulate"):
            self.behavior.act(self.context.aid, "regulate", factor)
            # Keep the dedup cache truthful: this direct write bypasses it, and
            # a stale cache would silently drop a later L2 re-dispatch that
            # would restore this load.
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

        # Signed actuated setpoint: ``factor`` scales the (signed) nominal
        # capacity, so ``factor * cap`` is the power physically realised after
        # the clamp + floors. The caller reconciles the ledger against it.
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

        After ``_apply_setpoint`` clamps/floors the requested delta, record what
        was ACTUALLY applied so ``_gossip_total_delta`` reflects real
        consumption. A load held below its request by a constraint clamp is also
        marked *saturated* (it won't move with more λ), so the dual denominator
        excludes it and the freed supply flows to the unconstrained loads. The
        constraint stays solved — only the ledger is corrected, not the clamp.
        """
        if (
            not self.enable_actuated_ledger_writeback
            or applied_sp is None
            or self._gossip is None
        ):
            return
        actuated_delta = applied_sp - self._gossip.starting_setpoint
        # Clamp / ramp held it below the requested magnitude ⇒ constraint-bound.
        held_below = abs(actuated_delta) < abs(requested_delta) - 1e-12
        saturated = held_below or _is_saturated(actuated_delta, dmin, dmax)
        self._gossip.memory[self_key] = (
            actuated_delta, counter, self.priority, saturated,
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
        """Inline local-gen fallback for isolated agents without a co-located
        LocalGenerationFallbackRole: if this agent is a generator with
        headroom, ramp up to cover as much of the deficit as possible.
        """
        if deficit <= 0:
            return
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap >= 0:
            return  # not a generator (generators have cap < 0)
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
    """Map a raw priority and imbalance direction to a gossip round.

    - Restoration (target > 0): loads ordered by priority (lower = more
      urgent = earlier); generators wait until all load tiers have acted.
    - Reduction (target < 0): generators (priority 0) first; loads shed in
      reverse priority order (high-priority shed last).
    """
    if target < 0:
        if priority == 0:
            return 0  # generators first
        # Invert: lower number = more important = shed later (higher round).
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
    enable_actuated_ledger_writeback: bool = True,
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
        enable_actuated_ledger_writeback=enable_actuated_ledger_writeback,
    )
