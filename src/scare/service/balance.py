from __future__ import annotations

import hashlib
import logging
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

from scare.base.diagnostics import record_negotiation
from scare.base.model import (
    AskEnergyMessage,
    AskForAvailableFlex,
    AvailableFlexAnswer,
    BalanceProblem,
    ConstraintViolation,
    ConstraintWarning,
    EnergyNegotiationMessage,
    FailureNotice,
    IslandingRequest,
    NegotiationFinishedEvent,
    ResponseEnergyMessage,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.util import (
    clamp_to_constraints,
    constraint_utilization,
    obs_capacity,
    obs_min_max,
    obs_priority,
    obs_sector,
    obs_setpoint,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

_DEFAULT_START_THRESHOLD = 1e-4
# All sectors share the same start threshold.  Monee now reports heat
# capacities in MW (``q_mw_heat`` ≈ 0.005 for residential), so a single
# stressed agent at util≈0.85 contributes ~1.25 kW = 0.00125 MW to the
# group's thermal-deficit target — comfortably above 1e-4 yet
# distinguishable from sub-kW solver noise.  The chapter's "10 W"
# literal predates the W→MW migration and was effectively masking all
# heat gossip; it has been removed.
_START_THRESHOLD: dict[Sector, float] = {}

# Per-group threshold floor: noise vs. signal in setpoint imbalance is
# fundamentally a *fraction* of group capacity, not a fixed magnitude.
# A group with Σ|cap| = 0.02 MW (4 × 5 kW residential heat) gets
# threshold ≈ 1e-4 MW (0.5 % of capacity), a group with Σ|cap| = 2 MW
# gets ≈ 0.01 MW.  Both reject sub-half-percent imbalances while
# accepting any meaningful deficit.
_THRESHOLD_CAPACITY_FRACTION: float = 0.005
# Absolute floor for empty / zero-capacity groups so we never end up
# with a zero threshold (which would treat solver noise as signal).
_THRESHOLD_ABS_FLOOR: float = 1e-6

# Heat-specific clearance utilization: above this fraction of the
# feasible band, an agent contributes its remaining headroom to the
# group's thermal-deficit target (drives the gossip negotiation).
# Below it, the agent is "comfortable" and reports zero deficit.
# Tuned slightly below the proactive-warning threshold (0.85) so
# warmups/cooldowns trigger gossip before a hard violation.
_HEAT_CLEAR_FRACTION: float = 0.6

_MAX_HOPS = 100

# Robbins-Monro step-size decay constant.  The per-step gain is
# ``gamma_s / (1 + k / k0)``: at k=0 the step matches the historical
# constant ``gamma_s``; by k=k0 it has halved; by k=k_max=100 with
# k0=20 it is roughly 1/6 of the initial value.  This satisfies
# ``Σ γ_k = ∞`` and ``Σ γ_k² < ∞``, the standard conditions for
# almost-sure convergence under the noise that constraint-feedback
# / saturation place the gossip in.  k0 ≈ n (typical group size of
# the simbench LV grids) keeps the early phase fast.
_STEP_DECAY_K0_DEFAULT: int = 20

# P2 stall-detection parameters.  The gap window holds the last
# ``_STALL_WINDOW_FACTOR · n_act`` post-update gap values; when the
# range across the window is below ``_STALL_TOL_FRACTION · |T|`` and
# the current gap still exceeds the per-group threshold, we declare
# the protocol stuck and emit IslandingRequest immediately rather
# than spinning to k_max.  Brings the termination logic in line with
# the dynamics of (F1) "saturation stalling".
_STALL_WINDOW_FACTOR: int = 2
_STALL_TOL_FRACTION: float = 0.005
_STALL_TOL_FLOOR: float = 1e-6

# Base wallclock deadline per sector.  Electricity is fastest (agents
# need few rounds), heat is slowest (high decision delay, low
# convergence rate).  The actual deadline is further scaled by the
# group size — larger groups need more rounds to converge.
_GOSSIP_TIMEOUT_BASE_S: dict[Sector, float] = {
    Sector.ELECTRICITY: 5.0,
    Sector.GAS: 15.0,
    Sector.HEAT: 30.0,
}
_GOSSIP_TIMEOUT_DEFAULT_S = 15.0
_GOSSIP_TIMEOUT_PER_AGENT_S = 0.5  # added per group member

# Neighbour is considered stale (pruned from the live set) after this
# many poll periods of silence.  Tuned per sector: heat ramps so slow
# that long gaps between messages are normal and should not trigger
# pruning, while electricity needs a tight threshold.
_HEARTBEAT_MAX_AGE_MULTIPLE: float = 8.0

# Number of discrete priority tiers for intra-sector ordering.
# Lower tier = higher urgency = participates earlier in gossip rounds.
_PRIORITY_TIERS = 10

# A single participant's delta is clipped to this multiple of the
# negotiation target magnitude.  A misbehaving or faulty agent cannot
# corrupt the ledger sum by reporting an implausibly large contribution.
_BYZANTINE_DELTA_CAP_MULTIPLE: float = 5.0


def _start_threshold(sector: Sector) -> float:
    return _START_THRESHOLD.get(sector, _DEFAULT_START_THRESHOLD)


def _heat_thermal_deficit_mw(obs: dict) -> float:
    """Return MW of demand reduction this heat agent should contribute to
    its group's thermal-deficit target.

    Computed as ``max(0, util - ϑ_clear) · |cap|`` with the dominant local
    constraint utilization; only loads (cap > 0) contribute, since heat
    generators *increasing* output relieves the constraint and is handled
    by the islanding fallback.  Capacity is read from ``q_mw_heat`` (or
    ``q_mw_set`` for branch-side heat exchangers), so the result is in
    MW and directly addable to ``obs_setpoint`` for the negotiation
    target.  Returns 0 for non-heat agents — the caller is responsible
    for restricting invocation to heat sector.
    """
    from scare.base.model import SECTOR_CONSTRAINTS

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
    # IslandingRequest and finishes with terminal "stalled".
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
        from scare.base.model import SECTOR_TIMESCALE

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

        # --- Local proactive constraint utilization ---
        # Populated by the co-located GridConstraintMonitor's
        # ConstraintWarning events.  Keyed by variable name; value is the
        # last-reported utilization in [0, 1].  Used to throttle the
        # gossip step so an agent close to a hard bound contributes less.
        self._proactive_util: dict[str, float] = {}

    def setup(self) -> None:
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
        # type.  IslandingFallbackRole is attached only to group leaders,
        # so non-leader members would crash mango.emit_event without
        # this no-op safety net.  The actual islanding logic still lives
        # on the leader; this handler just satisfies the dispatch path.
        self.context.subscribe_event(
            self, IslandingRequest, self._on_islanding_request_noop
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

    def _on_islanding_request_noop(
        self, _event: IslandingRequest, _src: Any
    ) -> None:
        # Intentionally empty — see setup() for the rationale.  The
        # leader's IslandingFallbackRole handles the actual response.
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
                    total_delta = sum(v[0] for v in self._gossip.memory.values())
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
        self._neighbour_last_seen[str(addr)] = self.context.current_timestamp

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
        """P2: terminate a stalled gossip and escalate to islanding.

        Only the originator records the ``stalled`` diary terminal so
        the ``started == Σ terminals`` invariant remains exact.
        Emits IslandingRequest with the residual deficit if this agent
        is the group leader (the same gate as in
        ``_finish_negotiation``); ``_finish_negotiation`` is then
        called with ``record_finished=False`` so it does not double-
        count a ``finished`` terminal.
        """
        if self._gossip is None:
            return
        total_delta = sum(v[0] for v in self._gossip.memory.values())
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

        With a positive target (lost-load restoration regime), higher-
        priority loads receive higher ``w`` so they saturate first as
        ``λ`` rises (waterfall: tier 1 hits ``δ_max`` before tier 2).
        Generators receive the lowest weight so they are last to shed
        when loads can't absorb the imbalance.  Sign is symmetrical
        for the lost-gen regime (``target < 0``).

        Returns 1.0 as the trivial floor when no priority is set.
        """
        p = self.priority
        P = _PRIORITY_TIERS
        if target_sign > 0:  # restoration: lost load, want to bring loads up
            if p > 0:  # load: higher priority → higher w → saturates first
                return 2.0 ** (P - min(p, P) + 1)
            # generator: lowest weight → only adjusts after loads
            return 1.0
        if target_sign < 0:  # curtailment: lost gen, want to shed low-prio loads / ramp gen
            if p > 0:  # load: REVERSE priority — low-priority sheds first
                return 2.0 ** min(p, P)
            # generator: highest weight → ramps up first
            return 2.0 ** (P + 1)
        return 1.0

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

    def _entry_responsiveness(self, prio: int, target_sign: int) -> float:
        """``a_i`` from a ledger entry's stored priority — used by the
        receiver to estimate ``Σ a_j`` for dual-step normalisation.
        """
        # Replicates ``_qp_priority_weight`` but for an arbitrary prio.
        P = _PRIORITY_TIERS
        if target_sign > 0:
            if prio > 0:
                return 2.0 ** (P - min(prio, P) + 1)
            return 1.0
        if target_sign < 0:
            if prio > 0:
                return 2.0 ** min(prio, P)
            return 2.0 ** (P + 1)
        return 1.0

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
        """Return group neighbours that have been heard from recently.

        A neighbour is considered live if either (a) we have never
        attempted contact yet (unknown → included) or (b) the last
        observed timestamp is within the heartbeat window.
        """
        all_neighbours = topology_neighbors(self, tid="groups")
        now = self.context.current_timestamp
        live = []
        for addr in all_neighbours:
            last = self._neighbour_last_seen.get(str(addr))
            if last is None or (now - last) <= self._heartbeat_max_age_s:
                live.append(addr)
        return live

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
        cap = obs_capacity(obs)
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
        """
        sp = obs_setpoint(obs)
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
            await self._start_gossip(-total_sp)

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

        # Clear violation flag at the start of each new negotiation so
        # that the monotonic floor is only breached while a violation is
        # actively present.
        self._constraint_violation_active = False

        neighbours = self._live_neighbours()
        self._touch_neighbours(neighbours)
        nid = str(uuid4())
        self_key = str(self.context.addr)

        obs = self.behavior.observe(self.context.aid) or {}
        starting_sp = obs_setpoint(obs)

        self._gossip = _GossipState(
            negotiation_id=nid,
            target=target,
            counter=0,
            current_delta=0.0,
            starting_setpoint=starting_sp,
            memory={self_key: (0.0, 0, self.priority, False)},
            is_originator=True,
        )

        if not neighbours:
            # Isolated agent: gossip can't help. Emit IslandingRequest
            # directly with the full deficit so a co-located
            # IslandingFallbackRole (or the agent itself) can activate
            # local DGs.  We also self-activate islanding inline for the
            # common case where the fallback role is absent.
            logger.info(
                "[%s] gossip skipped: singleton (target=%.4f) — escalating to islanding",
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
                from scare.base.diagnostics import record_event

                record_event(
                    t=self.context.current_timestamp,
                    kind="islanding_request",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=f"residual={target:.4f} (singleton)",
                )
                self.context.emit_event(
                    IslandingRequest(
                        sector=self.sector, residual_deficit=target
                    )
                )
                self._try_self_island(target)
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
            # peers).  Forward to a single deterministic next-hop
            # instead, just like every subsequent round.
            next_addr = _deterministic_next(neighbours, nid, 0)
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
                total_delta = sum(v[0] for v in self._gossip.memory.values())
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
                prev_total = sum(v[0] for v in self._gossip.memory.values())
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
            self._gossip = _GossipState(
                negotiation_id=nid,
                target=message.negotiation_target,
                counter=counter,
                current_delta=0.0,
                starting_setpoint=obs_setpoint(obs),
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
        cap = obs_capacity(obs)
        dmin, dmax = obs_min_max(obs)

        prev_own = self._gossip.memory.get(self_key, (0.0, 0, self.priority, False))[0]
        total_delta = sum(v[0] for v in self._gossip.memory.values())
        open_gap = target - total_delta

        # --- Constraint-aware participation scaling ---
        # Blend local utilization with the worst utilization reported by
        # a 1-N hop neighbour.  An agent near a stressed neighbourhood
        # throttles itself even if its own readings are still healthy.
        #
        # Heat-sector exception: thermal violations are dominated by the
        # *lower* bound (severed supply path → cooling junctions), and
        # there shedding the stressed agent *helps* (less demand on the
        # surviving thermal corridor).  Throttling stressed agents would
        # invert the desired response, so for heat we keep
        # participation_scale = 1 and rely on the priority ordering and
        # ``clamp_to_constraints`` for the per-step magnitude.
        participation_scale = 1.0
        if self.constraint_aware and self.sector != Sector.HEAT:
            from scare.base.model import SECTOR_CONSTRAINTS
            bounds = SECTOR_CONSTRAINTS.get(self.sector, {})
            for var, (lo, hi) in bounds.items():
                if var in obs:
                    util = constraint_utilization(float(obs[var]), lo, hi)
                    participation_scale = min(participation_scale, max(0.0, 1.0 - util))
            neigh_util = self._worst_neighbour_utilization()
            if neigh_util > 0.0:
                participation_scale = min(
                    participation_scale, max(0.0, 1.0 - neigh_util)
                )
            # Proactive-warning channel: if the co-located monitor has
            # flagged any local variable above PROACTIVE_WARNING_FRACTION,
            # the recorded utilization is already > 0.85 — translate it
            # into an additional throttle.  This is the consumer that
            # closes the open ConstraintWarning loop (previously the
            # event had no subscribers and crashed mango.emit_event).
            if self._proactive_util:
                worst_proactive = max(self._proactive_util.values())
                participation_scale = min(
                    participation_scale, max(0.0, 1.0 - worst_proactive)
                )

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
            sat_tol = 1e-9 + 1e-6 * max(abs(dmin), abs(dmax), 1.0)
            saturated = (
                new_delta <= dmin + sat_tol
                or new_delta >= dmax - sat_tol
            )
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
            total_delta_post = sum(v[0] for v in self._gossip.memory.values())
            residual = target - total_delta_post
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
                sat_tol = 1e-9 + 1e-6 * max(abs(dmin), abs(dmax), 1.0)
                saturated = (
                    new_delta <= dmin + sat_tol
                    or new_delta >= dmax - sat_tol
                )
                self._gossip.memory[self_key] = (
                    new_delta, counter, self.priority, saturated
                )
                self._gossip.current_delta = new_delta
                if cap != 0.0:
                    self._apply_setpoint(self._gossip.starting_setpoint + new_delta)

        # Recompute total after own update
        total_delta = sum(v[0] for v in self._gossip.memory.values())
        open_gap = target - total_delta

        # P2: stall detection — append the post-update gap to the window;
        # if the window range is below tolerance and the gap is still
        # above the per-group threshold, the protocol has saturated
        # without converging.  Emit IslandingRequest immediately rather
        # than spinning to k_max.
        stalled = self._update_gap_window_and_check_stall(open_gap, target)

        neighbours = self._live_neighbours()

        if stalled:
            await self._finish_negotiation_stalled()
            return

        if abs(open_gap) <= self.termination_tolerance or counter >= self.max_hops:
            await self._finish_negotiation()
        elif neighbours:
            next_addr = _deterministic_next(neighbours, nid, counter)
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
            total_delta = sum(v[0] for v in self._gossip.memory.values())
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

            # Unresolved deficit escalates to islanding fallback.  Only
            # the group leader emits the request — IslandingFallbackRole
            # is attached only there, and mango.emit_event raises
            # KeyError if no role on the same agent subscribes.  Members
            # converging with residual still surface the residual via
            # the normal NegotiationFinishedEvent broadcast.
            if (
                abs(residual) > self._per_group_threshold() * 10
                and topology_characteristic(self, tid="groups") == "leader"
            ):
                from scare.base.diagnostics import record_event

                record_event(
                    t=self.context.current_timestamp,
                    kind="islanding_request",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=f"residual={residual:.4f}",
                )
                self.context.emit_event(
                    IslandingRequest(
                        sector=self.sector, residual_deficit=residual
                    )
                )

        self.context.emit_event(
            NegotiationFinishedEvent(new_setpoint=new_sp, sector=self.sector)
        )

        neighbours = self._live_neighbours()

        # Broadcast convergence to all live group neighbours so each can
        # emit its own local event.  Pruned neighbours are skipped —
        # sending to an unreachable peer just wastes a scheduled message
        # that will stall the simulation-termination tracker.
        finished_msg = NegotiationFinishedEvent(new_setpoint=0, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(finished_msg, receiver_addr=addr)

        # Leader also notifies CP connectors
        if topology_characteristic(self, tid="groups") == "leader":
            for addr in topology_connectors(self, tid="groups"):
                await self.context.send_message(finished_msg, receiver_addr=addr)

        self._gossip = None
        self._active = False

    def flush_pending(self) -> None:
        """Record any still-active gossip as ``abandoned`` in the diary.

        Called from the scenario-level world teardown so a negotiation
        that was in flight when the simulation ended doesn't disappear
        silently from the per-event accounting.  After this call the
        ledger satisfies ``started == finished + timed_out + cancelled
        + abandoned`` for every nid.
        """
        if self._gossip is None:
            return
        if self._gossip.is_originator:
            total_delta = sum(v[0] for v in self._gossip.memory.values())
            record_negotiation(
                t=self.context.current_timestamp,
                aid=self.context.aid,
                sector=self.sector.value,
                nid=self._gossip.negotiation_id,
                event="abandoned",
                target=self._gossip.target,
                residual=self._gossip.target - total_delta,
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
        demand_by_priority: dict[int, float] = {}
        served_by_priority: dict[int, float] = {}
        for aid in member_aids:
            obs = self.behavior.observe(aid) or {}
            sector = obs_sector(obs, behavior=self.behavior, aid=aid)
            if sector is None:
                continue
            cap = obs_capacity(obs)
            sp = obs_setpoint(obs)
            available = cap - sp  # headroom
            # Per-sector breakdown for multi-dimensional ADMM
            sec_key = sector.value
            flex_by_sector[sec_key] = flex_by_sector.get(sec_key, 0.0) + available
            balance_by_sector[sec_key] = balance_by_sector.get(sec_key, 0.0) + sp
            # Priority-tier demand aggregation (loads only: cap > 0)
            if cap > 0:
                prio = obs_priority(obs)
                demand_by_priority[prio] = demand_by_priority.get(prio, 0.0) + abs(cap)
                served_by_priority[prio] = served_by_priority.get(prio, 0.0) + abs(sp)
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
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_start_balance(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        if topology_characteristic(self, tid="groups") == "leader":
            self.context.schedule_instant_task(self.trigger_balance_negotiation())

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
        cap = obs_capacity(obs)
        if cap == 0.0:
            return

        # Constraint-aware clamping: reduce the setpoint when local grid
        # measurements are near or beyond safety bounds.
        if self.constraint_aware:
            new_setpoint = clamp_to_constraints(new_setpoint, obs, self.sector)

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
            from scare.base.diagnostics import record_regulate

            self.behavior.act(self.context.aid, "regulate", factor)
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


    def _try_self_island(self, deficit: float) -> None:
        """Inline islanding for isolated agents without IslandingFallbackRole.

        If this agent is a generator with available headroom, ramp up to
        cover as much of the deficit as possible.
        """
        if deficit <= 0:
            return
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs)
        if cap >= 0:
            return  # not a generator (generators have negative capacity)
        sp = obs_setpoint(obs)
        headroom = abs(cap) - abs(sp)
        if headroom < 1e-6:
            return
        share = min(headroom, deficit)
        new_factor = min(1.0, (abs(sp) + share) / abs(cap))
        from scare.base.util import apply_regulate

        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            new_factor,
            sector=self.sector.value,
            reason="self_island",
            timestamp=self.context.current_timestamp,
        )
        if applied:
            logger.info(
                "[%s] self-island: ramped to %.1f%% (deficit=%.4f)",
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
    )
