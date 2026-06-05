"""Grid constraint monitoring and enforcement.

Per-agent local state estimation, conservative feasibility margins,
proactive curtailment signaling, and multi-hop constraint-state
propagation with deduplication.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import TYPE_CHECKING, Any

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors

from scare.base.diagnostics import record_event
from scare.base.model import (
    PROACTIVE_WARNING_FRACTION,
    SECTOR_CONSTRAINTS,
    SECTOR_TIMESCALE,
    AskEnergyMessage,
    BalanceProblem,
    ConstraintStateMessage,
    ConstraintViolation,
    ConstraintWarning,
    CurtailmentBid,
    CurtailmentNeed,
    CurtailmentRequest,
    ResponseEnergyMessage,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.trust import TrustLedger, TrustParams
from scare.base.util import (
    apply_regulate,
    constraint_utilization,
    has_heat_curtail_lock,
    lookup_priority,
    obs_capacity,
    obs_constraint_values,
    obs_priority,
    obs_setpoint,
    refresh_line_curtail_lock,
)
from scare.service.curtailment import (
    curtail_willingness,
    plan_auction_allocation,
    proximity_from_hops,
)
from scare.service.heat_frontier import HeatFrontierController

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# How many hops constraint state information propagates.
_DEFAULT_MAX_HOPS = 3

# Min utilization-unit change that triggers a fresh broadcast; below
# this the value is stable and the prior broadcast still holds.
_FORWARD_VALUE_TOL: float = 0.02

# Min sim-time between re-broadcasts of an unchanged value; keeps
# trust-ledger liveness ticks flowing without per-cycle flooding.
_FORWARD_FRESHNESS_S: float = 5.0

# Cache-gate tolerance for ``_monitor``: skip the whole pass when no
# variable moved by more than this and no violation is active.  Tighter
# than ``_FORWARD_VALUE_TOL`` (gates whether to look at values at all).
_VALUES_DELTA_TOL: float = 1e-4

# EMA smoothing for the local sensitivity estimate (dV/dP).  Low enough
# that noisy single samples don't swing it, high enough to adapt to
# post-failure topology changes.
_SENSITIVITY_EMA_ALPHA: float = 0.2

# Per-sector min |ΔP| before a sample is used; below this ΔV is
# noise-dominated and yields spurious sensitivity estimates.
_SENSITIVITY_MIN_DP: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,   # MW
    Sector.GAS: 1e-4,           # kg/s
    Sector.HEAT: 5e-4,          # MW (0.5 kW; registers ~30% regulation steps)
}

# Default sensitivity used before any samples have been collected.
_SENSITIVITY_DEFAULT: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,   # p.u. voltage per MW
    Sector.GAS: 0.5,            # p.u. pressure per kg/s
    Sector.HEAT: 1e-5,          # K per W
}

# Bounds on the sensitivity multiplier in the curtailment-auction
# willingness.  Sensitivity ranks loads *within* a priority tier; clamped
# (normalised by sector default) to a ≤16× tiebreaker so it stays far
# below the 1e4 tier step and priority remains lexicographic.
_SENS_MULT_MIN: float = 0.25
_SENS_MULT_MAX: float = 4.0

# Primary constraint variable per sector for sensitivity tracking.
_SECTOR_PRIMARY_VAR: dict[Sector, str] = {
    Sector.ELECTRICITY: "vm_pu",
    Sector.GAS: "pressure_pu",
    Sector.HEAT: "t_k",
}

# Sentinel bidder key for the auctioneer's OWN load.  The violating
# agent's own setpoint is the most direct lever on its junction (L0
# self-action), so it bids in its own auction.  Distinct from any
# ``str(addr)`` neighbour key.
_SELF_BID_KEY: str = "__self__"

# Curtailment-auction gating (``enable_curtail_auction_gating``).
# Consecutive rounds a variable's overshoot may fail to improve before
# the progress gate suspends re-arming it, and the min fractional-overshoot
# improvement (relative to span) that counts as progress.
_CURTAIL_NO_PROGRESS_LIMIT: int = 2
_CURTAIL_PROGRESS_TOL: float = 0.01

# Variables the auction must NOT fire on (under gating).  The auction's
# component-wide ``priority × own-sensitivity × reducible`` bidding is blind
# to WHICH node/branch is violated, so it would shed load that doesn't
# relieve the violation:
#   - ``t_k``: no load's curtailment moves another junction's return
#     temperature; the frontier controller owns this lever.
#   - ``loading_percent``: a BRANCH violation with no load to shed; the
#     dedicated line-relief path (``_send_line_overload_relief``) targets an
#     actual line endpoint, the correct lever.
# The auction still fires on node-local violations (``vm_pu``,
# ``pressure_pu``) where the node's own load and neighbours are the lever.
_CURTAIL_AUCTION_SKIP_VARS: frozenset[str] = frozenset({"t_k", "loading_percent"})

# Cross-sensitivity targeting (``enable_curtail_auction_targeting``).
# A bidder's electrical proximity to the violated origin scales its
# willingness within [PROX_MIN, PROX_MAX] — a bounded within-tier
# tiebreaker so priority stays dominant.  Proximity derives from cached
# multi-hop distance (more ``hops_remaining`` = closer); the auctioneer is
# the origin, so it self-bids at PROX_MAX.
_CURTAIL_PROX_MIN: float = 0.25
_CURTAIL_PROX_MAX: float = 4.0

# Min sim-seconds between line-relief re-assertions for the same branch
# (``enable_line_relief_reassert``); sized so re-assert never out-paces the
# gossip round it triggers.  Magnitude is recomputed from live overshoot
# each time, so it shrinks to zero as the line nears its bound (convergent).
_LINE_RELIEF_COOLDOWN_S: float = 2.0

# Aggressive per-round gain for branch-downstream line relief (vs the gentle
# 0.3 default): each round sheds a large share of downstream loads (priority
# orders WHO), walking a 10-20% overload down to ≤100% over rounds.
_LINE_RELIEF_GAIN: float = 1.5

# Reducible-draw threshold (MW) below which a downstream bidder is
# "exhausted" by the line-relief waterfall, escalating to the next tier.
_LINE_RELIEF_MIN_REDUCIBLE: float = 5e-4

# Schmitt-trigger release margin (loading-% points) for the line-relief lock
# hold.  Hysteresis: shed to ≤100%, but hold the downstream L2-clawback lock
# fresh until the line drops a full margin below the bound (≤85%), so L2
# can't re-serve the relieving loads and re-breach it (relief↔L2 limit-cycle).
# Released only on genuine headroom (topology change), not the relief's own
# settle point (~93-96%, which exists only because the loads stay shed).
_LINE_RELIEF_RELEASE_MARGIN: float = 15.0

# Heat frontier feedback period (s), faster than the heat SCADA poll so a
# rate-limited deeply-cold node converges within the run. The frontier control
# logic + tuning lives in :class:`~scare.service.heat_frontier.HeatFrontierController`.
_HEAT_FRONTIER_PERIOD_S: float = 1.0


class GridConstraintMonitor(Role):
    """Periodically checks local grid measurements against sector bounds
    and takes corrective action.  Per sector it:
    1. Reads local constraint variables (voltage, pressure, temperature).
    2. Emits ``ConstraintWarning`` when utilization exceeds
       ``PROACTIVE_WARNING_FRACTION``.
    3. Emits ``ConstraintViolation`` + ``BalanceProblem`` on a hard breach.
    4. Propagates ``ConstraintStateMessage`` to neighbours (2-3 hop picture).

    Multi-hop propagation dedupes per (origin, variable) to prevent
    exponential amplification in meshed topologies.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        node_id: Any = None,
        *,
        max_hops: int = _DEFAULT_MAX_HOPS,
        enable_curtailment_auction: bool = True,
        enable_curtail_auction_gating: bool = False,
        enable_curtail_auction_targeting: bool = False,
        enable_line_relief_reassert: bool = False,
        enable_branch_downstream_relief: bool = False,
        enable_line_relief_waterfall: bool = False,
        downstream_load_addrs: "list[Any] | None" = None,
        enable_multihop_constraint: bool = True,
        enable_heat_frontier: bool = True,
        enable_heat_priority_waterfall: bool = True,
        branch_id: Any = None,
        home_leader_addr: Any = None,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.node_id = node_id
        self.max_hops = max_hops
        self.enable_curtailment_auction = enable_curtailment_auction
        self.enable_curtail_auction_gating = enable_curtail_auction_gating
        self.enable_curtail_auction_targeting = enable_curtail_auction_targeting
        self.enable_line_relief_reassert = enable_line_relief_reassert
        self.enable_branch_downstream_relief = enable_branch_downstream_relief
        # Strict reverse-priority cascade for downstream line relief
        # (only meaningful with ``enable_branch_downstream_relief``).
        self.enable_line_relief_waterfall = enable_line_relief_waterfall
        # Loads electrically downstream of this branch — the only ones whose
        # curtailment reduces its flow.  Populated post-build for electricity
        # branch monitors under ``enable_branch_downstream_relief``.
        self._downstream_load_addrs: list[Any] = list(downstream_load_addrs or [])
        self.enable_multihop_constraint = enable_multihop_constraint
        self.enable_heat_frontier = enable_heat_frontier
        # Heat frontier priority-waterfall gate: a cold load defers its own
        # tier-blind shed while lower-priority reducible heat load remains in
        # its hydraulic region (shed lowest-priority first).
        self.enable_heat_priority_waterfall = enable_heat_priority_waterfall
        # Branch mode: ``branch_id`` set => running on a PowerLine branch
        # agent.  ``emit_event(BalanceProblem)`` is a no-op there (no
        # co-located negotiator), so on overload we send
        # ``StartBalanceNegotiation`` with a relief-MW override to
        # ``home_leader_addr`` (the endpoint group with lower priority-weighted
        # demand, picked at build time).
        self.branch_id = branch_id
        self.home_leader_addr = home_leader_addr

        # Neighbour constraint state cache:
        # (origin_addr_str, variable) -> ConstraintStateMessage
        self._neighbour_state: dict[tuple[str, str], ConstraintStateMessage] = {}

        # Heat-sector frontier controller: owns the priority-waterfall peer
        # cache (filled from heat ``t_k`` constraint-state messages) and the
        # frontier step state, and decides the regulation move toward the t_k
        # feasibility floor. See scare.service.heat_frontier.
        self._heat_frontier = HeatFrontierController(
            peer_freshness_s=2.0 * _FORWARD_FRESHNESS_S
        )

        # Dedup: (origin, variable) -> (best_hops_remaining, t_received,
        # value) already forwarded.  Bounds message volume via two rules:
        #   1. Forward an incoming copy only if its ``hops_remaining``
        #      strictly improves on what we last forwarded.
        #   2. Re-broadcast our own state only if the value moved by more
        #      than ``_FORWARD_VALUE_TOL`` or ``_FORWARD_FRESHNESS_S`` elapsed.
        # Never cleared per-cycle (that re-floods the whole group).
        self._state_forwarded: dict[
            tuple[str, str], tuple[int, float, float]
        ] = {}
        # Per-variable (t, util) of this agent's last OWN broadcast;
        # ``_propagate_state`` uses it to decide whether to flood again.
        self._last_local_broadcast: dict[str, tuple[float, float]] = {}

        # Variables with a violation emitted this episode (dedup guard).
        self._violation_emitted: set[str] = set()

        # Last-observed values per variable; ``_monitor`` short-circuits
        # when nothing moved beyond ``_VALUES_DELTA_TOL`` since the last tick.
        self._last_polled_values: dict[str, float] = {}

        # B.1: continuous coupling weights K_ij for the propagation overlay
        # (independent of the balance ledger — different topology/frequency).
        # Used to weight worst-neighbour utilisation by trust and to skip
        # forwarding to neighbours below the liveness threshold.
        poll_s = SECTOR_TIMESCALE.get(sector, {}).get("poll_period_s", 1.0)
        self._trust = TrustLedger(
            TrustParams(
                decay_rate_per_s=1.0 / max(poll_s * 8.0, 1.0),
                recover_rate=0.6,
                liveness_threshold=0.5,
                initial=1.0,
            )
        )

        # Local power-flow sensitivity: EMA of |dV/dP| from this agent's own
        # (P, V) history.  Lets the curtailment auction bid agents near the
        # violated variable more aggressively; no Jacobian/central view needed.
        self._sensitivity: float = _SENSITIVITY_DEFAULT.get(sector, 1e-3)
        self._last_p: float | None = None
        self._last_v: float | None = None

        # Curtailment auction state (auctioneer side):
        # auction_id -> {"bids": {sender_key: willingness}, "total", ...}.
        self._open_auctions: dict[str, dict[str, Any]] = {}
        # Per-variable in-flight guard (variable -> auction deadline).  A
        # persistent violation re-enters ``_request_curtailment`` every poll;
        # this prevents stacking auctions while letting curtailment iterate
        # round-by-round toward feasibility.
        self._curtail_inflight: dict[str, float] = {}
        # Progress gate (only under ``enable_curtail_auction_gating``):
        # variable -> {"best": best overshoot, "no_progress": rounds without
        # improvement}.  Suspends re-arming a lever that isn't moving its
        # constraint; reset when the variable returns in-bounds.
        self._curtail_progress: dict[str, dict[str, float]] = {}
        # Per-variable cooldown deadline for iterative line-relief re-assert
        # (``enable_line_relief_reassert``); cleared when in-bounds.
        self._relief_inflight: dict[str, float] = {}
        # Per-variable flag: the line-relief waterfall has only tier-1
        # reducible bidders left (can't relieve further without breaking the
        # hard-lock).  Stops re-arming; cleared when in-bounds.
        self._line_relief_tier1_residual: dict[str, bool] = {}

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 1.0)
        self.context.schedule_periodic_task(self._monitor, delay=poll)
        # Heat frontier controller: local feedback loop driving each heat load
        # to the regulation where t_k sits at the feasibility floor (max
        # feasible service).  Runs faster than the 5 s SCADA poll so a
        # rate-limited, deeply-cold node can converge within the run.
        if self.sector == Sector.HEAT and self.enable_heat_frontier:
            self.context.schedule_periodic_task(
                self._heat_frontier_control,
                delay=min(poll, _HEAT_FRONTIER_PERIOD_S),
            )

        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        self.context.subscribe_message(
            self,
            _wrap(self._handle_constraint_state),
            lambda msg, meta: isinstance(msg, ConstraintStateMessage)
            and msg.sector == self.sector,
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_curtailment_request),
            lambda msg, meta: isinstance(msg, CurtailmentRequest)
            and msg.sector == self.sector,
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_curtailment_need),
            lambda msg, meta: isinstance(msg, CurtailmentNeed)
            and msg.sector == self.sector,
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_curtailment_bid),
            lambda msg, meta: isinstance(msg, CurtailmentBid)
            and msg.sector == self.sector,
        )
        # Branch agents have no co-located negotiator, so the leader's
        # pre-gossip ``AskEnergyMessage`` would go unanswered.  Reply with
        # zeros — a branch is a sensor, no setpoint or flex.
        if self.branch_id is not None:
            self.context.subscribe_message(
                self,
                _wrap(self._handle_ask_energy_branch),
                lambda msg, meta: (
                    isinstance(msg, AskEnergyMessage)
                    and msg.sector == self.sector
                ),
            )

    async def _handle_ask_energy_branch(
        self, message: AskEnergyMessage, meta: dict
    ) -> None:
        """Stub zero-reply so the home leader's pre-gossip round completes."""
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=0.0,
            available=0.0,
        )
        await self.context.send_message(
            reply, receiver_addr=mango_sender_addr(meta)
        )

    # ------------------------------------------------------------------
    # Periodic monitoring
    # ------------------------------------------------------------------

    def _safe_observe(self) -> dict | None:
        """``observe()`` result, or ``None`` when the LP hasn't solved yet.
        Swallows ``AttributeError``/``KeyError`` during the bootstrap tick.
        """
        try:
            return self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return None

    def _try_emit_event(self, event) -> None:
        """Emit a local event, swallowing the ``KeyError`` mango raises when
        no co-located role subscribes (branch-mode agents have no negotiator).
        """
        try:
            self.context.emit_event(event)
        except KeyError:
            pass

    async def _handle_violation(
        self, obs: dict, var: str, val: float, lo: float, hi: float
    ) -> None:
        """Emit ``ConstraintViolation`` + ``BalanceProblem`` for a freshly
        breached variable; relief-route branch overloads to the home leader;
        (re-)arm curtailment while the violation persists.

        Event emission is deduped (one per episode, via
        ``_violation_emitted``).  Curtailment is (re-)armed on EVERY active
        poll so the round-to-round iteration drives the variable back toward
        feasibility; the in-flight guard prevents overlapping auctions.
        """
        # Branch-downstream relief owns a line overload only for a branch
        # ``loading_percent`` breach WITH a resolved downstream load set
        # (bridge branch); else fall back to legacy endpoint/auction levers.
        downstream_active = (
            self.enable_branch_downstream_relief
            and self.branch_id is not None
            and var == "loading_percent"
            and bool(self._downstream_load_addrs)
        )

        if var not in self._violation_emitted:
            self._violation_emitted.add(var)
            logger.warning(
                "[%s] CONSTRAINT VIOLATION %s=%.4f bounds=[%.4f,%.4f]",
                self.context.aid, var, val, lo, hi,
            )
            record_event(
                t=self.context.current_timestamp,
                kind="constraint_violation",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=f"{var}={val:.4f} bounds=[{lo:.4f},{hi:.4f}]",
            )
            self._try_emit_event(ConstraintViolation(
                sector=self.sector, variable=var, value=val,
                bound_low=lo, bound_high=hi, node_id=self.node_id,
            ))
            self._try_emit_event(BalanceProblem(
                sector=self.sector,
                imbalance=val - hi if val > hi else lo - val,
            ))
            # Branch-mode legacy endpoint relief (one-shot).  Deferred when
            # branch-downstream relief owns this overload, or when iterative
            # re-assert (``_reassert_line_relief``) handles it instead.
            if (
                self.branch_id is not None
                and var == "loading_percent"
                and self.home_leader_addr is not None
                and not downstream_active
                and not self.enable_line_relief_reassert
            ):
                await self._send_line_overload_relief(obs, val, lo, hi)
        # Iterative endpoint relief (re-asserts while overloaded).  Skipped
        # when branch-downstream relief owns the line.
        if (
            self.enable_line_relief_reassert
            and not downstream_active
            and self.branch_id is not None
            and var == "loading_percent"
            and self.home_leader_addr is not None
        ):
            await self._reassert_line_relief(obs, var, val, lo, hi)
        # Curtailment auction.  Skipped for gated skip-vars (``t_k`` /
        # ``loading_percent``) unless branch-downstream relief re-enables
        # ``loading_percent`` with a targeted downstream bidder set.
        if self.enable_curtailment_auction and (
            downstream_active
            or not (
                self.enable_curtail_auction_gating
                and var in _CURTAIL_AUCTION_SKIP_VARS
            )
        ):
            await self._request_curtailment(var, val, lo, hi)

    def _handle_warning(
        self, var: str, val: float, lo: float, hi: float, util: float
    ) -> None:
        """Emit ``ConstraintWarning`` for a variable above the proactive
        threshold but not yet over the bound."""
        self._try_emit_event(ConstraintWarning(
            sector=self.sector, variable=var, value=val,
            bound_low=lo, bound_high=hi, utilization=util,
            node_id=self.node_id,
        ))
        logger.debug(
            "[%s] constraint warning %s=%.4f util=%.2f",
            self.context.aid, var, val, util,
        )

    async def _monitor(self) -> None:
        obs = self._safe_observe()
        if not obs:
            return

        bounds = SECTOR_CONSTRAINTS.get(self.sector, {})
        values = obs_constraint_values(obs, self.sector)

        # Cache gate: skip the whole pass when no value moved beyond
        # ``_VALUES_DELTA_TOL`` since the last poll and no violation is
        # active.  An active violation must keep firing until it clears,
        # else downstream balance roles never see the "clear" transition.
        if not self._violation_emitted and self._last_polled_values:
            unchanged = all(
                math.isfinite(v)
                and var in self._last_polled_values
                and abs(v - self._last_polled_values[var]) < _VALUES_DELTA_TOL
                for var, v in values.items()
            )
            if unchanged and set(values) == set(self._last_polled_values):
                return
        self._last_polled_values = {
            var: float(v) for var, v in values.items() if math.isfinite(v)
        }

        self._update_sensitivity(obs)

        for var, val in values.items():
            # Skip readings the solver hasn't populated (isolated nodes
            # report t_k=0 / NaN post-failure).
            if not math.isfinite(val) or (var == "t_k" and val <= 0.0):
                continue

            lo, hi = bounds.get(var, (float("-inf"), float("inf")))
            util = constraint_utilization(val, lo, hi)

            if val < lo or val > hi:
                await self._handle_violation(obs, var, val, lo, hi)
                # Line over bound: hold the downstream L2-clawback lock fresh
                # (the auction writes ``curtail`` in bursts, so otherwise the
                # lock ages out between sheds and L2 re-serves mid-relief).
                self._hold_downstream_line_locks(var, val, hi)
            elif (
                self._is_line_relief_branch()
                and var == "loading_percent"
                and val > hi - _LINE_RELIEF_RELEASE_MARGIN
            ):
                # Hysteresis hold band: line just-cleared but lacks the release
                # margin.  Keep the lock fresh so L2 can't claw back the
                # relieving loads and re-breach; cleared only below the margin.
                self._hold_downstream_line_locks(var, val, hi)
            else:
                self._violation_emitted.discard(var)
                # Back in-bounds: clear the gates so a re-breach gets a fresh
                # round budget.
                self._curtail_progress.pop(var, None)
                self._relief_inflight.pop(var, None)
                self._line_relief_tier1_residual.pop(var, None)

            if util >= PROACTIVE_WARNING_FRACTION and var not in self._violation_emitted:
                self._handle_warning(var, val, lo, hi, util)

            if self.enable_multihop_constraint:
                await self._propagate_state(var, val, util, obs=obs)

    # ------------------------------------------------------------------
    # Multi-hop state propagation with deduplication
    # ------------------------------------------------------------------

    async def _propagate_state(
        self, variable: str, value: float, utilization: float,
        obs: dict | None = None,
    ) -> None:
        # Suppress re-broadcasts of an unchanged value unless the freshness
        # window elapsed (keeps trust-ledger liveness alive) or utilization
        # moved by more than ``_FORWARD_VALUE_TOL``.
        now = self.context.current_timestamp
        prev = self._last_local_broadcast.get(variable)
        if prev is not None:
            prev_t, prev_util = prev
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (stale or changed):
                return

        # Heat t_k broadcasts carry this load's (tier, reducible) so cold
        # neighbours can run the priority-waterfall gate.  Only set for a
        # curtailable heat load.
        prio_tier: int | None = None
        reducible: float | None = None
        if (
            self.sector == Sector.HEAT
            and variable == "t_k"
            and obs is not None
            and self.behavior.has_action(self.context.aid, "regulate")
        ):
            prio_tier = max(
                1, obs_priority(obs, behavior=self.behavior, aid=self.context.aid)
            )
            reducible = abs(
                obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
            )

        origin = self.context.addr
        msg = ConstraintStateMessage(
            sector=self.sector,
            variable=variable,
            value=value,
            utilization=utilization,
            hops_remaining=self.max_hops,
            origin_addr=origin,
            priority_tier=prio_tier,
            reducible=reducible,
        )
        origin_key = (str(origin), variable)
        self._state_forwarded[origin_key] = (self.max_hops, now, utilization)
        self._last_local_broadcast[variable] = (now, utilization)

        for addr in topology_neighbors(self, tid="groups"):
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_constraint_state(
        self, message: ConstraintStateMessage, meta: dict
    ) -> None:
        origin_key = (str(message.origin_addr), message.variable)

        # B.1: nudge the K-score of the link the message arrived on.
        sender = mango_sender_addr(meta)
        now = self.context.current_timestamp
        if sender is not None:
            self._trust.on_message_received(str(sender), now)

        self._neighbour_state[origin_key] = message

        # Heat priority-waterfall: cache the origin's (tier, reducible),
        # stamped with arrival time for freshness.
        if message.priority_tier is not None and message.reducible is not None:
            self._heat_frontier.note_peer_state(
                str(message.origin_addr), now,
                message.priority_tier, message.reducible,
            )

        # Dedup: forward only if the incoming copy improves on the last
        # forwarded one for this (origin, variable) — strictly larger
        # ``hops_remaining`` (closer to origin), freshness window elapsed,
        # or value moved beyond tolerance.
        prev = self._state_forwarded.get(origin_key)
        if prev is not None:
            prev_hops, prev_t, prev_util = prev
            improves_hops = message.hops_remaining > prev_hops
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(message.utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (improves_hops or stale or changed):
                return
        self._state_forwarded[origin_key] = (
            message.hops_remaining, now, message.utilization,
        )

        if message.hops_remaining <= 1:
            return  # TTL exhausted

        # ``enable_multihop_constraint=False`` also disables incoming-message
        # forwarding.  Required for ``component_level``, whose partition
        # collapses a sector into one group: ``topology_neighbors`` returns
        # O(N) addresses and one message fans out N·(N−1) on the first hop.
        # Cache + trust updates still fire; only redistribution stops.
        if not self.enable_multihop_constraint:
            return

        fwd = ConstraintStateMessage(
            sector=message.sector,
            variable=message.variable,
            value=message.value,
            utilization=message.utilization,
            hops_remaining=message.hops_remaining - 1,
            origin_addr=message.origin_addr,
            priority_tier=message.priority_tier,
            reducible=message.reducible,
        )
        for addr in topology_neighbors(self, tid="groups"):
            # Don't send back to origin or immediate sender.
            if addr == message.origin_addr or addr == sender:
                continue
            # B.1: skip neighbours below the liveness gate.
            if not self._trust.is_live(str(addr), now):
                continue
            await self.context.send_message(fwd, receiver_addr=addr)

    # ------------------------------------------------------------------
    # Branch-mode helpers
    # ------------------------------------------------------------------

    async def _reassert_line_relief(
        self, obs: dict, var: str, val: float, lo: float, hi: float
    ) -> None:
        """Re-send the relief target while the line stays overloaded so the
        home leader sheds round-by-round toward feasibility.

        Cooldown-guarded (``_LINE_RELIEF_COOLDOWN_S``) so it never out-paces
        the gossip round each send triggers; the magnitude (computed in
        :meth:`_send_line_overload_relief`) shrinks to zero as the line nears
        its bound (convergent).
        """
        now = self.context.current_timestamp
        deadline = self._relief_inflight.get(var)
        if deadline is not None and now < deadline:
            return
        self._relief_inflight[var] = now + _LINE_RELIEF_COOLDOWN_S
        await self._send_line_overload_relief(obs, val, lo, hi)

    async def _send_line_overload_relief(
        self, obs: dict, val: float, lo: float, hi: float
    ) -> None:
        """Send StartBalanceNegotiation with a relief-MW target.

        Line loaded at ``val`` percent, bounded ``[lo, hi]``; the target is
        the MW the home group must shed to bring it back in-range, scaled by
        the line flow (max of ``p_from_mw`` / ``p_to_mw``) so it's real MW.
        """
        if val > hi:
            overshoot_fraction = (val - hi) / 100.0
        elif val < lo:
            overshoot_fraction = (lo - val) / 100.0
        else:
            return

        flow_mw = max(
            abs(float(obs.get("p_from_mw", 0.0) or 0.0)),
            abs(float(obs.get("p_to_mw", 0.0) or 0.0)),
        )
        if flow_mw <= 1e-9:
            # No flow magnitude — fall back to a fractional signal so the
            # home leader still triggers a fresh round.
            relief_mw = overshoot_fraction
        else:
            relief_mw = flow_mw * overshoot_fraction

        # Negative target => group reduces net load by ``relief_mw``, handled
        # by the Layer-1 QP's reverse-priority curtailment schedule.
        try:
            await self.context.send_message(
                StartBalanceNegotiation(override_target=-relief_mw),
                receiver_addr=self.home_leader_addr,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[%s] line-overload relief send failed: %s",
                self.context.aid, exc,
            )

    def _is_line_relief_branch(self) -> bool:
        """True iff this monitor runs the branch-downstream line-relief lever
        (a branch agent with the flag on and a resolved downstream set)."""
        return (
            self.enable_branch_downstream_relief
            and self.branch_id is not None
            and bool(self._downstream_load_addrs)
        )

    def _hold_downstream_line_locks(self, var: str, val: float, hi: float) -> None:
        """Keep the downstream L2-clawback line locks fresh while the line is
        over (or in the release hysteresis band), so L2 can't re-serve a
        just-relieved load between sheds.  No-op unless this is the
        line-relief branch lever on a ``loading_percent`` reading at/above
        its bound."""
        if var != "loading_percent" or not self._is_line_relief_branch():
            return
        if val <= hi - _LINE_RELIEF_RELEASE_MARGIN:
            return
        now = self.context.current_timestamp
        for addr in self._downstream_load_addrs:
            aid = getattr(addr, "aid", None)
            if aid is not None:
                refresh_line_curtail_lock(self.behavior, aid, now)

    # ------------------------------------------------------------------
    # Curtailment
    # ------------------------------------------------------------------

    # Proportional gain on the normalized overshoot.  Small enough that a
    # borderline violation steps gently; persistence ratchets it up over
    # monitor cycles.  Prevents one-shot over-curtailment.
    _CURTAILMENT_GAIN: float = 0.3

    # How long the auctioneer waits for bids before allocating; short, the
    # monitor re-fires next cycle if the violation persists.
    _AUCTION_TIMEOUT_S: float = 2.0

    def _own_curtail_willingness(self, obs: dict) -> float:
        """Curtailment willingness for this agent's own load (bigger = more
        effective at absorbing curtailment).  Product of three local signals:
          - priority tier weight (``tier_priority_weight(regime=-1)``: tier 4
            1e8 / tier 3 1e4 / tier 2 1 / tier 1 0) — dominant, lexicographic;
          - bounded sensitivity multiplier (within-tier tiebreaker, see
            ``_SENS_MULT_MIN/MAX``);
          - current reducible output.

        Tier-1 LOADS (cap > 0) return exactly 0.0, not the 1e-9 floor: the
        floor would let a tier-1 self-only auction dispatch the full amount to
        self, breaking the hard-lock invariant.  Generators (cap < 0) keep the
        floor so PV stays shed-eligible under overvoltage.
        """
        from scare.service.balance import _PRIORITY_TIERS

        prio_tier = max(
            1, obs_priority(obs, behavior=self.behavior, aid=self.context.aid)
        )
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        reducible = abs(
            obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        )
        return curtail_willingness(
            priority_tier=prio_tier,
            capacity=cap,
            reducible=reducible,
            sensitivity=self._sensitivity,
            sensitivity_ref=_SENSITIVITY_DEFAULT.get(self.sector, 1e-3),
            priority_tiers=_PRIORITY_TIERS,
            sens_mult_min=_SENS_MULT_MIN,
            sens_mult_max=_SENS_MULT_MAX,
        )

    async def _request_curtailment(
        self, variable: str, value: float, lo: float, hi: float
    ) -> None:
        span = hi - lo
        if span <= 0:
            return

        # In-flight guard: a persistent violation re-enters every poll.  Skip
        # while an auction for this variable is open so rounds don't stack;
        # the round-by-round iteration (re-opened once the guard clears) is
        # what lets a gain-limited auction reach feasibility.
        now = self.context.current_timestamp
        deadline_prev = self._curtail_inflight.get(variable)
        if deadline_prev is not None and now < deadline_prev:
            return

        overshoot = (value - hi) / span if value > hi else (lo - value) / span

        # Is this a strict reverse-priority line-relief waterfall auction?
        _waterfall = (
            self.enable_line_relief_waterfall
            and self.enable_branch_downstream_relief
            and variable == "loading_percent"
            and bool(self._downstream_load_addrs)
        )

        if _waterfall:
            # The waterfall is monotone and self-terminating, so the generic
            # no-progress gate is the wrong stop (a tier-transition round can
            # briefly stall and trip it).  Stop only when the allocator reports
            # the sole reducible bidders left are tier-1 (relieving further
            # would break the hard-lock).
            if self._line_relief_tier1_residual.get(variable):
                return
        elif self.enable_curtail_auction_gating:
            # Progress gate (one check per round): if the overshoot keeps
            # failing to improve on its best, the auction isn't relieving it —
            # stop re-arming until the overshoot worsens or topology re-engages
            # the lever.  Avoids churn on an auction-unrelievable violation.
            prog = self._curtail_progress.setdefault(
                variable, {"best": float("inf"), "no_progress": 0.0}
            )
            if overshoot < prog["best"] - _CURTAIL_PROGRESS_TOL:
                prog["best"] = overshoot
                prog["no_progress"] = 0.0
            else:
                prog["no_progress"] += 1.0
                if prog["no_progress"] > _CURTAIL_NO_PROGRESS_LIMIT:
                    return

        # Total fractional reduction needed across group + self.  Two-phase
        # auction: broadcast the need, collect bids, allocate proportional to
        # willingness (priority × sensitivity × reducible).
        _downstream_line = (
            self.enable_branch_downstream_relief
            and variable == "loading_percent"
            and bool(self._downstream_load_addrs)
        )
        if _downstream_line:
            # Branch-downstream relief uses a high gain to drive a 10-20%
            # overload to feasibility in a few rounds (the gentle 0.3 schedule
            # sheds ~1%/round and can't clear it before the holon re-serves);
            # priority still orders WHO sheds, re-arming until ≤100%.
            total_amount = min(1.0, max(0.25, _LINE_RELIEF_GAIN * overshoot))
        else:
            total_amount = max(0.02, min(1.0, self._CURTAILMENT_GAIN * overshoot))

        # Seed the agent's OWN load as a candidate — it's the most direct
        # lever on its own junction.  Priority still decides absorption: a
        # high-priority self wins ~0 share until neighbours' reducible is spent.
        self_obs = self.behavior.observe(self.context.aid) or {}
        self_w_raw = (
            self._own_curtail_willingness(self_obs)
            if self.behavior.has_action(self.context.aid, "regulate")
            else None
        )
        # Drop a zero-willingness self (tier-1, or nothing to curtail) so the
        # all-zero even-split fallback in ``_allocate_auction`` can't shed it.
        self_w = (
            self_w_raw if (self_w_raw is not None and self_w_raw > 0.0) else None
        )
        # Targeting: the auctioneer is the violation origin (closest bidder),
        # so scale its self-bid by max proximity to compete on equal footing
        # with proximity-weighted neighbour bids.
        if self.enable_curtail_auction_targeting and self_w is not None:
            self_w *= _CURTAIL_PROX_MAX

        # Branch-downstream relief: bidders are the loads flowing through the
        # branch (so the shed reduces ITS loading), not the whole component.
        # Falls back to the component otherwise.
        if (
            self.enable_branch_downstream_relief
            and variable == "loading_percent"
            and self._downstream_load_addrs
        ):
            neighbors = list(self._downstream_load_addrs)
        else:
            neighbors = list(topology_neighbors(self, tid="groups"))

        if not neighbors and self_w is None:
            # Self locked and no neighbours to delegate to — nothing can be
            # allocated without breaking the hard-lock.  Clear the guard so a
            # later poll retries once neighbours/eligibility appear.
            self._curtail_inflight.pop(variable, None)
            return

        auction_id = str(uuid.uuid4())
        self._open_auctions[auction_id] = {
            "bids": {},
            "total": total_amount,
            "neighbours_contacted": len(neighbors),
            "bidders": {},   # sender_key -> addr
            "bid_meta": {},  # sender_key -> (tier, reducible)
            "var": variable,
            "self_willingness": self_w,
            "self_addr": self.context.addr,
            "waterfall": _waterfall,
        }
        self._curtail_inflight[variable] = now + self._AUCTION_TIMEOUT_S

        if not neighbors:
            # Self-only auction (isolated node / singleton group): allocate now.
            await self._allocate_auction(auction_id)
            return

        need_msg = CurtailmentNeed(
            sector=self.sector,
            total_amount=total_amount,
            auction_id=auction_id,
            origin_addr=self.context.addr,
            variable=variable,
        )
        for addr in neighbors:
            await self.context.send_message(need_msg, receiver_addr=addr)

        deadline = now + self._AUCTION_TIMEOUT_S
        self.context.schedule_timestamp_task(
            self._close_auction(auction_id), timestamp=deadline
        )

    async def _handle_curtailment_need(
        self, message: CurtailmentNeed, meta: dict
    ) -> None:
        if not self.behavior.has_action(self.context.aid, "regulate"):
            return
        obs = self.behavior.observe(self.context.aid)
        if not obs:
            return

        willingness = self._own_curtail_willingness(obs)
        # Targeting: scale by electrical proximity to the violation origin so
        # the share concentrates on loads that actually relieve THIS violation.
        # Bounded within-tier, so priority stays dominant.
        if self.enable_curtail_auction_targeting:
            willingness *= self._curtail_proximity(
                message.origin_addr, message.variable
            )
        # Carry tier + reducible so a waterfall auctioneer can shed in reverse-
        # priority order (ignored by the default proportional allocator).
        bid_tier = max(
            1, obs_priority(obs, behavior=self.behavior, aid=self.context.aid)
        )
        bid_reducible = abs(
            obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        )
        reply = CurtailmentBid(
            auction_id=message.auction_id,
            willingness=willingness,
            sector=self.sector,
            tier=bid_tier,
            reducible=bid_reducible,
        )
        await self.context.send_message(
            reply, receiver_addr=mango_sender_addr(meta)
        )

    def _curtail_proximity(self, origin_addr: Any, variable: str) -> float:
        """Bounded proximity multiplier in ``[_CURTAIL_PROX_MIN,
        _CURTAIL_PROX_MAX]`` for this bidder relative to the violation origin.

        Uses cached multi-hop distance: larger ``hops_remaining`` => fewer
        hops from origin => electrically closer => larger ∂constraint/∂Q.  No
        cached state => neutral 1.0, so targeting only redistributes toward
        demonstrably-close bidders, never starving an unknown one.
        """
        if not variable or origin_addr is None or self.max_hops <= 0:
            return 1.0
        state = self._neighbour_state.get((str(origin_addr), variable))
        if state is None:
            return 1.0
        return proximity_from_hops(
            state.hops_remaining, self.max_hops,
            prox_min=_CURTAIL_PROX_MIN, prox_max=_CURTAIL_PROX_MAX,
        )

    async def _handle_curtailment_bid(
        self, message: CurtailmentBid, meta: dict
    ) -> None:
        auction = self._open_auctions.get(message.auction_id)
        if auction is None:
            return
        sender = mango_sender_addr(meta)
        sender_key = str(sender)
        auction["bids"][sender_key] = message.willingness
        auction["bidders"][sender_key] = sender
        auction["bid_meta"][sender_key] = (  # for the waterfall allocator
            int(getattr(message, "tier", 0) or 0),
            float(getattr(message, "reducible", 0.0) or 0.0),
        )

        if len(auction["bids"]) >= auction["neighbours_contacted"]:
            await self._allocate_auction(message.auction_id)

    async def _close_auction(self, auction_id: str) -> None:
        if auction_id in self._open_auctions:
            await self._allocate_auction(auction_id)

    async def _allocate_auction(self, auction_id: str) -> None:
        auction = self._open_auctions.pop(auction_id, None)
        if auction is None:
            return
        # Clear the in-flight guard so the next poll can open the next round.
        self._curtail_inflight.pop(auction.get("var"), None)

        bids: dict[str, float] = dict(auction["bids"])
        bidders: dict[str, Any] = dict(auction["bidders"])
        total_amount: float = auction["total"]

        # Fold in the auctioneer's own bid (L0 self-curtail candidate).
        self_w = auction.get("self_willingness")
        if self_w is not None:
            bids[_SELF_BID_KEY] = self_w
            bidders[_SELF_BID_KEY] = auction.get("self_addr")

        if not bids:
            return

        async def _dispatch(key: str, addr: Any, share: float) -> None:
            if share <= 0.0:
                return
            if key == _SELF_BID_KEY:
                await self._curtail_self(share)
            elif addr is not None:
                await self.context.send_message(
                    CurtailmentRequest(sector=self.sector, amount=share),
                    receiver_addr=addr,
                )

        plan = plan_auction_allocation(
            bids,
            bidders,
            dict(auction.get("bid_meta", {})),
            total_amount,
            waterfall=bool(auction.get("waterfall")),
            min_reducible=_LINE_RELIEF_MIN_REDUCIBLE,
        )
        # Reverse-priority waterfall terminal state: only tier-1 reducible
        # bidders remain (relieving further would break the hard-lock).  Surface
        # the residual once and stop re-arming.
        if plan.tier1_exhausted:
            var = auction.get("var", "loading_percent")
            if not self._line_relief_tier1_residual.get(var):
                self._line_relief_tier1_residual[var] = True
                record_event(
                    t=self.context.current_timestamp,
                    kind="line_relief_tier1_residual",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=f"{var}: tiers 2-4 exhausted, line still over",
                )
            return

        for key, addr, share in plan.dispatches:
            await _dispatch(key, addr, share)

    async def _handle_curtailment_request(
        self, message: CurtailmentRequest, meta: dict
    ) -> None:
        await self._apply_curtail(message.amount, label="curtailed")

    async def _curtail_self(self, amount: float) -> None:
        """Apply the auctioneer's own winning share (L0 self-curtail)."""
        await self._apply_curtail(amount, label="self-curtailed")

    async def _apply_curtail(self, amount: float, *, label: str) -> None:
        if not self.behavior.has_action(self.context.aid, "regulate"):
            return
        obs = self.behavior.observe(self.context.aid)
        if not obs:
            return

        # Multiplicative reduction: amount=0.3 cuts current output by 30%.
        # Repeated requests compound toward zero, so the loop can't overshoot
        # in one step.
        current = float(obs.get("regulation", 1.0))
        amount = max(0.0, min(1.0, amount))
        new_factor = max(0.0, current * (1.0 - amount))

        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            new_factor,
            sector=self.sector.value,
            reason="curtail",
            timestamp=self.context.current_timestamp,
            priority_tier=lookup_priority(self.behavior, self.context.aid),
        )
        if applied:
            logger.info(
                "[%s] %s by %.1f%% (regulation %.3f -> %.3f)",
                self.context.aid,
                label,
                amount * 100,
                current,
                new_factor,
            )

    # ------------------------------------------------------------------
    # Heat frontier controller (serve at the t_k feasibility frontier)
    # ------------------------------------------------------------------

    async def _heat_frontier_control(self) -> None:
        """Drive this heat load's regulation to where its junction
        temperature sits at the feasibility floor (max feasible service):
        sheds a cold node to its partial frontier (not bang-bang to 0) and
        restores a warm one, using local dT/dreg sensitivity as gain.  Applies
        to ALL tiers incl. tier-1 (a partial feasible serve beats collapsing
        temperature to a zero-credited barrier).  Writes ``reason="curtail"``
        (shed) / ``"heat_recovery"`` (restore) so the heat curtail-lock makes
        the MW holon defer.

        Observation + ``apply_regulate`` plumbing live here; the step decision
        and the priority-waterfall peer gate live in ``self._heat_frontier``
        (:class:`~scare.service.heat_frontier.HeatFrontierController`).
        """
        if self.sector != Sector.HEAT:
            return
        if not self.behavior.has_action(self.context.aid, "regulate"):
            return
        obs = self._safe_observe()
        if not obs:
            return
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap <= 0:  # generator-class, nothing to curtail
            return
        bounds = SECTOR_CONSTRAINTS.get(Sector.HEAT, {}).get("t_k")
        if bounds is None:
            return
        lo, _hi = bounds
        try:
            t = float(obs.get("t_k"))
        except (TypeError, ValueError):
            return
        if not math.isfinite(t) or t <= 0.0:
            return  # junction not yet populated

        cur = float(obs.get("regulation", 1.0))
        my_tier = max(
            1, obs_priority(obs, behavior=self.behavior, aid=self.context.aid)
        )
        decision = self._heat_frontier.decide(
            t=t,
            lo=lo,
            cap=cap,
            cur=cur,
            sensitivity=self._sensitivity,
            now=self.context.current_timestamp,
            my_tier=my_tier,
            has_lock=has_heat_curtail_lock(self.behavior, self.context.aid),
            waterfall_enabled=self.enable_heat_priority_waterfall,
            aid=str(self.context.aid),
        )
        if decision is None:
            return

        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            decision.new_reg,
            sector=self.sector.value,
            reason=decision.reason,
            timestamp=self.context.current_timestamp,
            priority_tier=lookup_priority(self.behavior, self.context.aid),
        )
        if applied:
            logger.info(
                "[%s] heat frontier: t_k=%.1f target=%.1f regulation %.3f -> %.3f",
                self.context.aid, t,
                lo + HeatFrontierController.MARGIN_K, cur, decision.new_reg,
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def worst_neighbour_utilization(self) -> float:
        """Worst neighbour constraint utilization within multi-hop range,
        weighted by the link's coupling weight K_ij (B.1): ``K_ij * util``.
        A low-trust link contributes a proportionally weaker signal.
        """
        if not self._neighbour_state:
            return 0.0
        now = self.context.current_timestamp
        worst = 0.0
        for (origin_str, _var), msg in self._neighbour_state.items():
            k = self._trust.score(origin_str, now)
            weighted = k * msg.utilization
            if weighted > worst:
                worst = weighted
        return worst

    def is_locally_feasible(self) -> bool:
        """True if no local constraint is currently violated."""
        return len(self._violation_emitted) == 0

    def local_sensitivity(self) -> float:
        """Latest |dV/dP| estimate for this agent's primary constraint
        variable.  Strictly positive; defaults to a sector-typical prior
        until enough samples are collected."""
        return self._sensitivity

    def _update_sensitivity(self, obs: dict) -> None:
        var = _SECTOR_PRIMARY_VAR.get(self.sector)
        if var is None or var not in obs:
            return
        v = float(obs[var])
        if not math.isfinite(v):
            return
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        sp = obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        # Signed injection (sp negative for generators, cap < 0).
        p = sp if cap != 0.0 else 0.0
        if self._last_p is not None and self._last_v is not None:
            dp = p - self._last_p
            dv = v - self._last_v
            min_dp = _SENSITIVITY_MIN_DP.get(self.sector, 1e-6)
            if abs(dp) >= min_dp and math.isfinite(dv):
                sample = abs(dv / dp)
                # Clamp absurd jumps (post-failure snapshots) before the EMA.
                sample = min(sample, 10.0 * self._sensitivity + 1.0)
                self._sensitivity = (
                    (1.0 - _SENSITIVITY_EMA_ALPHA) * self._sensitivity
                    + _SENSITIVITY_EMA_ALPHA * sample
                )
        self._last_p = p
        self._last_v = v
