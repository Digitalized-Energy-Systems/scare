"""Grid constraint monitoring and enforcement.

Implements the MUST-level requirements from improvements.txt:
- Local state estimation at every agent
- Conservative feasibility margins
- Proactive curtailment signaling (CAN-level)
- Multi-hop constraint state propagation with deduplication (SHOULD-level)
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
    tier_priority_weight,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# How many hops constraint state information propagates.
_DEFAULT_MAX_HOPS = 3

# Minimum change in the monitored value (post-normalisation, in
# constraint-utilization units) that triggers a fresh broadcast.  Below
# this, the value is considered stable and the previous broadcast still
# represents network state.  Cuts redundant per-tick floods.
_FORWARD_VALUE_TOL: float = 0.02

# Minimum sim-time between re-broadcasts of an unchanged value.  Keeps
# liveness ticks flowing through the trust ledger while preventing the
# per-cycle flood that ``_state_forwarded.clear()`` used to enable.
_FORWARD_FRESHNESS_S: float = 5.0

# Cache-gate tolerance for ``_monitor``: when no constraint variable's
# value has moved by more than this since the last poll AND no
# violation is currently active, the entire monitor body skips.
# Tighter than ``_FORWARD_VALUE_TOL`` because here we are gating
# whether to even look at the values at all — the propagation guard
# downstream uses the coarser tolerance.
_VALUES_DELTA_TOL: float = 1e-4

# EMA smoothing factor for the local sensitivity estimate (dV / dP).
# 0 = never update, 1 = replace with latest sample.  A low value keeps
# noisy single samples from swinging the estimate, but high enough that
# the agent can adapt when topology changes after a failure.
_SENSITIVITY_EMA_ALPHA: float = 0.2

# Minimum |ΔP| required before a sample is used — below this the
# corresponding ΔV is dominated by measurement noise and would produce
# spurious sensitivity estimates.  Per sector.
_SENSITIVITY_MIN_DP: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,   # MW
    Sector.GAS: 1e-4,           # kg/s
    # Heat ``obs_setpoint`` is ``q_mw_heat`` in MW and individual heat
    # loads are ~0.0075–0.05 MW, so a full regulation swing moves P by at
    # most ~0.05 MW.  The previous 0.5 (calibrated as if P were in W) was
    # 10–60× above any achievable ΔP, so ``_update_sensitivity`` never
    # fired and ``_sensitivity`` stayed pinned at the 1e-5 default —
    # silently disabling the sensitivity term of the curtailment-auction
    # willingness.  5e-4 MW (0.5 kW) registers the ~30 % regulation steps
    # the auction applies while staying above measurement noise.
    Sector.HEAT: 5e-4,          # MW
}

# Default sensitivity used before any samples have been collected.
_SENSITIVITY_DEFAULT: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,   # p.u. voltage per MW
    Sector.GAS: 0.5,            # p.u. pressure per kg/s
    Sector.HEAT: 1e-5,          # K per W
}

# Bounds on the sensitivity multiplier in the curtailment-auction
# willingness.  Sensitivity ranks loads *within* a priority tier by how
# effectively curtailing them moves the violated variable, but the raw
# EMA can span many orders of magnitude — left unbounded and multiplied
# in, it would overcome the (1e4-per-step) priority tier weights and
# invert the waterfall (shed a higher-priority but more-sensitive load
# before a lower-priority one).  Normalising by the sector default and
# clamping to [0.25, 4] keeps sensitivity a ≤16× within-tier tiebreaker,
# far below the 1e4 tier step, so priority stays lexicographic.
_SENS_MULT_MIN: float = 0.25
_SENS_MULT_MAX: float = 4.0

# Primary constraint variable per sector for sensitivity tracking.
_SECTOR_PRIMARY_VAR: dict[Sector, str] = {
    Sector.ELECTRICITY: "vm_pu",
    Sector.GAS: "pressure_pu",
    Sector.HEAT: "t_k",
}

# Sentinel bidder key for the auctioneer's OWN load in a curtailment
# auction.  The violating agent's own setpoint is the most direct lever
# on its own junction (L0 self-action), so it competes in its own auction
# as a bidder rather than only ever curtailing neighbours.  Distinct from
# any ``str(addr)`` neighbour key.
_SELF_BID_KEY: str = "__self__"

# Curtailment-auction gating (``enable_curtail_auction_gating``).
# Consecutive auction rounds a variable may run without its overshoot
# improving before the progress gate suspends re-arming it.
_CURTAIL_NO_PROGRESS_LIMIT: int = 2
# Minimum improvement in the fractional overshoot (relative to the
# constraint span) that counts as progress for the gate above.
_CURTAIL_PROGRESS_TOL: float = 0.01

# Variables the auction must NOT fire on (under gating).  Two cases, same
# reason — the auction's component-wide ``priority × own-sensitivity ×
# reducible`` bidding is blind to WHICH node/branch is violated, so it sheds
# whatever load is most "willing" regardless of whether that load relieves
# the violation:
#   - ``t_k``           heat temperature: no load's curtailment moves another
#                       junction's return temperature — the frontier controller
#                       owns this lever.
#   - ``loading_percent`` line overload is a BRANCH violation; the branch has
#                       no load to shed, so the auction farms it out to the
#                       most-willing load in the component, which need not be
#                       on (or even near) the overloaded line.  The dedicated
#                       line-relief path (``_send_line_overload_relief``) targets
#                       an actual endpoint of the line and is the correct lever;
#                       the auction here only sheds unrelated nodes (measured
#                       net-negative on the settled worst-line loading).
# The auction still fires on node-local violations (``vm_pu``, ``pressure_pu``)
# where the violating node's OWN load and its electrical neighbours are the
# lever.
_CURTAIL_AUCTION_SKIP_VARS: frozenset[str] = frozenset({"t_k", "loading_percent"})

# Cross-sensitivity targeting (``enable_curtail_auction_targeting``).
# A bidder's electrical proximity to the violated origin scales its
# willingness within [PROX_MIN, PROX_MAX] — a bounded within-tier
# tiebreaker (like the sensitivity multiplier) so priority stays
# lexicographically dominant.  Proximity is derived from the cached
# multi-hop ``ConstraintStateMessage`` distance: a node that received the
# state with more ``hops_remaining`` is closer to the origin.  The
# auctioneer itself IS the origin, so it self-bids at PROX_MAX.
_CURTAIL_PROX_MIN: float = 0.25
_CURTAIL_PROX_MAX: float = 4.0

# Minimum sim-seconds between two line-relief re-assertions for the same
# branch (``enable_line_relief_reassert``).  Sized so the re-assert never
# out-paces the gossip round it triggers — modelled on the 2 s
# SlackBudgetMonitor refire cooldown that damps the balance layer.  The
# relief magnitude is recomputed from the live overshoot each time, so as
# the line approaches its bound the ask shrinks to zero (convergent, no
# over-shed); a still-overloaded line keeps drawing fresh relief rounds.
_LINE_RELIEF_COOLDOWN_S: float = 2.0

# Aggressive per-round gain for branch-downstream line relief (vs the gentle
# 0.3 default).  The downstream auction must clear a 10-20 % overload within
# the run, so each round sheds a large share of the downstream loads
# (priority orders WHO sheds first); the round-by-round re-arm then walks the
# line down to ≤100 %.  See ``_request_curtailment``.
_LINE_RELIEF_GAIN: float = 1.5

# Reducible-draw threshold (MW) below which a downstream bidder is treated as
# "exhausted" by the line-relief waterfall, so the cascade escalates to the
# next priority tier.  Matches the 0.5 kW bid-registration floor.
_LINE_RELIEF_MIN_REDUCIBLE: float = 5e-4

# Schmitt-trigger release margin (loading-percent points) for the line-relief
# lock hold.  The relief auction engages while the line is over its bound
# (loading > 100 %) but the downstream loads' L2-clawback lock is held fresh
# until the line drops a further ``_LINE_RELIEF_RELEASE_MARGIN`` below the
# bound — so once the line just-clears, L2 cannot immediately re-serve the
# relieving loads and re-breach it (the relief↔L2 limit-cycle).  Hysteresis:
# shed to ≤100 %, but don't release the shed until the line has ≥ this margin.
#
# Sized above the relief's typical settle point: the tapering per-round shed
# walks the line to just under 100 % (observed settle ~97-99 %), and that
# headroom exists *only because* the loads are held shed — restoring them would
# re-breach the line — so the hold must persist there.  Released only when the
# line sits a full margin below the bound (≤ 95 %), i.e. genuine headroom from
# a topology change, not the relief's own success.
_LINE_RELIEF_RELEASE_MARGIN: float = 5.0


class GridConstraintMonitor(Role):
    """Periodically checks local grid measurements against sector-specific
    bounds and takes corrective action.

    For each sector the agent participates in, it:
    1. Reads local constraint variables (voltage, pressure, temperature).
    2. Emits a ``ConstraintWarning`` event when utilization exceeds
       ``PROACTIVE_WARNING_FRACTION`` (proactive curtailment signaling).
    3. Emits a ``ConstraintViolation`` event and triggers a
       ``BalanceProblem`` when a hard bound is breached.
    4. Propagates ``ConstraintStateMessage`` to neighbours so they can
       build a 2-3 hop picture of constraint tightness.

    Multi-hop propagation includes deduplication: each message carries
    the origin address, and each agent tracks which (origin, variable)
    pairs it has already forwarded per generation counter.  This
    prevents exponential amplification in meshed topologies.
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
        # Strict reverse-priority cascade for the downstream line-relief
        # auction (only meaningful with ``enable_branch_downstream_relief``).
        self.enable_line_relief_waterfall = enable_line_relief_waterfall
        # Addresses of the loads electrically downstream of this branch (the
        # subtree fed through it) — the only loads whose curtailment reduces
        # this branch's flow.  Populated post-build for electricity branch
        # monitors when ``enable_branch_downstream_relief``; empty otherwise.
        self._downstream_load_addrs: list[Any] = list(downstream_load_addrs or [])
        self.enable_multihop_constraint = enable_multihop_constraint
        self.enable_heat_frontier = enable_heat_frontier
        # Priority-waterfall gate for the heat frontier controller: a cold
        # heat load defers its own (tier-blind) shed while lower-priority
        # reducible heat load remains in its hydraulic region, so shedding
        # follows the priority order (lowest-priority first).  See
        # ``_heat_frontier_control``.
        self.enable_heat_priority_waterfall = enable_heat_priority_waterfall
        # Branch mode: when ``branch_id`` is set the monitor is running
        # on a PowerLine branch agent.  Local ``emit_event(BalanceProblem)``
        # is a no-op on a branch (no co-located EnergyBalanceNegotiator),
        # so on overload we instead send ``StartBalanceNegotiation`` with
        # an explicit relief-MW override target to ``home_leader_addr``.
        # The home leader is picked at scenario-build time as the
        # endpoint group with the lower priority-weighted demand.
        self.branch_id = branch_id
        self.home_leader_addr = home_leader_addr

        # Neighbour constraint state cache:
        # (origin_addr_str, variable) -> ConstraintStateMessage
        self._neighbour_state: dict[tuple[str, str], ConstraintStateMessage] = {}

        # Heat priority-waterfall peer cache: origin_addr_str ->
        # (t_received, priority_tier, reducible).  Populated from heat
        # ``t_k`` ConstraintStateMessages that carry the priority-coordination
        # fields; read by the frontier controller's deferral gate with a
        # freshness window so a peer that has since shed (and stopped
        # re-broadcasting) ages out instead of pinning a stale defer.
        self._heat_peer_state: dict[str, tuple[float, int, float]] = {}

        # Deduplication: track which (origin, variable) we have already
        # forwarded with the (best_hops_remaining, t_received, value)
        # triple.  Two suppress rules combine to bound message volume
        # without losing freshness:
        #   1. We only forward an incoming copy if its ``hops_remaining``
        #      strictly improves on what we last forwarded.
        #   2. We only re-broadcast our own state if the value moved by
        #      more than ``_FORWARD_VALUE_TOL`` *or* the freshness window
        #      ``_FORWARD_FRESHNESS_S`` has elapsed.
        # Earlier revisions cleared this dict every monitor cycle, which
        # made each cycle re-flood the entire group with identical
        # information — task 0 produced 119 480 ``ConstraintStateMessage``
        # in 5 s on simbench_lv (≈ 24 k/s).  Per-cycle clearing is
        # explicitly removed; freshness is governed by the rules above.
        self._state_forwarded: dict[
            tuple[str, str], tuple[int, float, float]
        ] = {}
        # When this agent last broadcast its OWN state, indexed by
        # variable.  Used by ``_propagate_state`` to decide whether the
        # local poll has produced a new-enough datum to flood again.
        self._last_local_broadcast: dict[str, tuple[float, float]] = {}

        # Track whether we already emitted a violation this cycle to
        # avoid flooding.
        self._violation_emitted: set[str] = set()

        # Cache of last-observed constraint values per variable.  Used
        # by the polling watchdog (``_monitor``) to short-circuit when
        # nothing has moved beyond ``_VALUES_DELTA_TOL`` since the
        # last tick — most ticks on a steady grid would otherwise
        # repeat identical work (violation check, sensitivity update,
        # propagation guard) for no signal change.
        self._last_polled_values: dict[str, float] = {}

        # Heat frontier controller: sign of the last applied regulation step,
        # used to damp limit cycles (a too-large step relative to the
        # under-estimated dT/dreg can overshoot the feasibility band; halving
        # on each direction reversal makes the load converge to its frontier
        # instead of ping-ponging).
        self._frontier_last_dir: float = 0.0

        # B.1: continuous coupling weights K_ij for the constraint
        # propagation overlay.  Independent of the balance negotiator's
        # ledger because the topology and message frequencies differ.
        # Used to (a) weight the worst-neighbour utilisation by trust,
        # (b) skip forwarding to neighbours whose K is below the
        # liveness threshold.
        poll_s = SECTOR_TIMESCALE.get(sector, {}).get("poll_period_s", 1.0)
        self._trust = TrustLedger(
            TrustParams(
                decay_rate_per_s=1.0 / max(poll_s * 8.0, 1.0),
                recover_rate=0.6,
                liveness_threshold=0.5,
                initial=1.0,
            )
        )

        # --- Local power-flow sensitivity estimate ---
        # EMA of |dV/dP| observed from this agent's own (P, V) history.
        # Used by the curtailment auction so agents near the violated
        # variable bid more aggressively than those with little
        # influence.  No Jacobian / central view required.
        self._sensitivity: float = _SENSITIVITY_DEFAULT.get(sector, 1e-3)
        self._last_p: float | None = None
        self._last_v: float | None = None

        # --- Curtailment auction state (auctioneer side) ---
        # auction_id -> {"bids": {sender_key: willingness}, "total": float,
        #                "neighbours_contacted": int, "deadline": float}
        self._open_auctions: dict[str, dict[str, Any]] = {}
        # Per-variable in-flight guard: variable -> auction deadline.  A
        # persistent violation re-enters ``_request_curtailment`` every
        # monitor poll; this prevents stacking overlapping auctions for the
        # same variable while letting curtailment ITERATE round-by-round
        # toward feasibility once the previous round has allocated.
        self._curtail_inflight: dict[str, float] = {}
        # Progress gate state (only consulted when
        # ``enable_curtail_auction_gating``): variable -> {"best": best
        # fractional overshoot seen, "no_progress": consecutive rounds
        # without improvement}.  Reset when the variable returns in-bounds
        # (see ``_monitor``).  Suspends re-arming a lever that isn't
        # moving its constraint — see ``_request_curtailment``.
        self._curtail_progress: dict[str, dict[str, float]] = {}
        # Per-variable cooldown deadline for iterative line-relief re-assert
        # (``enable_line_relief_reassert``); cleared when the line returns
        # in-bounds (see ``_monitor``).
        self._relief_inflight: dict[str, float] = {}
        # Per-variable flag set by the line-relief waterfall allocator when the
        # only downstream bidders with reducible draw left are tier-1 (the line
        # cannot be relieved further without breaking the hard-lock).  Stops the
        # auction re-arming; cleared when the line returns in-bounds.
        self._line_relief_tier1_residual: dict[str, bool] = {}

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 1.0)
        self.context.schedule_periodic_task(self._monitor, delay=poll)
        # Heat frontier controller: drive each heat load to the regulation
        # where its t_k sits at the feasibility floor (max feasible service),
        # both shedding cold nodes to the partial frontier and restoring
        # recovered ones.  Supersedes the bang-bang gate behaviour.  Runs as
        # a local feedback loop at ``_HEAT_FRONTIER_PERIOD_S`` — faster than
        # the 5 s SCADA heat-decision poll, because each rate-limited step
        # only moves regulation a little and a deeply-cold node needs several
        # steps to converge to its frontier (t_k updates every energy-flow
        # recompute, well inside this period).  See ``_heat_frontier_control``.
        if self.sector == Sector.HEAT and self.enable_heat_frontier:
            self.context.schedule_periodic_task(
                self._heat_frontier_control,
                delay=min(poll, self._HEAT_FRONTIER_PERIOD_S),
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
        # Branch agents have no co-located EnergyBalanceNegotiator, so
        # the group leader's pre-gossip ``AskEnergyMessage`` would go
        # unanswered and the leader's response-count would never hit
        # ``_trigger_expected``.  Reply with zeros from here — the
        # branch contributes no setpoint and no flex; it's a sensor.
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
        """Stub reply so the home group leader's pre-gossip round
        completes promptly when the branch sits in its group topology.
        """
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
        """Return ``observe()`` result or ``None`` when the LP hasn't solved
        yet.  Swallows ``AttributeError``/``KeyError`` so periodic tasks
        don't log 200+ tracebacks during the bootstrap tick.
        """
        try:
            return self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return None

    def _try_emit_event(self, event) -> None:
        """Emit a local event, swallowing the ``KeyError`` that mango
        raises when no co-located role subscribes (branch-mode monitor
        agents have no EnergyBalanceNegotiator).
        """
        try:
            self.context.emit_event(event)
        except KeyError:
            pass

    async def _handle_violation(
        self, obs: dict, var: str, val: float, lo: float, hi: float
    ) -> None:
        """Emit ``ConstraintViolation`` + ``BalanceProblem`` for a freshly
        breached variable; relief-route branch overloads to the home
        leader; (re-)arm curtailment while the violation persists.

        Event emission is deduped (one ``ConstraintViolation`` /
        ``BalanceProblem`` per episode, via ``_violation_emitted``) to avoid
        flooding the bus.  Curtailment, by contrast, is (re-)armed on EVERY
        poll the violation is still active so a single gain-limited auction
        round does not have to clear the whole violation by itself — the
        round-to-round iteration drives the variable back toward feasibility.
        The in-flight guard in ``_request_curtailment`` keeps that from
        stacking overlapping auctions for the same variable.
        """
        # Branch-downstream relief owns a line overload only when this is a
        # branch ``loading_percent`` breach AND a downstream load set was
        # resolved for it (a bridge branch); otherwise fall back to the
        # legacy endpoint/auction levers.
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
            # Branch-mode line-relief.  When branch-downstream relief owns
            # this overload (it has a non-empty downstream set), the
            # downstream-targeted auction below is the lever and BOTH legacy
            # relief paths defer; otherwise the legacy endpoint relief fires
            # (one-shot here, or iteratively via ``_reassert_line_relief``).
            if (
                self.branch_id is not None
                and var == "loading_percent"
                and self.home_leader_addr is not None
                and not downstream_active
                and not self.enable_line_relief_reassert
            ):
                await self._send_line_overload_relief(obs, val, lo, hi)
        # Iterative endpoint relief (legacy lever, made to re-assert while
        # overloaded).  Skipped when branch-downstream relief owns the line.
        if (
            self.enable_line_relief_reassert
            and not downstream_active
            and self.branch_id is not None
            and var == "loading_percent"
            and self.home_leader_addr is not None
        ):
            await self._reassert_line_relief(obs, var, val, lo, hi)
        # Curtailment auction.  Skipped for the gated skip-vars (``t_k`` /
        # ``loading_percent``) EXCEPT when branch-downstream relief re-enables
        # ``loading_percent`` with a targeted (downstream) bidder set.
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
        threshold but not yet over the bound.
        """
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

        # Cache gate: skip the whole monitor pass when no value has
        # moved beyond ``_VALUES_DELTA_TOL`` since the last poll AND
        # there is no active violation that still needs re-evaluation
        # (an active violation must keep firing until the value drops
        # back inside bounds, otherwise downstream balance roles never
        # see the "clear" transition).
        if not self._violation_emitted and self._last_polled_values:
            unchanged = all(
                math.isfinite(v)
                and var in self._last_polled_values
                and abs(v - self._last_polled_values[var]) < _VALUES_DELTA_TOL
                for var, v in values.items()
            )
            if unchanged and set(values) == set(self._last_polled_values):
                return
        # Update cache for the next tick's comparison.
        self._last_polled_values = {
            var: float(v) for var, v in values.items() if math.isfinite(v)
        }

        # Deduplication state is now persistent across cycles — see
        # ``_propagate_state`` / ``_handle_constraint_state`` for the
        # value-delta + freshness-window suppression rules that replace
        # the per-cycle ``clear()``.

        # Update local sensitivity estimate from own (P, V) history.
        self._update_sensitivity(obs)

        for var, val in values.items():
            # Skip readings the solver hasn't populated (post-failure
            # infeasibility reports t_k=0, NaN on isolated nodes).
            if not math.isfinite(val) or (var == "t_k" and val <= 0.0):
                continue

            lo, hi = bounds.get(var, (float("-inf"), float("inf")))
            util = constraint_utilization(val, lo, hi)

            if val < lo or val > hi:
                await self._handle_violation(obs, var, val, lo, hi)
                # Line over its bound: hold the downstream loads' L2-clawback
                # lock fresh this poll (the auction only writes ``curtail`` in
                # bursts, so without this the lock ages out between sheds and
                # L2 re-serves mid-relief).
                self._hold_downstream_line_locks(var, val, hi)
            elif (
                self._is_line_relief_branch()
                and var == "loading_percent"
                and val > hi - _LINE_RELIEF_RELEASE_MARGIN
            ):
                # Hysteresis hold band: the line has just-cleared but lacks the
                # release margin.  Keep the lock fresh (no further shed) so L2
                # cannot immediately claw back the relieving loads and
                # re-breach the line.  State is NOT cleared until the line
                # drops a full margin below the bound (the else branch).
                self._hold_downstream_line_locks(var, val, hi)
            else:
                self._violation_emitted.discard(var)
                # Variable back in-bounds (with margin): clear the progress
                # gate so a later re-breach starts with a fresh round budget.
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
        # Suppress re-broadcasts of an unchanged value unless the
        # freshness window has elapsed (keeps trust-ledger liveness alive)
        # OR the utilization moved by more than ``_FORWARD_VALUE_TOL``.
        # Without this, every monitor cycle re-broadcasts the same value
        # to every neighbour, swamping the network — task-0 baseline had
        # 119 480 ConstraintStateMessages in 5 s on this path alone.
        now = self.context.current_timestamp
        prev = self._last_local_broadcast.get(variable)
        if prev is not None:
            prev_t, prev_util = prev
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (stale or changed):
                return

        # Heat t_k broadcasts carry this load's (tier, reducible) so cold
        # neighbours can run the priority-waterfall gate in
        # ``_heat_frontier_control``.  Only meaningful for a curtailable
        # heat load; left ``None`` otherwise.
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

        # Cache the latest state from this origin
        self._neighbour_state[origin_key] = message

        # Heat priority-waterfall: cache the origin's (tier, reducible) when
        # the message carries them, stamped with arrival time for freshness.
        if message.priority_tier is not None and message.reducible is not None:
            self._heat_peer_state[str(message.origin_addr)] = (
                now, message.priority_tier, message.reducible,
            )

        # --- Deduplication ---
        # Forward only if the incoming copy improves on what we've
        # already forwarded for this (origin, variable): either
        # ``hops_remaining`` is strictly larger (fresher / closer to
        # origin), or the freshness window has elapsed, or the value
        # moved by more than the tolerance.  Updated lazily; never
        # cleared (the per-cycle clear caused the message flood).
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

        # ``enable_multihop_constraint=False`` also disables incoming-
        # message forwarding (not just the own-state broadcast in
        # ``_propagate_state``).  Required for ``component_level``,
        # whose ``connected_component`` partition collapses an entire
        # sector into one ``tid="groups"`` group: ``topology_neighbors``
        # then returns O(N) addresses per agent, and a single message
        # fans out N · (N−1) on the first hop alone, OOM-killing the
        # worker (8.5 M log lines / 9 min on simbench_lv with N≈300).
        # Cache + trust-score updates above still fire so the local
        # picture stays current; only the redistribution stops.
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
            # Don't send back to the origin or the immediate sender
            if addr == message.origin_addr or addr == sender:
                continue
            # B.1: skip forwarding to neighbours below the liveness gate.
            if not self._trust.is_live(str(addr), now):
                continue
            await self.context.send_message(fwd, receiver_addr=addr)

    # ------------------------------------------------------------------
    # Branch-mode helpers
    # ------------------------------------------------------------------

    async def _reassert_line_relief(
        self, obs: dict, var: str, val: float, lo: float, hi: float
    ) -> None:
        """Iterative line-relief: re-send the relief target while the line
        stays overloaded, so the home leader sheds round-by-round toward
        feasibility instead of stopping after the legacy single shot.

        Cooldown-guarded (``_LINE_RELIEF_COOLDOWN_S``) so we never re-assert
        faster than the gossip round each send triggers; the magnitude is
        recomputed from the live overshoot inside
        :meth:`_send_line_overload_relief`, so it shrinks to zero as the line
        approaches its bound (convergent — no over-shed) and keeps drawing
        fresh relief while the line is still over.
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

        The line is loaded at ``val`` percent and bounded at
        ``[lo, hi]``; the relief target is the MW the home group must
        absorb to bring the line back into the feasible range.  The
        flow magnitude is read from the line's ``p_from_mw`` /
        ``p_to_mw`` (whichever larger), so the result is in real MW
        and directly usable as a gossip target.
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
            # No flow magnitude available — fall back to a fractional
            # signal so the home leader still triggers a fresh round.
            relief_mw = overshoot_fraction
        else:
            relief_mw = flow_mw * overshoot_fraction

        # Negative target ⇒ the group must reduce net load by
        # ``relief_mw``, which the Layer-1 QP handles in curtailment
        # regime via the reverse-priority schedule.
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
        """Keep the downstream loads' L2-clawback line locks fresh while the
        line is over (or in the release hysteresis band), so L2 cannot
        re-serve a just-relieved load between the auction's sheds.  No-op
        unless this is the line-relief branch lever on a ``loading_percent``
        reading at/above its bound."""
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

    # Proportional-controller gain applied to the normalized overshoot.
    # Small enough that a borderline violation produces a gentle step;
    # if the violation persists, the next monitor cycle re-emits and
    # the response ratchets up. Prevents one-shot over-curtailment.
    _CURTAILMENT_GAIN: float = 0.3

    # How long the auctioneer waits for bids before allocating with
    # whatever it has.  Short — the monitor re-fires on the next cycle
    # if the violation persists.
    _AUCTION_TIMEOUT_S: float = 2.0

    def _own_curtail_willingness(self, obs: dict) -> float:
        """Curtailment willingness for *this* agent's own load.

        Bigger = more happy / more effective at absorbing curtailment.
        Combines three purely local signals:
          - priority tier weight (``tier_priority_weight(regime=-1)``:
            tier 4 → 1e8, tier 3 → 1e4, tier 2 → 1, tier 1 → 0; tier-1's 0
            falls through the guard to a 1e-9 floor so tier-1 loads
            effectively never win, matching the hard-lock invariant) — the
            dominant, lexicographic term;
          - a bounded sensitivity multiplier (high = curtailment here
            cheaply moves the violated variable) — normalised by the
            sector default and clamped to ``[_SENS_MULT_MIN,
            _SENS_MULT_MAX]`` so it ranks loads *within* a tier without
            overcoming the 1e4 tier step (see those constants);
          - current reducible output (nothing to curtail → nothing to give).

        Tier-1 LOADS (cap > 0) return exactly 0.0 instead of the 1e-9
        floor.  The floor leaked in self-only auctions: with tier-1 self
        as the only bidder, ``sum_w = 1e-9 > 0`` and the allocator
        dispatched the full violation amount to self, defeating the
        hard-lock invariant at ``base/util.py:1080-1086`` (eval task
        1556: child-21/89 ratcheted 0.90 → 0.05).  Generators (cap < 0)
        still get the 1e-9 floor — they must remain shed-eligible under
        overvoltage so PV self-curtail under qv_droop violations works.
        """
        from scare.service.balance import _PRIORITY_TIERS

        prio_tier = max(
            1, obs_priority(obs, behavior=self.behavior, aid=self.context.aid)
        )
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if prio_tier <= 1 and cap > 0:
            return 0.0
        prio_weight = tier_priority_weight(
            prio_tier, regime=-1, priority_tiers=_PRIORITY_TIERS,
        )
        reducible = abs(
            obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid)
        )
        sens_ref = _SENSITIVITY_DEFAULT.get(self.sector, 1e-3)
        sens_mult = (
            self._sensitivity / sens_ref if sens_ref > 0.0 else 1.0
        )
        if not math.isfinite(sens_mult) or sens_mult <= 0.0:
            sens_mult = 1.0
        sens_mult = max(_SENS_MULT_MIN, min(_SENS_MULT_MAX, sens_mult))
        willingness = prio_weight * sens_mult * reducible
        if not math.isfinite(willingness) or willingness <= 0.0:
            willingness = 1e-9
        return willingness

    async def _request_curtailment(
        self, variable: str, value: float, lo: float, hi: float
    ) -> None:
        span = hi - lo
        if span <= 0:
            return

        # In-flight guard: a persistent violation re-enters here every poll
        # (``_handle_violation`` re-arms us unconditionally).  Skip while an
        # auction for this variable is still open so rounds don't stack;
        # once it allocates the guard clears and the next poll opens the
        # next round — that round-by-round iteration is what lets a
        # gain-limited auction actually reach feasibility.
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
            # The waterfall is monotone and self-terminating (it re-arms only
            # while the line is over and stops once it clears), so the generic
            # no-progress gate is the wrong stop — a tier-transition round can
            # briefly stall and the gate would abort before the next tier
            # engages.  Instead stop only when the allocator has reported that
            # the sole reducible bidders left are tier-1 (relieving further
            # would break the hard-lock).
            if self._line_relief_tier1_residual.get(variable):
                return
        elif self.enable_curtail_auction_gating:
            # Progress gate: the in-flight guard above means we reach here once
            # per auction round.  If this variable's overshoot keeps failing to
            # improve on its best-seen value, the auction is not relieving it —
            # stop re-arming until something else moves it (the overshoot
            # worsening or a topology change re-engages the lever).  Without
            # this a persistent, auction-unrelievable violation re-arms every
            # poll and churns (2.5–5× more applies than distinct violations,
            # mostly at the gain floor).  No-op unless gating is enabled.
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

        # Total fractional reduction needed across the group + self.
        # Announced via a two-phase auction: broadcast the *need*, collect
        # bids, then allocate proportional to each candidate's willingness
        # (priority × local sensitivity × reducible output).
        _downstream_line = (
            self.enable_branch_downstream_relief
            and variable == "loading_percent"
            and bool(self._downstream_load_addrs)
        )
        if _waterfall:
            # Waterfall sheds one priority tier at a time and the lock hold
            # below keeps L2 from re-serving, so it can afford to taper near
            # the bound: a low floor sheds in small steps (each tier-balance
            # drop stays under the no-regret tolerance) and lets the per-poll
            # re-arm walk the line down to *just* under 100 %, settling inside
            # the lock's release band rather than overshooting well below it.
            total_amount = min(1.0, max(0.05, _LINE_RELIEF_GAIN * overshoot))
        elif _downstream_line:
            # Branch-downstream line relief must drive the line to feasibility,
            # not nudge it: the gain-0.3 / 0.02-floor schedule sheds ~1 %/round
            # and can't clear a 10-20 % overload inside the run while the holon
            # re-serves the rest.  Use a high gain so each round sheds a large
            # share of the *downstream* loads (priority still orders WHO sheds),
            # converging in a few rounds; the auction re-arms until ≤100 %.
            total_amount = min(1.0, max(0.25, _LINE_RELIEF_GAIN * overshoot))
        else:
            total_amount = max(0.02, min(1.0, self._CURTAILMENT_GAIN * overshoot))

        # The violating agent's OWN load is the most direct lever on its own
        # junction (an upstream neighbour relieves a downstream node, but a
        # node's own extraction always drives its own return temperature),
        # so seed it as a candidate.  Priority still decides who actually
        # absorbs: a high-priority self competing against low-priority
        # neighbours wins ~0 share until the neighbours' reducible is spent.
        self_obs = self.behavior.observe(self.context.aid) or {}
        self_w_raw = (
            self._own_curtail_willingness(self_obs)
            if self.behavior.has_action(self.context.aid, "regulate")
            else None
        )
        # Tier-1 returns willingness 0.0 (see ``_own_curtail_willingness``);
        # drop it from the auction so the all-zero fallback even-split
        # in ``_allocate_auction`` can't shed self either.  A zero
        # willingness contributes nothing to dispatch in any case
        # (``_dispatch`` short-circuits on share <= 0), so this is safe
        # for the non-tier-1 "nothing to curtail" path too.
        self_w = (
            self_w_raw if (self_w_raw is not None and self_w_raw > 0.0) else None
        )
        # Targeting: the auctioneer IS the violation origin, so it is the
        # closest possible bidder — scale its self-bid by the max proximity
        # so it competes on the same footing as the proximity-weighted
        # neighbour bids (priority still decides who actually absorbs).
        if self.enable_curtail_auction_targeting and self_w is not None:
            self_w *= _CURTAIL_PROX_MAX

        # Branch-downstream relief: for a line overload with a resolved
        # downstream set, the bidders are the loads that actually flow through
        # the branch (so the shed reduces ITS loading), not the whole
        # component.  Priority still orders the shed (lowest-priority
        # downstream load first).  Falls back to the component otherwise.
        if (
            self.enable_branch_downstream_relief
            and variable == "loading_percent"
            and self._downstream_load_addrs
        ):
            neighbors = list(self._downstream_load_addrs)
        else:
            neighbors = list(topology_neighbors(self, tid="groups"))

        if not neighbors and self_w is None:
            # Self is locked (tier-1 or nothing to curtail) and there are
            # no neighbours to delegate to — no auction can be allocated
            # without violating the hard-lock invariant.  Clear the
            # in-flight guard so the next monitor poll can retry once
            # repartition adds neighbours or self becomes shed-eligible.
            self._curtail_inflight.pop(variable, None)
            return

        auction_id = str(uuid.uuid4())
        self._open_auctions[auction_id] = {
            "bids": {},
            "total": total_amount,
            "neighbours_contacted": len(neighbors),
            "bidders": {},  # sender_key -> addr
            "bid_meta": {},  # sender_key -> (tier, reducible)
            "var": variable,
            "self_willingness": self_w,
            "self_addr": self.context.addr,
            "waterfall": _waterfall,
        }
        self._curtail_inflight[variable] = now + self._AUCTION_TIMEOUT_S

        if not neighbors:
            # Self-only auction (isolated node / singleton group): allocate
            # immediately so the local lever still fires.
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
        # Targeting: scale by this bidder's electrical proximity to the
        # violation origin so the share concentrates on the loads that
        # actually relieve THIS violation (high ∂constraint/∂Q), not just
        # any reducible load in the component.  Bounded within-tier, so
        # priority stays dominant.
        if self.enable_curtail_auction_targeting:
            willingness *= self._curtail_proximity(
                message.origin_addr, message.variable
            )
        # Carry tier + reducible draw so a line-relief-waterfall auctioneer can
        # shed in strict reverse-priority order (cheap; ignored by the default
        # willingness-proportional allocator).
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
        _CURTAIL_PROX_MAX]`` for this bidder relative to the violation
        ``origin_addr``/``variable``.

        Uses the cached multi-hop ``ConstraintStateMessage`` distance: the
        origin broadcasts at ``hops_remaining = max_hops`` and each forward
        decrements, so a larger cached ``hops_remaining`` means this bidder
        sat fewer hops from the origin (electrically closer → larger
        ∂constraint/∂Q).  No cached state ⇒ the bidder never saw the
        violation propagate (beyond the propagation radius) ⇒ neutral 1.0,
        so targeting only ever *redistributes* toward demonstrably-close
        bidders and never starves an unknown one below baseline.
        """
        if not variable or origin_addr is None or self.max_hops <= 0:
            return 1.0
        state = self._neighbour_state.get((str(origin_addr), variable))
        if state is None:
            return 1.0
        # hops_remaining in [0, max_hops]; map to [MIN, MAX] linearly.
        frac = max(0.0, min(1.0, float(state.hops_remaining) / float(self.max_hops)))
        return _CURTAIL_PROX_MIN + (_CURTAIL_PROX_MAX - _CURTAIL_PROX_MIN) * frac

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
        # (tier, reducible) for the reverse-priority waterfall allocator.
        auction["bid_meta"][sender_key] = (
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
        # Clear the in-flight guard so the next monitor poll can open the
        # next round if the violation has not yet cleared.
        self._curtail_inflight.pop(auction.get("var"), None)

        bids: dict[str, float] = dict(auction["bids"])
        bidders: dict[str, Any] = dict(auction["bidders"])
        total_amount: float = auction["total"]

        # Fold in the auctioneer's own bid (the L0 self-curtail candidate).
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

        # Strict reverse-priority waterfall for branch-downstream line relief:
        # drive the lowest-priority tier with reducible draw toward zero, then
        # escalate to the next tier on the following rounds (an exhausted tier
        # reports reducible ≈ 0 and drops out).  Tier 1 is never shed (the
        # hard-lock); when the only reducible bidders left are tier 1, stop and
        # surface the residual.  This keeps the shed lowest-priority-first
        # (priority invariant stays clean) while still relieving the line.
        if auction.get("waterfall"):
            meta: dict[str, tuple] = dict(auction.get("bid_meta", {}))
            var = auction.get("var", "loading_percent")
            eligible = {
                k: tier
                for k, (tier, red) in meta.items()
                if tier >= 2 and red > _LINE_RELIEF_MIN_REDUCIBLE
            }
            if not eligible:
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
            target_tier = max(eligible.values())  # lowest priority present
            for key, tier in eligible.items():
                if tier == target_tier:
                    await _dispatch(key, bidders.get(key), total_amount)
            return

        sum_w = sum(bids.values())
        if sum_w <= 0.0:
            # All willingness scores zero — fall back to an even split
            # so at least something curtails and the violation can clear.
            share = total_amount / len(bids)
            for key, addr in bidders.items():
                await _dispatch(key, addr, share)
            return

        for key, w in bids.items():
            await _dispatch(key, bidders.get(key), total_amount * (w / sum_w))

    async def _handle_curtailment_request(
        self, message: CurtailmentRequest, meta: dict
    ) -> None:
        await self._apply_curtail(message.amount, label="curtailed")

    async def _curtail_self(self, amount: float) -> None:
        """Apply the auctioneer's own winning share — the L0 self-action
        lever (the violating load curtailing its own setpoint)."""
        await self._apply_curtail(amount, label="self-curtailed")

    async def _apply_curtail(self, amount: float, *, label: str) -> None:
        if not self.behavior.has_action(self.context.aid, "regulate"):
            return
        obs = self.behavior.observe(self.context.aid)
        if not obs:
            return

        # Multiplicative reduction: amount=0.3 means "cut current output
        # by 30%". Repeated requests compound toward zero rather than
        # jumping past it, so the control loop can't overshoot in a
        # single step.
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

    # Target junction temperature: hold t_k a small margin above the hard
    # floor so the load is served at (just inside) the feasibility frontier.
    _HEAT_FRONTIER_MARGIN_K: float = 3.0
    # Below ``target - DEADBAND`` -> shed; above ``target + RESTORE_BAND`` ->
    # restore.  The asymmetric, wide restore band is hysteresis: it stops the
    # restore<->re-violate limit cycle for nodes that re-cool when served.
    _HEAT_FRONTIER_DEADBAND_K: float = 2.0
    _HEAT_FRONTIER_RESTORE_BAND_K: float = 6.0
    # Proportional gain and per-poll step clamp.  The clamp bounds the move
    # even when ``_sensitivity`` is still at its prior (so a mis-estimate
    # can't slam the load to 0/1 in one step); as the dT/dP estimate is
    # learned the proportional term settles the load at the frontier.
    _HEAT_FRONTIER_GAIN: float = 0.5
    _HEAT_FRONTIER_MAX_STEP: float = 0.15
    # Feedback-loop period (s).  Faster than the heat SCADA poll so the
    # rate-limited controller can take enough steps to converge a deeply
    # cold node to its frontier within the run.
    _HEAT_FRONTIER_PERIOD_S: float = 1.0
    # Priority-waterfall gate.  A heat load broadcasts its (tier, reducible)
    # on its t_k state message; a cold load defers its own shed while a
    # peer in its hydraulic region with strictly lower priority (higher
    # tier number) still has reducible heat draw above this epsilon (MW).
    # Freshness window ages out peers that have since shed and stopped
    # re-broadcasting (re-broadcast fires on a t_k move or every
    # ``_FORWARD_FRESHNESS_S``), so a stale "still reducible" can't pin a
    # permanent defer.
    _WATERFALL_REDUCIBLE_EPS: float = 1e-4
    _HEAT_PEER_FRESHNESS_S: float = 2.0 * _FORWARD_FRESHNESS_S

    def _region_has_lower_priority_reducible(self, my_tier: int) -> float:
        """Total reducible heat draw of fresh same-region peers at a
        strictly lower priority (higher tier number) than ``my_tier``.

        Drives the frontier controller's deferral gate: while this is
        non-trivial, a cold load lets the lower-priority loads (and the
        priority-weighted curtailment auction) absorb the shed first
        instead of shedding itself tier-blind.
        """
        now = self.context.current_timestamp
        total = 0.0
        for _origin, (t_rx, tier, reducible) in self._heat_peer_state.items():
            if now - t_rx > self._HEAT_PEER_FRESHNESS_S:
                continue
            if tier > my_tier and reducible > self._WATERFALL_REDUCIBLE_EPS:
                total += reducible
        return total

    async def _heat_frontier_control(self) -> None:
        """Drive this heat load's regulation toward the point where its
        junction temperature sits at the feasibility floor — the maximum
        feasible service.  Sheds a cold node to its partial frontier (not
        bang-bang to 0) and restores a comfortably-warm one, using the local
        dT/dreg sensitivity as the gain.  Applies to ALL tiers, incl. tier-1
        (holding a critical heat load at full draw collapses its temperature
        and the served barrier then credits zero — a partial feasible serve
        beats that).  Writes ``reason="curtail"`` (shed) / ``"heat_recovery"``
        (restore) so the heat curtail-lock makes the MW holon defer.
        """
        if self.sector != Sector.HEAT:
            return
        if not self.behavior.has_action(self.context.aid, "regulate"):
            return
        obs = self._safe_observe()
        if not obs:
            return
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap <= 0:  # generator-class — nothing to curtail
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
            return  # solver hasn't populated this junction yet

        cur = float(obs.get("regulation", 1.0))
        target = lo + self._HEAT_FRONTIER_MARGIN_K
        too_cold = t < target - self._HEAT_FRONTIER_DEADBAND_K
        # Only restore loads WE (the auction/frontier controller) shed for
        # temperature — i.e. that still hold a curtail-lock.  A load shed by
        # an L2 *priority* decision sets no lock; restoring it on a warm
        # reading alone would claw back the priority cascade (the inversion
        # bug that retired the old blind heat-recovery loop).
        can_restore = (
            t > target + self._HEAT_FRONTIER_RESTORE_BAND_K
            and cur < 1.0
            and has_heat_curtail_lock(self.behavior, self.context.aid)
        )
        if not (too_cold or can_restore):
            return  # inside the hold band

        # Priority-waterfall gate (shed direction only): if a strictly
        # lower-priority heat load in our hydraulic region still has
        # reducible draw, defer — let it (and the priority-weighted
        # curtailment auction) absorb the shed first rather than shedding
        # this higher-priority load tier-blind.  The gate opens once the
        # lower tiers have shed (their reducible decays toward 0), so a
        # genuinely needed shed of this load still happens, just last in
        # priority order.  Restores are never gated here.
        if too_cold and self.enable_heat_priority_waterfall:
            my_tier = max(
                1, obs_priority(obs, behavior=self.behavior, aid=self.context.aid)
            )
            if self._region_has_lower_priority_reducible(my_tier) > 0.0:
                logger.debug(
                    "[%s] heat frontier: defer shed (t_k=%.1f, tier=%s) — "
                    "lower-priority reducible load remains in region",
                    self.context.aid, t, my_tier,
                )
                return

        # d(t_k)/d(reg) for heat is NEGATIVE (more extraction -> colder); the
        # EMA stores the magnitude |dt_k/dP|, dP/dreg = cap.  Floor the
        # magnitude away from 0 so the step is finite; the clamp below bounds
        # it regardless of the estimate's quality.
        dtk_dreg_mag = max(self._sensitivity * cap, 1e-6)
        delta_t = target - t  # >0 want warmer (shed); <0 want cooler (restore)
        delta_reg = -self._HEAT_FRONTIER_GAIN * delta_t / dtk_dreg_mag
        delta_reg = max(
            -self._HEAT_FRONTIER_MAX_STEP,
            min(self._HEAT_FRONTIER_MAX_STEP, delta_reg),
        )
        # Anti-limit-cycle damping: if this step reverses the previous one
        # (the load overshot its frontier last time), halve it so the
        # amplitude decays toward the frontier instead of ping-ponging.
        if self._frontier_last_dir != 0.0 and (
            delta_reg * self._frontier_last_dir < 0.0
        ):
            delta_reg *= 0.5
        new_reg = max(0.0, min(1.0, cur + delta_reg))
        if abs(new_reg - cur) < 1e-3:
            return
        self._frontier_last_dir = 1.0 if delta_reg > 0.0 else -1.0

        reason = "curtail" if new_reg < cur else "heat_recovery"
        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            new_reg,
            sector=self.sector.value,
            reason=reason,
            timestamp=self.context.current_timestamp,
            priority_tier=lookup_priority(self.behavior, self.context.aid),
        )
        if applied:
            logger.info(
                "[%s] heat frontier: t_k=%.1f target=%.1f regulation %.3f -> %.3f",
                self.context.aid, t, target, cur, new_reg,
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def worst_neighbour_utilization(self) -> float:
        """Return the worst constraint utilization reported by any
        neighbour within multi-hop range, weighted by the continuous
        coupling weight K_ij of the link the report arrived on (B.1).

        A low-trust link contributes a proportionally weaker signal, so
        the negotiator's pessimism factor degrades smoothly as
        connectivity becomes unreliable.  The weighted utilisation is
        ``K_ij * util_neighbour``.
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
        variable.  Values are strictly positive (the estimator takes
        absolute values of each observed pair) and default to a
        sector-typical prior until enough samples are collected."""
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
        # Signed injection.  For generators (cap < 0) sp is negative.
        p = sp if cap != 0.0 else 0.0
        if self._last_p is not None and self._last_v is not None:
            dp = p - self._last_p
            dv = v - self._last_v
            min_dp = _SENSITIVITY_MIN_DP.get(self.sector, 1e-6)
            if abs(dp) >= min_dp and math.isfinite(dv):
                sample = abs(dv / dp)
                # EMA update.  Clamp absurd values (post-failure
                # snapshots can show huge jumps that dominate the
                # estimate otherwise).
                sample = min(sample, 10.0 * self._sensitivity + 1.0)
                self._sensitivity = (
                    (1.0 - _SENSITIVITY_EMA_ALPHA) * self._sensitivity
                    + _SENSITIVITY_EMA_ALPHA * sample
                )
        self._last_p = p
        self._last_v = v
