"""Grid constraint monitoring and enforcement with multi-hop state propagation."""

from __future__ import annotations

import logging
import math
import uuid
from typing import TYPE_CHECKING, Any

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors
from monee.model.child import ExtPowerGrid

from scare.base.model import (
    DEENERGISED_PRESSURE_HIGH_PU,
    DEENERGISED_PRESSURE_PU,
    DEENERGISED_VM_PU,
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
from scare.base.runtime.diagnostics import record_event
from scare.base.util import (
    LINE_CONGESTION_REASON,
    apply_regulate,
    constraint_utilization,
    feeder_max_voltage,
    has_heat_curtail_lock,
    lookup_priority,
    obs_capacity,
    obs_constraint_values,
    obs_priority,
    obs_setpoint,
    publish_line_congestion_price,
    publish_line_relief_headroom,
    publish_node_voltage,
    qv_relief_avail,
    refresh_line_curtail_lock,
    sector_from_grid,
)
from scare.service.balance.trust import TrustLedger, TrustParams
from scare.service.control.curtailment import (
    curtail_willingness,
    plan_auction_allocation,
    proximity_from_hops,
)
from scare.service.control.heat_frontier import HeatFrontierController

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Hops constraint state propagates.
_DEFAULT_MAX_HOPS = 3

# Min utilization change that triggers a fresh broadcast.
_FORWARD_VALUE_TOL: float = 0.02

# Min sim-time between re-broadcasts of an unchanged value (keeps trust
# liveness ticking without per-cycle flooding).
_FORWARD_FRESHNESS_S: float = 5.0

# Cache-gate tolerance for ``_monitor``; tighter than ``_FORWARD_VALUE_TOL``.
_VALUES_DELTA_TOL: float = 1e-4

# EMA smoothing for the local sensitivity estimate (dV/dP).
_SENSITIVITY_EMA_ALPHA: float = 0.2

# Per-sector min |ΔP| before a sample is used; below this ΔV is noise.
_SENSITIVITY_MIN_DP: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,  # MW
    Sector.GAS: 1e-4,  # kg/s
    Sector.HEAT: 5e-4,  # MW (0.5 kW; registers ~30% regulation steps)
}

# Default sensitivity before any samples collected.
_SENSITIVITY_DEFAULT: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,  # p.u. voltage per MW
    Sector.GAS: 0.5,  # p.u. pressure per kg/s
    Sector.HEAT: 10.0,  # K per MW (samples are dT/dP_MW; ≡ 1e-5 K/W)
}

# Bounds on the auction willingness sensitivity multiplier; a within-tier
# tiebreaker kept far below the tier step so priority stays lexicographic.
_SENS_MULT_MIN: float = 0.25
_SENS_MULT_MAX: float = 4.0

# Primary constraint variable per sector for sensitivity tracking.
_SECTOR_PRIMARY_VAR: dict[Sector, str] = {
    Sector.ELECTRICITY: "vm_pu",
    Sector.GAS: "pressure_pu",
    Sector.HEAT: "t_k",
}

# Sentinel bidder key for the auctioneer's OWN load (its setpoint is the
# most direct lever on its junction); distinct from any ``str(addr)`` key.
_SELF_BID_KEY: str = "__self__"

# Auction gating (``enable_curtail_auction_gating``): no-progress rounds
# before the gate suspends re-arming, and the overshoot improvement that
# counts as progress.
_CURTAIL_NO_PROGRESS_LIMIT: int = 2
_CURTAIL_PROGRESS_TOL: float = 0.01

# Coordinated hand-off (``enable_qv_auction_coordination``): defer to the
# reactive lever only while voltage is measurably dropping. ``_..._TOL`` is the
# min p.u. drop per poll counting as progress; ``_..._DEFERS`` is a backstop
# bounding pathological oscillation, not the primary stop.
_QV_MAX_CONSECUTIVE_DEFERS: int = 6
_QV_DEFER_PROGRESS_TOL: float = 1e-3

# The auction never fires on ``loading_percent``: its node-blind bidding can't
# relieve a branch (the line-relief path owns it; downstream relief re-enables
# it with a targeted bidder set). ``t_k`` is skipped only while the heat
# frontier controller is enabled to own it — see ``_auction_skips_var``. The
# auction still fires on ``vm_pu`` / ``pressure_pu`` where local load is the lever.

# Targeting (``enable_curtail_auction_targeting``): proximity to the violated
# origin scales willingness within these bounds (within-tier tiebreaker, from
# cached multi-hop distance). Auctioneer is the origin, self-bids at PROX_MAX.
_CURTAIL_PROX_MIN: float = 0.25
_CURTAIL_PROX_MAX: float = 4.0

# Min sim-seconds between line-relief re-assertions per branch
# (``enable_line_relief_reassert``); never out-paces the gossip round it triggers.
_LINE_RELIEF_COOLDOWN_S: float = 2.0

# Consecutive polls classifying an overload as export (reverse flow) before
# load-shed relief is suppressed and generators are curtailed; a single
# transient reverse-flow sample must not trigger a non-reverting curtail.
_EXPORT_DEBOUNCE_POLLS: int = 2

# Sim-seconds a resolved downstream topology stays valid. No topology event
# (branch failure, tie close) reaches branch monitors, so re-resolve on a TTL
# when consulted.
_DOWNSTREAM_TOPOLOGY_TTL_S: float = 10.0

# Aggressive per-round gain for branch-downstream line relief (vs 0.3 default),
# walking a 10-20% overload down to ≤100% over rounds; priority orders WHO.
_LINE_RELIEF_GAIN: float = 1.5

# Reducible-draw threshold (MW) below which a downstream bidder is exhausted,
# escalating the waterfall to the next tier.
_LINE_RELIEF_MIN_REDUCIBLE: float = 5e-4

# Schmitt-trigger release margin (loading-% points): hold the L2-clawback lock
# until the line drops this far below bound, avoiding a relief↔L2 limit-cycle.
# Released only on genuine headroom, not the relief's own settle point.
_LINE_RELIEF_RELEASE_MARGIN: float = 15.0

# Congestion-price controller (``enable_line_congestion_price``). Integral gain
# on the normalized loading overshoot ((val-hi)/100) per poll: moderate so the
# price climbs to the export-clearing level over a few polls instead of slamming
# PV to 0 in one shot. Restore step decays the price each headroom poll so PV
# ramps back to serve local load. Headroom margin (loading-% points below the
# limit) gates the decay — inside the band the last ceiling is held (a stalled
# monitor must not release curtailment and re-overload). Price is capped below 1
# so a downstream generator is never pinned fully to 0 by a single branch.
_LINE_CONGESTION_GAIN: float = 0.35
_LINE_CONGESTION_RESTORE_STEP: float = 0.05
_LINE_CONGESTION_HEADROOM_MARGIN: float = 8.0
_LINE_CONGESTION_PRICE_MAX: float = 0.95
# Freshness (sim-s) of a published congestion price; matches the line-curtail
# lock TTL so a monitor that stops publishing releases the ceiling on the same
# horizon the old hard lock aged out.
_LINE_CONGESTION_TTL_S: float = 3.0

# Heat frontier feedback period (s), faster than the heat SCADA poll so a
# rate-limited deeply-cold node converges within the run. See HeatFrontierController.
_HEAT_FRONTIER_PERIOD_S: float = 1.0


class GridConstraintMonitor(Role):
    """Checks local grid measurements against sector bounds and takes action:
    warnings, violations + ``BalanceProblem`` on breach, and multi-hop
    ``ConstraintStateMessage`` propagation deduped per (origin, variable).
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        node_id: Any = None,
        *,
        max_hops: int = _DEFAULT_MAX_HOPS,
        enable_curtailment_auction: bool = True,
        enable_generation_priority_curtailment: bool = False,
        enable_line_congestion_price: bool = True,
        enable_curtail_auction_gating: bool = False,
        enable_curtail_auction_targeting: bool = False,
        enable_line_relief_reassert: bool = False,
        enable_branch_downstream_relief: bool = False,
        enable_line_relief_waterfall: bool = False,
        downstream_load_addrs: list[Any] | None = None,
        enable_multihop_constraint: bool = True,
        enable_heat_frontier: bool = True,
        enable_heat_priority_waterfall: bool = True,
        enable_qv_auction_coordination: bool = False,
        enable_qv_feeder_gate: bool = True,
        branch_id: Any = None,
        home_leader_addr: Any = None,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.node_id = node_id
        self.max_hops = max_hops
        self.enable_curtailment_auction = enable_curtailment_auction
        # Over-voltage relief curtails generation only (never sheds load).
        self.enable_generation_priority_curtailment = (
            enable_generation_priority_curtailment
        )
        # Soft congestion-price line relief (reversible gen ceiling, no lock).
        self.enable_line_congestion_price = enable_line_congestion_price
        # Per-branch congestion price (1 - gen ceiling); AIMD integrator state.
        self._line_congestion_price: float = 0.0
        self.enable_curtail_auction_gating = enable_curtail_auction_gating
        self.enable_curtail_auction_targeting = enable_curtail_auction_targeting
        self.enable_line_relief_reassert = enable_line_relief_reassert
        self.enable_branch_downstream_relief = enable_branch_downstream_relief
        # Strict reverse-priority cascade for downstream line relief.
        self.enable_line_relief_waterfall = enable_line_relief_waterfall
        # Loads downstream of this branch (the only ones whose curtailment
        # reduces its flow); populated post-build for electricity branch monitors.
        self._downstream_load_addrs: list[Any] = list(downstream_load_addrs or [])
        self.enable_multihop_constraint = enable_multihop_constraint
        self.enable_heat_frontier = enable_heat_frontier
        # Heat priority-waterfall gate: a cold load defers its own shed while
        # lower-priority reducible heat load remains in its hydraulic region.
        self.enable_heat_priority_waterfall = enable_heat_priority_waterfall
        # Coordinated Q(U)/auction hand-off: credit remaining reactive relief
        # before sizing an over-voltage shed, so active is shed only for the residual.
        self.enable_qv_auction_coordination = enable_qv_auction_coordination
        # Phase-2: shed active (don't defer to reactive) when ANY feeder node
        # is over-voltage, per the gossip neighbour cache.
        self.enable_qv_feeder_gate = enable_qv_feeder_gate
        # Branch mode (``branch_id`` set): ``emit_event(BalanceProblem)`` is a
        # no-op (no co-located negotiator), so on overload send
        # ``StartBalanceNegotiation`` with a relief-MW override to ``home_leader_addr``.
        self.branch_id = branch_id
        self.home_leader_addr = home_leader_addr

        # (origin_addr_str, variable) -> ConstraintStateMessage
        self._neighbour_state: dict[tuple[str, str], ConstraintStateMessage] = {}

        # Heat frontier controller: owns the priority-waterfall peer cache and
        # frontier step state, decides the move toward the t_k feasibility floor.
        self._heat_frontier = HeatFrontierController(
            peer_freshness_s=2.0 * _FORWARD_FRESHNESS_S
        )

        # Dedup of forwarded state: (origin, variable) -> (best_hops, t, value).
        # Forward incoming only on better hops; re-broadcast own only on value
        # change or freshness elapse. Never cleared per-cycle (re-floods the group).
        self._state_forwarded: dict[tuple[str, str], tuple[int, float, float]] = {}
        # Per-variable (t, util) of this agent's last own broadcast.
        self._last_local_broadcast: dict[str, tuple[float, float]] = {}

        # Variables with a violation emitted this episode (dedup guard).
        self._violation_emitted: set[str] = set()

        # Last-observed values; ``_monitor`` short-circuits when nothing moved.
        self._last_polled_values: dict[str, float] = {}

        # B.1: coupling weights K_ij for the propagation overlay (independent of
        # the balance ledger). Weight worst-neighbour util by trust, skip
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

        # Local power-flow sensitivity: EMA of |dV/dP| from own (P, V) history,
        # letting the auction bid agents near the violation more aggressively.
        self._sensitivity: float = _SENSITIVITY_DEFAULT.get(sector, 1e-3)
        self._last_p: float | None = None
        self._last_v: float | None = None

        # Auctioneer-side state: auction_id -> {"bids", "total", ...}.
        self._open_auctions: dict[str, dict[str, Any]] = {}
        # Per-variable in-flight guard (variable -> deadline): prevents stacking
        # auctions while letting curtailment iterate round-by-round.
        self._curtail_inflight: dict[str, float] = {}
        # Progress gate (``enable_curtail_auction_gating``): variable ->
        # {"best", "no_progress"}; suspends re-arming an ineffective lever.
        self._curtail_progress: dict[str, dict[str, float]] = {}
        # Coordinated hand-off: consecutive reactive-defers (backstop cap) and
        # the value at last defer (to check droop progress before re-deferring).
        self._qv_defer_count: dict[str, int] = {}
        self._qv_last_value: dict[str, float] = {}
        # Per-variable cooldown for iterative line-relief re-assert.
        self._relief_inflight: dict[str, float] = {}
        # Branch flow-direction context, resolved lazily from the live net and
        # refreshed on a TTL: which endpoint is upstream (slack side) and which
        # generators sit downstream — the export-overload relief targets.
        self._downstream_resolved: bool = False
        self._downstream_resolved_t: float = float("-inf")
        self._upstream_is_from: bool | None = None
        self._downstream_gen_aids: list[str] = []
        # Per-variable consecutive export-classified polls (debounce).
        self._export_streak: dict[str, int] = {}
        # Per-variable flag: waterfall has only tier-1 reducible bidders left
        # (can't relieve further without breaking the hard-lock).
        self._line_relief_tier1_residual: dict[str, bool] = {}
        # Heat waterfall actuator: per-peer cooldown deadline for the curtail
        # requests a deferring cold load sends to lower-priority peers.
        self._waterfall_request_cooldown: dict[str, float] = {}

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 1.0)
        self.context.schedule_periodic_task(self._monitor, delay=poll)
        # Heat frontier feedback loop driving each load to the t_k feasibility
        # floor; faster than the SCADA poll so a deeply-cold node converges.
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
            lambda msg, meta: (
                isinstance(msg, ConstraintStateMessage) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_curtailment_request),
            lambda msg, meta: (
                isinstance(msg, CurtailmentRequest) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_curtailment_need),
            lambda msg, meta: (
                isinstance(msg, CurtailmentNeed) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_curtailment_bid),
            lambda msg, meta: (
                isinstance(msg, CurtailmentBid) and msg.sector == self.sector
            ),
        )
        # Branch agents have no negotiator; reply zeros to the leader's
        # pre-gossip ``AskEnergyMessage`` (a branch is a sensor, no flex).
        if self.branch_id is not None:
            self.context.subscribe_message(
                self,
                _wrap(self._handle_ask_energy_branch),
                lambda msg, meta: (
                    isinstance(msg, AskEnergyMessage) and msg.sector == self.sector
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
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    # ------------------------------------------------------------------
    # Periodic monitoring
    # ------------------------------------------------------------------

    def _safe_observe(self) -> dict | None:
        """``observe()`` result, or ``None`` when the LP hasn't solved yet."""
        try:
            return self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return None

    def _try_emit_event(self, event) -> None:
        """Emit a local event, swallowing the ``KeyError`` when no role subscribes."""
        try:
            self.context.emit_event(event)
        except KeyError:
            pass

    def _auction_skips_var(self, var: str) -> bool:
        """Variables the node-blind auction never fires on: ``loading_percent``
        always (the line-relief path owns branches); ``t_k`` only while the
        heat frontier controller is enabled to own it — with the frontier
        ablated the auction is the only heat lever left."""
        if var == "loading_percent":
            return True
        return var == "t_k" and self.enable_heat_frontier

    async def _handle_violation(
        self, obs: dict, var: str, val: float, lo: float, hi: float
    ) -> None:
        """Emit ``ConstraintViolation`` + ``BalanceProblem`` for a breached
        variable; relief-route branch overloads; (re-)arm curtailment.

        Event emission is deduped per episode; curtailment is re-armed every
        active poll, the in-flight guard preventing overlapping auctions.
        """
        # Branch-downstream relief owns a line overload only for a branch
        # ``loading_percent`` breach with a resolved downstream load set.
        downstream_active = (
            self.enable_branch_downstream_relief
            and self.branch_id is not None
            and var == "loading_percent"
            and bool(self._downstream_load_addrs)
        )
        # Export-driven (reverse-flow) overload: shedding downstream load
        # INCREASES net export and worsens it, so every load-shed relief path
        # is suppressed and downstream generation is curtailed instead.
        # Debounced over consecutive polls, and owned only while curtailable
        # downstream generators exist — otherwise the ordinary relief chain
        # stays in charge rather than suppressing everything.
        is_export = (
            self.branch_id is not None
            and var == "loading_percent"
            and val > hi
            and self._flow_is_export(obs) is True
        )
        if is_export:
            self._export_streak[var] = self._export_streak.get(var, 0) + 1
        else:
            self._export_streak.pop(var, None)
        export_overload = (
            is_export
            and self._export_streak[var] >= _EXPORT_DEBOUNCE_POLLS
            and bool(self._downstream_generator_aids())
        )
        # Load-shed suppression for an export overload. Legacy: only once the
        # debounced ``export_overload`` is confirmed — but then the waterfall
        # sheds downstream load on the first poll or two before generation
        # curtail (the correct lever) engages, and that shed doesn't revert once
        # the line clears. Under generation-priority curtailment, stop shedding
        # the MOMENT an overload looks export-driven with curtailable downstream
        # generation; the gen-curtail actuation below stays debounced so a single
        # transient reverse sample can't latch a non-reverting curtail.
        if self.enable_generation_priority_curtailment or self.enable_line_congestion_price:
            suppress_load_shed = is_export and bool(self._downstream_generator_aids())
        else:
            suppress_load_shed = export_overload

        if var not in self._violation_emitted:
            self._violation_emitted.add(var)
            logger.warning(
                "[%s] CONSTRAINT VIOLATION %s=%.4f bounds=[%.4f,%.4f]",
                self.context.aid,
                var,
                val,
                lo,
                hi,
            )
            record_event(
                t=self.context.current_timestamp,
                kind="constraint_violation",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=f"{var}={val:.4f} bounds=[{lo:.4f},{hi:.4f}]",
            )
            self._try_emit_event(
                ConstraintViolation(
                    sector=self.sector,
                    variable=var,
                    value=val,
                    bound_low=lo,
                    bound_high=hi,
                    node_id=self.node_id,
                )
            )
            self._try_emit_event(
                BalanceProblem(
                    sector=self.sector,
                    imbalance=val - hi if val > hi else lo - val,
                )
            )
            # Branch-mode legacy endpoint relief (one-shot); deferred when
            # downstream relief or iterative re-assert handles it instead.
            if (
                self.branch_id is not None
                and var == "loading_percent"
                and self.home_leader_addr is not None
                and not downstream_active
                and not suppress_load_shed
                and not self.enable_line_relief_reassert
            ):
                await self._send_line_overload_relief(obs, val, lo, hi)
        # Iterative endpoint relief (re-asserts while overloaded); skipped when
        # branch-downstream relief owns the line.
        if (
            self.enable_line_relief_reassert
            and not downstream_active
            and not suppress_load_shed
            and self.branch_id is not None
            and var == "loading_percent"
            and self.home_leader_addr is not None
        ):
            await self._reassert_line_relief(obs, var, val, lo, hi)
        # Curtailment auction; skip-vars are scoped out (the node-blind
        # auction can't relieve them) unless downstream relief re-enables
        # ``loading_percent`` with a targeted bidder set.
        if (
            self.enable_curtailment_auction
            and not suppress_load_shed
            and (downstream_active or not self._auction_skips_var(var))
        ):
            await self._request_curtailment(var, val, lo, hi)
        # Export gen relief: the soft congestion-price controller (driven every
        # poll from ``_monitor``) owns it when enabled; else the legacy hard
        # curtail-to-0 export relief.
        if export_overload and not self.enable_line_congestion_price:
            await self._relieve_export_overload(obs, var, val, hi)

    def _handle_warning(
        self, var: str, val: float, lo: float, hi: float, util: float
    ) -> None:
        """Emit ``ConstraintWarning`` for a variable above the warning threshold."""
        self._try_emit_event(
            ConstraintWarning(
                sector=self.sector,
                variable=var,
                value=val,
                bound_low=lo,
                bound_high=hi,
                utilization=util,
                node_id=self.node_id,
            )
        )
        logger.debug(
            "[%s] constraint warning %s=%.4f util=%.2f",
            self.context.aid,
            var,
            val,
            util,
        )

    async def _monitor(self) -> None:
        obs = self._safe_observe()
        if not obs:
            return

        bounds = SECTOR_CONSTRAINTS.get(self.sector, {})
        values = obs_constraint_values(obs, self.sector)

        # Cache gate: skip the pass when no value moved and no violation is
        # active. An active violation must keep firing until it clears, else
        # downstream balance roles never see the "clear" transition.
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

        # Phase-2: publish this node's voltage to the shared feeder ledger so
        # other inverters' auctions can see feeder-wide over-voltage.
        if (
            self.enable_qv_auction_coordination
            and self.enable_qv_feeder_gate
            and self.sector is Sector.ELECTRICITY
        ):
            _vm = self._last_polled_values.get("vm_pu")
            if _vm is not None:
                publish_node_voltage(
                    self.behavior, self.context.aid, _vm, self.context.current_timestamp
                )

        self._update_sensitivity(obs)

        for var, val in values.items():
            # Skip readings the solver hasn't populated or that signal a
            # de-energised junction: isolated heat nodes report t_k=0 / NaN
            # post-failure, a gas region cut off from its source collapses to
            # pressure_pu~0 (or saturates the relaxed-Weymouth box at ~sqrt(3)),
            # and an electricity node cut off from its slack collapses to
            # vm_pu~0 (see DEENERGISED_*). None is an actionable breach (no
            # curtailment lever re-energises a source-isolated region); genuine
            # out-of-bound readings sit well inside the gap, so they still fire.
            if (
                not math.isfinite(val)
                or (var == "t_k" and val <= 0.0)
                or (
                    var == "pressure_pu"
                    and (
                        val <= DEENERGISED_PRESSURE_PU
                        or val >= DEENERGISED_PRESSURE_HIGH_PU
                    )
                )
                or (var == "vm_pu" and val <= DEENERGISED_VM_PU)
            ):
                continue

            lo, hi = bounds.get(var, (float("-inf"), float("inf")))
            util = constraint_utilization(val, lo, hi)

            # Every poll: feed the line-relief hand-off its loading headroom and
            # keep any live restore-ramp lock fresh, so the bounded hand-back can
            # proceed once the line clears without the lock ageing out (which
            # would let L2 slam the load to full and re-overload the line).
            self._maintain_line_relief_handoff(var, val, hi)
            if self.enable_line_congestion_price:
                self._maintain_congestion_price(obs, var, val, hi)

            if val < lo or val > hi:
                await self._handle_violation(obs, var, val, lo, hi)
                # Hold the L2-clawback lock fresh; else it ages out between
                # bursty sheds and L2 re-serves mid-relief.
                self._hold_downstream_line_locks(var, val, hi)
            elif (
                self._is_line_relief_branch()
                and var == "loading_percent"
                and val > hi - _LINE_RELIEF_RELEASE_MARGIN
            ):
                # Hysteresis hold band: line cleared but lacks the release
                # margin; hold the lock so L2 can't claw back and re-breach.
                self._hold_downstream_line_locks(var, val, hi)
            else:
                self._violation_emitted.discard(var)
                # Back in-bounds: clear gates so a re-breach gets a fresh budget.
                self._curtail_progress.pop(var, None)
                self._relief_inflight.pop(var, None)
                self._line_relief_tier1_residual.pop(var, None)
                self._export_streak.pop(var, None)

            if (
                util >= PROACTIVE_WARNING_FRACTION
                and var not in self._violation_emitted
            ):
                self._handle_warning(var, val, lo, hi, util)

            if self.enable_multihop_constraint:
                await self._propagate_state(var, val, util, obs=obs)

    # ------------------------------------------------------------------
    # Multi-hop state propagation with deduplication
    # ------------------------------------------------------------------

    async def _propagate_state(
        self,
        variable: str,
        value: float,
        utilization: float,
        obs: dict | None = None,
    ) -> None:
        # Suppress re-broadcasts of an unchanged value unless freshness elapsed
        # or utilization moved beyond ``_FORWARD_VALUE_TOL``.
        now = self.context.current_timestamp
        prev = self._last_local_broadcast.get(variable)
        if prev is not None:
            prev_t, prev_util = prev
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (stale or changed):
                return

        # Heat t_k broadcasts carry (tier, reducible) so cold neighbours can
        # run the priority-waterfall gate; only set for a curtailable heat load.
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

        # B.1: nudge the K-score of the arriving link.
        sender = mango_sender_addr(meta)
        now = self.context.current_timestamp
        if sender is not None:
            self._trust.on_message_received(str(sender), now)

        self._neighbour_state[origin_key] = message

        # Heat priority-waterfall: cache the origin's (tier, reducible).
        if message.priority_tier is not None and message.reducible is not None:
            self._heat_frontier.note_peer_state(
                str(message.origin_addr),
                now,
                message.priority_tier,
                message.reducible,
            )

        # Dedup: forward only if the incoming copy improves on the last
        # forwarded one — larger ``hops_remaining``, freshness elapsed, or
        # value moved beyond tolerance.
        prev = self._state_forwarded.get(origin_key)
        if prev is not None:
            prev_hops, prev_t, prev_util = prev
            improves_hops = message.hops_remaining > prev_hops
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(message.utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (improves_hops or stale or changed):
                return
        self._state_forwarded[origin_key] = (
            message.hops_remaining,
            now,
            message.utilization,
        )

        if message.hops_remaining <= 1:
            return  # TTL exhausted

        # ``enable_multihop_constraint=False`` also disables forwarding (needed
        # for ``component_level``, where one group fans out N·(N−1) per hop).
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
        """Re-send the relief target while overloaded so the home leader sheds
        round-by-round. Cooldown-guarded so it never out-paces its gossip round.
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
        """Send StartBalanceNegotiation with a relief-MW target: the MW the home
        group must shed, scaled by line flow (max ``p_from_mw`` / ``p_to_mw``).
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
            # No flow magnitude — fall back to a fractional signal.
            relief_mw = overshoot_fraction
        else:
            relief_mw = flow_mw * overshoot_fraction

        # Negative target => group reduces net load, via the Layer-1 QP's
        # reverse-priority curtailment schedule.
        try:
            await self.context.send_message(
                StartBalanceNegotiation(override_target=-relief_mw),
                receiver_addr=self.home_leader_addr,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[%s] line-overload relief send failed: %s",
                self.context.aid,
                exc,
            )

    def _is_line_relief_branch(self) -> bool:
        """True iff this monitor runs the branch-downstream line-relief lever."""
        return (
            self.enable_branch_downstream_relief
            and self.branch_id is not None
            and bool(self._downstream_load_addrs)
        )

    def _hold_downstream_line_locks(self, var: str, val: float, hi: float) -> None:
        """Keep the L2-clawback line locks fresh while the line is over (or in
        the hysteresis band), so L2 can't re-serve a just-relieved load. No-op
        unless this is the line-relief branch lever on ``loading_percent``."""
        if var != "loading_percent" or not self._is_line_relief_branch():
            return
        if val <= hi - _LINE_RELIEF_RELEASE_MARGIN:
            return
        now = self.context.current_timestamp
        for addr in self._downstream_load_addrs:
            aid = getattr(addr, "aid", None)
            if aid is not None:
                refresh_line_curtail_lock(self.behavior, aid, now)

    def _maintain_line_relief_handoff(self, var: str, val: float, hi: float) -> None:
        """Publish this branch's loading headroom (``hi - val``) to its
        downstream loads every poll, and keep any live restore-ramp lock fresh
        so ``apply_regulate``'s bounded hand-back can proceed in the cleared
        region. ``refresh_line_curtail_lock`` only re-stamps EXISTING locks, so
        a fully-restored (lock-dropped) load is left alone. No-op unless this is
        the line-relief branch lever on ``loading_percent``."""
        if var != "loading_percent" or not self._is_line_relief_branch():
            return
        now = self.context.current_timestamp
        headroom = hi - val
        for addr in self._downstream_load_addrs:
            aid = getattr(addr, "aid", None)
            if aid is None:
                continue
            publish_line_relief_headroom(self.behavior, aid, headroom, now)
            refresh_line_curtail_lock(self.behavior, aid, now)

    def _maintain_congestion_price(
        self, obs: dict, var: str, val: float, hi: float
    ) -> None:
        """Soft congestion-price controller for an export (reverse-flow) branch
        overload. Runs EVERY poll (not just on breach) so the price can decay and
        the generation ceiling recover once the line clears.

        AIMD-style: integrate the price up on overshoot while the flow is export
        with curtailable downstream gens; decay it on genuine loading headroom;
        hold it inside the hysteresis band (a stalled monitor must not release
        the ceiling and re-overload). The price is published per downstream gen
        (summed across branches by ``line_congestion_ceiling``) and the gens are
        curtailed DOWN to the ceiling immediately; the gossip ``_apply_setpoint``
        enforces the same ceiling softly, so PV can ramp back to serve local load
        up to the export-clearing level without a curtail-lock pinning it at 0.
        """
        if var != "loading_percent" or self.branch_id is None:
            return
        gens = self._downstream_generator_aids()
        if not gens:
            # No lever here; let the price decay so any stale ceiling lifts.
            self._line_congestion_price = max(
                0.0, self._line_congestion_price - _LINE_CONGESTION_RESTORE_STEP
            )
        else:
            overshoot = (val - hi) / 100.0 if val > hi else 0.0
            is_export = overshoot > 0.0 and self._flow_is_export(obs) is True
            if is_export:
                self._line_congestion_price = min(
                    _LINE_CONGESTION_PRICE_MAX,
                    self._line_congestion_price
                    + _LINE_CONGESTION_GAIN * overshoot,
                )
            elif val <= hi - _LINE_CONGESTION_HEADROOM_MARGIN:
                self._line_congestion_price = max(
                    0.0, self._line_congestion_price - _LINE_CONGESTION_RESTORE_STEP
                )
            # else: hysteresis band / non-export overload — hold last price.

        now = self.context.current_timestamp
        price = self._line_congestion_price
        ceiling = max(0.0, 1.0 - price)
        for aid in gens:
            publish_line_congestion_price(
                self.behavior, str(self.branch_id), aid, price, now
            )
            if price <= 0.0:
                continue
            gen_obs = self.behavior.observe(aid) or {}
            current = float(gen_obs.get("regulation", 1.0))
            if current > ceiling + 1e-6:
                apply_regulate(
                    self.behavior,
                    aid,
                    ceiling,
                    sector=self.sector.value,
                    reason=LINE_CONGESTION_REASON,
                    timestamp=now,
                    priority_tier=lookup_priority(self.behavior, aid),
                )
        if price > 0.0:
            record_event(
                t=now,
                kind="line_congestion_price",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=(
                    f"val={val:.1f} hi={hi:.1f} price={price:.3f} "
                    f"ceiling={ceiling:.3f} gens={len(gens)}"
                ),
            )

    # ------------------------------------------------------------------
    # Flow direction / export-overload relief
    # ------------------------------------------------------------------

    def _flow_is_export(self, obs: dict) -> bool | None:
        """True when the branch carries reverse (downstream→slack, export)
        flow, False for forward flow, None when undeterminable. Sign
        convention: ``p_from_mw > 0`` (equivalently ``p_to_mw < 0``) is
        from→to flow."""
        self._ensure_downstream_topology()
        if self._upstream_is_from is None:
            return None
        try:
            p_from = float(obs.get("p_from_mw", 0.0) or 0.0)
            p_to = float(obs.get("p_to_mw", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if abs(p_from) <= 1e-9 and abs(p_to) <= 1e-9:
            return None
        flow_from_to = p_from > 0.0 if abs(p_from) >= abs(p_to) else p_to < 0.0
        return flow_from_to is not self._upstream_is_from

    def _ensure_downstream_topology(self) -> None:
        """Resolve the downstream topology on first use and re-resolve after
        ``_DOWNSTREAM_TOPOLOGY_TTL_S``: failures and tie closes reshape the
        graph but no topology event reaches branch monitors, so a TTL is the
        cheapest correct invalidation."""
        now = self.context.current_timestamp
        if (
            self._downstream_resolved
            and (now - self._downstream_resolved_t) < _DOWNSTREAM_TOPOLOGY_TTL_S
        ):
            return
        self._downstream_resolved_t = now
        self._resolve_downstream_topology()

    def _resolve_downstream_topology(self) -> None:
        """Cut this branch and BFS the electricity graph from the slacks to
        find which endpoint is upstream and which generators sit downstream
        (the export-relief targets). Open ties and failed branches are
        non-conductive. Leaves ``_upstream_is_from`` None on a meshed /
        unclean cut."""
        self._downstream_resolved = True
        self._upstream_is_from = None
        self._downstream_gen_aids = []
        net = getattr(self.behavior, "_net", None)
        if net is None or self.branch_id is None:
            return
        try:
            branches = list(net.branches)
            childs = list(net.childs)
        except Exception:  # noqa: BLE001
            return

        adj: dict[Any, list[Any]] = {}
        for branch in branches:
            try:
                if branch.id == self.branch_id or branch.model.is_cp():
                    continue
                if (
                    not getattr(branch, "active", True)
                    or not getattr(branch.model, "active", True)
                    or not int(getattr(branch.model, "on_off", 1) or 0)
                ):
                    continue
                node = net.node_by_id(branch.id[0])
            except Exception:  # noqa: BLE001
                continue
            if sector_from_grid(getattr(node, "grid", None)) is not Sector.ELECTRICITY:
                continue
            a, b = branch.id[0], branch.id[1]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        slack_nodes = {
            child.node_id for child in childs if isinstance(child.model, ExtPowerGrid)
        }
        if not slack_nodes:
            return

        def _reach(start: set[Any]) -> set[Any]:
            seen = set(start)
            frontier = list(start)
            while frontier:
                nxt: list[Any] = []
                for n in frontier:
                    for nb in adj.get(n, ()):
                        if nb not in seen:
                            seen.add(nb)
                            nxt.append(nb)
                frontier = nxt
            return seen

        fed = _reach(slack_nodes)
        a, b = self.branch_id[0], self.branch_id[1]
        a_up, b_up = a in fed, b in fed
        if a_up == b_up:
            return  # no clean cut: direction stays undeterminable
        self._upstream_is_from = a_up

        down = _reach({b if a_up else a})
        gens: list[str] = []
        for child in childs:
            if child.node_id not in down or isinstance(child.model, ExtPowerGrid):
                continue
            try:
                cap = obs_capacity(dict(child.model.values))
            except Exception:  # noqa: BLE001
                continue
            if cap >= 0:
                continue
            aid = f"child-{child.id}"
            if self.behavior.has_action(aid, "regulate"):
                gens.append(aid)
        self._downstream_gen_aids = gens

    def _downstream_generator_aids(self) -> list[str]:
        self._ensure_downstream_topology()
        return self._downstream_gen_aids

    async def _relieve_export_overload(
        self, obs: dict, var: str, val: float, hi: float
    ) -> None:
        """Curtail downstream generation for an export (reverse-flow) overload.
        Load-shed paths are suppressed for these — shedding raises net export.
        Cooldown-guarded and re-armed each poll until the line clears."""
        now = self.context.current_timestamp
        deadline = self._relief_inflight.get(var)
        if deadline is not None and now < deadline:
            return
        gens = self._downstream_generator_aids()
        if not gens:
            # No lever here; leave the shared cooldown unburnt so the
            # ordinary relief chain isn't starved.
            record_event(
                t=now,
                kind="line_export_relief_no_generators",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=f"{var}={val:.1f} hi={hi:.1f}",
            )
            return
        self._relief_inflight[var] = now + _LINE_RELIEF_COOLDOWN_S
        amount = min(1.0, max(0.25, _LINE_RELIEF_GAIN * (val - hi) / 100.0))
        curtailed = 0
        for aid in gens:
            gen_obs = self.behavior.observe(aid) or {}
            current = float(gen_obs.get("regulation", 1.0))
            new_factor = max(0.0, current * (1.0 - amount))
            applied = apply_regulate(
                self.behavior,
                aid,
                new_factor,
                sector=self.sector.value,
                reason="curtail",
                timestamp=now,
                priority_tier=lookup_priority(self.behavior, aid),
            )
            if applied:
                curtailed += 1
        record_event(
            t=now,
            kind="line_export_relief",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=(
                f"{var}={val:.1f} hi={hi:.1f} amount={amount:.2f} "
                f"gens={len(gens)} curtailed={curtailed}"
            ),
        )

    # ------------------------------------------------------------------
    # Curtailment
    # ------------------------------------------------------------------

    # Proportional gain on normalized overshoot; gentle so persistence ratchets
    # it up over cycles rather than over-curtailing in one shot.
    _CURTAILMENT_GAIN: float = 0.3

    # How long the auctioneer waits for bids; short, the monitor re-fires next
    # cycle if the violation persists.
    _AUCTION_TIMEOUT_S: float = 2.0

    def _own_curtail_willingness(
        self, obs: dict, *, injection_relief: bool = False
    ) -> float:
        """Curtailment willingness for this agent's own load: priority tier
        weight (dominant, lexicographic) × bounded sensitivity multiplier ×
        reducible output.

        Tier-1 LOADS (cap > 0) return exactly 0.0, not the 1e-9 floor (which
        would let a tier-1 self-only auction shed itself, breaking the
        hard-lock); generators (cap < 0) keep the floor so PV stays shed-eligible.
        ``injection_relief`` restricts an over-voltage auction to generators.
        """
        from scare.service.balance.balance import _PRIORITY_TIERS

        prio_tier = max(
            1, obs_priority(obs, behavior=self.behavior, aid=self.context.aid)
        )
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        reducible = abs(obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid))
        return curtail_willingness(
            priority_tier=prio_tier,
            capacity=cap,
            reducible=reducible,
            sensitivity=self._sensitivity,
            sensitivity_ref=_SENSITIVITY_DEFAULT.get(self.sector, 1e-3),
            priority_tiers=_PRIORITY_TIERS,
            sens_mult_min=_SENS_MULT_MIN,
            sens_mult_max=_SENS_MULT_MAX,
            injection_relief=injection_relief,
        )

    async def _request_curtailment(
        self, variable: str, value: float, lo: float, hi: float
    ) -> None:
        span = hi - lo
        if span <= 0:
            return

        # Excess-injection violation: over-voltage is relieved ONLY by cutting
        # generation. Bid generators, exclude loads — shedding load on a
        # PV-surplus feeder raises voltage and needlessly drops served demand.
        # The Q(U)/auction coordination substitutes reactive for the ACTIVE
        # over-voltage shed it defers; that shed must target generation (its
        # premise), so it implies generation-priority even if the standalone
        # flag is off — otherwise the coordinated path sheds loads, which raises
        # voltage on an export feeder (wrong direction).
        injection_relief = (
            (
                self.enable_generation_priority_curtailment
                or self.enable_qv_auction_coordination
            )
            and self.sector is Sector.ELECTRICITY
            and variable == "vm_pu"
            and value > hi
        )

        # Gas OVER-pressure: shedding load shrinks the Weymouth drops and
        # RAISES pressure — positive feedback. The slack pressure regulator
        # owns that side; never arm a load-shed for it.
        if self.sector is Sector.GAS and variable == "pressure_pu" and value > hi:
            return

        # In-flight guard: skip while an auction for this variable is open so
        # rounds don't stack; re-opening once it clears reaches feasibility.
        now = self.context.current_timestamp
        deadline_prev = self._curtail_inflight.get(variable)
        if deadline_prev is not None and now < deadline_prev:
            return

        overshoot = (value - hi) / span if value > hi else (lo - value) / span

        # Coordinated hand-off: at a Q(U)-droop node, credit the reactive
        # lever's remaining relief (not yet in ``value``) and size the active
        # shed to the residual; skip the auction entirely when reactive covers
        # the overshoot, re-arming next poll if it falls short.
        if (
            self.enable_qv_auction_coordination
            and self.sector == Sector.ELECTRICITY
            and variable == _SECTOR_PRIMARY_VAR.get(self.sector)
            and value > hi
        ):
            relief = qv_relief_avail(self.behavior, self.context.aid, now)
            if relief <= 0.0:
                self._qv_defer_count.pop(variable, None)
                self._qv_last_value.pop(variable, None)
            else:
                residual = max(0.0, (value - relief - hi) / span)
                if residual > 0.0:
                    # Reactive covers part of it — shed active for the residual.
                    self._qv_defer_count.pop(variable, None)
                    self._qv_last_value.pop(variable, None)
                    overshoot = residual
                else:
                    # Reactive claims to cover it all. Defer only while voltage
                    # is measurably dropping; on stall (or backstop) escalate
                    # and shed active for the measured overshoot.
                    last = self._qv_last_value.get(variable)
                    improving = last is None or value < last - _QV_DEFER_PROGRESS_TOL
                    # Phase-2 feeder gate: never defer to local reactive while
                    # another node on the feeder is over-voltage — the retained
                    # active PV is what's holding the feeder over, so shed it.
                    feeder_over = (
                        self.enable_qv_feeder_gate and self._feeder_overvoltage(hi)
                    )
                    cnt = self._qv_defer_count.get(variable, 0)
                    if (
                        improving
                        and not feeder_over
                        and cnt < _QV_MAX_CONSECUTIVE_DEFERS
                    ):
                        self._qv_defer_count[variable] = cnt + 1
                        self._qv_last_value[variable] = value
                        self._curtail_inflight.pop(variable, None)
                        record_event(
                            t=now,
                            kind="curtail_deferred_to_qv_relief",
                            aid=self.context.aid,
                            sector=self.sector.value,
                            detail=f"v={value:.4f} hi={hi:.4f} relief={relief:.5f} "
                            f"defer={self._qv_defer_count[variable]}",
                        )
                        return
                    # Stalled (or backstop): escalate, shed full overshoot.
                    self._qv_defer_count.pop(variable, None)
                    self._qv_last_value.pop(variable, None)
                    record_event(
                        t=now,
                        kind="curtail_qv_defer_escalated",
                        aid=self.context.aid,
                        sector=self.sector.value,
                        detail=f"v={value:.4f} hi={hi:.4f} relief={relief:.5f}",
                    )

        # Strict reverse-priority line-relief waterfall auction?
        _waterfall = (
            self.enable_line_relief_waterfall
            and self.enable_branch_downstream_relief
            and variable == "loading_percent"
            and bool(self._downstream_load_addrs)
        )

        if _waterfall:
            # The waterfall self-terminates, so the generic no-progress gate is
            # the wrong stop. Stop only when only tier-1 bidders remain
            # (relieving further would break the hard-lock).
            if self._line_relief_tier1_residual.get(variable):
                return
        elif self.enable_curtail_auction_gating:
            # Progress gate: if the overshoot keeps failing to improve, stop
            # re-arming until it worsens or topology re-engages the lever.
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

        # Total fractional reduction across group + self. Two-phase auction:
        # broadcast need, collect bids, allocate proportional to willingness.
        _downstream_line = (
            self.enable_branch_downstream_relief
            and variable == "loading_percent"
            and bool(self._downstream_load_addrs)
        )
        if _downstream_line:
            # High gain to drive a 10-20% overload to feasibility in a few
            # rounds; priority still orders WHO sheds, re-arming until ≤100%.
            total_amount = min(1.0, max(0.25, _LINE_RELIEF_GAIN * overshoot))
        else:
            total_amount = max(0.02, min(1.0, self._CURTAILMENT_GAIN * overshoot))

        # Seed the agent's OWN load as a candidate (most direct lever on its
        # junction); priority still decides absorption.
        self_obs = self.behavior.observe(self.context.aid) or {}
        self_w_raw = (
            self._own_curtail_willingness(self_obs, injection_relief=injection_relief)
            if self.behavior.has_action(self.context.aid, "regulate")
            else None
        )
        # Drop a zero-willingness self so the all-zero even-split fallback in
        # ``_allocate_auction`` can't shed it.
        self_w = self_w_raw if (self_w_raw is not None and self_w_raw > 0.0) else None
        # Targeting: auctioneer is the origin (closest bidder), so scale its
        # self-bid by max proximity to compete with neighbour bids.
        if self.enable_curtail_auction_targeting and self_w is not None:
            self_w *= _CURTAIL_PROX_MAX

        # Branch-downstream relief: bidders are the loads flowing through the
        # branch (shed reduces its loading); else fall back to the component.
        if (
            self.enable_branch_downstream_relief
            and variable == "loading_percent"
            and self._downstream_load_addrs
        ):
            neighbors = list(self._downstream_load_addrs)
        else:
            neighbors = list(topology_neighbors(self, tid="groups"))

        if not neighbors and self_w is None:
            # Self locked, no neighbours — nothing allocable without breaking
            # the hard-lock. Clear the guard so a later poll retries.
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
            "injection_relief": injection_relief,
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
            injection_relief=injection_relief,
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

        willingness = self._own_curtail_willingness(
            obs, injection_relief=bool(getattr(message, "injection_relief", False))
        )
        # Targeting: scale by proximity to the origin so the share concentrates
        # on relieving loads (bounded within-tier, priority stays dominant).
        if self.enable_curtail_auction_targeting:
            willingness *= self._curtail_proximity(
                message.origin_addr, message.variable
            )
        # Carry tier + reducible for a waterfall auctioneer's reverse-priority
        # shed (ignored by the default proportional allocator).
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
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    def _curtail_proximity(self, origin_addr: Any, variable: str) -> float:
        """Bounded proximity multiplier for this bidder relative to the origin,
        from cached multi-hop distance (more ``hops_remaining`` => closer). No
        cached state => neutral 1.0 (never starves an unknown bidder).
        """
        if not variable or origin_addr is None or self.max_hops <= 0:
            return 1.0
        state = self._neighbour_state.get((str(origin_addr), variable))
        if state is None:
            return 1.0
        return proximity_from_hops(
            state.hops_remaining,
            self.max_hops,
            prox_min=_CURTAIL_PROX_MIN,
            prox_max=_CURTAIL_PROX_MAX,
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
        # Clear the in-flight guard so the next poll can open a new round.
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
        # Waterfall terminal state: only tier-1 bidders remain (relieving
        # further breaks the hard-lock). Surface once and stop re-arming.
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

        # Multiplicative reduction: amount=0.3 cuts output 30%; repeated
        # requests compound toward zero, so one step can't overshoot.
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
        """Drive this heat load's regulation to the t_k feasibility floor (max
        feasible service): partial-shed a cold node, restore a warm one, gained
        by local dT/dreg. Applies to all tiers incl. tier-1. Writes
        ``reason="curtail"``/``"heat_recovery"`` so the MW holon defers.

        Observation + ``apply_regulate`` plumbing live here; the step decision
        and peer gate live in ``self._heat_frontier``.
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
        if decision.reason == "defer_waterfall":
            # Own shed held — actively shed the lower-priority peer instead.
            await self._request_waterfall_peer_shed(my_tier)
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
                self.context.aid,
                t,
                lo + HeatFrontierController.MARGIN_K,
                cur,
                decision.new_reg,
            )

    # Bounded multiplicative shed step per peer request; the receiver's
    # ``_apply_curtail`` compounds repeated requests toward zero, so a single
    # step can't overshoot, and the per-peer cooldown paces escalation while
    # the requester's own poll re-fires each cycle the node stays cold.
    _HEAT_WATERFALL_SHED_AMOUNT: float = 0.5
    _HEAT_WATERFALL_REQUEST_COOLDOWN_S: float = 1.0

    async def _request_waterfall_peer_shed(self, my_tier: int) -> None:
        """Actuate the heat priority waterfall: while this cold load's own
        shed is deferred, send a bounded ``CurtailmentRequest`` to the
        lowest-priority reducible peer in range (one per poll). The receiver
        curtails with ``reason="curtail"``, taking the heat curtail-lock, so
        L2 defers and its own frontier restores it once the region warms.
        """
        now = self.context.current_timestamp
        for origin, tier, reducible in self._heat_frontier.waterfall_request_targets(
            my_tier, now
        ):
            deadline = self._waterfall_request_cooldown.get(origin)
            if deadline is not None and now < deadline:
                continue
            state = self._neighbour_state.get((origin, "t_k"))
            addr = getattr(state, "origin_addr", None) if state is not None else None
            if addr is None:
                continue
            self._waterfall_request_cooldown[origin] = (
                now + self._HEAT_WATERFALL_REQUEST_COOLDOWN_S
            )
            await self.context.send_message(
                CurtailmentRequest(
                    sector=self.sector, amount=self._HEAT_WATERFALL_SHED_AMOUNT
                ),
                receiver_addr=addr,
            )
            record_event(
                t=now,
                kind="heat_waterfall_peer_shed",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=(
                    f"target={origin} tier={tier} reducible={reducible:.4f} "
                    f"amount={self._HEAT_WATERFALL_SHED_AMOUNT}"
                ),
            )
            return

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def worst_neighbour_utilization(self) -> float:
        """Worst neighbour utilization in multi-hop range, weighted by the
        link's coupling weight K_ij (B.1) so low-trust links count less."""
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

    def _feeder_overvoltage(self, hi: float) -> bool:
        """True iff any OTHER feeder node reports a voltage above ``hi`` (shared
        ledger), so an inverter sheds active when the feeder is over. Stale data
        only over-reports — the safe direction."""
        mx = feeder_max_voltage(
            self.behavior, self.context.current_timestamp, exclude_aid=self.context.aid
        )
        return mx is not None and mx > hi + 1e-9

    def is_locally_feasible(self) -> bool:
        """True if no local constraint is currently violated."""
        return len(self._violation_emitted) == 0

    def local_sensitivity(self) -> float:
        """Latest |dV/dP| estimate for the primary constraint variable; defaults
        to a sector-typical prior until enough samples are collected."""
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
                    1.0 - _SENSITIVITY_EMA_ALPHA
                ) * self._sensitivity + _SENSITIVITY_EMA_ALPHA * sample
        self._last_p = p
        self._last_v = v
