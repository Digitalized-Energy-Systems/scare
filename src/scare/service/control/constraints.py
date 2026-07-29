"""Grid constraint monitoring and enforcement with multi-hop state propagation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from mango import Role
from mango import sender_addr as mango_sender_addr

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
    is_energised_reading,
)
from scare.base.runtime.diagnostics import record_event
from scare.base.util import (
    apply_regulate,
    async_dispatch,
    constraint_utilization,
    feeder_max_voltage,
    has_heat_curtail_lock,
    lookup_priority,
    obs_capacity,
    obs_constraint_values,
    obs_priority,
    obs_setpoint,
    publish_node_voltage,
    safe_observe,
)
from scare.service.balance.trust import TrustLedger, TrustParams
from scare.service.control.congestion_relief import CongestionRelief
from scare.service.control.constraint_propagation import StatePropagator
from scare.service.control.constraint_tuning import (
    _DEFAULT_MAX_HOPS,
    _EXPORT_DEBOUNCE_POLLS,
    _FORWARD_FRESHNESS_S,
    _HEAT_FRONTIER_PERIOD_S,
    _LINE_RELIEF_RELEASE_MARGIN,
    _SECTOR_PRIMARY_VAR,
    _SENSITIVITY_DEFAULT,
    _SENSITIVITY_EMA_ALPHA,
    _SENSITIVITY_MIN_DP,
    _VALUES_DELTA_TOL,
)
from scare.service.control.curtailment_auction import CurtailmentAuction
from scare.service.control.heat_frontier import HeatFrontierController

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class SensitivityEstimator:
    """EMA of |dV/dP| for the sector's primary constraint variable, seeded with a
    sector-typical prior. Behavior/aid for the obs lookups are passed per update."""

    def __init__(self, sector: Sector, seed: float | None = None) -> None:
        self._sector = sector
        self._value: float = (
            float(seed)
            if seed is not None and seed > 0.0
            else _SENSITIVITY_DEFAULT.get(sector, 1e-3)
        )
        self._last_p: float | None = None
        self._last_v: float | None = None

    @property
    def value(self) -> float:
        return self._value

    def update(self, obs: dict, behavior: Any, aid: str) -> None:
        var = _SECTOR_PRIMARY_VAR.get(self._sector)
        if var is None or var not in obs:
            return
        v = float(obs[var])
        if not math.isfinite(v):
            return
        cap = obs_capacity(obs, behavior=behavior, aid=aid)
        sp = obs_setpoint(obs, behavior=behavior, aid=aid)
        # Signed injection (sp negative for generators, cap < 0).
        p = sp if cap != 0.0 else 0.0
        if self._last_p is not None and self._last_v is not None:
            dp = p - self._last_p
            dv = v - self._last_v
            min_dp = _SENSITIVITY_MIN_DP.get(self._sector, 1e-6)
            if abs(dp) >= min_dp and math.isfinite(dv):
                sample = abs(dv / dp)
                # Clamp absurd jumps (post-failure snapshots) before the EMA.
                sample = min(sample, 10.0 * self._value + 1.0)
                self._value = (
                    1.0 - _SENSITIVITY_EMA_ALPHA
                ) * self._value + _SENSITIVITY_EMA_ALPHA * sample
        self._last_p = p
        self._last_v = v


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
        heat_sensitivity_seed_k_per_mw: float | None = None,
        heat_component_id: Any = None,
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
        self.enable_curtail_auction_gating = enable_curtail_auction_gating
        self.enable_curtail_auction_targeting = enable_curtail_auction_targeting
        self.enable_line_relief_reassert = enable_line_relief_reassert
        self.enable_branch_downstream_relief = enable_branch_downstream_relief
        # Strict reverse-priority cascade for downstream line relief.
        self.enable_line_relief_waterfall = enable_line_relief_waterfall
        self.enable_multihop_constraint = enable_multihop_constraint
        self.enable_heat_frontier = enable_heat_frontier
        # Heat priority-waterfall gate: a cold load defers its own shed while
        # lower-priority reducible heat load remains in its hydraulic region.
        self.enable_heat_priority_waterfall = enable_heat_priority_waterfall
        # Static water-subnetwork id (build-time connected component); scopes
        # waterfall partners to peers that actually share hydraulics.
        self._heat_component_id = heat_component_id
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

        # Variables with a violation emitted this episode (dedup guard).
        self._violation_emitted: set[str] = set()

        # Last-observed values; ``_monitor`` short-circuits when nothing moved.
        self._last_polled_values: dict[str, float] = {}

        # Heat frontier controller: owns the priority-waterfall peer cache and
        # frontier step state, decides the move toward the t_k feasibility floor.
        self._heat_frontier = HeatFrontierController(
            peer_freshness_s=2.0 * _FORWARD_FRESHNESS_S,
            component_id=heat_component_id,
        )

        # Local power-flow sensitivity: EMA of |dV/dP| from own (P, V) history,
        # letting the auction bid agents near the violation more aggressively.
        self._sens = SensitivityEstimator(
            sector,
            seed=(
                heat_sensitivity_seed_k_per_mw if sector is Sector.HEAT else None
            ),
        )

        poll_s = SECTOR_TIMESCALE.get(sector, {}).get("poll_period_s", 1.0)
        self._propagator = StatePropagator(
            self,
            TrustLedger(
                TrustParams(
                    decay_rate_per_s=1.0 / max(poll_s * 8.0, 1.0),
                    recover_rate=0.6,
                    liveness_threshold=0.5,
                    initial=1.0,
                )
            ),
        )
        self._relief = CongestionRelief(self, downstream_load_addrs)
        self._auction = CurtailmentAuction(self)

        # Heat waterfall actuator: per-peer cooldown deadline for the curtail
        # requests a deferring cold load sends to lower-priority peers.
        self._waterfall_request_cooldown: dict[str, float] = {}

    # Helper-owned state kept reachable under its original name: the residual
    # monitor logic and the tests read (and mutate in place) these maps.
    @property
    def _neighbour_state(self):
        return self._propagator._neighbour_state

    @property
    def _downstream_load_addrs(self):
        return self._relief._downstream_load_addrs

    @property
    def _export_streak(self):
        return self._relief._export_streak

    @property
    def _relief_inflight(self):
        return self._relief._relief_inflight

    @property
    def _open_auctions(self):
        return self._auction._open_auctions

    @property
    def _curtail_inflight(self):
        return self._auction._curtail_inflight

    @property
    def _curtail_progress(self):
        return self._auction._curtail_progress

    @property
    def _line_relief_tier1_residual(self):
        return self._auction._line_relief_tier1_residual

    def set_downstream_loads(self, addrs: list[Any]) -> None:
        """Set the loads downstream of this branch, once addresses resolve."""
        self._relief._downstream_load_addrs = list(addrs)

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

        _wrap = async_dispatch(self)

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
        return safe_observe(self.behavior, self.context.aid)

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
        # Export (reverse-flow) overload: shedding downstream load raises net
        # export and worsens it, so all load-shed paths are suppressed and
        # downstream generation is curtailed instead. Debounced over polls, and
        # owned only while curtailable downstream generators exist.
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
        # Legacy debounce sheds 1-2 polls of downstream load pre-gen-curtail and
        # never reverts; so under gen-priority/congestion-price suppress load-shed
        # the moment an overload looks export-driven with curtailable downstream
        # gens (pre-debounce), while gen-curtail below stays debounced.
        if (
            self.enable_generation_priority_curtailment
            or self.enable_line_congestion_price
        ):
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
        if (
            not self._violation_emitted
            and self._last_polled_values
            and not self._needs_per_poll_tick()
        ):
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

        self._sens.update(obs, self.behavior, self.context.aid)

        for var, val in values.items():
            # Skip de-energised / non-finite readings (see is_energised_reading):
            # no curtail lever re-energises them; genuine out-of-bound values sit
            # inside the band and still fire.
            if not is_energised_reading(var, val):
                continue

            lo, hi = bounds.get(var, (float("-inf"), float("inf")))
            util = constraint_utilization(val, lo, hi)

            # Feed the line-relief hand-off its headroom and re-stamp any live
            # restore-ramp lock. On a line-relief branch _needs_per_poll_tick()
            # forces per-poll execution, so the cache gate can't short-circuit
            # here and the lock is re-stamped every poll (liveness guaranteed).
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
        return await self._propagator._propagate_state(
            variable, value, utilization, obs
        )

    async def _handle_constraint_state(
        self, message: ConstraintStateMessage, meta: dict
    ) -> None:
        return await self._propagator._handle_constraint_state(message, meta)

    # ------------------------------------------------------------------
    # Branch-mode helpers
    # ------------------------------------------------------------------

    async def _reassert_line_relief(
        self, obs: dict, var: str, val: float, lo: float, hi: float
    ) -> None:
        return await self._relief._reassert_line_relief(obs, var, val, lo, hi)

    async def _send_line_overload_relief(
        self, obs: dict, val: float, lo: float, hi: float
    ) -> None:
        return await self._relief._send_line_overload_relief(obs, val, lo, hi)

    def _is_line_relief_branch(self) -> bool:
        return self._relief._is_line_relief_branch()

    def _needs_per_poll_tick(self) -> bool:
        return self._relief._needs_per_poll_tick()

    def _hold_downstream_line_locks(self, var: str, val: float, hi: float) -> None:
        return self._relief._hold_downstream_line_locks(var, val, hi)

    def _maintain_line_relief_handoff(self, var: str, val: float, hi: float) -> None:
        return self._relief._maintain_line_relief_handoff(var, val, hi)

    def _maintain_congestion_price(
        self, obs: dict, var: str, val: float, hi: float
    ) -> None:
        return self._relief._maintain_congestion_price(obs, var, val, hi)

    # ------------------------------------------------------------------
    # Flow direction / export-overload relief
    # ------------------------------------------------------------------

    def _flow_is_export(self, obs: dict) -> bool | None:
        return self._relief._flow_is_export(obs)

    def _ensure_downstream_topology(self) -> None:
        return self._relief._ensure_downstream_topology()

    def _resolve_downstream_topology(self) -> None:
        return self._relief._resolve_downstream_topology()

    def _downstream_generator_aids(self) -> list[str]:
        return self._relief._downstream_generator_aids()

    async def _relieve_export_overload(
        self, obs: dict, var: str, val: float, hi: float
    ) -> None:
        return await self._relief._relieve_export_overload(obs, var, val, hi)

    # ------------------------------------------------------------------
    # Curtailment
    # ------------------------------------------------------------------

    def _own_curtail_willingness(
        self, obs: dict, *, injection_relief: bool = False
    ) -> float:
        return self._auction._own_curtail_willingness(
            obs, injection_relief=injection_relief
        )

    async def _request_curtailment(
        self, variable: str, value: float, lo: float, hi: float
    ) -> None:
        return await self._auction._request_curtailment(variable, value, lo, hi)

    async def _handle_curtailment_need(
        self, message: CurtailmentNeed, meta: dict
    ) -> None:
        return await self._auction._handle_curtailment_need(message, meta)

    def _curtail_proximity(self, origin_addr: Any, variable: str) -> float:
        return self._auction._curtail_proximity(origin_addr, variable)

    async def _handle_curtailment_bid(
        self, message: CurtailmentBid, meta: dict
    ) -> None:
        return await self._auction._handle_curtailment_bid(message, meta)

    async def _close_auction(self, auction_id: str) -> None:
        return await self._auction._close_auction(auction_id)

    async def _allocate_auction(self, auction_id: str) -> None:
        return await self._auction._allocate_auction(auction_id)

    async def _handle_curtailment_request(
        self, message: CurtailmentRequest, meta: dict
    ) -> None:
        return await self._auction._handle_curtailment_request(message, meta)

    async def _curtail_self(self, amount: float) -> None:
        return await self._auction._curtail_self(amount)

    async def _apply_curtail(self, amount: float, *, label: str) -> None:
        return await self._auction._apply_curtail(amount, label=label)

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
            sensitivity=self._sens.value,
            now=self.context.current_timestamp,
            my_tier=my_tier,
            has_lock=has_heat_curtail_lock(self.behavior, self.context.aid),
            waterfall_enabled=self.enable_heat_priority_waterfall,
            aid=str(self.context.aid),
        )
        too_cold = t < (
            lo + HeatFrontierController.MARGIN_K - HeatFrontierController.DEADBAND_K
        )
        if decision is None:
            # Fully-shed (or sub-threshold) but still cold: peer shed is the
            # only remaining lever — keep requesting while the node is cold.
            if too_cold and self.enable_heat_priority_waterfall:
                await self._request_waterfall_peer_shed(my_tier)
            return
        if decision.reason == "defer_waterfall":
            # Own shed held — actively shed the lower-priority peers instead.
            await self._request_waterfall_peer_shed(my_tier, decision.needed_mw)
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
        # Safety-valve escalation: shedding self (insufficient peers or defer
        # budget exhausted) still shifts what it can onto lower tiers.
        if decision.reason == "curtail" and self.enable_heat_priority_waterfall:
            await self._request_waterfall_peer_shed(my_tier, decision.needed_mw)

    # Bounded multiplicative shed step per peer request; the receiver's
    # ``_apply_curtail`` compounds repeated requests toward zero, so a single
    # step can't overshoot, and the per-peer cooldown paces escalation while
    # the requester's own poll re-fires each cycle the node stays cold.
    _HEAT_WATERFALL_SHED_AMOUNT: float = 0.5
    _HEAT_WATERFALL_REQUEST_COOLDOWN_S: float = 1.0
    _HEAT_WATERFALL_MAX_TARGETS: int = 3

    async def _request_waterfall_peer_shed(
        self, my_tier: int, needed_mw: float = 0.0
    ) -> None:
        """Actuate the heat priority waterfall: send bounded
        ``CurtailmentRequest``s to the lowest-priority reducible peers in the
        own hydraulic component, until the expected relief covers this poll's
        needed shed (or the per-poll target cap). The receiver curtails with
        ``reason="curtail"``, taking the heat curtail-lock, so L2 defers and
        its own frontier restores it once the region warms.
        """
        now = self.context.current_timestamp
        sent = 0
        expected_mw = 0.0
        for origin, tier, reducible in self._heat_frontier.waterfall_request_targets(
            my_tier, now, needed_mw
        ):
            if sent >= self._HEAT_WATERFALL_MAX_TARGETS:
                break
            if needed_mw > 0.0 and expected_mw >= needed_mw:
                break
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
            sent += 1
            expected_mw += self._HEAT_WATERFALL_SHED_AMOUNT * reducible
            record_event(
                t=now,
                kind="heat_waterfall_peer_shed",
                aid=self.context.aid,
                sector=self.sector.value,
                detail=(
                    f"target={origin} tier={tier} reducible={reducible:.4f} "
                    f"amount={self._HEAT_WATERFALL_SHED_AMOUNT} "
                    f"needed={needed_mw:.4f}"
                ),
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def worst_neighbour_utilization(self) -> float:
        return self._propagator.worst_neighbour_utilization()

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
        return self._sens.value
