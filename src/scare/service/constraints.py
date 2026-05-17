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
from typing import TYPE_CHECKING, Any

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors

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
from scare.base.util import obs_priority
from scare.base.util import (
    constraint_utilization,
    obs_capacity,
    obs_constraint_values,
    obs_setpoint,
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
# per-cycle flood that ``_forwarded.clear()`` used to enable.
_FORWARD_FRESHNESS_S: float = 5.0

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
    Sector.HEAT: 0.5,           # W or kg/s scaled
}

# Default sensitivity used before any samples have been collected.
_SENSITIVITY_DEFAULT: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,   # p.u. voltage per MW
    Sector.GAS: 0.5,            # p.u. pressure per kg/s
    Sector.HEAT: 1e-5,          # K per W
}

# Primary constraint variable per sector for sensitivity tracking.
_SECTOR_PRIMARY_VAR: dict[Sector, str] = {
    Sector.ELECTRICITY: "vm_pu",
    Sector.GAS: "pressure_pu",
    Sector.HEAT: "t_k",
}


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
        enable_multihop_constraint: bool = True,
        enable_heat_recovery: bool = True,
        branch_id: Any = None,
        home_leader_addr: Any = None,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.node_id = node_id
        self.max_hops = max_hops
        self.enable_curtailment_auction = enable_curtailment_auction
        self.enable_multihop_constraint = enable_multihop_constraint
        self.enable_heat_recovery = enable_heat_recovery
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
        self._forwarded: dict[
            tuple[str, str], tuple[int, float, float]
        ] = {}
        # When this agent last broadcast its OWN state, indexed by
        # variable.  Used by ``_propagate_state`` to decide whether the
        # local poll has produced a new-enough datum to flood again.
        self._last_local_propagate: dict[str, tuple[float, float]] = {}

        # Track whether we already emitted a violation this cycle to
        # avoid flooding.
        self._violation_emitted: set[str] = set()

        # B.1: continuous coupling weights K_ij for the constraint
        # propagation overlay.  Independent of the balance negotiator's
        # ledger because the topology and message frequencies differ.
        # Used to (a) weight the worst-neighbour utilisation by trust,
        # (b) skip forwarding to neighbours whose K is below the
        # liveness threshold.
        from scare.base.trust import TrustLedger, TrustParams

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

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 1.0)
        self.context.schedule_periodic_task(self._monitor, delay=poll)
        # Heat sector: gradually un-shed previously curtailed loads once
        # the local thermal stress has cleared.  The Level-1 gossip
        # produces shed-only deltas during a violation; without an
        # explicit recovery loop the load stays at the reduced factor
        # forever.  Run at the heat poll period, slightly slower than
        # the monitor so the violation flag has time to drop.
        if self.sector == Sector.HEAT and self.enable_heat_recovery:
            self.context.schedule_periodic_task(
                self._heat_recovery, delay=poll * 1.5
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

    async def _monitor(self) -> None:
        # ``observe`` can raise ``AttributeError`` when the simulation
        # hasn't run an LP solve yet (``_net_results`` is None on the
        # very first scheduled tick).  This is benign — the monitor
        # just has nothing to look at — so swallow it instead of
        # letting mango's task wrapper log a noisy traceback.  The
        # next periodic tick will see the populated results.
        try:
            obs = self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return
        if not obs:
            return

        bounds = SECTOR_CONSTRAINTS.get(self.sector, {})
        values = obs_constraint_values(obs, self.sector)

        # Deduplication state is now persistent across cycles — see
        # ``_propagate_state`` / ``_handle_constraint_state`` for the
        # value-delta + freshness-window suppression rules that replace
        # the per-cycle ``clear()``.

        # Update local sensitivity estimate from own (P, V) history.
        self._update_sensitivity(obs)

        for var, val in values.items():
            # Skip readings the solver hasn't populated (post-failure
            # infeasibility reports t_k=0, NaN shows up on isolated nodes).
            # Acting on those triggers spurious curtailment cascades.
            if not math.isfinite(val) or (var == "t_k" and val <= 0.0):
                continue

            lo, hi = bounds.get(var, (float("-inf"), float("inf")))
            util = constraint_utilization(val, lo, hi)

            # --- Hard violation ---
            if val < lo or val > hi:
                if var not in self._violation_emitted:
                    self._violation_emitted.add(var)
                    violation = ConstraintViolation(
                        sector=self.sector,
                        variable=var,
                        value=val,
                        bound_low=lo,
                        bound_high=hi,
                        node_id=self.node_id,
                    )
                    logger.warning(
                        "[%s] CONSTRAINT VIOLATION %s=%.4f bounds=[%.4f,%.4f]",
                        self.context.aid,
                        var,
                        val,
                        lo,
                        hi,
                    )
                    from scare.base.diagnostics import record_event

                    record_event(
                        t=self.context.current_timestamp,
                        kind="constraint_violation",
                        aid=self.context.aid,
                        sector=self.sector.value,
                        detail=f"{var}={val:.4f} bounds=[{lo:.4f},{hi:.4f}]",
                    )
                    # Local emits raise ``KeyError`` in mango when no
                    # co-located role subscribes — branch agents in
                    # branch mode have no EnergyBalanceNegotiator, and
                    # the inter-agent signal goes via ``_send_line_overload_relief``
                    # below.  Swallow defensively so the monitor task
                    # doesn't abort before that send executes.
                    try:
                        self.context.emit_event(violation)
                    except KeyError:
                        pass
                    try:
                        self.context.emit_event(
                            BalanceProblem(
                                sector=self.sector,
                                imbalance=val - hi if val > hi else lo - val,
                            )
                        )
                    except KeyError:
                        pass
                    if (
                        self.branch_id is not None
                        and var == "loading_percent"
                        and self.home_leader_addr is not None
                    ):
                        # Branch mode: BalanceProblem is local and has no
                        # listener here.  Drive the home group leader to
                        # rebalance by sending an explicit
                        # StartBalanceNegotiation with a relief-MW target.
                        await self._send_line_overload_relief(
                            obs, val, lo, hi
                        )
                    if self.enable_curtailment_auction:
                        await self._request_curtailment(var, val, lo, hi)
            else:
                self._violation_emitted.discard(var)

            # --- Proactive warning ---
            if util >= PROACTIVE_WARNING_FRACTION and var not in self._violation_emitted:
                warning = ConstraintWarning(
                    sector=self.sector,
                    variable=var,
                    value=val,
                    bound_low=lo,
                    bound_high=hi,
                    utilization=util,
                    node_id=self.node_id,
                )
                # Same defensive pattern as the ``ConstraintViolation``
                # / ``BalanceProblem`` emits above: branch-mode monitor
                # agents have no co-located negotiator to subscribe.
                try:
                    self.context.emit_event(warning)
                except KeyError:
                    pass
                logger.debug(
                    "[%s] constraint warning %s=%.4f util=%.2f",
                    self.context.aid,
                    var,
                    val,
                    util,
                )

            # --- Propagate state to neighbours ---
            if self.enable_multihop_constraint:
                await self._propagate_state(var, val, util)

    # ------------------------------------------------------------------
    # Multi-hop state propagation with deduplication
    # ------------------------------------------------------------------

    async def _propagate_state(
        self, variable: str, value: float, utilization: float
    ) -> None:
        # Suppress re-broadcasts of an unchanged value unless the
        # freshness window has elapsed (keeps trust-ledger liveness alive)
        # OR the utilization moved by more than ``_FORWARD_VALUE_TOL``.
        # Without this, every monitor cycle re-broadcasts the same value
        # to every neighbour, swamping the network — task-0 baseline had
        # 119 480 ConstraintStateMessages in 5 s on this path alone.
        now = self.context.current_timestamp
        prev = self._last_local_propagate.get(variable)
        if prev is not None:
            prev_t, prev_util = prev
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (stale or changed):
                return

        origin = self.context.addr
        msg = ConstraintStateMessage(
            sector=self.sector,
            variable=variable,
            value=value,
            utilization=utilization,
            hops_remaining=self.max_hops,
            origin_addr=origin,
        )
        origin_key = (str(origin), variable)
        self._forwarded[origin_key] = (self.max_hops, now, utilization)
        self._last_local_propagate[variable] = (now, utilization)

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

        # --- Deduplication ---
        # Forward only if the incoming copy improves on what we've
        # already forwarded for this (origin, variable): either
        # ``hops_remaining`` is strictly larger (fresher / closer to
        # origin), or the freshness window has elapsed, or the value
        # moved by more than the tolerance.  Updated lazily; never
        # cleared (the per-cycle clear caused the message flood).
        prev = self._forwarded.get(origin_key)
        if prev is not None:
            prev_hops, prev_t, prev_util = prev
            improves_hops = message.hops_remaining > prev_hops
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(message.utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (improves_hops or stale or changed):
                return
        self._forwarded[origin_key] = (
            message.hops_remaining, now, message.utilization,
        )

        if message.hops_remaining <= 1:
            return  # TTL exhausted

        fwd = ConstraintStateMessage(
            sector=message.sector,
            variable=message.variable,
            value=message.value,
            utilization=message.utilization,
            hops_remaining=message.hops_remaining - 1,
            origin_addr=message.origin_addr,
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

    async def _request_curtailment(
        self, variable: str, value: float, lo: float, hi: float
    ) -> None:
        span = hi - lo
        if span <= 0:
            return
        overshoot = (value - hi) / span if value > hi else (lo - value) / span

        neighbors = list(topology_neighbors(self, tid="groups"))
        if not neighbors:
            return

        # Total fractional reduction needed across the group.  Announced
        # via a two-phase auction: broadcast the *need*, collect bids,
        # then allocate proportional to each neighbour's self-reported
        # willingness (priority × local sensitivity × reducible output).
        total_amount = max(0.02, min(1.0, self._CURTAILMENT_GAIN * overshoot))

        import uuid
        auction_id = str(uuid.uuid4())
        self._open_auctions[auction_id] = {
            "bids": {},
            "total": total_amount,
            "neighbours_contacted": len(neighbors),
            "bidders": {},  # sender_key -> addr
        }

        need_msg = CurtailmentNeed(
            sector=self.sector,
            total_amount=total_amount,
            auction_id=auction_id,
        )
        for addr in neighbors:
            await self.context.send_message(need_msg, receiver_addr=addr)

        deadline = (
            self.context.current_timestamp + self._AUCTION_TIMEOUT_S
        )
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

        # Willingness: bigger = more happy / more effective at absorbing
        # curtailment.  Combines three purely local signals:
        #   - priority tier weight (exponential schedule consistent with
        #     the curtailment-regime w(π, -1) = 2^π used in the Layer-1
        #     QP: tier 10 → 1024, tier 1 → 2; gives a 512× spread instead
        #     of the 10× linear spread used historically)
        #   - local |dV/dP| sensitivity (high = curtailment here cheaply
        #     moves the violated variable)
        #   - current reducible output magnitude (nothing to curtail → nothing
        #     to contribute).
        from scare.service.balance import _PRIORITY_TIERS

        prio_tier = max(1, obs_priority(obs, behavior=self.behavior, aid=self.context.aid))
        prio_weight = 2.0 ** min(prio_tier, _PRIORITY_TIERS)
        reducible = abs(obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid))
        willingness = prio_weight * self._sensitivity * reducible
        if not math.isfinite(willingness) or willingness <= 0.0:
            willingness = 1e-9

        reply = CurtailmentBid(
            auction_id=message.auction_id,
            willingness=willingness,
            sector=self.sector,
        )
        await self.context.send_message(
            reply, receiver_addr=mango_sender_addr(meta)
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

        if len(auction["bids"]) >= auction["neighbours_contacted"]:
            await self._allocate_auction(message.auction_id)

    async def _close_auction(self, auction_id: str) -> None:
        if auction_id in self._open_auctions:
            await self._allocate_auction(auction_id)

    async def _allocate_auction(self, auction_id: str) -> None:
        auction = self._open_auctions.pop(auction_id, None)
        if auction is None:
            return
        bids: dict[str, float] = auction["bids"]
        bidders: dict[str, Any] = auction["bidders"]
        total_amount: float = auction["total"]

        if not bids:
            return

        sum_w = sum(bids.values())
        if sum_w <= 0.0:
            # All willingness scores zero — fall back to an even split
            # so at least something curtails and the violation can clear.
            share = total_amount / len(bids)
            for key, addr in bidders.items():
                await self.context.send_message(
                    CurtailmentRequest(sector=self.sector, amount=share),
                    receiver_addr=addr,
                )
            return

        for key, w in bids.items():
            addr = bidders.get(key)
            if addr is None:
                continue
            share = total_amount * (w / sum_w)
            if share <= 0.0:
                continue
            await self.context.send_message(
                CurtailmentRequest(sector=self.sector, amount=share),
                receiver_addr=addr,
            )

    async def _handle_curtailment_request(
        self, message: CurtailmentRequest, meta: dict
    ) -> None:
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
        amount = max(0.0, min(1.0, message.amount))
        new_factor = max(0.0, current * (1.0 - amount))

        from scare.base.util import apply_regulate

        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            new_factor,
            sector=self.sector.value,
            reason="curtail",
            timestamp=self.context.current_timestamp,
        )
        if applied:
            logger.info(
                "[%s] curtailed by %.1f%% (regulation %.3f -> %.3f)",
                self.context.aid,
                amount * 100,
                current,
                new_factor,
            )

    # ------------------------------------------------------------------
    # Heat recovery (un-shed)
    # ------------------------------------------------------------------

    # Fraction of the feasible band below which the agent is considered
    # comfortably clear and may begin un-shedding.  Matched to
    # ``_HEAT_CLEAR_FRACTION`` in service/balance.py: an agent only
    # contributes to the deficit target above this point, so the same
    # threshold defines "no longer stressed".
    _HEAT_RECOVERY_CLEAR_FRACTION: float = 0.6

    async def _heat_recovery(self) -> None:
        if self.sector != Sector.HEAT:
            return
        if not self.is_locally_feasible():
            return
        if not self.behavior.has_action(self.context.aid, "regulate"):
            return
        # ``observe`` raises ``AttributeError`` when the simulation hasn't
        # run a solve yet (``_net_results`` is None on the very first
        # scheduled tick).  Match the defensive pattern in ``_monitor``
        # so the periodic task doesn't log 200+ tracebacks per run.
        try:
            obs = self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return
        if not obs:
            return

        bounds = SECTOR_CONSTRAINTS.get(self.sector, {})
        worst_util = 0.0
        for var, val in obs_constraint_values(obs, self.sector).items():
            if not math.isfinite(val) or (var == "t_k" and val <= 0.0):
                return  # solver hasn't populated this junction yet
            lo, hi = bounds.get(var, (float("-inf"), float("inf")))
            worst_util = max(worst_util, constraint_utilization(val, lo, hi))
        if worst_util > self._HEAT_RECOVERY_CLEAR_FRACTION:
            return

        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap <= 0:  # generators are not subject to recovery (DGs ramp via their own role)
            return
        current = float(obs.get("regulation", 1.0))
        if current >= 1.0:
            return

        # Ramp at the heat-sector convergence rate per recovery period.
        # 0.15 / s × 1.5×poll(=5 s) ≈ +1.125 per cycle if unbounded; the
        # min(1.0, ...) clamp caps a single step at full restoration.
        rate_per_s = SECTOR_TIMESCALE.get(self.sector, {}).get(
            "convergence_rate", 0.15
        )
        period_s = SECTOR_TIMESCALE.get(self.sector, {}).get(
            "poll_period_s", 5.0
        ) * 1.5
        new_factor = min(1.0, current + rate_per_s * period_s * 0.2)
        if new_factor <= current + 1e-4:
            return

        from scare.base.util import apply_regulate

        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            new_factor,
            sector=self.sector.value,
            reason="heat_recovery",
            timestamp=self.context.current_timestamp,
        )
        if applied:
            logger.info(
                "[%s] heat recovery: regulation %.3f -> %.3f (util=%.2f)",
                self.context.aid,
                current,
                new_factor,
                worst_util,
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
