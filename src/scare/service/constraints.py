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
    BalanceProblem,
    ConstraintStateMessage,
    ConstraintViolation,
    ConstraintWarning,
    CurtailmentRequest,
    Sector,
)
from scare.base.util import constraint_utilization, obs_constraint_values

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# How many hops constraint state information propagates.
_DEFAULT_MAX_HOPS = 3


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
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.node_id = node_id
        self.max_hops = max_hops

        # Neighbour constraint state cache:
        # (origin_addr_str, variable) -> ConstraintStateMessage
        self._neighbour_state: dict[tuple[str, str], ConstraintStateMessage] = {}

        # Deduplication: track which (origin, variable, hops_remaining)
        # we have already forwarded to prevent loops in meshed topologies.
        # Keyed by (origin_addr_str, variable); value is the lowest
        # hops_remaining we have seen — we only forward if the incoming
        # hops_remaining is higher (i.e. fresher / closer to origin).
        self._forwarded: dict[tuple[str, str], int] = {}

        # Track whether we already emitted a violation this cycle to
        # avoid flooding.
        self._violation_emitted: set[str] = set()

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 1.0)
        self.context.schedule_periodic_task(self._monitor, delay=poll)

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

    # ------------------------------------------------------------------
    # Periodic monitoring
    # ------------------------------------------------------------------

    async def _monitor(self) -> None:
        obs = self.behavior.observe(self.context.aid)
        if not obs:
            return

        bounds = SECTOR_CONSTRAINTS.get(self.sector, {})
        values = obs_constraint_values(obs, self.sector)

        # Reset deduplication table each monitoring cycle so fresh
        # information can propagate.
        self._forwarded.clear()

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
                    self.context.emit_event(violation)
                    self.context.emit_event(
                        BalanceProblem(
                            sector=self.sector,
                            imbalance=val - hi if val > hi else lo - val,
                        )
                    )
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
                self.context.emit_event(warning)
                logger.debug(
                    "[%s] constraint warning %s=%.4f util=%.2f",
                    self.context.aid,
                    var,
                    val,
                    util,
                )

            # --- Propagate state to neighbours ---
            await self._propagate_state(var, val, util)

    # ------------------------------------------------------------------
    # Multi-hop state propagation with deduplication
    # ------------------------------------------------------------------

    async def _propagate_state(
        self, variable: str, value: float, utilization: float
    ) -> None:
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
        self._forwarded[origin_key] = self.max_hops

        for addr in topology_neighbors(self, tid="groups"):
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_constraint_state(
        self, message: ConstraintStateMessage, meta: dict
    ) -> None:
        origin_key = (str(message.origin_addr), message.variable)

        # Cache the latest state from this origin
        self._neighbour_state[origin_key] = message

        # --- Deduplication ---
        # Only forward if we haven't already forwarded a fresher copy
        # (higher hops_remaining) from the same origin for this variable.
        prev_hops = self._forwarded.get(origin_key)
        if prev_hops is not None and message.hops_remaining <= prev_hops:
            return  # already forwarded a fresher or equal copy
        self._forwarded[origin_key] = message.hops_remaining

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
        sender = mango_sender_addr(meta)
        for addr in topology_neighbors(self, tid="groups"):
            # Don't send back to the origin or the immediate sender
            if addr == message.origin_addr or addr == sender:
                continue
            await self.context.send_message(fwd, receiver_addr=addr)

    # ------------------------------------------------------------------
    # Curtailment
    # ------------------------------------------------------------------

    # Proportional-controller gain applied to the normalized overshoot.
    # Small enough that a borderline violation produces a gentle step;
    # if the violation persists, the next monitor cycle re-emits and
    # the response ratchets up. Prevents one-shot over-curtailment.
    _CURTAILMENT_GAIN: float = 0.3

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

        # Total fractional reduction across the group, split evenly so
        # aggregate curtailment tracks the overshoot instead of scaling
        # with fan-out. Floor keeps tiny overshoots from being ignored.
        total_amount = max(0.02, min(1.0, self._CURTAILMENT_GAIN * overshoot))
        per_neighbor = total_amount / len(neighbors)

        msg = CurtailmentRequest(sector=self.sector, amount=per_neighbor)
        for addr in neighbors:
            await self.context.send_message(msg, receiver_addr=addr)

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

        self.behavior.act(self.context.aid, "regulate", new_factor)
        logger.info(
            "[%s] curtailed by %.1f%% (regulation %.3f -> %.3f)",
            self.context.aid,
            amount * 100,
            current,
            new_factor,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def worst_neighbour_utilization(self) -> float:
        """Return the worst constraint utilization reported by any
        neighbour within multi-hop range.  Used by the balance negotiator
        to apply conservative margins."""
        if not self._neighbour_state:
            return 0.0
        return max(s.utilization for s in self._neighbour_state.values())

    def is_locally_feasible(self) -> bool:
        """True if no local constraint is currently violated."""
        return len(self._violation_emitted) == 0
