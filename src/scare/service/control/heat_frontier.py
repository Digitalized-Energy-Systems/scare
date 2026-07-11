"""Heat-sector frontier feedback controller.

Owns the heat priority-waterfall peer cache and frontier step state, and decides
the regulation move that drives a heat load's junction temperature to the
feasibility floor (max feasible service) rather than bang-bang to zero. A plain
object (pure arithmetic + state), unit-testable without a mango context.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class FrontierDecision(NamedTuple):
    new_reg: float
    # "curtail" (shed self) | "heat_recovery" (restore self) |
    # "defer_waterfall" (hold: ask a lower-priority peer to shed instead;
    # new_reg is the unchanged current regulation, nothing to actuate)
    reason: str
    # MW-equivalent of this poll's shed step — sizes the waterfall peer
    # requests (0.0 on restore decisions).
    needed_mw: float = 0.0


class HeatFrontierController:
    # Hold t_k a small margin above the hard floor.
    MARGIN_K: float = 3.0
    # Below ``target - DEADBAND`` -> shed; above ``target + RESTORE_BAND`` ->
    # restore. The wide, asymmetric restore band is hysteresis against the
    # restore<->re-violate limit cycle for nodes that re-cool when served.
    DEADBAND_K: float = 2.0
    RESTORE_BAND_K: float = 6.0
    # Proportional gain and per-poll step clamp (clamp bounds the move on a
    # stale sensitivity; the P term settles at the frontier as dT/dP learns).
    GAIN: float = 0.5
    MAX_STEP: float = 0.15
    # Waterfall targeting floor: a peer is a meaningful shed target only above
    # this absolute draw (MW) or this share of the deferring load's own needed
    # relief. Kilowatt remnants can't warm a node — and because the peer shed
    # is multiplicative (never reaches zero), an existence-based guard on them
    # deadlocked the deferring load's own shed forever (eval_full_v2 just_heat).
    WATERFALL_TARGET_MIN_MW: float = 1e-3
    WATERFALL_TARGET_NEED_SHARE: float = 0.1
    # Defer the own shed only while lower-priority reducible covers this share
    # of the own step (sufficiency, not existence).
    WATERFALL_SUFFICIENCY: float = 1.0
    # Defer budget: after this many consecutive deferred polls without the
    # node warming by DEFER_IMPROVE_K, resume the own shed — peer sheds are
    # evidently not reaching this node (wrong region / too shallow), and t_k
    # feasibility must win over ordering.
    WATERFALL_DEFER_POLLS: int = 5
    WATERFALL_DEFER_IMPROVE_K: float = 0.5

    def __init__(
        self, *, peer_freshness_s: float, component_id: Any = None
    ) -> None:
        self._peer_freshness_s = peer_freshness_s
        # Static water-subnetwork id of the own node; peers advertising a
        # different component can't warm this node and are never waterfall
        # partners. ``None`` (either side) admits all — legacy mesh reach.
        self._component_id = component_id
        # origin -> (t_received, priority_tier, reducible, component_id) from
        # heat ``t_k`` constraint-state messages; read with a freshness window.
        self._peer_state: dict[str, tuple[float, int, float, Any]] = {}
        # Sign of the last committed step; halving on reversal damps the
        # frontier limit cycle.
        self._last_dir: float = 0.0
        # Defer budget state, live only while the node stays too cold.
        self._defer_streak: int = 0
        self._defer_anchor_t: float | None = None
        self._defer_exhausted: bool = False

    def note_peer_state(
        self,
        origin: str,
        t_received: float,
        tier: int,
        reducible: float,
        component_id: Any = None,
    ) -> None:
        """Cache a peer heat load's (tier, reducible), stamped for freshness."""
        self._peer_state[origin] = (t_received, tier, reducible, component_id)

    def _same_component(self, peer_component: Any) -> bool:
        if self._component_id is None or peer_component is None:
            return True
        return peer_component == self._component_id

    def _target_floor_mw(self, needed_mw: float) -> float:
        return max(
            self.WATERFALL_TARGET_MIN_MW,
            self.WATERFALL_TARGET_NEED_SHARE * needed_mw,
        )

    def waterfall_request_targets(
        self, my_tier: int, now: float, needed_mw: float = 0.0
    ) -> list[tuple[str, int, float]]:
        """Fresh same-component strictly-lower-priority peers with meaningful
        reducible draw, ordered lowest-priority (highest tier) then
        most-reducible first — the shed order for the waterfall's peer curtail
        requests."""
        floor = self._target_floor_mw(needed_mw)
        peers = [
            (origin, tier, reducible)
            for origin, (t_rx, tier, reducible, comp) in self._peer_state.items()
            if now - t_rx <= self._peer_freshness_s
            and tier > my_tier
            and reducible >= floor
            and self._same_component(comp)
        ]
        peers.sort(key=lambda p: (-p[1], -p[2]))
        return peers

    def region_has_lower_priority_reducible(
        self, my_tier: int, now: float, needed_mw: float = 0.0
    ) -> float:
        """Total meaningful reducible draw of fresh same-component peers at
        strictly lower priority than ``my_tier``."""
        return sum(
            reducible
            for _origin, _tier, reducible in self.waterfall_request_targets(
                my_tier, now, needed_mw
            )
        )

    def _reset_defer(self) -> None:
        self._defer_streak = 0
        self._defer_anchor_t = None
        self._defer_exhausted = False

    def _defer_allowed(
        self, t: float, now: float, my_tier: int, needed_mw: float
    ) -> bool:
        """Sufficiency + budget gate for holding the own shed this poll."""
        total = self.region_has_lower_priority_reducible(my_tier, now, needed_mw)
        if total < self.WATERFALL_SUFFICIENCY * needed_mw:
            return False
        # Real warming since the streak anchor -> deferral is working; re-arm.
        if (
            self._defer_anchor_t is not None
            and t >= self._defer_anchor_t + self.WATERFALL_DEFER_IMPROVE_K
        ):
            self._reset_defer()
        if self._defer_exhausted:
            return False
        if self._defer_anchor_t is None:
            self._defer_anchor_t = t
        self._defer_streak += 1
        if self._defer_streak > self.WATERFALL_DEFER_POLLS:
            self._defer_exhausted = True
            return False
        return True

    def decide(
        self,
        *,
        t: float,
        lo: float,
        cap: float,
        cur: float,
        sensitivity: float,
        now: float,
        my_tier: int,
        has_lock: bool,
        waterfall_enabled: bool,
        aid: str = "",
    ) -> FrontierDecision | None:
        """Return the regulation move toward the t_k frontier, or ``None`` to
        hold (inside band, waterfall-deferred, or step below commit threshold).
        Mutates step-direction state only when a move is committed.
        """
        target = lo + self.MARGIN_K
        too_cold = t < target - self.DEADBAND_K
        # Only restore loads WE shed for temperature (still hold a curtail-lock).
        # An L2 priority shed sets no lock; restoring on warmth alone would claw
        # back the priority cascade.
        can_restore = t > target + self.RESTORE_BAND_K and cur < 1.0 and has_lock
        if not too_cold:
            self._reset_defer()
        if not (too_cold or can_restore):
            return None  # inside the hold band

        # |d(t_k)/d(reg)| = sensitivity * cap; floor away from 0 for a finite
        # step (clamp bounds it regardless).
        dtk_dreg_mag = max(sensitivity * cap, 1e-6)
        delta_t = target - t  # >0 want warmer (shed); <0 want cooler (restore)
        delta_reg = -self.GAIN * delta_t / dtk_dreg_mag
        delta_reg = max(-self.MAX_STEP, min(self.MAX_STEP, delta_reg))
        # Anti-limit-cycle: halve a step that reverses the previous one.
        if self._last_dir != 0.0 and delta_reg * self._last_dir < 0.0:
            delta_reg *= 0.5
        new_reg = max(0.0, min(1.0, cur + delta_reg))
        needed_mw = abs(delta_reg) * cap if too_cold else 0.0

        # Waterfall gate (shed only): hold the own shed while lower-priority
        # same-component reducible draw can COVER this step (sufficiency) and
        # deferring is still making the node warmer (budget) — and surface the
        # defer so the monitor actively requests those peers to shed.
        if too_cold and waterfall_enabled and needed_mw > 0.0:
            if self._defer_allowed(t, now, my_tier, needed_mw):
                logger.debug(
                    "[%s] heat frontier: defer shed (t_k=%.1f, tier=%s) — "
                    "sufficient lower-priority reducible load in region",
                    aid,
                    t,
                    my_tier,
                )
                return FrontierDecision(cur, "defer_waterfall", needed_mw)

        if abs(new_reg - cur) < 1e-3:
            return None
        self._last_dir = 1.0 if delta_reg > 0.0 else -1.0
        reason = "curtail" if new_reg < cur else "heat_recovery"
        return FrontierDecision(
            new_reg, reason, needed_mw if reason == "curtail" else 0.0
        )
