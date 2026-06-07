"""Heat-sector frontier feedback controller.

Owns the heat priority-waterfall peer cache and frontier step state, and decides
the regulation move that drives a heat load's junction temperature to the
feasibility floor (max feasible service) rather than bang-bang to zero. A plain
object (pure arithmetic + state), unit-testable without a mango context.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class FrontierDecision(NamedTuple):
    new_reg: float
    reason: str  # "curtail" (shed) | "heat_recovery" (restore)


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
    # Waterfall gate: a same-region lower-priority peer is "still reducible"
    # while its draw exceeds this eps (MW).
    WATERFALL_REDUCIBLE_EPS: float = 1e-4

    def __init__(self, *, peer_freshness_s: float) -> None:
        self._peer_freshness_s = peer_freshness_s
        # origin -> (t_received, priority_tier, reducible) from heat ``t_k``
        # constraint-state messages; read with a freshness window.
        self._peer_state: dict[str, tuple[float, int, float]] = {}
        # Sign of the last committed step; halving on reversal damps the
        # frontier limit cycle.
        self._last_dir: float = 0.0

    def note_peer_state(
        self,
        origin: str,
        t_received: float,
        tier: int,
        reducible: float,
    ) -> None:
        """Cache a peer heat load's (tier, reducible), stamped for freshness."""
        self._peer_state[origin] = (t_received, tier, reducible)

    def region_has_lower_priority_reducible(self, my_tier: int, now: float) -> float:
        """Total reducible draw of fresh same-region peers at strictly lower
        priority than ``my_tier``. While non-zero, this load's shed is held so
        lower-priority loads absorb first."""
        total = 0.0
        for _origin, (t_rx, tier, reducible) in self._peer_state.items():
            if now - t_rx > self._peer_freshness_s:
                continue
            if tier > my_tier and reducible > self.WATERFALL_REDUCIBLE_EPS:
                total += reducible
        return total

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
        if not (too_cold or can_restore):
            return None  # inside the hold band

        # Waterfall gate (shed only): defer while a lower-priority same-region
        # load still has reducible draw.
        if too_cold and waterfall_enabled:
            if self.region_has_lower_priority_reducible(my_tier, now) > 0.0:
                logger.debug(
                    "[%s] heat frontier: defer shed (t_k=%.1f, tier=%s) — "
                    "lower-priority reducible load remains in region",
                    aid,
                    t,
                    my_tier,
                )
                return None

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
        if abs(new_reg - cur) < 1e-3:
            return None
        self._last_dir = 1.0 if delta_reg > 0.0 else -1.0
        reason = "curtail" if new_reg < cur else "heat_recovery"
        return FrontierDecision(new_reg, reason)
