"""Heat-sector frontier feedback controller.

Extracted from :class:`scare.service.constraints.GridConstraintMonitor`. Owns
the heat priority-waterfall peer cache and the frontier step state, and decides
the regulation move that drives a heat load's junction temperature to the
feasibility floor (max feasible service) rather than bang-bang to zero.

The controller is a plain object the role composes: it holds state and pure
arithmetic, so it is unit-testable without a mango context. The role keeps the
periodic-task scheduling, observation, and ``apply_regulate`` side-effects.
(Promoting it to a standalone mango Role is a clean follow-up once its shared
inputs — local sensitivity and the constraint-state feed — are re-plumbed.)
"""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class FrontierDecision(NamedTuple):
    new_reg: float
    reason: str  # "curtail" (shed) | "heat_recovery" (restore)


class HeatFrontierController:
    # Hold t_k a small margin above the hard floor (served just inside the
    # feasibility frontier).
    MARGIN_K: float = 3.0
    # Below ``target - DEADBAND`` -> shed; above ``target + RESTORE_BAND`` ->
    # restore. The wide, asymmetric restore band is hysteresis against the
    # restore<->re-violate limit cycle for nodes that re-cool when served.
    DEADBAND_K: float = 2.0
    RESTORE_BAND_K: float = 6.0
    # Proportional gain and per-poll step clamp. The clamp bounds the move even
    # on a stale sensitivity estimate; the proportional term settles at the
    # frontier as dT/dP learns.
    GAIN: float = 0.5
    MAX_STEP: float = 0.15
    # Priority-waterfall gate: a same-region peer of strictly lower priority is
    # "still reducible" while its draw exceeds this eps (MW).
    WATERFALL_REDUCIBLE_EPS: float = 1e-4

    def __init__(self, *, peer_freshness_s: float) -> None:
        self._peer_freshness_s = peer_freshness_s
        # origin_addr_str -> (t_received, priority_tier, reducible). Filled from
        # heat ``t_k`` constraint-state messages; the deferral gate reads it
        # with a freshness window so a peer that has since shed ages out.
        self._peer_state: dict[str, tuple[float, int, float]] = {}
        # Sign of the last committed regulation step; halving on a direction
        # reversal damps the frontier limit cycle.
        self._last_dir: float = 0.0

    def note_peer_state(
        self, origin: str, t_received: float, tier: int, reducible: float,
    ) -> None:
        """Cache a peer heat load's (tier, reducible), stamped for freshness."""
        self._peer_state[origin] = (t_received, tier, reducible)

    def region_has_lower_priority_reducible(self, my_tier: int, now: float) -> float:
        """Total reducible heat draw of fresh same-region peers at strictly
        lower priority (higher tier number) than ``my_tier``. While non-zero,
        the deferral gate holds this load's shed so lower-priority loads absorb
        first."""
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
        hold (inside the band, deferred by the priority waterfall, or step below
        the commit threshold). Mutates the internal step-direction state only
        when a move is committed.
        """
        target = lo + self.MARGIN_K
        too_cold = t < target - self.DEADBAND_K
        # Only restore loads WE shed for temperature (still holding a
        # curtail-lock). An L2 priority shed sets no lock; restoring on a warm
        # reading alone would claw back the priority cascade.
        can_restore = (
            t > target + self.RESTORE_BAND_K and cur < 1.0 and has_lock
        )
        if not (too_cold or can_restore):
            return None  # inside the hold band

        # Priority-waterfall gate (shed only): defer while a strictly
        # lower-priority same-region heat load still has reducible draw.
        if too_cold and waterfall_enabled:
            if self.region_has_lower_priority_reducible(my_tier, now) > 0.0:
                logger.debug(
                    "[%s] heat frontier: defer shed (t_k=%.1f, tier=%s) — "
                    "lower-priority reducible load remains in region",
                    aid, t, my_tier,
                )
                return None

        # d(t_k)/d(reg) is NEGATIVE (more extraction -> colder); the EMA stores
        # the magnitude |dt_k/dP|, dP/dreg = cap. Floor away from 0 for a finite
        # step (the clamp below bounds it regardless).
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
