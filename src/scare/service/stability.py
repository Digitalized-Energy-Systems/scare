from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role

from scare.base.model import NegotiationFinishedEvent, SECTOR_TIMESCALE, Sector
from scare.base.util import apply_regulate, lookup_priority, obs_capacity

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class NodeObserver(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        *,
        control_until_s: float = 30.0,
        poll_period_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.control_until_s = control_until_s
        self.poll_period_s = poll_period_s

    def setup(self) -> None:
        self.context.schedule_periodic_task(self._observe, delay=self.poll_period_s)

    async def _observe(self) -> None:
        if self.context.current_timestamp > self.control_until_s:
            return
        try:
            obs = self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return
        if obs:
            logger.debug("[%s] obs=%s", self.context.aid, obs)


class GenerationController(Role):
    """Apply the local agent's gossip-decided setpoint as a regulate action.

    Subscribes to the agent's own ``NegotiationFinishedEvent`` and converts
    the new setpoint into a clamped factor.  Honours the CLPU ramp and
    restoration monotonic floor from the scenario config so this path does
    not bypass the safety nets gossip applies in ``_apply_setpoint``.
    """

    def __init__(
        self, behavior: RestorationEnvironmentBehavior, sector: Sector
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        # Last applied factor + timestamp; lets the CLPU ramp bound
        # increases at ``convergence_rate`` per sim-second.
        self._last_factor: float | None = None
        self._last_t: float | None = None
        self._floor: float = 0.0

    def setup(self) -> None:
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_negotiation_finished
        )

    def _on_negotiation_finished(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        if event.sector != self.sector:
            return
        try:
            obs = self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return
        if not obs:
            return
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap == 0.0:
            return
        # Both load (cap>0, sp in [0,cap]) and generator (cap<0, sp in
        # [cap,0]) give ``new_setpoint / cap`` in [0,1] when gossip honoured
        # its box constraints.  abs() defensively clamps an opposite-sign
        # setpoint, but log it so the bug surfaces rather than being hidden.
        raw_factor = event.new_setpoint / cap
        if (cap > 0 and event.new_setpoint < -1e-9) or (
            cap < 0 and event.new_setpoint > 1e-9
        ):
            logger.warning(
                "[%s] stability: new_setpoint sign disagrees with cap "
                "(sp=%.4g, cap=%.4g) — clamping with abs()",
                self.context.aid,
                event.new_setpoint,
                cap,
            )
            raw_factor = abs(raw_factor)
        factor = max(0.0, min(1.0, raw_factor))

        # Honour the same safety nets gossip applies in ``_apply_setpoint``
        # so the stability path can't jump a factor in one tick.
        cfg = getattr(self.behavior, "_scare_config", None)
        enable_floor = getattr(cfg, "enable_monotonic_floor", True)
        enable_ramp = getattr(cfg, "enable_clpu_ramp", True)

        now = self.context.current_timestamp
        # CLPU ramp: bound ramp-up to ``convergence_rate`` per sim-second.
        # Decreases pass through immediately — shedding can't wait.
        if (
            enable_ramp
            and self._last_factor is not None
            and self._last_t is not None
            and factor > self._last_factor
        ):
            rate = SECTOR_TIMESCALE.get(self.sector, {}).get("convergence_rate", 0.6)
            dt = max(0.0, now - self._last_t)
            factor = min(factor, self._last_factor + rate * dt)

        # Monotonic restoration floor for loads: once restored to a factor,
        # the stability path may not regress below it.  Generators (cap < 0)
        # are exempt — they ramp both ways to balance.
        if enable_floor and cap > 0 and factor < self._floor:
            factor = self._floor
        if cap > 0:
            self._floor = max(self._floor, factor)

        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            factor,
            sector=self.sector.value,
            reason="stability",
            timestamp=now,
            priority_tier=lookup_priority(self.behavior, self.context.aid),
        )
        if applied:
            self._last_factor = factor
            self._last_t = now
