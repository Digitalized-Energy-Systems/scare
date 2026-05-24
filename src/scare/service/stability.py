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

    Subscribes to ``NegotiationFinishedEvent`` (emitted from the same
    agent's own ``_handle_negotiation_finished_msg``) and converts the
    new per-agent setpoint into a clamped factor.  Honours the CLPU
    ramp + restoration monotonic floor from the active scenario config
    so this side-channel doesn't bypass the safety nets the gossip
    layer applies on its own ``_apply_setpoint`` path.
    """

    def __init__(
        self, behavior: RestorationEnvironmentBehavior, sector: Sector
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        # Track the last applied factor + timestamp so the CLPU ramp can
        # bound increases at ``convergence_rate`` per sim-second, matching
        # the rate the gossip layer applies inside ``_apply_setpoint``.
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
        # Both load (cap>0, sp∈[0,cap]) and generator (cap<0, sp∈[cap,0])
        # produce ``new_setpoint / cap ∈ [0,1]`` when the gossip honoured
        # its own box constraints.  Keep the abs() as a defensive clamp
        # against a degenerate ``new_setpoint`` of opposite sign — but
        # log it so the bug shows up instead of being silently swallowed.
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

        # Read the active scenario config (stashed on behavior by the
        # scenario builder).  Honour the same safety nets the gossip
        # layer applies in ``_apply_setpoint`` so the stability path
        # can't bypass them — without this, child-418 (heat) was seen
        # jumping from 1.0 → 0.0 in one tick via the stability reason.
        cfg = getattr(self.behavior, "_scare_config", None)
        enable_floor = getattr(cfg, "enable_monotonic_floor", True)
        enable_ramp = getattr(cfg, "enable_clpu_ramp", True)

        now = self.context.current_timestamp
        # CLPU ramp: bound ramp-up to ``convergence_rate`` per sim-second.
        # Decreases pass through immediately — shedding under stress
        # cannot wait for the ramp budget.
        if (
            enable_ramp
            and self._last_factor is not None
            and self._last_t is not None
            and factor > self._last_factor
        ):
            rate = SECTOR_TIMESCALE.get(self.sector, {}).get("convergence_rate", 0.6)
            dt = max(0.0, now - self._last_t)
            factor = min(factor, self._last_factor + rate * dt)

        # Monotonic restoration floor for loads: once restored to a given
        # factor, the stability path may not regress unless a constraint
        # violation is actively pulling it back.  Generators (cap < 0)
        # are exempt — they ramp both directions to balance.
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
