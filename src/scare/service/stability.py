from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role

from scare.base.model import NegotiationFinishedEvent, Sector
from scare.base.util import apply_regulate, obs_capacity

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
        obs = self.behavior.observe(self.context.aid)
        if obs:
            logger.debug("[%s] obs=%s", self.context.aid, obs)


class GenerationController(Role):
    def __init__(
        self, behavior: RestorationEnvironmentBehavior, sector: Sector
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector

    def setup(self) -> None:
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_negotiation_finished
        )

    def _on_negotiation_finished(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        if event.sector != self.sector:
            return
        obs = self.behavior.observe(self.context.aid)
        if not obs:
            return
        cap = obs_capacity(obs)
        if cap == 0.0:
            return
        factor = max(0.0, min(1.0, abs(event.new_setpoint / cap)))
        apply_regulate(
            self.behavior,
            self.context.aid,
            factor,
            sector=self.sector.value,
            reason="stability",
            timestamp=self.context.current_timestamp,
        )
