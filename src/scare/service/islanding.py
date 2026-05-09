"""Islanding fallback for unresolved negotiation deficits.

Implements the SHOULD-level requirement from improvements.txt §5:
"Fallback / islanding capability — if an agent cannot find a feasible
restoration through negotiation with neighbors, it should be able to
form a local island using local DGs or storage."

When the gossip negotiation converges but a significant residual deficit
remains, an ``IslandingRequest`` event is emitted.  This role reacts by
identifying local generators (DGs/storage) within the group and matching
their output to the local load demand, creating a soft island where local
generation follows local consumption.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_neighbors

from scare.base.model import (
    SECTOR_TIMESCALE,
    IslandingRequest,
    Sector,
)
from scare.base.util import obs_capacity, obs_priority, obs_sector, obs_setpoint

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class IslandingFallbackRole(Role):
    """Activates local DGs to cover residual load deficit when gossip
    negotiation fails to fully resolve the imbalance.

    The role listens for ``IslandingRequest`` events (emitted by the
    ``EnergyBalanceNegotiator`` when negotiation converges with a
    significant residual) and responds by:

    1. Scanning group members for generators (priority == 0) with
       available headroom.
    2. Distributing the residual deficit across available generators
       proportional to their capacity.
    3. Setting each generator's regulation factor to cover its share.

    This is a last-resort mechanism — it only activates after normal
    negotiation has been exhausted.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector

    def setup(self) -> None:
        self.context.subscribe_event(
            self, IslandingRequest, self._on_islanding_request
        )

    def _on_islanding_request(
        self, event: IslandingRequest, _src: Any
    ) -> None:
        if event.sector != self.sector:
            return
        self.context.schedule_instant_task(
            self._activate_islanding(event.residual_deficit)
        )

    async def _activate_islanding(self, deficit: float) -> None:
        """Find local generators and distribute the deficit across them."""
        if deficit <= 0:
            return

        neighbours = topology_neighbors(self, tid="groups")
        member_aids = [self.context.aid] + [a.aid for a in neighbours]

        # Identify generators with available headroom
        generators: list[tuple[str, float, float]] = []  # (aid, capacity, headroom)
        for aid in member_aids:
            obs = self.behavior.observe(aid) or {}
            if obs_sector(obs, behavior=self.behavior, aid=aid) != self.sector:
                continue
            if obs_priority(obs) != 0:
                continue  # not a generator
            cap = obs_capacity(obs)
            sp = obs_setpoint(obs)
            headroom = abs(cap) - abs(sp)
            if headroom > 1e-6:
                generators.append((aid, abs(cap), headroom))

        if not generators:
            logger.warning(
                "[%s] islanding: no local DGs available for sector %s",
                self.context.aid,
                self.sector.value,
            )
            return

        total_headroom = sum(h for _, _, h in generators)
        if total_headroom < 1e-6:
            return

        covered = 0.0
        for aid, cap, headroom in generators:
            # Distribute deficit proportional to each generator's headroom
            share = min(headroom, deficit * (headroom / total_headroom))
            obs = self.behavior.observe(aid) or {}
            current_sp = abs(obs_setpoint(obs))
            new_factor = min(1.0, (current_sp + share) / cap) if cap > 0 else 0.0

            from scare.base.util import apply_regulate

            applied = apply_regulate(
                self.behavior,
                aid,
                new_factor,
                sector=self.sector.value,
                reason="islanding",
                timestamp=self.context.current_timestamp,
            )
            if applied:
                covered += share
                logger.info(
                    "[%s] islanding: DG %s ramped to %.1f%% (share=%.4f)",
                    self.context.aid,
                    aid,
                    new_factor * 100,
                    share,
                )

        logger.info(
            "[%s] islanding: covered %.4f of %.4f deficit in sector %s",
            self.context.aid,
            covered,
            deficit,
            self.sector.value,
        )
        from scare.base.diagnostics import record_event

        record_event(
            t=self.context.current_timestamp,
            kind="islanding_covered",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=f"covered={covered:.4f} of deficit={deficit:.4f}",
        )
