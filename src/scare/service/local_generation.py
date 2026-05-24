"""Local-generation fallback for unresolved negotiation deficits.

Implements the SHOULD-level requirement from improvements.txt §5:
"Fallback / islanding capability — if an agent cannot find a feasible
restoration through negotiation with neighbors, it should be able to
form a local island using local DGs or storage."

Despite the §5 wording, this role does *not* perform real electrical
islanding (no switch is opened, no f/V control, no supply≡demand
match).  It is a dispatch heuristic: when gossip converges with a
significant residual deficit, the role mops up what gossip could not
redistribute by ramping local generator-class children up to their
remaining headroom.  Genuine physical islanding (grid-forming, sub-
island anchoring) lives in monee's ``enable_islanding`` extension and
the ``GridReconfigurator`` / ``GridTieSwitchOperator`` roles.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_neighbors

from scare.base.diagnostics import record_event
from scare.base.model import (
    SECTOR_TIMESCALE,
    LocalGenerationApproval,
    Sector,
)
from scare.base.util import (
    apply_regulate,
    lookup_priority,
    obs_capacity,
    obs_priority,
    obs_sector,
    obs_setpoint,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class LocalGenerationFallbackRole(Role):
    """Activates local DGs to cover residual load deficit when gossip
    negotiation fails to fully resolve the imbalance.

    The role listens for ``LocalGenerationApproval`` events/messages.
    The approval is dispatched by L2's ``HolonicCommunityRole`` after
    the holon has had a chance to absorb the residual cross-group, or
    emitted locally by the L1 negotiator when no holon layer is wired
    (non-holonic config).  On receipt the role:

    1. Scans group members for generators (priority == 0) with
       available headroom.
    2. Distributes the residual deficit across available generators
       proportional to their capacity.
    3. Sets each generator's regulation factor to cover its share.

    This is a last-resort mechanism — it activates only after L1
    gossip and the L2 absorption attempt have both been exhausted.
    It only adjusts setpoints; it never opens a switch or otherwise
    alters physical topology.
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
        # Local-event subscription handles the non-holonic config where
        # the L1 negotiator emits the approval directly because no
        # holon peer exists to mediate.
        self.context.subscribe_event(
            self, LocalGenerationApproval, self._on_local_gen_approval_event
        )
        # Message subscription handles the L2-mediated path: a holon
        # peer (lex-smallest co-recipient) returns the approval after
        # triggering its own rebalance attempt.
        self.context.subscribe_message(
            self,
            self._on_local_gen_approval_message,
            lambda msg, meta: isinstance(msg, LocalGenerationApproval),
        )

    def _on_local_gen_approval_event(
        self, event: LocalGenerationApproval, _src: Any
    ) -> None:
        if event.sector != self.sector:
            return
        self.context.schedule_instant_task(
            self._activate_local_gen(event.residual_deficit)
        )

    def _on_local_gen_approval_message(
        self, message: LocalGenerationApproval, _meta: Any
    ) -> None:
        if message.sector != self.sector:
            return
        self.context.schedule_instant_task(
            self._activate_local_gen(message.residual_deficit)
        )

    async def _activate_local_gen(self, deficit: float) -> None:
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
            if obs_priority(obs, behavior=self.behavior, aid=aid) != 0:
                continue  # not a generator
            cap = obs_capacity(obs, behavior=self.behavior, aid=aid)
            sp = obs_setpoint(obs, behavior=self.behavior, aid=aid)
            headroom = abs(cap) - abs(sp)
            if headroom > 1e-6:
                generators.append((aid, abs(cap), headroom))

        if not generators:
            logger.warning(
                "[%s] local-gen fallback: no local DGs available for sector %s",
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
            current_sp = abs(obs_setpoint(obs, behavior=self.behavior, aid=aid))
            new_factor = min(1.0, (current_sp + share) / cap) if cap > 0 else 0.0

            applied = apply_regulate(
                self.behavior,
                aid,
                new_factor,
                sector=self.sector.value,
                reason="local_gen_fallback",
                timestamp=self.context.current_timestamp,
                priority_tier=lookup_priority(self.behavior, aid),
            )
            if applied:
                covered += share
                logger.info(
                    "[%s] local-gen fallback: DG %s ramped to %.1f%% (share=%.4f)",
                    self.context.aid,
                    aid,
                    new_factor * 100,
                    share,
                )

        logger.info(
            "[%s] local-gen fallback: covered %.4f of %.4f deficit in sector %s",
            self.context.aid,
            covered,
            deficit,
            self.sector.value,
        )
        record_event(
            t=self.context.current_timestamp,
            kind="local_gen_covered",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=f"covered={covered:.4f} of deficit={deficit:.4f}",
        )
