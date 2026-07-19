"""Multi-hop constraint-state propagation for the grid monitor.

Owns the neighbour-state cache, the forward-dedup ledger and the trust ledger
that weights (and gates) forwarding.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors

from scare.base.model import (
    ConstraintStateMessage,
    Sector,
)
from scare.base.util import (
    obs_priority,
    obs_setpoint,
)
from scare.service.control.constraint_tuning import (
    _FORWARD_FRESHNESS_S,
    _FORWARD_VALUE_TOL,
)

if TYPE_CHECKING:
    from scare.service.control.constraints import GridConstraintMonitor

logger = logging.getLogger(__name__)


class StatePropagator:
    """Broadcasts this agent's constraint state to its group and folds in what
    neighbours report, deduplicated by hop count and freshness. Reads sector,
    behavior and config through its owning role.
    """

    def __init__(self, role: GridConstraintMonitor, trust: Any) -> None:
        self._role = role
        # (origin_addr_str, variable) -> ConstraintStateMessage
        self._neighbour_state: dict[tuple[str, str], ConstraintStateMessage] = {}
        # Dedup of forwarded state: (origin, variable) -> (best_hops, t, util).
        # Forward incoming only on better hops; re-broadcast own only on value
        # change / freshness. Never cleared per-cycle (would re-flood the group).
        self._state_forwarded: dict[tuple[str, str], tuple[int, float, float]] = {}
        # Per-variable (t, util) of this agent's last own broadcast.
        self._last_local_broadcast: dict[str, tuple[float, float]] = {}
        # B.1: coupling weights K_ij for the propagation overlay (independent of
        # the balance ledger). Weight worst-neighbour util by trust, skip
        # forwarding to neighbours below the liveness threshold.
        self._trust = trust

    async def _propagate_state(
        self,
        variable: str,
        value: float,
        utilization: float,
        obs: dict | None = None,
    ) -> None:
        # Suppress re-broadcasts of an unchanged value unless freshness elapsed
        # or utilization moved beyond ``_FORWARD_VALUE_TOL``.
        now = self._role.context.current_timestamp
        prev = self._last_local_broadcast.get(variable)
        if prev is not None:
            prev_t, prev_util = prev
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (stale or changed):
                return

        # Heat t_k broadcasts carry (tier, reducible) so cold neighbours can
        # run the priority-waterfall gate; only set for a curtailable heat load.
        prio_tier: int | None = None
        reducible: float | None = None
        if (
            self._role.sector == Sector.HEAT
            and variable == "t_k"
            and obs is not None
            and self._role.behavior.has_action(self._role.context.aid, "regulate")
        ):
            prio_tier = max(
                1,
                obs_priority(
                    obs, behavior=self._role.behavior, aid=self._role.context.aid
                ),
            )
            reducible = abs(
                obs_setpoint(
                    obs, behavior=self._role.behavior, aid=self._role.context.aid
                )
            )

        origin = self._role.context.addr
        msg = ConstraintStateMessage(
            sector=self._role.sector,
            variable=variable,
            value=value,
            utilization=utilization,
            hops_remaining=self._role.max_hops,
            origin_addr=origin,
            priority_tier=prio_tier,
            reducible=reducible,
            component_id=self._role._heat_component_id
            if prio_tier is not None
            else None,
        )
        origin_key = (str(origin), variable)
        self._state_forwarded[origin_key] = (self._role.max_hops, now, utilization)
        self._last_local_broadcast[variable] = (now, utilization)

        for addr in topology_neighbors(self._role, tid="groups"):
            await self._role.context.send_message(msg, receiver_addr=addr)

    async def _handle_constraint_state(
        self, message: ConstraintStateMessage, meta: dict
    ) -> None:
        origin_key = (str(message.origin_addr), message.variable)

        # B.1: nudge the K-score of the arriving link.
        sender = mango_sender_addr(meta)
        now = self._role.context.current_timestamp
        if sender is not None:
            self._trust.on_message_received(str(sender), now)

        self._neighbour_state[origin_key] = message

        # Heat priority-waterfall: cache the origin's (tier, reducible).
        if message.priority_tier is not None and message.reducible is not None:
            self._role._heat_frontier.note_peer_state(
                str(message.origin_addr),
                now,
                message.priority_tier,
                message.reducible,
                component_id=getattr(message, "component_id", None),
            )

        # Dedup: forward only if the incoming copy improves on the last
        # forwarded one — larger ``hops_remaining``, freshness elapsed, or
        # value moved beyond tolerance.
        prev = self._state_forwarded.get(origin_key)
        if prev is not None:
            prev_hops, prev_t, prev_util = prev
            improves_hops = message.hops_remaining > prev_hops
            stale = (now - prev_t) >= _FORWARD_FRESHNESS_S
            changed = abs(message.utilization - prev_util) >= _FORWARD_VALUE_TOL
            if not (improves_hops or stale or changed):
                return
        self._state_forwarded[origin_key] = (
            message.hops_remaining,
            now,
            message.utilization,
        )

        if message.hops_remaining <= 1:
            return  # TTL exhausted

        # ``enable_multihop_constraint=False`` also disables forwarding (needed
        # for ``component_level``, where one group fans out N·(N−1) per hop).
        # Cache + trust updates still fire; only redistribution stops.
        if not self._role.enable_multihop_constraint:
            return

        fwd = ConstraintStateMessage(
            sector=message.sector,
            variable=message.variable,
            value=message.value,
            utilization=message.utilization,
            hops_remaining=message.hops_remaining - 1,
            origin_addr=message.origin_addr,
            priority_tier=message.priority_tier,
            reducible=message.reducible,
        )
        for addr in topology_neighbors(self._role, tid="groups"):
            # Don't send back to origin or immediate sender.
            if addr == message.origin_addr or addr == sender:
                continue
            # B.1: skip neighbours below the liveness gate.
            if not self._trust.is_live(str(addr), now):
                continue
            await self._role.context.send_message(fwd, receiver_addr=addr)

    def worst_neighbour_utilization(self) -> float:
        """Worst neighbour utilization in multi-hop range, weighted by the
        link's coupling weight K_ij (B.1) so low-trust links count less."""
        if not self._neighbour_state:
            return 0.0
        now = self._role.context.current_timestamp
        worst = 0.0
        for (origin_str, _var), msg in self._neighbour_state.items():
            k = self._trust.score(origin_str, now)
            weighted = k * msg.utilization
            if weighted > worst:
                worst = weighted
        return worst
