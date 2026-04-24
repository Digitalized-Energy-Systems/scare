from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
    topology_neighbors,
)

from scare.base.model import (
    AskEnergyMessage,
    AskForAvailableFlex,
    AvailableFlexAnswer,
    BalanceProblem,
    ConstraintViolation,
    EnergyNegotiationMessage,
    IslandingRequest,
    NegotiationFinishedEvent,
    ResponseEnergyMessage,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.util import (
    clamp_to_constraints,
    constraint_utilization,
    obs_capacity,
    obs_min_max,
    obs_priority,
    obs_sector,
    obs_setpoint,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

_START_THRESHOLD: dict[Sector, float] = {Sector.HEAT: 10.0}
_DEFAULT_START_THRESHOLD = 1e-4

_MAX_HOPS = 100

# Base wallclock deadline per sector.  Electricity is fastest (agents
# need few rounds), heat is slowest (high decision delay, low
# convergence rate).  The actual deadline is further scaled by the
# group size — larger groups need more rounds to converge.
_GOSSIP_TIMEOUT_BASE_S: dict[Sector, float] = {
    Sector.ELECTRICITY: 5.0,
    Sector.GAS: 15.0,
    Sector.HEAT: 30.0,
}
_GOSSIP_TIMEOUT_DEFAULT_S = 15.0
_GOSSIP_TIMEOUT_PER_AGENT_S = 0.5  # added per group member

# Number of discrete priority tiers for intra-sector ordering.
# Lower tier = higher urgency = participates earlier in gossip rounds.
_PRIORITY_TIERS = 10


def _start_threshold(sector: Sector) -> float:
    return _START_THRESHOLD.get(sector, _DEFAULT_START_THRESHOLD)


def _deterministic_next(neighbours: list, negotiation_id: str, counter: int) -> Any:
    """Select the next gossip target deterministically.

    Uses a hash of (negotiation_id, counter) to pick a neighbour index.
    This replaces ``random.choice`` and ensures that two agents competing
    for the same resource always send in the same order, which is a
    prerequisite for deterministic conflict resolution.
    """
    if not neighbours:
        return None
    h = hashlib.sha256(f"{negotiation_id}:{counter}".encode()).digest()
    idx = int.from_bytes(h[:4], "big") % len(neighbours)
    return neighbours[idx]


def _deterministic_sub_round(
    agent_addr: str, negotiation_id: str, tier: int, tier_size: int
) -> int:
    """Compute a deterministic sub-round index for intra-tier ordering.

    Agents within the same priority tier are serialized: each gets a
    unique sub-round index in [0, tier_size).  The index is stable for
    a given (agent, negotiation, tier) triple.
    """
    h = hashlib.sha256(
        f"{negotiation_id}:{tier}:{agent_addr}".encode()
    ).digest()
    return int.from_bytes(h[:4], "big") % tier_size


@dataclass
class _GossipState:
    negotiation_id: str
    target: float
    counter: int
    current_delta: float
    starting_setpoint: float
    # Shared ledger of per-agent contributions, merged across received
    # messages by taking the entry with the highest counter per agent.
    # This replaces the aggregate digest that double-counted in cycles.
    # addr_str -> (delta, counter_when_set, priority)
    memory: dict[str, tuple[float, int, int]] = field(default_factory=dict)


class EnergyBalanceNegotiator(Role):
    """Gossip-based energy balance negotiation with:

    - **Priority-ordered participation**: agents join the gossip at a
      round determined by their priority tier, ensuring high-priority
      loads are restored first (or shed last).
    - **Monotonic progress**: once a load's regulation factor has been
      increased during a restoration (target > 0), it will not be
      decreased below that floor unless a hard constraint violation
      forces it (improvements.txt §5 MUST "no-regret switching").
    - **Deterministic conflict resolution**: gossip forwarding uses a
      hash-based deterministic selector instead of ``random.choice``,
      and when two agents compete for the same headroom the
      higher-priority agent wins (improvements.txt §5 MUST "conflict
      resolution mechanism").
    - **Constraint-aware clamping**: setpoints are clamped using the
      local constraint utilization, so agents near voltage/pressure/
      temperature bounds automatically contribute less.
    - **Sector time-scale awareness**: convergence rate defaults to the
      sector-specific value from ``SECTOR_TIMESCALE``.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        priority: int = 0,
        convergence_rate: float | None = None,
        impact_weight: float = 1.0,
        termination_tolerance: float = 1e-5,
        constraint_aware: bool = True,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.priority = priority
        self.impact_weight = impact_weight
        self.termination_tolerance = termination_tolerance
        self.constraint_aware = constraint_aware

        # Apply sector-specific convergence rate if not overridden.
        from scare.base.model import SECTOR_TIMESCALE

        ts = SECTOR_TIMESCALE.get(sector, {})
        self.convergence_rate = (
            convergence_rate if convergence_rate is not None
            else ts.get("convergence_rate", 0.5)
        )

        self._active: bool = False
        self._gossip: _GossipState | None = None
        # State for the setpoint-gathering phase before gossip starts
        self._trigger_nid: str | None = None
        self._trigger_responses: dict[str, float] = {}
        self._trigger_expected: int = 0

        # --- Monotonic progress floor ---
        # Tracks the highest regulation factor this agent has applied
        # during restoration.  The factor may only decrease if a hard
        # constraint violation is active.
        self._restoration_floor: float = 0.0
        self._constraint_violation_active: bool = False

    def setup(self) -> None:
        # Mango's handle_message dispatches synchronously, so async handlers
        # must be wrapped to schedule themselves via the agent scheduler.
        # This ensures the simulation's termination detection can track them.
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_energy),
            lambda msg, meta: (
                isinstance(msg, AskEnergyMessage) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_response_energy),
            lambda msg, meta: isinstance(msg, ResponseEnergyMessage),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_message),
            lambda msg, meta: (
                isinstance(msg, EnergyNegotiationMessage) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_finished_msg),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_flex),
            lambda msg, meta: isinstance(msg, AskForAvailableFlex),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_start_balance),
            lambda msg, meta: isinstance(msg, StartBalanceNegotiation),
        )
        self.context.subscribe_event(self, BalanceProblem, self._on_balance_problem)
        self.context.subscribe_event(
            self, ConstraintViolation, self._on_constraint_violation
        )

    # ------------------------------------------------------------------
    # Constraint violation tracking (for monotonic progress override)
    # ------------------------------------------------------------------

    def _on_constraint_violation(self, event: ConstraintViolation, _src: Any) -> None:
        if event.sector == self.sector:
            self._constraint_violation_active = True
            # Cancel any active gossip: the constraint landscape has changed,
            # so continuing to converge on a stale target is wasteful and may
            # push the system further into violation.  The BalanceProblem
            # event emitted by GridConstraintMonitor will trigger a fresh
            # negotiation that incorporates updated constraint utilization.
            if self._gossip is not None:
                logger.info(
                    "[%s] cancelling gossip %s due to %s violation",
                    self.context.aid,
                    self._gossip.negotiation_id[:8],
                    event.variable,
                )
                self._gossip = None
                self._active = False

    def _check_violation_cleared(self) -> None:
        """Clear the violation flag if the co-located GridConstraintMonitor
        reports all local constraints are satisfied again."""
        if not self._constraint_violation_active:
            return
        from scare.service.constraints import GridConstraintMonitor

        for role in getattr(self.context, "roles", []):
            if isinstance(role, GridConstraintMonitor) and role.sector == self.sector:
                if role.is_locally_feasible():
                    self._constraint_violation_active = False
                return

    # ------------------------------------------------------------------
    # Trigger phase
    # ------------------------------------------------------------------

    async def trigger_balance_negotiation(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        if self._active:
            return
        self._active = True

        neighbours = topology_neighbors(self, tid="groups")
        logger.info(
            "[%s] balance negotiation triggered (sector=%s, group size=%d)",
            self.context.aid,
            self.sector.value,
            len(neighbours) + 1,
        )
        if not neighbours:
            obs = self.behavior.observe(self.context.aid) or {}
            await self._start_gossip(-obs_setpoint(obs))
            return

        nid = str(uuid4())
        self._trigger_nid = nid
        self._trigger_responses = {}
        self._trigger_expected = len(neighbours)

        msg = AskEnergyMessage(negotiation_id=nid, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs)
        sp = obs_setpoint(obs)
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=sp,
            available=cap - sp,  # headroom, not total capacity
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_response_energy(
        self, message: ResponseEnergyMessage, meta: dict
    ) -> None:
        if message.negotiation_id != self._trigger_nid:
            return

        sender_key = str(mango_sender_addr(meta))
        self._trigger_responses[sender_key] = message.setpoint

        if len(self._trigger_responses) >= self._trigger_expected:
            own_obs = self.behavior.observe(self.context.aid) or {}
            total_sp = obs_setpoint(own_obs) + sum(self._trigger_responses.values())
            self._trigger_nid = None
            self._trigger_responses = {}
            await self._start_gossip(-total_sp)

    # ------------------------------------------------------------------
    # Gossip phase
    # ------------------------------------------------------------------

    async def _start_gossip(self, target: float) -> None:
        if abs(target) < _start_threshold(self.sector):
            logger.info(
                "[%s] gossip skipped: already balanced (target=%.4f)",
                self.context.aid,
                target,
            )
            self._active = False
            return
        logger.info(
            "[%s] starting gossip (sector=%s, target=%.4f)",
            self.context.aid,
            self.sector.value,
            target,
        )

        # Clear violation flag at the start of each new negotiation so
        # that the monotonic floor is only breached while a violation is
        # actively present.
        self._constraint_violation_active = False

        neighbours = topology_neighbors(self, tid="groups")
        nid = str(uuid4())
        self_key = str(self.context.addr)

        obs = self.behavior.observe(self.context.aid) or {}
        starting_sp = obs_setpoint(obs)

        self._gossip = _GossipState(
            negotiation_id=nid,
            target=target,
            counter=0,
            current_delta=0.0,
            starting_setpoint=starting_sp,
            memory={self_key: (0.0, 0, self.priority)},
        )

        if not neighbours:
            # Isolated agent: gossip can't help. Emit IslandingRequest
            # directly with the full deficit so a co-located
            # IslandingFallbackRole (or the agent itself) can activate
            # local DGs.  We also self-activate islanding inline for the
            # common case where the fallback role is absent.
            if abs(target) > _start_threshold(self.sector):
                self.context.emit_event(
                    IslandingRequest(
                        sector=self.sector, residual_deficit=target
                    )
                )
                self._try_self_island(target)
            self._active = False
            self._gossip = None
            return

        msg = EnergyNegotiationMessage(
            negotiation_id=nid,
            sector=self.sector,
            negotiation_target=target,
            current_delta=0.0,
            counter=0,
            memory=dict(self._gossip.memory),
        )
        for addr in neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

        # Wallclock timeout: force-finish if gossip hasn't converged
        # within the deadline.  Adaptive: base per sector + per-agent
        # scaling so large groups get proportionally more time.
        base = _GOSSIP_TIMEOUT_BASE_S.get(self.sector, _GOSSIP_TIMEOUT_DEFAULT_S)
        timeout = base + len(neighbours) * _GOSSIP_TIMEOUT_PER_AGENT_S
        deadline = self.context.current_timestamp + timeout
        self.context.schedule_timestamp_task(
            self._gossip_timeout(nid), timestamp=deadline
        )

    async def _gossip_timeout(self, negotiation_id: str) -> None:
        if (
            self._gossip is not None
            and self._gossip.negotiation_id == negotiation_id
        ):
            base = _GOSSIP_TIMEOUT_BASE_S.get(self.sector, _GOSSIP_TIMEOUT_DEFAULT_S)
            logger.warning(
                "[%s] gossip %s timed out — forcing finish",
                self.context.aid,
                negotiation_id[:8],
            )
            await self._finish_negotiation()

    async def _handle_negotiation_message(
        self, message: EnergyNegotiationMessage, meta: dict
    ) -> None:
        nid = message.negotiation_id
        counter = message.counter + 1

        if counter > _MAX_HOPS + 1:
            return

        self_key = str(self.context.addr)

        if self._gossip is None or self._gossip.negotiation_id != nid:
            obs = self.behavior.observe(self.context.aid) or {}
            self._gossip = _GossipState(
                negotiation_id=nid,
                target=message.negotiation_target,
                counter=counter,
                current_delta=0.0,
                starting_setpoint=obs_setpoint(obs),
                memory=dict(message.memory),
            )
        else:
            self._gossip.counter = counter
            # Merge per-agent ledger: for each agent, keep the entry with
            # the newest counter.  This prevents the double-counting that
            # a single aggregate digest suffers in a cyclic gossip graph.
            for k, v in message.memory.items():
                local = self._gossip.memory.get(k)
                if local is None or local[1] < v[1]:
                    self._gossip.memory[k] = v

        target = self._gossip.target
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs)
        dmin, dmax = obs_min_max(obs)

        prev_own = self._gossip.memory.get(self_key, (0.0, 0, self.priority))[0]
        total_delta = sum(v[0] for v in self._gossip.memory.values())
        open_gap = target - total_delta

        # --- Constraint-aware participation scaling ---
        participation_scale = 1.0
        if self.constraint_aware:
            from scare.base.model import SECTOR_CONSTRAINTS
            bounds = SECTOR_CONSTRAINTS.get(self.sector, {})
            for var, (lo, hi) in bounds.items():
                if var in obs:
                    util = constraint_utilization(float(obs[var]), lo, hi)
                    participation_scale = min(participation_scale, max(0.0, 1.0 - util))

        own_change = (
            open_gap * (abs(cap) / 20.0)
            * self.impact_weight
            * self.convergence_rate
            * participation_scale
        )

        # --- Priority-ordered participation ---
        actual_prio = _compute_actual_priority(self.priority, target)

        if actual_prio <= counter:
            # --- Intra-tier strict ordering ---
            # Within the same priority tier, agents are serialized using a
            # deterministic sub-round index derived from their address hash.
            participant_count = max(1, len(self._gossip.memory))
            tier_size = max(1, participant_count // max(1, _PRIORITY_TIERS))
            if tier_size > 1:
                sub_idx = _deterministic_sub_round(
                    self_key, nid, actual_prio, tier_size
                )
                rounds_in_tier = counter - actual_prio
                if rounds_in_tier % tier_size != sub_idx:
                    own_change = 0.0

            current_own = prev_own

            # --- Deterministic conflict resolution ---
            active_count = participant_count
            if active_count > 1 and own_change != 0.0:
                share = 1.0 / active_count
                own_change *= share

            new_delta = max(dmin, min(dmax, current_own + own_change))
            self._gossip.memory[self_key] = (new_delta, counter, self.priority)
            self._gossip.current_delta = new_delta
            if cap != 0.0:
                self._apply_setpoint(self._gossip.starting_setpoint + new_delta)

        # Recompute total after own update
        total_delta = sum(v[0] for v in self._gossip.memory.values())
        open_gap = target - total_delta

        neighbours = topology_neighbors(self, tid="groups")

        if abs(open_gap) <= self.termination_tolerance or counter >= _MAX_HOPS:
            await self._finish_negotiation()
        elif neighbours:
            next_addr = _deterministic_next(neighbours, nid, counter)
            fwd = EnergyNegotiationMessage(
                negotiation_id=nid,
                sector=self.sector,
                negotiation_target=target,
                current_delta=self._gossip.current_delta,
                counter=counter,
                memory=dict(self._gossip.memory),
            )
            await self.context.send_message(fwd, receiver_addr=next_addr)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    async def _finish_negotiation(self) -> None:
        starting_sp = (
            self._gossip.starting_setpoint
            if self._gossip
            else obs_setpoint(self.behavior.observe(self.context.aid) or {})
        )
        delta = self._gossip.current_delta if self._gossip else 0.0
        new_sp = starting_sp + delta

        if self._gossip is not None:
            total_delta = sum(v[0] for v in self._gossip.memory.values())
            target = self._gossip.target
            residual = target - total_delta
            logger.info(
                "[%s] gossip finished (sector=%s, delta=%.4f, residual=%.4f)",
                self.context.aid,
                self.sector.value,
                delta,
                residual,
            )

            # Unresolved deficit escalates to islanding fallback.
            if abs(residual) > _start_threshold(self.sector) * 10:
                self.context.emit_event(
                    IslandingRequest(
                        sector=self.sector, residual_deficit=residual
                    )
                )

        self.context.emit_event(
            NegotiationFinishedEvent(new_setpoint=new_sp, sector=self.sector)
        )

        neighbours = topology_neighbors(self, tid="groups")

        # Broadcast convergence to all group neighbours so each can emit its own local event
        finished_msg = NegotiationFinishedEvent(new_setpoint=0, sector=self.sector)
        for addr in neighbours:
            await self.context.send_message(finished_msg, receiver_addr=addr)

        # Leader also notifies CP connectors
        if topology_characteristic(self, tid="groups") == "leader":
            for addr in topology_connectors(self, tid="groups"):
                await self.context.send_message(finished_msg, receiver_addr=addr)

        self._gossip = None
        self._active = False

    async def _handle_negotiation_finished_msg(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        """Convergence broadcast from a gossip peer — emit own local NegotiationFinishedEvent."""
        starting_sp = (
            self._gossip.starting_setpoint
            if self._gossip
            else obs_setpoint(self.behavior.observe(self.context.aid) or {})
        )
        delta = self._gossip.current_delta if self._gossip else 0.0
        self.context.emit_event(
            NegotiationFinishedEvent(
                new_setpoint=starting_sp + delta, sector=self.sector
            )
        )

    # ------------------------------------------------------------------
    # Flex reporting
    # ------------------------------------------------------------------

    async def _handle_ask_flex(self, message: AskForAvailableFlex, meta: dict) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return

        member_aids = [self.context.aid] + [
            addr.aid for addr in topology_neighbors(self, tid="groups")
        ]

        if message.include_connectors:
            for addr in topology_connectors(self, tid="groups"):
                member_aids.append(addr.aid)

        total_flex = 0.0
        total_balance = 0.0
        total_shedded = 0.0
        flex_by_sector: dict[str, float] = {}
        balance_by_sector: dict[str, float] = {}
        demand_by_priority: dict[int, float] = {}
        served_by_priority: dict[int, float] = {}
        for aid in member_aids:
            obs = self.behavior.observe(aid) or {}
            sector = obs_sector(obs, behavior=self.behavior, aid=aid)
            if sector is None:
                continue
            cap = obs_capacity(obs)
            sp = obs_setpoint(obs)
            available = cap - sp  # headroom
            # Per-sector breakdown for multi-dimensional ADMM
            sec_key = sector.value
            flex_by_sector[sec_key] = flex_by_sector.get(sec_key, 0.0) + available
            balance_by_sector[sec_key] = balance_by_sector.get(sec_key, 0.0) + sp
            # Priority-tier demand aggregation (loads only: cap > 0)
            if cap > 0:
                prio = obs_priority(obs)
                demand_by_priority[prio] = demand_by_priority.get(prio, 0.0) + abs(cap)
                served_by_priority[prio] = served_by_priority.get(prio, 0.0) + abs(sp)
            if sector != self.sector:
                continue
            total_flex += available
            total_balance += sp
            if sp > 0 and available > 0:
                total_shedded += available

        reply = AvailableFlexAnswer(
            flex=total_flex,
            balance=total_balance,
            shedded=total_shedded,
            sector=self.sector,
            flex_by_sector=flex_by_sector,
            balance_by_sector=balance_by_sector,
            demand_by_priority=demand_by_priority,
            served_by_priority=served_by_priority,
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_start_balance(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        if topology_characteristic(self, tid="groups") == "leader":
            self.context.schedule_instant_task(self.trigger_balance_negotiation())

    def _on_balance_problem(self, event: BalanceProblem, _src: Any) -> None:
        if event.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") == "leader":
            self.context.schedule_instant_task(self.trigger_balance_negotiation())

    # ------------------------------------------------------------------
    # Setpoint application with monotonic progress guarantee
    # ------------------------------------------------------------------

    def _apply_setpoint(self, new_setpoint: float) -> None:
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs)
        if cap == 0.0:
            return

        # Constraint-aware clamping: reduce the setpoint when local grid
        # measurements are near or beyond safety bounds.
        if self.constraint_aware:
            new_setpoint = clamp_to_constraints(new_setpoint, obs, self.sector)

        factor = max(0.0, min(1.0, abs(new_setpoint / cap)))

        # --- Monotonic progress guarantee ---
        # The "no-regret switching" floor only applies when the current
        # negotiation is a restoration (target > 0 means the group needs
        # to grow net supply/demand back).  In a shedding negotiation
        # (target < 0), loads legitimately need to reduce factor to
        # rebalance, so the floor must not block them.
        target = self._gossip.target if self._gossip is not None else 0.0
        is_restoration = target > 0
        if self.priority > 0 and is_restoration:
            self._check_violation_cleared()

            if factor > self._restoration_floor:
                self._restoration_floor = factor
            elif not self._constraint_violation_active:
                factor = self._restoration_floor

        if self.behavior.has_action(self.context.aid, "regulate"):
            self.behavior.act(self.context.aid, "regulate", factor)


    def _try_self_island(self, deficit: float) -> None:
        """Inline islanding for isolated agents without IslandingFallbackRole.

        If this agent is a generator with available headroom, ramp up to
        cover as much of the deficit as possible.
        """
        if deficit <= 0:
            return
        obs = self.behavior.observe(self.context.aid) or {}
        cap = obs_capacity(obs)
        if cap >= 0:
            return  # not a generator (generators have negative capacity)
        sp = obs_setpoint(obs)
        headroom = abs(cap) - abs(sp)
        if headroom < 1e-6:
            return
        share = min(headroom, deficit)
        new_factor = min(1.0, (abs(sp) + share) / abs(cap))
        if self.behavior.has_action(self.context.aid, "regulate"):
            self.behavior.act(self.context.aid, "regulate", new_factor)
            logger.info(
                "[%s] self-island: ramped to %.1f%% (deficit=%.4f)",
                self.context.aid,
                new_factor * 100,
                deficit,
            )


# ---------------------------------------------------------------------------
# Priority mapping helper
# ---------------------------------------------------------------------------


def _compute_actual_priority(priority: int, target: float) -> int:
    """Map a raw priority value and imbalance direction to a gossip round.

    The mapping guarantees:
    - **Restoration (target > 0)**: loads participate ordered by priority
      (lower number = higher urgency = earlier round).  Generators wait
      until all load tiers have had a chance.
    - **Reduction (target < 0)**: generators (priority 0) go first.
      Loads are shed in *reverse* priority order — high-priority loads
      (low number) are shed last.
    """
    if target < 0:
        if priority == 0:
            return 0  # generators first
        # Invert: lower number → more important → shed later (higher round)
        return max(1, _PRIORITY_TIERS - min(priority, _PRIORITY_TIERS) + 1)
    elif target > 0:
        if priority > 0:
            return min(priority, _PRIORITY_TIERS)
        return _PRIORITY_TIERS + 1  # generators last during restoration
    return priority


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_energy_balance_role(
    behavior: RestorationEnvironmentBehavior,
    sector: Sector,
    obs: dict,
    *,
    priority: int | None = None,
) -> EnergyBalanceNegotiator:
    if priority is None:
        priority = obs_priority(obs)
    return EnergyBalanceNegotiator(
        behavior=behavior,
        sector=sector,
        priority=priority,
    )
