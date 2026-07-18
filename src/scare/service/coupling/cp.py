from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from distributed_resource_optimization import (
    create_admm_sharing_data,
    create_admm_start,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
)

from scare.base.channel import (
    CPAllocation,
    CPCommitment,
    CPSetpoint,
    HolonAllocation,
    L3RebalanceWakeup,
    MonotonicVersion,
    SeenVersions,
)
from scare.base.model import (
    AskEnergyMessage,
    AskForAvailableFlex,
    AvailableFlexAnswer,
    NegotiationFinishedEvent,
    OptimizationFinishedLocalEvent,
    ResponseEnergyMessage,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.runtime.diagnostics import record_event
from scare.base.runtime.trace import optimization
from scare.base.topology.topology_mirror import LivePeerFilter
from scare.base.util import (
    apply_regulate,
    clamp_to_constraints,
    kgps_to_mw,
    mw_to_kgps,
    obs_setpoint,
)
from scare.community.supply_priority_admm import allocate_supply_priority
from scare.service.coupling.cp_envelope import CoalitionEnvelope
from scare.service.coupling.cp_flex import (
    aggregate_flex_answers,
    compute_sector_priorities,
)
from scare.service.coupling.cp_l3 import CPComponentView, compute_cp_setpoint

if TYPE_CHECKING:
    from distributed_resource_optimization import ADMMFlexActor
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Sector -> obs key holding that sector's current setpoint.
_ACCESS_KEYS: dict[Sector, str] = {
    Sector.ELECTRICITY: "el_mw",
    Sector.GAS: "gas_mass_flow_kgs",
    Sector.HEAT: "heat_mw",
}

# ADMM result index for each sector
_RESULT_INDEX: dict[Sector, int] = {
    Sector.ELECTRICITY: 0,
    Sector.HEAT: 1,
    Sector.GAS: 2,
}

# Fixed-point tolerance: re-trigger CP ADMM only if the sector setpoint moved
# more than this, else the loop ping-pongs. Units per _ACCESS_KEYS.
_CP_SETPOINT_TOLERANCE: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,  # MW
    Sector.GAS: 1e-4,  # kg/s
    Sector.HEAT: 1e-4,  # MW (~100 W)
}
_CP_DEFAULT_TOLERANCE = 0.01

# Reactive-trigger noise filter (HolonAllocation path): below DEAD_BAND the L2
# signal is suppressed; MIN_GAP_S cools down bursts so the CP can't self-thrash.
_PREDICATE_DEAD_BAND_MW: float = 1e-4
_PREDICATE_MIN_GAP_S: float = 1.0

# Flex-collection deadline (mirrors the curtailment auction's _close_auction):
# a dropped/late AvailableFlexAnswer must not wedge the CP for the episode.
_FLEX_ROUND_TIMEOUT_S: float = 2.0


async def _reply_ask_energy(
    behavior: RestorationEnvironmentBehavior, context: Any, message: AskEnergyMessage, meta: dict
) -> None:
    """Reply to an AskEnergyMessage with the CP's current sector setpoint.

    available=0: a CP has no spare flex, only the conversion knob. Shared by
    both CP roles.
    """
    try:
        obs = behavior.observe(context.aid) or {}
    except (AttributeError, KeyError):
        obs = {}
    key = _ACCESS_KEYS.get(message.sector)
    if key and key in obs:
        raw = obs[key]
        reg = obs.get("regulation", 1.0)
        try:
            value = float(raw) * float(reg)
        except (TypeError, ValueError):
            value = 0.0
    else:
        value = obs_setpoint(obs)
    if not math.isfinite(value):
        value = 0.0
    reply = ResponseEnergyMessage(
        negotiation_id=message.negotiation_id,
        setpoint=value,
        available=0.0,
    )
    await context.send_message(reply, receiver_addr=mango_sender_addr(meta))


class CpActuator:
    """Applies a [EL, HEAT, GAS] ADMM result via the CP's single regulation knob,
    driving the strongest-signal (largest native-MW) sector."""

    def __init__(self, behavior: RestorationEnvironmentBehavior) -> None:
        self._behavior = behavior

    def apply(self, aid: str, result: list[float], timestamp: float) -> float | None:
        obs = self._behavior.observe(aid) or {}
        best_factor: float | None = None
        best_weight = -1.0
        for sector, idx in _RESULT_INDEX.items():
            key = _ACCESS_KEYS[sector]
            if key not in obs or idx >= len(result):
                continue
            value = result[idx]
            if sector == Sector.GAS:
                value = mw_to_kgps(value)
            value = clamp_to_constraints(value, obs, sector)
            cap = float(obs.get(key, 0.0))
            if cap == 0.0:
                continue
            factor = max(0.0, min(1.0, abs(value / cap)))
            weight = abs(result[idx])
            if weight > best_weight:
                best_weight = weight
                best_factor = factor

        if best_factor is not None:
            apply_regulate(
                self._behavior,
                aid,
                best_factor,
                sector="cp",
                reason="cp_admm",
                timestamp=timestamp,
            )
        return best_factor


class FlexRound:
    """The CP flex-collection buffer shared by the legacy and L3 ADMM drivers.

    Owns the answer buffer + round identity only; the ``_active``/``_l3_active``
    locks, driver routing, and deadline scheduling stay on the Role.
    """

    def __init__(self) -> None:
        self.answers: list[AvailableFlexAnswer] = []
        self.expected: int = 0
        self.round_id: str = ""
        self._counter: int = 0

    def open(self, aid: str, n_expected: int) -> str:
        self._counter += 1
        self.round_id = f"{aid}-flex-{self._counter}"
        self.answers = []
        self.expected = n_expected
        return self.round_id

    def add(self, message: AvailableFlexAnswer) -> bool:
        """Record a same-round answer; True once the expected set is complete.
        Rejects late answers after dispatch (expected<=0) and stale round ids."""
        if self.expected <= 0:
            return False
        if (getattr(message, "round_id", "") or "") != self.round_id:
            return False
        self.answers.append(message)
        return len(self.answers) >= self.expected

    def drain(self) -> list[AvailableFlexAnswer]:
        """Snapshot the answers and close the round (zero expected, clear buffer,
        blank round id) so a late answer can't seed a second concurrent run."""
        answers = self.answers[:]
        self.answers = []
        self.expected = 0
        self.round_id = ""
        return answers

    def close_empty(self) -> None:
        """Timeout with no answers: zero expected so no run fires."""
        self.expected = 0


class EnergyConverterRole(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        flex_actor: ADMMFlexActor,
        sectors: list[Sector],
        *,
        live_connector_filter: LivePeerFilter | None = None,
        coupling_ratios: dict[tuple[str, str], float] | None = None,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.flex_actor = flex_actor
        self._actuator = CpActuator(behavior)
        self.sectors = sectors
        # Sibling DynamicConnectorRole gating reachable group leaders.
        # None => static topology: every connector is admitted.
        self._live_connector_filter = live_connector_filter
        # Per-(in,out) sector efficiency advertised when joining a coalition.
        # Empty => no coupling advertised (coalition path skips this CP).
        self._coupling_ratios: dict[tuple[str, str], float] = (
            dict(coupling_ratios) if coupling_ratios else {}
        )
        # Active coalition envelope; read by _run_admm to clamp per-sector bounds,
        # written by _handle_cp_commitment. Empty => CP runs free.
        self._envelope = CoalitionEnvelope()

        self._active: bool = False
        # Flex-collection buffer shared by the legacy and L3 ADMM drivers.
        self._flex = FlexRound()

        # Last-observed group setpoint per sector; feeds the fixed-point gate.
        self._last_sector_setpoint: dict[Sector, float] = {}

        # Count of same-sign imbalance skips (no cross-sector trade helps).
        self._same_sign_skip_count: int = 0

        # HolonAllocation signal: a holon committed a cross-sector shift, letting
        # CP ADMM fire without waiting for downstream gossip.
        self._seen_holon_alloc = SeenVersions()
        self._holon_alloc_signal: dict[tuple[str, Sector], float] = {}
        self._last_holon_predicate_fire_t: float = -1e9

        self._cp_version = MonotonicVersion()

        # Multi-sector L3 wiring, injected via wire_multi_sector_l3 after world
        # construction. Unwired => legacy per-CP path (see CPComponentView.enabled).
        self._component = CPComponentView()

        self._l3_round_counter: int = 0
        # True while the L3 collect->solve->dispatch cycle is in flight so reactive
        # triggers coalesce. Distinct from _active (legacy per-CP path).
        self._l3_active: bool = False

    # ------------------------------------------------------------------
    # Multi-sector L3 wiring + helpers
    # ------------------------------------------------------------------

    def wire_multi_sector_l3(
        self,
        *,
        topology_mirror: Any,
        my_node_id: Any,
        cp_meta_by_aid: dict[str, dict[str, Any]],
        leader_addrs_by_sector: dict[Sector, dict[str, Any]],
        leader_node_ids: dict[str, Any],
    ) -> None:
        """Inject post-construction state for the multi-sector L3 path.

        Called by scenario.restoration after the topology mirror + every CP agent
        are built; pre-call the role is legacy per-CP. topology_mirror is the
        shared instance; cp_meta_by_aid is {cp_aid: {sectors, capacity_mw,
        coupling_ratios, addr, node_id}}; leader_addrs_by_sector is {Sector:
        {leader_aid: addr}}; leader_node_ids feeds reachability filtering.
        """
        self._component.wire(
            topology_mirror=topology_mirror,
            my_node_id=my_node_id,
            cp_meta_by_aid=cp_meta_by_aid,
            leader_addrs_by_sector=leader_addrs_by_sector,
            leader_node_ids=leader_node_ids,
        )

    def _multi_sector_l3_enabled(self) -> bool:
        return self._component.enabled()

    def _is_cps_cluster_leader(self) -> bool:
        """Raw cps-cluster-leader test. Gates the leader-ONLY sites (L2.5
        envelope, legacy per-CP trigger), which must stay leader-scoped even
        under L3 or a non-leader CP would race the L3 path."""
        return topology_characteristic(self, tid="cps") == "leader"

    def _acts_as_cp_leader(self) -> bool:
        """This CP should run coordination logic: cluster leader OR the L3 wiring
        is live (leader test first, so ``_component.enabled()`` is only read when
        not leader — same short-circuit as the original ``char != 'leader' and
        not L3`` gates). Gates the full-form sites (holon-alloc, negotiation-
        finished, cp-allocation)."""
        return self._is_cps_cluster_leader() or self._multi_sector_l3_enabled()

    def _cp_peers_in_component(self) -> dict[str, dict[str, Any]]:
        return self._component.cp_peers(self.context.aid)

    def _leader_addrs_in_component(self) -> dict[Sector, dict[str, Any]]:
        return self._component.leader_addrs(self.context.aid)

    def _schedule_trigger(self) -> None:
        """Route the reactive trigger to the L3 path when wired, else per-CP."""
        if self._multi_sector_l3_enabled():
            self.context.schedule_instant_task(self.trigger_multi_sector_l3())
        else:
            self.context.schedule_instant_task(self.trigger_cp_negotiation())

    def _is_l3_coordinator(self) -> bool:
        return self._component.is_coordinator(self.context.aid)

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))

            return _sync

        logger.debug(
            "[%s] EnergyConverterRole setup: sectors=%s",
            self.context.aid,
            [s.value for s in self.sectors],
        )
        # Purely event-driven; wakes on a cross-sector decision input:
        # NegotiationFinishedEvent (L1), HolonAllocation (L2),
        # CPCommitment (L2.5 envelope), CPAllocation (L3 coord).
        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_energy),
            lambda msg, meta: isinstance(msg, AskEnergyMessage),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_finished),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_flex_answer),
            lambda msg, meta: isinstance(msg, AvailableFlexAnswer),
        )
        # Direct L2 -> L3 trigger: a holon allocation creates cross-sector flow;
        # engage before L1 gossip resolves targets.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_holon_allocation),
            lambda msg, meta: isinstance(msg, HolonAllocation),
        )
        # L2.5 coalition envelope: narrows per-sector ADMM bounds until ttl_s.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_cp_commitment),
            lambda msg, meta: isinstance(msg, CPCommitment),
        )
        # L3 dispatch: every CP applies the coord's per-sector setpoint.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_cp_allocation),
            lambda msg, meta: (
                isinstance(msg, CPAllocation) and msg.cp_aid == str(self.context.aid)
            ),
        )

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        await _reply_ask_energy(self.behavior, self.context, message, meta)

    async def _handle_holon_allocation(
        self, message: HolonAllocation, meta: dict
    ) -> None:
        """Direct L2 -> L3 trigger: a holon's per-member allocation is a leading
        indicator of a sector-balance shift, so engage before gossip resolves.
        """
        char = topology_characteristic(self, tid="cps")
        logger.debug(
            "[%s] CP received holon-allocation: sector=%s n_targets=%d v=%d from %s char=%s",
            self.context.aid,
            message.sector.value,
            len(message.targets_mw),
            message.version,
            message.publisher,
            char,
        )
        if not self._acts_as_cp_leader():
            return
        # Echo guard: skip an allocation our own CP setpoint caused.
        if (
            message.caused_by.get(self.context.aid, -1) == self._cp_version.current
            and self._cp_version.current > 0
        ):
            return
        if not self._seen_holon_alloc.is_fresh(message.publisher, message.version):
            return

        # Signal = Σ|targets_mw|: rebalance intent regardless of shed
        # distribution; sign is meaningless as per-member allocations can cancel.
        signal = sum(abs(v) for v in message.targets_mw.values())
        key = (message.publisher, message.sector)
        self._holon_alloc_signal[key] = signal
        self._seen_holon_alloc.mark(message.publisher, message.version)

        if self._active:
            return
        if signal < _PREDICATE_DEAD_BAND_MW:
            return

        now = float(self.context.current_timestamp)
        if now - self._last_holon_predicate_fire_t < _PREDICATE_MIN_GAP_S:
            return
        self._last_holon_predicate_fire_t = now

        logger.info(
            "[%s] CP holon-allocation predicate fired: sector=%s "
            "sum_abs_targets=%.4f (publisher=%s v=%d)",
            self.context.aid,
            message.sector.value,
            signal,
            message.publisher,
            message.version,
        )
        self._schedule_trigger()

    async def _handle_cp_commitment(self, message: CPCommitment, meta: dict) -> None:
        """Write directional coalition flows into envelope state with a TTL.

        _run_admm clamps per-sector bounds to stay inside the commitment.
        Idempotent latest-wins; honoured only if cp_id addresses this CP.
        """
        if message.cp_id and message.cp_id != str(self.context.aid):
            return
        if not self._is_cps_cluster_leader():
            return
        # Translate sector-value keys to Sector enums; unknown strings dropped.
        flows: dict[Sector, float] = {}
        for sec_v, mw in message.target_flows_mw.items():
            try:
                flows[Sector(sec_v)] = float(mw)
            except (ValueError, TypeError):
                continue
        if not flows:
            return
        now = float(self.context.current_timestamp)
        self._envelope.set(flows, message.ttl_s, message.coalition_id, now=now)
        logger.info(
            "[%s] CP envelope set by coalition %s: flows=%s ttl=%.2fs",
            self.context.aid,
            message.coalition_id,
            {s.value: round(v, 4) for s, v in flows.items()},
            float(message.ttl_s),
        )
        record_event(
            t=now,
            kind="cp_envelope_set",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"coalition={message.coalition_id} ttl={float(message.ttl_s):.2f} "
                f"flows={{{', '.join(f'{s.value}: {v:.4f}' for s, v in flows.items())}}}"
            ),
        )

    def _envelope_active(self) -> bool:
        return self._envelope.active(float(self.context.current_timestamp))

    def _clamp_to_envelope(self, result: list) -> list:
        """Replace ADMM result dims with coalition-committed flows when an L2.5
        envelope is active, else return untouched. Records cp_envelope_clamp.
        """
        now = float(self.context.current_timestamp)
        pre_clamp = self._envelope.clamp(result, _RESULT_INDEX, now=now)
        if pre_clamp is None:
            logger.info("[%s] ADMM result: %s", self.context.aid, result)
            return result
        logger.info(
            "[%s] CP ADMM result clamped to coalition envelope %s: %s",
            self.context.aid,
            self._envelope.coalition_id,
            result,
        )
        record_event(
            t=now,
            kind="cp_envelope_clamp",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"coalition={self._envelope.coalition_id} pre={pre_clamp} post={result}"
            ),
        )
        return result

    async def _handle_negotiation_finished(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        char = topology_characteristic(self, tid="cps")
        logger.debug(
            "[%s] CP received NegotiationFinishedEvent (sector=%s, new_sp=%.4f, my_cps_char=%s)",
            self.context.aid,
            message.sector.value,
            message.new_setpoint,
            char,
        )
        # Non-leader coordinators must still wake on gossip convergence in L3;
        # trigger_multi_sector_l3 re-gates on the actual coordinator.
        if not self._acts_as_cp_leader():
            return
        if self._active:
            return
        # Fixed-point gate: skip re-trigger if the setpoint barely moved.
        sector = message.sector
        tol = _CP_SETPOINT_TOLERANCE.get(sector, _CP_DEFAULT_TOLERANCE)
        prev = self._last_sector_setpoint.get(sector)
        new = message.new_setpoint
        if prev is not None and abs(new - prev) < tol:
            logger.debug(
                "[%s] CP re-trigger suppressed (sector=%s, |Δ|=%.6g < %.6g)",
                self.context.aid,
                sector.value,
                abs(new - prev),
                tol,
            )
            return
        self._last_sector_setpoint[sector] = new
        self._schedule_trigger()

    def _live_connectors(self, connectors: list) -> list:
        """Filter connectors through the sibling DynamicConnectorRole (else
        passthrough), so every fan-out honours reachability uniformly.
        """
        if self._live_connector_filter is None:
            return list(connectors)
        kept = [c for c in connectors if self._live_connector_filter.is_live(c)]
        if len(kept) != len(connectors) and logger.isEnabledFor(logging.DEBUG):
            ctx = getattr(self, "context", None)
            logger.debug(
                "[%s] CP filter dropped %d unreachable connectors (kept=%d)",
                getattr(ctx, "aid", "<detached>"),
                len(connectors) - len(kept),
                len(kept),
            )
        return kept

    async def trigger_cp_negotiation(self) -> None:
        if not self._is_cps_cluster_leader():
            return
        if self._active:
            return
        self._active = True

        group_leaders = self._live_connectors(topology_connectors(self, tid="cps"))
        if not group_leaders:
            logger.info(
                "[%s] CP trigger skipped: no connected group leaders",
                self.context.aid,
            )
            self._active = False
            return

        round_id = self._open_flex_round(len(group_leaders))

        logger.info(
            "[%s] CP triggered: asking %d group leaders for flex",
            self.context.aid,
            len(group_leaders),
        )
        msg = AskForAvailableFlex(include_connectors=False, round_id=round_id)
        for addr in group_leaders:
            await self.context.send_message(msg, receiver_addr=addr)

    def _open_flex_round(self, n_expected: int) -> str:
        """Arm a flex-collection round: reset the answer buffer, stamp a fresh
        round id, and schedule the deadline that closes it on partial answers
        so one dropped reply can't wedge the CP."""
        round_id = self._flex.open(str(self.context.aid), n_expected)
        deadline = float(self.context.current_timestamp) + _FLEX_ROUND_TIMEOUT_S
        self.context.schedule_timestamp_task(
            self._close_flex_round(round_id), timestamp=deadline
        )
        return round_id

    async def _close_flex_round(self, round_id: str) -> None:
        """Deadline task: close a still-open round on whatever answers arrived
        and clear the active flag (nothing arrived => plain unwedge)."""
        if round_id != self._flex.round_id:
            return
        if not (self._active or self._l3_active):
            return
        if self._flex.expected <= 0:
            return  # complete set already dispatched to the ADMM driver
        if len(self._flex.answers) >= self._flex.expected:
            return
        record_event(
            t=float(self.context.current_timestamp),
            kind="cp_flex_round_timeout",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"round={round_id} answers={len(self._flex.answers)}"
                f"/{self._flex.expected}"
            ),
        )
        if not self._flex.answers:
            self._flex.close_empty()
            self._active = False
            self._l3_active = False
            return
        if self._l3_active:
            await self._run_multi_sector_admm()
        else:
            await self._run_admm()

    async def _handle_flex_answer(
        self, message: AvailableFlexAnswer, meta: dict
    ) -> None:
        # Route to the L3 (_l3_active, set by trigger_multi_sector_l3) vs legacy
        # (_active, set by trigger_cp_negotiation) driver; the two are mutually
        # exclusive by design and both guards below reject re-entry.
        if not self._l3_active and not self._active:
            return
        # add() rejects a late same-round answer arriving during the ADMM await
        # (expected<=0, so it can't seed a second concurrent run) and a stale
        # round id; it returns True only once the expected set is complete.
        if not self._flex.add(message):
            return
        if self._l3_active:
            await self._run_multi_sector_admm()
        else:
            await self._run_admm()

    async def _run_admm(self) -> None:
        answers = self._flex.drain()

        agg = aggregate_flex_answers(answers)
        imbalance_by_sector = agg.imbalance_by_sector
        unmet_by_sector_total = agg.unmet_by_sector_total

        # Combine balance + unsigned unmet so a sector with disconnected loads
        # (balance≈0) still shows a positive deficit.
        imb_el = imbalance_by_sector.get(
            Sector.ELECTRICITY, 0.0
        ) + unmet_by_sector_total.get(Sector.ELECTRICITY, 0.0)
        imb_heat = imbalance_by_sector.get(
            Sector.HEAT, 0.0
        ) + unmet_by_sector_total.get(Sector.HEAT, 0.0)
        imb_gas = imbalance_by_sector.get(Sector.GAS, 0.0) + unmet_by_sector_total.get(
            Sector.GAS, 0.0
        )

        # Imbalances arrive in natural units (gas kg/s); ADMM is all-MW, so only
        # gas needs conversion.
        T = np.array([imb_el, imb_heat, kgps_to_mw(imb_gas)])

        if np.all(T >= 0) or np.all(T <= 0):
            self._same_sign_skip_count += 1
            # Log first skip and every 10th thereafter, to avoid flooding.
            if self._same_sign_skip_count == 1 or self._same_sign_skip_count % 10 == 0:
                logger.info(
                    "[%s] CP ADMM skipped (same-sign T=%s, n=%d)",
                    self.context.aid,
                    T.tolist(),
                    self._same_sign_skip_count,
                )
            # Record the wake-up despite the bail, so CP engagement isn't
            # undercounted on structurally same-sign grids.
            record_event(
                t=float(self.context.current_timestamp),
                kind="cp_admm_skipped_same_sign",
                aid=str(self.context.aid),
                sector="cp",
                detail=f"T={T.tolist()} n_skips={self._same_sign_skip_count}",
            )
            self._active = False
            return

        priorities = compute_sector_priorities(np, agg)

        coordinator = create_sharing_target_distance_admm_coordinator()
        start_msg = create_admm_start(
            create_admm_sharing_data(T.tolist(), priorities=priorities.tolist())
        )

        try:
            with optimization("admm_cp", logger=logger, aid=self.context.aid):
                await start_coordinated_optimization(
                    [self.flex_actor], coordinator, start_msg
                )
            result = self._clamp_to_envelope(list(self.flex_actor.x))
            # The actuator must run before emit_event: with no subscriber,
            # emit_event raises KeyError that would discard the result.
            applied_factor = self._actuator.apply(
                self.context.aid, result, self.context.current_timestamp
            )
            try:
                self.context.emit_event(OptimizationFinishedLocalEvent(result=result))
            except KeyError:
                pass
            group_leaders = self._live_connectors(topology_connectors(self, tid="cps"))
            # Publish CPSetpoint alongside StartBalanceNegotiation so holons can
            # re-evaluate. Layout per _RESULT_INDEX [0=EL, 1=HEAT, 2=GAS] MW.
            sector_flows: dict[str, float] = {}
            for sector, idx in _RESULT_INDEX.items():
                if idx < len(result):
                    sector_flows[sector.value] = float(result[idx])
            cp_decision = CPSetpoint(
                publisher=str(self.context.aid),
                version=self._cp_version.next(),
                caused_by={},
                timestamp_s=float(self.context.current_timestamp),
                cp_id=str(self.context.aid),
                sector_flows_mw=sector_flows,
                regulation_factor=1.0 if applied_factor is None else float(applied_factor),
            )
            record_event(
                t=float(self.context.current_timestamp),
                kind="cp_setpoint",
                aid=str(self.context.aid),
                sector="cp",
                detail=(
                    f"flows={{{', '.join(f'{s}: {v:.4f}' for s, v in sector_flows.items())}}} "
                    f"reg={1.0 if applied_factor is None else float(applied_factor):.3f} "
                    f"envelope_active={self._envelope_active()}"
                ),
            )
            logger.debug(
                "[%s] CP publish: sector_flows=%s reg=%.3f v=%d to %d holons",
                self.context.aid,
                {s: round(v, 3) for s, v in sector_flows.items()},
                cp_decision.regulation_factor,
                cp_decision.version,
                len(group_leaders),
            )
            for addr in group_leaders:
                await self.context.send_message(
                    StartBalanceNegotiation(), receiver_addr=addr
                )
                await self.context.send_message(cp_decision, receiver_addr=addr)
        except Exception as exc:
            logger.error("[%s] ADMM failed: %s", self.context.aid, exc)
            # Fallback: trigger intra-group gossip so groups still rebalance.
            for addr in self._live_connectors(topology_connectors(self, tid="cps")):
                await self.context.send_message(
                    StartBalanceNegotiation(), receiver_addr=addr
                )

        self._active = False

    # ------------------------------------------------------------------
    # Multi-sector L3 driver
    # ------------------------------------------------------------------

    async def trigger_multi_sector_l3(self) -> None:
        """L3-coord entrypoint for the joint multi-sector ADMM round.

        Only the lex-smallest CP aid in the component runs; others wait for its
        CPAllocation broadcast. Collects component-leader flex, runs the
        supply-priority kernel, broadcasts per-CP CPAllocation. Falls back to
        trigger_cp_negotiation when multi-sector wiring is unavailable.
        """
        if not self._multi_sector_l3_enabled():
            await self.trigger_cp_negotiation()
            return
        # Deliberately NOT gated on cps-cluster-leader: the coord is the
        # lex-smallest aid in the component, often not the cluster leader, so
        # gating here would leave L3 silent while L2 defers.
        if not self._is_l3_coordinator():
            return
        if self._l3_active or self._active:
            return

        component_leaders = self._leader_addrs_in_component()
        # Flatten {sector: {aid: addr}} -> list[addr], deduped by aid.
        all_addrs: list[Any] = []
        seen_aids: set[str] = set()
        for sec_table in component_leaders.values():
            for aid, addr in sec_table.items():
                if aid in seen_aids:
                    continue
                seen_aids.add(aid)
                all_addrs.append(addr)
        if not all_addrs:
            logger.info(
                "[%s] L3 trigger skipped: no reachable leaders in component",
                self.context.aid,
            )
            return

        self._l3_active = True
        round_id = self._open_flex_round(len(all_addrs))

        logger.info(
            "[%s] L3 coord triggered: asking %d leaders in MS component for flex",
            self.context.aid,
            len(all_addrs),
        )
        msg = AskForAvailableFlex(include_connectors=False, round_id=round_id)
        for addr in all_addrs:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _run_multi_sector_admm(self) -> None:
        """Run the joint supply-priority ADMM over collected leader flex and
        dispatch. Called by _handle_flex_answer on the L3-coord path.
        """
        try:
            await self._run_multi_sector_admm_inner()
        finally:
            self._l3_active = False

    async def _run_multi_sector_admm_inner(self) -> None:
        answers = self._flex.drain()
        if not answers:
            return

        # Build (supply, demand) actor lists, one actor per leader. The ADMM is
        # sector-agnostic, so passing every touched sector solves them jointly.
        actor_supplies: list[dict[str, float]] = []
        actor_demands: list[dict[str, dict[int, float]]] = []
        for a in answers:
            actor_supplies.append(dict(a.supply_by_sector or {}))
            actor_demands.append(
                {
                    sec: dict(tmap)
                    for sec, tmap in (a.demand_by_sector_priority or {}).items()
                }
            )

        sectors = sorted({s for d in actor_demands for s in d})
        if not sectors:
            return
        tiers_present: set[int] = set()
        for d in actor_demands:
            for tmap in d.values():
                tiers_present.update(tmap.keys())
        tiers = sorted(t for t in tiers_present if t >= 1)
        if not tiers:
            return
        total_demand = sum(
            float(v)
            for d in actor_demands
            for tmap in d.values()
            for v in tmap.values()
        )
        if total_demand < 1e-6:
            return

        round_id = f"l3-r{self._l3_round_counter}"
        self._l3_round_counter += 1

        try:
            service_fraction, _per_actor_x, meta = await allocate_supply_priority(
                sectors=sectors,
                tiers=tiers,
                actor_supplies=actor_supplies,
                actor_demands=actor_demands,
                actor_ub_overrides=None,
                priority_tiers=4,
                max_iters=50,
                abs_tol=1e-3,
                enable_priority_weighting=True,
            )
        except Exception as exc:
            logger.error(
                "[%s] multi-sector L3 ADMM failed: %s",
                self.context.aid,
                exc,
            )
            record_event(
                t=float(self.context.current_timestamp),
                kind="l3_admm_failed",
                aid=str(self.context.aid),
                sector="cp",
                detail=f"multi_sector: {exc}",
            )
            return

        # Noise scrub: clamp sub-tolerance ADMM residuals to 0 to restore
        # priority-monotonicity without touching genuine partial service.
        _FRACTION_NOISE = 1e-3
        for sec_key in service_fraction:
            for tier, frac in list(service_fraction[sec_key].items()):
                if 0.0 < float(frac) < _FRACTION_NOISE:
                    service_fraction[sec_key][tier] = 0.0

        logger.info(
            "[%s] L3 multi-sector ADMM result: round=%s sectors=%s tiers=%s "
            "n_leaders=%d fractions=%s",
            self.context.aid,
            round_id,
            sectors,
            tiers,
            len(answers),
            service_fraction,
        )
        record_event(
            t=float(self.context.current_timestamp),
            kind="l3_admm_result",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"round={round_id} sectors={sectors} n_leaders={len(answers)} "
                f"fractions={service_fraction}"
            ),
        )

        # L3 does NOT dispatch ComponentAllocation: per-sector priority
        # allocation is L2's job. L3's sole output is CP setpoints, fed by the
        # per-sector marginals below.

        # marginal(sector) = 1 - min served fraction over positive-demand tiers:
        # 0 = fully served, near 1 = stressed. Dispatch a per-CP setpoint.
        now = float(self.context.current_timestamp)
        marginal_by_sector: dict[str, float] = {}
        for sec in sectors:
            tmap = service_fraction.get(sec, {})
            if not tmap:
                marginal_by_sector[sec] = 0.0
                continue
            lowest = min(tmap.values())
            marginal_by_sector[sec] = max(0.0, 1.0 - float(lowest))

        cp_peers = self._cp_peers_in_component()
        for cp_aid, meta in cp_peers.items():
            cp_setpoint = compute_cp_setpoint(meta, marginal_by_sector)
            allocation_msg = CPAllocation(
                publisher=str(self.context.aid),
                version=self._cp_version.next(),
                caused_by={},
                timestamp_s=now,
                cp_aid=cp_aid,
                round_id=round_id,
                sector_flows_mw=cp_setpoint,
            )
            cp_addr = meta.get("addr")
            if cp_addr is None:
                continue
            await self.context.send_message(allocation_msg, receiver_addr=cp_addr)

        # Wake L2 in every touched sector: new CP setpoints change leader flex
        # next gossip pass. L3RebalanceWakeup lifts the leader's L2 short-circuit.
        component_leaders = self._leader_addrs_in_component()
        for sector, sec_table in component_leaders.items():
            tier_map = service_fraction.get(sector.value, {})
            if not tier_map:
                continue
            wakeup = L3RebalanceWakeup(
                publisher=str(self.context.aid),
                version=self._cp_version.next(),
                caused_by={},
                timestamp_s=now,
                sector=sector,
            )
            for addr in sec_table.values():
                await self.context.send_message(wakeup, receiver_addr=addr)

    async def _handle_cp_allocation(self, message: CPAllocation, meta: dict) -> None:
        """Apply an L3-coord setpoint via the CP actuator so it shares the legacy
        per-CP path. Idempotent: apply_regulate dedups same-value writes.
        """
        # CPAllocation is per-CP addressed (subscribe filter on cp_aid): in L3
        # every CP applies its own setpoint, so this is the full-form gate.
        if not self._acts_as_cp_leader():
            return
        flows_mw = dict(message.sector_flows_mw)
        # Translate to the flat [el, heat, gas] vector; missing sectors stay 0.
        result = [0.0, 0.0, 0.0]
        for sec, idx in _RESULT_INDEX.items():
            v = flows_mw.get(sec.value)
            if v is not None:
                result[idx] = float(v)
        applied_factor = self._actuator.apply(
            self.context.aid, result, self.context.current_timestamp
        )
        record_event(
            t=float(self.context.current_timestamp),
            kind="cp_setpoint",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"source=l3 round={message.round_id} flows={flows_mw} "
                f"reg={1.0 if applied_factor is None else float(applied_factor):.3f}"
            ),
        )


class MultiCommunityCPRole(Role):
    """CP-side coordination role for the ``component_level`` baseline.

    With the ADMM/holonic layers off, each bridged community sends an
    independent NegotiationFinishedEvent ask. This role reconciles them under an
    EMA-blended target with a deadband + cooldown so the CP can't ping-pong
    between contradictory commits. Replaces the EnergyConverterRole pipeline
    (no ADMM/flex-actor), keeping only the minimal AskEnergyMessage reply. The
    EMA resets on every BranchFailureEvent (see on_branch_failure).
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sectors: list[Sector],
        *,
        ema_alpha: float = 0.3,
        deadband_mw: float = 0.05,
        min_interval_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sectors = list(sectors)
        self._ema_alpha = float(ema_alpha)
        self._deadband_mw = float(deadband_mw)
        self._min_interval_s = float(min_interval_s)
        # Per-sector EMA target, seeded lazily to the first observed setpoint so
        # it doesn't start at 0 (which biases the CP toward off).
        self._target_by_sector: dict[Sector, float] = {}
        # Last committed setpoint per sector; deadband anchors against it.
        self._committed_by_sector: dict[Sector, float] = {}
        # Sim-time of the last apply_regulate call; cooldown gate.
        self._last_commit_t: float = -1e9

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))

            return _sync

        # Minimal flex-query reply so a leader treating this CP as a connector
        # doesn't stall. available=0: no spare flex, only the conversion knob.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_energy),
            lambda msg, meta: isinstance(msg, AskEnergyMessage),
        )
        # Per-community ask: each gossip convergence broadcasts a settled
        # NegotiationFinishedEvent setpoint.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_finished),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        # on_branch_failure is wired by the scenario builder via behavior_in.
        # CPs bridge via cross-link (not community members), so the global
        # branch-failure event is the only reset signal reaching every CP.

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        await _reply_ask_energy(self.behavior, self.context, message, meta)

    async def _handle_negotiation_finished(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        sector = message.sector
        if sector not in self.sectors:
            return
        proposed = float(message.new_setpoint)
        if not math.isfinite(proposed):
            return

        # EMA blend; first observation seeds the target (else the first commit
        # chases 0 → proposed over several rounds).
        if sector not in self._target_by_sector:
            self._target_by_sector[sector] = proposed
        else:
            alpha = self._ema_alpha
            self._target_by_sector[sector] = (
                alpha * proposed + (1.0 - alpha) * self._target_by_sector[sector]
            )

        await self._maybe_commit(sector)

    async def _maybe_commit(self, sector: Sector) -> None:
        target = self._target_by_sector.get(sector)
        if target is None:
            return
        now = float(self.context.current_timestamp)
        if now - self._last_commit_t < self._min_interval_s:
            return

        committed = self._committed_by_sector.get(sector, 0.0)
        if abs(target - committed) < self._deadband_mw:
            return

        try:
            obs = self.behavior.observe(self.context.aid) or {}
        except (AttributeError, KeyError):
            obs = {}
        clamped = clamp_to_constraints(target, obs, sector)
        key = _ACCESS_KEYS.get(sector)
        cap = float(obs.get(key, 0.0)) if key else 0.0
        if cap == 0.0 or not math.isfinite(cap):
            return
        factor = max(0.0, min(1.0, abs(clamped / cap)))

        apply_regulate(
            self.behavior,
            self.context.aid,
            factor,
            sector="cp",
            reason="cp_multi_community",
            timestamp=now,
        )
        self._committed_by_sector[sector] = float(clamped)
        self._last_commit_t = now
        record_event(
            t=now,
            kind="cp_setpoint",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"source=multi_community sector={sector.value} "
                f"target={clamped:.4f} factor={factor:.3f} "
                f"deadband={self._deadband_mw:.3f} cooldown={self._min_interval_s:.2f}"
            ),
        )

    def on_branch_failure(self, branch_id: Any) -> None:
        """Reset the per-sector EMA on every observed branch failure.

        We don't decide whether this failure islands a bridged community; the
        EMA re-seeds within a few events so a spurious reset is cheap.
        _committed_by_sector + cooldown are kept (physical setpoint unchanged).
        """
        if not self._target_by_sector:
            return
        self._target_by_sector.clear()
        try:
            t = float(self.context.current_timestamp)
        except Exception:
            t = 0.0
        record_event(
            t=t,
            kind="cp_ema_reset",
            aid=str(self.context.aid),
            sector="cp",
            detail=f"trigger=branch_failure branch={branch_id}",
        )
