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
    ComponentAllocation,
    CPAllocation,
    CPCommitment,
    CPFlexReport,
    CPSetpoint,
    HolonAllocation,
    L3RebalanceWakeup,
    MonotonicVersion,
    SeenVersions,
)
from scare.base.diagnostics import record_event
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
from scare.base.topology_mirror import LivePeerFilter
from scare.base.util import (
    apply_regulate,
    clamp_to_constraints,
    kgps_to_mw,
    mw_to_kgps,
    obs_setpoint,
)
from scare.community.supply_priority_admm import allocate_supply_priority
from scare.service.cp_envelope import CoalitionEnvelope
from scare.service.cp_flex import aggregate_flex_answers, compute_sector_priorities
from scare.service.cp_l3 import CPComponentView, compute_cp_setpoint

if TYPE_CHECKING:
    from distributed_resource_optimization import ADMMFlexActor
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)

# Maps sector to the obs key that holds the current setpoint for that sector
_ACCESS_KEYS: dict[Sector, str] = {
    Sector.ELECTRICITY: "el_mw",
    Sector.GAS: "gas_kgps",
    Sector.HEAT: "heat_mw",
}

# ADMM result index for each sector
_RESULT_INDEX: dict[Sector, int] = {
    Sector.ELECTRICITY: 0,
    Sector.HEAT: 1,
    Sector.GAS: 2,
}

# CP fixed-point tolerance: a NegotiationFinishedEvent re-triggers CP ADMM
# only if the sector's new setpoint moved more than this; below it the loop
# is at a fixed-point and re-triggering would ping-pong. Units per _ACCESS_KEYS.
_CP_SETPOINT_TOLERANCE: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,   # MW
    Sector.GAS: 1e-4,           # kg/s
    Sector.HEAT: 1e-4,          # MW (~100 W)
}
_CP_DEFAULT_TOLERANCE = 0.01

# Reactive-trigger noise filter (HolonAllocation path). Below the dead-band the
# L2 cross-sector signal is treated as noise and suppressed; MIN_GAP_S enforces
# a cooldown so a burst of L2 dispatches cannot self-thrash the CP.
_PREDICATE_DEAD_BAND_MW: float = 1e-4
_PREDICATE_MIN_GAP_S: float = 1.0


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
        self.sectors = sectors
        # Optional sibling DynamicConnectorRole classifying which group leaders
        # the CP can still physically reach. None => static topology: every
        # connector from topology_connectors is admitted.
        self._live_connector_filter = live_connector_filter
        # Per-(in_sector, out_sector) efficiency advertised when this CP joins a
        # coalition, mapping output supply to input draw. None => no coupling
        # advertised (the cross-sector coalition path skips this CP).
        self._coupling_ratios: dict[tuple[str, str], float] = (
            dict(coupling_ratios) if coupling_ratios else {}
        )
        # Active cross-sector coalition envelope. Empty => CP runs free. Read by
        # _run_admm to clamp per-sector bounds; written by _handle_cp_commitment.
        self._envelope = CoalitionEnvelope()

        self._active: bool = False
        self._flex_answers: list[AvailableFlexAnswer] = []
        self._flex_expected: int = 0

        # Per-sector last-observed group setpoint; the fixed-point gate uses it
        # to suppress re-triggering when a negotiation converged to ~the same point.
        self._last_sector_setpoint: dict[Sector, float] = {}

        # ADMM declines when the imbalance vector is same-sign across all sectors
        # (no cross-sector trade helps). Log every Nth such skip at INFO.
        self._same_sign_skip_count: int = 0

        # HolonAllocation signal: a holon committed a cross-sector shift; we may
        # fire CP ADMM directly without waiting for the downstream gossip.
        self._seen_holon_alloc = SeenVersions()
        self._holon_alloc_signal: dict[tuple[str, Sector], float] = {}
        self._last_holon_predicate_fire_t: float = -1e9

        self._cp_version = MonotonicVersion()

        # Multi-sector L3 wiring + component-scoping queries, injected via
        # wire_multi_sector_l3 after world construction. Unwired => legacy
        # per-CP path runs (see CPComponentView.enabled).
        self._component = CPComponentView()

        # Latest CPAllocation this CP applied (warm-start + no-op dispatch suppression).
        self._last_l3_setpoint_by_sector: dict[str, float] = {}
        # Round counter for the multi-sector ADMM the L3 coord drives.
        self._l3_round_counter: int = 0
        # True while the L3 coord's collect->solve->dispatch cycle is in flight so
        # reactive triggers coalesce. Distinct from _active (legacy per-CP path).
        self._l3_active: bool = False

    # ------------------------------------------------------------------
    # Multi-sector L3 (Option B) wiring + helpers
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
        """Inject post-construction state the multi-sector L3 path needs to elect
        a coordinator and drive a joint ADMM. Called by scenario.restoration after
        the topology mirror + every CP agent are built; pre-call the role is
        legacy per-CP.

        topology_mirror is the shared instance the community + holonic roles also
        consult (so failures propagate consistently across L2/L3); my_node_id is
        this CP's host node; cp_meta_by_aid is {cp_aid: {sectors, capacity_mw,
        coupling_ratios, addr, node_id}} for every CP; leader_addrs_by_sector is
        {Sector: {leader_aid: addr}}; leader_node_ids is {leader_aid: node_id}
        for reachability filtering.
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

    def _cp_peers_in_component(self) -> dict[str, dict[str, Any]]:
        return self._component.cp_peers(self.context.aid)

    def _leader_addrs_in_component(self) -> dict[Sector, dict[str, Any]]:
        return self._component.leader_addrs(self.context.aid)

    def _schedule_trigger(self) -> None:
        """Route the reactive trigger to the multi-sector L3 path when wired,
        else the legacy per-CP path. Used by the holon-allocation / coalition
        handlers.
        """
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
            self.context.aid, [s.value for s in self.sectors],
        )
        # CP triggering is purely event-driven; the role wakes only on a
        # cross-sector decision input:
        #   - NegotiationFinishedEvent (L1 gossip converged on a new group setpoint)
        #   - HolonAllocation (L2 holon dispatched a fresh per-member plan)
        #   - CPCommitment (L2.5 coalition issued a directional envelope)
        #   - CPAllocation (multi-sector L3 coord assigned this CP)
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
        # Direct L2 -> L3 trigger: a holon committed a per-member allocation
        # creating cross-sector flow; engage before L1 gossip resolves targets.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_holon_allocation),
            lambda msg, meta: isinstance(msg, HolonAllocation),
        )
        # Cross-sector coalition envelope: L2.5 dispatches a CPCommitment per
        # re-assert tick while active; on receipt we narrow per-sector ADMM
        # bounds so the coalition's decision is honoured until ttl_s expiry.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_cp_commitment),
            lambda msg, meta: isinstance(msg, CPCommitment),
        )
        # Multi-sector L3 dispatch: every CP (incl. coord) applies its per-sector
        # setpoint from the elected coordinator via the regulation path.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_cp_allocation),
            lambda msg, meta: isinstance(msg, CPAllocation)
            and msg.cp_aid == str(self.context.aid),
        )

    async def _handle_ask_energy(self, message: AskEnergyMessage, meta: dict) -> None:
        try:
            obs = self.behavior.observe(self.context.aid) or {}
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
        # available=0: a CP has no spare flex of its own, only the conversion knob.
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=value,
            available=0.0,
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_holon_allocation(
        self, message: HolonAllocation, meta: dict
    ) -> None:
        """Direct L2 -> L3 trigger. A holon published its per-member ADMM
        allocation; the target magnitudes are a leading indicator the groups are
        about to shift sector balance, so the CP can engage before the gossip
        chain resolves new targets and broadcasts NegotiationFinishedEvent.
        """
        char = topology_characteristic(self, tid="cps")
        logger.debug(
            "[%s] CP received holon-allocation: sector=%s n_targets=%d v=%d from %s char=%s",
            self.context.aid, message.sector.value,
            len(message.targets_mw), message.version, message.publisher, char,
        )
        # Cps-leader gate, active only in legacy single-sector mode; multi-sector
        # L3 uses the _is_l3_coordinator election inside the dispatcher.
        if char != "leader" and not self._multi_sector_l3_enabled():
            return
        # Echo guard: a holon allocation our own CP setpoint caused isn't news.
        if (
            message.caused_by.get(self.context.aid, -1) == self._cp_version.current
            and self._cp_version.current > 0
        ):
            return
        if not self._seen_holon_alloc.is_fresh(message.publisher, message.version):
            return

        # Per-(publisher, sector) signal = Σ|targets_mw|: captures the holon's
        # rebalance intent regardless of shed distribution. Sign is meaningless
        # here since L2's per-member allocations can cancel within a sector.
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
            self.context.aid, message.sector.value, signal,
            message.publisher, message.version,
        )
        self._schedule_trigger()

    async def _handle_cp_commitment(
        self, message: CPCommitment, meta: dict
    ) -> None:
        """Cross-sector coalition envelope handler.

        Writes directional sector flows into local envelope state with a TTL;
        _run_admm clamps per-sector bounds so subsequent L3 rounds stay inside
        the commitment. Idempotent (latest-wins): re-asserts refresh TTL and
        overwrite flows. Honoured only if message.cp_id addresses this CP, so a
        single broadcast can target multiple CPs without cross-action.
        """
        if message.cp_id and message.cp_id != str(self.context.aid):
            return
        if topology_characteristic(self, tid="cps") != "leader":
            return
        # Translate sector-value-keyed flows to Sector enums; unknown sector
        # strings are silently dropped.
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
            self.context.aid, message.coalition_id,
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
        """Replace each sector dimension of the ADMM result with the
        coalition-committed flow when an L2.5 envelope is active; else return it
        untouched. Records cp_envelope_clamp when a clamp lands.
        """
        now = float(self.context.current_timestamp)
        pre_clamp = self._envelope.clamp(result, _RESULT_INDEX, now=now)
        if pre_clamp is None:
            logger.info("[%s] ADMM result: %s", self.context.aid, result)
            return result
        logger.info(
            "[%s] CP ADMM result clamped to coalition envelope %s: %s",
            self.context.aid, self._envelope.coalition_id, result,
        )
        record_event(
            t=now,
            kind="cp_envelope_clamp",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"coalition={self._envelope.coalition_id} "
                f"pre={pre_clamp} post={result}"
            ),
        )
        return result

    async def _handle_negotiation_finished(
        self, message: NegotiationFinishedEvent, meta: dict
    ) -> None:
        char = topology_characteristic(self, tid="cps")
        logger.debug(
            "[%s] CP received NegotiationFinishedEvent (sector=%s, new_sp=%.4f, my_cps_char=%s)",
            self.context.aid, message.sector.value, message.new_setpoint, char,
        )
        if char != "leader":
            return
        if self._active:
            return
        # Fixed-point gate: skip re-trigger if this sector's group setpoint
        # hasn't moved enough to change the ADMM answer.
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
        """Filter connectors through the sibling DynamicConnectorRole when
        attached, else passthrough. Centralised so every fan-out to group
        leaders honours reachability uniformly.
        """
        if self._live_connector_filter is None:
            return list(connectors)
        kept = [c for c in connectors if self._live_connector_filter.is_live(c)]
        if len(kept) != len(connectors) and logger.isEnabledFor(logging.DEBUG):
            ctx = getattr(self, "context", None)
            logger.debug(
                "[%s] CP filter dropped %d unreachable connectors (kept=%d)",
                getattr(ctx, "aid", "<detached>"),
                len(connectors) - len(kept), len(kept),
            )
        return kept

    async def trigger_cp_negotiation(self) -> None:
        if topology_characteristic(self, tid="cps") != "leader":
            return
        if self._active:
            return
        self._active = True

        group_leaders = self._live_connectors(topology_connectors(self, tid="cps"))
        if not group_leaders:
            logger.info(
                "[%s] CP trigger skipped: no connected group leaders", self.context.aid,
            )
            self._active = False
            return

        self._flex_answers = []
        self._flex_expected = len(group_leaders)

        logger.info(
            "[%s] CP triggered: asking %d group leaders for flex",
            self.context.aid, len(group_leaders),
        )
        msg = AskForAvailableFlex(include_connectors=False)
        for addr in group_leaders:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_flex_answer(
        self, message: AvailableFlexAnswer, meta: dict
    ) -> None:
        # Route to the right ADMM driver based on which flow opened the
        # collection.  ``_l3_active`` is set by ``trigger_multi_sector_l3``;
        # ``_active`` is set by the legacy ``trigger_cp_negotiation``.
        # Mutually exclusive by design — both guards reject re-entry.
        if not self._l3_active and not self._active:
            return
        self._flex_answers.append(message)
        if len(self._flex_answers) < self._flex_expected:
            return
        if self._l3_active:
            await self._run_multi_sector_admm()
        else:
            await self._run_admm()

    async def _run_admm(self) -> None:
        answers = self._flex_answers[:]
        self._flex_answers = []
        self._flex_expected = 0

        agg = aggregate_flex_answers(answers)
        imbalance_by_sector = agg.imbalance_by_sector
        unmet_by_sector_total = agg.unmet_by_sector_total

        # Combine balance + unsigned unmet into T so a sector whose loads are
        # disconnected (balance≈0) is still represented as a positive deficit.
        imb_el = (
            imbalance_by_sector.get(Sector.ELECTRICITY, 0.0)
            + unmet_by_sector_total.get(Sector.ELECTRICITY, 0.0)
        )
        imb_heat = (
            imbalance_by_sector.get(Sector.HEAT, 0.0)
            + unmet_by_sector_total.get(Sector.HEAT, 0.0)
        )
        imb_gas = (
            imbalance_by_sector.get(Sector.GAS, 0.0)
            + unmet_by_sector_total.get(Sector.GAS, 0.0)
        )

        # Imbalances arrive in natural sector units (el MW, heat MW, gas kg/s);
        # ADMM lives in MW across all dimensions, so only gas needs conversion.
        T = np.array([imb_el, imb_heat, kgps_to_mw(imb_gas)])

        if np.all(T >= 0) or np.all(T <= 0):
            self._same_sign_skip_count += 1
            # Log first skip and every 10th thereafter at INFO, without flooding.
            if (
                self._same_sign_skip_count == 1
                or self._same_sign_skip_count % 10 == 0
            ):
                logger.info(
                    "[%s] CP ADMM skipped (same-sign T=%s, n=%d)",
                    self.context.aid,
                    T.tolist(),
                    self._same_sign_skip_count,
                )
            # Record the wake-up even though ADMM bailed before publishing, so
            # CP engagement isn't undercounted on structurally same-sign grids.
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
            await start_coordinated_optimization(
                [self.flex_actor], coordinator, start_msg
            )
            result = self._clamp_to_envelope(list(self.flex_actor.x))
            # _apply_result must run before emit_event: nothing subscribes to
            # OptimizationFinishedLocalEvent so emit_event raises KeyError, which
            # would otherwise discard the result via the outer except.
            applied_factor = self._apply_result(result)
            try:
                self.context.emit_event(OptimizationFinishedLocalEvent(result=result))
            except KeyError:
                pass
            group_leaders = self._live_connectors(topology_connectors(self, tid="cps"))
            # Publish CPSetpoint alongside StartBalanceNegotiation so subscribed
            # holons can re-evaluate directly. Layout per _RESULT_INDEX
            # [0=EL_MW, 1=HEAT_MW, 2=GAS_MW].
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
                regulation_factor=float(applied_factor or 1.0),
            )
            record_event(
                t=float(self.context.current_timestamp),
                kind="cp_setpoint",
                aid=str(self.context.aid),
                sector="cp",
                detail=(
                    f"flows={{{', '.join(f'{s}: {v:.4f}' for s, v in sector_flows.items())}}} "
                    f"reg={float(applied_factor or 1.0):.3f} "
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
            # Fallback: trigger intra-group gossip so groups still rebalance
            # locally even though cross-sector optimisation failed.
            for addr in self._live_connectors(topology_connectors(self, tid="cps")):
                await self.context.send_message(
                    StartBalanceNegotiation(), receiver_addr=addr
                )

        self._active = False

    # ------------------------------------------------------------------
    # Multi-sector L3 driver (Option B)
    # ------------------------------------------------------------------

    async def trigger_multi_sector_l3(self) -> None:
        """L3-coord entrypoint for the joint multi-sector ADMM round.

        Only the lex-smallest CP aid in the component runs; other CPs early-out
        and wait for the coord's CPAllocation broadcast. Flow: collect flex from
        every leader in the component, wait for replies via _handle_flex_answer,
        run the supply-priority kernel scoped to the component, then broadcast a
        per-CP CPAllocation (each CP applies via _apply_result).

        Falls back to trigger_cp_negotiation when the multi-sector wiring is
        unavailable.
        """
        if not self._multi_sector_l3_enabled():
            await self.trigger_cp_negotiation()
            return
        # Deliberately NOT gated on cps-cluster-leader: the L3 coord identity is
        # the lex-smallest aid in the multi-sector component (see
        # _is_l3_coordinator), independent of cps-cluster leadership. With CPs
        # scattered across clusters the coord often isn't the cluster leader, so
        # gating here would leave L3 silent while L2 defers.
        if not self._is_l3_coordinator():
            # Non-coord: wait for the coord's CPAllocation broadcast.
            return
        if self._l3_active or self._active:
            return

        component_leaders = self._leader_addrs_in_component()
        # Flatten {sector: {aid: addr}} -> list[addr], deduped by aid (a leader
        # spanning multiple sectors appears once per sector).
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
        self._flex_answers = []
        self._flex_expected = len(all_addrs)

        logger.info(
            "[%s] L3 coord triggered: asking %d leaders in MS component for flex",
            self.context.aid, len(all_addrs),
        )
        msg = AskForAvailableFlex(include_connectors=False)
        for addr in all_addrs:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _run_multi_sector_admm(self) -> None:
        """Run the joint multi-sector supply-priority ADMM over the collected
        leader flex answers and dispatch the result. Called by
        _handle_flex_answer once the L3-coord path has the expected replies;
        distinct from _run_admm (legacy per-CP).
        """
        try:
            await self._run_multi_sector_admm_inner()
        finally:
            self._l3_active = False

    async def _run_multi_sector_admm_inner(self) -> None:
        answers = self._flex_answers[:]
        self._flex_answers = []
        self._flex_expected = 0
        if not answers:
            return

        # Build (supply, demand) actor lists from the leader replies, one actor
        # per leader. The supply-priority ADMM is sector-agnostic, so passing
        # every sector touched by any leader yields the joint multi-sector solve.
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

        sectors = sorted({
            s for d in actor_demands for s in d
        })
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
            float(v) for d in actor_demands
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
                "[%s] multi-sector L3 ADMM failed: %s", self.context.aid, exc,
            )
            record_event(
                t=float(self.context.current_timestamp),
                kind="l3_admm_failed",
                aid=str(self.context.aid),
                sector="cp",
                detail=f"multi_sector: {exc}",
            )
            return

        # Numeric noise scrub: the ADMM leaves ~1e-3-scale residuals on cells it
        # can't serve, and their cross-cell ordering isn't priority-monotone at
        # that noise level. Clamping fractions below the PI claim tolerance to 0
        # restores priority-monotonicity without touching genuine partial service.
        _FRACTION_NOISE = 1e-3
        for sec_key in service_fraction:
            for tier, frac in list(service_fraction[sec_key].items()):
                if 0.0 < float(frac) < _FRACTION_NOISE:
                    service_fraction[sec_key][tier] = 0.0

        logger.info(
            "[%s] L3 multi-sector ADMM result: round=%s sectors=%s tiers=%s "
            "n_leaders=%d fractions=%s",
            self.context.aid, round_id, sectors, tiers, len(answers),
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

        # L3 does NOT dispatch ComponentAllocation to leaders: per-sector
        # priority allocation is L2's job (it refines per-tier fractions on the
        # post-CP state in parallel). L3's sole output is CP setpoints; the ADMM
        # result here only feeds the per-sector marginal values below.

        # Compute per-sector marginal values and dispatch a per-CP setpoint to
        # every CP in the component. marginal(sector) = 1 - min served fraction
        # over tiers with positive demand: 0 = fully served, near 1 = stressed.
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

        # Wake L2 in every sector the ADMM touched: CPs just committed new
        # setpoints, so post-commit LP routing changes leader flex next gossip
        # pass. L3RebalanceWakeup flags the leader's _rebalance_dirty so its L2
        # short-circuit lifts. Pure kick message (sector filter only), no dispatch.
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

    async def _handle_cp_allocation(
        self, message: CPAllocation, meta: dict
    ) -> None:
        """Apply a setpoint dispatched by the L3 coord via _apply_result so the
        regulate ledger, diagnostics and downstream LP see the same path as the
        legacy per-CP ADMM. Idempotent: apply_regulate dedups same-value writes.
        """
        if topology_characteristic(self, tid="cps") != "leader":
            return
        flows_mw = dict(message.sector_flows_mw)
        # Translate to the flat [el, heat, gas] vector _apply_result expects;
        # missing sectors stay at 0.
        result = [0.0, 0.0, 0.0]
        for sec, idx in _RESULT_INDEX.items():
            v = flows_mw.get(sec.value)
            if v is not None:
                result[idx] = float(v)
        self._last_l3_setpoint_by_sector = {
            sec.value: float(result[idx]) for sec, idx in _RESULT_INDEX.items()
        }
        applied_factor = self._apply_result(result)
        record_event(
            t=float(self.context.current_timestamp),
            kind="cp_setpoint",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"source=l3 round={message.round_id} flows={flows_mw} "
                f"reg={float(applied_factor or 1.0):.3f}"
            ),
        )

    def _apply_result(self, result: list[float]) -> float | None:
        obs = self.behavior.observe(self.context.aid) or {}
        # result layout [0=EL, 1=HEAT, 2=GAS]. A CP has one regulation knob:
        # compute a factor per sector and apply the strongest-signal one (largest
        # |value| after clamping) so the most-demanded sector drives the setpoint.
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
            weight = abs(value)
            if weight > best_weight:
                best_weight = weight
                best_factor = factor

        if best_factor is not None:
            apply_regulate(
                self.behavior,
                self.context.aid,
                best_factor,
                sector="cp",
                reason="cp_admm",
                timestamp=self.context.current_timestamp,
            )
        return best_factor


class MultiCommunityCPRole(Role):
    """CP-side coordination role for the ``component_level`` baseline.

    The baseline forms one community per connected component of each per-sector
    subgraph and exposes the CP as a connector to each community it bridges (via
    the same cps↔groups cross-link the SCARe CP-ADMM path uses). Group leaders
    dispatch :class:`NegotiationFinishedEvent` after each local rebalance, but
    with the holonic + cross-sector ADMM layers off there is no single
    coordinator — each community contributes an independent ask.

    This role reconciles those asks under an EMA-blended target with a deadband
    + cooldown so the CP (member of two communities at once) cannot ping-pong
    between contradictory commits. It replaces the EnergyConverterRole pipeline
    (no ADMM, no flex-actor) and keeps only the minimal AskEnergyMessage reply.

    State: ``_target_by_sector`` is the per-sector EMA target updated on each
    event (target ← α·new + (1−α)·target); ``_committed_by_sector`` is the last
    setpoint committed via apply_regulate (a commit fires only when
    |target−committed| > deadband and the cooldown has elapsed);
    ``_last_commit_t`` gates that cooldown. The EMA is reset on every
    :class:`BranchFailureEvent` (wired to :meth:`on_branch_failure`) so a failure
    islanding a bridged community doesn't leave stale signal averaged in;
    ``_committed_*`` is retained since the physical setpoint doesn't move on the
    event itself.
    """

    def __init__(
        self,
        behavior: "RestorationEnvironmentBehavior",
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
        # Per-sector smoothed target. Seeded lazily to the first observed
        # setpoint so the EMA doesn't start at 0 (which biases the CP toward off).
        self._target_by_sector: dict[Sector, float] = {}
        # Last committed setpoint per sector; the deadband check anchors against
        # it so small drift doesn't generate spurious regulate calls.
        self._committed_by_sector: dict[Sector, float] = {}
        # Sim-time of the last apply_regulate call; cooldown gate.
        self._last_commit_t: float = -1e9

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        # Reply minimally to community gossip's flex-query so a leader treating
        # this CP as a connector doesn't stall. available=0: the CP brings no
        # spare flex of its own, only the cross-sector conversion knob.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_energy),
            lambda msg, meta: isinstance(msg, AskEnergyMessage),
        )
        # Per-community signal: on each gossip convergence the community leader
        # broadcasts NegotiationFinishedEvent with its settled setpoint, which we
        # treat as that community's ask of the CP.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_finished),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        # on_branch_failure is wired by the scenario builder via behavior_in on
        # BranchFailureEvent. CPs aren't community members in the groups topology
        # (they bridge via the cross-link), so the RepartitionHandlerRole's
        # CommunityReassignedEvent never reaches them — the branch-failure global
        # event is the only signal landing on every CP, which the safety reset needs.

    async def _handle_ask_energy(
        self, message: AskEnergyMessage, meta: dict
    ) -> None:
        try:
            obs = self.behavior.observe(self.context.aid) or {}
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
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

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
        # would chase 0 → proposed across several rounds).
        if sector not in self._target_by_sector:
            self._target_by_sector[sector] = proposed
        else:
            alpha = self._ema_alpha
            self._target_by_sector[sector] = (
                alpha * proposed
                + (1.0 - alpha) * self._target_by_sector[sector]
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

        We don't try to decide whether *this* failure islands a bridged
        community (that would need the SCARe physical-graph mirror); the EMA
        re-seeds within a few post-failure events anyway, so a spurious reset is
        cheap. ``_committed_by_sector`` and the cooldown are retained because the
        physical setpoint doesn't move on the failure, so the deadband still
        anchors against the live operating point.
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
