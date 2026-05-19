from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.express.topology import (
    topology_characteristic,
    topology_connectors,
)

from scare.base.channel import (
    CPSetpoint,
    HolonAllocation,
    MonotonicVersion,
    SectorImbalanceUpdate,
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
from scare.base.util import clamp_to_constraints, kgps_to_mw, mw_to_kgps, obs_setpoint

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

# --- CP ↔ sector fixed-point tolerance ---
# A NegotiationFinishedEvent re-triggers CP ADMM only if the sector's
# new setpoint differs from the last one observed here by more than
# this tolerance.  Below the tolerance the loop has reached a
# fixed-point on the CP side and re-triggering would just ping-pong
# back to the group and waste messages.  Units match _ACCESS_KEYS
# (MW for electricity, kg/s for gas, W for heat).
_CP_SETPOINT_TOLERANCE: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,   # MW
    Sector.GAS: 1e-4,           # kg/s
    Sector.HEAT: 1e-4,          # MW (~100 W)
}
_CP_DEFAULT_TOLERANCE = 0.01

# --- Predicate-driven trigger (channel/decision path) ---
# Below ``_PREDICATE_DEAD_BAND`` the cross-sector imbalance is treated
# as noise and the predicate stays False.  ``_PREDICATE_MIN_GAP_S``
# enforces a cooldown between predicate-driven fires so the role
# cannot self-thrash if two beacons publish in rapid succession.
_PREDICATE_DEAD_BAND_MW: float = 1e-4
_PREDICATE_MIN_GAP_S: float = 1.0


class EnergyConverterRole(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        flex_actor: ADMMFlexActor,
        sectors: list[Sector],
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.flex_actor = flex_actor
        self.sectors = sectors

        self._active: bool = False
        self._flex_answers: list[AvailableFlexAnswer] = []
        self._flex_expected: int = 0

        # Per-sector last-observed group setpoint; used to suppress
        # re-triggering CP ADMM when the balance negotiation converged
        # to essentially the same point as before (fixed-point gate).
        self._last_sector_setpoint: dict[Sector, float] = {}

        # Skip-count throttle: ADMM may decline to run because the
        # imbalance vector has the same sign across all sectors (no
        # cross-sector trade improves the objective).  Log every Nth
        # such skip at INFO so it is visible without flooding.
        self._same_sign_skip_count: int = 0

        # --- Predicate path state ---
        # Per-publisher version memory for SectorImbalanceUpdate so we
        # don't re-evaluate the predicate against stale beacon decisions.
        self._seen_beacons = SeenVersions()
        # Latest signed imbalance per (publisher, sector).  Aggregated
        # by sector in ``_predicate_inputs`` when the predicate runs.
        self._beacon_imbalance: dict[tuple[str, Sector], float] = {}
        self._last_predicate_fire_t: float = -1e9

        # --- L2 -> L3 channel state ---
        # Same shape as the beacon path but for ``HolonAllocation``: a
        # holon committed a cross-sector setpoint shift; we may want
        # to fire CP ADMM directly without waiting for the downstream
        # gossip to materialise.
        self._seen_holon_alloc = SeenVersions()
        self._holon_alloc_signal: dict[tuple[str, Sector], float] = {}
        self._last_holon_predicate_fire_t: float = -1e9

        # --- L3 publishing identity ---
        self._cp_version = MonotonicVersion()

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        logger.debug(
            "[%s] EnergyConverterRole setup: sectors=%s",
            self.context.aid, [s.value for s in self.sectors],
        )
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
        # Predicate path: receive sector-imbalance beacons from group
        # leaders.  This runs *alongside* the legacy NegotiationFinishedEvent
        # path — both can trigger ADMM independently, and the existing
        # ``self._active`` guard prevents concurrent runs.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_sector_imbalance),
            lambda msg, meta: isinstance(msg, SectorImbalanceUpdate),
        )
        # Direct L2 -> L3 trigger.  When a holon commits a per-member
        # allocation that creates cross-sector flow, the CP can decide
        # to engage before the L1 gossip resolves the new targets.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_holon_allocation),
            lambda msg, meta: isinstance(msg, HolonAllocation),
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
        # CP agents report available=0: they have no spare flex of their own
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=value,
            available=0.0,
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

    async def _handle_sector_imbalance(
        self, message: SectorImbalanceUpdate, meta: dict
    ) -> None:
        """Predicate-driven trigger path (channel/decision design).

        Updates the per-publisher imbalance memory, then evaluates the
        trigger predicate over the aggregated sector vector.  Fires
        ``trigger_cp_negotiation`` independently of whether L1 gossip
        has finished — the missing path that left L3 silent on
        ``simbench_lv_cp_heavy`` with the holon layer disabled.
        """
        # Only the CP leader runs ADMM.  Non-leaders silently drop
        # beacons; they exist for symmetry with the legacy path.
        char = topology_characteristic(self, tid="cps")
        logger.debug(
            "[%s] CP received beacon: sector=%s imb=%.5f v=%d from %s char=%s",
            self.context.aid, message.sector.value, message.local_imbalance_mw,
            message.version, message.publisher, char,
        )
        if char != "leader":
            return
        # Echo / staleness guard.  ``caused_by[my_aid] == latest`` means
        # the beacon is reporting on state we just published; skip it.
        my_latest = self._seen_beacons.latest(self.context.aid)
        if message.caused_by.get(self.context.aid, -1) == my_latest and my_latest >= 0:
            return
        if not self._seen_beacons.is_fresh(message.publisher, message.version):
            return

        self._beacon_imbalance[(message.publisher, message.sector)] = (
            float(message.local_imbalance_mw)
        )
        self._seen_beacons.mark(message.publisher, message.version)

        if self._active:
            return

        T = self._predicate_inputs()
        if not self._predicate_should_run(T):
            return

        now = float(self.context.current_timestamp)
        if now - self._last_predicate_fire_t < _PREDICATE_MIN_GAP_S:
            return
        self._last_predicate_fire_t = now

        logger.info(
            "[%s] CP predicate fired: T=%s (publisher=%s v=%d)",
            self.context.aid,
            {s.value: round(v, 4) for s, v in T.items()},
            message.publisher,
            message.version,
        )
        self.context.schedule_instant_task(self.trigger_cp_negotiation())

    def _predicate_inputs(self) -> dict[Sector, float]:
        """Aggregate the latest per-publisher imbalances by sector.

        A publisher only contributes its most-recent value per sector
        (the dict is keyed by ``(publisher, sector)``).  Sums across
        publishers give the sector-level signal the predicate compares
        across sectors.
        """
        by_sector: dict[Sector, float] = {}
        for (_, sec), v in self._beacon_imbalance.items():
            if sec in self.sectors:
                by_sector[sec] = by_sector.get(sec, 0.0) + v
        return by_sector

    def _predicate_should_run(self, T: dict[Sector, float]) -> bool:
        """Predicate is a *wake-up hint*, not an ADMM-feasibility test.

        The historical same-sign-skip lives inside ``_run_admm`` (which
        runs its own ``AskForAvailableFlex`` round after we trigger),
        so the predicate doesn't need to duplicate it.  Beacons only
        cover sectors whose group leaders had a CP connector
        registered (and mango's per-agent single-conn_type rule means
        multi-sector CPs only show up to one sector's leaders), which
        would otherwise make the cross-sector check unreachable.

        The predicate just asks: did any subscribed sector report
        stress beyond the dead-band?  If yes, wake the CP up; its
        collection round will gather the real cross-sector picture.
        """
        if not T:
            return False
        return any(abs(v) >= _PREDICATE_DEAD_BAND_MW for v in T.values())

    async def _handle_holon_allocation(
        self, message: HolonAllocation, meta: dict
    ) -> None:
        """Direct L2 -> L3 trigger via the channel/decision pattern.

        A holon just published its per-member ADMM allocation.  The
        magnitudes of those targets, signed in load convention, are a
        leading indicator that the affected groups are about to shift
        their sector balance — CP can engage now rather than waiting
        for the gossip chain to resolve the new targets and broadcast
        ``NegotiationFinishedEvent``.

        Aggregation reuses the same dictionary the beacon path fills,
        keyed by ``(publisher, sector)``, so the predicate sees a
        unified view across both channels.
        """
        char = topology_characteristic(self, tid="cps")
        logger.debug(
            "[%s] CP received holon-allocation: sector=%s n_targets=%d v=%d from %s char=%s",
            self.context.aid, message.sector.value,
            len(message.targets_mw), message.version, message.publisher, char,
        )
        if char != "leader":
            return
        # Echo guard: a holon allocation triggered by our own CP
        # setpoint isn't fresh news to us.
        if (
            message.caused_by.get(self.context.aid, -1) == self._cp_version.current
            and self._cp_version.current > 0
        ):
            return
        if not self._seen_holon_alloc.is_fresh(message.publisher, message.version):
            return

        # Use the aggregate target magnitude as the per-(publisher, sector)
        # signal.  Sum of |targets_mw| captures the holon's intent to
        # rebalance regardless of how the shed is distributed across
        # members; sign isn't meaningful here because L2's per-member
        # allocations can cancel out within a sector.
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
        self.context.schedule_instant_task(self.trigger_cp_negotiation())

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
        # Fixed-point gate: skip re-trigger if this sector's group
        # setpoint has not moved enough to change the ADMM answer.
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
        self.context.schedule_instant_task(self.trigger_cp_negotiation())

    async def trigger_cp_negotiation(self) -> None:
        if topology_characteristic(self, tid="cps") != "leader":
            return
        if self._active:
            return
        self._active = True

        group_leaders = topology_connectors(self, tid="cps")
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
        if not self._active:
            return

        self._flex_answers.append(message)

        if len(self._flex_answers) >= self._flex_expected:
            await self._run_admm()

    async def _run_admm(self) -> None:
        import numpy as np
        from distributed_resource_optimization import (
            create_admm_sharing_data,
            create_admm_start,
            create_sharing_target_distance_admm_coordinator,
            start_coordinated_optimization,
        )

        from scare.base.util import aggregate_priority_weight

        answers = self._flex_answers[:]
        self._flex_answers = []
        self._flex_expected = 0

        # Per-sector aggregation.  Two channels:
        # * ``balance_by_sector`` — net signed setpoint (generation + load
        #   in load-convention).  Reflects what's actually flowing.
        # * ``unmet_by_sector`` — load that the LP could not deliver
        #   (regulation forced to 0 by monee's disconnect handling).
        #   Without this, a sector that loses all its loads to physical
        #   disconnect reports ``balance = 0`` and the CP layer treats it
        #   as balanced — exactly when CP help is most needed.  Adding
        #   ``unmet`` shifts the imbalance toward the deficit side so
        #   ``T`` carries a real positive entry for the disconnected
        #   sector and ADMM can find a cross-sector shift instead of
        #   skipping with ``same-sign T``.
        imbalance_by_sector: dict[Sector, float] = {}
        unmet_by_sector_total: dict[Sector, float] = {}
        sector_priority_weight: dict[Sector, float] = {}
        for answer in answers:
            imbalance_by_sector[answer.sector] = (
                imbalance_by_sector.get(answer.sector, 0.0) + answer.balance
            )
            for sec_str, val in (getattr(answer, "unmet_by_sector", {}) or {}).items():
                try:
                    sec_enum = Sector(sec_str)
                except ValueError:
                    continue
                unmet_by_sector_total[sec_enum] = (
                    unmet_by_sector_total.get(sec_enum, 0.0) + float(val)
                )
            w = aggregate_priority_weight(
                answer.demand_by_priority, answer.served_by_priority
            )
            sector_priority_weight[answer.sector] = (
                sector_priority_weight.get(answer.sector, 0.0) + w
            )

        # Combine balance + unmet into the T vector.  Unmet is unsigned
        # (always positive deficit) so it shifts T toward positive in
        # sectors with disconnected loads.
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

        # Imbalances are now reported in their natural sector units —
        # electricity in MW, heat in MW (was W), gas in kg/s.  ADMM
        # lives in MW across all dimensions, so only gas needs unit
        # conversion; the historical /1e6 on heat is no longer correct
        # since monee switched ``q_w_heat`` → ``q_mw_heat``.
        T = np.array([imb_el, imb_heat, kgps_to_mw(imb_gas)])

        if np.all(T >= 0) or np.all(T <= 0):
            self._same_sign_skip_count += 1
            # First skip per leader and every 10th thereafter at INFO so
            # the suppressed coordination is visible without log flooding.
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
            self._active = False
            return

        # Per-sector priority weights for the ADMM sharing problem.
        # Sectors with higher-priority unserved demand get stronger
        # weight, biasing the CP operating point toward those sectors.
        w_el = sector_priority_weight.get(Sector.ELECTRICITY, 1.0)
        w_heat = sector_priority_weight.get(Sector.HEAT, 1.0)
        w_gas = sector_priority_weight.get(Sector.GAS, 1.0)
        w_max = max(w_el, w_heat, w_gas, 1e-9)
        priorities = np.array([w_el, w_heat, w_gas]) / w_max  # normalise to [0, 1]
        priorities = np.maximum(priorities, 0.01)  # floor to avoid zero-weight

        coordinator = create_sharing_target_distance_admm_coordinator()
        start_msg = create_admm_start(
            create_admm_sharing_data(T.tolist(), priorities=priorities.tolist())
        )

        try:
            await start_coordinated_optimization(
                [self.flex_actor], coordinator, start_msg
            )
            result = list(self.flex_actor.x)
            logger.info("[%s] ADMM result: %s", self.context.aid, result)
            # ``_apply_result`` must run *before* ``emit_event``: nothing
            # subscribes to ``OptimizationFinishedLocalEvent``, so the
            # underlying ``RoleHandler.emit_event`` raises ``KeyError``
            # on its dict lookup.  Previously that KeyError was caught by
            # the outer ``except Exception`` and the ADMM result was
            # discarded — the CP layer was diagnostic-only.
            applied_factor = self._apply_result(result)
            try:
                self.context.emit_event(OptimizationFinishedLocalEvent(result=result))
            except KeyError:
                pass
            group_leaders = topology_connectors(self, tid="cps")
            # Publish CPSetpoint Decision (channel/decision pattern)
            # alongside the legacy StartBalanceNegotiation so subscribed
            # holons can re-evaluate directly.  Result layout matches
            # _RESULT_INDEX: [0=EL_MW, 1=HEAT_MW, 2=GAS_MW].
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
            # Fallback: trigger intra-group gossip so groups can still
            # rebalance locally even though cross-sector optimisation failed.
            for addr in topology_connectors(self, tid="cps"):
                await self.context.send_message(
                    StartBalanceNegotiation(), receiver_addr=addr
                )

        self._active = False

    def _apply_result(self, result: list[float]) -> float | None:
        obs = self.behavior.observe(self.context.aid) or {}
        # result layout: [0=EL, 1=HEAT, 2=GAS]
        # A CP has a single regulation knob.  Compute a factor per sector
        # and apply the one with the strongest signal (largest |value/cap|
        # ratio after clamping).  This ensures the most constrained or
        # most demanded sector drives the operating point.
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
            from scare.base.util import apply_regulate

            apply_regulate(
                self.behavior,
                self.context.aid,
                best_factor,
                sector="cp",
                reason="cp_admm",
                timestamp=self.context.current_timestamp,
            )
        return best_factor
