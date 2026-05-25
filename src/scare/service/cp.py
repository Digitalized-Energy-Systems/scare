from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, NamedTuple

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
    aggregate_priority_weight,
    apply_regulate,
    clamp_to_constraints,
    kgps_to_mw,
    mw_to_kgps,
    obs_setpoint,
    tier_priority_weight_strict,
)
from scare.community.supply_priority_admm import allocate_supply_priority

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

# --- Reactive-trigger noise filter (used by HolonAllocation path) ---
# Below ``_PREDICATE_DEAD_BAND`` the L2-dispatched cross-sector signal
# is treated as noise and the trigger is suppressed.  ``_PREDICATE_MIN_GAP_S``
# enforces a cooldown so a burst of L2 dispatches cannot self-thrash
# the CP.  Both were previously also used by the deprecated beacon-
# driven ``_handle_sector_imbalance`` path; that path has been removed
# in favour of purely event-driven CP triggering.
_PREDICATE_DEAD_BAND_MW: float = 1e-4
_PREDICATE_MIN_GAP_S: float = 1.0


class _FlexAggregate(NamedTuple):
    """Per-sector aggregation of a batch of ``AvailableFlexAnswer``."""
    imbalance_by_sector: dict[Sector, float]
    unmet_by_sector_total: dict[Sector, float]
    sector_priority_weight: dict[Sector, float]
    top_unmet_tier_per_sector: dict[Sector, int]
    top_unmet_mag_per_sector: dict[Sector, float]


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
        # Concept C — Layer 3 dynamic topology.  Optional sibling role
        # (:class:`DynamicConnectorRole`) that classifies which group
        # leaders the CP can still physically reach.  ``None`` keeps
        # the legacy static-topology behaviour: every connector
        # returned by ``topology_connectors`` is admitted.
        self._live_connector_filter = live_connector_filter
        # Cross-sector coalition advertisement: per-(in_sector,
        # out_sector) efficiency.  Used when this CP joins a coalition
        # to tell the initiator how its output supply maps to input
        # draw.  ``None`` ⇒ CP doesn't advertise coupling (legacy
        # behaviour; the cross-sector coalition path will skip it).
        self._coupling_ratios: dict[tuple[str, str], float] = (
            dict(coupling_ratios) if coupling_ratios else {}
        )
        # Active envelope from a cross-sector coalition commitment.
        # ``None`` ⇒ no envelope (CP runs free).  Read by ``_run_admm``
        # to clamp per-sector bounds; written by ``_handle_cp_commitment``.
        self._envelope_flows_mw: dict[Sector, float] | None = None
        self._envelope_expires_at: float = -1.0
        self._envelope_coalition_id: str = ""

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

        # --- Multi-sector L3 (Option B) wiring ---
        # Injected via ``wire_multi_sector_l3`` after world construction
        # so the topology mirror + the CP meta table (which require the
        # full world graph) can reach the role.  ``None`` until wired —
        # legacy per-CP path runs in that case.
        self._topology_mirror: Any = None
        self._my_node_id: Any = None
        # All CPs in the world: ``{cp_aid: {"sectors": [Sector, ...],
        # "capacity_mw": float, "coupling_ratios": dict[(in, out), float],
        # "addr": Address, "node_id": Any}}``.  The L3 coord uses this
        # to broadcast :class:`CPAllocation` to other CPs in its multi-
        # sector component.
        self._cp_meta_by_aid: dict[str, dict[str, Any]] = {}
        # Per-sector ``{leader_aid: addr}`` for every group leader; the
        # L3 coord filters this through ``topology_mirror.reachable_from``
        # to find the leaders in its multi-sector component.
        self._leader_addrs_by_sector: dict[Sector, dict[str, Any]] = {}
        # ``{leader_aid: node_id}`` so the L3 coord can resolve which
        # leaders are reachable from its own node.
        self._leader_node_ids: dict[str, Any] = {}

        # --- Multi-sector L3 coordinator runtime state ---
        # Latest CPAllocation result this CP applied (for warm-start
        # tracking and to suppress no-op re-dispatches).
        self._last_l3_setpoint_by_sector: dict[str, float] = {}
        # Round counter for the multi-sector ADMM the L3 coord drives.
        self._l3_round_counter: int = 0
        # Set to True while the L3 coord's collect→solve→dispatch cycle
        # is in flight, so reactive triggers coalesce instead of
        # stacking.  Distinct from ``_active`` (which guards the
        # legacy per-CP ADMM path).
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
        """Inject the post-construction state the multi-sector L3 path
        needs to elect a coordinator and drive a joint multi-sector
        ADMM.  Called by ``scenario.restoration`` after the topology
        mirror + every CP agent have been built; pre-call the role
        behaves as legacy per-CP.

        Parameters
        ----------
        topology_mirror
            Shared :class:`TopologyMirror` instance.  Same instance the
            community + holonic roles consult, so failures propagate
            consistently across L2 and L3.
        my_node_id
            This CP agent's host node id in the monee graph.
        cp_meta_by_aid
            ``{cp_aid: {sectors, capacity_mw, coupling_ratios, addr,
            node_id}}`` for every CP in the world.  L3 coord
            broadcasts to other CPs via this map; non-coord CPs apply
            allocations they receive.
        leader_addrs_by_sector
            ``{Sector: {leader_aid: addr}}`` for every group leader.
            L3 coord filters this through the mirror to find leaders
            in its multi-sector component.
        leader_node_ids
            ``{leader_aid: node_id}`` for reachability filtering.
        """
        self._topology_mirror = topology_mirror
        self._my_node_id = my_node_id
        self._cp_meta_by_aid = dict(cp_meta_by_aid)
        self._leader_addrs_by_sector = dict(leader_addrs_by_sector)
        self._leader_node_ids = dict(leader_node_ids)

    def _multi_sector_l3_enabled(self) -> bool:
        """True iff this CP has the runtime state needed to drive the
        multi-sector L3 path.  Falls back to the legacy per-CP path
        when False (e.g. tests that build the role directly without
        wiring the mirror, or campaigns that disable enable_cp_admm).
        """
        return (
            self._topology_mirror is not None
            and self._my_node_id is not None
            and bool(self._cp_meta_by_aid)
        )

    def _multi_sector_component_reachable(self) -> set:
        """Return the set of node ids in this CP's multi-sector
        connected component — i.e. nodes mutually reachable on the
        active branch subgraph AND through active CP bridges.  Used
        to scope both leader-flex collection and CP-allocation
        dispatch to the right physical island.
        """
        try:
            return self._topology_mirror.reachable_from(
                self._my_node_id, sector=None, allow_cp_bridges=True,
            )
        except Exception as exc:
            logger.debug(
                "[%s] multi-sector reachable_from failed: %s",
                self.context.aid, exc,
            )
            return {self._my_node_id}

    def _cp_peers_in_component(self) -> dict[str, dict[str, Any]]:
        """Return ``{cp_aid: meta}`` for every CP whose host node is
        in this CP's multi-sector component.  Always includes self
        (so the lex-smallest aid is well-defined even for a singleton
        component).
        """
        reachable = self._multi_sector_component_reachable()
        out: dict[str, dict[str, Any]] = {}
        for aid, meta in self._cp_meta_by_aid.items():
            node = meta.get("node_id")
            if aid == self.context.aid or node is None or node in reachable:
                out[aid] = meta
        # Defensive: always self.
        if self.context.aid not in out and self.context.aid in self._cp_meta_by_aid:
            out[self.context.aid] = self._cp_meta_by_aid[self.context.aid]
        return out

    def _leader_addrs_in_component(self) -> dict[Sector, dict[str, Any]]:
        """Return ``{Sector: {leader_aid: addr}}`` filtered to leaders
        whose host node is reachable from this CP on the active
        multi-sector subgraph.  This is what the L3 coord asks for
        flex; identical to L2's per-component peer set logic but
        widened to span every sector touched by the multi-sector
        component.
        """
        reachable = self._multi_sector_component_reachable()
        out: dict[Sector, dict[str, Any]] = {}
        for sector, table in self._leader_addrs_by_sector.items():
            sec_out: dict[str, Any] = {}
            for aid, addr in table.items():
                node = self._leader_node_ids.get(aid)
                if node is not None and node in reachable:
                    sec_out[aid] = addr
            if sec_out:
                out[sector] = sec_out
        return out

    def _schedule_trigger(self) -> None:
        """Route the reactive trigger to the multi-sector L3 path when
        wired, else to the legacy per-CP path.  Used by the predicate /
        beacon / holon-allocation / coalition handlers that previously
        always scheduled ``trigger_cp_negotiation``.
        """
        if self._multi_sector_l3_enabled():
            self.context.schedule_instant_task(self.trigger_multi_sector_l3())
        else:
            self.context.schedule_instant_task(self.trigger_cp_negotiation())

    def _is_l3_coordinator(self) -> bool:
        """True iff this CP has the lex-smallest aid among CPs in its
        multi-sector component.  The L3 coord drives the joint ADMM;
        other CPs in the same component wait for ``CPAllocation``.

        Defensive: if this CP isn't in the cp_meta table (the wiring
        was incomplete), defer to legacy per-CP and return True only
        when this CP is the trivially-lex-smallest of {self}.
        """
        peers = self._cp_peers_in_component()
        if not peers:
            return True
        return min(peers) == self.context.aid

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        logger.debug(
            "[%s] EnergyConverterRole setup: sectors=%s",
            self.context.aid, [s.value for s in self.sectors],
        )
        # CP triggering is purely event-driven.  The role wakes up only
        # when an actual cross-sector decision input arrives:
        #   - NegotiationFinishedEvent (L1 gossip converged on a new
        #     group setpoint)
        #   - HolonAllocation (L2 holon dispatched a fresh per-member plan)
        #   - CPCommitment (L2.5 coalition issued a directional envelope)
        #   - CPAllocation (multi-sector L3 coord assigned this CP)
        # The previous design also ran a periodic ``_heartbeat_l3`` and
        # subscribed to ``SectorImbalanceUpdate`` beacons; both were
        # implicitly periodic (heartbeat by timer, beacon by publisher
        # cadence) and could fire ADMM even when no upstream state had
        # changed.  They have been removed — the CP only acts on new
        # information now.
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
        # Direct L2 -> L3 trigger.  When a holon commits a per-member
        # allocation that creates cross-sector flow, the CP can decide
        # to engage before the L1 gossip resolves the new targets.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_holon_allocation),
            lambda msg, meta: isinstance(msg, HolonAllocation),
        )
        # Cross-sector coalition envelope.  L2.5 dispatches a
        # CPCommitment per re-assert tick while the coalition is
        # active; on receipt we narrow the CP's per-sector ADMM bounds
        # so the coalition's directional decision is honoured until
        # ttl_s expiry.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_cp_commitment),
            lambda msg, meta: isinstance(msg, CPCommitment),
        )
        # Multi-sector L3 dispatch: non-coord CPs receive their
        # per-sector flow setpoint from the elected L3 coordinator
        # and apply it via the existing regulation path.  The coord's
        # local setpoint goes through the same handler so all CPs
        # behave uniformly.
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
        # CP agents report available=0: they have no spare flex of their own
        reply = ResponseEnergyMessage(
            negotiation_id=message.negotiation_id,
            setpoint=value,
            available=0.0,
        )
        await self.context.send_message(reply, receiver_addr=mango_sender_addr(meta))

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
        # Cps-leader gate — only active in legacy single-sector mode.
        # Multi-sector L3 path uses the ``_is_l3_coordinator`` election
        # inside the dispatcher.
        if char != "leader" and not self._multi_sector_l3_enabled():
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
        self._schedule_trigger()

    async def _handle_cp_commitment(
        self, message: CPCommitment, meta: dict
    ) -> None:
        """Cross-sector coalition envelope handler.

        Writes the directional sector flows into local envelope state
        with a TTL; ``_run_admm`` reads this and clamps per-sector
        bounds so subsequent L3 rounds stay inside the coalition's
        commitment.  Idempotent: re-asserted commits just refresh the
        TTL and overwrite the flows (latest-wins semantics matches the
        coalition store).

        The commitment is only honoured if the message addresses
        *this* CP — addressed by aid in ``message.cp_id`` so a single
        coalition broadcast can include multiple CPs without each
        recipient acting on the others.
        """
        if message.cp_id and message.cp_id != str(self.context.aid):
            return
        if topology_characteristic(self, tid="cps") != "leader":
            return
        # Translate sector-value-keyed flows to Sector enums.  Unknown
        # sector strings are silently dropped — defensive against
        # forward-compat channel additions.
        flows: dict[Sector, float] = {}
        for sec_v, mw in message.target_flows_mw.items():
            try:
                flows[Sector(sec_v)] = float(mw)
            except (ValueError, TypeError):
                continue
        if not flows:
            return
        now = float(self.context.current_timestamp)
        self._envelope_flows_mw = flows
        self._envelope_expires_at = now + float(message.ttl_s)
        self._envelope_coalition_id = message.coalition_id
        logger.info(
            "[%s] CP envelope set by coalition %s: flows=%s ttl=%.2fs",
            self.context.aid, message.coalition_id,
            {s.value: round(v, 4) for s, v in flows.items()},
            float(message.ttl_s),
        )
        # Diagnostic ledger entry — picked up by ``event_log()`` so the
        # post-run analysis can plot envelope-active intervals and the
        # cumulative cross-sector transfer the coalition committed to.
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
        if self._envelope_flows_mw is None:
            return False
        if self.context.current_timestamp > self._envelope_expires_at:
            self._envelope_flows_mw = None
            return False
        return True

    def _clamp_to_envelope(self, result: list) -> list:
        """Replace each sector dimension of the ADMM result with the
        coalition-committed flow when an L2.5 envelope is active; return
        the untouched result otherwise.  Logs + records ``cp_envelope_clamp``
        when a clamp lands.
        """
        if not self._envelope_active():
            logger.info("[%s] ADMM result: %s", self.context.aid, result)
            return result
        envelope = self._envelope_flows_mw or {}
        pre_clamp = list(result)
        for sector, idx in _RESULT_INDEX.items():
            if idx >= len(result) or sector not in envelope:
                continue
            result[idx] = float(envelope[sector])
        logger.info(
            "[%s] CP ADMM result clamped to coalition envelope %s: %s",
            self.context.aid, self._envelope_coalition_id, result,
        )
        record_event(
            t=float(self.context.current_timestamp),
            kind="cp_envelope_clamp",
            aid=str(self.context.aid),
            sector="cp",
            detail=(
                f"coalition={self._envelope_coalition_id} "
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
        self._schedule_trigger()

    def _live_connectors(self, connectors: list) -> list:
        """Filter ``connectors`` through the sibling
        :class:`DynamicConnectorRole` when one is attached, otherwise
        passthrough.  Centralised so every callsite that fans flex
        requests / decisions out to group leaders honours Concept-C
        dynamics uniformly (legacy behaviour preserved when no filter
        is wired).
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

    def _aggregate_flex_answers(
        self, answers: list[AvailableFlexAnswer]
    ) -> _FlexAggregate:
        """Aggregate a batch of flex answers per-sector.

        Tracks signed balance, unsigned ``unmet`` (LP-undelivered demand
        — surfaces sectors whose disconnected loads would otherwise
        cancel against generation in ``balance``), and the most
        critical (lowest-tier) unmet (sector, tier) pair so the ADMM
        weight is top-tier-dominant rather than magnitude-weighted.
        """
        agg = _FlexAggregate({}, {}, {}, {}, {})
        for answer in answers:
            agg.imbalance_by_sector[answer.sector] = (
                agg.imbalance_by_sector.get(answer.sector, 0.0) + answer.balance
            )
            for sec_str, val in (getattr(answer, "unmet_by_sector", {}) or {}).items():
                try:
                    sec_enum = Sector(sec_str)
                except ValueError:
                    continue
                agg.unmet_by_sector_total[sec_enum] = (
                    agg.unmet_by_sector_total.get(sec_enum, 0.0) + float(val)
                )
            w = aggregate_priority_weight(
                answer.demand_by_priority, answer.served_by_priority
            )
            agg.sector_priority_weight[answer.sector] = (
                agg.sector_priority_weight.get(answer.sector, 0.0) + w
            )
            dem_map = getattr(answer, "demand_by_sector_priority", {}) or {}
            srv_map = getattr(answer, "served_by_sector_priority", {}) or {}
            for sec_str, tier_to_dem in dem_map.items():
                try:
                    sec_enum = Sector(sec_str)
                except ValueError:
                    continue
                sec_srv = srv_map.get(sec_str, {})
                for tier, dem in tier_to_dem.items():
                    unmet = max(0.0, float(dem) - float(sec_srv.get(tier, 0.0)))
                    if unmet < 1e-9:
                        continue
                    cur_tier = agg.top_unmet_tier_per_sector.get(sec_enum)
                    if cur_tier is None or int(tier) < cur_tier:
                        agg.top_unmet_tier_per_sector[sec_enum] = int(tier)
                        agg.top_unmet_mag_per_sector[sec_enum] = unmet
                    elif int(tier) == cur_tier:
                        agg.top_unmet_mag_per_sector[sec_enum] = (
                            agg.top_unmet_mag_per_sector.get(sec_enum, 0.0) + unmet
                        )
        return agg

    def _compute_sector_priorities(self, np, agg: _FlexAggregate):
        """F4 top-tier-dominant priority weights for the ADMM sharing
        problem.  A sector whose lowest unmet tier is ``t`` outranks any
        sector with top tier ``t' > t``; within a tier, magnitude is a
        bounded ``log1p`` tiebreaker.  Falls back to the aggregated
        magnitude weight when ``demand_by_sector_priority`` is absent.
        Result normalised to ``[0.01, 1]``.
        """
        def _sector_w(sec: Sector) -> float:
            top_tier = agg.top_unmet_tier_per_sector.get(sec)
            if top_tier is None:
                return agg.sector_priority_weight.get(sec, 1.0) or 1.0
            # Strict-monotone schedule: tier 1 gets the highest weight
            # so the L3 CP ADMM's S-coefficient pulls toward sectors
            # with high-priority unmet demand.  Uses the L2/L3 helper,
            # not the L1 QP schedule (which returns 0 for tier 1).
            base = tier_priority_weight_strict(top_tier, priority_tiers=4)
            mag = agg.top_unmet_mag_per_sector.get(sec, 0.0)
            return base * (1.0 + 0.5 * math.log1p(mag))

        w_el = _sector_w(Sector.ELECTRICITY)
        w_heat = _sector_w(Sector.HEAT)
        w_gas = _sector_w(Sector.GAS)
        w_max = max(w_el, w_heat, w_gas, 1e-9)
        priorities = np.array([w_el, w_heat, w_gas]) / w_max
        return np.maximum(priorities, 0.01)

    async def _run_admm(self) -> None:
        answers = self._flex_answers[:]
        self._flex_answers = []
        self._flex_expected = 0

        agg = self._aggregate_flex_answers(answers)
        imbalance_by_sector = agg.imbalance_by_sector
        unmet_by_sector_total = agg.unmet_by_sector_total
        sector_priority_weight = agg.sector_priority_weight
        top_unmet_tier_per_sector = agg.top_unmet_tier_per_sector
        top_unmet_mag_per_sector = agg.top_unmet_mag_per_sector

        # Combine balance + unsigned ``unmet`` into the T vector so a
        # sector whose loads are disconnected (balance≈0) is still
        # represented as a positive deficit.
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
            # Ledger entry so the CP's wake-up activity is visible to the
            # post-run analysis even when ADMM bails before publishing a
            # setpoint.  Without this the cp_setpoint counter undercounts
            # CP engagement on grids where T is structurally same-sign.
            record_event(
                t=float(self.context.current_timestamp),
                kind="cp_admm_skipped_same_sign",
                aid=str(self.context.aid),
                sector="cp",
                detail=f"T={T.tolist()} n_skips={self._same_sign_skip_count}",
            )
            self._active = False
            return

        priorities = self._compute_sector_priorities(np, agg)

        coordinator = create_sharing_target_distance_admm_coordinator()
        start_msg = create_admm_start(
            create_admm_sharing_data(T.tolist(), priorities=priorities.tolist())
        )

        try:
            await start_coordinated_optimization(
                [self.flex_actor], coordinator, start_msg
            )
            result = self._clamp_to_envelope(list(self.flex_actor.x))
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
            group_leaders = self._live_connectors(topology_connectors(self, tid="cps"))
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
            # Ledger entry — picked up by ``event_log()`` so the plots
            # can reconstruct each CP's per-sector flow timeline.
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
            # Fallback: trigger intra-group gossip so groups can still
            # rebalance locally even though cross-sector optimisation failed.
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

        Only the lex-smallest CP aid in the multi-sector component
        actually runs.  Other CPs early-out and wait for the
        :class:`CPAllocation` broadcast from the coord.

        Flow:
          1. Collect flex from every group leader in the multi-sector
             component (one ``AskForAvailableFlex`` per leader).
          2. Wait for replies via the existing ``_handle_flex_answer``
             buffer; once the expected count lands, call
             :meth:`_run_multi_sector_admm`.
          3. The ADMM is the same supply-priority kernel L2 uses, but
             scoped to the multi-sector component (spans every sector
             touched by this component).
          4. Dispatch :class:`ComponentAllocation` to each leader (per-
             sector slice of the joint result), then compute per-sector
             marginal values and broadcast :class:`CPAllocation` to
             every CP in the component (including self) so each CP
             can apply its setpoint via the existing
             :meth:`_apply_result` path.

        Falls back to the legacy per-CP path via
        :meth:`trigger_cp_negotiation` when the multi-sector wiring is
        unavailable (e.g. ``wire_multi_sector_l3`` wasn't called).
        """
        if not self._multi_sector_l3_enabled():
            await self.trigger_cp_negotiation()
            return
        # Note: deliberately do NOT gate on cps-cluster-leader here.
        # The L3 coordinator identity is determined by lex-smallest
        # aid in the multi-sector component (see ``_is_l3_coordinator``)
        # which is independent of cps-cluster topology leadership.  In
        # grids with CPs scattered across multiple cps-clusters, the
        # L3 coord often isn't the cps-cluster leader; gating here
        # caused the "silent shed" pattern in the 2026-05-23 smoke
        # where L2 deferred but L3 never fired.
        if not self._is_l3_coordinator():
            # Non-coord: wait for the coord's CPAllocation broadcast.
            return
        if self._l3_active or self._active:
            return

        component_leaders = self._leader_addrs_in_component()
        # Flatten {sector: {aid: addr}} → list[addr], one per leader
        # (some leaders may appear in multiple sectors if their group
        # spans them, but ``leader_addrs_by_sector`` is per-sector so
        # the same aid won't double-list under a single sector).
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
        """Run the joint multi-sector supply-priority ADMM over the
        collected leader flex answers and dispatch the result.

        Called by :meth:`_handle_flex_answer` when the L3-coord path
        has gathered the expected replies.  Distinct from
        :meth:`_run_admm` (the legacy per-CP path).
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

        # Build (supply, demand) actor lists from the leader replies.
        # One actor per leader.  The supply-priority ADMM is sector-
        # agnostic about how many sectors live in a single round, so
        # passing in every sector touched by any leader gives us the
        # multi-sector joint solve.
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

        # Numeric noise scrub: the multi-sector supply-priority ADMM
        # leaves ~1e-3-scale residuals on cells it can't serve.  The
        # ADMM converges within its own tolerance, but the cross-cell
        # ordering of those residuals isn't priority-monotone at that
        # noise level — the per-sector L2 path produced ~1e-7-scale
        # residuals that didn't trip PI; the multi-sector pool is
        # noisier because more cells share one waterfall target.
        # Below ``_FRACTION_NOISE`` (= PI claim tolerance) we clamp
        # to exactly 0, restoring priority-monotone-by-construction
        # at sub-tolerance without affecting cells with genuine
        # partial service.
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

        # L3 *does not* dispatch ComponentAllocation to leaders any
        # more.  The per-sector priority allocation is L2's job — L2
        # runs in parallel with L3 and refines per-sector per-tier
        # service fractions on the post-CP state.  L3's sole output
        # is the CP setpoints; using the multi-sector ADMM result
        # only to compute per-sector marginal values for the
        # gradient-step setpoint decision.
        #
        # Earlier shipping had L3 broadcasting ComponentAllocation to
        # every leader (which then conflicted with L2's own dispatch
        # and made L1 gossip the de-facto arbiter — see the
        # 2026-05-23 PI regression).  Removed.

        # Compute per-sector marginal values from the service fractions
        # and dispatch a per-CP setpoint to every CP in the multi-sector
        # component.  Convention: marginal_value(sector) = 1 - min over
        # tiers of the served fraction at tiers with positive demand.
        # 0 = everything served, no scarcity.  Closer to 1 = stressed.
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
            cp_setpoint = self._compute_cp_setpoint(meta, marginal_by_sector)
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

        # S2 — wake L2 in every sector the multi-sector ADMM touched.
        # The CPs have just committed new setpoints; the post-CP-commit
        # LP routing will change leader flex on the next gossip pass.
        # Sending an ``L3RebalanceWakeup`` to every leader in the
        # multi-sector component (per sector) flags ``_rebalance_dirty``
        # so the leader's L2 short-circuit lifts and the next watchdog /
        # reactive trigger re-evaluates with the new state.  No payload
        # beyond the sector filter — this is purely a "kick" message,
        # not a dispatch.
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

    def _compute_cp_setpoint(
        self,
        cp_meta: dict[str, Any],
        marginal_by_sector: dict[str, float],
    ) -> dict[str, float]:
        """Pick a setpoint for one CP given the per-sector marginal
        values from the L3 ADMM.

        Heuristic gradient step: for each ``(in_sector, out_sector)``
        coupling pair, run the CP iff the destination's marginal value
        × ratio exceeds the source's marginal value (i.e. conversion
        relieves a more-stressed cell at a less-stressed cell's cost).
        Setpoint magnitude is ``capacity × max(0, marginal_out × ratio
        − marginal_in)`` — proportional to how lopsided the imbalance
        is.  Conservative by construction: a balanced pair gives a
        zero step, so no CP commitment in zero-deficit scenarios.

        Returns ``{sector_value: signed_flow_mw}``.  Sign convention:
        positive ⇒ flow into sector (CP consumes / load-like);
        negative ⇒ flow out of sector (CP produces / generator-like).
        Matches the existing :class:`CPSetpoint.sector_flows_mw`
        convention so :meth:`_apply_result` consumes it unchanged.
        """
        capacity = float(cp_meta.get("capacity_mw") or 0.0)
        if capacity <= 0:
            return {}
        ratios = cp_meta.get("coupling_ratios") or {}
        if not ratios:
            return {}

        best_in: str | None = None
        best_out: str | None = None
        best_step: float = 0.0
        best_ratio: float = 1.0
        for (sec_in, sec_out), ratio in ratios.items():
            try:
                r = float(ratio)
            except (TypeError, ValueError):
                continue
            m_in = float(marginal_by_sector.get(str(sec_in), 0.0))
            m_out = float(marginal_by_sector.get(str(sec_out), 0.0))
            step = m_out * r - m_in
            if step > best_step:
                best_step = step
                best_in = str(sec_in)
                best_out = str(sec_out)
                best_ratio = r

        if best_in is None or best_step <= 0.0:
            return {}

        # Magnitude of conversion to commit, clamped to capacity.  The
        # step ∈ (0, 1] (marginal values are clipped to [0, 1] above)
        # so capacity × step lands in (0, capacity].
        magnitude = capacity * min(1.0, best_step)
        return {
            best_in: float(magnitude),                 # consume from source
            best_out: -float(magnitude * best_ratio),  # produce into destination
        }

    async def _handle_cp_allocation(
        self, message: CPAllocation, meta: dict
    ) -> None:
        """Apply a setpoint dispatched by the L3 coord.  Routes through
        the existing :meth:`_apply_result` so the regulate ledger,
        diagnostics and downstream LP all see the same path the legacy
        per-CP ADMM would have produced.

        Idempotent on repeated identical broadcasts: ``apply_regulate``
        already dedups same-value writes within tolerance.
        """
        if topology_characteristic(self, tid="cps") != "leader":
            return
        flows_mw = dict(message.sector_flows_mw)
        # Translate the dict to the flat [el, heat, gas] result vector
        # _apply_result expects.  Missing sectors stay at 0.
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

    The baseline forms one community per connected component of each
    per-sector subgraph (see :func:`scare.base.community.\
connected_component_partition`) and exposes the CP as a connector to
    each of the per-sector communities it bridges (via the same
    ``cps``↔``groups`` cross-topology link the SCARe CP-ADMM path uses).
    Group leaders therefore continue to dispatch
    :class:`NegotiationFinishedEvent` to the CP after each local
    rebalance, but with the holonic + cross-sector ADMM layers turned
    off, the CP no longer has a single coordinator deciding its
    setpoint; each community contributes an independent ask.

    This role reconciles those asks under an EMA-blended target with a
    deadband + cooldown so the CP, which is "in" two communities at
    once, cannot ping-pong between contradictory commits.  It
    deliberately replaces the legacy :class:`EnergyConverterRole`
    pipeline (no ADMM, no flex-actor, no per-sector beacons) and only
    keeps the minimal :class:`AskEnergyMessage` reply so co-located
    gossip rounds don't stall waiting for the CP.

    State machine
    -------------
    - ``_target_by_sector``  EMA-blended desired setpoint per sector.
      Updated on each incoming :class:`NegotiationFinishedEvent`:
      ``target ← α · new_setpoint + (1 − α) · target``.
    - ``_committed_by_sector``  Last setpoint actually committed via
      :func:`scare.base.util.apply_regulate`.  A commit fires only when
      ``|target − committed| > deadband`` AND the per-CP cooldown has
      elapsed, so noisy fluctuations and high-frequency oscillation are
      suppressed.
    - ``_last_commit_t``  Sim-time of the last commit; gates the
      cooldown check above.
    - EMA is reset on every observed :class:`BranchFailureEvent` (the
      scenario builder wires it to :meth:`on_branch_failure` via
      ``behavior_in``) so a failure that islands one of the
      communities the CP bridges does not leave stale signal averaged
      into the post-failure decision.  ``_committed_*`` is retained
      because the physical setpoint at the CP does not move on the
      event itself; only the EMA's belief about what each community
      wants is dropped.
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
        # Per-sector smoothed target.  Initialised lazily to the first
        # observed setpoint so the EMA doesn't start at 0 (which would
        # bias every CP toward "off" until enough rounds accumulate).
        self._target_by_sector: dict[Sector, float] = {}
        # Last committed setpoint per sector — used by the deadband
        # check, so a small drift between proposed targets and the
        # active commit doesn't generate spurious regulate calls.
        self._committed_by_sector: dict[Sector, float] = {}
        # Sim-time of the last apply_regulate call; cooldown gate.
        self._last_commit_t: float = -1e9

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        # Reply minimally to community gossip's flex-query round so a
        # leader treating this CP as a (cps-cross-link) connector does
        # not stall waiting for a response.  ``available=0`` mirrors
        # ``EnergyConverterRole._handle_ask_energy``: the CP brings no
        # spare flex of its own, only the cross-sector conversion knob.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_ask_energy),
            lambda msg, meta: isinstance(msg, AskEnergyMessage),
        )
        # Per-community signal: every time a community's gossip
        # converges, its leader broadcasts NegotiationFinishedEvent to
        # the CP connectors of the relevant sector.  Each event tells
        # us what target setpoint the community has settled on; we
        # treat that as the community's ask of the CP.
        self.context.subscribe_message(
            self,
            _wrap(self._handle_negotiation_finished),
            lambda msg, meta: isinstance(msg, NegotiationFinishedEvent),
        )
        # Dynamic re-partition handshake is wired by the scenario
        # builder via ``behavior_in(... on_global_event=BranchFailureEvent,
        # role_types=MultiCommunityCPRole)`` and dispatches to
        # :meth:`on_branch_failure` below.  CPs themselves are not
        # community members in the ``groups`` topology (they bridge
        # via the ``cps``↔``groups`` cross-link), so the
        # :class:`RepartitionHandlerRole` path that fires
        # ``CommunityReassignedEvent`` on regular members never
        # reaches a CP.  The branch-failure global event is the only
        # signal that lands on every CP regardless of topology
        # placement, which is exactly the property we need for a
        # safety reset.

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

        # EMA blend.  First observation seeds the target — without
        # seeding, the first commit would chase 0 → proposed across
        # several rounds, behaviour the deadband then masks.
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

        Dispatched from the scenario builder's ``behavior_in`` hook on
        :class:`BranchFailureEvent`.  Conservative by design: we don't
        try to decide whether *this* failure islands one of the
        communities we bridge, because doing so would require the same
        physical-graph mirror the SCARe variant keeps for L2/L3 — and
        the EMA's α = 0.3 default re-seeds within a few rounds of
        post-failure :class:`NegotiationFinishedEvent` deliveries
        anyway, so a spurious reset is cheap.

        ``_committed_by_sector`` is retained on purpose: the actual
        regulation setpoint at the CP did not move on the failure, so
        the deadband check still anchors against the live operating
        point.  Cooldown is retained for the same reason.
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
