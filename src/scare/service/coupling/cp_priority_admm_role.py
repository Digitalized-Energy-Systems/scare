"""L3 priority-cascaded sharing ADMM role — replicated, coordinator-free.

Per-CP wiring around the kernel in :mod:`scare.service.coupling.cp_priority_admm`.
Runs the kernel locally on a gossiped peer view and commits only its own
regulation factor via :func:`apply_regulate`; the kernel being deterministic,
every CP with a fresh view reaches the same allocation independently.

Observes two channels: :class:`HolonSummary` (per-sector demand/supply, aggregated
across the latest summary per leader) and :class:`CPSummary` (peer CP capacity,
delta-gated + watchdog). Branch failures invalidate the reachable-peer set via the
shared :class:`GridTopologyMirror`. MVP runs at horizon H = 1.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from distributed_resource_optimization.algorithm.admm.lexicographic import (
    GossipCascadeInit,
    GossipIter,
    create_gossip_cascade_participant,
    create_gossip_cascade_start,
    solve_cp_distributed_lexicographic_cascade,
)
from distributed_resource_optimization.carrier.mango import MangoCarrier
from mango import Role
from mango import sender_addr as mango_sender_addr

from scare.base.channel import (
    CPSetpoint,
    CPSummary,
    HolonSummary,
    MonotonicVersion,
)
from scare.base.model import Sector
from scare.base.util import (
    apply_regulate,
    async_dispatch,
    kgps_to_mw,
    publish_cp_supply,
)
from scare.service.coupling.cp_priority_admm import (
    CPSpec,
    SectorDemand,
)


class _ReachableCPCarrier(MangoCarrier):
    """DRO :class:`MangoCarrier` with the gossip broadcast set restricted to CP
    peers currently reachable across the cross-sector graph (post-failure). This
    ``others`` override is scare's only carrier customisation.
    """

    def __init__(self, role: CPPriorityAdmmRole) -> None:
        super().__init__(role)
        self._role = role

    def others(self, participant_id: str) -> list[Any]:
        # Reachable peers only; falls back to all known peers without the mirror.
        reachable = self._role._reachable_peer_cp_ids()
        return [
            addr
            for cp_id, addr in self._role._peer_cp_addrs.items()
            if cp_id in reachable
        ]


if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

    from scare.base.topology.topology_mirror import GridTopologyMirror

logger = logging.getLogger(__name__)

# Distinguishes "caller didn't resolve reachability" from a resolved None
# (mirror unwired => admit everything).
_UNRESOLVED: Any = object()


class CPPriorityAdmmRole(Role):
    """Replicated-kernel L3 role on a single CP."""

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        cp_id: str,
        *,
        capacity_by_sector: dict[str, float],
        bridged_sectors: list[Sector],
        home_node_id: Any = None,
        watchdog_s: float = 30.0,
        summary_ttl_s: float | None = None,
        rebalance_min_gap_s: float = 0.5,
        horizon: int = 1,
        rho: float = 1.0,
        admm_max_iters: int = 200,
        admm_abs_tol: float = 1e-3,
        algorithm: str = "lexicographic",
        r_regularization: float = 0.1,
        scale_free: bool = False,
        publish_supply_credit: bool = False,
        heat_supply_from_deficit: bool = False,
        demand_union: bool = False,
        gossip_warm_start: bool = False,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.cp_id = cp_id
        self.capacity_by_sector = dict(capacity_by_sector)
        self.bridged_sectors = list(bridged_sectors)
        # When set, heat's L3 base supply is delivered heat (Σ served) not the
        # unbounded heat-slack budget, so unmet demand drives heat CPs to ramp.
        self.heat_supply_from_deficit = bool(heat_supply_from_deficit)
        # Gossip only: build the round demand set over every sector, not just the
        # (dynamically elected, usually non-heat) initiator's bridged sectors, so
        # heat demand always reaches the round. See RestorationConfiguration.
        self.demand_union = bool(demand_union)
        # Gossip only: carry ADMM state across rounds (see
        # RestorationConfiguration.enable_cp_gossip_warm_start).
        self.gossip_warm_start = bool(gossip_warm_start)
        self.home_node_id = home_node_id
        self.watchdog_s = watchdog_s
        # Freshness bound on cached HolonSummary entries; the leader watchdog
        # force-republishes every watchdog_s, so 2x covers one missed beat.
        self.summary_ttl_s = (
            float(summary_ttl_s) if summary_ttl_s is not None else 2.0 * watchdog_s
        )
        self.rebalance_min_gap_s = rebalance_min_gap_s
        self.horizon = int(horizon)
        self.rho = float(rho)
        self.admm_max_iters = int(admm_max_iters)
        self.admm_abs_tol = float(admm_abs_tol)
        self.algorithm = str(algorithm)
        self.r_regularization = float(r_regularization)
        self.scale_free = bool(scale_free)
        self.publish_supply_credit = bool(publish_supply_credit)

        self._version = MonotonicVersion()
        # Peer caches keyed by publisher aid.
        self._peer_cps: dict[str, CPSummary] = {}
        # sector_value -> {leader_aid: HolonSummary}. Under demand_union a CP
        # caches summaries for every sector (it may build demand for sectors it
        # doesn't itself bridge), so pre-seed all sectors' buckets.
        summary_sectors = list(Sector) if self.demand_union else self.bridged_sectors
        self._leader_summaries: dict[str, dict[str, HolonSummary]] = {
            s.value: {} for s in summary_sectors
        }
        # Reply path for the post-commit L2 wake-up, learned from the summaries
        # themselves: mango's connector cross-product carries no locality, so the
        # senders are the only measured leader set.
        self._leader_addrs: dict[str, dict[str, Any]] = {
            s.value: {} for s in summary_sectors
        }

        # Throttle / dirty-tracking.
        self._dirty: bool = True
        self._last_rebalance_t: float = -1e9
        self._last_committed_factor: float | None = None
        self._last_wake_factor: float | None = None

        # Injected after construction (scenario-build time).
        self._topology_mirror: GridTopologyMirror | None = None
        self._peer_cp_addrs: dict[str, Any] = {}
        self._peer_cp_node_ids: dict[str, Any] = {}

        # Gossip-mode state, populated lazily in setup(); unused in replicated modes.
        self._gossip_participant = None
        self._gossip_carrier: _ReachableCPCarrier | None = None
        self._gossip_round_id: int = 0

    # ------------------------------------------------------------------
    # Wiring (called by scenario.restoration after the world is built)
    # ------------------------------------------------------------------

    def wire(
        self,
        *,
        topology_mirror: GridTopologyMirror,
        peer_cp_addrs: dict[str, Any],
        peer_cp_node_ids: dict[str, Any],
    ) -> None:
        """Inject the reachability mirror and CP address book. Mirrors
        :meth:`EnergyConverterRole.wire_multi_sector_l3` so the builder wires
        either L3 path with the same data.
        """
        self._topology_mirror = topology_mirror
        self._peer_cp_addrs = dict(peer_cp_addrs)
        self._peer_cp_node_ids = dict(peer_cp_node_ids)

    # ------------------------------------------------------------------
    # Mango lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        _wrap = async_dispatch(self)

        # Inbound from L2. Normally only summaries on sectors we bridge; under
        # demand_union take every sector so the round demand set can include
        # sectors this CP doesn't bridge (e.g. heat, when this CP is the elected
        # gossip initiator but bridges only electricity+gas).
        bridged = {s.value for s in self.bridged_sectors}
        self.context.subscribe_message(
            self,
            _wrap(self._on_holon_summary),
            lambda msg, meta: (
                isinstance(msg, HolonSummary)
                and (self.demand_union or msg.sector.value in bridged)
            ),
        )
        # Inbound from peer CPs on the CP-only mesh.
        self.context.subscribe_message(
            self,
            _wrap(self._on_cp_summary),
            lambda msg, meta: (
                isinstance(msg, CPSummary) and msg.publisher != self.cp_id
            ),
        )

        # Gossip mode: route Init/Iter to the gossip participant.
        if self.algorithm == "gossip":
            self._gossip_carrier = _ReachableCPCarrier(self)
            self._gossip_participant = create_gossip_cascade_participant(
                cp_id=self.cp_id,
                capacity_by_sector=self.capacity_by_sector,
                on_commit=self._on_gossip_commit,
                warm_start=self.gossip_warm_start,
            )
            self.context.subscribe_message(
                self,
                _wrap(self._on_gossip_message),
                lambda msg, meta: isinstance(msg, (GossipCascadeInit, GossipIter)),
            )

        # Announce capacity so peers include us from their first kernel run.
        self.context.schedule_instant_task(self._publish(force=True))
        # Watchdog: re-publish (for late joiners) and run a kernel pass.
        try:
            self.context.schedule_periodic_task(
                coroutine_creator=self._tick,
                delay=self.watchdog_s,
            )
        except (AttributeError, TypeError):
            # Some test contexts lack periodic scheduling; still works event-driven.
            pass

    # ------------------------------------------------------------------
    # Publish / subscribe
    # ------------------------------------------------------------------

    async def _publish(self, *, force: bool = False) -> None:
        """Send a fresh :class:`CPSummary` to every known peer CP. Delta-gated on
        capacity; ``force=True`` bypasses the gate.
        """
        if not self._peer_cp_addrs and not force:
            return
        # Delta gate: skip when capacity hasn't materially shifted.
        if not force and not self._capacity_changed():
            return
        self._last_published_capacity = dict(self.capacity_by_sector)

        summary = CPSummary(
            publisher=self.cp_id,
            version=self._version.next(),
            caused_by={},
            timestamp_s=float(self.context.current_timestamp),
            capacity_by_sector=dict(self.capacity_by_sector),
            home_node_id=self.home_node_id,
        )
        for addr in self._peer_cp_addrs.values():
            try:
                await self.context.send_message(summary, receiver_addr=addr)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[%s] CPSummary send failed: %s",
                    self.cp_id,
                    exc,
                )

    def _capacity_changed(self) -> bool:
        if getattr(self, "_last_published_capacity", None) is None:
            return True
        prev = self._last_published_capacity
        if set(prev) != set(self.capacity_by_sector):
            return True
        for sec, v in self.capacity_by_sector.items():
            if abs(float(v) - float(prev.get(sec, 0.0))) > 1e-6:
                return True
        return False

    async def _on_holon_summary(self, message: HolonSummary, meta: dict) -> None:
        """Cache the latest summary per (sector, leader) and mark dirty."""
        sender = mango_sender_addr(meta)
        leader_aid = getattr(sender, "aid", None) or str(sender)
        if not leader_aid:
            return
        bucket = self._leader_summaries.setdefault(message.sector.value, {})
        prior = bucket.get(leader_aid)
        if prior is not None and message.version <= prior.version:
            return  # stale
        bucket[leader_aid] = message
        self._leader_addrs.setdefault(message.sector.value, {})[leader_aid] = sender
        self._dirty = True
        await self._maybe_rebalance()

    async def _on_cp_summary(self, message: CPSummary, meta: dict) -> None:
        """Cache the latest peer CP summary keyed by publisher aid."""
        prior = self._peer_cps.get(message.publisher)
        if prior is not None and message.version <= prior.version:
            return
        self._peer_cps[message.publisher] = message
        self._dirty = True
        await self._maybe_rebalance()

    async def _tick(self) -> None:
        """Watchdog: refresh our published summary and re-evaluate even when no
        peer summary arrived this interval.
        """
        await self._publish(force=True)
        self._dirty = True
        await self._maybe_rebalance()

    # ------------------------------------------------------------------
    # Throttled kernel run + commit
    # ------------------------------------------------------------------

    async def _maybe_rebalance(self) -> None:
        if not self._dirty:
            return
        now = float(self.context.current_timestamp)
        if now - self._last_rebalance_t < self.rebalance_min_gap_s:
            return
        self._last_rebalance_t = now
        self._dirty = False
        await self._run_kernel_and_commit()

    def _own_spec(self) -> CPSpec:
        return CPSpec(
            cp_id=self.cp_id,
            capacity_by_sector=dict(self.capacity_by_sector),
        )

    def _reachable_peer_cp_ids(self, reachable: Any = _UNRESOLVED) -> set[str]:
        """CPs reachable through the live cross-sector subgraph. Without the
        mirror, all announced peers are assumed reachable; with it, BFS from our
        node and admit only peers whose host node is reachable. Pass a
        pre-computed ``_reachable_node_set()`` result to reuse one BFS.
        """
        if reachable is _UNRESOLVED:
            reachable = self._reachable_node_set()
        if reachable is None:
            return set(self._peer_cps.keys())
        out: set[str] = set()
        for aid in self._peer_cps:
            node = self._peer_cp_node_ids.get(aid)
            if node is None or node in reachable:
                out.add(aid)
        return out

    def _reachable_node_set(self) -> set[Any] | None:
        """Live cross-sector reachable node set from this CP, or None when the
        mirror is unwired (admit everything, mirroring peer-CP behaviour)."""
        if self._topology_mirror is None or self.home_node_id is None:
            return None
        try:
            return self._topology_mirror.reachable_from(
                self.home_node_id,
                sector=None,
                allow_cp_bridges=True,
            )
        except Exception:  # noqa: BLE001
            return None

    def _summary_admitted(
        self, summary: HolonSummary, now: float, reachable: set[Any] | None
    ) -> bool:
        """Freshness + reachability gate for a cached leader summary; leaders
        without a timestamp/home_node_id are admitted (additive, like unknown
        peers)."""
        ts = getattr(summary, "timestamp_s", None)
        if ts is not None and (now - float(ts)) > self.summary_ttl_s:
            return False
        node = getattr(summary, "home_node_id", None)
        if reachable is not None and node is not None and node not in reachable:
            return False
        return True

    def _trace_demands(self, demands: list[SectorDemand], participants: int) -> None:
        """Log the kernel's INPUTS beside its output.

        Without this only the committed factor is observable, and a factor of 0
        is ambiguous: the kernel may be trading off scarce input, or it may be
        seeing no deficit at all. ``base_supply`` is the discriminator.

        A shedding holon computes ``served < demand`` CORRECTLY; what makes the
        kernel see ``supply == demand`` is that the shed value is never
        published — the delta gate in ``SummaryPublisher._summary_changed``
        compares an absolute ``holon_summary_inversion_tol`` against kg/s for
        gas but MW for electricity/heat, so no gas change ever clears it, and
        the only forced republish is the watchdog. Read this trace together
        with the ``L3 supply`` provenance line, which separates the served /
        slack / pool terms.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        for d in demands:
            total = float(sum(float(v[0]) for v in d.demand_by_tier.values()))
            supply = float(d.base_supply[0]) if len(d.base_supply) else 0.0
            tiers = {
                t: round(float(v[0]), 6) for t, v in sorted(d.demand_by_tier.items())
            }
            logger.debug(
                "[%s] L3 demand sector=%s demand=%.6f supply=%.6f deficit=%.6f "
                "by_tier=%s cap=%.6f participants=%d",
                self.cp_id,
                d.sector,
                total,
                supply,
                total - supply,
                tiers,
                float(self.capacity_by_sector.get(d.sector, 0.0)),
                participants,
            )

    def _build_demands(self, reachable: Any = _UNRESOLVED) -> list[SectorDemand]:
        """Aggregate the latest HolonSummary per leader into one
        :class:`SectorDemand` per bridged sector; sectors with no summaries are
        skipped (CP holds its current factor). Summaries from unreachable
        (islanded) or stale leaders are dropped, matching the peer-CP filter.
        Pass a pre-computed ``_reachable_node_set()`` result to reuse one BFS.
        """
        H = self.horizon
        now = float(self.context.current_timestamp)
        if reachable is _UNRESOLVED:
            reachable = self._reachable_node_set()
        demands: list[SectorDemand] = []
        # Under demand_union, build across every sector present in the community
        # (not just this initiator's bridged sectors); sectors with no admitted
        # summaries fall through the empty-demand guard below.
        build_sectors = list(Sector) if self.demand_union else self.bridged_sectors
        for sector in build_sectors:
            sec_v = sector.value
            bucket = self._leader_summaries.get(sec_v, {})
            agg_demand: dict[int, float] = {}
            agg_supply: float = 0.0
            heat_deficit_mode = (
                self.heat_supply_from_deficit and sec_v == Sector.HEAT.value
            )
            # Input-sector capping: under heat→L3 mode CP-input sectors (el, gas)
            # also use the delivered-supply reframe so the CP draw is bounded by
            # slack_budget_by_sector, not the phantom aggregate generator pool.
            input_capped_mode = self.heat_supply_from_deficit and sec_v in (
                Sector.ELECTRICITY.value,
                Sector.GAS.value,
            )
            # Provenance of agg_supply, for the L3 trace: a phantom surplus is
            # only diagnosable if the served / slack / pool terms are separable.
            parts = {"served": 0.0, "slack": 0.0, "pool": 0.0}
            n_admitted = 0
            for summary in bucket.values():
                if not self._summary_admitted(summary, now, reachable):
                    continue
                n_admitted += 1
                if heat_deficit_mode:
                    # Heat is temperature-limited, not MW-limited; unbounded heat
                    # slack reads as infinite so L3 never ramps a heat CP. Use
                    # delivered heat (Σ served) so unmet demand becomes the gap.
                    served_map = (summary.served_by_sector_priority or {}).get(
                        sec_v, {}
                    )
                    served = sum(float(v) for v in served_map.values())
                    agg_supply += served
                    parts["served"] += served
                elif input_capped_mode:
                    # B_s = delivered + binding slack's eff_budget: keep existing
                    # service feasible, then cap extra draw at the slack budget
                    # rather than the unbounded |cap| pool.
                    served_map = (summary.served_by_sector_priority or {}).get(
                        sec_v, {}
                    )
                    served = sum(float(v) for v in served_map.values())
                    agg_supply += served
                    parts["served"] += served
                    slack_map = summary.slack_budget_by_sector or {}
                    slack = float(slack_map.get(sec_v, 0.0))
                    agg_supply += slack
                    parts["slack"] += slack
                else:
                    supply_dict = summary.supply_by_sector or {}
                    pool = float(supply_dict.get(sec_v, 0.0))
                    agg_supply += pool
                    parts["pool"] += pool
                d_map = (summary.demand_by_sector_priority or {}).get(sec_v, {})
                for tier, val in d_map.items():
                    agg_demand[int(tier)] = agg_demand.get(int(tier), 0.0) + float(val)
            if not agg_demand and agg_supply == 0.0:
                continue
            if logger.isEnabledFor(logging.DEBUG):
                mode = (
                    "heat_deficit"
                    if heat_deficit_mode
                    else ("input_capped" if input_capped_mode else "pool")
                )
                logger.debug(
                    "[%s] L3 supply sector=%s mode=%s n_summaries=%d/%d "
                    "raw_served=%.6f raw_slack=%.6f raw_pool=%.6f raw_total=%.6f "
                    "raw_demand=%.6f",
                    self.cp_id,
                    sec_v,
                    mode,
                    n_admitted,
                    len(bucket),
                    parts["served"],
                    parts["slack"],
                    parts["pool"],
                    agg_supply,
                    float(sum(agg_demand.values())),
                )
            # Kernel works in MW; HolonSummary carries gas in native kg/s.
            # Convert so it nets against CP capacity (the ~55× HHV factor would
            # otherwise break the gas waterfall).
            if sec_v == Sector.GAS.value:
                agg_supply = kgps_to_mw(agg_supply)
                agg_demand = {t: kgps_to_mw(v) for t, v in agg_demand.items()}
            demands.append(
                SectorDemand(
                    sector=sec_v,
                    demand_by_tier={
                        t: np.full(H, v, dtype=float) for t, v in agg_demand.items()
                    },
                    base_supply=np.full(H, agg_supply, dtype=float),
                )
            )
        return demands

    # ------------------------------------------------------------------
    # Gossip mode — initiator gate, message routing, commit callback
    # ------------------------------------------------------------------

    def _am_gossip_initiator(self, reachable_nodes: Any = _UNRESOLVED) -> bool:
        """Lowest cp_id among self + reachable peers wins the initiator slot.
        Re-evaluated per tick, so initiator death hands the slot to the next CP
        with no handover protocol; the participant's per-(cp_id, round_id) dedup
        ensures one negotiation at a time.
        """
        peers = self._reachable_peer_cp_ids(reachable_nodes) | {self.cp_id}
        return self.cp_id == min(peers)

    async def _on_gossip_message(
        self, message: GossipCascadeInit | GossipIter, meta: dict
    ) -> None:
        """Route an inbound gossip message to the local participant."""
        if self._gossip_participant is None or self._gossip_carrier is None:
            return
        await self._gossip_participant.on_exchange_message(
            self._gossip_carrier,
            message,
            meta,
        )

    def _publish_supply_credit(self, factor: float) -> None:
        """Hand this CP's committed production back to the L2 supply pool.

        L2 sums ``supply_by_sector`` over node children, and a converter is a
        monee branch — so without this its output is invisible to the leaders
        that decide how much load to shed. Only the produced side is credited
        (signed capacity < 0); the consumed side stays out, since crediting a
        draw as supply would be nonsense and booking it as demand belongs to
        whichever leader owns that sector's pool.
        """
        if not self.publish_supply_credit:
            return
        produced = {
            sec: abs(cap) * factor
            for sec, cap in self.capacity_by_sector.items()
            if cap < 0.0
        }
        # Split each sector's output across the leaders that actually fed us
        # demand for it, so summing the credits over leaders reproduces exactly
        # what this CP made. Crediting every leader the full amount would
        # multiply the pool by the leader count (see publish_cp_supply).
        by_leader: dict[str, dict[str, float]] = {}
        for sec, mw in produced.items():
            leaders = sorted(self._leader_summaries.get(sec, {}))
            if not leaders:
                continue
            share = mw / float(len(leaders))
            for leader_aid in leaders:
                by_leader.setdefault(leader_aid, {})[sec] = share
        publish_cp_supply(
            self.behavior,
            self.cp_id,
            by_leader,
            float(self.context.current_timestamp),
        )

    # Min flow shift before waking L2; mirrors the receiver's own predicate
    # (``HolonicCommunityRole._CP_PREDICATE_DEAD_BAND_MW``) so a commit that
    # would be dead-banded there is never sent.
    _L2_WAKE_DEAD_BAND_MW: float = 1e-3

    async def _wake_l2(self, factor: float) -> None:
        """Tell the leaders that fed us demand that our setpoint moved.

        L2 has no trigger of its own once its reactive burst dies down:
        ``RebalanceRound.dirty`` clears on the first round and the periodic
        ``_try_rebalance`` runs at ``holon_watchdog_s`` (30 s — a whole episode),
        so only heat, which owns a separate poll, keeps rebalancing. This is the
        L3->L2 edge ``EnergyConverterRole`` published as :class:`CPSetpoint`;
        ``CPPriorityAdmmRole`` replaced that role without carrying the edge over,
        leaving ``HolonicCommunityRole._handle_cp_setpoint`` with no publisher.
        On ``simbench_lv_gas_dependent`` gas leaders then decided twice (t=0.08,
        t=0.18) — both before any P2G had produced — and that blanket 0.0
        allocation stood for the remaining 29.8 s, so gas served stayed 0.

        Addressed to the summary senders, not to a connector list: mango's
        connectors are a locality-free cross-product (see
        ``balance._credit_cp_supply``).
        """
        prev = self._last_wake_factor
        if prev is not None:
            peak_cap = max(
                (abs(c) for c in self.capacity_by_sector.values()), default=0.0
            )
            if peak_cap * abs(factor - prev) < self._L2_WAKE_DEAD_BAND_MW:
                return
        self._last_wake_factor = factor

        flows = {sec: cap * factor for sec, cap in self.capacity_by_sector.items()}
        setpoint = CPSetpoint(
            publisher=self.cp_id,
            version=self._version.next(),
            caused_by={},
            timestamp_s=float(self.context.current_timestamp),
            cp_id=self.cp_id,
            sector_flows_mw=flows,
            regulation_factor=float(factor),
        )
        for sector in self.bridged_sectors:
            for addr in self._leader_addrs.get(sector.value, {}).values():
                try:
                    await self.context.send_message(setpoint, receiver_addr=addr)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[%s] L2 wake-up send failed: %s", self.cp_id, exc)

    def _on_gossip_commit(
        self,
        r: np.ndarray,
        converged: bool,
        iterations: int,
    ) -> None:
        """End-of-round callback (fired on every CP); applies this CP's own row
        of the answer via :func:`apply_regulate`.
        """
        if r.size == 0:
            return
        my_factor = float(np.clip(r[0], 0.0, 1.0))
        applied = apply_regulate(
            self.behavior,
            self.cp_id,
            my_factor,
            sector="cp",
            reason="cp_priority_admm",
            timestamp=float(self.context.current_timestamp),
        )
        if applied:
            self._last_committed_factor = my_factor
        self._publish_supply_credit(my_factor)
        # Sync callback (DRO's on_commit), so the wake-up has to be scheduled.
        self.context.schedule_instant_task(self._wake_l2(my_factor))
        logger.debug(
            "[%s] gossip cascade committed factor=%.4f cap=%s "
            "(iters=%d, converged=%s, applied=%s)",
            self.cp_id,
            my_factor,
            {k: round(v, 6) for k, v in sorted(self.capacity_by_sector.items())},
            iterations,
            converged,
            applied,
        )

    async def _run_gossip_round(self) -> None:
        """Kick off a new gossip round iff we're the current initiator.
        Non-initiators are pure responders, driven by the initiator's broadcast.
        """
        if self._gossip_participant is None or self._gossip_carrier is None:
            return
        # One BFS per round, shared by the initiator gate, demand build and
        # participant list.
        reachable_nodes = self._reachable_node_set()
        if not self._am_gossip_initiator(reachable_nodes):
            return
        # Skip if a cascade is in flight: a new round's _begin_round cancels the running
        # (uncommitted) cascade, and re-triggers outpace round completion, so without this
        # guard apply_regulate never fires. Let it self-terminate at round_timeout_s.
        if self._gossip_participant.is_round_active():
            return
        demands = self._build_demands(reachable_nodes)
        if not demands:
            logger.debug(
                "[%s] L3 gossip round SKIPPED: no demands built (summaries cached: %s)",
                self.cp_id,
                {s: len(b) for s, b in self._leader_summaries.items()},
            )
            return
        participants = sorted(
            self._reachable_peer_cp_ids(reachable_nodes) | {self.cp_id}
        )
        self._trace_demands(demands, len(participants))
        self._gossip_round_id += 1
        # SIM-second timeouts (carrier clock): a round commits (apply_regulate) on convergence
        # or round_timeout_s. 30s round/1s iter never completed in the ~10s sim -> CP converter
        # never fired, elec slack over-drew; 2.0s/0.2s commits-on-timeout (partial still curtails).
        start = create_gossip_cascade_start(
            round_id=self._gossip_round_id,
            participants=participants,
            demands=demands,
            horizon=self.horizon,
            rho=self.rho,
            inner_iters_max=self.admm_max_iters,
            inner_abs_tol=self.admm_abs_tol,
            r_regularization=self.r_regularization,
            normalize=self.scale_free,
            r_regularization_relative=self.scale_free,
            minimize_usage=self.scale_free,
            adaptive_rho=True,
            rho_mu=10.0,
            rho_tau=2.0,
            iter_timeout_s=0.2,
            round_timeout_s=2.0,
        )
        await self._gossip_participant.on_exchange_message(
            self._gossip_carrier,
            start,
            meta=None,
        )

    # ------------------------------------------------------------------
    # Replicated kernel path (legacy / default)
    # ------------------------------------------------------------------

    async def _run_kernel_and_commit(self) -> None:
        if self.algorithm == "gossip":
            await self._run_gossip_round()
            return
        reachable_nodes = self._reachable_node_set()
        demands = self._build_demands(reachable_nodes)
        if not demands:
            logger.debug(
                "[%s] L3 kernel SKIPPED: no demands built (summaries cached: %s)",
                self.cp_id,
                {s: len(b) for s, b in self._leader_summaries.items()},
            )
            return
        reachable = self._reachable_peer_cp_ids(reachable_nodes)
        cps: list[CPSpec] = [self._own_spec()]
        for aid in sorted(reachable):
            s = self._peer_cps.get(aid)
            if s is None:
                continue
            cps.append(CPSpec(cp_id=aid, capacity_by_sector=dict(s.capacity_by_sector)))
        self._trace_demands(demands, len(cps))

        try:
            # Lexicographic-cascade sharing ADMM: one round per priority tier
            # (highest first) under σ + Σ_i r_i·c_{i,s} ≤ B_s − θ. B folds in the
            # slack budget, capping the CPs' cross-sector draw at it.
            result = solve_cp_distributed_lexicographic_cascade(
                cps=cps,
                demands=demands,
                horizon=self.horizon,
                rho=self.rho,
                inner_iters_max=self.admm_max_iters,
                inner_abs_tol=self.admm_abs_tol,
                r_regularization=self.r_regularization,
                normalize=self.scale_free,
                r_regularization_relative=self.scale_free,
                minimize_usage=self.scale_free,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] cp L3 kernel (%s) failed: %s",
                self.cp_id,
                self.algorithm,
                exc,
            )
            return

        factor_arr = result.factor_by_cp.get(self.cp_id)
        if factor_arr is None or len(factor_arr) == 0:
            return
        my_factor = float(np.clip(factor_arr[0], 0.0, 1.0))

        applied = apply_regulate(
            self.behavior,
            self.cp_id,
            my_factor,
            sector="cp",
            reason="cp_priority_admm",
            timestamp=float(self.context.current_timestamp),
        )
        if applied:
            self._last_committed_factor = my_factor
        self._publish_supply_credit(my_factor)
        await self._wake_l2(my_factor)
        logger.debug(
            "[%s] L3 kernel committed factor=%.4f (peers=%d, sectors=%d, "
            "iters=%d, converged=%s, applied=%s)",
            self.cp_id,
            my_factor,
            len(cps) - 1,
            len(demands),
            result.iterations,
            result.converged,
            applied,
        )
