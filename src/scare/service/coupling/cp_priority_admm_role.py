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

from scare.base.channel import CPSummary, HolonSummary, MonotonicVersion
from scare.base.model import Sector
from scare.base.util import apply_regulate, kgps_to_mw
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
        rebalance_min_gap_s: float = 0.5,
        horizon: int = 1,
        rho: float = 1.0,
        admm_max_iters: int = 200,
        admm_abs_tol: float = 1e-3,
        algorithm: str = "lexicographic",
        r_regularization: float = 0.1,
        heat_supply_from_deficit: bool = False,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.cp_id = cp_id
        self.capacity_by_sector = dict(capacity_by_sector)
        self.bridged_sectors = list(bridged_sectors)
        # When set, heat's L3 base supply is delivered heat (Σ served) not the
        # unbounded heat-slack budget, so unmet demand drives heat CPs to ramp.
        self.heat_supply_from_deficit = bool(heat_supply_from_deficit)
        self.home_node_id = home_node_id
        self.watchdog_s = watchdog_s
        self.rebalance_min_gap_s = rebalance_min_gap_s
        self.horizon = int(horizon)
        self.rho = float(rho)
        self.admm_max_iters = int(admm_max_iters)
        self.admm_abs_tol = float(admm_abs_tol)
        self.algorithm = str(algorithm)
        self.r_regularization = float(r_regularization)

        self._version = MonotonicVersion()
        # Peer caches keyed by publisher aid.
        self._peer_cps: dict[str, CPSummary] = {}
        # sector_value -> {leader_aid: HolonSummary}
        self._leader_summaries: dict[str, dict[str, HolonSummary]] = {
            s.value: {} for s in self.bridged_sectors
        }

        # Throttle / dirty-tracking.
        self._dirty: bool = True
        self._last_rebalance_t: float = -1e9
        self._last_committed_factor: float | None = None

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
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))

            return _sync

        # Inbound from L2 — only summaries on sectors we bridge.
        bridged = {s.value for s in self.bridged_sectors}
        self.context.subscribe_message(
            self,
            _wrap(self._on_holon_summary),
            lambda msg, meta: (
                isinstance(msg, HolonSummary) and msg.sector.value in bridged
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

    def _reachable_peer_cp_ids(self) -> set[str]:
        """CPs reachable through the live cross-sector subgraph. Without the
        mirror, all announced peers are assumed reachable; with it, BFS from our
        node and admit only peers whose host node is reachable.
        """
        if self._topology_mirror is None or self.home_node_id is None:
            return set(self._peer_cps.keys())
        try:
            reachable = self._topology_mirror.reachable_from(
                self.home_node_id,
                sector=None,
                allow_cp_bridges=True,
            )
        except Exception:
            return set(self._peer_cps.keys())
        out: set[str] = set()
        for aid in self._peer_cps:
            node = self._peer_cp_node_ids.get(aid)
            if node is None or node in reachable:
                out.add(aid)
        return out

    def _build_demands(self) -> list[SectorDemand]:
        """Aggregate the latest HolonSummary per leader into one
        :class:`SectorDemand` per bridged sector; sectors with no summaries are
        skipped (CP holds its current factor).
        """
        H = self.horizon
        demands: list[SectorDemand] = []
        for sector in self.bridged_sectors:
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
            for summary in bucket.values():
                if heat_deficit_mode:
                    # Heat is temperature-limited, not MW-limited; unbounded heat
                    # slack reads as infinite so L3 never ramps a heat CP. Use
                    # delivered heat (Σ served) so unmet demand becomes the gap.
                    served_map = (summary.served_by_sector_priority or {}).get(
                        sec_v, {}
                    )
                    agg_supply += sum(float(v) for v in served_map.values())
                elif input_capped_mode:
                    # B_s = delivered + binding slack's eff_budget: keep existing
                    # service feasible, then cap extra draw at the slack budget
                    # rather than the unbounded |cap| pool.
                    served_map = (summary.served_by_sector_priority or {}).get(
                        sec_v, {}
                    )
                    agg_supply += sum(float(v) for v in served_map.values())
                    slack_map = summary.slack_budget_by_sector or {}
                    agg_supply += float(slack_map.get(sec_v, 0.0))
                else:
                    supply_dict = summary.supply_by_sector or {}
                    agg_supply += float(supply_dict.get(sec_v, 0.0))
                d_map = (summary.demand_by_sector_priority or {}).get(sec_v, {})
                for tier, val in d_map.items():
                    agg_demand[int(tier)] = agg_demand.get(int(tier), 0.0) + float(val)
            if not agg_demand and agg_supply == 0.0:
                continue
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

    def _am_gossip_initiator(self) -> bool:
        """Lowest cp_id among self + reachable peers wins the initiator slot.
        Re-evaluated per tick, so initiator death hands the slot to the next CP
        with no handover protocol; the participant's per-(cp_id, round_id) dedup
        ensures one negotiation at a time.
        """
        reachable = self._reachable_peer_cp_ids() | {self.cp_id}
        return self.cp_id == min(reachable)

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
        logger.debug(
            "[%s] gossip cascade committed factor=%.4f "
            "(iters=%d, converged=%s, applied=%s)",
            self.cp_id,
            my_factor,
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
        if not self._am_gossip_initiator():
            return
        demands = self._build_demands()
        if not demands:
            return
        participants = sorted(self._reachable_peer_cp_ids() | {self.cp_id})
        self._gossip_round_id += 1
        # Timeouts: a round can take seconds on large grids; round_timeout_s
        # absorbs the tail, iter_timeout_s catches the per-iter broadcast median.
        start = create_gossip_cascade_start(
            round_id=self._gossip_round_id,
            participants=participants,
            demands=demands,
            horizon=self.horizon,
            rho=self.rho,
            inner_iters_max=self.admm_max_iters,
            inner_abs_tol=self.admm_abs_tol,
            r_regularization=self.r_regularization,
            adaptive_rho=True,
            rho_mu=10.0,
            rho_tau=2.0,
            iter_timeout_s=1.0,
            round_timeout_s=30.0,
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
        demands = self._build_demands()
        if not demands:
            return
        reachable = self._reachable_peer_cp_ids()
        cps: list[CPSpec] = [self._own_spec()]
        for aid in sorted(reachable):
            s = self._peer_cps.get(aid)
            if s is None:
                continue
            cps.append(CPSpec(cp_id=aid, capacity_by_sector=dict(s.capacity_by_sector)))

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
