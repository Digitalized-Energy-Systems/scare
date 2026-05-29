"""L3 priority-cascaded sharing ADMM role — replicated, coordinator-free.

Per-CP role that holds the wiring side of the L3 redesign whose
compute kernel lives in :mod:`scare.service.cp_priority_admm`.  The
role runs that kernel locally on a gossiped peer view and commits
*only* its own regulation factor via :func:`apply_regulate`.  There is
no coordinator election, no per-component fan-out, and no `CPAllocation`
envelope — every CP that holds a quorum-fresh view reaches the same
allocation independently because the kernel is deterministic.

Mesh
----

Each role observes two channels of typed summaries:

- :class:`HolonSummary` from every leader in every sector this CP
  bridges, sourced from the existing
  ``holon_summary_<sector>`` overlays.  The CP joins those topologies
  at scenario build time.  Demand / supply per sector is reconstructed
  from these by aggregating across the latest summary per leader.
- :class:`CPSummary` from every other CP in the cross-sector component,
  sourced from a new global ``cp_summary`` full-mesh topology.  Each
  CP publishes its own summary event-driven (with a delta gate and a
  long watchdog) so peers can build their replicated view.

Lifecycle
---------

* On install, the role publishes its initial :class:`CPSummary` and
  schedules a watchdog tick that re-publishes the current capacity
  unconditionally every ``watchdog_s`` (default 30 s) so late-joining
  peers always observe the frontier.
* Each incoming :class:`HolonSummary` flips a ``dirty`` flag; the
  next minimum-gap-throttled tick consumes it and runs
  :func:`solve_cp_priority_admm` on the current replicated view.
* The kernel returns this CP's regulation factor for the horizon-0
  step; the role commits it via ``apply_regulate(...,
  sector="cp", reason="cp_priority_admm")``.  No further messages are
  sent — peers will pick up the implied effect on the next round of
  L2 summaries.
* Branch failures invalidate the CP's reachable-peer set via the
  shared :class:`~scare.base.topology_mirror.GridTopologyMirror`.  The
  next kernel run automatically excludes islanded peers because the
  role consults the mirror when filtering ``_peer_summaries``.

Receding-horizon support is reserved on the kernel side (``H``-axis
in every input array); the role's MVP runs at ``H = 1``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from mango import Role
from mango import sender_addr as mango_sender_addr

from scare.base.channel import CPSummary, HolonSummary, MonotonicVersion
from scare.base.model import Sector
from scare.base.util import apply_regulate, kgps_to_mw
from distributed_resource_optimization.algorithm.distributed_lexicographic_cascade.core import (  # noqa: E501
    solve_cp_distributed_lexicographic_cascade,
)

from scare.service.cp_priority_admm import (
    CPSpec,
    SectorDemand,
    solve_cp_priority_admm,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior
    from scare.base.topology_mirror import GridTopologyMirror

logger = logging.getLogger(__name__)


class CPPriorityAdmmRole(Role):
    """Replicated-kernel L3 role on a single CP."""

    def __init__(
        self,
        behavior: "RestorationEnvironmentBehavior",
        cp_id: str,
        *,
        capacity_by_sector: dict[str, float],
        bridged_sectors: list[Sector],
        home_node_id: Any = None,
        watchdog_s: float = 30.0,
        rebalance_min_gap_s: float = 0.5,
        horizon: int = 1,
        rho: float = 1.0,
        priority_weight_base: float = 1.0e4,
        r_damping: float = 0.3,
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
        # When set, the heat sector's L3 base supply is the *delivered*
        # heat (Σ served) rather than the (unbounded) heat-slack budget,
        # so the unmet heat demand drives heat-producing CPs to ramp.
        self.heat_supply_from_deficit = bool(heat_supply_from_deficit)
        self.home_node_id = home_node_id
        self.watchdog_s = watchdog_s
        self.rebalance_min_gap_s = rebalance_min_gap_s
        self.horizon = int(horizon)
        self.rho = float(rho)
        self.priority_weight_base = float(priority_weight_base)
        self.r_damping = float(r_damping)
        self.admm_max_iters = int(admm_max_iters)
        self.admm_abs_tol = float(admm_abs_tol)
        self.algorithm = str(algorithm)
        self.r_regularization = float(r_regularization)

        self._version = MonotonicVersion()
        # Peer state caches keyed by publisher aid.
        self._peer_cps: dict[str, CPSummary] = {}
        # sector_value -> {leader_aid: HolonSummary}
        self._leader_summaries: dict[str, dict[str, HolonSummary]] = {
            s.value: {} for s in self.bridged_sectors
        }

        # Throttle / dirty-tracking.
        self._dirty: bool = True
        self._last_rebalance_t: float = -1e9
        self._last_committed_factor: float | None = None

        # Wiring injected after construction (scenario-build time).
        self._topology_mirror: "GridTopologyMirror | None" = None
        self._peer_cp_addrs: dict[str, Any] = {}
        self._peer_cp_node_ids: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Wiring (called by scenario.restoration after the world is built)
    # ------------------------------------------------------------------

    def wire(
        self,
        *,
        topology_mirror: "GridTopologyMirror",
        peer_cp_addrs: dict[str, Any],
        peer_cp_node_ids: dict[str, Any],
    ) -> None:
        """Inject the cross-sector reachability mirror plus the
        address book of every CP in the world.  Mirrors
        :meth:`EnergyConverterRole.wire_multi_sector_l3` so the
        scenario builder can wire either L3 path with the same data.
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

        # Initial publish: announce our existence and capacity so peer
        # CPs can include us in their replicated view from the first
        # kernel run.
        self.context.schedule_instant_task(self._publish(force=True))
        # Watchdog: periodically re-publish (so a peer that joins late
        # eventually sees us) and run a kernel pass.
        try:
            self.context.schedule_periodic_task(
                coroutine_creator=self._tick, delay=self.watchdog_s,
            )
        except (AttributeError, TypeError):
            # Some test contexts don't support periodic scheduling;
            # the role still works event-driven.
            pass

    # ------------------------------------------------------------------
    # Publish / subscribe
    # ------------------------------------------------------------------

    async def _publish(self, *, force: bool = False) -> None:
        """Send a fresh :class:`CPSummary` to every peer CP we know
        about.  Delta-gated on capacity changes; ``force=True`` bypasses
        the gate (used on first publish, on watchdog tick, and after a
        post-rebalance state shift).
        """
        if not self._peer_cp_addrs and not force:
            return
        # Delta gate: skip when capacity hasn't materially shifted.
        # Capacity rarely moves in steady state; the gate keeps the
        # mesh quiet between failures.
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
                    "[%s] CPSummary send failed: %s", self.cp_id, exc,
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

    async def _on_holon_summary(
        self, message: HolonSummary, meta: dict
    ) -> None:
        """Cache the latest summary per (sector, leader) and mark the
        kernel as dirty so the next throttled tick re-runs it.
        """
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
        """Watchdog: refresh our published summary and re-evaluate.
        Set ``_dirty`` so :meth:`_maybe_rebalance` runs even when no
        peer summary has arrived during the last interval.
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
        """CPs reachable through the live cross-sector subgraph.

        Without the mirror (test contexts) we assume every announced
        peer is reachable.  With the mirror, we BFS from our own node
        through same-sector edges and CP bridges and admit only peers
        whose host node is in the reachable set.
        """
        if self._topology_mirror is None or self.home_node_id is None:
            return set(self._peer_cps.keys())
        try:
            reachable = self._topology_mirror.reachable_from(
                self.home_node_id, sector=None, allow_cp_bridges=True,
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
        :class:`SectorDemand` per bridged sector.  Sectors with no
        observed summaries yet contribute an empty demand (zero
        supply, no demand) — the kernel handles this gracefully and
        the CP simply stays at its current factor until summaries
        arrive.
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
            for summary in bucket.values():
                if heat_deficit_mode:
                    # Heat delivery is temperature-limited, not MW-limited:
                    # the unbounded heat slack would otherwise report an
                    # effectively-infinite pool and L3 would never ramp a
                    # heat CP.  Use the *delivered* heat (Σ served) as the
                    # base supply so the unmet demand (nominal − delivered)
                    # becomes the gap the reachable CHP/P2H units fill.
                    served_map = (summary.served_by_sector_priority or {}).get(sec_v, {})
                    agg_supply += sum(float(v) for v in served_map.values())
                else:
                    supply_dict = summary.supply_by_sector or {}
                    agg_supply += float(supply_dict.get(sec_v, 0.0))
                d_map = (summary.demand_by_sector_priority or {}).get(sec_v, {})
                for tier, val in d_map.items():
                    agg_demand[int(tier)] = agg_demand.get(int(tier), 0.0) + float(val)
            if not agg_demand and agg_supply == 0.0:
                continue
            # The kernel works in MW across every dimension: a CP's
            # ``capacity_by_sector`` is MW (gas is converted via
            # ``kgps_to_mw`` in ``_cp_signed_capacity_by_sector``).  The
            # ``HolonSummary`` it consumes, however, carries gas supply
            # and demand in the sector's native kg/s.  Convert the gas
            # dimension here so ``base_supply``/``demand`` are unit-
            # consistent with the CP capacity they are netted against
            # (``supply_net = base_supply − Σ r·c``, hard budget cap
            # ``≤ Bₛ``); without this the ~55× HHV factor makes the gas
            # budget and served-demand waterfall wrong.
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

    async def _run_kernel_and_commit(self) -> None:
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
            if self.algorithm == "lexicographic":
                # Distributed lexicographic-cascade sharing ADMM: Π
                # rounds (one per priority tier, highest first), each
                # maximising served demand subject to the hard
                # ``σ + Σ_i r_i·c_{i,s} ≤ B_s − θ`` constraint.  Since
                # ``B`` folds in the slack budget, this hard-caps the
                # CPs' cross-sector draw at the operator budget — the
                # formally-correct replacement for the penalty kernel's
                # soft over-draw marginal (which limit-cycles / offsets).
                # SCARE's ``CPSpec`` / ``SectorDemand`` duck-type onto the
                # DRO kernel (it reads only ``.cp_id`` /
                # ``.capacity_by_sector`` and ``.sector`` /
                # ``.demand_by_tier`` / ``.base_supply``).
                result = solve_cp_distributed_lexicographic_cascade(
                    cps=cps,
                    demands=demands,
                    horizon=self.horizon,
                    rho=self.rho,
                    inner_iters_max=self.admm_max_iters,
                    inner_abs_tol=self.admm_abs_tol,
                    r_regularization=self.r_regularization,
                )
            else:
                result = solve_cp_priority_admm(
                    cps=cps,
                    demands=demands,
                    horizon=self.horizon,
                    rho=self.rho,
                    max_iters=self.admm_max_iters,
                    abs_tol=self.admm_abs_tol,
                    priority_weight_base=self.priority_weight_base,
                    r_damping=self.r_damping,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] cp L3 kernel (%s) failed: %s",
                self.cp_id, self.algorithm, exc,
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
            self.cp_id, my_factor, len(cps) - 1, len(demands),
            result.iterations, result.converged, applied,
        )
