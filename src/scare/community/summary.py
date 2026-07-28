"""Layer 2.5 — holon-summary mesh + cross-holon coalition formation.

Each group leader publishes a :class:`HolonSummary` on a sector-wide
full mesh (``holon_summary_<sector>``) and runs a cross-holon
priority-inversion check. On detection a deterministically-elected
initiator forms an ad-hoc coalition, runs a scoped supply-priority
allocation over the members' flex, and broadcasts per-tier
service-fraction constraints they apply on L1 dispatch.

Additive to :class:`HolonicCommunityRole`: L2's ADMM keeps running and
coalition constraints are TTL-bounded. Inversion = a higher-priority
tier served at a strictly smaller frac than a lower-priority one beyond
``inversion_tol`` (mirrors ``experiment/eval/claims.py``). Initiator =
lex-smallest publisher with non-empty state, collapsing N duplicate
detections into one.

Constraints expire on ``now > issued_at + ttl_s`` (control returns to
L2) or on a ``BranchFailureEvent`` (topology changed). No two-phase
commit: last-write-wins at L1, the next tick re-asserts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_characteristic

from scare.base.channel import (
    CoalitionAcceptance,
    CoalitionConstraint,
    CoalitionInvitation,
    HolonSummary,
    MonotonicVersion,
)
from scare.base.model import NegotiationFinishedEvent, Sector, StartBalanceNegotiation
from scare.base.util import (
    async_dispatch,
)
from scare.community.coalition_store import CoalitionConstraintStore
from scare.community.summary_coalition import CoalitionManager
from scare.community.summary_inversion import InversionDetector
from scare.community.summary_publish import SummaryPublisher
from scare.community.summary_state import (
    _ActiveCoalition,
    _ActiveCrossSectorCoalition,
    _CoalitionAggregate,
    _PendingCoalition,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class HolonSummaryRole(Role):
    """Cross-holon priority observability + coalition formation.

    Installed on every group leader. Non-leaders stay quiescent:
    ``_tick``'s leader-check returns early, so no publish fires.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        period_s: float = 1.0,
        watchdog_s: float = 30.0,
        inversion_tol: float = 1e-3,
        enable_coalition: bool = True,
        coalition_accept_window_s: float = 1.0,
        coalition_constraint_ttl_s: float = 8.0,
        priority_tiers: int = 4,
        admm_max_iters: int = 50,
        admm_abs_tol: float = 1e-3,
        my_node_id: Any = None,
        member_node_ids: dict[str, Any] | None = None,
        mirror: Any = None,
        constraint_store: CoalitionConstraintStore | None = None,
        enable_cross_sector_coalitions: bool = False,
        cp_meta: dict[str, dict[str, Any]] | None = None,
        peer_leader_addrs: dict[Sector, dict[str, Any]] | None = None,
        enable_heat_cp_supply: bool = False,
        heat_refresh_s: float = 2.0,
        cp_budget_nominal: bool = True,
        coalition_delivered_supply: bool = True,
        cp_commitment_actuatable: bool = False,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.period_s = period_s
        # Heat→L3: heat's summary triggers are off, so refresh faster to
        # keep the delivered-heat deficit flowing to the CP-ADMM.
        self.enable_heat_cp_supply = bool(enable_heat_cp_supply)
        self.heat_refresh_s = float(heat_refresh_s)
        self.cp_budget_nominal = bool(cp_budget_nominal)
        # Credit pool generators at delivered |sp| not rated |cap| (a curtailed
        # gen can't fund at nameplate); also gates cross-sector coalitions on the
        # CP transfer being actuatable.
        self.coalition_delivered_supply = bool(coalition_delivered_supply)
        # Whether a CPCommitment consumer exists (legacy EnergyConverterRole L3).
        # Default priority-ADMM L3 has none, so a promised CP transfer never
        # actuates — don't raise own-sector fractions on it (they'd be
        # slack-funded, the child-118 overdraw).
        self.cp_commitment_actuatable = bool(cp_commitment_actuatable)
        # Slow safety-net cadence re-running publish + check + re-assert
        # even when idle; the dominant trigger is event-driven (see setup).
        self.watchdog_s = watchdog_s
        self.inversion_tol = inversion_tol
        self.enable_coalition = enable_coalition
        self.coalition_accept_window_s = coalition_accept_window_s
        self.coalition_constraint_ttl_s = coalition_constraint_ttl_s
        self.priority_tiers = priority_tiers
        self.admm_max_iters = admm_max_iters
        self.admm_abs_tol = admm_abs_tol
        # Deliverability wiring (leader node, member aid→node, mirror).
        # Any being None degrades to raw-supply ADMM (no caps).
        self._my_node_id = my_node_id
        self._member_node_ids: dict[str, Any] = dict(member_node_ids or {})
        self._mirror = mirror
        # Shared store between L2.5 (writer) and L2 (reader). None ⇒ L2's
        # later rounds overwrite per-tier without checking coalitions.
        self._constraint_store = constraint_store
        self._version = MonotonicVersion()
        self._peer_addrs: dict[str, Any] = {}

        # ---- Cross-sector coalition state ----
        # When enabled, cross-sector invariants run after intra-sector
        # checks and may open coalitions spanning a CP.
        self.enable_cross_sector_coalitions = enable_cross_sector_coalitions
        # sector -> {aid -> addr}.
        self._peer_leader_addrs: dict[Sector, dict[str, Any]] = dict(
            peer_leader_addrs or {}
        )

        self._publisher = SummaryPublisher(self)
        # Cooldown = ``period_s`` so a persistent inversion re-detects
        # each tick, converging while L2 rebalances on its slow heartbeat.
        self._detector = InversionDetector(self, period_s)
        self._coalitions = CoalitionManager(self, cp_meta)

    # Helper-owned state kept reachable under its original name: residual role
    # logic and the tests read (and mutate in place) these maps.
    @property
    def _peer_summaries(self) -> dict[str, HolonSummary]:
        return self._publisher._peer_summaries

    @property
    def _pending_coalitions(self) -> dict[str, _PendingCoalition]:
        return self._coalitions._pending_coalitions

    @property
    def _active_coalitions(self) -> dict[str, _ActiveCoalition]:
        return self._coalitions._active_coalitions

    @property
    def _active_xs_coalitions(self) -> dict[str, _ActiveCrossSectorCoalition]:
        return self._coalitions._active_xs_coalitions

    @property
    def _cp_meta(self) -> dict[str, dict[str, Any]]:
        return self._coalitions._cp_meta

    @property
    def _topology_tid(self) -> str:
        return f"holon_summary_{self.sector.value}"

    def setup(self) -> None:
        logger.debug(
            "[%s] HolonSummaryRole setup: sector=%s period_s=%.2f tid=%s",
            self.context.aid,
            self.sector.value,
            self.period_s,
            self._topology_tid,
        )

        _wrap = async_dispatch(self)

        # Subscribe to summaries from same-sector peers.
        self.context.subscribe_message(
            self,
            _wrap(self._on_summary),
            lambda msg, meta: (
                isinstance(msg, HolonSummary) and msg.sector == self.sector
            ),
        )
        # Coalition control-plane subs, sector-filtered so a leader is
        # never pulled into another sector's coalition.
        self.context.subscribe_message(
            self,
            _wrap(self._on_invitation),
            lambda msg, meta: (
                isinstance(msg, CoalitionInvitation) and msg.sector == self.sector
            ),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._on_acceptance),
            lambda msg, meta: (
                isinstance(msg, CoalitionAcceptance) and msg.sector == self.sector
            ),
        )
        # Inbound constraints from other initiators, stored so this
        # leader's L2 ADMM consults them first (coalition wins per cell).
        self.context.subscribe_message(
            self,
            _wrap(self._on_constraint),
            lambda msg, meta: (
                isinstance(msg, CoalitionConstraint) and msg.sector == self.sector
            ),
        )
        # Event-driven publish: the per-tier vector only moves on L1
        # gossip convergence or L2 dispatch, so subscribe to both.
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_local_state_change
        )
        self.context.subscribe_message(
            self,
            _wrap(self._on_l2_dispatch),
            lambda msg, meta: isinstance(msg, StartBalanceNegotiation),
        )
        # Immediate first publish so peer summaries are in flight before
        # the L2 holon ADMM lands its initial allocation.
        self.context.schedule_instant_task(self._tick())
        # Watchdog: low-cadence safety net for missed events.
        self.context.schedule_periodic_task(self._tick, delay=self.watchdog_s)
        # L3 refresh: drive a faster delta-gated republish so the CP-ADMM's
        # view stays current. Applies to EVERY sector that feeds L3, not just
        # heat: ``enable_heat_cp_supply`` also flips electricity and gas into
        # the delivered-supply reframe (see ``_build_demands``'s
        # ``input_capped_mode``), so all three are read through ``served``.
        # While this was heat-only, el/gas published just twice in a 30 s run —
        # the mandatory first tick at t~0.08 and the watchdog at t=30 — so L3
        # spent the whole run reading a PRE-FAILURE snapshot in which
        # ``served == demand``.
        if self.enable_heat_cp_supply:
            self.context.schedule_periodic_task(
                self._publish_and_check, delay=self.heat_refresh_s
            )

    async def _tick(self) -> None:
        if topology_characteristic(self, tid="groups") != "leader":
            return
        # Watchdog path: bypass the delta gate so the version frontier
        # advances even when idle.
        await self._publish(force=True)
        self._check_invariants()
        if self.enable_cross_sector_coalitions:
            self._check_cross_sector_invariants()
        await self._reassert_active_coalitions()

    def _on_local_state_change(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        """L1 gossip converged — delta-gated publish + re-check."""
        if event.sector != self.sector:
            return
        if topology_characteristic(self, tid="groups") != "leader":
            return
        self.context.schedule_instant_task(self._publish_and_check())

    async def _on_l2_dispatch(
        self, message: StartBalanceNegotiation, meta: dict
    ) -> None:
        """L2 dispatched a fresh allocation — delta-gated publish + check."""
        if topology_characteristic(self, tid="groups") != "leader":
            return
        await self._publish_and_check()

    async def _publish_and_check(self) -> None:
        await self._publish()
        self._check_invariants()
        if self.enable_cross_sector_coalitions:
            self._check_cross_sector_invariants()

    def _summary_changed(
        self,
        served: dict[int, float],
        demand: dict[int, float],
    ) -> bool:
        return self._publisher._summary_changed(served, demand)

    async def _publish(self, *, force: bool = False) -> None:
        return await self._publisher._publish(force=force)

    async def _on_summary(self, message: HolonSummary, meta: dict) -> None:
        return await self._publisher._on_summary(message, meta)

    # ------------------------------------------------------------------
    # Detection + initiator election
    # ------------------------------------------------------------------

    def _is_elected_initiator(self) -> bool:
        return self._publisher._is_elected_initiator()

    def _check_invariants(self) -> None:
        return self._detector._check_invariants()

    # ------------------------------------------------------------------
    # Cross-sector coalition detection + allocation
    # ------------------------------------------------------------------

    def _check_cross_sector_invariants(self) -> None:
        return self._detector._check_cross_sector_invariants()

    def _find_inversion_pair(
        self,
        own_summaries: dict[str, HolonSummary],
        peer_summaries: dict[str, HolonSummary],
    ) -> tuple[int, int, float, float] | None:
        return self._detector._find_inversion_pair(own_summaries, peer_summaries)

    @staticmethod
    def _aggregate_tier(
        summaries: dict[str, HolonSummary],
    ) -> tuple[dict[int, float], dict[int, float]]:
        return InversionDetector._aggregate_tier(summaries)

    async def _open_cross_sector_coalition(
        self,
        *,
        cp_aid: str,
        own_sec: Sector,
        peer_sec: Sector,
        t_own_high: int,
        t_peer_low: int,
    ) -> None:
        return await self._coalitions._open_cross_sector_coalition(
            cp_aid=cp_aid,
            own_sec=own_sec,
            peer_sec=peer_sec,
            t_own_high=t_own_high,
            t_peer_low=t_peer_low,
        )

    async def _dispatch_active_xs_coalition(
        self, active: _ActiveCrossSectorCoalition
    ) -> None:
        return await self._coalitions._dispatch_active_xs_coalition(active)

    # ------------------------------------------------------------------
    # Coalition initiator path
    # ------------------------------------------------------------------

    async def _open_coalition(
        self,
        target_tiers: tuple[int, ...],
        demand_at_tier: dict[int, float],
    ) -> None:
        return await self._coalitions._open_coalition(target_tiers, demand_at_tier)

    async def _on_invitation(self, message: CoalitionInvitation, meta: dict) -> None:
        return await self._coalitions._on_invitation(message, meta)

    def _local_acceptance(
        self,
        coalition_id: str,
        target_tiers_in: tuple[int, ...],
    ) -> CoalitionAcceptance | None:
        return self._coalitions._local_acceptance(coalition_id, target_tiers_in)

    async def _on_constraint(self, message: CoalitionConstraint, meta: dict) -> None:
        """Persist an incoming constraint so this leader's L2 ADMM
        consults it. Trusts the initiator's TTL.
        """
        if self._constraint_store is None:
            return
        now = float(self.context.current_timestamp)
        self._constraint_store.set(
            coalition_id=message.coalition_id,
            sector=message.sector,
            service_fraction_by_tier=message.service_fraction_by_tier,
            issued_at=float(message.timestamp_s) or now,
            ttl_s=float(message.ttl_s),
        )

    async def _on_acceptance(self, message: CoalitionAcceptance, meta: dict) -> None:
        return await self._coalitions._on_acceptance(message, meta)

    @staticmethod
    def _cap_fractions_by_feasibility(
        fractions: dict[int, float],
        demand_by_tier: dict[int, float],
        served_by_tier: dict[int, float],
    ) -> int:
        return CoalitionManager._cap_fractions_by_feasibility(
            fractions, demand_by_tier, served_by_tier
        )

    def _aggregate_coalition_supply_demand(
        self,
        accepting: list[CoalitionAcceptance],
        sector_str: str,
    ) -> _CoalitionAggregate:
        return self._coalitions._aggregate_coalition_supply_demand(
            accepting, sector_str
        )

    async def _close_and_allocate(self, coalition_id: str) -> None:
        return await self._coalitions._close_and_allocate(coalition_id)

    async def _dispatch_active_coalition(self, active: _ActiveCoalition) -> None:
        return await self._coalitions._dispatch_active_coalition(active)

    async def _reassert_active_coalitions(self) -> None:
        return await self._coalitions._reassert_active_coalitions()

    # ------------------------------------------------------------------
    # Failure invalidation
    # ------------------------------------------------------------------

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Drop all active coalition constraints for this sector.

        Wired on ``BranchFailureEvent``. Invalidates on any failure (not
        just in-sector) since cross-sector coupling can make a CP failure
        relevant everywhere; L2's retrigger then re-allocates.
        """
        n_active = len(self._active_coalitions)
        n_pending = len(self._pending_coalitions)
        n_xs = len(self._active_xs_coalitions)
        n_store = (
            self._constraint_store.clear(self.sector)
            if self._constraint_store is not None
            else 0
        )
        if not n_active and not n_pending and not n_store and not n_xs:
            return
        self._active_coalitions.clear()
        self._pending_coalitions.clear()
        # Cross-sector coalitions invalidate on any failure (conservative).
        self._active_xs_coalitions.clear()
        logger.info(
            "[%s] branch failure invalidated %d active + %d pending "
            "+ %d cross-sector + %d stored coalitions (sector=%s)",
            self.context.aid,
            n_active,
            n_pending,
            n_xs,
            n_store,
            self.sector.value,
        )
