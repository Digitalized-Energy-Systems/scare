"""Branch line-relief and congestion-price levers for the grid monitor.

Two coupled controllers on an overloaded branch: the discrete relief target
sent to the home leader (plus the downstream curtail locks it must hold), and
the continuous AIMD congestion price that caps downstream generation. Both
need a per-poll tick even when nothing moved -- see ``needs_per_poll_tick``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from monee.model.child import ExtPowerGrid

from scare.base.addressing import child_aid
from scare.base.model import (
    Sector,
    StartBalanceNegotiation,
)
from scare.base.runtime.diagnostics import record_event
from scare.base.util import (
    LINE_CONGESTION_REASON,
    apply_regulate,
    lookup_priority,
    obs_capacity,
    publish_line_congestion_price,
    publish_line_relief_headroom,
    refresh_line_curtail_lock,
    sector_from_grid,
)
from scare.service.control.constraint_tuning import (
    _DOWNSTREAM_TOPOLOGY_TTL_S,
    _LINE_CONGESTION_GAIN,
    _LINE_CONGESTION_HEADROOM_MARGIN,
    _LINE_CONGESTION_PRICE_MAX,
    _LINE_CONGESTION_RESTORE_STEP,
    _LINE_RELIEF_COOLDOWN_S,
    _LINE_RELIEF_GAIN,
    _LINE_RELIEF_RELEASE_MARGIN,
)

if TYPE_CHECKING:
    from scare.service.control.constraints import GridConstraintMonitor

logger = logging.getLogger(__name__)


class CongestionRelief:
    """Owns the branch's flow-direction context, its relief cooldowns and the
    congestion-price integrator. Reads behavior, sector and config through its
    owning role.
    """

    def __init__(
        self, role: GridConstraintMonitor, downstream_load_addrs: list[Any] | None
    ) -> None:
        self._role = role
        # Per-branch congestion price (1 - gen ceiling); AIMD integrator state.
        self._line_congestion_price: float = 0.0
        # Loads downstream of this branch (the only ones whose curtailment
        # reduces its flow); populated post-build for electricity branch
        # monitors -- see ``GridConstraintMonitor.set_downstream_loads``.
        self._downstream_load_addrs: list[Any] = list(downstream_load_addrs or [])
        # Per-variable cooldown for iterative line-relief re-assert.
        self._relief_inflight: dict[str, float] = {}
        # Branch flow-direction context, resolved lazily from the live net and
        # refreshed on a TTL: which endpoint is upstream (slack side) and which
        # generators sit downstream -- the export-overload relief targets.
        self._downstream_resolved: bool = False
        self._downstream_resolved_t: float = float("-inf")
        self._upstream_is_from: bool | None = None
        self._downstream_gen_aids: list[str] = []
        # Per-variable consecutive export-classified polls (debounce).
        self._export_streak: dict[str, int] = {}

    async def _reassert_line_relief(
        self, obs: dict, var: str, val: float, lo: float, hi: float
    ) -> None:
        """Re-send the relief target while overloaded so the home leader sheds
        round-by-round. Cooldown-guarded so it never out-paces its gossip round.
        """
        now = self._role.context.current_timestamp
        deadline = self._relief_inflight.get(var)
        if deadline is not None and now < deadline:
            return
        self._relief_inflight[var] = now + _LINE_RELIEF_COOLDOWN_S
        await self._role._send_line_overload_relief(obs, val, lo, hi)

    async def _send_line_overload_relief(
        self, obs: dict, val: float, lo: float, hi: float
    ) -> None:
        """Send StartBalanceNegotiation with a relief-MW target: the MW the home
        group must shed, scaled by line flow (max ``p_from_mw`` / ``p_to_mw``).
        """
        if val > hi:
            overshoot_fraction = (val - hi) / 100.0
        elif val < lo:
            overshoot_fraction = (lo - val) / 100.0
        else:
            return

        flow_mw = max(
            abs(float(obs.get("p_from_mw", 0.0) or 0.0)),
            abs(float(obs.get("p_to_mw", 0.0) or 0.0)),
        )
        if flow_mw <= 1e-9:
            # No flow magnitude — fall back to a fractional signal.
            relief_mw = overshoot_fraction
        else:
            relief_mw = flow_mw * overshoot_fraction

        # Negative target => group reduces net load, via the Layer-1 QP's
        # reverse-priority curtailment schedule.
        try:
            await self._role.context.send_message(
                StartBalanceNegotiation(override_target=-relief_mw),
                receiver_addr=self._role.home_leader_addr,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[%s] line-overload relief send failed: %s",
                self._role.context.aid,
                exc,
            )

    def _is_line_relief_branch(self) -> bool:
        """True iff this monitor runs the branch-downstream line-relief lever."""
        return (
            self._role.enable_branch_downstream_relief
            and self._role.branch_id is not None
            and bool(self._downstream_load_addrs)
        )

    def _needs_per_poll_tick(self) -> bool:
        """Branches whose per-poll maintenance must run even when values are
        stable and no violation is active: the line-relief lever (keep a live
        restore-ramp lock fresh in the cleared region) and an active congestion
        price (keep decaying it back to 1.0). Skipping these is the limit-cycle
        the cache gate would otherwise cause."""
        return self._role._is_line_relief_branch() or (
            self._role.enable_line_congestion_price
            and self._role.branch_id is not None
            and self._line_congestion_price > 0.0
        )

    def _hold_downstream_line_locks(self, var: str, val: float, hi: float) -> None:
        """Keep the L2-clawback line locks fresh while the line is over (or in
        the hysteresis band), so L2 can't re-serve a just-relieved load. No-op
        unless this is the line-relief branch lever on ``loading_percent``."""
        if var != "loading_percent" or not self._role._is_line_relief_branch():
            return
        if val <= hi - _LINE_RELIEF_RELEASE_MARGIN:
            return
        now = self._role.context.current_timestamp
        for addr in self._downstream_load_addrs:
            aid = getattr(addr, "aid", None)
            if aid is not None:
                refresh_line_curtail_lock(self._role.behavior, aid, now)

    def _maintain_line_relief_handoff(self, var: str, val: float, hi: float) -> None:
        """Publish this branch's loading headroom (``hi - val``) to its
        downstream loads every poll, and keep any live restore-ramp lock fresh
        so ``apply_regulate``'s bounded hand-back can proceed in the cleared
        region. ``refresh_line_curtail_lock`` only re-stamps EXISTING locks, so
        a fully-restored (lock-dropped) load is left alone. No-op unless this is
        the line-relief branch lever on ``loading_percent``."""
        if var != "loading_percent" or not self._role._is_line_relief_branch():
            return
        now = self._role.context.current_timestamp
        headroom = hi - val
        for addr in self._downstream_load_addrs:
            aid = getattr(addr, "aid", None)
            if aid is None:
                continue
            publish_line_relief_headroom(self._role.behavior, aid, headroom, now)
            refresh_line_curtail_lock(self._role.behavior, aid, now)

    def _maintain_congestion_price(
        self, obs: dict, var: str, val: float, hi: float
    ) -> None:
        """Soft congestion-price controller for an export (reverse-flow) branch
        overload. Runs on every poll that survives the ``_monitor`` cache gate,
        so the price can decay and the generation ceiling recover once the line
        clears. While the price is >0 the branch always needs a per-poll tick, so
        the gate never short-circuits an active price; a short-circuit is only
        possible once the price is already 0 (ceiling fully recovered).

        AIMD-style: integrate the price up on overshoot while the flow is export
        with curtailable downstream gens; decay it on genuine loading headroom;
        hold it inside the hysteresis band (a stalled monitor must not release
        the ceiling and re-overload). The price is published per downstream gen
        (summed across branches by ``line_congestion_ceiling``) and the gens are
        curtailed DOWN to the ceiling immediately; the gossip ``_apply_setpoint``
        enforces the same ceiling softly, so PV can ramp back to serve local load
        up to the export-clearing level without a curtail-lock pinning it at 0.
        """
        if var != "loading_percent" or self._role.branch_id is None:
            return
        gens = self._role._downstream_generator_aids()
        if not gens:
            # No lever here; let the price decay so any stale ceiling lifts.
            self._line_congestion_price = max(
                0.0, self._line_congestion_price - _LINE_CONGESTION_RESTORE_STEP
            )
        else:
            overshoot = (val - hi) / 100.0 if val > hi else 0.0
            is_export = overshoot > 0.0 and self._role._flow_is_export(obs) is True
            if is_export:
                self._line_congestion_price = min(
                    _LINE_CONGESTION_PRICE_MAX,
                    self._line_congestion_price + _LINE_CONGESTION_GAIN * overshoot,
                )
            elif val <= hi - _LINE_CONGESTION_HEADROOM_MARGIN:
                self._line_congestion_price = max(
                    0.0, self._line_congestion_price - _LINE_CONGESTION_RESTORE_STEP
                )
            # else: hysteresis band / non-export overload — hold last price.

        now = self._role.context.current_timestamp
        price = self._line_congestion_price
        ceiling = max(0.0, 1.0 - price)
        for aid in gens:
            publish_line_congestion_price(
                self._role.behavior, str(self._role.branch_id), aid, price, now
            )
            if price <= 0.0:
                continue
            gen_obs = self._role.behavior.observe(aid) or {}
            current = float(gen_obs.get("regulation", 1.0))
            if current > ceiling + 1e-6:
                apply_regulate(
                    self._role.behavior,
                    aid,
                    ceiling,
                    sector=self._role.sector.value,
                    reason=LINE_CONGESTION_REASON,
                    timestamp=now,
                    priority_tier=lookup_priority(self._role.behavior, aid),
                )
        if price > 0.0:
            record_event(
                t=now,
                kind="line_congestion_price",
                aid=self._role.context.aid,
                sector=self._role.sector.value,
                detail=(
                    f"val={val:.1f} hi={hi:.1f} price={price:.3f} "
                    f"ceiling={ceiling:.3f} gens={len(gens)}"
                ),
            )

    def _flow_is_export(self, obs: dict) -> bool | None:
        """True when the branch carries reverse (downstream→slack, export)
        flow, False for forward flow, None when undeterminable. Sign
        convention: ``p_from_mw > 0`` (equivalently ``p_to_mw < 0``) is
        from→to flow."""
        self._role._ensure_downstream_topology()
        if self._upstream_is_from is None:
            return None
        try:
            p_from = float(obs.get("p_from_mw", 0.0) or 0.0)
            p_to = float(obs.get("p_to_mw", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if abs(p_from) <= 1e-9 and abs(p_to) <= 1e-9:
            return None
        flow_from_to = p_from > 0.0 if abs(p_from) >= abs(p_to) else p_to < 0.0
        return flow_from_to is not self._upstream_is_from

    def _ensure_downstream_topology(self) -> None:
        """Resolve the downstream topology on first use and re-resolve after
        ``_DOWNSTREAM_TOPOLOGY_TTL_S``: failures and tie closes reshape the
        graph but no topology event reaches branch monitors, so a TTL is the
        cheapest correct invalidation."""
        now = self._role.context.current_timestamp
        if (
            self._downstream_resolved
            and (now - self._downstream_resolved_t) < _DOWNSTREAM_TOPOLOGY_TTL_S
        ):
            return
        self._downstream_resolved_t = now
        self._role._resolve_downstream_topology()

    def _resolve_downstream_topology(self) -> None:
        """Cut this branch and BFS the electricity graph from the slacks to
        find which endpoint is upstream and which generators sit downstream
        (the export-relief targets). Open ties and failed branches are
        non-conductive. Leaves ``_upstream_is_from`` None on a meshed /
        unclean cut."""
        self._downstream_resolved = True
        self._upstream_is_from = None
        self._downstream_gen_aids = []
        net = getattr(self._role.behavior, "_net", None)
        if net is None or self._role.branch_id is None:
            return
        try:
            branches = list(net.branches)
            childs = list(net.childs)
        except Exception:  # noqa: BLE001
            return

        adj: dict[Any, list[Any]] = {}
        for branch in branches:
            try:
                if branch.id == self._role.branch_id or branch.model.is_cp():
                    continue
                if (
                    not getattr(branch, "active", True)
                    or not getattr(branch.model, "active", True)
                    or not int(getattr(branch.model, "on_off", 1) or 0)
                ):
                    continue
                node = net.node_by_id(branch.id[0])
            except Exception:  # noqa: BLE001
                continue
            if sector_from_grid(getattr(node, "grid", None)) is not Sector.ELECTRICITY:
                continue
            a, b = branch.id[0], branch.id[1]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        slack_nodes = {
            child.node_id for child in childs if isinstance(child.model, ExtPowerGrid)
        }
        if not slack_nodes:
            return

        def _reach(start: set[Any]) -> set[Any]:
            seen = set(start)
            frontier = list(start)
            while frontier:
                nxt: list[Any] = []
                for n in frontier:
                    for nb in adj.get(n, ()):
                        if nb not in seen:
                            seen.add(nb)
                            nxt.append(nb)
                frontier = nxt
            return seen

        fed = _reach(slack_nodes)
        a, b = self._role.branch_id[0], self._role.branch_id[1]
        a_up, b_up = a in fed, b in fed
        if a_up == b_up:
            return  # no clean cut: direction stays undeterminable
        self._upstream_is_from = a_up

        down = _reach({b if a_up else a})
        gens: list[str] = []
        for child in childs:
            if child.node_id not in down or isinstance(child.model, ExtPowerGrid):
                continue
            try:
                cap = obs_capacity(dict(child.model.values))
            except Exception:  # noqa: BLE001
                continue
            if cap >= 0:
                continue
            aid = child_aid(child.id)
            if self._role.behavior.has_action(aid, "regulate"):
                gens.append(aid)
        self._downstream_gen_aids = gens

    def _downstream_generator_aids(self) -> list[str]:
        self._role._ensure_downstream_topology()
        return self._downstream_gen_aids

    async def _relieve_export_overload(
        self, obs: dict, var: str, val: float, hi: float
    ) -> None:
        """Curtail downstream generation for an export (reverse-flow) overload.
        Load-shed paths are suppressed for these — shedding raises net export.
        Cooldown-guarded and re-armed each poll until the line clears."""
        now = self._role.context.current_timestamp
        deadline = self._relief_inflight.get(var)
        if deadline is not None and now < deadline:
            return
        gens = self._role._downstream_generator_aids()
        if not gens:
            # No lever here; leave the shared cooldown unburnt so the
            # ordinary relief chain isn't starved.
            record_event(
                t=now,
                kind="line_export_relief_no_generators",
                aid=self._role.context.aid,
                sector=self._role.sector.value,
                detail=f"{var}={val:.1f} hi={hi:.1f}",
            )
            return
        self._relief_inflight[var] = now + _LINE_RELIEF_COOLDOWN_S
        amount = min(1.0, max(0.25, _LINE_RELIEF_GAIN * (val - hi) / 100.0))
        curtailed = 0
        for aid in gens:
            gen_obs = self._role.behavior.observe(aid) or {}
            current = float(gen_obs.get("regulation", 1.0))
            new_factor = max(0.0, current * (1.0 - amount))
            applied = apply_regulate(
                self._role.behavior,
                aid,
                new_factor,
                sector=self._role.sector.value,
                reason="curtail",
                timestamp=now,
                priority_tier=lookup_priority(self._role.behavior, aid),
            )
            if applied:
                curtailed += 1
        record_event(
            t=now,
            kind="line_export_relief",
            aid=self._role.context.aid,
            sector=self._role.sector.value,
            detail=(
                f"{var}={val:.1f} hi={hi:.1f} amount={amount:.2f} "
                f"gens={len(gens)} curtailed={curtailed}"
            ),
        )
