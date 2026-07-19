"""The two-phase curtailment auction run by the grid monitor.

Broadcast a need, collect willingness bids, allocate the shed. Also holds the
levers layered on top of it: the Q(U) hand-off that defers to reactive relief,
the progress gate that stops re-arming an ineffective auction, and the strict
reverse-priority waterfall used for branch line relief.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors

from scare.base.model import (
    CurtailmentBid,
    CurtailmentNeed,
    CurtailmentRequest,
    Sector,
)
from scare.base.runtime.diagnostics import record_event
from scare.base.util import (
    apply_regulate,
    lookup_priority,
    obs_capacity,
    obs_priority,
    obs_setpoint,
    qv_relief_avail,
)
from scare.service.control.constraint_tuning import (
    _CURTAIL_NO_PROGRESS_LIMIT,
    _CURTAIL_PROGRESS_TOL,
    _CURTAIL_PROX_MAX,
    _CURTAIL_PROX_MIN,
    _LINE_RELIEF_GAIN,
    _LINE_RELIEF_MIN_REDUCIBLE,
    _QV_DEFER_PROGRESS_TOL,
    _QV_MAX_CONSECUTIVE_DEFERS,
    _SECTOR_PRIMARY_VAR,
    _SELF_BID_KEY,
    _SENS_MULT_MAX,
    _SENS_MULT_MIN,
    _SENSITIVITY_DEFAULT,
)
from scare.service.control.curtailment import (
    curtail_willingness,
    plan_auction_allocation,
    proximity_from_hops,
)

if TYPE_CHECKING:
    from scare.service.control.constraints import GridConstraintMonitor

logger = logging.getLogger(__name__)


class CurtailmentAuction:
    """Auctioneer and bidder sides of the curtailment auction. Owns the open
    auctions, the per-variable in-flight/progress guards and the Q(U) defer
    state; reads behavior, sector, sensitivity and config through its owning
    role.
    """

    # Proportional gain on normalized overshoot; gentle so persistence ratchets
    # it up over cycles rather than over-curtailing in one shot.
    _CURTAILMENT_GAIN: float = 0.3

    # How long the auctioneer waits for bids; short, the monitor re-fires next
    # cycle if the violation persists.
    _AUCTION_TIMEOUT_S: float = 2.0

    def __init__(self, role: GridConstraintMonitor) -> None:
        self._role = role
        # Auctioneer-side state: auction_id -> {"bids", "total", ...}.
        self._open_auctions: dict[str, dict[str, Any]] = {}
        # Monotonic counter backing reproducible auction ids -- see base.util.ids.
        self._auction_seq = 0
        # Per-variable in-flight guard (variable -> deadline): prevents stacking
        # auctions while letting curtailment iterate round-by-round.
        self._curtail_inflight: dict[str, float] = {}
        # Progress gate (``enable_curtail_auction_gating``): variable ->
        # {"best", "no_progress"}; suspends re-arming an ineffective lever.
        self._curtail_progress: dict[str, dict[str, float]] = {}
        # Coordinated hand-off: consecutive reactive-defers (backstop cap) and
        # the value at last defer (to check droop progress before re-deferring).
        self._qv_defer_count: dict[str, int] = {}
        self._qv_last_value: dict[str, float] = {}
        # Per-variable flag: waterfall has only tier-1 reducible bidders left
        # (can't relieve further without breaking the hard-lock).
        self._line_relief_tier1_residual: dict[str, bool] = {}

    def _own_curtail_willingness(
        self, obs: dict, *, injection_relief: bool = False
    ) -> float:
        """Curtailment willingness for this agent's own load: priority tier
        weight (dominant, lexicographic) × bounded sensitivity multiplier ×
        reducible output.

        Tier-1 LOADS (cap > 0) return exactly 0.0, not the 1e-9 floor (which
        would let a tier-1 self-only auction shed itself, breaking the
        hard-lock); generators (cap < 0) keep the floor so PV stays shed-eligible.
        ``injection_relief`` restricts an over-voltage auction to generators.
        """
        from scare.service.balance.balance import _PRIORITY_TIERS

        prio_tier = max(
            1,
            obs_priority(obs, behavior=self._role.behavior, aid=self._role.context.aid),
        )
        cap = obs_capacity(
            obs, behavior=self._role.behavior, aid=self._role.context.aid
        )
        reducible = abs(
            obs_setpoint(obs, behavior=self._role.behavior, aid=self._role.context.aid)
        )
        return curtail_willingness(
            priority_tier=prio_tier,
            capacity=cap,
            reducible=reducible,
            sensitivity=self._role._sens.value,
            sensitivity_ref=_SENSITIVITY_DEFAULT.get(self._role.sector, 1e-3),
            priority_tiers=_PRIORITY_TIERS,
            sens_mult_min=_SENS_MULT_MIN,
            sens_mult_max=_SENS_MULT_MAX,
            injection_relief=injection_relief,
        )

    async def _request_curtailment(
        self, variable: str, value: float, lo: float, hi: float
    ) -> None:
        span = hi - lo
        if span <= 0:
            return

        # Over-voltage is relieved only by cutting generation, so bid gens and
        # exclude loads (shedding load on a PV-surplus feeder raises voltage).
        # Q(U)/auction coordination substitutes reactive for the active shed it
        # defers, so it must also target generation — hence it implies
        # injection_relief even when the standalone gen-priority flag is off.
        injection_relief = (
            (
                self._role.enable_generation_priority_curtailment
                or self._role.enable_qv_auction_coordination
            )
            and self._role.sector is Sector.ELECTRICITY
            and variable == "vm_pu"
            and value > hi
        )

        # Gas OVER-pressure: shedding load shrinks the Weymouth drops and
        # RAISES pressure — positive feedback. The slack pressure regulator
        # owns that side; never arm a load-shed for it.
        if self._role.sector is Sector.GAS and variable == "pressure_pu" and value > hi:
            return

        # Heat OVER-temperature: shedding load removes cooling draw from the
        # hot junction — wrong direction (matters in frontier-ablated arms,
        # where the t_k auction skip is off). The CP heat-outlet guard owns
        # the injector side; never arm a load-shed for it.
        if self._role.sector is Sector.HEAT and variable == "t_k" and value > hi:
            return

        # In-flight guard: skip while an auction for this variable is open so
        # rounds don't stack; re-opening once it clears reaches feasibility.
        now = self._role.context.current_timestamp
        deadline_prev = self._curtail_inflight.get(variable)
        if deadline_prev is not None and now < deadline_prev:
            return

        overshoot = (value - hi) / span if value > hi else (lo - value) / span

        # Coordinated hand-off: at a Q(U)-droop node, credit the reactive
        # lever's remaining relief (not yet in ``value``) and size the active
        # shed to the residual; skip the auction entirely when reactive covers
        # the overshoot, re-arming next poll if it falls short.
        if (
            self._role.enable_qv_auction_coordination
            and self._role.sector == Sector.ELECTRICITY
            and variable == _SECTOR_PRIMARY_VAR.get(self._role.sector)
            and value > hi
        ):
            relief = qv_relief_avail(self._role.behavior, self._role.context.aid, now)
            if relief <= 0.0:
                self._qv_defer_count.pop(variable, None)
                self._qv_last_value.pop(variable, None)
            else:
                residual = max(0.0, (value - relief - hi) / span)
                if residual > 0.0:
                    # Reactive covers part of it — shed active for the residual.
                    self._qv_defer_count.pop(variable, None)
                    self._qv_last_value.pop(variable, None)
                    overshoot = residual
                else:
                    # Reactive claims to cover it all. Defer only while voltage
                    # is measurably dropping; on stall (or backstop) escalate
                    # and shed active for the measured overshoot.
                    last = self._qv_last_value.get(variable)
                    improving = last is None or value < last - _QV_DEFER_PROGRESS_TOL
                    # Phase-2 feeder gate: never defer to local reactive while
                    # another node on the feeder is over-voltage — the retained
                    # active PV is what's holding the feeder over, so shed it.
                    feeder_over = (
                        self._role.enable_qv_feeder_gate
                        and self._role._feeder_overvoltage(hi)
                    )
                    cnt = self._qv_defer_count.get(variable, 0)
                    if (
                        improving
                        and not feeder_over
                        and cnt < _QV_MAX_CONSECUTIVE_DEFERS
                    ):
                        self._qv_defer_count[variable] = cnt + 1
                        self._qv_last_value[variable] = value
                        self._curtail_inflight.pop(variable, None)
                        record_event(
                            t=now,
                            kind="curtail_deferred_to_qv_relief",
                            aid=self._role.context.aid,
                            sector=self._role.sector.value,
                            detail=f"v={value:.4f} hi={hi:.4f} relief={relief:.5f} "
                            f"defer={self._qv_defer_count[variable]}",
                        )
                        return
                    # Stalled (or backstop): escalate, shed full overshoot.
                    self._qv_defer_count.pop(variable, None)
                    self._qv_last_value.pop(variable, None)
                    record_event(
                        t=now,
                        kind="curtail_qv_defer_escalated",
                        aid=self._role.context.aid,
                        sector=self._role.sector.value,
                        detail=f"v={value:.4f} hi={hi:.4f} relief={relief:.5f}",
                    )

        # Strict reverse-priority line-relief waterfall auction?
        _waterfall = (
            self._role.enable_line_relief_waterfall
            and self._role.enable_branch_downstream_relief
            and variable == "loading_percent"
            and bool(self._role._downstream_load_addrs)
        )

        if _waterfall:
            # The waterfall self-terminates, so the generic no-progress gate is
            # the wrong stop. Stop only when only tier-1 bidders remain
            # (relieving further would break the hard-lock).
            if self._line_relief_tier1_residual.get(variable):
                return
        elif self._role.enable_curtail_auction_gating:
            # Progress gate: if the overshoot keeps failing to improve, stop
            # re-arming until it worsens or topology re-engages the lever.
            prog = self._curtail_progress.setdefault(
                variable, {"best": float("inf"), "no_progress": 0.0}
            )
            if overshoot < prog["best"] - _CURTAIL_PROGRESS_TOL:
                prog["best"] = overshoot
                prog["no_progress"] = 0.0
            else:
                prog["no_progress"] += 1.0
                if prog["no_progress"] > _CURTAIL_NO_PROGRESS_LIMIT:
                    return

        # Total fractional reduction across group + self. Two-phase auction:
        # broadcast need, collect bids, allocate proportional to willingness.
        _downstream_line = (
            self._role.enable_branch_downstream_relief
            and variable == "loading_percent"
            and bool(self._role._downstream_load_addrs)
        )
        if _downstream_line:
            # High gain to drive a 10-20% overload to feasibility in a few
            # rounds; priority still orders WHO sheds, re-arming until ≤100%.
            total_amount = min(1.0, max(0.25, _LINE_RELIEF_GAIN * overshoot))
        else:
            total_amount = max(0.02, min(1.0, self._CURTAILMENT_GAIN * overshoot))

        # Seed the agent's OWN load as a candidate (most direct lever on its
        # junction); priority still decides absorption.
        self_obs = self._role.behavior.observe(self._role.context.aid) or {}
        self_w_raw = (
            self._role._own_curtail_willingness(
                self_obs, injection_relief=injection_relief
            )
            if self._role.behavior.has_action(self._role.context.aid, "regulate")
            else None
        )
        # Drop a zero-willingness self so the all-zero even-split fallback in
        # ``_allocate_auction`` can't shed it.
        self_w = self_w_raw if (self_w_raw is not None and self_w_raw > 0.0) else None
        # Targeting: auctioneer is the origin (closest bidder), so scale its
        # self-bid by max proximity to compete with neighbour bids.
        if self._role.enable_curtail_auction_targeting and self_w is not None:
            self_w *= _CURTAIL_PROX_MAX

        # Branch-downstream relief: bidders are the loads flowing through the
        # branch (shed reduces its loading); else fall back to the component.
        if (
            self._role.enable_branch_downstream_relief
            and variable == "loading_percent"
            and self._role._downstream_load_addrs
        ):
            neighbors = list(self._role._downstream_load_addrs)
        else:
            neighbors = list(topology_neighbors(self._role, tid="groups"))

        if not neighbors and self_w is None:
            # Self locked, no neighbours — nothing allocable without breaking
            # the hard-lock. Clear the guard so a later poll retries.
            self._curtail_inflight.pop(variable, None)
            return

        self._auction_seq += 1
        auction_id = f"{self._role.context.aid}/{self._auction_seq}"
        self._open_auctions[auction_id] = {
            "bids": {},
            "total": total_amount,
            "neighbours_contacted": len(neighbors),
            "bidders": {},  # sender_key -> addr
            "bid_meta": {},  # sender_key -> (tier, reducible)
            "var": variable,
            "self_willingness": self_w,
            "self_addr": self._role.context.addr,
            "waterfall": _waterfall,
            "injection_relief": injection_relief,
        }
        self._curtail_inflight[variable] = now + self._AUCTION_TIMEOUT_S

        if not neighbors:
            # Self-only auction (isolated node / singleton group): allocate now.
            await self._role._allocate_auction(auction_id)
            return

        need_msg = CurtailmentNeed(
            sector=self._role.sector,
            total_amount=total_amount,
            auction_id=auction_id,
            origin_addr=self._role.context.addr,
            variable=variable,
            injection_relief=injection_relief,
        )
        for addr in neighbors:
            await self._role.context.send_message(need_msg, receiver_addr=addr)

        deadline = now + self._AUCTION_TIMEOUT_S
        self._role.context.schedule_timestamp_task(
            self._role._close_auction(auction_id), timestamp=deadline
        )

    async def _handle_curtailment_need(
        self, message: CurtailmentNeed, meta: dict
    ) -> None:
        if not self._role.behavior.has_action(self._role.context.aid, "regulate"):
            return
        obs = self._role.behavior.observe(self._role.context.aid)
        if not obs:
            return

        willingness = self._role._own_curtail_willingness(
            obs, injection_relief=bool(getattr(message, "injection_relief", False))
        )
        # Targeting: scale by proximity to the origin so the share concentrates
        # on relieving loads (bounded within-tier, priority stays dominant).
        if self._role.enable_curtail_auction_targeting:
            willingness *= self._role._curtail_proximity(
                message.origin_addr, message.variable
            )
        # Carry tier + reducible for a waterfall auctioneer's reverse-priority
        # shed (ignored by the default proportional allocator).
        bid_tier = max(
            1,
            obs_priority(obs, behavior=self._role.behavior, aid=self._role.context.aid),
        )
        bid_reducible = abs(
            obs_setpoint(obs, behavior=self._role.behavior, aid=self._role.context.aid)
        )
        reply = CurtailmentBid(
            auction_id=message.auction_id,
            willingness=willingness,
            sector=self._role.sector,
            tier=bid_tier,
            reducible=bid_reducible,
        )
        await self._role.context.send_message(
            reply, receiver_addr=mango_sender_addr(meta)
        )

    def _curtail_proximity(self, origin_addr: Any, variable: str) -> float:
        """Bounded proximity multiplier for this bidder relative to the origin,
        from cached multi-hop distance (more ``hops_remaining`` => closer). No
        cached state => neutral 1.0 (never starves an unknown bidder).
        """
        if not variable or origin_addr is None or self._role.max_hops <= 0:
            return 1.0
        state = self._role._neighbour_state.get((str(origin_addr), variable))
        if state is None:
            return 1.0
        return proximity_from_hops(
            state.hops_remaining,
            self._role.max_hops,
            prox_min=_CURTAIL_PROX_MIN,
            prox_max=_CURTAIL_PROX_MAX,
        )

    async def _handle_curtailment_bid(
        self, message: CurtailmentBid, meta: dict
    ) -> None:
        auction = self._open_auctions.get(message.auction_id)
        if auction is None:
            return
        sender = mango_sender_addr(meta)
        sender_key = str(sender)
        auction["bids"][sender_key] = message.willingness
        auction["bidders"][sender_key] = sender
        auction["bid_meta"][sender_key] = (  # for the waterfall allocator
            int(getattr(message, "tier", 0) or 0),
            float(getattr(message, "reducible", 0.0) or 0.0),
        )

        if len(auction["bids"]) >= auction["neighbours_contacted"]:
            await self._role._allocate_auction(message.auction_id)

    async def _close_auction(self, auction_id: str) -> None:
        if auction_id in self._open_auctions:
            await self._role._allocate_auction(auction_id)

    async def _allocate_auction(self, auction_id: str) -> None:
        auction = self._open_auctions.pop(auction_id, None)
        if auction is None:
            return
        # Clear the in-flight guard so the next poll can open a new round.
        self._curtail_inflight.pop(auction.get("var"), None)

        bids: dict[str, float] = dict(auction["bids"])
        bidders: dict[str, Any] = dict(auction["bidders"])
        total_amount: float = auction["total"]

        # Fold in the auctioneer's own bid (L0 self-curtail candidate).
        self_w = auction.get("self_willingness")
        if self_w is not None:
            bids[_SELF_BID_KEY] = self_w
            bidders[_SELF_BID_KEY] = auction.get("self_addr")

        if not bids:
            return

        async def _dispatch(key: str, addr: Any, share: float) -> None:
            if share <= 0.0:
                return
            if key == _SELF_BID_KEY:
                await self._role._curtail_self(share)
            elif addr is not None:
                await self._role.context.send_message(
                    CurtailmentRequest(sector=self._role.sector, amount=share),
                    receiver_addr=addr,
                )

        plan = plan_auction_allocation(
            bids,
            bidders,
            dict(auction.get("bid_meta", {})),
            total_amount,
            waterfall=bool(auction.get("waterfall")),
            min_reducible=_LINE_RELIEF_MIN_REDUCIBLE,
        )
        # Waterfall terminal state: only tier-1 bidders remain (relieving
        # further breaks the hard-lock). Surface once and stop re-arming.
        if plan.tier1_exhausted:
            var = auction.get("var", "loading_percent")
            if not self._line_relief_tier1_residual.get(var):
                self._line_relief_tier1_residual[var] = True
                record_event(
                    t=self._role.context.current_timestamp,
                    kind="line_relief_tier1_residual",
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    detail=f"{var}: tiers 2-4 exhausted, line still over",
                )
            return

        for key, addr, share in plan.dispatches:
            await _dispatch(key, addr, share)

    async def _handle_curtailment_request(
        self, message: CurtailmentRequest, meta: dict
    ) -> None:
        await self._role._apply_curtail(message.amount, label="curtailed")

    async def _curtail_self(self, amount: float) -> None:
        """Apply the auctioneer's own winning share (L0 self-curtail)."""
        await self._role._apply_curtail(amount, label="self-curtailed")

    async def _apply_curtail(self, amount: float, *, label: str) -> None:
        if not self._role.behavior.has_action(self._role.context.aid, "regulate"):
            return
        obs = self._role.behavior.observe(self._role.context.aid)
        if not obs:
            return

        # Multiplicative reduction: amount=0.3 cuts output 30%; repeated
        # requests compound toward zero, so one step can't overshoot.
        current = float(obs.get("regulation", 1.0))
        amount = max(0.0, min(1.0, amount))
        new_factor = max(0.0, current * (1.0 - amount))

        applied = apply_regulate(
            self._role.behavior,
            self._role.context.aid,
            new_factor,
            sector=self._role.sector.value,
            reason="curtail",
            timestamp=self._role.context.current_timestamp,
            priority_tier=lookup_priority(self._role.behavior, self._role.context.aid),
        )
        if applied:
            logger.info(
                "[%s] %s by %.1f%% (regulation %.3f -> %.3f)",
                self._role.context.aid,
                label,
                amount * 100,
                current,
                new_factor,
            )
