"""Ad-hoc coalition formation across holons.

An elected initiator invites peers, aggregates their flex, runs a scoped
supply-priority allocation and broadcasts TTL-bounded per-tier service-fraction
constraints. Covers both the single-sector and the cross-sector (CP-mediated)
variants -- they share the re-assert and branch-failure paths.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_neighbors

from scare.base.channel import (
    CoalitionAcceptance,
    CoalitionConstraint,
    CoalitionInvitation,
    CPCommitment,
)
from scare.base.model import Sector, StartBalanceNegotiation
from scare.base.runtime.diagnostics import record_event
from scare.base.util import (
    clamp_tier_monotonic,
    lookup_slack,
    lookup_slack_eff_budget,
    obs_capacity,
    obs_priority,
    obs_sector,
    obs_setpoint,
)
from scare.community.deliverability import per_actor_deliverable_caps
from scare.community.summary_state import (
    CrossSectorChannel,
    _ActiveCoalition,
    _ActiveCrossSectorCoalition,
    _CoalitionAggregate,
    _PendingCoalition,
)
from scare.community.supply_priority_admm import allocate_supply_priority

if TYPE_CHECKING:
    from scare.community.summary import HolonSummaryRole

logger = logging.getLogger(__name__)


class CoalitionManager:
    """Owns the pending/active coalition tables for both variants and drives
    them through invite -> accept -> allocate -> dispatch -> re-assert.
    """

    def __init__(self, role: HolonSummaryRole, cp_meta: dict | None = None) -> None:
        self._role = role
        # coalition_id -> invitation state awaiting acceptances.
        self._pending_coalitions: dict[str, Any] = {}
        # coalition_id -> allocated coalition held for its TTL.
        self._active_coalitions: dict[str, Any] = {}
        # Cross-sector (CP-mediated) counterpart of ``_active_coalitions``.
        self._active_xs_coalitions: dict[str, Any] = {}
        # Monotonic counter backing reproducible coalition ids.
        self._coalition_counter = 0
        # publisher_aid -> latest CPCommitment metadata for the XS path.
        self._cp_meta: dict[str, Any] = dict(cp_meta or {})

    async def _open_coalition(
        self,
        target_tiers: tuple[int, ...],
        demand_at_tier: dict[int, float],
    ) -> None:
        """Build the coalition member list and broadcast invitations.

        Members = publishers with non-zero demand in any target tier
        (plus self); zero-demand peers would just add message volume.
        """
        if not target_tiers:
            return
        member_aids: list[str] = [str(self._role.context.aid)]
        for aid, summary in self._role._peer_summaries.items():
            if aid == str(self._role.context.aid):
                continue
            if any(
                float(summary.per_tier_demand_mw.get(t, 0.0)) > 1e-9
                for t in target_tiers
            ):
                member_aids.append(aid)
        if len(member_aids) < 2:
            return

        self._coalition_counter += 1
        coalition_id = f"{self._role.context.aid}#{self._coalition_counter}"
        now = float(self._role.context.current_timestamp)
        pending = _PendingCoalition(
            coalition_id=coalition_id,
            sector=self._role.sector,
            target_tiers=tuple(target_tiers),
            member_aids=tuple(member_aids),
            started_at=now,
        )
        # Pre-seed our own acceptance locally; skip the round-trip.
        own_acc = self._role._local_acceptance(coalition_id, target_tiers)
        if own_acc is not None:
            pending.acceptances[str(self._role.context.aid)] = own_acc
        pending.addr_by_aid[str(self._role.context.aid)] = None  # sentinel

        invitation = CoalitionInvitation(
            publisher=str(self._role.context.aid),
            version=self._role._version.next(),
            caused_by={},
            timestamp_s=now,
            coalition_id=coalition_id,
            sector=self._role.sector,
            target_tiers=tuple(target_tiers),
            member_aids=tuple(member_aids),
            ttl_s=float(self._role.coalition_constraint_ttl_s),
        )

        n_sent = 0
        for aid in member_aids:
            if aid == str(self._role.context.aid):
                continue
            addr = self._role._peer_addrs.get(aid)
            if addr is None:
                continue
            pending.addr_by_aid[aid] = addr
            await self._role.context.send_message(invitation, receiver_addr=addr)
            n_sent += 1

        if n_sent == 0:
            return

        self._pending_coalitions[coalition_id] = pending
        logger.info(
            "[%s] coalition %s opened: tiers=%s members=%d (invitations=%d)",
            self._role.context.aid,
            coalition_id,
            target_tiers,
            len(member_aids),
            n_sent,
        )
        # Allocate after the acceptance window.
        try:
            self._role.context.schedule_timestamp_task(
                self._role._close_and_allocate(coalition_id),
                timestamp=now + float(self._role.coalition_accept_window_s),
            )
        except Exception:
            # No absolute-timestamp scheduling: fall back to instant with
            # whatever acceptances have arrived.
            self._role.context.schedule_instant_task(
                self._role._close_and_allocate(coalition_id)
            )

    async def _on_invitation(self, message: CoalitionInvitation, meta: dict) -> None:
        """Reply with an acceptance when in the member list.

        No flex gate: the initiator pre-filtered, and a no-flex member
        just contributes zeros.
        """
        own_aid = str(self._role.context.aid)
        if own_aid not in message.member_aids:
            return
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        target_tiers = tuple(message.target_tiers) if message.target_tiers else ()
        acceptance = self._role._local_acceptance(message.coalition_id, target_tiers)
        if acceptance is None:
            return
        await self._role.context.send_message(acceptance, receiver_addr=sender)

    def _local_acceptance(
        self,
        coalition_id: str,
        target_tiers_in: tuple[int, ...],
    ) -> CoalitionAcceptance | None:
        """Build this leader's acceptance payload from a fresh observe.

        Per-(sector, tier) demand (restricted to ``target_tiers_in``) +
        per-sector supply (all sectors, since the LP may route freed
        supply through CPs).
        """
        try:
            member_aids = [self._role.context.aid] + [
                addr.aid for addr in topology_neighbors(self._role, tid="groups")
            ]
        except Exception:
            member_aids = [self._role.context.aid]

        target_tiers = {int(t) for t in target_tiers_in if t is not None}
        supply_by_sector: dict[str, float] = {}
        demand_by_sector_priority: dict[str, dict[int, float]] = {}
        served_by_sector_priority: dict[str, dict[int, float]] = {}
        # Spatial demand footprint for per-actor reachability, keyed by
        # node (not aid) so the initiator needs no global aid→node map.
        demand_nodes_by_tier: dict[int, dict[Any, float]] = {}
        for aid in member_aids:
            try:
                obs = self._role.behavior.observe(aid) or {}
            except (AttributeError, KeyError):
                return None
            sec = obs_sector(obs, behavior=self._role.behavior, aid=aid)
            if sec is None:
                continue
            sec_v = sec.value
            # A promoted island reference is a generator, never a load: its free
            # Var reads 0 at init and flips positive when it absorbs, so the plain
            # cap-sign test below would drop it from supply and then bill it as
            # tiered demand (project_islanding_former_guard_off).
            if self._role._grid_former_policy.is_former(aid):
                supply_by_sector[sec_v] = supply_by_sector.get(
                    sec_v, 0.0
                ) + self._role._grid_former_policy.supply_credit(
                    aid, obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
                )
                continue
            cap = obs_capacity(obs, behavior=self._role.behavior, aid=aid)
            if cap < 0:  # generator-class — register supply
                # Slack advertises its (effective) budget (see publish).
                if lookup_slack(self._role.behavior, aid) is not None:
                    eff = lookup_slack_eff_budget(self._role.behavior, aid)
                    add = float(eff) if eff is not None else abs(float(cap))
                elif self._role.coalition_delivered_supply:
                    # Generator: credit DELIVERED |sp|, not RATED |cap| — a
                    # curtailed generator can't fund the pool at its nameplate
                    # (mirrors the L2 supply pool in balance.py).
                    sp = obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
                    add = abs(float(sp))
                else:
                    add = abs(float(cap))
                supply_by_sector[sec_v] = supply_by_sector.get(sec_v, 0.0) + add
                continue
            if cap <= 0:  # zero-capacity / passive — skip
                continue
            if sec != self._role.sector:
                continue  # sector-scoped invariant: skip other-sector demand
            tier = obs_priority(obs, behavior=self._role.behavior, aid=aid)
            if target_tiers and tier not in target_tiers:
                continue
            sp = obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
            demand_by_sector_priority.setdefault(sec_v, {})
            demand_by_sector_priority[sec_v][tier] = demand_by_sector_priority[
                sec_v
            ].get(tier, 0.0) + abs(float(cap))
            served_by_sector_priority.setdefault(sec_v, {})
            served_by_sector_priority[sec_v][tier] = served_by_sector_priority[
                sec_v
            ].get(tier, 0.0) + abs(float(sp))
            node = self._role._member_node_ids.get(str(aid))
            if node is not None:
                bucket = demand_nodes_by_tier.setdefault(tier, {})
                bucket[node] = bucket.get(node, 0.0) + abs(float(cap))

        return CoalitionAcceptance(
            publisher=str(self._role.context.aid),
            version=self._role._version.next(),
            caused_by={},
            timestamp_s=float(self._role.context.current_timestamp),
            coalition_id=coalition_id,
            sector=self._role.sector,
            accepted=True,
            supply_by_sector=supply_by_sector,
            demand_by_sector_priority=demand_by_sector_priority,
            served_by_sector_priority=served_by_sector_priority,
            home_node_id=self._role._my_node_id,
            demand_nodes_by_tier=demand_nodes_by_tier,
        )

    async def _on_acceptance(self, message: CoalitionAcceptance, meta: dict) -> None:
        pending = self._pending_coalitions.get(message.coalition_id)
        if pending is None or pending.run:
            return  # not ours, or already allocated
        sender = mango_sender_addr(meta)
        sender_aid = getattr(sender, "aid", None) or message.publisher or str(sender)
        if sender_aid not in pending.member_aids:
            return
        pending.acceptances[sender_aid] = message
        if sender is not None:
            pending.addr_by_aid[sender_aid] = sender

    async def _close_and_allocate(self, coalition_id: str) -> None:
        pending = self._pending_coalitions.pop(coalition_id, None)
        if pending is None or pending.run:
            return
        pending.run = True

        accepting = [a for a in pending.acceptances.values() if a.accepted]
        if len(accepting) < 2:
            logger.info(
                "[%s] coalition %s closed without enough acceptances "
                "(%d/%d), no allocation",
                self._role.context.aid,
                coalition_id,
                len(accepting),
                len(pending.member_aids),
            )
            return

        sector_str = self._role.sector.value
        agg = self._role._aggregate_coalition_supply_demand(accepting, sector_str)
        total_supply = agg.total_supply
        total_observed_served = agg.total_observed_served
        demand_by_tier = agg.demand_by_tier
        served_by_tier = agg.served_by_tier
        actor_supplies = agg.actor_supplies
        actor_demands = agg.actor_demands
        actor_node_ids = agg.actor_node_ids
        actor_demand_nodes_by_tier = agg.actor_demand_nodes_by_tier

        if not demand_by_tier:
            return

        tiers_for_admm = sorted(t for t in demand_by_tier.keys() if t >= 1)
        if not tiers_for_admm:
            return

        # Budget cap: when bottlenecked, scale supply by delivery vs
        # ``min(supply, demand)`` (raw nameplate over-scales when supply >> demand).
        total_demand_sector = sum(demand_by_tier.values())
        budget_scale = 1.0
        if (
            total_observed_served > 0.0
            and total_supply > 0.0
            and total_demand_sector > 1e-9
            and total_observed_served < total_demand_sector - 1e-9
        ):
            denom = max(min(total_supply, total_demand_sector), 1e-9)
            # Cap at 1.0 so delivery never inflates supply past nameplate.
            budget_scale = min(1.0, total_observed_served / denom)
            for supply_map in actor_supplies:
                if sector_str in supply_map:
                    supply_map[sector_str] = (
                        float(supply_map[sector_str]) * budget_scale
                    )

        # Deliverability caps: bound each actor's ub to its reachable
        # demand so it isn't committed to unreachable load. None ⇒ raw caps.
        try:
            actor_ub_overrides = per_actor_deliverable_caps(
                actor_node_ids=actor_node_ids,
                actor_demand_nodes_by_tier=actor_demand_nodes_by_tier,
                sector=self._role.sector,
                mirror=self._role._mirror,
            )
        except Exception as exc:
            logger.warning(
                "[%s] coalition %s: deliverability caps failed (%s) — "
                "falling back to raw supply",
                self._role.context.aid,
                coalition_id,
                exc,
            )
            actor_ub_overrides = None

        try:
            service_fraction_map, _per_actor_x, _meta = await allocate_supply_priority(
                sectors=[sector_str],
                tiers=tiers_for_admm,
                actor_supplies=actor_supplies,
                actor_demands=actor_demands,
                actor_ub_overrides=actor_ub_overrides,
                priority_tiers=self._role.priority_tiers,
                max_iters=int(self._role.admm_max_iters),
                abs_tol=float(self._role.admm_abs_tol),
            )
            fractions = dict(service_fraction_map.get(sector_str, {}))
            alloc_method = "admm"
        except Exception as exc:
            logger.warning(
                "[%s] coalition %s ADMM failed (%s) — falling back to "
                "centralised greedy",
                self._role.context.aid,
                coalition_id,
                exc,
            )
            record_event(
                t=float(self._role.context.current_timestamp),
                kind="coalition_admm_failed",
                aid=self._role.context.aid,
                sector=self._role.sector.value,
                detail=f"id={coalition_id} exc={exc!r}",
            )
            # Safety net: priority-greedy on the aggregate pool so a
            # solver hiccup never loses the coalition.
            remaining_supply = max(total_supply, 0.0)
            fractions = {}
            for tier in tiers_for_admm:
                dem = demand_by_tier[tier]
                if dem <= 1e-9:
                    fractions[tier] = 1.0
                    continue
                served = min(remaining_supply, dem)
                fractions[tier] = max(0.0, min(1.0, served / dem))
                remaining_supply -= served
            alloc_method = "greedy_fallback"

        n_capped = self._role._cap_fractions_by_feasibility(
            fractions, demand_by_tier, served_by_tier
        )

        record_event(
            t=float(self._role.context.current_timestamp),
            kind="coalition_allocation",
            aid=self._role.context.aid,
            sector=self._role.sector.value,
            detail=(
                f"id={coalition_id} method={alloc_method} "
                f"tiers={pending.target_tiers} "
                f"n_accept={len(accepting)} supply={total_supply:.4f} "
                f"observed_served={total_observed_served:.4f} "
                f"budget_scale={budget_scale:.3f} "
                f"feasibility_caps={n_capped} "
                f"demand_by_tier={{"
                f"{', '.join(f'{t}: {v:.4f}' for t, v in sorted(demand_by_tier.items()))}}} "
                f"served_by_tier={{"
                f"{', '.join(f'{t}: {v:.4f}' for t, v in sorted(served_by_tier.items()))}}} "
                f"fractions={{{', '.join(f'{t}: {v:.3f}' for t, v in sorted(fractions.items()))}}}"
            ),
        )
        logger.info(
            "[%s] coalition %s allocated (%s): supply=%.4f "
            "demand_by_tier=%s fractions=%s",
            self._role.context.aid,
            coalition_id,
            alloc_method,
            total_supply,
            {t: round(v, 4) for t, v in sorted(demand_by_tier.items())},
            {t: round(v, 3) for t, v in sorted(fractions.items())},
        )

        # Record the active coalition so each ``_tick`` re-asserts the
        # constraint until TTL expiry.
        addrs: list[Any] = []
        for acc in accepting:
            aid = acc.publisher
            if aid == str(self._role.context.aid):
                continue  # initiator dispatches to itself separately
            addr = pending.addr_by_aid.get(aid)
            if addr is not None:
                addrs.append(addr)
        now = float(self._role.context.current_timestamp)
        active = _ActiveCoalition(
            coalition_id=coalition_id,
            sector=self._role.sector,
            service_fraction_by_tier=fractions,
            member_addrs=addrs,
            issued_at=now,
            ttl_s=float(self._role.coalition_constraint_ttl_s),
        )
        self._active_coalitions[coalition_id] = active

        # Persist locally + broadcast so every member's L2 ADMM merges
        # the same fractions for the TTL window instead of overwriting.
        if self._role._constraint_store is not None:
            self._role._constraint_store.set(
                coalition_id=coalition_id,
                sector=self._role.sector,
                service_fraction_by_tier=fractions,
                issued_at=now,
                ttl_s=float(self._role.coalition_constraint_ttl_s),
            )
        constraint_msg = CoalitionConstraint(
            publisher=str(self._role.context.aid),
            version=self._role._version.next(),
            caused_by={},
            timestamp_s=now,
            coalition_id=coalition_id,
            sector=self._role.sector,
            service_fraction_by_tier=dict(fractions),
            ttl_s=float(self._role.coalition_constraint_ttl_s),
        )
        # Broadcast to every same-sector peer leader, not just members, so
        # an outside chunk initiator's L2 dispatch also merges these first.
        broadcast_targets = set()
        for addr in active.member_addrs:
            broadcast_targets.add(addr)
        for aid, addr in self._role._peer_addrs.items():
            if aid == str(self._role.context.aid):
                continue
            broadcast_targets.add(addr)
        # Self-loop so the initiator's own L2 rebalance merges the new
        # fractions immediately rather than waiting for its heartbeat.
        own_addr = getattr(self._role.context, "addr", None)
        if own_addr is not None:
            broadcast_targets.add(own_addr)
        for addr in broadcast_targets:
            await self._role.context.send_message(constraint_msg, receiver_addr=addr)

        await self._role._dispatch_active_coalition(active)

    async def _dispatch_active_coalition(self, active: _ActiveCoalition) -> None:
        """Send the constraint as a StartBalanceNegotiation to every accepting
        member plus self, tier-monotonic-clamped (mirrors the
        component-allocation path) so a coalition map can't serve a
        lower-priority tier above a higher one.
        """
        service_fraction_by_sector_priority = {
            active.sector.value: clamp_tier_monotonic(
                dict(active.service_fraction_by_tier)
            ),
        }
        payload = StartBalanceNegotiation(
            service_fraction_by_sector_priority=service_fraction_by_sector_priority,
        )
        for addr in active.member_addrs:
            await self._role.context.send_message(payload, receiver_addr=addr)
        # Self so the message lands on this leader's own
        # EnergyBalanceNegotiator via the same handler path L2 uses.
        own_addr = getattr(self._role.context, "addr", None)
        if own_addr is not None:
            await self._role.context.send_message(payload, receiver_addr=own_addr)

    async def _reassert_active_coalitions(self) -> None:
        """Per-tick TTL pruning + re-broadcast.

        Re-broadcasting each tick is what holds a constraint against L2:
        if L2 overwrites the L1 fraction, the next tick re-asserts it.
        """
        now = float(self._role.context.current_timestamp)
        # Prune expired store records so L2's merge sees only valid ones.
        if self._role._constraint_store is not None:
            self._role._constraint_store.prune(now)
        if self._active_coalitions:
            expired: list[str] = []
            for coalition_id, active in self._active_coalitions.items():
                if now > active.issued_at + active.ttl_s:
                    expired.append(coalition_id)
                    continue
                await self._role._dispatch_active_coalition(active)
            for cid in expired:
                self._active_coalitions.pop(cid, None)
                logger.debug(
                    "[%s] coalition %s expired (ttl reached)",
                    self._role.context.aid,
                    cid,
                )
        # Cross-sector coalitions: same shape, separate dispatch.
        if self._active_xs_coalitions:
            expired_xs: list[str] = []
            for cid, active in self._active_xs_coalitions.items():
                if now > active.issued_at + active.ttl_s:
                    expired_xs.append(cid)
                    continue
                await self._role._dispatch_active_xs_coalition(active)
            for cid in expired_xs:
                self._active_xs_coalitions.pop(cid, None)
                logger.debug(
                    "[%s] cross-sector coalition %s expired (ttl reached)",
                    self._role.context.aid,
                    cid,
                )

    @staticmethod
    def _cap_fractions_by_feasibility(
        fractions: dict[int, float],
        demand_by_tier: dict[int, float],
        served_by_tier: dict[int, float],
    ) -> int:
        """No-op shim kept for callers that read its return value.

        Capping at the observed served fraction self-reinforced into a
        lock at the degenerate point; budgeting now lives in
        ``budget_scale`` + the ADMM box constraints.
        """
        return 0

    def _aggregate_coalition_supply_demand(
        self,
        accepting: list[CoalitionAcceptance],
        sector_str: str,
    ) -> _CoalitionAggregate:
        """Sum supply/demand/served across accepting members; copy
        per-actor inputs into fresh structures for the allocator.
        """
        total_supply = 0.0
        total_observed_served = 0.0
        demand_by_tier: dict[int, float] = {}
        served_by_tier: dict[int, float] = {}
        actor_supplies: list[dict[str, float]] = []
        actor_demands: list[dict[str, dict[int, float]]] = []
        actor_node_ids: list[Any] = []
        actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]] = []
        for acc in accepting:
            total_supply += float(acc.supply_by_sector.get(sector_str, 0.0))
            for tier, dem in acc.demand_by_sector_priority.get(sector_str, {}).items():
                demand_by_tier[tier] = demand_by_tier.get(tier, 0.0) + float(dem)
            for tier, srv in acc.served_by_sector_priority.get(sector_str, {}).items():
                served_by_tier[tier] = served_by_tier.get(tier, 0.0) + float(srv)
                total_observed_served += float(srv)
            actor_supplies.append(dict(acc.supply_by_sector))
            actor_demands.append(
                {k: dict(v) for k, v in acc.demand_by_sector_priority.items()}
            )
            actor_node_ids.append(acc.home_node_id)
            actor_demand_nodes_by_tier.append(
                {
                    tier: dict(nodes)
                    for tier, nodes in (acc.demand_nodes_by_tier or {}).items()
                }
            )
        return _CoalitionAggregate(
            total_supply=total_supply,
            total_observed_served=total_observed_served,
            demand_by_tier=demand_by_tier,
            served_by_tier=served_by_tier,
            actor_supplies=actor_supplies,
            actor_demands=actor_demands,
            actor_node_ids=actor_node_ids,
            actor_demand_nodes_by_tier=actor_demand_nodes_by_tier,
        )

    async def _open_cross_sector_coalition(
        self,
        *,
        cp_aid: str,
        own_sec: Sector,
        peer_sec: Sector,
        t_own_high: int,
        t_peer_low: int,
    ) -> None:
        """Build a cross-sector allocation and dispatch it immediately.

        No invitation round: uses build-time ``_cp_meta`` (CP capacity +
        coupling) and the registry's current per-tier state.
        """
        # Don't raise own-sector service fractions on a CP transfer that has no
        # actuator: under the default priority-ADMM L3 there is no CPCommitment
        # consumer, so the promised inflow never materialises and the raised
        # fractions get funded by the slack instead (the child-118 overdraw).
        if (
            self._role.coalition_delivered_supply
            and not self._role.cp_commitment_actuatable
        ):
            record_event(
                t=float(self._role.context.current_timestamp),
                kind="cross_sector_coalition_skipped_unactuatable",
                aid=self._role.context.aid,
                sector=own_sec.value,
                detail=f"cp={cp_aid} peer={peer_sec.value} (no CPCommitment consumer)",
            )
            return

        meta = self._cp_meta.get(cp_aid)
        if meta is None:
            return
        coupling = meta.get("coupling_ratios", {}) or {}
        rated = meta.get("rated_capacity_mw", {}) or {}
        cp_addr = meta.get("addr")
        if cp_addr is None:
            return

        # CP pushes into own_sec and draws from peer_sec. Coupling keyed
        # (in, out).
        key = (peer_sec.value, own_sec.value)
        eta = float(coupling.get(key, 0.0))
        cp_cap_out = float(rated.get(own_sec.value, 0.0))
        if eta <= 0.0 or cp_cap_out <= 0.0:
            return  # CP can't push this direction

        channel = CrossSectorChannel.for_behavior(self._role.behavior)
        own_summaries = channel.read(own_sec)
        peer_summaries = channel.read(peer_sec)

        own_dem, own_ser = self._role._aggregate_tier(own_summaries)
        peer_dem, peer_ser = self._role._aggregate_tier(peer_summaries)

        deficit_own_high = max(
            0.0, own_dem.get(t_own_high, 0.0) - own_ser.get(t_own_high, 0.0)
        )
        served_peer_low = peer_ser.get(t_peer_low, 0.0)
        # Own-sec MW the CP can produce (after η) from peer's tier_low.
        peer_freeable_own = served_peer_low * eta

        # Bounded by deficit, peer's available served, and CP rated.
        transfer_out = min(deficit_own_high, peer_freeable_own, cp_cap_out)
        if transfer_out <= 1e-6:
            logger.info(
                "[%s] cross-sector coalition skipped: nothing to transfer "
                "(deficit=%.4f, peer_freeable=%.4f, cp_cap=%.4f)",
                self._role.context.aid,
                deficit_own_high,
                peer_freeable_own,
                cp_cap_out,
            )
            return
        transfer_in = transfer_out / eta

        new_ser_own = own_ser.get(t_own_high, 0.0) + transfer_out
        own_total = own_dem.get(t_own_high, 0.0)
        new_frac_own = min(1.0, new_ser_own / own_total) if own_total > 1e-9 else 1.0

        new_ser_peer = max(0.0, served_peer_low - transfer_in)
        peer_total = peer_dem.get(t_peer_low, 0.0)
        new_frac_peer = (
            max(0.0, new_ser_peer / peer_total) if peer_total > 1e-9 else 1.0
        )

        service_fraction_by_sector_tier: dict[str, dict[int, float]] = {
            own_sec.value: {t_own_high: new_frac_own},
            peer_sec.value: {t_peer_low: new_frac_peer},
        }
        cp_targets: dict[str, dict[str, float]] = {
            cp_aid: {own_sec.value: +transfer_out, peer_sec.value: -transfer_in}
        }

        # Leader addresses: own_sec from our peer book, peer_sec from the
        # scenario-supplied ``_peer_leader_addrs``.
        own_addrs: list[Any] = []
        for aid in own_summaries:
            if aid == str(self._role.context.aid):
                continue
            addr = self._role._peer_addrs.get(aid)
            if addr is not None:
                own_addrs.append(addr)
        peer_addrs: list[Any] = []
        peer_book = self._role._peer_leader_addrs.get(peer_sec, {})
        for aid in peer_summaries:
            addr = peer_book.get(aid)
            if addr is not None:
                peer_addrs.append(addr)

        self._coalition_counter += 1
        coalition_id = f"xs:{self._role.context.aid}#{self._coalition_counter}"
        now = float(self._role.context.current_timestamp)
        active = _ActiveCrossSectorCoalition(
            coalition_id=coalition_id,
            service_fraction_by_sector_tier=service_fraction_by_sector_tier,
            leader_addrs_by_sector={
                own_sec.value: own_addrs,
                peer_sec.value: peer_addrs,
            },
            cp_targets_mw=cp_targets,
            cp_addrs={cp_aid: cp_addr},
            sectors=(own_sec, peer_sec),
            issued_at=now,
            ttl_s=float(self._role.coalition_constraint_ttl_s),
        )
        self._active_xs_coalitions[coalition_id] = active

        # Persist so L2 (per-sector ADMM) and L3 (CP ADMM) see it.
        if self._role._constraint_store is not None:
            self._role._constraint_store.set(
                coalition_id=coalition_id,
                sector=own_sec,
                service_fraction_by_tier={t_own_high: new_frac_own},
                issued_at=now,
                ttl_s=float(self._role.coalition_constraint_ttl_s),
            )
            # Distinct id so the peer-side record isn't overwritten.
            self._role._constraint_store.set(
                coalition_id=f"{coalition_id}/peer",
                sector=peer_sec,
                service_fraction_by_tier={t_peer_low: new_frac_peer},
                issued_at=now,
                ttl_s=float(self._role.coalition_constraint_ttl_s),
            )
            self._role._constraint_store.set_cp_envelope(
                coalition_id=coalition_id,
                cp_id=cp_aid,
                target_flows_mw=cp_targets[cp_aid],
                issued_at=now,
                ttl_s=float(self._role.coalition_constraint_ttl_s),
            )

        record_event(
            t=now,
            kind="cross_sector_coalition_allocation",
            aid=self._role.context.aid,
            sector=own_sec.value,
            detail=(
                f"id={coalition_id} cp={cp_aid} "
                f"transfer_out={transfer_out:.4f} transfer_in={transfer_in:.4f} "
                f"own_frac={{{t_own_high}: {new_frac_own:.3f}}} "
                f"peer_frac={{{t_peer_low}: {new_frac_peer:.3f}}}"
            ),
        )

        await self._role._dispatch_active_xs_coalition(active)

    async def _dispatch_active_xs_coalition(
        self, active: _ActiveCrossSectorCoalition
    ) -> None:
        """Send one ``StartBalanceNegotiation`` per sector + one
        ``CPCommitment`` per CP. Idempotent (initial fire + re-asserts).
        """
        for sec_v, addrs in active.leader_addrs_by_sector.items():
            tier_map = active.service_fraction_by_sector_tier.get(sec_v, {})
            if not tier_map:
                continue
            payload = StartBalanceNegotiation(
                service_fraction_by_sector_priority={sec_v: dict(tier_map)},
            )
            for addr in addrs:
                await self._role.context.send_message(payload, receiver_addr=addr)
        # Self-dispatch so own L1 applies the fraction too.
        own_v = self._role.sector.value
        own_tier_map = active.service_fraction_by_sector_tier.get(own_v, {})
        if own_tier_map:
            own_addr = getattr(self._role.context, "addr", None)
            if own_addr is not None:
                await self._role.context.send_message(
                    StartBalanceNegotiation(
                        service_fraction_by_sector_priority={own_v: dict(own_tier_map)},
                    ),
                    receiver_addr=own_addr,
                )
        for cp_aid, flows in active.cp_targets_mw.items():
            cp_addr = active.cp_addrs.get(cp_aid)
            if cp_addr is None:
                continue
            commit = CPCommitment(
                publisher=str(self._role.context.aid),
                version=self._role._version.next(),
                caused_by={},
                timestamp_s=float(self._role.context.current_timestamp),
                coalition_id=active.coalition_id,
                cp_id=cp_aid,
                target_flows_mw=dict(flows),
                ttl_s=float(active.ttl_s),
            )
            await self._role.context.send_message(commit, receiver_addr=cp_addr)
