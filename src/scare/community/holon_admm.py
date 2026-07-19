"""Per-holon L2 ADMM solvers: the legacy multi-sector sharing formulation and
the tier-stratified supply-priority formulation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    create_admm_sharing_data,
    create_admm_start,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.flex_actor import (
    ADMMFlexActor,
)
from mango.express.topology import topology_connectors

from scare.base.channel import HolonAllocation
from scare.base.model import (
    AvailableFlexAnswer,
    HolonicAssignment,
    StartBalanceNegotiation,
)
from scare.base.runtime.trace import optimization
from scare.base.util import clamp_tier_monotonic, compute_priority_weighted_shares
from scare.community.deliverability import per_actor_deliverable_caps
from scare.community.holon_flex import extract_demand_sectors_tiers
from scare.community.supply_priority_admm import allocate_supply_priority

logger = logging.getLogger(__name__)


class SectorAdmmRunner:
    """Runs the holon-scoped ADMM over the collected member flex and pushes the
    result down to the member leaders. Reads round state, config and the peer
    resolver through its owning role.
    """

    def __init__(self, role: Any) -> None:
        self._role = role

    def has_supply_priority_work(self) -> bool:
        """True iff queued answers have both per-tier demand and supply."""
        answers = self._role._round.answers
        any_demand = any(a.demand_by_sector_priority for a in answers)
        any_supply = any(
            float(s) > 1e-9
            for a in answers
            for s in (a.supply_by_sector or {}).values()
        )
        return any_demand and any_supply

    async def run_legacy(self) -> None:
        """Run ADMM sharing across member groups.

        Each group is a multi-dimensional actor (one dim per sector), so
        resources balance across sectors at once (e.g. gas surplus covering
        heat deficit via CHP). ``T`` is per-sector total imbalance. Priority-
        aware: ``S`` pulls each actor toward its priority-weighted share.
        """
        role = self._role
        rnd = role._round

        if not rnd.active:
            return
        answers, senders = rnd.drain()

        if len(answers) < 2:
            return

        # Union of sectors across member groups.
        all_sectors: list[str] = sorted(
            {s for a in answers for s in a.balance_by_sector}
        )
        n_dims = len(all_sectors) if all_sectors else 1

        # Fall back to 1D if no per-sector data.
        if n_dims <= 1 and not any(a.balance_by_sector for a in answers):
            all_sectors = [role.sector.value]
            n_dims = 1

        sector_idx = {s: i for i, s in enumerate(all_sectors)}

        actors: list[ADMMFlexActor] = []
        total_T = np.zeros(n_dims)

        # Per-group bounds first, then priority shares.
        group_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for answer in answers:
            lb = np.zeros(n_dims)
            ub = np.full(n_dims, 1e-6)

            if answer.balance_by_sector:
                for sec, bal in answer.balance_by_sector.items():
                    if sec in sector_idx:
                        i = sector_idx[sec]
                        lb[i] = -abs(bal)
                        flex_val = answer.flex_by_sector.get(sec, 0.0)
                        ub[i] = max(abs(flex_val), 1e-6)
                        total_T[i] += bal
            else:
                lb[0] = -abs(answer.balance)
                ub[0] = max(abs(answer.flex), 1e-6)
                total_T[0] += answer.balance

            group_bounds.append((lb, ub))

        if not np.all(np.isfinite(total_T)):
            logger.warning(
                "[%s] holon ADMM skipped: non-finite target T=%s",
                role.context.aid,
                total_T.tolist(),
            )
            rnd.active = False
            return

        # Feasibility cap: when |T[i]| exceeds available budgets, the distance
        # term plateaus and the library spuriously logs "max iterations".
        # Bound |T| by the budget envelope (sign preserved).
        budget_pos = np.zeros(n_dims)
        budget_neg = np.zeros(n_dims)
        for lb_g, ub_g in group_bounds:
            budget_pos += np.maximum(ub_g, 0.0)
            budget_neg += np.maximum(-lb_g, 0.0)
        for i in range(n_dims):
            if total_T[i] > 0 and budget_pos[i] < total_T[i]:
                total_T[i] = budget_pos[i]
            elif total_T[i] < 0 and budget_neg[i] < -total_T[i]:
                total_T[i] = -budget_neg[i]

        if np.all(np.abs(total_T) < 1e-6):
            logger.info(
                "[%s] holon ADMM skipped: balanced (sectors=%s)",
                role.context.aid,
                all_sectors,
            )
            rnd.active = False
            return

        # Priority-weighted S: waterfall shares under strict priority ordering.
        # Budget = surplus + flex headroom; the flex term keeps the budget
        # positive in a pure-deficit holon with no surplus.
        total_surplus = sum(max(0.0, -a.balance) for a in answers)
        total_flex = sum(max(0.0, a.flex) for a in answers)
        total_available = total_surplus + total_flex
        if role.enable_priority_allocation:
            priority_shares = compute_priority_weighted_shares(
                [a.demand_by_priority for a in answers],
                [a.served_by_priority for a in answers],
                total_available,
            )
        else:
            # Priority disabled — distribute uniformly (balance-only ablation).
            even = total_available / max(1, len(answers))
            priority_shares = [even for _ in answers]

        for idx, answer in enumerate(answers):
            lb, ub = group_bounds[idx]
            C = np.zeros((0, n_dims))
            d = np.zeros(0)
            # S = local-QP linear cost (negative attracts), proportional to the
            # priority-weighted share so ADMM steers toward high-priority demand.
            S = np.zeros(n_dims)
            if priority_shares[idx] > 1e-9:
                # Normalise by total target (stability vs rho), then split the
                # pull across dims by per-dimension balance.
                t_mag = max(np.max(np.abs(total_T)), 1e-6)
                pull = priority_shares[idx] / t_mag
                if answer.balance_by_sector and n_dims > 1:
                    bal_vec = np.zeros(n_dims)
                    for sec, bal in answer.balance_by_sector.items():
                        if sec in sector_idx:
                            bal_vec[sector_idx[sec]] = abs(bal)
                    bal_sum = np.sum(bal_vec)
                    if bal_sum > 1e-9:
                        S = -(pull * bal_vec / bal_sum)
                    else:
                        S = np.full(n_dims, -pull / n_dims)
                else:
                    S[0] = -pull
            lb = np.nan_to_num(lb, nan=0.0, posinf=0.0, neginf=0.0)
            ub = np.nan_to_num(ub, nan=1e-6, posinf=1e6, neginf=1e-6)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            actors.append(ADMMFlexActor(lb=lb, u=ub, C=C, d=d, S=S))

        coordinator = create_sharing_target_distance_admm_coordinator()
        # Iter cap / tol from config. Defaults (50 @ 1e-3) relaxed from 1000 /
        # 1e-4 so concurrent ADMMs don't block discrete-time progress.
        coordinator.max_iters = int(role.admm_max_iters)
        coordinator.abs_tol = float(role.admm_abs_tol)
        start_msg = create_admm_start(create_admm_sharing_data(total_T.tolist()))

        try:
            with optimization(
                "admm_holon",
                logger=logger,
                aid=role.context.aid,
                n_actors=len(actors),
                sectors=all_sectors,
            ):
                await start_coordinated_optimization(actors, coordinator, start_msg)
            results = [a.x.tolist() for a in actors]
            logger.info(
                "[%s] holon ADMM result (sectors=%s): %s (T=%s)",
                role.context.aid,
                all_sectors,
                results,
                total_T.tolist(),
            )
            role._record_event(
                "holon_admm_result",
                f"sectors={all_sectors} T={total_T.tolist()}",
            )
        except Exception as exc:
            logger.error("[%s] holon ADMM failed: %s", role.context.aid, exc)
            role._record_event("holon_admm_failed", str(exc))
            # On failure still trigger intra-group gossip for local rebalance.

        # Trigger intra-group rebalancing, routing each allocation as the gossip
        # override target. ``actors[idx].x`` is a vector over ``all_sectors``;
        # pick the member's sector and negate (target = -allocation). Unmapped
        # members fall back to local recompute.
        sender_to_actor: dict[str, tuple[Any, AvailableFlexAnswer]] = {}
        for sender, answer, actor in zip(senders, answers, actors):
            sender_to_actor[str(sender)] = (actor, answer)

        triggers = role._resolve_holon_members()
        # Carry per-member overrides on ``HolonAllocation`` AND push the legacy
        # ``StartBalanceNegotiation`` so L1's override path keeps working.
        allocation_targets: dict[str, float] = {}
        for addr in triggers:
            entry = sender_to_actor.get(str(addr))
            override: float | None = None
            if entry is not None:
                actor_obj, answer = entry
                # Index actor_obj.x by the member's natural sector.
                try:
                    x_vec = list(actor_obj.x)
                    if answer.sector.value in sector_idx:
                        override = -float(x_vec[sector_idx[answer.sector.value]])
                    elif x_vec:
                        override = -float(x_vec[0])
                except Exception:  # pragma: no cover - defensive
                    override = None
            if override is not None:
                allocation_targets[str(addr)] = override
            await role.context.send_message(
                StartBalanceNegotiation(override_target=override),
                receiver_addr=addr,
            )

        # Publish HolonAllocation to CP connectors so L3 reacts directly,
        # skipping the L2->L1->gossip->L3 detour (``tid="groups"`` link).
        if allocation_targets:
            try:
                cp_connectors = list(topology_connectors(role, tid="groups"))
            except Exception:
                cp_connectors = []
            if cp_connectors:
                assignment = role.context.get_or_create_model(HolonicAssignment)
                holon_id = str(assignment.holon_id) if assignment.holon_id else ""
                decision = HolonAllocation(
                    publisher=str(role.context.aid),
                    version=role._version.next(),
                    caused_by={},
                    timestamp_s=float(role.context.current_timestamp),
                    sector=role.sector,
                    targets_mw=allocation_targets,
                    holon_id=holon_id,
                    residual=0.0,
                )
                logger.debug(
                    "[%s] holon publish: sector=%s n_targets=%d v=%d to %d cps",
                    role.context.aid,
                    role.sector.value,
                    len(allocation_targets),
                    decision.version,
                    len(cp_connectors),
                )
                for addr in cp_connectors:
                    await role.context.send_message(decision, receiver_addr=addr)

    async def run_supply_priority(self) -> None:
        """Supply-priority ADMM.

        Differs from the demand-side path: ``T`` is total demand per
        (sector, tier) cell (priority-weighted), and each actor's ``ub`` /
        coupling reflect generator capacity, so a supply-rich group can serve
        holon-wide tier-X demand. Output: per-(sector, tier) service fractions
        sent to each leader, giving a consistent shed-low/serve-high pattern.
        """
        role = self._role
        rnd = role._round
        if not rnd.active:
            return
        answers, senders = rnd.drain()

        if len(answers) < 2:
            return

        actor_supplies = [a.supply_by_sector or {} for a in answers]
        actor_demands = [a.demand_by_sector_priority or {} for a in answers]

        # Active cells. A sector with supply but no demand bypasses the ADMM;
        # no demand anywhere ⇒ fall back to the legacy path.
        sectors, tiers, total_demand = extract_demand_sectors_tiers(actor_demands)
        if not sectors or not tiers or total_demand < 1e-6:
            rnd.restore(answers, senders)
            await self.run_legacy()
            return

        # F6 deliverability caps: when member nodes + mirror are available,
        # cap each cell at demand reachable from the actor's home node so the
        # ADMM never commits unroutable supply. Unreachable leader ⇒ caps 0;
        # reachable ⇒ uncapped (None), so coupling is the only binding constraint.
        peers = role._peers
        actor_ub_overrides: list[dict[tuple[str, int], float] | None] | None = None
        if peers.topology_mirror is not None and peers.leader_node_ids:
            try:
                actor_node_ids: list[Any | None] = []
                actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]] = []
                for sender, answer in zip(senders, answers):
                    leader_aid = getattr(sender, "aid", str(sender))
                    node_id = peers.leader_node_ids.get(leader_aid)
                    actor_node_ids.append(node_id)
                    # Map this leader's tier-aggregated demand onto its home
                    # node; an unreachable leader contributes nothing.
                    per_tier: dict[int, dict[Any, float]] = {}
                    if node_id is not None:
                        for sec, tier_map in (
                            answer.demand_by_sector_priority or {}
                        ).items():
                            if sec not in sectors:
                                continue
                            for tier, dem in tier_map.items():
                                # Bind inner dict first: a combined
                                # ``setdefault(k,{})[n] = ...[k].get(...)`` would
                                # KeyError (RHS evaluated before LHS subscript).
                                inner = per_tier.setdefault(int(tier), {})
                                inner[node_id] = inner.get(node_id, 0.0) + float(dem)
                    actor_demand_nodes_by_tier.append(per_tier)

                actor_ub_overrides = per_actor_deliverable_caps(
                    actor_node_ids=actor_node_ids,
                    actor_demand_nodes_by_tier=actor_demand_nodes_by_tier,
                    sector=role.sector,
                    mirror=peers.topology_mirror,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] supply-priority holon: deliverability caps "
                    "failed (%s: %s) — falling back to raw supply",
                    role.context.aid,
                    type(exc).__name__,
                    exc,
                )
                actor_ub_overrides = None

        try:
            with optimization(
                "admm_supply_priority",
                logger=logger,
                scope="holon",
                aid=role.context.aid,
                n_actors=len(actor_supplies),
            ):
                service_fraction, _per_actor_x, meta = await allocate_supply_priority(
                    sectors=sectors,
                    tiers=tiers,
                    actor_supplies=actor_supplies,
                    actor_demands=actor_demands,
                    actor_ub_overrides=actor_ub_overrides,
                    priority_tiers=role.priority_tiers,
                    max_iters=int(role.admm_max_iters),
                    abs_tol=float(role.admm_abs_tol),
                    enable_priority_weighting=role.enable_priority_allocation,
                )
        except Exception as exc:
            logger.error(
                "[%s] supply-priority holon ADMM failed: %s",
                role.context.aid,
                exc,
            )
            role._record_event("holon_admm_failed", f"supply_priority: {exc}")
            return

        total_T = meta["T_per_cell"]
        sum_x_per_cell = meta["sum_x_per_cell"]
        priorities = meta["priorities"]
        n_tier = len(tiers)
        sec_idx = {s: i for i, s in enumerate(sectors)}
        tier_idx = {t: j for j, t in enumerate(tiers)}

        def _flat_idx(s: str, t: int) -> int:
            return sec_idx[s] * n_tier + tier_idx[t]

        logger.info(
            "[%s] supply-priority holon ADMM result: sectors=%s tiers=%s "
            "T=%s sum_x=%s holon_supply=%.4f",
            role.context.aid,
            sectors,
            tiers,
            total_T,
            sum_x_per_cell,
            sum(meta["actor_supply_total"]),
        )
        role._record_event(
            "holon_admm_result",
            f"supply_priority sectors={sectors} tiers={tiers} fractions={service_fraction}",
        )
        role._record_event(
            "holon_priority_allocation",
            str(
                {
                    f"{sec}:tier{tier}": {
                        "T": round(float(total_T[_flat_idx(sec, tier)]), 6),
                        "weight": float(priorities[_flat_idx(sec, tier)]),
                        "sum_x": round(float(sum_x_per_cell[_flat_idx(sec, tier)]), 6),
                        "service_frac": round(service_fraction[sec][tier], 4),
                    }
                    for sec in sectors
                    for tier in tiers
                }
            ),
        )

        # Coalition merge: an active coalition fraction overrides L2's per-cell
        # result; untouched cells keep L2's. Without it, last-write-wins would
        # reset the regulation each round.
        if role._coalition_constraint_store is not None:
            now = float(role.context.current_timestamp)
            service_fraction = role._coalition_constraint_store.merge_into(
                service_fraction,
                role.sector,
                now,
            )
            # Post-merge tier-monotonic clamp: same rationale as the
            # component-allocation path — a per-tier coalition override must
            # not lift a lower tier above a higher one.
            for tier_map in service_fraction.values():
                clamp_tier_monotonic(tier_map)

        # Send the holon-global fraction map to every member leader (L1 honour).
        triggers = role._resolve_holon_members()
        for addr in triggers:
            await role.context.send_message(
                StartBalanceNegotiation(
                    service_fraction_by_sector_priority=service_fraction,
                ),
                receiver_addr=addr,
            )
