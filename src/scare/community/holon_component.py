"""Per-(sector, active-component) L2 ADMM (``admm_scope="component"``).

One ADMM per active connected component; each leader is one actor (the holon
abstraction is unused). Per round: each leader self-collects flex, pushes a
``ComponentAdmmReport`` to the coordinator (lex-smallest reachable leader),
which debounces a burst into one solve and dispatches a ``ComponentAllocation``
to every leader, who applies it to its members. Invariant: every load at the
same tier in a (sector, component) is served at the same fraction, so
cross-leader inversions cannot arise.
"""

from __future__ import annotations

import logging
from typing import Any

from mango.express.topology import topology_characteristic

from scare.base.channel import (
    ComponentAdmmReport,
    ComponentAllocation,
)
from scare.base.model import StartBalanceNegotiation
from scare.base.runtime.trace import optimization
from scare.base.util import clamp_tier_monotonic
from scare.community.holon_flex import (
    aggregate_holon_flex,
    extract_demand_sectors_tiers,
)
from scare.community.supply_priority_admm import allocate_supply_priority

logger = logging.getLogger(__name__)

# Two fraction maps within this per-tier tolerance count as the same
# allocation (matches the actuator dedup tolerance, so a skipped re-dispatch
# would have been all no-ops anyway).
_FRACTION_EQUAL_TOL: float = 1e-3


def _fraction_maps_equal(
    a: dict[str, dict[int, float]],
    b: dict[str, dict[int, float]],
    *,
    tol: float = _FRACTION_EQUAL_TOL,
) -> bool:
    if set(a) != set(b):
        return False
    for sec in a:
        ta, tb = a[sec], b[sec]
        if set(ta) != set(tb):
            return False
        if any(abs(ta[t] - tb[t]) > tol for t in ta):
            return False
    return True


class ComponentCoordinator:
    """Both sides of the component-scoped L2 round: the coordinator's report
    buffer + debounced solve + versioned dispatch, and the leaf's apply/ACK.
    Reads round state, peers and config through its owning role.
    """

    def __init__(self, role: Any) -> None:
        self._role = role
        # The coordinator (lex-smallest reachable leader aid) buffers
        # ``ComponentAdmmReport`` per leader and runs one ADMM over N actors.
        # ``dispatch_pending`` debounces a report burst into one solve.
        self.round_counter: int = 0
        self.report_buffer: dict[str, tuple[str, Any]] = {}
        self.dispatch_pending: bool = False
        # Latest dispatched fraction; new reports merge against it.
        self.last_fraction: dict[str, dict[int, float]] | None = None

        # Monotone counter per outgoing ``ComponentAllocation``. Receivers echo
        # the last version applied; the coordinator re-sends to stale receivers.
        self.allocation_version_counter: int = 0
        # Latest dispatched allocation, re-sent to stale leaders. None until first.
        self.last_dispatched_allocation: Any = None  # ComponentAllocation
        # Latest (coordinator aid, version) applied as a leaf. Versions are
        # per-publisher (re-elected coordinator's low versions aren't judged vs
        # the old high counter); echoed only when publisher == coordinator.
        self.last_applied_version: int = -1
        self.last_applied_publisher: str | None = None

    async def run_scoped(self) -> None:
        """Component-scoped variant of the supply-priority solve.

        Flex already collected. Coordinator buffers its own report + schedules
        a debounced solve; non-coordinator pushes a ``ComponentAdmmReport`` and
        awaits the ``ComponentAllocation``. The round is drained so a follow-up
        trigger doesn't re-fire on the stale buffer.
        """
        role = self._role
        rnd = role._round
        if not rnd.active:
            return
        answers, senders = rnd.drain()

        if not answers:
            return

        supply, demand, served = aggregate_holon_flex(answers)
        # Report on any supply OR demand: demand-only feeds the T vector,
        # supply-only the pool. ``and`` would drop load-only communities.
        any_supply = any(v > 1e-9 for v in supply.values())
        any_demand = any(mw > 1e-9 for tmap in demand.values() for mw in tmap.values())
        if not (any_supply or any_demand):
            return

        coord_aid = role._component_coordinator_aid()
        if coord_aid is None:
            # No peer topology; degenerate to the per-holon path.
            rnd.restore(answers, senders)
            await role._admm.run_supply_priority()
            return

        round_id = f"r{self.round_counter}"
        self.round_counter += 1
        now = float(role.context.current_timestamp)
        leader_aid = role.context.aid

        # Echo the last applied version as an implicit ACK, but only if it was
        # published by the CURRENT coordinator; else a re-elected coordinator
        # (counter restarts low) is wedged by the old coordinator's high version.
        echo_version = (
            self.last_applied_version
            if self.last_applied_publisher == coord_aid
            else -1
        )
        report = ComponentAdmmReport(
            publisher=leader_aid,
            version=role._version.next(),
            timestamp_s=now,
            round_id=round_id,
            sector=role.sector,
            leader_aid=leader_aid,
            supply_by_sector=supply,
            demand_by_sector_priority=demand,
            served_by_sector_priority=served,
            last_applied_allocation_version=echo_version,
        )

        if coord_aid == leader_aid:
            # I'm the coordinator — buffer my report, trigger the solve.
            self.report_buffer[leader_aid] = (round_id, report)
            await self.maybe_run(reason="self_report")
            return

        # Push to the coordinator. No reply timeout: a drop is retried by the
        # next trigger (idempotent — buffer keyed by leader_aid).
        peers = role._resolve_component_peer_addrs()
        coord_addr = peers.get(coord_aid)
        if coord_addr is None:
            # Aid known but address unresolved — fall back to the per-holon path.
            rnd.restore(answers, senders)
            await role._admm.run_supply_priority()
            return
        await role.context.send_message(report, receiver_addr=coord_addr)
        role._record_event(
            "component_report_sent",
            f"coord={coord_aid} leader={leader_aid} round={round_id} "
            f"supply={sum(supply.values()):.4f} demand_tiers={len(demand)}",
        )

    async def handle_report(self, message: ComponentAdmmReport, meta: dict) -> None:
        """Coordinator-side: buffer a peer's report (keyed by ``leader_aid``,
        freshest wins) and schedule the debounced solve. Non-coordinators drop;
        reports from peers no longer in our component are dropped too.
        """
        role = self._role
        if role.admm_scope != "component":
            return
        if role._component_coordinator_aid() != role.context.aid:
            return
        # Only buffer reports from leaders still in our active component.
        component_peers = role._resolve_component_peer_addrs()
        if message.leader_aid not in component_peers:
            return
        # Per-publisher staleness guard (mirrors ``_on_summary``): under
        # latency/loss a reordered older report must not overwrite a fresher
        # one (the buffer is last-arrival-wins otherwise).
        prior = self.report_buffer.get(message.leader_aid)
        if prior is not None and int(getattr(message, "version", 0)) <= int(
            getattr(prior[1], "version", -1)
        ):
            return
        self.report_buffer[message.leader_aid] = (message.round_id, message)
        # Packet-loss recovery: if the sender's echoed version trails our latest
        # dispatch, re-send to just this peer. getattr-guarded for legacy reports.
        try:
            await self.resend_if_stale(
                message.leader_aid,
                component_peers.get(message.leader_aid),
                int(getattr(message, "last_applied_allocation_version", -1)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[%s] resend-if-stale failed for leader=%s: %s",
                role.context.aid,
                message.leader_aid,
                exc,
            )
        await self.maybe_run(reason="peer_report")

    async def resend_if_stale(
        self,
        leader_aid: str,
        leader_addr: Any | None,
        applied_version: int,
    ) -> None:
        """Re-send the latest ``ComponentAllocation`` to ``leader_addr`` when
        its echoed version trails ``allocation_version_counter`` (dispatch
        lost). Idempotent (leaf ignores version ≤ applied); self-skipped.
        """
        role = self._role
        if leader_addr is None:
            return
        if leader_aid == role.context.aid:
            return
        if self.last_dispatched_allocation is None:
            return
        if applied_version >= self.allocation_version_counter:
            return
        await role.context.send_message(
            self.last_dispatched_allocation,
            receiver_addr=leader_addr,
        )
        role._record_event(
            "component_alloc_resent",
            f"target={leader_aid} applied_version={applied_version} "
            f"latest_version={self.allocation_version_counter}",
        )

    async def maybe_run(self, *, reason: str) -> None:
        """Debounce: collapse a report burst into one solve. The first arrival
        owns the solve (``dispatch_pending``), one per burst.
        """
        if self.dispatch_pending:
            return
        self.dispatch_pending = True
        self._role.context.schedule_instant_task(
            self.run_now(reason=reason),
        )

    async def run_now(self, *, reason: str) -> None:
        """Run the per-component ADMM over the buffer and dispatch fractions to
        every leader (incl. self). Clears ``dispatch_pending``; the buffer is
        not drained so late reports overwrite for the next solve.
        """
        try:
            await self._run_now_inner(reason=reason)
        finally:
            self.dispatch_pending = False

    async def _run_now_inner(self, *, reason: str) -> None:
        role = self._role
        if not self.report_buffer:
            return
        # Only leaders still in this coordinator's component view; a
        # disconnected leader's report stays buffered but is skipped.
        component_peers = role._resolve_component_peer_addrs()
        leader_aids = sorted(
            aid for aid in self.report_buffer if aid in component_peers
        )
        if not leader_aids:
            return
        reports = [self.report_buffer[a][1] for a in leader_aids]

        actor_supplies = [r.supply_by_sector for r in reports]
        actor_demands = [r.demand_by_sector_priority for r in reports]

        sectors, tiers, total_demand = extract_demand_sectors_tiers(actor_demands)
        if not sectors or not tiers or total_demand < 1e-6:
            return

        try:
            with optimization(
                "admm_supply_priority",
                logger=logger,
                scope="component",
                aid=role.context.aid,
                n_actors=len(actor_supplies),
            ):
                service_fraction, _per_actor_x, meta = await allocate_supply_priority(
                    sectors=sectors,
                    tiers=tiers,
                    actor_supplies=actor_supplies,
                    actor_demands=actor_demands,
                    actor_ub_overrides=None,
                    priority_tiers=role.priority_tiers,
                    max_iters=int(role.admm_max_iters),
                    abs_tol=float(role.admm_abs_tol),
                    enable_priority_weighting=role.enable_priority_allocation,
                )
        except Exception as exc:
            logger.error(
                "[%s] component-scope ADMM failed: %s",
                role.context.aid,
                exc,
            )
            role._record_event("holon_admm_failed", f"component_scope: {exc}")
            return

        # No sub-tolerance noise scrub: clamping near-zero to exact 0 would lock
        # it in via the cooldown gate. The PI claim's 1e-3 tolerance absorbs it.

        # Complete + monotone per-tier vector. A round may solve a tier subset,
        # so fold in the previous dispatch's tiers (fresh wins) then clamp
        # non-increasing in tier number, avoiding a stale-higher inversion.
        # Bounded to this coordinator's solve+history (not all P tiers): forcing
        # unknown tiers down over-sheds; handoff inversion needs L2.5 instead.
        sec_val = role.sector.value
        prev_fraction = self.last_fraction
        prev_own = (prev_fraction or {}).get(sec_val, {})
        merged_own = dict(prev_own)
        merged_own.update(service_fraction.get(sec_val, {}))
        clamp_tier_monotonic(merged_own)
        service_fraction = {**service_fraction, sec_val: merged_own}

        # Allocation-unchanged gate: when the merged solve equals the last
        # dispatched fractions (within the actuator dedup tolerance), skip the
        # re-dispatch entirely. This (with the leaf-side "anything actually
        # changed" nudge gate) bounds the dispatch→rebalance→report→solve loop
        # under enable_change_only_dispatch, where no time throttle applies.
        # Leaders that missed the previous dispatch still recover via
        # ``resend_if_stale`` (version echo), so skipping is safe.
        if (
            self.allocation_version_counter > 0
            and prev_fraction is not None
            and _fraction_maps_equal(prev_fraction, service_fraction)
        ):
            logger.debug(
                "[%s] component-scope ADMM: allocation unchanged — dispatch skipped",
                role.context.aid,
            )
            return
        logger.info(
            "[%s] component-scope ADMM result (reason=%s): sectors=%s "
            "tiers=%s n_communities=%d fractions=%s",
            role.context.aid,
            reason,
            sectors,
            tiers,
            len(reports),
            service_fraction,
        )
        role._record_event(
            "holon_admm_result",
            f"component_scope reason={reason} sectors={sectors} tiers={tiers} "
            f"n_communities={len(reports)} fractions={service_fraction}",
        )

        # Dispatch to every leader incl. self; don't skip this authoritative
        # dispatch, or a stale per-load L2 priority floor (set in apply_regulate)
        # lets a fresh L1 gossip invert priority. The upward change-detection
        # bounds the cascade instead.
        round_id = max(
            (self.report_buffer[a][0] for a in leader_aids),
            default="",
        )
        now = float(role.context.current_timestamp)
        # Bump version before building the message so leaf ACKs line up. Stash
        # for re-sends to stale leaders (``resend_if_stale``).
        self.allocation_version_counter += 1
        allocation = ComponentAllocation(
            publisher=role.context.aid,
            version=self.allocation_version_counter,
            timestamp_s=now,
            round_id=round_id,
            sector=role.sector,
            service_fraction_by_tier=service_fraction.get(role.sector.value, {}),
        )
        # Rebase only on actual dispatch: both the skip gate's prev_fraction
        # and the monotonic-merge anchor must reference the last DISPATCHED
        # map, or skipped rounds accumulate unbounded drift.
        self.last_fraction = service_fraction
        self.last_dispatched_allocation = allocation
        for addr in component_peers.values():
            await role.context.send_message(allocation, receiver_addr=addr)

    async def handle_allocation(self, message: ComponentAllocation, meta: dict) -> None:
        """Leaf-side: apply the coordinator's per-tier fraction to this leader's
        own community members (no holon hop) via the L1 honour path. Every
        sector leader handles it, covering communities outside any holon.
        Coalition merge: an active store fraction wins per-tier.
        """
        role = self._role
        if role.admm_scope != "component":
            return
        # Cheap drift guard: act only as a current group leader.
        if topology_characteristic(role, tid="groups") != "leader":
            return
        # Version gate BEFORE applying: a delayed/duplicated older allocation
        # from the same coordinator must not overwrite a fresher one. Versions
        # are per-publisher; a different publisher (coordinator re-election)
        # always passes and resets the counter below.
        try:
            msg_version = int(message.version)
        except (TypeError, ValueError):
            msg_version = None  # legacy allocation: apply, ack stays -1
        msg_publisher = str(getattr(message, "publisher", "") or "")
        if (
            msg_version is not None
            and msg_publisher == self.last_applied_publisher
            and msg_version <= self.last_applied_version
        ):
            return
        # Rebuild a {sector: {tier: frac}} envelope for the L1 honour path.
        service_fraction: dict[str, dict[int, float]] = {
            role.sector.value: dict(message.service_fraction_by_tier),
        }
        if role._coalition_constraint_store is not None:
            now = float(role.context.current_timestamp)
            service_fraction = role._coalition_constraint_store.merge_into(
                service_fraction,
                role.sector,
                now,
            )
            # Re-clamp after the coalition merge: merge_into can let a store
            # fraction lift a low tier above a high one, reintroducing the
            # observed gas tier-1-victim inversions.
            merged = service_fraction.get(role.sector.value)
            if merged:
                clamp_tier_monotonic(merged)
        # Send to SELF: the negotiator applies the fraction to every community
        # member, covering loads outside any holon. This authoritative dispatch
        # also refreshes the per-load L2 priority floor (in ``apply_regulate``);
        # the cascade is bounded upstream by the L1→L2 change-detection, not by
        # skipping it here.
        await role.context.send_message(
            StartBalanceNegotiation(
                service_fraction_by_sector_priority=service_fraction,
            ),
            receiver_addr=role.context.addr,
        )
        # ACK: record (publisher, version) so the next report echoes it (the
        # coordinator detects drops via the echo). A new publisher resets the
        # counter — versions are not comparable across coordinators.
        if msg_version is not None:
            self.last_applied_publisher = msg_publisher
            self.last_applied_version = msg_version
        role._record_event(
            "holon_priority_allocation",
            f"component_scope round={message.round_id} "
            f"version={getattr(message, 'version', 0)} "
            f"fractions={service_fraction}",
        )
