"""Publication of this leader's :class:`HolonSummary` onto the sector mesh,
and the peer-summary cache built from what the mesh reports back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import sender_addr as mango_sender_addr
from mango.express.topology import topology_characteristic, topology_neighbors

from scare.base.channel import (
    HolonSummary,
)
from scare.base.model import Sector
from scare.base.util import (
    kgps_to_mw,
    lookup_slack,
    lookup_slack_eff_budget,
    obs_capacity,
    obs_priority,
    obs_sector,
    obs_setpoint,
)
from scare.community.summary_state import (
    CrossSectorChannel,
)

if TYPE_CHECKING:
    from scare.community.summary import HolonSummaryRole

logger = logging.getLogger(__name__)


class SummaryPublisher:
    """Decides when the local summary changed enough to republish, publishes
    it, and keeps the freshest per-publisher peer summary. Reads behavior,
    sector and config through its owning role.
    """

    def __init__(self, role: HolonSummaryRole) -> None:
        self._role = role
        # publisher_aid -> latest HolonSummary seen on the mesh.
        self._peer_summaries: dict[str, Any] = {}
        # Last published per-tier maps; a republish needs a real change.
        self._last_published_demand: dict[int, float] = {}
        self._last_published_served: dict[int, float] = {}

    async def _publish(self, *, force: bool = False) -> None:
        """Aggregate this leader's community state per tier and broadcast
        to all same-sector peers via ``holon_summary_<sector>``.
        """
        try:
            peers = list(topology_neighbors(self._role, tid=self._role._topology_tid))
        except Exception:
            return
        if not peers:
            return

        per_tier_served: dict[int, float] = {}
        per_tier_demand: dict[int, float] = {}
        supply_total: float = 0.0
        slack_budget_total: float = 0.0  # slack-only; caps CP input draw
        slack_headroom_total: float = 0.0  # budget still unused right now
        try:
            member_aids = [self._role.context.aid] + [
                addr.aid for addr in topology_neighbors(self._role, tid="groups")
            ]
        except Exception:
            member_aids = [self._role.context.aid]

        for aid in member_aids:
            try:
                obs = self._role.behavior.observe(aid) or {}
            except (AttributeError, KeyError):
                return
            sector = obs_sector(obs, behavior=self._role.behavior, aid=aid)
            if sector != self._role.sector:
                continue
            # A promoted island reference is a generator, never a load: its
            # free Var reads 0 at init and flips positive when it absorbs, so
            # the plain cap-sign test below would drop it from supply and then
            # bill it as tiered demand (project_islanding_former_guard_off).
            if self._role._grid_former_policy.is_former(aid):
                supply_total += self._role._grid_former_policy.supply_credit(
                    aid, obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
                )
                continue
            cap = obs_capacity(obs, behavior=self._role.behavior, aid=aid)
            if cap < 0:
                # Generator / slack injector feeds L3's supply pool. A
                # slack advertises its (effective) budget, not raw |cap|.
                slack_meta = lookup_slack(self._role.behavior, aid)
                if slack_meta is not None:
                    eff = lookup_slack_eff_budget(self._role.behavior, aid)
                    v = float(eff) if eff is not None else abs(cap)
                    supply_total += v
                    # CP input cap uses nominal, not wound-down eff budget: a
                    # floor gives a converter (Ση<1) zero headroom → r=0, starving
                    # the over-draw actuator (CP kernel splits native-vs-CP, one B).
                    if self._role.cp_budget_nominal:
                        budget = abs(slack_meta.cap)
                    else:
                        budget = v
                    slack_budget_total += budget
                    # obs_setpoint on a slack is its LP-chosen operating point,
                    # i.e. how much of ``budget`` is already flowing and hence
                    # already counted in some community's ``per_tier_served``.
                    used = abs(
                        obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
                    )
                    slack_headroom_total += max(0.0, budget - used)
                else:
                    supply_total += abs(cap)
                continue
            if cap == 0:
                continue
            sp = obs_setpoint(obs, behavior=self._role.behavior, aid=aid)
            tier = obs_priority(obs, behavior=self._role.behavior, aid=aid)
            per_tier_demand[tier] = per_tier_demand.get(tier, 0.0) + abs(cap)
            per_tier_served[tier] = per_tier_served.get(tier, 0.0) + abs(sp)

        # Delta gate: skip when no tier moved by more than ``inversion_tol``.
        # The watchdog forces through to advance the version frontier.
        gated = not force and not self._role._summary_changed(
            per_tier_served, per_tier_demand
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[%s] L2 summary publish sector=%s force=%s gated=%s "
                "demand=%.6f served=%.6f supply=%.6f slack=%.6f headroom=%.6f "
                "d_by_tier=%s s_by_tier=%s prev_s_by_tier=%s tol=%g",
                self._role.context.aid,
                self._role.sector.value,
                force,
                gated,
                float(sum(per_tier_demand.values())),
                float(sum(per_tier_served.values())),
                supply_total,
                slack_budget_total,
                slack_headroom_total,
                {t: round(v, 8) for t, v in sorted(per_tier_demand.items())},
                {t: round(v, 8) for t, v in sorted(per_tier_served.items())},
                {t: round(v, 8) for t, v in sorted(self._last_published_served.items())},
                self._role.inversion_tol,
            )
        if gated:
            return

        # Cache before ``send_message`` so a re-entrant publish sees the
        # latest baseline.
        self._last_published_served = dict(per_tier_served)
        self._last_published_demand = dict(per_tier_demand)

        sec_key = self._role.sector.value
        supply_by_sector = {sec_key: supply_total} if supply_total > 0.0 else {}
        slack_budget_by_sector = (
            {sec_key: slack_budget_total} if slack_budget_total > 0.0 else {}
        )
        slack_headroom_by_sector = (
            {sec_key: slack_headroom_total} if slack_headroom_total > 0.0 else {}
        )
        demand_by_sector_priority = (
            {sec_key: dict(per_tier_demand)} if per_tier_demand else {}
        )
        served_by_sector_priority = (
            {sec_key: dict(per_tier_served)} if per_tier_served else {}
        )

        summary = HolonSummary(
            publisher=str(self._role.context.aid),
            version=self._role._version.next(),
            caused_by={},
            timestamp_s=float(self._role.context.current_timestamp),
            sector=self._role.sector,
            per_tier_served_mw=per_tier_served,
            per_tier_demand_mw=per_tier_demand,
            supply_by_sector=supply_by_sector,
            demand_by_sector_priority=demand_by_sector_priority,
            served_by_sector_priority=served_by_sector_priority,
            slack_budget_by_sector=slack_budget_by_sector,
            slack_headroom_by_sector=slack_headroom_by_sector,
            home_node_id=self._role._my_node_id,
        )
        # Record our own summary too — the invariant check treats self
        # as just another publisher.
        self._peer_summaries[str(self._role.context.aid)] = summary
        # Cross-sector visibility: mirror into the shared per-sector channel
        # other-sector roles read.
        CrossSectorChannel.for_behavior(self._role.behavior).publish(
            self._role.sector, str(self._role.context.aid), summary
        )
        for addr in peers:
            await self._role.context.send_message(summary, receiver_addr=addr)

    def _summary_changed(
        self,
        served: dict[int, float],
        demand: dict[int, float],
    ) -> bool:
        """True iff any tier moved by more than ``inversion_tol`` vs the
        last published vectors (union of tiers, so a drop to 0 counts).

        Deltas are compared in MW. ``inversion_tol`` is an ABSOLUTE threshold,
        but gas is carried in native kg/s, so comparing it raw made the gate
        ~42x looser for gas than for electricity/heat: on
        ``simbench_lv_gas_dependent`` the whole grid's gas demand is 0.0036
        kg/s against a 1e-3 tolerance, so every one of the 17 load-carrying
        leaders could shed its ENTIRE load without the gate opening. Gas
        ``served`` then stayed frozen at the pre-dispatch t~0.08 snapshot
        (where ``served == demand`` because nothing is regulated yet) for the
        whole run, and the L3 CP kernel — which reads exactly this field —
        was told gas was 100% served while 60% of it was shed.
        """
        if not self._last_published_served and not self._last_published_demand:
            return True  # first publish
        tiers = (
            set(served)
            | set(demand)
            | set(self._last_published_served)
            | set(self._last_published_demand)
        )
        tol = self._role.inversion_tol
        scale = self._tol_scale()
        for t in tiers:
            d_served = served.get(t, 0.0) - self._last_published_served.get(t, 0.0)
            if abs(d_served) * scale > tol:
                return True
            d_demand = demand.get(t, 0.0) - self._last_published_demand.get(t, 0.0)
            if abs(d_demand) * scale > tol:
                return True
        return False

    def _tol_scale(self) -> float:
        """Native-units-to-MW multiplier for this sector's delta comparison.

        Electricity/heat per-tier values are already MW (scale 1.0); gas is
        kg/s. Only the COMPARISON is rescaled — the published values stay in
        native units, since the L3 consumer does its own ``kgps_to_mw``.
        """
        if self._role.sector == Sector.GAS:
            return kgps_to_mw(1.0)
        return 1.0

    async def _on_summary(self, message: HolonSummary, meta: dict) -> None:
        sender = mango_sender_addr(meta)
        if sender is None:
            return
        # Normalise to the bare aid string to match the self-key from
        # ``_publish``; mixed keys would break the lex-smallest election.
        key = getattr(sender, "aid", None) or str(sender)
        prior = self._peer_summaries.get(key)
        if prior is not None and message.version <= prior.version:
            return  # stale
        self._peer_summaries[key] = message
        # Full address for sending coalition messages back to this peer.
        self._role._peer_addrs[key] = sender
        CrossSectorChannel.for_behavior(self._role.behavior).publish(
            message.sector, key, message
        )
        # Peer view shifted — re-run detection now, not next watchdog.
        if topology_characteristic(self._role, tid="groups") == "leader":
            self._role._check_invariants()
            if self._role.enable_cross_sector_coalitions:
                self._role._check_cross_sector_invariants()

    def _is_elected_initiator(self) -> bool:
        """True when this leader is the lex-smallest publisher with
        non-empty summary state.

        A brief election flip (a silent leader first publishing)
        double-fires but is absorbed by last-write-wins at L1.
        """
        if not self._peer_summaries:
            return False
        publishers = sorted(self._peer_summaries.keys())
        return publishers[0] == str(self._role.context.aid)
