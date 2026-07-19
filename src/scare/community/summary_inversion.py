"""Cross-holon priority-inversion detection.

A higher-priority tier served at a strictly smaller fraction than a lower one
(beyond ``inversion_tol``) is an inversion; mirrors ``experiment/eval/claims.py``.
Cooldown-gated so one persistent inversion emits once, not every tick.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mango.express.topology import topology_characteristic

from scare.base.channel import (
    HolonSummary,
)
from scare.base.runtime.diagnostics import record_event
from scare.community.summary_state import (
    CrossSectorChannel,
)

if TYPE_CHECKING:
    from scare.community.summary import HolonSummaryRole

logger = logging.getLogger(__name__)


class InversionDetector:
    """Scans the peer-summary mesh (and the cross-sector CP view) for tier
    inversions and decides whether this leader should act on one.
    """

    def __init__(self, role: HolonSummaryRole, inversion_cooldown_s: float) -> None:
        self._role = role
        self._inversion_cooldown_s = inversion_cooldown_s
        self._last_inversion_emit_t: float = -1e9
        self._last_xs_inversion_emit_t: float = -1e9

    def _check_invariants(self) -> None:
        """Aggregate peer summaries by tier, detect inversions, and (on the
        elected initiator only) open one coalition per inversion cohort, not N.
        """
        if topology_characteristic(self._role, tid="groups") != "leader":
            return
        # Need a peer summary besides our own to call it "cross-holon";
        # the first tick fires before any peer summary arrives.
        if len(self._role._peer_summaries) < 2:
            return
        if not self._role._is_elected_initiator():
            return

        served_at_tier: dict[int, float] = {}
        demand_at_tier: dict[int, float] = {}
        for s in self._role._peer_summaries.values():
            for tier, served in s.per_tier_served_mw.items():
                served_at_tier[tier] = served_at_tier.get(tier, 0.0) + float(served)
            for tier, demand in s.per_tier_demand_mw.items():
                demand_at_tier[tier] = demand_at_tier.get(tier, 0.0) + float(demand)

        if not demand_at_tier:
            return

        tiers_sorted = sorted(
            t for t, d in demand_at_tier.items() if d > 1e-9 and t >= 1
        )
        if len(tiers_sorted) < 2:
            return
        fracs: dict[int, float] = {
            t: served_at_tier.get(t, 0.0) / demand_at_tier[t] for t in tiers_sorted
        }

        total_served = sum(served_at_tier.get(t, 0.0) for t in tiers_sorted)
        total_demand = sum(demand_at_tier[t] for t in tiers_sorted)
        if total_demand <= 1e-9 or total_served >= total_demand - 1e-6:
            return

        now = float(self._role.context.current_timestamp)
        if now - self._last_inversion_emit_t < self._inversion_cooldown_s:
            return

        emitted = False
        # Worst inversion (largest gap) per tick: bundling every pair
        # would drop mid tiers in lockstep; successive ticks clear them.
        worst_pair: tuple[int, int] | None = None
        worst_gap: float = 0.0
        for i in range(1, len(tiers_sorted)):
            t_prev, t_cur = tiers_sorted[i - 1], tiers_sorted[i]
            f_prev, f_cur = fracs[t_prev], fracs[t_cur]
            gap = f_cur - f_prev
            if gap > self._role.inversion_tol:
                record_event(
                    t=now,
                    kind="priority_inversion_detected",
                    aid=self._role.context.aid,
                    sector=self._role.sector.value,
                    detail=(
                        f"tier_high={t_prev} (frac={f_prev:.3f}) "
                        f"tier_low={t_cur} (frac={f_cur:.3f}) "
                        f"n_publishers={len(self._role._peer_summaries)}"
                    ),
                )
                emitted = True
                if gap > worst_gap:
                    worst_gap = gap
                    worst_pair = (t_prev, t_cur)
        if emitted:
            self._last_inversion_emit_t = now
            logger.info(
                "[%s] cross-holon priority inversion detected (sector=%s, "
                "n_publishers=%d, n_tiers=%d, fracs=%s)",
                self._role.context.aid,
                self._role.sector.value,
                len(self._role._peer_summaries),
                len(tiers_sorted),
                {t: round(f, 3) for t, f in fracs.items()},
            )
            if self._role.enable_coalition and worst_pair is not None:
                # Instant task so the check stays synchronous and the
                # rest of the tick still runs.
                self._role.context.schedule_instant_task(
                    self._role._open_coalition(worst_pair, dict(demand_at_tier))
                )

    def _check_cross_sector_invariants(self) -> None:
        """Detect cross-sector priority inversions across CP-bridged
        sector pairs and, if elected initiator (lex-smallest aid across
        both sides' publishers), open a cross-sector coalition.
        """
        if not self._role._cp_meta:
            return
        channel = CrossSectorChannel.for_behavior(self._role.behavior)
        own_sec = self._role.sector
        own_aid = str(self._role.context.aid)
        own_summaries = channel.read(own_sec)
        if not own_summaries:
            return
        now = float(self._role.context.current_timestamp)
        if now - self._last_xs_inversion_emit_t < self._inversion_cooldown_s:
            return

        for cp_aid, meta in self._role._cp_meta.items():
            sectors_bridged = meta.get("sectors", [])
            if own_sec not in sectors_bridged:
                continue
            for peer_sec in sectors_bridged:
                if peer_sec == own_sec:
                    continue
                peer_summaries = channel.read(peer_sec)
                if not peer_summaries:
                    continue
                pair = self._role._find_inversion_pair(own_summaries, peer_summaries)
                if pair is None:
                    continue
                t_own_high, t_peer_low, frac_own, frac_peer = pair
                # Initiator = lex-smallest aid across both sides; skip
                # otherwise.
                union_aids = sorted(
                    set(own_summaries.keys()) | set(peer_summaries.keys())
                )
                if not union_aids or union_aids[0] != own_aid:
                    continue
                self._last_xs_inversion_emit_t = now
                record_event(
                    t=now,
                    kind="cross_sector_inversion_detected",
                    aid=self._role.context.aid,
                    sector=own_sec.value,
                    detail=(
                        f"cp={cp_aid} own_sec={own_sec.value} "
                        f"tier_high={t_own_high} frac_high={frac_own:.3f} "
                        f"peer_sec={peer_sec.value} tier_low={t_peer_low} "
                        f"frac_low={frac_peer:.3f}"
                    ),
                )
                logger.info(
                    "[%s] cross-sector inversion: cp=%s %s.t%d=%.3f vs "
                    "%s.t%d=%.3f — opening coalition",
                    self._role.context.aid,
                    cp_aid,
                    own_sec.value,
                    t_own_high,
                    frac_own,
                    peer_sec.value,
                    t_peer_low,
                    frac_peer,
                )
                self._role.context.schedule_instant_task(
                    self._role._open_cross_sector_coalition(
                        cp_aid=cp_aid,
                        own_sec=own_sec,
                        peer_sec=peer_sec,
                        t_own_high=t_own_high,
                        t_peer_low=t_peer_low,
                    )
                )
                return  # one per tick

    def _find_inversion_pair(
        self,
        own_summaries: dict[str, HolonSummary],
        peer_summaries: dict[str, HolonSummary],
    ) -> tuple[int, int, float, float] | None:
        """Return ``(t_own_high, t_peer_low, frac_own, frac_peer)`` if an
        inversion exists, else None.

        Inversion: an own-side tier with strict priority over a peer tier
        (lower number = higher priority) served at least ``inversion_tol``
        below the peer's fraction.
        """
        own_dem, own_ser = self._role._aggregate_tier(own_summaries)
        peer_dem, peer_ser = self._role._aggregate_tier(peer_summaries)
        if not own_dem or not peer_dem:
            return None
        for t_own in sorted(own_dem.keys()):
            if t_own < 1 or own_dem[t_own] <= 1e-9:
                continue
            f_own = own_ser.get(t_own, 0.0) / own_dem[t_own]
            for t_peer in sorted(peer_dem.keys()):
                if t_peer <= t_own:
                    continue  # not lower-priority
                if peer_dem[t_peer] <= 1e-9:
                    continue
                f_peer = peer_ser.get(t_peer, 0.0) / peer_dem[t_peer]
                if f_peer > f_own + self._role.inversion_tol:
                    return t_own, t_peer, f_own, f_peer
        return None

    @staticmethod
    def _aggregate_tier(
        summaries: dict[str, HolonSummary],
    ) -> tuple[dict[int, float], dict[int, float]]:
        dem: dict[int, float] = {}
        ser: dict[int, float] = {}
        for s in summaries.values():
            for tier, v in s.per_tier_demand_mw.items():
                dem[tier] = dem.get(tier, 0.0) + float(v)
            for tier, v in s.per_tier_served_mw.items():
                ser[tier] = ser.get(tier, 0.0) + float(v)
        return dem, ser
