"""Per-leader store for active L2.5 coalition constraints.

Bridges the coalition writer (HolonSummaryRole) and L2 reader
(HolonicCommunityRole) on the same leader. Constraints expire on
``issued_at + ttl_s`` or on a ``BranchFailureEvent`` (early
invalidation) so the post-failure L2 ADMM round can redecide.
Local to each leader; records are ``coalition_id``-keyed (last-wins).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from scare.base.model import Sector

logger = logging.getLogger(__name__)


@dataclass
class _CoalitionRecord:
    sector: Sector
    service_fraction_by_tier: dict[int, float]
    issued_at: float
    ttl_s: float


@dataclass
class _CPEnvelopeRecord:
    """Cross-sector coalition commitment for a single CP.

    ``target_flows_mw[sector_value]`` is signed in load convention
    (positive = consume, negative = produce). The CP role clamps its
    ADMM per-sector lb/ub to ``[committed +/- tol]`` within the TTL.
    """

    coalition_id: str
    cp_id: str
    target_flows_mw: dict[str, float]
    issued_at: float
    ttl_s: float


class CoalitionConstraintStore:
    """Active coalition constraints for one leader.

    Per-(sector, tier) service fractions (``set`` / ``merge_into``,
    coalition wins on overlap) plus directional cross-sector CP setpoint
    envelopes (``set_cp_envelope`` / ``cp_envelope_for``, read by L3's
    EnergyConverterRole). ``prune`` drops past-TTL; ``clear`` invalidates
    on failure.
    """

    def __init__(self) -> None:
        self._records: dict[str, _CoalitionRecord] = {}
        # Keyed by cp_id, not coalition_id: one coalition holds a CP at a
        # time, so a re-asserted commit overwrites (latest-wins).
        self._cp_envelopes: dict[str, _CPEnvelopeRecord] = {}

    def set(
        self,
        coalition_id: str,
        sector: Sector,
        service_fraction_by_tier: dict[int, float],
        issued_at: float,
        ttl_s: float,
    ) -> None:
        self._records[coalition_id] = _CoalitionRecord(
            sector=sector,
            service_fraction_by_tier=dict(service_fraction_by_tier),
            issued_at=float(issued_at),
            ttl_s=float(ttl_s),
        )

    def prune(self, now: float) -> int:
        """Drop expired records (fractions and CP envelopes); return count."""
        expired = [
            cid for cid, rec in self._records.items() if now > rec.issued_at + rec.ttl_s
        ]
        for cid in expired:
            self._records.pop(cid, None)
        expired_cp = [
            cp_id
            for cp_id, env in self._cp_envelopes.items()
            if now > env.issued_at + env.ttl_s
        ]
        for cp_id in expired_cp:
            self._cp_envelopes.pop(cp_id, None)
        return len(expired) + len(expired_cp)

    def clear(self, sector: Sector | None = None) -> int:
        """Drop all records (``sector`` None) or those matching ``sector``.

        Called on a ``BranchFailureEvent`` so the next L2 rebalance can
        redecide. Sector-scoped clear only drops CP envelopes whose
        ``target_flows_mw`` touches that sector.
        """
        if sector is None:
            n = len(self._records) + len(self._cp_envelopes)
            self._records.clear()
            self._cp_envelopes.clear()
            return n
        expired = [cid for cid, rec in self._records.items() if rec.sector == sector]
        for cid in expired:
            self._records.pop(cid, None)
        sec_v = sector.value
        expired_cp = [
            cp_id
            for cp_id, env in self._cp_envelopes.items()
            if sec_v in env.target_flows_mw
        ]
        for cp_id in expired_cp:
            self._cp_envelopes.pop(cp_id, None)
        return len(expired) + len(expired_cp)

    # CP envelope API (cross-sector coalition)

    def set_cp_envelope(
        self,
        coalition_id: str,
        cp_id: str,
        target_flows_mw: dict[str, float],
        issued_at: float,
        ttl_s: float,
    ) -> None:
        """Record (overwrite) the committed directional sector flows for ``cp_id``."""
        self._cp_envelopes[cp_id] = _CPEnvelopeRecord(
            coalition_id=coalition_id,
            cp_id=cp_id,
            target_flows_mw=dict(target_flows_mw),
            issued_at=float(issued_at),
            ttl_s=float(ttl_s),
        )

    def cp_envelope_for(self, cp_id: str, now: float) -> dict[str, float] | None:
        """Return active ``target_flows_mw`` for ``cp_id``, or None if missing/expired."""
        env = self._cp_envelopes.get(cp_id)
        if env is None:
            return None
        if now > env.issued_at + env.ttl_s:
            return None
        return dict(env.target_flows_mw)

    def has_active_cp_envelope(self, cp_id: str, now: float) -> bool:
        return self.cp_envelope_for(cp_id, now) is not None

    def merge_into(
        self,
        service_fraction: dict[str, dict[int, float]],
        sector: Sector,
        now: float,
    ) -> dict[str, dict[int, float]]:
        """Return a copy of ``service_fraction`` with active coalition
        per-tier fractions overlaid for ``sector`` (input not mutated).
        Untouched tiers and other sectors are left unchanged.
        """
        merged = {sec: dict(tier_map) for sec, tier_map in service_fraction.items()}
        sec_v = sector.value
        for rec in self._records.values():
            if rec.sector != sector:
                continue
            if now > rec.issued_at + rec.ttl_s:
                continue
            merged.setdefault(sec_v, {})
            for tier, frac in rec.service_fraction_by_tier.items():
                merged[sec_v][tier] = float(frac)
        return merged

    def has_active(self, sector: Sector, now: float) -> bool:
        """True iff at least one non-expired record exists for ``sector``."""
        for rec in self._records.values():
            if rec.sector != sector:
                continue
            if now <= rec.issued_at + rec.ttl_s:
                return True
        return False

    def active_tiers(self, sector: Sector, now: float) -> set[int]:
        """Tiers covered by non-expired records for ``sector``.

        Lets L2 skip per-tier dispatch for cells the coalition claims:
        it re-asserts absolute fractions each tick, so an extra L2 delta
        on the same knob would oscillate.
        """
        tiers: set[int] = set()
        for rec in self._records.values():
            if rec.sector != sector:
                continue
            if now > rec.issued_at + rec.ttl_s:
                continue
            tiers.update(rec.service_fraction_by_tier.keys())
        return tiers

    def __len__(self) -> int:
        return len(self._records)
