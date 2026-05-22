"""Per-leader store for active L2.5 coalition constraints.

Bridges :class:`HolonSummaryRole` (writer) and
:class:`HolonicCommunityRole` (reader) on the same leader agent:

- The coalition initiator allocates per-(sector, tier) service
  fractions and broadcasts them as :class:`CoalitionConstraint` to
  every accepting member.
- Each member's :class:`HolonSummaryRole` writes the received
  constraint into its local store.
- When :class:`HolonicCommunityRole` is about to send a
  :class:`StartBalanceNegotiation` with its own ADMM result, it
  calls :meth:`merge_into` to override per-tier fractions with any
  active coalition fractions for the same sector.

Constraints expire on ``issued_at + ttl_s`` (natural TTL) or on a
``BranchFailureEvent`` reaching the leader (early invalidation),
implementing the design directive that failures invalidate
coalition constraints immediately so the post-failure L2 ADMM
round is free to redecide allocations.

The store is local to each leader; coalitions broadcast their
constraints over the sector-wide ``holon_summary_<sector>`` mesh,
so every coalition member's store ends up holding the same
``coalition_id``-keyed record (last-version-wins).  No cross-agent
shared state — the store lives inside one Python process per
leader and is shared between the leader's L2.5 and L2 roles via
construction-time injection.
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

    ``target_flows_mw[sector_value]`` is signed in load convention —
    positive means the CP commits to *consuming* from that sector,
    negative means *producing* into it.  The CP role narrows its own
    ADMM's per-sector lb/ub to ``[committed - tol, committed + tol]``
    inside the TTL window, then releases on expiry.
    """

    coalition_id: str
    cp_id: str
    target_flows_mw: dict[str, float]
    issued_at: float
    ttl_s: float


class CoalitionConstraintStore:
    """Active coalition constraints for one leader.

    ``set`` adds (or replaces by ``coalition_id``).  ``merge_into``
    overlays the active per-(sector, tier) fractions onto a base
    ``service_fraction`` map produced by L2's ADMM, with coalition
    winning on overlap.  ``prune`` drops entries past TTL; ``clear``
    invalidates everything for a sector (or all sectors) on a
    failure.

    Cross-sector extension: ``set_cp_envelope`` / ``cp_envelope_for``
    /``prune`` /``clear`` also handle directional CP setpoint
    commitments issued by cross-sector coalitions.  L3
    (:class:`EnergyConverterRole`) reads its own envelope on every
    ADMM round and clamps the per-sector ``ub``/``lb`` accordingly.
    """

    def __init__(self) -> None:
        self._records: dict[str, _CoalitionRecord] = {}
        # cp_id -> envelope record.  Keyed by CP aid (not coalition_id)
        # because at most one cross-sector coalition can hold a given
        # CP at a time — second commit overwrites the first, which is
        # the desired latest-wins behaviour for re-asserted commits.
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
        """Drop expired records (both sector fractions and CP
        envelopes).  Returns the total number dropped.
        """
        expired = [
            cid for cid, rec in self._records.items()
            if now > rec.issued_at + rec.ttl_s
        ]
        for cid in expired:
            self._records.pop(cid, None)
        expired_cp = [
            cp_id for cp_id, env in self._cp_envelopes.items()
            if now > env.issued_at + env.ttl_s
        ]
        for cp_id in expired_cp:
            self._cp_envelopes.pop(cp_id, None)
        return len(expired) + len(expired_cp)

    def clear(self, sector: Sector | None = None) -> int:
        """Drop all records (when ``sector`` is None) or just those
        matching ``sector``.  Returns the number dropped.

        Called on a ``BranchFailureEvent`` — the post-failure topology
        invalidates the recently-computed fractions, and the next L2
        rebalance is free to redecide.

        ``sector=None`` clears CP envelopes too; sector-scoped clear
        only drops CP envelopes whose ``target_flows_mw`` includes the
        affected sector (so a heat-only coalition is not invalidated
        by an electricity branch failure).
        """
        if sector is None:
            n = len(self._records) + len(self._cp_envelopes)
            self._records.clear()
            self._cp_envelopes.clear()
            return n
        expired = [
            cid for cid, rec in self._records.items()
            if rec.sector == sector
        ]
        for cid in expired:
            self._records.pop(cid, None)
        sec_v = sector.value
        expired_cp = [
            cp_id for cp_id, env in self._cp_envelopes.items()
            if sec_v in env.target_flows_mw
        ]
        for cp_id in expired_cp:
            self._cp_envelopes.pop(cp_id, None)
        return len(expired) + len(expired_cp)

    # ------------------------------------------------------------------
    # CP envelope API (cross-sector coalition)
    # ------------------------------------------------------------------

    def set_cp_envelope(
        self,
        coalition_id: str,
        cp_id: str,
        target_flows_mw: dict[str, float],
        issued_at: float,
        ttl_s: float,
    ) -> None:
        """Record (or overwrite) the directional sector flows a
        cross-sector coalition has committed for ``cp_id``.

        Keyed by ``cp_id`` so a re-asserted commit replaces the
        previous record cleanly (latest-wins).  The CP role reads via
        :meth:`cp_envelope_for` on every ADMM round.
        """
        self._cp_envelopes[cp_id] = _CPEnvelopeRecord(
            coalition_id=coalition_id,
            cp_id=cp_id,
            target_flows_mw=dict(target_flows_mw),
            issued_at=float(issued_at),
            ttl_s=float(ttl_s),
        )

    def cp_envelope_for(
        self, cp_id: str, now: float
    ) -> dict[str, float] | None:
        """Return active ``target_flows_mw`` for ``cp_id`` or None.

        Returns None if no record exists, or the record has expired
        (caller should typically have called :meth:`prune` already).
        """
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
        """Return a new service-fraction map with active coalition
        fractions overlaid for ``sector``.  ``service_fraction`` is
        not mutated.

        For every active (non-expired) record matching ``sector``,
        the record's ``service_fraction_by_tier`` entries override
        the corresponding tier in ``service_fraction[sector.value]``.
        Tiers the coalition didn't touch keep the base value.
        Sectors other than ``sector`` are returned unchanged.
        """
        merged = {
            sec: dict(tier_map) for sec, tier_map in service_fraction.items()
        }
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
        """Return the union of tiers covered by non-expired records for
        ``sector``.

        Used by :class:`HolonicCommunityRole._run_tier_stratified_admm`
        to suppress L2 per-tier dispatch for cells the coalition has
        already claimed — the coalition re-asserts its absolute service
        fractions every tick, so an additional L2 incremental delta on
        the same regulation knob would cause oscillation.
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
