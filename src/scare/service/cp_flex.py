"""Per-sector aggregation of CP flex answers and ADMM sector-priority weights.

Extracted verbatim (behaviour-preserving) from
:class:`scare.service.cp.EnergyConverterRole`. These are pure functions over a
batch of :class:`~scare.base.model.AvailableFlexAnswer` plus the resulting
aggregate, so they can be unit-tested without a mango role/context.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from scare.base.model import AvailableFlexAnswer, Sector
from scare.base.util import aggregate_priority_weight, tier_priority_weight_strict


class FlexAggregate(NamedTuple):
    """Per-sector aggregation of a batch of ``AvailableFlexAnswer``."""

    imbalance_by_sector: dict[Sector, float]
    unmet_by_sector_total: dict[Sector, float]
    sector_priority_weight: dict[Sector, float]
    top_unmet_tier_per_sector: dict[Sector, int]
    top_unmet_mag_per_sector: dict[Sector, float]


def aggregate_flex_answers(answers: list[AvailableFlexAnswer]) -> FlexAggregate:
    """Aggregate a batch of flex answers per sector.

    Tracks signed balance, unsigned unmet (LP-undelivered demand — surfaces
    sectors whose disconnected loads would otherwise cancel against generation
    in balance), and the lowest-tier unmet (sector, tier) pair so the ADMM
    weight is top-tier-dominant rather than magnitude-weighted.
    """
    agg = FlexAggregate({}, {}, {}, {}, {})
    for answer in answers:
        agg.imbalance_by_sector[answer.sector] = (
            agg.imbalance_by_sector.get(answer.sector, 0.0) + answer.balance
        )
        for sec_str, val in (getattr(answer, "unmet_by_sector", {}) or {}).items():
            try:
                sec_enum = Sector(sec_str)
            except ValueError:
                continue
            agg.unmet_by_sector_total[sec_enum] = (
                agg.unmet_by_sector_total.get(sec_enum, 0.0) + float(val)
            )
        w = aggregate_priority_weight(
            answer.demand_by_priority, answer.served_by_priority
        )
        agg.sector_priority_weight[answer.sector] = (
            agg.sector_priority_weight.get(answer.sector, 0.0) + w
        )
        dem_map = getattr(answer, "demand_by_sector_priority", {}) or {}
        srv_map = getattr(answer, "served_by_sector_priority", {}) or {}
        for sec_str, tier_to_dem in dem_map.items():
            try:
                sec_enum = Sector(sec_str)
            except ValueError:
                continue
            sec_srv = srv_map.get(sec_str, {})
            for tier, dem in tier_to_dem.items():
                unmet = max(0.0, float(dem) - float(sec_srv.get(tier, 0.0)))
                if unmet < 1e-9:
                    continue
                cur_tier = agg.top_unmet_tier_per_sector.get(sec_enum)
                if cur_tier is None or int(tier) < cur_tier:
                    agg.top_unmet_tier_per_sector[sec_enum] = int(tier)
                    agg.top_unmet_mag_per_sector[sec_enum] = unmet
                elif int(tier) == cur_tier:
                    agg.top_unmet_mag_per_sector[sec_enum] = (
                        agg.top_unmet_mag_per_sector.get(sec_enum, 0.0) + unmet
                    )
    return agg


def compute_sector_priorities(np, agg: FlexAggregate):
    """Top-tier-dominant priority weights for the ADMM sharing problem. A sector
    whose lowest unmet tier is t outranks any sector with top tier t' > t;
    within a tier, magnitude is a bounded log1p tiebreaker. Falls back to the
    aggregated magnitude weight when demand_by_sector_priority is absent.
    Normalised to [0.01, 1].
    """

    def _sector_w(sec: Sector) -> float:
        top_tier = agg.top_unmet_tier_per_sector.get(sec)
        if top_tier is None:
            return agg.sector_priority_weight.get(sec, 1.0) or 1.0
        # Strict-monotone schedule: tier 1 gets the highest weight so the ADMM
        # pulls toward sectors with high-priority unmet demand. Uses the L2/L3
        # helper, not the L1 QP schedule (which returns 0 for tier 1).
        base = tier_priority_weight_strict(top_tier, priority_tiers=4)
        mag = agg.top_unmet_mag_per_sector.get(sec, 0.0)
        return base * (1.0 + 0.5 * math.log1p(mag))

    w_el = _sector_w(Sector.ELECTRICITY)
    w_heat = _sector_w(Sector.HEAT)
    w_gas = _sector_w(Sector.GAS)
    w_max = max(w_el, w_heat, w_gas, 1e-9)
    priorities = np.array([w_el, w_heat, w_gas]) / w_max
    return np.maximum(priorities, 0.01)
