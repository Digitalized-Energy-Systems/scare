"""Pure flex-aggregation algebra for the L2 holon ADMM.

Extracted from :class:`scare.community.holonic.HolonicCommunityRole`:

* :func:`aggregate_holon_flex` rolls a batch of
  :class:`~scare.base.model.AvailableFlexAnswer` into one
  ``(supply_by_sector, demand_by_sector_priority, served_by_sector_priority)``
  triple — the per-actor inputs for the supply-priority ADMM.
* :func:`extract_demand_sectors_tiers` derives the active
  ``(sectors, tiers, total_demand)`` from a batch of per-actor demand maps.

Both the per-holon path (over flex answers) and the component-coordinator path
(over ``ComponentAdmmReport`` demand maps) share this algebra. Side-effect-free,
so unit-testable without a mango role/context.
"""

from __future__ import annotations

from scare.base.model import AvailableFlexAnswer


def aggregate_holon_flex(
    answers: list[AvailableFlexAnswer],
) -> tuple[
    dict[str, float],
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
]:
    """Sum ``answers`` into ``(supply_by_sector, demand_by_sector_priority,
    served_by_sector_priority)``."""
    supply: dict[str, float] = {}
    demand: dict[str, dict[int, float]] = {}
    served: dict[str, dict[int, float]] = {}
    for a in answers:
        for sec, val in (a.supply_by_sector or {}).items():
            supply[sec] = supply.get(sec, 0.0) + float(val)
        for sec, tmap in (a.demand_by_sector_priority or {}).items():
            bucket = demand.setdefault(sec, {})
            for tier, mw in tmap.items():
                bucket[int(tier)] = bucket.get(int(tier), 0.0) + float(mw)
        for sec, tmap in (a.served_by_sector_priority or {}).items():
            bucket = served.setdefault(sec, {})
            for tier, mw in tmap.items():
                bucket[int(tier)] = bucket.get(int(tier), 0.0) + float(mw)
    return supply, demand, served


def extract_demand_sectors_tiers(
    actor_demands: list[dict[str, dict[int, float]]],
) -> tuple[list[str], list[int], float]:
    """Active ``(sectors, tiers, total_demand)`` across a batch of per-actor
    ``demand_by_sector_priority`` maps.

    ``sectors`` are those with any demand cell (sorted); ``tiers`` are the
    present tiers ``>= 1`` (sorted); ``total_demand`` is the sum over all cells.
    An empty ``sectors``/``tiers`` or a near-zero ``total_demand`` signals the
    caller to take its no-demand fallback.
    """
    sectors = sorted({s for d in actor_demands for s in (d or {})})
    tiers_present: set[int] = set()
    for d in actor_demands:
        for tmap in (d or {}).values():
            tiers_present.update(tmap.keys())
    tiers = sorted(t for t in tiers_present if t >= 1)
    total_demand = sum(
        float(v)
        for d in actor_demands
        for tmap in (d or {}).values()
        for v in tmap.values()
    )
    return sectors, tiers, total_demand
