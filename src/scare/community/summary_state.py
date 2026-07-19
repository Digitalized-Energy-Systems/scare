"""Shared state records for the layer-2.5 holon-summary mesh.

Split out so the summary role and its helper controllers (publication,
inversion detection, coalition management) share one definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from scare.base.channel import (
    CoalitionAcceptance,
    HolonSummary,
)
from scare.base.model import Sector

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior


@dataclass
class _PendingCoalition:
    """Initiator-side state during the invitation/acceptance window.

    ``run`` flips True once allocated so a late acceptance can't
    re-trigger it.
    """

    coalition_id: str
    sector: Sector
    target_tiers: tuple[int, ...]
    member_aids: tuple[str, ...]
    started_at: float
    addr_by_aid: dict[str, Any] = field(default_factory=dict)
    acceptances: dict[str, CoalitionAcceptance] = field(default_factory=dict)
    run: bool = False


@dataclass
class _ActiveCoalition:
    """Initiator-side TTL record of an allocated coalition.

    Re-asserted every ``_tick`` until TTL expiry or a same-sector
    ``BranchFailureEvent``. ``member_addrs`` holds only accepting members.
    """

    coalition_id: str
    sector: Sector
    service_fraction_by_tier: dict[int, float]
    member_addrs: list[Any]
    issued_at: float
    ttl_s: float


class CrossSectorChannel:
    """Typed publish/read facade over the per-behavior cross-sector HolonSummary
    bus (``behavior._scare_xs_summaries``). Today single-process and lazy-init;
    the one seam a distributed transport would reimplement."""

    def __init__(self, store: dict[Sector, dict[str, HolonSummary]]) -> None:
        self._store = store

    @classmethod
    def for_behavior(
        cls, behavior: RestorationEnvironmentBehavior
    ) -> CrossSectorChannel:
        store = getattr(behavior, "_scare_xs_summaries", None)
        if store is None:
            store = {}
            behavior._scare_xs_summaries = store
        return cls(store)

    def publish(self, sector: Sector, key: str, summary: HolonSummary) -> None:
        self._store.setdefault(sector, {})[key] = summary

    def read(self, sector: Sector) -> dict[str, HolonSummary]:
        return self._store.get(sector, {})


@dataclass
class _ActiveCrossSectorCoalition:
    """Initiator-side TTL record of an allocated cross-sector coalition.

    Unlike :class:`_ActiveCoalition` the dispatch spans multiple sectors
    and includes per-CP commitments (directional flows).
    """

    coalition_id: str
    service_fraction_by_sector_tier: dict[str, dict[int, float]]
    leader_addrs_by_sector: dict[str, list[Any]]
    cp_targets_mw: dict[str, dict[str, float]]  # cp_aid -> sector_v -> mw
    cp_addrs: dict[str, Any]
    sectors: tuple[Sector, ...]
    issued_at: float
    ttl_s: float


class _CoalitionAggregate(NamedTuple):
    """Per-sector aggregation across a coalition's accepting members."""

    total_supply: float
    total_observed_served: float
    demand_by_tier: dict[int, float]
    served_by_tier: dict[int, float]
    actor_supplies: list[dict[str, float]]
    actor_demands: list[dict[str, dict[int, float]]]
    actor_node_ids: list[Any]
    actor_demand_nodes_by_tier: list[dict[int, dict[Any, float]]]
