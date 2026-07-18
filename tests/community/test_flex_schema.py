"""Lock for the SectorTierFlex mixin: the three Decision subclasses all carry the
flex quad from one place, with unchanged names/defaults."""

from __future__ import annotations

import dataclasses as dc

from scare.base.channel import (
    CoalitionAcceptance,
    ComponentAdmmReport,
    HolonSummary,
    SectorTierFlex,
)

FLEX_QUAD = {
    "supply_by_sector",
    "demand_by_sector_priority",
    "served_by_sector_priority",
}


def test_mixin_is_the_single_source():
    assert {f.name for f in dc.fields(SectorTierFlex)} == FLEX_QUAD


def test_all_three_dtos_carry_the_flex_quad():
    for cls in (HolonSummary, CoalitionAcceptance, ComponentAdmmReport):
        assert FLEX_QUAD <= {f.name for f in dc.fields(cls)}, cls.__name__


def test_flex_quad_defaults_are_empty_dicts():
    for cls in (HolonSummary, CoalitionAcceptance, ComponentAdmmReport):
        obj = cls(publisher="p", version=1)
        assert obj.supply_by_sector == {}
        assert obj.demand_by_sector_priority == {}
        assert obj.served_by_sector_priority == {}
