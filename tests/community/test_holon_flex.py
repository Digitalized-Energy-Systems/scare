"""Unit tests for the pure holon flex algebra extracted from HolonicCommunityRole."""

from __future__ import annotations

from scare.base.model import AvailableFlexAnswer, Sector
from scare.community.holon_flex import (
    aggregate_holon_flex,
    extract_demand_sectors_tiers,
)


def _ans(**kw):
    return AvailableFlexAnswer(
        flex=0.0,
        balance=0.0,
        shedded=0.0,
        sector=Sector.ELECTRICITY,
        supply_by_sector=kw.get("supply", {}),
        demand_by_sector_priority=kw.get("demand", {}),
        served_by_sector_priority=kw.get("served", {}),
    )


# --------------------------------------------------------------------------- #
# aggregate_holon_flex
# --------------------------------------------------------------------------- #


def test_aggregate_sums_supply_across_answers():
    answers = [
        _ans(supply={"electricity": 2.0, "heat": 1.0}),
        _ans(supply={"electricity": 3.0}),
    ]
    supply, _demand, _served = aggregate_holon_flex(answers)
    assert supply == {"electricity": 5.0, "heat": 1.0}


def test_aggregate_sums_demand_and_served_per_cell():
    answers = [
        _ans(
            demand={"electricity": {1: 2.0, 2: 1.0}}, served={"electricity": {1: 1.0}}
        ),
        _ans(
            demand={"electricity": {1: 0.5}, "heat": {3: 4.0}},
            served={"electricity": {1: 0.5}},
        ),
    ]
    _supply, demand, served = aggregate_holon_flex(answers)
    assert demand == {"electricity": {1: 2.5, 2: 1.0}, "heat": {3: 4.0}}
    assert served == {"electricity": {1: 1.5}}


def test_aggregate_tolerates_empty_and_none_fields():
    supply, demand, served = aggregate_holon_flex([_ans(), _ans()])
    assert supply == {} and demand == {} and served == {}


def test_aggregate_coerces_tier_keys_to_int():
    # String/float tier keys are normalised to int so cells merge correctly.
    answers = [
        _ans(demand={"gas": {2: 1.0}}),
        _ans(demand={"gas": {2.0: 1.0}}),
    ]
    _s, demand, _v = aggregate_holon_flex(answers)
    assert demand == {"gas": {2: 2.0}}


# --------------------------------------------------------------------------- #
# extract_demand_sectors_tiers
# --------------------------------------------------------------------------- #


def test_extract_basic():
    actor_demands = [
        {"electricity": {1: 2.0, 3: 1.0}},
        {"heat": {2: 0.5}, "electricity": {1: 1.0}},
    ]
    sectors, tiers, total = extract_demand_sectors_tiers(actor_demands)
    assert sectors == ["electricity", "heat"]  # sorted
    assert tiers == [1, 2, 3]  # sorted, present
    assert total == 4.5


def test_extract_filters_non_positive_tiers():
    sectors, tiers, total = extract_demand_sectors_tiers(
        [{"electricity": {0: 5.0, 2: 1.0}}]
    )
    assert sectors == ["electricity"]
    assert tiers == [2]  # tier 0 dropped (must be >= 1)
    assert total == 6.0  # but its demand still counts in the total


def test_extract_empty_inputs():
    assert extract_demand_sectors_tiers([]) == ([], [], 0.0)
    assert extract_demand_sectors_tiers([{}, None]) == ([], [], 0.0)
