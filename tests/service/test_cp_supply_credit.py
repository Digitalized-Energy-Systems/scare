"""Converter output must reach the L2 supply pool.

``supply_by_sector`` is summed over node children, but every converter is a
monee *branch* and so has no member aid. On ``simbench_lv_gas_dependent``
(``gas_gen_share=0`` — gas exists only as P2G output) that made every gas holon
read ``supply=0.0000`` and shed all 27 gas loads, tier 1 included, within 0.2 s,
pinning gas PWSF at exactly 0.
"""

from __future__ import annotations

import pytest

from scare.base.config import RestorationConfiguration
from scare.base.util import lookup_cp_supply, mw_to_kgps, publish_cp_supply
from scare.base.util.blackboard import _CP_SUPPLY_TTL_S
from scare.service.balance.balance import _credit_cp_supply


class _Behavior:
    """Minimal stand-in: the blackboard only needs an attribute holder."""


def test_publish_then_lookup_roundtrips():
    b = _Behavior()
    publish_cp_supply(b, "branch-1", {"L": {"gas": 0.0036}}, now=1.0)
    assert lookup_cp_supply(b, "branch-1", "L", now=1.0) == {"gas": 0.0036}


def test_lookup_expires_after_ttl():
    b = _Behavior()
    publish_cp_supply(b, "branch-1", {"L": {"gas": 0.0036}}, now=1.0)
    assert (
        lookup_cp_supply(b, "branch-1", "L", now=1.0 + _CP_SUPPLY_TTL_S - 0.01)
        is not None
    )
    assert lookup_cp_supply(b, "branch-1", "L", now=1.0 + _CP_SUPPLY_TTL_S) is None


def test_lookup_missing_is_none():
    assert lookup_cp_supply(_Behavior(), "nope", "L", now=0.0) is None


def test_zero_and_negative_entries_are_dropped():
    """Only produced quantities are credited; a draw is not negative supply."""
    b = _Behavior()
    publish_cp_supply(
        b,
        "branch-1",
        {"L": {"gas": 0.004, "electricity": -0.007, "heat": 0.0}},
        now=0.0,
    )
    assert lookup_cp_supply(b, "branch-1", "L", now=0.0) == {"gas": 0.004}


def test_credit_adds_converter_output_to_an_empty_gas_pool():
    """The regression: a gas pool with no native generation."""
    b = _Behavior()
    for i, mw in enumerate((0.0036, 0.0024)):
        publish_cp_supply(b, f"branch-{i}", {"L": {"gas": mw}}, now=0.0)
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["branch-0", "branch-1"], b, 0.0, "L")
    assert pool["gas"] == pytest.approx(mw_to_kgps(0.0060))


def test_gas_credit_is_converted_to_the_pools_native_kgps():
    """A CP's capacities are MW; the gas pool is ``mass_flow_kgs``.

    Crediting MW straight in would overstate gas supply by 3.6*HHV = 42.4x —
    a converter would look like it out-produced the whole fleet.
    """
    b = _Behavior()
    publish_cp_supply(b, "cp", {"L": {"gas": 1.0}}, now=0.0)
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["cp"], b, 0.0, "L")
    assert pool["gas"] == pytest.approx(mw_to_kgps(1.0))
    assert pool["gas"] < 0.05


def test_electricity_and_heat_credits_stay_in_mw():
    b = _Behavior()
    publish_cp_supply(b, "cp", {"L": {"electricity": 0.4, "heat": 0.25}}, now=0.0)
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["cp"], b, 0.0, "L")
    assert pool["electricity"] == pytest.approx(0.4)
    assert pool["heat"] == pytest.approx(0.25)


def test_credit_is_scoped_to_the_leaders_own_connectors():
    """A holon may only count converters it is actually coupled to."""
    b = _Behavior()
    publish_cp_supply(b, "mine", {"L": {"gas": 0.004}}, now=0.0)
    publish_cp_supply(b, "someone-elses", {"L": {"gas": 99.0}}, now=0.0)
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["mine"], b, 0.0, "L")
    assert pool["gas"] == pytest.approx(mw_to_kgps(0.004))


def test_credit_adds_to_existing_native_supply():
    b = _Behavior()
    publish_cp_supply(b, "branch-0", {"L": {"gas": 0.004}}, now=0.0)
    pool = {"gas": 0.010}
    _credit_cp_supply(pool, ["branch-0"], b, 0.0, "L")
    assert pool["gas"] == pytest.approx(0.010 + mw_to_kgps(0.004))


def test_stale_credit_is_not_counted():
    """A failed converter stops contributing on its own."""
    b = _Behavior()
    publish_cp_supply(b, "branch-0", {"L": {"gas": 0.004}}, now=0.0)
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["branch-0"], b, _CP_SUPPLY_TTL_S + 1.0, "L")
    assert pool == {}


@pytest.mark.parametrize("aids", [None, []])
def test_no_connectors_is_a_noop(aids):
    pool = {"gas": 0.010}
    _credit_cp_supply(pool, aids, _Behavior(), 0.0, "L")
    assert pool == {"gas": 0.010}


def test_config_default_is_on():
    assert RestorationConfiguration().enable_cp_supply_credit is True


def test_credit_addressed_to_another_leader_is_invisible():
    """The anti-inflation contract: a leader sees only its own share."""
    b = _Behavior()
    publish_cp_supply(b, "cp", {"leader-A": {"gas": 0.004}}, now=0.0)
    assert lookup_cp_supply(b, "cp", "leader-B", now=0.0) is None
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["cp"], b, 0.0, "leader-B")
    assert pool == {}


def test_split_across_leaders_is_conservative():
    """Summing a CP's credit over every leader must equal what it produced.

    mango hands all N leaders of a sector the same connector list, and
    holon_component sums supply across leaders — so an unaddressed credit would
    be counted N times. This is the property that prevents that.
    """
    b = _Behavior()
    produced = 0.009
    leaders = ["l1", "l2", "l3"]
    share = produced / len(leaders)
    publish_cp_supply(b, "cp", {ldr: {"gas": share} for ldr in leaders}, now=0.0)
    total = 0.0
    for ldr in leaders:
        pool: dict[str, float] = {}
        _credit_cp_supply(pool, ["cp"], b, 0.0, ldr)
        total += pool.get("gas", 0.0)
    assert total == pytest.approx(mw_to_kgps(produced))
