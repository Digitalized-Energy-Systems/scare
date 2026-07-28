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
from scare.base.util import lookup_cp_supply, publish_cp_supply
from scare.base.util.blackboard import _CP_SUPPLY_TTL_S
from scare.service.balance.balance import _credit_cp_supply


class _Behavior:
    """Minimal stand-in: the blackboard only needs an attribute holder."""


def test_publish_then_lookup_roundtrips():
    b = _Behavior()
    publish_cp_supply(b, "branch-1", {"gas": 0.0036}, now=1.0)
    assert lookup_cp_supply(b, "branch-1", now=1.0) == {"gas": 0.0036}


def test_lookup_expires_after_ttl():
    b = _Behavior()
    publish_cp_supply(b, "branch-1", {"gas": 0.0036}, now=1.0)
    assert lookup_cp_supply(b, "branch-1", now=1.0 + _CP_SUPPLY_TTL_S - 0.01) is not None
    assert lookup_cp_supply(b, "branch-1", now=1.0 + _CP_SUPPLY_TTL_S) is None


def test_lookup_missing_is_none():
    assert lookup_cp_supply(_Behavior(), "nope", now=0.0) is None


def test_zero_and_negative_entries_are_dropped():
    """Only produced quantities are credited; a draw is not negative supply."""
    b = _Behavior()
    publish_cp_supply(b, "branch-1", {"gas": 0.004, "electricity": -0.007, "heat": 0.0}, now=0.0)
    assert lookup_cp_supply(b, "branch-1", now=0.0) == {"gas": 0.004}


def test_credit_adds_converter_output_to_an_empty_gas_pool():
    """The regression: a gas pool with no native generation."""
    b = _Behavior()
    for i, mw in enumerate((0.0036, 0.0024)):
        publish_cp_supply(b, f"branch-{i}", {"gas": mw}, now=0.0)
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["branch-0", "branch-1"], b, now=0.0)
    assert pool["gas"] == pytest.approx(0.0060)


def test_credit_is_scoped_to_the_leaders_own_connectors():
    """A holon may only count converters it is actually coupled to."""
    b = _Behavior()
    publish_cp_supply(b, "mine", {"gas": 0.004}, now=0.0)
    publish_cp_supply(b, "someone-elses", {"gas": 99.0}, now=0.0)
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["mine"], b, now=0.0)
    assert pool["gas"] == pytest.approx(0.004)


def test_credit_adds_to_existing_native_supply():
    b = _Behavior()
    publish_cp_supply(b, "branch-0", {"gas": 0.004}, now=0.0)
    pool = {"gas": 0.010}
    _credit_cp_supply(pool, ["branch-0"], b, now=0.0)
    assert pool["gas"] == pytest.approx(0.014)


def test_stale_credit_is_not_counted():
    """A failed converter stops contributing on its own."""
    b = _Behavior()
    publish_cp_supply(b, "branch-0", {"gas": 0.004}, now=0.0)
    pool: dict[str, float] = {}
    _credit_cp_supply(pool, ["branch-0"], b, now=_CP_SUPPLY_TTL_S + 1.0)
    assert pool == {}


@pytest.mark.parametrize("aids", [None, []])
def test_no_connectors_is_a_noop(aids):
    pool = {"gas": 0.010}
    _credit_cp_supply(pool, aids, _Behavior(), now=0.0)
    assert pool == {"gas": 0.010}


def test_config_default_is_on():
    assert RestorationConfiguration().enable_cp_supply_credit is True
