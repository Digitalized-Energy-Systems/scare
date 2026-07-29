"""Heat tier MW spent at the corners instead of one uniform fraction.

The oracle and both baselines put 82.8-84.1% of heat loads at a corner (0 or 1)
and take 0/90 t_k violations on eval_full_v2_20260728-202054 cold_day_stress;
SCARE's uniform per-tier fraction puts 54.5% there and takes 46/90.
"""

from types import SimpleNamespace

import pytest

from scare.base.model import Sector
from scare.service.balance.balance import L2DispatchHandler


def _negotiator(loads):
    """``loads``: {aid: (cap_mw, tier)}.  Every entry is a heat load."""
    obs = {
        aid: {"q_mw_heat": cap, "priority": tier, "regulation": 1.0}
        for aid, (cap, tier) in loads.items()
    }
    role = SimpleNamespace(
        behavior=SimpleNamespace(observe=lambda aid: obs.get(aid)),
        _grid_former_policy=SimpleNamespace(is_former=lambda aid: False),
    )
    return L2DispatchHandler.__new__(L2DispatchHandler), role


def _factors(loads, service_fraction):
    neg, role = _negotiator(loads)
    neg._role = role
    return neg._heat_corner_factors(list(loads), service_fraction)


def test_tier_mw_is_preserved_exactly():
    """Redistribution only — the tier's served MW must not move, or the tier
    ladder and every PWSF aggregate would shift with it."""
    loads = {f"child-{i}": (0.0075, 3) for i in range(10)}
    frac = 0.64
    got = _factors(loads, {Sector.HEAT.value: {3: frac}})
    total_cap = sum(c for c, _ in loads.values())
    served = sum(got[a] * loads[a][0] for a in loads)
    assert served == pytest.approx(frac * total_cap)


def test_allocation_is_at_the_corners():
    """At most ONE boundary load per tier may sit in the partial middle."""
    loads = {f"child-{i}": (0.0075, 3) for i in range(10)}
    got = _factors(loads, {Sector.HEAT.value: {3: 0.64}})
    middle = [f for f in got.values() if 1e-9 < f < 1 - 1e-9]
    assert len(middle) <= 1
    assert sum(1 for f in got.values() if f == pytest.approx(1.0)) == 6


def test_tiers_are_independent():
    """Each tier spends its own allocation; a tier at 1.0 is untouched."""
    loads = {"a": (0.01, 1), "b": (0.01, 1), "c": (0.02, 4), "d": (0.02, 4)}
    got = _factors(loads, {Sector.HEAT.value: {1: 1.0, 4: 0.5}})
    assert got["a"] == 1.0 and got["b"] == 1.0
    assert sorted([got["c"], got["d"]]) == [0.0, 1.0]


def test_ordering_is_deterministic_largest_first():
    """Reproducibility: unseeded ordering has broken campaign determinism
    before, so selection must be a pure function of (cap, aid)."""
    loads = {"child-9": (0.01, 2), "child-1": (0.03, 2), "child-5": (0.02, 2)}
    runs = [_factors(loads, {Sector.HEAT.value: {2: 0.5}}) for _ in range(5)]
    assert all(r == runs[0] for r in runs)
    assert runs[0]["child-1"] == 1.0  # largest cap served first


def test_zero_fraction_sheds_the_whole_tier():
    loads = {"a": (0.01, 4), "b": (0.02, 4)}
    got = _factors(loads, {Sector.HEAT.value: {4: 0.0}})
    assert got == {"a": 0.0, "b": 0.0}


def test_tier_without_an_allocation_is_left_alone():
    """No entry for the tier means 'preserve current state' upstream, so the
    map must not name those loads at all."""
    loads = {"a": (0.01, 2), "b": (0.01, 3)}
    got = _factors(loads, {Sector.HEAT.value: {2: 0.5}})
    assert "b" not in got
