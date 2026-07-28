"""The summary delta gate must mean the same physical quantity in every sector.

``holon_summary_inversion_tol`` is an ABSOLUTE threshold, but per-tier values
are MW for electricity/heat and native kg/s for gas. Compared raw, the gate was
~42x looser for gas: on ``simbench_lv_gas_dependent`` the whole grid's gas
demand is 0.0036 kg/s against a 1e-3 tolerance, so all 17 load-carrying leaders
could shed their ENTIRE load without the gate opening. Gas ``served`` then
stayed frozen at the pre-dispatch t~0.08 snapshot — where ``served == demand``
because nothing is regulated yet — and the L3 CP kernel, which reads exactly
this field, was told gas was 100% served while 60% of it was shed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scare.base.model import Sector
from scare.base.util import kgps_to_mw
from scare.community.summary_publish import SummaryPublisher


def _publisher(sector: Sector, tol: float = 1e-3) -> SummaryPublisher:
    return SummaryPublisher(SimpleNamespace(sector=sector, inversion_tol=tol))


def _seed(pub: SummaryPublisher, served: dict, demand: dict) -> None:
    pub._last_published_served = dict(served)
    pub._last_published_demand = dict(demand)


def test_first_publish_always_passes():
    assert _publisher(Sector.GAS)._summary_changed({1: 0.0008}, {1: 0.0008})


@pytest.mark.parametrize("sector", [Sector.ELECTRICITY, Sector.HEAT])
def test_mw_sectors_keep_the_absolute_tolerance(sector):
    """Electricity/heat are already MW — behaviour must be unchanged."""
    pub = _publisher(sector)
    _seed(pub, {1: 0.5}, {1: 0.5})
    assert not pub._summary_changed({1: 0.4995}, {1: 0.5})  # 5e-4 < 1e-3
    assert pub._summary_changed({1: 0.498}, {1: 0.5})  # 2e-3 > 1e-3


def test_gas_shedding_its_entire_load_opens_the_gate():
    """The regression: a gas leader sheds 100% of its load. Raw kg/s deltas are
    all far below 1e-3, so the pre-fix gate stayed shut and L3 never learned."""
    pub = _publisher(Sector.GAS)
    _seed(pub, {3: 0.0006, 4: 0.0002}, {3: 0.0006, 4: 0.0002})
    # Total shed: served -> 0 on both tiers. Raw delta 6e-4 < 1e-3 tol.
    assert pub._summary_changed({3: 0.0, 4: 0.0}, {3: 0.0006, 4: 0.0002})


def test_gas_partial_shed_seen_in_the_field_opens_the_gate():
    """Measured on simbench_lv_gas_dependent: child-148 went 0.000800 ->
    0.000155 kg/s served. Raw delta 6.45e-4 < 1e-3 was gated pre-fix."""
    pub = _publisher(Sector.GAS)
    _seed(pub, {3: 0.0006, 4: 0.0002}, {3: 0.0006, 4: 0.0002})
    assert pub._summary_changed({3: 0.000155, 4: 0.0}, {3: 0.0006, 4: 0.0002})


def test_gas_noise_below_the_tolerance_is_still_gated():
    """The gate must still suppress genuine noise, or every tick republishes.
    1e-3 MW back-converted is ~2.36e-5 kg/s; stay an order of magnitude under."""
    pub = _publisher(Sector.GAS)
    _seed(pub, {3: 0.0006}, {3: 0.0006})
    tiny = 1e-3 / kgps_to_mw(1.0) / 10.0
    assert not pub._summary_changed({3: 0.0006 - tiny}, {3: 0.0006})


def test_gas_threshold_sits_at_the_mw_equivalent():
    """Scale factor is exactly the kg/s -> MW conversion, so the tolerance
    means 1e-3 MW in every sector."""
    pub = _publisher(Sector.GAS)
    assert pub._tol_scale() == pytest.approx(kgps_to_mw(1.0))
    assert _publisher(Sector.ELECTRICITY)._tol_scale() == 1.0
    assert _publisher(Sector.HEAT)._tol_scale() == 1.0

    _seed(pub, {1: 0.001}, {1: 0.001})
    just_under = 1e-3 / kgps_to_mw(1.0) * 0.9
    just_over = 1e-3 / kgps_to_mw(1.0) * 1.1
    assert not pub._summary_changed({1: 0.001 - just_under}, {1: 0.001})
    assert pub._summary_changed({1: 0.001 - just_over}, {1: 0.001})


def test_demand_side_is_scaled_too():
    """A load disconnecting moves demand, not served; it must also open."""
    pub = _publisher(Sector.GAS)
    _seed(pub, {2: 0.0004}, {2: 0.0004})
    assert pub._summary_changed({2: 0.0004}, {2: 0.0})
