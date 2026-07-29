"""What the L3 CP kernel is told about each sector's supply.

Two independent ways ``CPPriorityAdmmRole._build_demands`` mis-stated it, both
measured on ``eval_full_v2_20260728-202054``, where SCARE's coupling-point fleet
delivered 4-30% of what the component-level baseline delivered on every grid and
exactly 0.0000 MW of CP gas on 5 of 10.

1. **Level.** ``base_supply`` for a CP-input sector was ``Σ served + the whole
   slack budget``. ``served`` is a flow already drawing on that budget, so the
   slack was counted twice and a shedding sector reported a surplus: task 004610
   published ``served=0.2378 slack=0.2763`` against ``demand=0.4456`` — a 47%
   shed read as +0.068 MW spare. That zeroes the waterfall marginal, and the
   ``minimize_usage`` ridge then pulls every ``r`` to 0.

2. **Dynamics.** ``base_supply`` for a sector the fleet PRODUCES into is
   ``Σ served``, which already contains the fleet's own delivery; the kernel
   adds it back as ``Σ r_i·c_i``. The round-to-round map is therefore
   ``r_{k+1} = R − r_k`` — gain exactly −1, which orbits instead of converging.
   Fingerprint in the campaign: lag-1 autocorrelation of the fleet-mean factor
   negative in 22 of 25 sampled tasks (median −0.74, four below −0.97) while the
   consecutive-PAIR sum stayed near-constant, and el_dependent ending with 27
   CPs at exactly 0.0 and 20 at exactly 1.0.

Both fixes are flagged (``enable_cp_slack_headroom`` /
``enable_cp_own_supply_netting``) and the off-path is asserted here too, so the
pre-fix behaviour stays available as the A/B counterfactual.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from scare.base.channel import CPSummary
from scare.base.model import Sector
from scare.service.coupling.cp_priority_admm_role import CPPriorityAdmmRole
from tests.conftest import MockBehavior
from tests.service.test_cp_priority_admm_wiring import (
    _Addr,
    _inject_holon_summary,
    _make_role,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _p2h_role(**attrs: Any) -> tuple[CPPriorityAdmmRole, MockBehavior]:
    role, _, behavior = _make_role(
        "p2h-A",
        capacity_by_sector={"heat": -0.05, "electricity": 0.05},
        bridged_sectors=[Sector.HEAT, Sector.ELECTRICITY],
    )
    role.heat_supply_from_deficit = True
    for k, v in attrs.items():
        setattr(role, k, v)
    return role, behavior


def _base_supply(role: CPPriorityAdmmRole, sector: Sector) -> float:
    return float(
        {d.sector: d for d in role._build_demands()}[sector.value].base_supply[0]
    )


# ---------------------------------------------------------------------------
# 1. Level — unused headroom, not the whole budget
# ---------------------------------------------------------------------------


def test_input_sector_base_supply_uses_headroom_not_budget() -> None:
    """Budget 0.168 with 0.15 of it already flowing leaves 0.018 to offer."""
    role, _ = _p2h_role(slack_headroom=True)
    _inject_holon_summary(
        role,
        leader_aid="el-leader",
        sector=Sector.ELECTRICITY,
        supply_mw=10.0,
        slack_budget_mw=0.168,
        slack_headroom_mw=0.018,
        demand_by_tier={1: 0.4},
        served_by_tier={1: 0.37},
    )
    assert _base_supply(role, Sector.ELECTRICITY) == pytest.approx(0.37 + 0.018)


def test_shedding_sector_no_longer_reports_a_surplus() -> None:
    """The regression, at task 004610's published numbers."""
    served, budget, headroom, demand = 0.237774, 0.276287, 0.038513, 0.445600

    def supply_with(flag: bool) -> float:
        role, _ = _p2h_role(slack_headroom=flag)
        _inject_holon_summary(
            role,
            leader_aid="el-leader",
            sector=Sector.ELECTRICITY,
            supply_mw=10.0,
            slack_budget_mw=budget,
            slack_headroom_mw=headroom,
            demand_by_tier={3: demand},
            served_by_tier={3: served},
        )
        return _base_supply(role, Sector.ELECTRICITY)

    assert supply_with(False) > demand  # sign-flipped: phantom surplus
    assert supply_with(True) < demand  # real deficit
    assert supply_with(True) == pytest.approx(served + headroom)


def test_a_maxed_slack_offers_nothing() -> None:
    """No headroom left ⇒ base supply is exactly the delivered flow, so a CP
    that wants to draw has to outbid a load rather than get it free."""
    role, _ = _p2h_role(slack_headroom=True)
    _inject_holon_summary(
        role,
        leader_aid="el-leader",
        sector=Sector.ELECTRICITY,
        supply_mw=10.0,
        slack_budget_mw=0.168,
        slack_headroom_mw=0.0,
        demand_by_tier={1: 0.4},
        served_by_tier={1: 0.37},
    )
    assert _base_supply(role, Sector.ELECTRICITY) == pytest.approx(0.37)


def test_gas_headroom_is_converted_out_of_kg_per_s() -> None:
    """Gas rides the same path; the summary is native kg/s and the kernel is MW."""
    from scare.base.util import kgps_to_mw

    role, _, _ = _make_role(
        "p2g-A",
        capacity_by_sector={"gas": -0.03, "electricity": 0.05},
        bridged_sectors=[Sector.GAS, Sector.ELECTRICITY],
    )
    role.heat_supply_from_deficit = True
    role.slack_headroom = True
    _inject_holon_summary(
        role,
        leader_aid="gas-leader",
        sector=Sector.GAS,
        supply_mw=5.0,
        slack_budget_mw=0.01,
        slack_headroom_mw=0.002,
        demand_by_tier={1: 0.04},
        served_by_tier={1: 0.03},
    )
    assert _base_supply(role, Sector.GAS) == pytest.approx(kgps_to_mw(0.032))


# ---------------------------------------------------------------------------
# 2. Dynamics — the fleet must not credit itself
# ---------------------------------------------------------------------------


def _heat_role(*, netting: bool, regulation: float) -> CPPriorityAdmmRole:
    role, behavior = _p2h_role(own_supply_netting=netting)
    behavior.set_obs("p2h-A", {"regulation": regulation})
    return role


def test_own_production_is_netted_out_of_the_heat_base_supply() -> None:
    role = _heat_role(netting=True, regulation=0.6)
    _inject_holon_summary(
        role,
        leader_aid="heat-leader",
        sector=Sector.HEAT,
        supply_mw=0.0,
        demand_by_tier={1: 0.8},
        served_by_tier={1: 0.3},
    )
    # 0.3 delivered − this CP's own 0.6 × 0.05.
    assert _base_supply(role, Sector.HEAT) == pytest.approx(0.27)


def test_netting_makes_base_supply_independent_of_the_previous_factor() -> None:
    """The property the fix exists for.

    Physically ``served(r) = base + r·|c|``. Read whole, that makes the kernel's
    answer a function of the factor it committed last round (gain −1). Netted,
    the same physical state yields the same ``base`` whatever the fleet is
    currently doing — which is what lets the cascade have a fixed point.
    """
    base, cap = 0.24, 0.05

    def base_supply_for(r: float, *, netting: bool) -> float:
        role = _heat_role(netting=netting, regulation=r)
        _inject_holon_summary(
            role,
            leader_aid="heat-leader",
            sector=Sector.HEAT,
            supply_mw=0.0,
            demand_by_tier={1: 0.8},
            served_by_tier={1: base + r * cap},
        )
        return _base_supply(role, Sector.HEAT)

    factors = (0.0, 0.25, 0.6, 1.0)
    assert [base_supply_for(r, netting=True) for r in factors] == pytest.approx(
        [base] * 4
    )

    unnetted = [base_supply_for(r, netting=False) for r in factors]
    assert unnetted[-1] - unnetted[0] == pytest.approx(cap)  # gain −1 restored


def test_netting_counts_reachable_peers_and_never_goes_negative() -> None:
    """The netted fleet is exactly the kernel's participant set, and a lagging
    ``served`` must not drive base supply below zero."""
    role = _heat_role(netting=True, regulation=1.0)
    role._peer_cps["p2h-B"] = CPSummary(
        publisher="p2h-B",
        version=1,
        caused_by={},
        timestamp_s=100.0,
        capacity_by_sector={"heat": -0.04},
        regulation=0.5,
    )
    _inject_holon_summary(
        role,
        leader_aid="heat-leader",
        sector=Sector.HEAT,
        supply_mw=0.0,
        demand_by_tier={1: 0.8},
        served_by_tier={1: 0.3},
    )
    # own 1.0 × 0.05 + peer 0.5 × 0.04 = 0.07.
    assert _base_supply(role, Sector.HEAT) == pytest.approx(0.23)

    role._peer_cps["p2h-B"].capacity_by_sector = {"heat": -4.0}
    assert _base_supply(role, Sector.HEAT) == 0.0


def test_a_consuming_peer_is_not_netted() -> None:
    """Only the produced side is inside ``served``; a CP that draws from the
    sector is booked as demand elsewhere and must not be subtracted."""
    role = _heat_role(netting=True, regulation=1.0)
    role._peer_cps["g2p-B"] = CPSummary(
        publisher="g2p-B",
        version=1,
        caused_by={},
        timestamp_s=100.0,
        capacity_by_sector={"heat": 0.04},
        regulation=1.0,
    )
    _inject_holon_summary(
        role,
        leader_aid="heat-leader",
        sector=Sector.HEAT,
        supply_mw=0.0,
        demand_by_tier={1: 0.8},
        served_by_tier={1: 0.3},
    )
    assert _base_supply(role, Sector.HEAT) == pytest.approx(0.25)  # own 0.05 only


def test_a_never_committed_cp_is_netted_at_its_born_factor() -> None:
    """Converters are born ``regulation=1`` and contribute to ``served`` from
    the first solve, so the netting has to see that, not a 0 default."""
    role, behavior = _p2h_role(own_supply_netting=True)
    behavior.set_obs("p2h-A", {})  # no regulation key at all
    _inject_holon_summary(
        role,
        leader_aid="heat-leader",
        sector=Sector.HEAT,
        supply_mw=0.0,
        demand_by_tier={1: 0.8},
        served_by_tier={1: 0.3},
    )
    assert role._standing_regulation() == pytest.approx(1.0)
    assert _base_supply(role, Sector.HEAT) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# The factor has to reach the peers that net it
# ---------------------------------------------------------------------------


def test_cp_summary_carries_the_standing_regulation() -> None:
    role, ctx, behavior = _make_role("p2h-A", capacity_by_sector={"heat": -0.05})
    behavior.set_obs("p2h-A", {"regulation": 0.42})
    role._peer_cp_addrs = {"p2h-B": _Addr("p2h-B")}
    asyncio.run(role._publish(force=True))
    assert ctx.sent[-1].payload.regulation == pytest.approx(0.42)


def test_a_factor_change_alone_reopens_the_delta_gate() -> None:
    """The gate was capacity-only, so a factor that moved without a capacity
    shift never reached the peers that subtract it."""
    role, ctx, behavior = _make_role("p2h-A", capacity_by_sector={"heat": -0.05})
    behavior.set_obs("p2h-A", {"regulation": 1.0})
    role._peer_cp_addrs = {"p2h-B": _Addr("p2h-B")}
    asyncio.run(role._publish(force=True))
    n = len(ctx.sent)

    asyncio.run(role._publish())  # nothing moved
    assert len(ctx.sent) == n

    behavior.set_obs("p2h-A", {"regulation": 0.2})  # capacity unchanged
    asyncio.run(role._publish())
    assert len(ctx.sent) == n + 1
    assert ctx.sent[-1].payload.regulation == pytest.approx(0.2)
