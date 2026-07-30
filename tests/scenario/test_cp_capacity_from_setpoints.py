"""CP capacities must come from sized setpoints, never from a monee solver Var.

``_cp_signed_capacity_by_sector`` fed the L3 CP-ADMM. Its p2g branch read
``el_mw``, which on ``monee.model.multi.PowerToGas`` is ``Var(1.1, min=0)`` — a
decision variable seeded at 1.1, not a rating. Every P2G therefore reported an
identical 1.1 MW while real rates varied 5x, overstating fleet gas output 214x
(20.02 MW against a measured 0.093) and fleet electricity draw to 64x the whole
grid's demand. The lexicographic cascade then drove every P2G to r=0 to protect
the electricity row, so gas coupling points never dispatched at all.

The invariant: signed capacity must reconcile with what the grid can actually
deliver, i.e. ``experiment.eval.metrics.cp_generation_breakdown`` at
``regulation == 1``.
"""

from __future__ import annotations

import pytest
from monee.model.core import Var
from monee.model.multi import PowerToGas

from scare.scenario.restoration import _cp_signed_capacity_by_sector


def test_powertogas_el_mw_is_a_var_not_a_rating():
    """Guards the premise: if monee ever makes ``el_mw`` a plain float rating,
    this test fails and the derivation below should be revisited."""
    p2g = PowerToGas(efficiency=0.7, mass_flow_setpoint_kgs=2.8e-05)
    assert isinstance(p2g.el_mw, Var)
    assert p2g.gas_mass_flow_kgs == pytest.approx(-2.8e-05)


def test_p2g_capacity_comes_from_the_mass_flow_setpoint():
    """Two differently-sized P2Gs must get different capacities. Pre-fix both
    returned {'electricity': 1.1, 'gas': -0.77} from the Var's initial value."""
    from scare.base.util import kgps_to_mw

    small = _cp_signed_capacity_by_sector(
        "p2g", dict(PowerToGas(efficiency=0.7, mass_flow_setpoint_kgs=2.8e-05).values)
    )
    large = _cp_signed_capacity_by_sector(
        "p2g", dict(PowerToGas(efficiency=0.7, mass_flow_setpoint_kgs=1.4e-04).values)
    )
    assert small != large
    # Gas is the OUTPUT (negative = produces) and equals the setpoint in MW.
    assert small["gas"] == pytest.approx(-kgps_to_mw(2.8e-05))
    assert large["gas"] == pytest.approx(-kgps_to_mw(1.4e-04))
    # 5x the setpoint => 5x the capacity, both sectors.
    assert large["gas"] / small["gas"] == pytest.approx(5.0)
    assert large["electricity"] / small["electricity"] == pytest.approx(5.0)
    # Electricity is the INPUT (positive = consumes), output / efficiency.
    assert small["electricity"] == pytest.approx(kgps_to_mw(2.8e-05) / 0.7)
    assert small["electricity"] > -small["gas"]  # losses => draw exceeds output


def test_p2g_capacity_is_nowhere_near_the_var_seed():
    """The specific regression: 0.77 MW gas / 1.1 MW el per unit was the Var."""
    caps = _cp_signed_capacity_by_sector(
        "p2g", dict(PowerToGas(efficiency=0.7, mass_flow_setpoint_kgs=2.8e-05).values)
    )
    assert abs(caps["gas"]) < 0.01
    assert caps["electricity"] < 0.01


def test_zero_setpoint_yields_no_capacity():
    assert (
        _cp_signed_capacity_by_sector(
            "p2g", dict(PowerToGas(efficiency=0.7, mass_flow_setpoint_kgs=0.0).values)
        )
        == {}
    )


@pytest.mark.parametrize(
    "grid", ["simbench_lv_gas_dependent", "simbench_lv_el_dependent"]
)
def test_fleet_capacity_reconciles_with_deliverable_output(grid):
    """End-to-end on the real grids: summed signed PRODUCTION capacity must
    match what the fleet can actually deliver at regulation 1."""
    from experiment.eval.metrics import cp_generation_breakdown
    from experiment.scenarios import GRIDS
    from scare.scenario.restoration import _detect_cp_type_for_node, _model_type_name

    net = GRIDS[grid]()
    truth = cp_generation_breakdown(net)

    agg: dict[str, float] = {}
    for branch in net.branches:
        bt = _model_type_name(branch).lower()
        ct = next(
            (
                c
                for k, c in (
                    ("powertogas", "p2g"),
                    ("gastopower", "g2p"),
                    ("powertoheat", "p2h"),
                )
                if k in bt
            ),
            None,
        )
        if ct is None:
            continue
        for k, v in _cp_signed_capacity_by_sector(
            ct, dict(branch.model.values)
        ).items():
            agg[k] = agg.get(k, 0.0) + v
    for node in net.nodes:
        ct = _detect_cp_type_for_node(node, net)
        if ct is None or "chp" not in ct.lower():
            continue
        for k, v in _cp_signed_capacity_by_sector(ct, dict(node.model.values)).items():
            agg[k] = agg.get(k, 0.0) + v

    for sector, key in (("gas", "gas_mw"), ("heat", "heat_mw")):
        deliverable = truth.get(key, 0.0)
        if deliverable <= 0:
            continue
        produced = -min(0.0, agg.get(sector, 0.0))
        assert produced == pytest.approx(deliverable, rel=0.1), (
            f"{grid}/{sector}: kernel capacity {produced} vs deliverable {deliverable}"
        )
