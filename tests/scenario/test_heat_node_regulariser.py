"""``heat_node_reg_kgs`` must reach the sim AND the oracle identically.

The nodal heat balance in monee keeps a non-zero derivative in ``T_n`` only via
a conduction term scaled by ``grid.node_heat_reg_kgs``. No grid class declared
that attribute, so the guard was unreachable: a zero-flow junction's temperature
was pinned only by the ``t_pu in [0.3, 2.0]`` Var box, and the MIQCQP parked it
at exactly ``0.3 * t_ref_k``.

It is deliberately a **scenario** key, not a ``RestorationConfiguration`` field:
the oracle solve never reads the config, so a config field would regularise the
MAS physics and leave the oracle unregularised — an actuator-parity break rather
than an ablation.
"""

from __future__ import annotations

import monee.model as mm
import pytest

from experiment.scenarios import apply_heat_node_regulariser


def _mes_with_water():
    grid = mm.create_water_grid("water")
    net = mm.Network(grid)
    net.node(
        mm.Junction(),
        child_ids=[net.child(mm.ExtHydrGrid(pressure_pu=1.0, t_k=356))],
    )
    net.node(mm.Junction(), child_ids=[net.child(mm.Sink(mass_flow_kgs=0.1))])
    return net


def test_water_grid_declares_the_attribute_the_balance_reads():
    assert mm.create_water_grid("water").node_heat_reg_kgs == 0.0


def test_applies_to_every_water_grid_and_reports_the_count():
    net = _mes_with_water()
    assert apply_heat_node_regulariser(net, 1e-3) == 1
    assert all(g.node_heat_reg_kgs == 1e-3 for g in net.grids)


def test_zero_is_a_real_setting_not_a_no_op_path():
    """0.0 must still be applied — it is the byte-identical baseline arm."""
    net = _mes_with_water()
    apply_heat_node_regulariser(net, 1e-3)
    assert apply_heat_node_regulariser(net, 0.0) == 1
    assert all(g.node_heat_reg_kgs == 0.0 for g in net.grids)


def test_a_net_without_a_heat_sector_reports_zero_rather_than_claiming_success():
    grid = mm.create_power_grid("power")
    net = mm.Network(grid)
    net.node(mm.Bus(base_kv=0.4))
    assert apply_heat_node_regulariser(net, 1e-3) == 0


def test_gas_and_power_grids_do_not_acquire_a_heat_coefficient():
    """Duck-typed on the declared attribute so a non-heat grid is untouched."""
    for factory, name in ((mm.create_power_grid, "p"), (mm.create_gas_grid, "g")):
        grid = factory(name)
        assert not hasattr(grid, "node_heat_reg_kgs")


@pytest.mark.parametrize("value", [1e-6, 1e-3, 0.5])
def test_value_is_stored_as_a_float(value):
    net = _mes_with_water()
    apply_heat_node_regulariser(net, value)
    stored = next(g.node_heat_reg_kgs for g in net.grids)
    assert isinstance(stored, float) and stored == pytest.approx(value)
