"""The CP-necessity grids encode their dependence in the factory recipe.

Guards the knobs the measured dependence rests on (validated with the
no-failure oracle LP: p2g gas 1.000 -> 0.450 and chp electricity 1.000 ->
0.734 when every coupling point is deactivated). Reads the factory closure
rather than building — a build is ~20 s.
"""

import pytest

from experiment.eval.grid_scenario_table import _closure_params
from experiment.scenarios import GRIDS


@pytest.mark.parametrize(
    "grid", ["simbench_lv_gas_dependent", "simbench_lv_el_dependent"]
)
def test_registered(grid):
    assert grid in GRIDS


def test_p2g_grid_has_no_native_gas_production():
    p = _closure_params("simbench_lv_gas_dependent")
    assert p["gas_kwargs"]["gas_gen_share"] == 0.0
    assert set(p["couplings"]) == {"p2g", "p2h"}
    # Surplus PV is what P2G converts; without it the electrical balance
    # cannot carry the converter draw.
    assert p["primary_gen_scale"] > 1.0


def test_chp_grid_starves_primary_generation():
    p = _closure_params("simbench_lv_el_dependent")
    assert set(p["couplings"]) == {"chp", "p2h"}
    assert p["primary_gen_scale"] < 0.5
    # CHPs need fuel: gas keeps its distributed feed-in.
    assert p["gas_kwargs"]["gas_gen_share"] > 0.0


@pytest.mark.parametrize(
    "grid", ["simbench_lv_gas_dependent", "simbench_lv_el_dependent"]
)
def test_heat_leans_on_converters(grid):
    p = _closure_params(grid)
    # Token distributed heat fleet only. 0.0 is NOT usable: the DHS goes
    # infeasible on the long supply pipes without any local injection.
    assert 0.0 < p["heat_kwargs"]["node_heat_gen_share"] <= 0.2


@pytest.mark.parametrize(
    "grid", ["simbench_lv_gas_dependent", "simbench_lv_el_dependent"]
)
def test_no_heat_slack_budget(grid):
    # Bounding the heat slack is infeasible rather than merely costly on these
    # grids (hydraulically pinned), so heat dependence comes from the supply
    # mix. Re-introducing a cap here would silently break every task.
    assert _closure_params(grid).get("heat_slack_budget_kgs") is None
