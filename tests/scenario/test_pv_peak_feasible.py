"""``pv_peak`` must leave the physics LP solvable before anything happens.

In ``eval_full_v2_20260724-141520`` the simulation solve was infeasible at
``t=0`` on 350 of the 365 ``pv_peak`` tasks — with no failure applied and no
agent having written a setpoint, so every one of them fed the *unsolved* net to
its observers for the first few steps. The binding row was the substation's
MISOCP thermal cap (``_ELL_THERMAL_HEADROOM = 3.0`` x rated), not the slack
budget the modifier widens.
"""

import pytest

from experiment.scenarios import GRIDS
from experiment.scenarios.modifiers import _is_transformer, apply_pv_peak

_MIN_DT_H = 1e-9


def _step0_solves(net) -> bool:
    from mango_energy_environments.base.monee import create_physics_stepper

    result = create_physics_stepper(net, solve_time_limit_s=120).step(_MIN_DT_H)
    return not getattr(result, "failed", False)


def test_transformer_detected_by_voltage_level():
    # from_pandapower routes through MATPOWER: the substation is a plain
    # GenericPowerBranch at tap 1.0, so only the base_kv step identifies it.
    net = GRIDS["simbench_lv_small"]()
    trafos = [b for b in net.branches if _is_transformer(net, b)]
    assert [tuple(b.id) for b in trafos] == [(1, 5, 0)]


def test_uprate_scales_only_transformers():
    stock = GRIDS["simbench_lv_small"]()
    ratings = {
        tuple(b.id): float(b.model.max_i_ka)
        for b in stock.branches
        if hasattr(b.model, "max_i_ka")
    }

    net = GRIDS["simbench_lv_small"]()
    apply_pv_peak(net, trafo_ampacity_scale=2.0)
    for b in net.branches:
        if not hasattr(b.model, "max_i_ka"):
            continue
        expected = ratings[tuple(b.id)] * (2.0 if _is_transformer(net, b) else 1.0)
        assert float(b.model.max_i_ka) == pytest.approx(expected)


def test_scale_one_is_a_no_op_on_ratings():
    stock = GRIDS["simbench_lv_small"]()
    ratings = {
        tuple(b.id): float(b.model.max_i_ka)
        for b in stock.branches
        if hasattr(b.model, "max_i_ka")
    }
    net = GRIDS["simbench_lv_small"]()
    apply_pv_peak(net, trafo_ampacity_scale=1.0)
    for b in net.branches:
        if hasattr(b.model, "max_i_ka"):
            assert float(b.model.max_i_ka) == pytest.approx(ratings[tuple(b.id)])


@pytest.mark.slow
def test_pv_peak_step0_is_feasible_at_campaign_defaults():
    net = GRIDS["simbench_lv_small"]()
    apply_pv_peak(net)
    assert _step0_solves(net)


@pytest.mark.slow
def test_stock_rating_is_what_made_it_infeasible():
    """Pins the diagnosis: same scenario, only the uprate removed."""
    net = GRIDS["simbench_lv_small"]()
    apply_pv_peak(net, trafo_ampacity_scale=1.0)
    assert not _step0_solves(net)
