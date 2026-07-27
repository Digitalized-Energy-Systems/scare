"""A promoted grid-former needs dispatch headroom over its setpoint.

``apply_microgrid_islanding`` rated a promoted ``GridFormingGenerator`` at the
magnitude of its *pre-failure setpoint* (~1.4 kW for simbench PV). That is
enough to anchor an island's voltage but not to carry its load, and the island
solve is then infeasible with no way out: de-energising the unservable buses
does not help, only capacity does (both verified below). It is the
active-power twin of the reactive-capability bug already fixed in the same
function.

Family C of ``eval_full_v2_20260724-141520`` — 6 tasks, the worst per-task
damage in the campaign (median 94 failed steps of ~140).
"""

import logging

import pytest

from monee.model.child import PowerGenerator
from monee.model.extension.islanding.el import GridFormingGenerator

from experiment.hpc.config import TaskSpec
from experiment.scenarios import GRIDS, apply_microgrid_islanding

_MIN_DT_H = 1e-9


def _promoted_ratings(net) -> dict:
    """``{child_id: p_mw_max}`` for promoted formers only.

    Not ``GridFormingMixin``: ``ExtPowerGrid`` shares that mixin and leaves
    ``p_mw.max`` at None, which is not a rating.
    """
    return {
        c.id: float(c.model.p_mw.max)
        for c in net.childs
        if isinstance(c.model, GridFormingGenerator)
    }

#: eval_full_v2 task 2044's infeasibility_snapshot.json at sim_t=2.18.
_INACTIVE = [(6, 104, 0), (96, 8, 0)]
_REGULATIONS = {
    2: 0.0, 4: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 25: 0.0, 31: 0.0, 35: 0.0,
    36: 0.0, 43: 0.0, 46: 0.0, 51: 0.0, 60: 0.0, 62: 0.0, 64: 0.0, 71: 0.0,
    80: 0.0, 86: 0.0, 92: 0.932401668148459, 100: 0.0, 114: 0.0, 115: 0.0,
    116: 0.0, 119: 0.0, 123: 0.0, 126: 0.0, 127: 0.0, 130: 0.0, 138: 0.0,
    221: 0.9947553216343291, 227: 0.9906953226187393, 239: 0.991636328549582,
    251: 0.9999999396472401, 252: 0.9992252850297415, 266: 0.0,
    308: 0.7, 316: 0.7, 320: 0.7, 324: 0.7, 328: 0.7, 344: 0.7, 378: 0.7,
    388: 0.7, 394: 0.85, 452: 0.7, 480: 0.7,
}
_SCENARIO = {
    "kind": "microgrid", "failure_type": "island", "max_failures": 2,
    "priority_assignment": "skewed", "slack_budget_pct": 0.3,
}


def test_headroom_scales_the_active_rating():
    net = GRIDS["simbench_lv_small"]()
    setpoints = {
        c.id: abs(float(c.model.p_mw))
        for c in net.childs
        if isinstance(c.model, PowerGenerator)
    }
    assert setpoints, "grid has no PowerGenerators to promote"

    apply_microgrid_islanding(net, promote_all_generators=True, grid_former_headroom=3.0)
    promoted = _promoted_ratings(net)
    checked = 0
    for cid, setpoint in setpoints.items():
        if cid in promoted and setpoint > 1e-6:
            assert promoted[cid] == pytest.approx(3.0 * setpoint)
            checked += 1
    assert checked, "no promoted former matched a generator setpoint"


def test_headroom_one_restores_setpoint_pinned_behaviour():
    net = GRIDS["simbench_lv_small"]()
    setpoints = {
        c.id: abs(float(c.model.p_mw))
        for c in net.childs
        if isinstance(c.model, PowerGenerator)
    }
    apply_microgrid_islanding(net, promote_all_generators=True, grid_former_headroom=1.0)
    for cid, rating in _promoted_ratings(net).items():
        if setpoints.get(cid, 0.0) > 1e-6:
            assert rating == pytest.approx(setpoints[cid])


def test_headroom_never_shrinks_the_rating():
    net = GRIDS["simbench_lv_small"]()
    apply_microgrid_islanding(net, promote_all_generators=True, grid_former_headroom=0.1)
    ref = GRIDS["simbench_lv_small"]()
    apply_microgrid_islanding(ref, promote_all_generators=True, grid_former_headroom=1.0)
    assert _promoted_ratings(net) == _promoted_ratings(ref)


def _replay(headroom):
    """Rebuild task 2044's captured LP state at a given former headroom."""
    from experiment.hpc.runner import _apply_scenario
    from mango_energy_environments.base.monee import create_physics_stepper

    net = GRIDS["simbench_lv"]()
    _apply_scenario(
        net,
        TaskSpec(
            task_id=2044, grid="simbench_lv", seed=200000004, n_failures=2,
            variant="component_level",
            scenario={**_SCENARIO, "grid_former_headroom": headroom},
        ),
        logging.getLogger("test"),
    )
    for bid in _INACTIVE:
        net.branch_by_id(bid).active = False
    for cid, value in _REGULATIONS.items():
        try:
            net.child_by_id(cid).model.regulation = value
        except Exception:  # noqa: BLE001 — snapshot may name a pruned child
            pass
    try:
        result = create_physics_stepper(net, solve_time_limit_s=300).step(_MIN_DT_H)
    except Exception:  # noqa: BLE001
        return False
    return not getattr(result, "failed", False)


@pytest.mark.slow
def test_captured_island_state_solves_at_the_default_headroom():
    assert _replay(4.0)


@pytest.mark.slow
def test_captured_island_state_is_infeasible_when_pinned_to_the_setpoint():
    """Pins the diagnosis: nothing else about the state changed."""
    assert not _replay(1.0)
