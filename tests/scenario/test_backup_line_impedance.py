"""Backup ties must be electrically like the lines they back up.

``add_backup_lines`` sized electricity ties from the sector median of
``length_m`` / ``r_ohm_per_m`` / ``x_ohm_per_m``. Those attributes live on
:class:`monee.model.branch.PowerLine`, but simbench arrives via
``from_pandapower_net`` -> MATPOWER as ``GenericPowerBranch``, which carries
``br_r_pu`` / ``br_x_pu`` instead — so **0 of 128** electricity branches
exposed them and ``_median_attr``'s ``1.0`` fallback won, over the sane
defaults in ``_create_backup_branch``. Every tie was built at 1.0 ohm/m over
1.0 m: ``r_pu = 1.0 / (0.4**2 / 1) = 6.25``, some 400x the median line.

Closing one then could not carry load without pushing buses under the 0.5 pu
floor, and the physics LP went infeasible the moment the reconfigurator
switched a tie in — 41 tasks in ``eval_full_v2_20260724-141520``, with an IIS
spanning the whole network minus the slack bus.
"""

from statistics import median

import pytest
from monee.model.node import Bus

from experiment.scenarios import GRIDS

GRID = "simbench_lv_reconfig"
CUT = (7, 106, 0)
TIE = (49, 114, 0)
_MIN_DT_H = 1e-9


def _el_branches(net):
    return [
        b
        for b in net.branches
        if isinstance(net.node_by_id(b.from_node_id).model, Bus)
        and isinstance(net.node_by_id(b.to_node_id).model, Bus)
    ]


def _r_pu(net, branch) -> float | None:
    """Per-unit resistance, however this branch stores it.

    ``PowerLine`` (what the backup ties are) seeds ``br_r_pu`` at 0 and only
    fills it in ``equations()``, so reading the attribute on a freshly built
    net silently reports a short circuit; derive it instead.
    """
    model = branch.model
    if hasattr(model, "calc_r_x"):
        r, _ = model.calc_r_x(
            branch.grid,
            net.node_by_id(branch.from_node_id).model,
            net.node_by_id(branch.to_node_id).model,
        )
        return float(r)
    try:
        return float(model.br_r_pu)
    except (AttributeError, TypeError, ValueError):
        return None


def _pu(net):
    """(backup, native) br_r_pu lists."""
    backup, native = [], []
    for b in _el_branches(net):
        r = _r_pu(net, b)
        if r is None or r <= 0.0:
            continue
        (backup if getattr(b.model, "backup", False) else native).append(r)
    return backup, native


def _step0_solves(net) -> bool:
    from mango_energy_environments.base.monee import create_physics_stepper

    try:
        result = create_physics_stepper(net, solve_time_limit_s=120).step(_MIN_DT_H)
    except Exception:  # noqa: BLE001 — a raising solve is a failure too
        return False
    return not getattr(result, "failed", False)


@pytest.mark.slow
def test_backup_ties_match_the_grid_they_back_up():
    net = GRIDS[GRID]()
    backup, native = _pu(net)
    assert backup, "no electricity backup ties were built"
    ref = median(native)
    # Was 424x before the fix; a tie should be a plausible line, not a
    # near-open circuit. Generous bound — the point is the order of magnitude.
    for r in backup:
        assert r <= 10 * ref, f"backup tie r_pu={r:.4f} vs median native {ref:.5f}"


@pytest.mark.slow
def test_no_backup_tie_dominates_the_impedance_range():
    """Before the fix the grid's max line resistance *was* a backup tie."""
    net = GRIDS[GRID]()
    backup, native = _pu(net)
    assert max(backup) <= max(native)


@pytest.mark.slow
def test_failure_plus_tie_closure_solves():
    """The exact reconfigurator sequence that went infeasible in eval_full_v2."""
    net = GRIDS[GRID]()
    net.branch_by_id(CUT).active = False
    net.branch_by_id(TIE).model.on_off = 1
    assert _step0_solves(net)


@pytest.mark.slow
@pytest.mark.parametrize("mutate", ["failure_only", "tie_only"], ids=str)
def test_each_action_alone_still_solves(mutate):
    net = GRIDS[GRID]()
    if mutate == "failure_only":
        net.branch_by_id(CUT).active = False
    else:
        net.branch_by_id(TIE).model.on_off = 1
    assert _step0_solves(net)


@pytest.mark.slow
def test_gas_and_heat_ties_keep_their_native_median():
    """Only electricity lacked the geometric attributes; don't regress the
    sectors whose median was real all along."""
    net = GRIDS[GRID]()
    for sector in ("gas", "water"):
        ids = {
            n.id
            for n in net.nodes
            if sector in str(getattr(n.grid, "name", "") or "").lower()
        }
        pipes = [b for b in net.branches if b.id[0] in ids and b.id[1] in ids]
        backups = [b for b in pipes if getattr(b.model, "backup", False)]
        others = [b for b in pipes if not getattr(b.model, "backup", False)]
        assert backups, f"no {sector} backup ties"
        want = median([float(b.model.length_m) for b in others])
        for b in backups:
            assert float(b.model.length_m) == pytest.approx(want)
