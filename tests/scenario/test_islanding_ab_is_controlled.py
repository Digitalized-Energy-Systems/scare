"""The microgrid arm must differ from its clean twin by islanding alone.

Two defects made ``extension_campaign_20260728-143126``'s islanding A/B
uninterpretable — the SCARE arm read -0.161 PWSF and the oracle arm read a
near-wash — and neither was islanding:

1. ``apply_slack_budget`` sizes the operator budget off a ``PowerGenerator`` /
   ``Source`` isinstance scan, but ``apply_microgrid_islanding`` runs first and
   promotes those children to ``GridForming*`` — which are NOT subclasses. All
   76 promoted units fell out of the budget basis, so the microgrid arm ran a
   38.9 % smaller electricity and 25.3 % smaller gas budget on a physically
   identical grid. Gas is budget-bound in both arms (the oracle draws exactly
   its budget), so this alone moved the headline.

2. ``monee.Network.__deepcopy__`` is an attribute whitelist that carried the
   islanding extension but not the ``net.islanding_config`` attribute
   ``enable_islanding`` also sets. Every solver returns
   ``prepare_solve_network``'s copy, so the oracle graded an islanded net as
   un-islanded: ``find_ignored_nodes`` fell back to ExtGrid-only leading and
   zeroed 47 loads whose ``oracle_regulations`` were 1.0 — exactly the load the
   extension had restored, worth +0.067 PWSF.
"""

import pytest
from monee.model.extension.islanding.gas import GasIslandingMode
from monee.solver.core import find_ignored_nodes

from experiment.eval.metrics import _disconnected_node_ids
from experiment.scenarios import GRIDS, apply_microgrid_islanding, apply_slack_budget
from scare.base.util import islanding_config_of

_CARRIERS = ("electricity", "water", "gas")
#: seed 0 of the campaign — three branch failures that sever 60 nodes when the
#: islanding config is not seen, 1 when it is.
_FAILED_BRANCHES = [(90, 82, 0), (52, 70, 0), (104, 43, 0)]


def _budgets(net) -> dict[str, tuple[float, float]]:
    """``{aid: (operator budget, LP envelope)}`` per slack child."""
    out: dict[str, tuple[float, float]] = {}
    for child in net.childs:
        m = child.model
        if hasattr(m, "_scare_slack_budget_mw"):
            out[f"child-{child.id}"] = (
                float(m._scare_slack_budget_mw),
                float(m.p_mw.max),
            )
        elif hasattr(m, "_scare_slack_budget_kgs"):
            out[f"child-{child.id}"] = (
                float(m._scare_slack_budget_kgs),
                float(-m.mass_flow_kgs.min),
            )
    return out


def _fail(net) -> None:
    for bid in _FAILED_BRANCHES:
        branch = net.branch_by_id(bid)
        branch.active = False
        branch.model.active = False


def test_promotion_leaves_the_slack_budget_untouched():
    clean = GRIDS["simbench_lv"]()
    apply_slack_budget(clean, 0.45)

    micro = GRIDS["simbench_lv"]()
    promoted = apply_microgrid_islanding(
        micro, carriers=_CARRIERS, promote_all_generators=True
    )
    apply_slack_budget(micro, 0.45)

    assert sum(promoted.values()) > 0, "nothing was promoted — test is vacuous"
    assert _budgets(micro) == _budgets(clean)


def test_budget_is_independent_of_grid_former_headroom():
    """Sizing off the promoted model's *rating* would confound the other way."""
    budgets = []
    for headroom in (1.0, 4.0, 10.0):
        net = GRIDS["simbench_lv"]()
        apply_microgrid_islanding(
            net,
            carriers=_CARRIERS,
            promote_all_generators=True,
            grid_former_headroom=headroom,
        )
        apply_slack_budget(net, 0.45)
        budgets.append(_budgets(net))
    assert budgets[0] == budgets[1] == budgets[2]


def test_a_promoted_gas_former_cannot_absorb_gas():
    """A ``Source`` injects; ``GridFormingSource`` bounds its balancing Var
    symmetrically, and nothing in the objective prices the positive half — so
    the LP parked the promoted fleet there as free sinks, drawing a net 0.0102
    kg/s (44 % of the gas slack budget) that the compliance wind-down paid for
    by shedding real load."""
    net = GRIDS["simbench_lv"]()
    apply_microgrid_islanding(net, carriers=_CARRIERS, promote_all_generators=True)

    formers = [
        c for c in net.childs if type(c.model).__name__ == "GridFormingSource"
    ]
    gas = [c for c in formers if "gas" in str(net.node_by_id(c.node_id).grid.name).lower()]
    assert gas, "nothing promoted on the gas grid — test is vacuous"
    for child in gas:
        var = child.model.mass_flow_kgs
        assert var.max == 0.0, f"child-{child.id} may absorb gas"
        # Injection headroom must survive: the former still anchors the island.
        assert var.min is not None and var.min < 0.0


def _gas_formers(net):
    return [
        c
        for c in net.childs
        if type(c.model).__name__ == "GridFormingSource"
        and "gas" in str(net.node_by_id(c.node_id).grid.name).lower()
    ]


def test_a_non_leading_gas_former_holds_its_pre_promotion_setpoint():
    """Where the ext grid leads the component ``stamp_gf_leadership`` marks
    EVERY former non-leading, ``overwrite()`` pins nothing and ``equations()``
    returned ``[]`` — leaving ``mass_flow_kgs`` in the node balance alone, pinned
    by no equation and priced by no objective (plain energy flow carries none).
    The LP returned an arbitrary degenerate split: the fleet delivered 0.0013 of
    its 0.0118 kg/s while the slack ran to its budget and the rest became shed
    gas."""
    net = GRIDS["simbench_lv"]()
    apply_microgrid_islanding(net, carriers=_CARRIERS, promote_all_generators=True)
    GasIslandingMode().stamp_gf_leadership(net)

    formers = _gas_formers(net)
    assert formers, "nothing promoted on the gas grid — test is vacuous"
    # The whole point: with a slack in the component, none of them leads.
    assert not any(m.model._gf_leading for m in formers)
    for child in formers:
        assert child.model.equations(None, None), f"child-{child.id} left unpinned"


def test_a_LEADING_gas_former_keeps_its_free_balancing_var():
    """The pin must lift the moment the unit actually becomes the island
    reference — that Var is what absorbs the island's imbalance."""
    net = GRIDS["simbench_lv"]()
    apply_microgrid_islanding(net, carriers=_CARRIERS, promote_all_generators=True)
    former = _gas_formers(net)[0].model
    former._gf_leading = True
    assert former.equations(None, None) == []


def test_promotion_preserves_each_source_injection_headroom():
    """The former's injection bound is the magnitude of the ``Source`` it
    replaced, so the clean arm's supply is still physically reachable."""
    clean = GRIDS["simbench_lv"]()
    pre = {}
    for child in clean.childs:
        grid = str(clean.node_by_id(child.node_id).grid.name).lower()
        if "gas" in grid and type(child.model).__name__ == "Source":
            pre[child.id] = abs(float(child.model.mass_flow_kgs))

    micro = GRIDS["simbench_lv"]()
    apply_microgrid_islanding(micro, carriers=_CARRIERS, promote_all_generators=True)
    post = {
        c.id: abs(float(c.model.mass_flow_kgs.min))
        for c in micro.childs
        if type(c.model).__name__ == "GridFormingSource"
        and "gas" in str(micro.node_by_id(c.node_id).grid.name).lower()
    }
    assert pre and post.keys() == pre.keys()
    assert sum(post.values()) == pytest.approx(sum(pre.values()), rel=1e-9)


def test_islanding_config_survives_the_copy_every_solver_returns():
    net = GRIDS["simbench_lv"]()
    apply_microgrid_islanding(net, carriers=_CARRIERS, promote_all_generators=True)
    _fail(net)

    solved = net.copy()  # what ``result.network`` is
    assert islanding_config_of(solved) is not None
    assert len(find_ignored_nodes(solved, islanding_config_of(solved))) == len(
        find_ignored_nodes(net, islanding_config_of(net))
    )


def test_grading_a_solver_returned_net_keeps_the_islands_energised():
    net = GRIDS["simbench_lv"]()
    apply_microgrid_islanding(net, carriers=_CARRIERS, promote_all_generators=True)
    _fail(net)

    live = _disconnected_node_ids(net)
    assert _disconnected_node_ids(net.copy()) == live

    # Guard against a vacuous pass: without the anchors the same failures do
    # strand a large subtree, so the assertion above has something to protect.
    bare = GRIDS["simbench_lv"]()
    _fail(bare)
    assert len(_disconnected_node_ids(bare)) > len(live)
