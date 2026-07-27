"""The oracle's DHS linearisation pins each water pipe to its as-built flow
direction, so a failure that removes a junction's inbound pipe leaves the
junction unsuppliable and the LP infeasible (eval_full_v2: 31 of 35 oracle
failures, IIS = ``mass_flow_neg_kgs [LB]`` vs that junction's mass balance).

``_build_min_shed_problem`` re-orients the pipes the post-failure topology
needs reversed; on a topology whose as-built directions still work it must
change nothing, or every already-solving task's model shifts with it.
"""

from __future__ import annotations

import monee.express as mx
import monee.model as mm
from monee.model.formulation.milp.heat import REVERSE_ATTR

from experiment.eval.oracle import _build_min_shed_problem


def _dhs_line(n_junctions: int = 4):
    """``ext -> j0 -> j1 -> ... -> jn``, every junction drawing mass, plus a
    tie from the feed to the far end so a cut is recoverable by reversal."""
    net = mm.Network()
    juncs = [mx.create_water_junction(net) for _ in range(n_junctions)]
    for a, b in zip(juncs[:-1], juncs[1:]):
        mx.create_water_pipe(
            net, from_node_id=a, to_node_id=b, diameter_m=0.15, length_m=100.0
        )
    mx.create_water_pipe(
        net,
        from_node_id=juncs[0],
        to_node_id=juncs[-1],
        diameter_m=0.15,
        length_m=100.0,
    )
    mx.create_water_ext_grid(net, juncs[0], t_k=360.0)
    for junc in juncs[1:]:
        mx.create_water_sink(net, junc, mass_flow_kgs=0.5)
    return net, juncs


def _reversed_ids(net):
    return {b.id for b in net.branches if getattr(b.model, REVERSE_ATTR, False)}


def test_intact_topology_keeps_every_pipe_as_built():
    net, _ = _dhs_line()

    _build_min_shed_problem(net, None)

    assert _reversed_ids(net) == set()


def test_cut_supply_pipe_reverses_the_pipes_below_it():
    net, juncs = _dhs_line()
    net.branch_by_id((juncs[0], juncs[1], 0)).active = False

    _build_min_shed_problem(net, None)

    assert _reversed_ids(net) == {
        (juncs[1], juncs[2], 0),
        (juncs[2], juncs[3], 0),
    }
