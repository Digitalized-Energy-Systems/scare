"""A dropped oracle task must persist WHICH constraint was infeasible.

``repr(GurobiIISReport)`` renders only counts. eval_full_v2 therefore recorded
``GurobiIISReport(constraints=1, bounds=1)`` for 29 of its 35 oracle failures,
and the member names lived only in each task's ``run.log`` — so the fact that
all 29 share a single structural conflict was invisible from ``summary.csv``.
Naming them collapses those 29 into one signature:
``branch_<A>_<B>_0__mass_flow_neg_kgs [LB]`` versus ``node_<A>_eq_35``.
"""

from __future__ import annotations

from experiment.eval.oracle import _IIS_NAMES_IN_REASON, _iis_signature


class _Report:
    """Stands in for ``monee.solver.gurobipy.GurobiIISReport``."""

    def __init__(self, constraints, bounds):
        self.constraints = list(constraints)
        self.bounds = list(bounds)

    def __repr__(self):
        return (
            f"GurobiIISReport(constraints={len(self.constraints)}, "
            f"bounds={len(self.bounds)})"
        )


def test_the_campaigns_dominant_signature_is_named():
    r = _Report(["node_378_eq_35"], ["branch_378_297_0__mass_flow_neg_kgs [LB]"])
    out = _iis_signature(r)
    assert "node_378_eq_35" in out
    assert "branch_378_297_0__mass_flow_neg_kgs [LB]" in out
    assert "bounds(1)" in out and "constraints(1)" in out


def test_two_failures_with_the_same_structure_share_a_groupable_prefix():
    a = _iis_signature(
        _Report(["node_362_eq_35"], ["branch_362_275_0__mass_flow_neg_kgs [LB]"])
    )
    b = _iis_signature(
        _Report(["node_326_eq_35"], ["branch_326_316_0__mass_flow_neg_kgs [LB]"])
    )
    assert a != b, "distinct branches must stay distinguishable"
    assert a.count("mass_flow_neg_kgs") == b.count("mass_flow_neg_kgs") == 1


def test_empty_iis_falls_back_to_repr():
    """An empty IIS means unbounded rather than infeasible; keep the old text."""
    r = _Report([], [])
    assert _iis_signature(r) == repr(r)


def test_report_without_the_attributes_falls_back_to_repr():
    class _Other:
        def __repr__(self):
            return "PyomoReport(...)"

    assert _iis_signature(_Other()) == "PyomoReport(...)"


def test_long_iis_is_capped_and_says_how_many_were_elided():
    n = _IIS_NAMES_IN_REASON + 3
    r = _Report([f"c{i}" for i in range(n)], [])
    out = _iis_signature(r)
    assert f"constraints({n})" in out
    assert "+3 more" in out
    assert f"c{n - 1}" not in out


def test_single_line_so_it_survives_a_csv_cell():
    r = _Report(["node_1_eq_35", "node_2_eq_35"], ["b [LB]", "c [UB]"])
    assert "\n" not in _iis_signature(r)
