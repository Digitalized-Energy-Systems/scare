"""Tests for the disconnected/stranded handling in
:func:`experiment.eval.claims._check_priority_invariant`.

Loads with ``disconnected=1`` come from monee's ``find_ignored_nodes`` —
nodes with no path to a grid-forming source through the active branch
topology.  They are physically unservable regardless of priority and
must not contribute to the per-tier aggregation, otherwise a single
stranded high-tier load drags down its tier's aggregate fraction and
produces a spurious inversion against the still-served lower tiers.
"""

from __future__ import annotations

import csv
from pathlib import Path

from experiment.eval.claims import _check_priority_invariant


def _write_served_by_load(tmp_path: Path, rows: list[dict]) -> Path:
    out = tmp_path / "served_by_load.csv"
    cols = (
        "aid", "sector", "tier", "node_id", "component",
        "demand", "served", "fraction", "disconnected",
    )
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    return out


def test_stranded_load_excluded_from_per_tier_aggregation(tmp_path):
    # tier-6 has one stranded load (frac would be 0) and two served
    # loads (frac=1.0).  tier-7 has two served loads.  Without the
    # fix the stranded load drags tier-6's aggregate fraction below
    # tier-7's, producing a spurious inversion.
    rows = [
        # tier-6, served
        {"aid": "a", "sector": "heat", "tier": 6, "node_id": 1, "component": 0,
         "demand": 5.0, "served": 5.0, "fraction": 1.0, "disconnected": 0},
        {"aid": "b", "sector": "heat", "tier": 6, "node_id": 2, "component": 0,
         "demand": 5.0, "served": 5.0, "fraction": 1.0, "disconnected": 0},
        # tier-6, STRANDED (no source path)
        {"aid": "c", "sector": "heat", "tier": 6, "node_id": 3, "component": 0,
         "demand": 6.0, "served": 0.0, "fraction": 0.0, "disconnected": 1},
        # tier-7, served
        {"aid": "d", "sector": "heat", "tier": 7, "node_id": 4, "component": 0,
         "demand": 5.0, "served": 5.0, "fraction": 1.0, "disconnected": 0},
        {"aid": "e", "sector": "heat", "tier": 7, "node_id": 5, "component": 0,
         "demand": 5.0, "served": 5.0, "fraction": 1.0, "disconnected": 0},
    ]
    res = _check_priority_invariant(_write_served_by_load(tmp_path, rows))
    assert res["passed"] is True
    assert res["detail"]["n_inversions"] == 0
    assert res["detail"]["n_loads_stranded"] == 1
    assert abs(res["detail"]["stranded_demand_mw"] - 6.0) < 1e-9


def test_genuine_inversion_still_detected(tmp_path):
    # Same shape but no stranded load — instead tier-6 is shed (by a
    # SCARE decision) and tier-7 is fully served.  This is a *real*
    # inversion the check must still flag.
    rows = [
        {"aid": "a", "sector": "heat", "tier": 6, "node_id": 1, "component": 0,
         "demand": 10.0, "served": 3.0, "fraction": 0.3, "disconnected": 0},
        {"aid": "d", "sector": "heat", "tier": 7, "node_id": 4, "component": 0,
         "demand": 10.0, "served": 10.0, "fraction": 1.0, "disconnected": 0},
    ]
    res = _check_priority_invariant(_write_served_by_load(tmp_path, rows))
    assert res["passed"] is False
    assert res["detail"]["n_inversions"] == 1
    assert res["detail"]["n_loads_stranded"] == 0


def test_all_loads_stranded_in_a_component_skips_silently(tmp_path):
    # When every load in a component is stranded the component
    # disappears from the per-(sector, component) aggregation entirely.
    # Result is "passed, nothing to check, N loads accounted for as
    # stranded" — no false-positive inversion, no false-positive pass
    # that hides system loss.
    rows = [
        {"aid": "a", "sector": "heat", "tier": 6, "node_id": 1, "component": 0,
         "demand": 5.0, "served": 0.0, "fraction": 0.0, "disconnected": 1},
        {"aid": "b", "sector": "heat", "tier": 7, "node_id": 2, "component": 0,
         "demand": 5.0, "served": 0.0, "fraction": 0.0, "disconnected": 1},
    ]
    res = _check_priority_invariant(_write_served_by_load(tmp_path, rows))
    assert res["passed"] is True
    assert res["detail"]["n_inversions"] == 0
    assert res["detail"]["n_loads_stranded"] == 2
    assert res["detail"]["n_components_checked"] == 0


def test_disconnected_field_missing_treated_as_not_stranded(tmp_path):
    # Older artefacts may lack the ``disconnected`` column; the check
    # must not crash and must fall back to "not stranded".
    rows = [
        {"aid": "a", "sector": "heat", "tier": 6, "node_id": 1, "component": 0,
         "demand": 10.0, "served": 3.0, "fraction": 0.3},
        {"aid": "d", "sector": "heat", "tier": 7, "node_id": 4, "component": 0,
         "demand": 10.0, "served": 10.0, "fraction": 1.0},
    ]
    res = _check_priority_invariant(_write_served_by_load(tmp_path, rows))
    assert res["passed"] is False
    assert res["detail"]["n_inversions"] == 1
    assert res["detail"]["n_loads_stranded"] == 0
