"""Tests for the disconnected/stranded handling in
:func:`experiment.eval.claims._check_priority_invariant`.

Loads with ``disconnected=1`` (no path to a grid-forming source) are
physically unservable regardless of priority and must be dropped from
the per-tier aggregation; otherwise a stranded high-tier load drags its
tier's aggregate fraction down and fakes an inversion against lower tiers.
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
    # tier-3 has one stranded load plus two served; tier-4 fully served.
    # The stranded load must not drag tier-3's aggregate below tier-4's.
    rows = [
        {"aid": "a", "sector": "electricity", "tier": 3, "node_id": 1, "component": 0,
         "demand": 5.0, "served": 5.0, "fraction": 1.0, "disconnected": 0},
        {"aid": "b", "sector": "electricity", "tier": 3, "node_id": 2, "component": 0,
         "demand": 5.0, "served": 5.0, "fraction": 1.0, "disconnected": 0},
        # tier-3, stranded (no source path)
        {"aid": "c", "sector": "electricity", "tier": 3, "node_id": 3, "component": 0,
         "demand": 6.0, "served": 0.0, "fraction": 0.0, "disconnected": 1},
        {"aid": "d", "sector": "electricity", "tier": 4, "node_id": 4, "component": 0,
         "demand": 5.0, "served": 5.0, "fraction": 1.0, "disconnected": 0},
        {"aid": "e", "sector": "electricity", "tier": 4, "node_id": 5, "component": 0,
         "demand": 5.0, "served": 5.0, "fraction": 1.0, "disconnected": 0},
    ]
    res = _check_priority_invariant(_write_served_by_load(tmp_path, rows))
    assert res["passed"] is True
    assert res["detail"]["n_inversions"] == 0
    assert res["detail"]["n_loads_stranded"] == 1
    assert abs(res["detail"]["stranded_demand_mw"] - 6.0) < 1e-9


def test_genuine_inversion_still_detected(tmp_path):
    # No stranded load: tier-3 shed while tier-4 fully served is a real
    # inversion the check must still flag.
    rows = [
        {"aid": "a", "sector": "electricity", "tier": 3, "node_id": 1, "component": 0,
         "demand": 10.0, "served": 3.0, "fraction": 0.3, "disconnected": 0},
        {"aid": "d", "sector": "electricity", "tier": 4, "node_id": 4, "component": 0,
         "demand": 10.0, "served": 10.0, "fraction": 1.0, "disconnected": 0},
    ]
    res = _check_priority_invariant(_write_served_by_load(tmp_path, rows))
    assert res["passed"] is False
    assert res["detail"]["n_inversions"] == 1
    assert res["detail"]["n_loads_stranded"] == 0


def test_all_loads_stranded_in_a_component_skips_silently(tmp_path):
    # An all-stranded component drops out of the aggregation entirely:
    # passes with nothing to check, loads counted as stranded.
    rows = [
        {"aid": "a", "sector": "electricity", "tier": 3, "node_id": 1, "component": 0,
         "demand": 5.0, "served": 0.0, "fraction": 0.0, "disconnected": 1},
        {"aid": "b", "sector": "electricity", "tier": 4, "node_id": 2, "component": 0,
         "demand": 5.0, "served": 0.0, "fraction": 0.0, "disconnected": 1},
    ]
    res = _check_priority_invariant(_write_served_by_load(tmp_path, rows))
    assert res["passed"] is True
    assert res["detail"]["n_inversions"] == 0
    assert res["detail"]["n_loads_stranded"] == 2
    assert res["detail"]["n_components_checked"] == 0


def test_disconnected_field_missing_treated_as_not_stranded(tmp_path):
    # Missing ``disconnected`` column → no crash, falls back to "not stranded".
    rows = [
        {"aid": "a", "sector": "electricity", "tier": 3, "node_id": 1, "component": 0,
         "demand": 10.0, "served": 3.0, "fraction": 0.3},
        {"aid": "d", "sector": "electricity", "tier": 4, "node_id": 4, "component": 0,
         "demand": 10.0, "served": 10.0, "fraction": 1.0},
    ]
    res = _check_priority_invariant(_write_served_by_load(tmp_path, rows))
    assert res["passed"] is False
    assert res["detail"]["n_inversions"] == 1
    assert res["detail"]["n_loads_stranded"] == 0
