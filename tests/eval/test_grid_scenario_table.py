"""Tests for the grid-scenario table generator
(:mod:`experiment.eval.grid_scenario_table`).

Uses ``build=False`` throughout so the grid factories are only introspected
(closure params), never built — fast and offline. Node / CP counts therefore
render as ``---``.
"""

from __future__ import annotations

from experiment.eval.grid_scenario_table import (
    _slack_label,
    _tex_escape,
    collect_grid_scenarios,
    render_latex,
)
from experiment.hpc.config import CampaignConfig


def _cfg(experiments: list[dict]) -> CampaignConfig:
    return CampaignConfig.from_dict(
        {
            "name": "t",
            "out_root": "x",
            "experiments": experiments,
        }
    )


class TestSlackLabel:
    def test_percent_is_plain_text(self):
        # Escaping is the renderer's job — the label itself is plain.
        assert _slack_label(0.45) == "45%"
        assert _slack_label(0.3) == "30%"
        assert _slack_label(None) == "∞"

    def test_render_escapes_percent_exactly_once(self):
        cfg = _cfg(
            [
                {
                    "name": "e",
                    "grids": ["simbench_lv"],
                    "scenarios": [{"kind": "clean", "slack_budget_pct": 0.45}],
                }
            ]
        )
        tex = render_latex(collect_grid_scenarios(cfg, build=False))
        assert r"45\%" in tex
        assert r"45\\%" not in tex

    def test_unbudgeted_renders_as_infty(self):
        cfg = _cfg(
            [
                {
                    "name": "pv",
                    "grids": ["simbench_lv"],
                    "scenarios": [{"kind": "pv_peak"}],  # no slack_budget_pct
                }
            ]
        )
        tex = render_latex(collect_grid_scenarios(cfg, build=False))
        assert r"$\infty$" in tex
        assert "unbudgeted" not in tex


class TestCollect:
    def test_collapses_slack_into_one_row_per_grid(self):
        # One grid under several slack budgets collapses to ONE row
        # carrying every distinct budget.
        cfg = _cfg(
            [
                {
                    "name": "a",
                    "grids": ["simbench_lv"],
                    "scenarios": [{"kind": "clean", "slack_budget_pct": 0.45}],
                },
                {
                    "name": "b",
                    "grids": ["simbench_lv"],
                    "scenarios": [
                        {"kind": "clean", "slack_budget_pct": 0.45},  # dup
                        {"kind": "clean", "slack_budget_pct": 0.30},  # new
                        {"kind": "pv_peak"},  # unbudgeted
                    ],
                },
            ]
        )
        scen = collect_grid_scenarios(cfg, build=False)
        assert [s.grid_name for s in scen] == ["simbench_lv"]
        assert [s.scenario_id for s in scen] == ["S1"]
        # Distinct budgets preserved in first-appearance order, ∞ for none.
        assert scen[0].slack_budgets == [0.45, 0.30, None]

    def test_distinct_grids_get_canonical_ids(self):
        # IDs are pinned to the dissertation's tab:grid-scenarios, not
        # first-appearance order: reconfig is S4 even when listed second.
        cfg = _cfg(
            [
                {
                    "name": "e",
                    "grids": ["simbench_lv", "simbench_lv_reconfig"],
                    "scenarios": [{"kind": "clean", "slack_budget_pct": 0.45}],
                }
            ]
        )
        scen = collect_grid_scenarios(cfg, build=False)
        assert [s.grid_name for s in scen] == [
            "simbench_lv",
            "simbench_lv_reconfig",
        ]
        assert [s.scenario_id for s in scen] == ["S1", "S4"]

    def test_canonical_order_beats_appearance_order(self):
        cfg = _cfg(
            [
                {
                    "name": "e",
                    "grids": ["simbench_lv_small", "simbench_lv"],
                    "scenarios": [{"kind": "clean", "slack_budget_pct": 0.45}],
                }
            ]
        )
        scen = collect_grid_scenarios(cfg, build=False)
        assert [s.scenario_id for s in scen] == ["S1", "S5"]
        assert [s.grid_name for s in scen] == [
            "simbench_lv",
            "simbench_lv_small",
        ]

    def test_missing_slack_is_none(self):
        cfg = _cfg(
            [
                {
                    "name": "pv",
                    "grids": ["simbench_lv"],
                    "scenarios": [{"kind": "pv_peak"}],  # no slack_budget_pct
                }
            ]
        )
        (scen,) = collect_grid_scenarios(cfg, build=False)
        assert scen.slack_budgets == [None]
        # Name no longer carries the slack — that's its own column now.
        assert "∞" not in scen.name

    def test_unknown_grid_skipped(self):
        cfg = _cfg(
            [
                {
                    "name": "x",
                    "grids": ["does_not_exist"],
                    "scenarios": [{"kind": "clean", "slack_budget_pct": 0.45}],
                }
            ]
        )
        assert collect_grid_scenarios(cfg, build=False) == []

    def test_facts_from_closure_without_build(self):
        # reconfig grid: backup detected purely from the factory closure.
        cfg = _cfg(
            [
                {
                    "name": "r",
                    "grids": ["simbench_lv_reconfig"],
                    "scenarios": [{"kind": "clean", "slack_budget_pct": 0.45}],
                }
            ]
        )
        (scen,) = collect_grid_scenarios(cfg, build=False)
        assert scen.facts.has_backup is True
        assert scen.facts.backup_lines_per_sector == 5
        assert scen.facts.simbench_code == "1-LV-rural3--1-no_sw"
        # No build requested → counts are unknown.
        assert scen.facts.n_nodes is None
        assert scen.facts.n_cps is None


class TestRender:
    def test_latex_is_well_formed(self):
        cfg = _cfg(
            [
                {
                    "name": "e",
                    "grids": ["simbench_lv", "simbench_lv_reconfig"],
                    "scenarios": [{"kind": "clean", "slack_budget_pct": 0.45}],
                }
            ]
        )
        tex = render_latex(collect_grid_scenarios(cfg, build=False))
        assert tex.count(r"\begin{table}") == 1
        assert tex.count(r"\end{table}") == 1
        assert r"\toprule" in tex and r"\bottomrule" in tex
        # One header row + two body rows, each terminated by ``\\``.
        assert tex.count(r"\\") == 3
        assert r"5/sector" in tex  # reconfig backup column

    def test_tex_escape(self):
        assert _tex_escape("a_b") == r"a\_b"
        assert _tex_escape("50%") == r"50\%"
