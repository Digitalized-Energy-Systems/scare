"""Tests for the display-alias layer (:mod:`experiment.eval.aliases`)."""

from __future__ import annotations

from experiment.eval.aliases import (
    alias_ablation,
    alias_experiment,
    alias_grid,
    alias_scenario,
    alias_sweep,
    alias_variant,
)


class TestTableAliases:
    def test_grid_carries_dissertation_id(self):
        assert alias_grid("simbench_lv") == "S1 · LV-L"
        assert alias_grid("simbench_lv_small") == "S5 · LV-S"
        assert alias_grid("simbench_mvlv") == "S8 · MV–LV"

    def test_variant_names(self):
        assert alias_variant("scare") == "SCARE"
        assert alias_variant("oracle") == "Oracle"
        assert alias_variant("single_level") == "Single-level"
        assert alias_variant("component_level") == "Component-level"

    def test_experiment_names(self):
        assert alias_experiment("functional_baseline") == "Functional baseline"
        assert alias_experiment("cold_day_stress") == "Cold-day stress"

    def test_names_no_longer_than_keys(self):
        from experiment.eval.aliases import _load

        tables = _load()
        for section in ("grids", "experiments", "variants", "ablation_flags"):
            for key, name in tables[section].items():
                assert len(name) <= len(key), f"{section}: {key!r} -> {name!r}"

    def test_unknown_passes_through(self):
        assert alias_grid("no_such_grid") == "no_such_grid"
        assert alias_experiment("no_such_experiment") == "no_such_experiment"
        assert alias_variant("no_such_variant") == "no_such_variant"


class TestAblationAlias:
    def test_default_is_full_system(self):
        assert alias_ablation("default") == "full system"
        assert alias_ablation("") == "full system"
        assert alias_ablation(None) == "full system"

    def test_disabled_flag_reads_as_removal(self):
        assert alias_ablation("enable_holonic=False") == "no holonic layer"
        assert (
            alias_ablation("enable_holonic=False;enable_cp_admm=False")
            == "no holonic layer, no CP-ADMM"
        )

    def test_enabled_flag_reads_as_armed(self):
        assert (
            alias_ablation("enable_cross_sector_coalitions=True")
            == "with cross-sector coalitions"
        )

    def test_parameter_renders_as_assignment(self):
        assert alias_ablation("holon_admm_scope=sector") == "holon-ADMM scope = sector"

    def test_unknown_flag_uses_heuristic(self):
        assert alias_ablation("enable_cp_qp_thing=False") == "no CP QP thing"


class TestSweepAlias:
    def test_default_stays_default(self):
        assert alias_sweep("default") == "default"

    def test_value_rendered(self):
        assert alias_sweep("slack_target_fraction=0.5") == "slack target fraction = 0.5"


class TestScenarioAlias:
    def test_default(self):
        assert alias_scenario("default") == "default"
        assert alias_scenario(None) == "default"

    def test_clean_with_failures(self):
        key = (
            "kind=clean;max_failures=3;priority_assignment=skewed;slack_budget_pct=0.45"
        )
        assert (
            alias_scenario(key) == "random failures ×3 · skewed priorities · slack 45%"
        )

    def test_concentrated(self):
        assert (
            alias_scenario("kind=clean;failure_type=concentrated;n_failures=5")
            == "concentrated ×5"
        )

    def test_mixed_share(self):
        assert (
            alias_scenario("kind=clean;failure_type=mixed;generator_share=0.5")
            == "mixed (50% generators)"
        )

    def test_cold_day(self):
        assert (
            alias_scenario("kind=cold_day;heat_load_scale=1.5")
            == "cold-day (heat ×1.5)"
        )

    def test_unknown_keys_survive(self):
        # Distinct scenarios must never collapse to the same label.
        a = alias_scenario("kind=clean;max_failures=3;linepack=True")
        b = alias_scenario("kind=clean;max_failures=3")
        assert a != b
        assert "linepack=True" in a

    def test_dict_input(self):
        assert (
            alias_scenario(
                {"kind": "clean", "failure_type": "generator", "max_failures": 3}
            )
            == "generator outage ×3"
        )
