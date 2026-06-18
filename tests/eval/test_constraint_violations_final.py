"""Tests for the end-of-sim hard-bound feasibility scan in
:mod:`experiment.eval.metrics`.

``constraint_violations_final`` backs the ``constraint_compliance`` claim.
Covers the violation classifier in isolation and an end-to-end scan over a
real monee grid with an injected breach.
"""

from __future__ import annotations

import pytest

from experiment.eval.metrics import (
    _bound_overshoot,
    _violation_row,
    constraint_rows,
    constraint_violations_final,
)
from scare.base.model import Sector

# ---------------------------------------------------------------------------
# Classifier logic
# ---------------------------------------------------------------------------


class TestViolationClassifier:
    def test_voltage_over_bound_is_a_violation(self):
        r = _violation_row("node", 5, Sector.ELECTRICITY, "vm_pu", 1.08, 0.95, 1.05)
        assert r["violated"] is True
        assert r["overshoot"] == pytest.approx(0.6)

    def test_numerical_wiggle_at_bound_does_not_fire(self):
        # 1.053 is over 1.05 but within the 0.005 p.u. tolerance.
        r = _violation_row("node", 5, Sector.ELECTRICITY, "vm_pu", 1.053, 0.95, 1.05)
        assert r["violated"] is False

    def test_loading_is_one_sided(self):
        over = _violation_row(
            "branch",
            (1, 2),
            Sector.ELECTRICITY,
            "loading_percent",
            114.0,
            -100.0,
            100.0,
        )
        assert over["violated"] is True
        assert over["overshoot"] == pytest.approx(0.14)
        # The lower bound is a formula artefact — a "negative loading" must
        # never be flagged.
        under = _violation_row(
            "branch",
            (1, 2),
            Sector.ELECTRICITY,
            "loading_percent",
            -150.0,
            -100.0,
            100.0,
        )
        assert under["violated"] is False
        assert under["overshoot"] == 0.0

    def test_cold_heat_node_is_a_violation(self):
        r = _violation_row("node", 5, Sector.HEAT, "t_k", 240.0, 313.15, 403.15)
        assert r["violated"] is True
        assert r["overshoot"] > 1.0  # deeply cold

    def test_overshoot_uncapped_unlike_utilization(self):
        # _bound_overshoot grows past 1.0 so the worst breach can be ranked.
        assert _bound_overshoot(1.20, 0.95, 1.05, one_sided=False) == pytest.approx(3.0)
        assert _bound_overshoot(1.00, 0.95, 1.05, one_sided=False) == 0.0


# ---------------------------------------------------------------------------
# End-to-end scan over a real grid
# ---------------------------------------------------------------------------


def _build_grid():
    from experiment.scenarios import GRIDS

    name = "simbench_lv_low" if "simbench_lv_low" in GRIDS else next(iter(GRIDS))
    return GRIDS[name]()


class TestScanOnRealGrid:
    def test_clean_grid_passes(self):
        net = _build_grid()
        cv = constraint_violations_final(net)
        assert cv["passed"] is True
        assert cv["n_violations"] == 0
        assert cv["n_checked"] > 0

    def test_injected_overvoltage_is_flagged(self):
        net = _build_grid()
        bus = next(n for n in net.nodes if type(n.model).__name__ == "Bus")
        bus.model.vm_pu = 1.20  # gross overvoltage

        cv = constraint_violations_final(net)
        assert cv["passed"] is False
        assert cv["n_violations"] >= 1
        assert cv["by_sector"]["electricity"]["n_violations"] >= 1
        # Per-variable-type tally: the overvoltage lands in the gating
        # ``voltage`` bucket, separate from electricity's ``line_load`` bucket.
        assert cv["by_variable"]["voltage"]["n_violations"] >= 1
        assert cv["by_variable"]["voltage"]["gating"] is True
        worst = cv["violations"][0]
        assert worst["variable"] == "vm_pu"
        assert worst["value"] == pytest.approx(1.20)

    def test_by_variable_separates_voltage_from_line_loading(self):
        # Electricity carries two gating variables; the per-sector count
        # conflates them but ``by_variable`` keeps them apart.
        net = _build_grid()
        cv = constraint_violations_final(net)
        by_var = cv["by_variable"]
        # A clean LV grid still *checks* voltage and line loading separately.
        assert "voltage" in by_var
        assert by_var["voltage"]["n_checked"] > 0
        if "temperature" in by_var:
            assert by_var["temperature"]["gating"] is False

    def _gas_junction(self, net):
        from experiment.eval.metrics import sector_from_grid

        return next(
            n
            for n in net.nodes
            if type(n.model).__name__ == "Junction"
            and sector_from_grid(getattr(n, "grid", None)) == Sector.GAS
        )

    def test_deenergised_gas_junction_is_not_flagged(self):
        # A gas region cut off from its ExtHydrGrid stays graph-connected via
        # the supply/return loop (so it isn't an "ignored" node), yet the LP
        # collapses its pressure to ~0. That is de-energisation, not an
        # actionable under-pressure breach — it must drop out of the scan.
        net = _build_grid()
        node = self._gas_junction(net)
        node.model.pressure_pu = 0.0

        rows = constraint_rows(net)
        assert all(
            r["id"] != node.id or r["variable"] != "pressure_pu" for r in rows
        )

    def test_genuine_under_pressure_still_gates(self):
        # A junction that is pressurised but below the 0.85 floor is a real
        # breach (well above the de-energised floor) and must still gate.
        net = _build_grid()
        node = self._gas_junction(net)
        node.model.pressure_pu = 0.70

        cv = constraint_violations_final(net)
        assert cv["passed"] is False
        assert cv["by_sector"]["gas"]["n_violations"] >= 1
        flagged = [
            r
            for r in constraint_rows(net)
            if r["id"] == node.id and r["variable"] == "pressure_pu"
        ]
        assert flagged and flagged[0]["violated"] is True

    def test_disconnected_node_is_not_scanned(self):
        # A deactivated node must drop out of the scan entirely — its loads
        # already count as served=0 and the oracle excludes it too, so a
        # garbage reading there must not be flagged as a breach.
        net = _build_grid()
        node = net.nodes[0]
        node.active = False
        rows = constraint_rows(net)
        assert all(r["id"] != node.id or r["kind"] != "node" for r in rows)
