"""Smoke tests for the CP-focused plotting helpers.

These exercise the plumbing (JSON parsing, figure construction, file
writing) on a synthetic event ledger, not the visual output. Rendering
needs no headless browser, so the tests run offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.eval.cp_plots import (
    coalition_lifecycle_gantt,
    cp_setpoint_timeline,
    cross_sector_transfer_distribution,
    envelope_clamp_arrows,
    flag_on_off_comparison,
    render_all,
    render_comparison,
)


def _synthetic_events() -> list[dict]:
    """Mock ledger with one full coalition life-cycle on one P2H:
    failure, CP setpoints, inversion, coalition allocation, envelope
    set/clamp, then envelope expiry.
    """
    return [
        {"t": 0.0, "kind": "failure", "aid": "branch", "sector": "", "detail": ""},
        {
            "t": 0.5,
            "kind": "cp_setpoint",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": "flows={electricity: 0.0000, heat: 0.0000} reg=1.000 envelope_active=False",
        },
        {
            "t": 1.0,
            "kind": "cross_sector_inversion_detected",
            "aid": "leader-el-1",
            "sector": "electricity",
            "detail": "cp=p2h-1 own_sec=electricity tier_high=1 frac_high=0.300 peer_sec=heat tier_low=5 frac_low=1.000",
        },
        {
            "t": 1.0,
            "kind": "cross_sector_coalition_allocation",
            "aid": "leader-el-1",
            "sector": "electricity",
            "detail": (
                "id=xs:leader-el-1#1 cp=p2h-1 transfer_out=0.5000 transfer_in=1.0000 "
                "own_frac={1: 0.550} peer_frac={5: 0.000}"
            ),
        },
        {
            "t": 1.0,
            "kind": "cp_envelope_set",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": (
                "coalition=xs:leader-el-1#1 ttl=4.00 "
                "flows={electricity: 0.5000, heat: -1.0000}"
            ),
        },
        {
            "t": 1.5,
            "kind": "cp_setpoint",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": "flows={electricity: 0.5000, heat: -1.0000} reg=0.700 envelope_active=True",
        },
        {
            "t": 2.0,
            "kind": "cp_envelope_clamp",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": (
                "coalition=xs:leader-el-1#1 pre=[0.7, -0.6, 0.0] post=[0.5, -1.0, 0.0]"
            ),
        },
        {
            "t": 2.0,
            "kind": "cp_setpoint",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": "flows={electricity: 0.5000, heat: -1.0000} reg=0.700 envelope_active=True",
        },
        {
            "t": 5.5,
            "kind": "cp_setpoint",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": "flows={electricity: 0.2000, heat: -0.4000} reg=0.400 envelope_active=False",
        },
    ]


def _synthetic_summary_on() -> dict:
    return {
        "all": {
            "failure": 1,
            "cp_setpoint": 4,
            "cross_sector_inversion_detected": 1,
            "cross_sector_coalition_allocation": 1,
            "cp_envelope_set": 1,
            "cp_envelope_clamp": 1,
        },
        "cross_sector": {
            "cross_sector_inversion_detected": 1,
            "cross_sector_coalition_allocation": 1,
            "cp_envelope_set": 1,
            "cp_envelope_clamp": 1,
            "cp_setpoint": 4,
        },
    }


def _synthetic_summary_off() -> dict:
    return {
        "all": {"failure": 1, "cp_setpoint": 3},
        "cross_sector": {
            "cross_sector_inversion_detected": 0,
            "cross_sector_coalition_allocation": 0,
            "cp_envelope_set": 0,
            "cp_envelope_clamp": 0,
            "cp_setpoint": 3,
        },
    }


@pytest.fixture
def run_dir(tmp_path) -> Path:
    """One synthetic run with events.json + summary.json populated."""
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    (on_dir / "events.json").write_text(json.dumps(_synthetic_events()))
    (on_dir / "summary.json").write_text(json.dumps(_synthetic_summary_on()))
    return on_dir


@pytest.fixture
def off_run_dir(tmp_path) -> Path:
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    # CP setpoints but no cross-sector events.
    off_events = [
        {"t": 0.0, "kind": "failure", "aid": "branch", "sector": "", "detail": ""},
        {
            "t": 0.5,
            "kind": "cp_setpoint",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": "flows={electricity: 0.0000, heat: 0.0000} reg=1.000 envelope_active=False",
        },
        {
            "t": 1.5,
            "kind": "cp_setpoint",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": "flows={electricity: 0.1000, heat: -0.2000} reg=0.900 envelope_active=False",
        },
        {
            "t": 2.5,
            "kind": "cp_setpoint",
            "aid": "p2h-1",
            "sector": "cp",
            "detail": "flows={electricity: 0.2000, heat: -0.4000} reg=0.800 envelope_active=False",
        },
    ]
    (off_dir / "events.json").write_text(json.dumps(off_events))
    (off_dir / "summary.json").write_text(json.dumps(_synthetic_summary_off()))
    return off_dir


class TestPlots:
    def test_cp_setpoint_timeline_writes_html(self, run_dir: Path):
        stem = cp_setpoint_timeline(
            run_dir / "events.json",
            run_dir / "plots" / "cp_setpoint_timeline",
        )
        assert stem.with_suffix(".html").exists()

    def test_coalition_lifecycle_gantt_writes_html(self, run_dir: Path):
        stem = coalition_lifecycle_gantt(
            run_dir / "events.json",
            run_dir / "plots" / "coalition_lifecycle_gantt",
        )
        assert stem.with_suffix(".html").exists()

    def test_envelope_clamp_arrows_writes_html(self, run_dir: Path):
        stem = envelope_clamp_arrows(
            run_dir / "events.json",
            run_dir / "plots" / "envelope_clamp_arrows",
        )
        assert stem.with_suffix(".html").exists()

    def test_transfer_distribution_writes_html(self, run_dir: Path):
        stem = cross_sector_transfer_distribution(
            run_dir / "events.json",
            run_dir / "plots" / "cross_sector_transfer_distribution",
        )
        assert stem.with_suffix(".html").exists()

    def test_render_all_writes_every_figure(self, run_dir: Path):
        stems = render_all(run_dir)
        for name, stem in stems.items():
            assert stem.with_suffix(".html").exists(), f"{name} missing"


class TestComparison:
    def test_flag_comparison_writes_html(
        self,
        run_dir: Path,
        off_run_dir: Path,
        tmp_path: Path,
    ):
        out = tmp_path / "compare"
        stem = flag_on_off_comparison(
            off_run_dir / "summary.json",
            run_dir / "summary.json",
            out / "flag_on_off",
        )
        assert stem.with_suffix(".html").exists()

    def test_render_comparison_bundle(
        self,
        run_dir: Path,
        off_run_dir: Path,
        tmp_path: Path,
    ):
        out = tmp_path / "compare"
        stem = render_comparison(off_run_dir, run_dir, out)
        assert stem.with_suffix(".html").exists()


class TestEmptyLedger:
    """An empty or missing ledger must still produce a valid placeholder
    figure rather than crashing.
    """

    def test_empty_ledger_does_not_crash(self, tmp_path: Path):
        ledger = tmp_path / "empty.json"
        ledger.write_text("[]")
        stem = cp_setpoint_timeline(ledger, tmp_path / "out" / "x")
        assert stem.with_suffix(".html").exists()

    def test_missing_ledger_does_not_crash(self, tmp_path: Path):
        stem = cp_setpoint_timeline(
            tmp_path / "does_not_exist.json",
            tmp_path / "out" / "x",
        )
        assert stem.with_suffix(".html").exists()
