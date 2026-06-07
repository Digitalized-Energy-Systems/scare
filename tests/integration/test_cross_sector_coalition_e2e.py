"""End-to-end test for the L2.5 cross-sector coalition pathway.

Drives the full :func:`create_restoration_scenario_world` +
:func:`start_restoration_simulation` pipeline twice on the same network
+ failure injection, toggling only
:attr:`RestorationConfiguration.enable_cross_sector_coalitions`. Each
run dumps the diagnostics ledger to ``events.json`` (full event_log
snapshot) and ``summary.json`` (per-kind counts + cross-sector roll-up)
so the plotting layer can read it back.

Asserts: with the flag off, no inversion / coalition / cp_envelope_*
events appear (the path is a genuine no-op); with the flag on, the sim
runs to completion and dumps the ledger (a specific inversion count is
not asserted, since the small net may not produce one in-window).

The GEKKO solver is stubbed to run without a numerical-solver install.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mango_energy_environments.environments.restoration.multi_energy_monee as _restoration_mod
import pytest
from mango_energy_environments import fetch_example_net

from scare.base.config import RestorationConfiguration
from scare.base.runtime import diagnostics
from scare.base.util import create_failures
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)


class _FakeEnergyFlowResult:
    def __init__(self, net):
        self.network = net


@pytest.fixture(autouse=True)
def _stub_energyflow(monkeypatch):
    monkeypatch.setattr(
        _restoration_mod,
        "energyflow",
        lambda net: _FakeEnergyFlowResult(net),
    )


def _dump_events(out_dir: Path, label: str) -> dict[str, dict]:
    """Serialise diagnostics state to ``out_dir/<label>/`` as
    ``events.json`` (every EventRecord) and ``summary.json``
    (event_kind -> count + cross-sector roll-up).
    """
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    events = [asdict(r) for r in diagnostics.event_log()]
    (label_dir / "events.json").write_text(json.dumps(events, indent=2))

    summary = diagnostics.event_summary()
    xs_kinds = (
        "cross_sector_inversion_detected",
        "cross_sector_coalition_allocation",
        "cp_envelope_set",
        "cp_envelope_clamp",
        "cp_setpoint",
    )
    cross_sector_roll = {k: summary.get(k, 0) for k in xs_kinds}
    (label_dir / "summary.json").write_text(
        json.dumps(
            {"all": summary, "cross_sector": cross_sector_roll},
            indent=2,
            sort_keys=True,
        )
    )
    return {"events": events, "summary": summary, "cross_sector": cross_sector_roll}


def _build_priorities(net) -> dict[str, int]:
    """Alternate loads tier-1 / tier-9 so the cross-sector inversion
    has something to detect within the short simulation window.
    """
    from scare.base.util import obs_capacity

    priorities: dict[str, int] = {}
    load_index = 0
    for child in net.childs:
        obs = dict(child.model.values)
        cap = obs_capacity(obs)
        aid = f"child-{child.id}"
        if cap > 0:  # loads only
            priorities[aid] = 1 if (load_index % 2 == 0) else 9
            load_index += 1
        else:
            priorities[aid] = 0  # generators
    return priorities


async def _run_once(
    *,
    enable_cross_sector: bool,
    simulation_duration_s: float,
    out_dir: Path,
    label: str,
):
    diagnostics.arm()  # also clears all logs

    net = fetch_example_net()
    priorities = _build_priorities(net)
    config = RestorationConfiguration(
        enable_cross_sector_coalitions=enable_cross_sector,
        # Default 1s period leaves only a few ticks in an 8s window.
        holon_summary_period_s=0.5,
    )

    world = create_restoration_scenario_world(
        net,
        priorities=priorities,
        simulation_duration_s=simulation_duration_s,
        config=config,
    )

    failures = create_failures(
        net,
        "branch",
        num_failures=1,
        delay_s_max=1.0,
    )
    await start_restoration_simulation(
        world,
        failures,
        simulation_duration_s=simulation_duration_s,
    )

    artefacts = _dump_events(out_dir, label)
    return world, artefacts


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cross_sector_coalition_flag_off(tmp_path):
    """With the flag off, no cross-sector pathway fires."""
    world, art = await _run_once(
        enable_cross_sector=False,
        simulation_duration_s=8.0,
        out_dir=tmp_path,
        label="off",
    )
    assert world.clock.time > 0
    xs = art["cross_sector"]
    assert xs["cross_sector_inversion_detected"] == 0
    assert xs["cross_sector_coalition_allocation"] == 0
    assert xs["cp_envelope_set"] == 0
    assert xs["cp_envelope_clamp"] == 0


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cross_sector_coalition_flag_on(tmp_path):
    """With the flag on, the simulation runs end-to-end and dumps the
    ledger. An inversion is not asserted (small net, random failure
    draw); this checks the pathway is wired and stable.
    """
    world, art = await _run_once(
        enable_cross_sector=True,
        simulation_duration_s=8.0,
        out_dir=tmp_path,
        label="on",
    )
    assert world.clock.time > 0
    # The ledger snapshot must exist on disk for downstream tooling.
    assert (tmp_path / "on" / "events.json").exists()
    assert (tmp_path / "on" / "summary.json").exists()


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_cross_sector_coalition_side_by_side(tmp_path):
    """Run both configurations on the same network shape and compare;
    the side-by-side artefacts match the plotting helpers' input format.
    """
    _, art_off = await _run_once(
        enable_cross_sector=False,
        simulation_duration_s=8.0,
        out_dir=tmp_path,
        label="off",
    )
    _, art_on = await _run_once(
        enable_cross_sector=True,
        simulation_duration_s=8.0,
        out_dir=tmp_path,
        label="on",
    )

    # Off must have zero cross-sector activity.
    for k in (
        "cross_sector_inversion_detected",
        "cross_sector_coalition_allocation",
        "cp_envelope_set",
        "cp_envelope_clamp",
    ):
        assert art_off["cross_sector"][k] == 0, (
            f"flag-off run produced {k} = {art_off['cross_sector'][k]}; "
            f"cross-sector path is leaking"
        )

    # Not asserting > 0 (the net may not invert), but the CP layer runs
    # either way, so on's cp_setpoint count must be >= off's.
    assert (
        art_on["cross_sector"]["cp_setpoint"] >= art_off["cross_sector"]["cp_setpoint"]
    )
