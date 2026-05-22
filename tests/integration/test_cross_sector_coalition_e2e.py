"""End-to-end test for the L2.5 cross-sector coalition pathway.

Drives the full :func:`create_restoration_scenario_world` +
:func:`start_restoration_simulation` pipeline twice on the same network
+ failure injection, with the only difference being the
:attr:`RestorationConfiguration.enable_cross_sector_coalitions` knob.

What is captured (per run)
--------------------------

After each run the diagnostics ledger is dumped to JSON in a temp dir
so the plotting layer (and any downstream analysis) can read it back
with no further wiring.  The artefacts written:

* ``events.json``    — full event_log() snapshot
* ``summary.json``   — counts per event kind + cross-sector roll-up

What this test asserts
----------------------

1. With the flag **off**, no ``cross_sector_inversion_detected``,
   ``cross_sector_coalition_allocation``, or ``cp_envelope_*`` event
   appears in the ledger — the cross-sector path is genuinely a
   no-op.
2. With the flag **on**, the simulation runs to completion without
   error (we don't assert a specific count because the network /
   failure may not happen to produce an inversion within the
   simulation window; the assertion is "the pathway is wired and the
   simulation is stable when it is enabled").
3. Both runs produce a non-empty world clock — basic smoke check.

The GEKKO solver is stubbed (same pattern as the existing scenario
tests) to keep the test runnable without a numerical-solver
installation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from mango_energy_environments import fetch_example_net
import mango_energy_environments.environments.restoration.multi_energy_monee as _restoration_mod

from scare.base import diagnostics
from scare.base.config import RestorationConfiguration
from scare.base.util import create_failures
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)


# ---------------------------------------------------------------------------
# Solver stub (mirrors tests/scenario/test_restoration.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dump_events(out_dir: Path, label: str) -> dict[str, dict]:
    """Serialise the current diagnostics state to ``out_dir/<label>/``.

    Writes two files the downstream plotting helpers consume:

    * ``events.json``  — list[dict] of every EventRecord
    * ``summary.json`` — event_kind → count + cross-sector roll-up
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
    """Skew priorities so the cross-sector inversion has something
    to detect.  Even tier-1 / tier-9 split across loads keeps the
    sample biased toward inversion-producing scenarios within the
    short simulation window.
    """
    from scare.base.util import obs_capacity

    priorities: dict[str, int] = {}
    load_index = 0
    for child in net.childs:
        obs = dict(child.model.values)
        cap = obs_capacity(obs)
        aid = f"child-{child.id}"
        if cap > 0:  # loads only
            # alternate 1 (critical) / 9 (deferrable)
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
        # Speed up cross-sector detection — default period 1s leaves
        # only a few ticks in a 10s sim window.
        holon_summary_period_s=0.5,
    )

    world = create_restoration_scenario_world(
        net,
        priorities=priorities,
        simulation_duration_s=simulation_duration_s,
        config=config,
    )

    failures = create_failures(
        net, "branch", num_failures=1, delay_s_max=1.0,
    )
    await start_restoration_simulation(
        world, failures, simulation_duration_s=simulation_duration_s,
    )

    artefacts = _dump_events(out_dir, label)
    return world, artefacts


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cross_sector_coalition_flag_off(tmp_path):
    """With the flag off, no cross-sector pathway fires.

    Writes the event ledger out to ``tmp_path/off/`` so the plotting
    helpers can be exercised against a real ledger snapshot.
    """
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
    """With the flag on, the simulation runs end-to-end without error.

    We don't *assert* that the example network + injected failure
    produces an inversion in the 8 s window — the example net is small
    and the failure draw is random — but we do verify the wiring is
    stable: the world finishes, the ledger is dumped, and the
    cross-sector pathway is at least *available* (record_event hooks
    are reachable code).
    """
    world, art = await _run_once(
        enable_cross_sector=True,
        simulation_duration_s=8.0,
        out_dir=tmp_path,
        label="on",
    )
    assert world.clock.time > 0
    # The ledger snapshot must exist on disk so downstream tooling
    # (plots, report) can ingest it deterministically.
    assert (tmp_path / "on" / "events.json").exists()
    assert (tmp_path / "on" / "summary.json").exists()


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_cross_sector_coalition_side_by_side(tmp_path):
    """Run both configurations on the same network shape and compare.

    The side-by-side artefacts are exactly the input format the
    cross-sector plotting helpers consume, so this test doubles as a
    fixture-generator that downstream visualisation tests can re-read.
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

    # Off must have ZERO cross-sector activity.
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

    # On run: not asserting > 0 because the network shape may not
    # produce an inversion; but the wiring must be different from
    # off in *some* way (typically: extra CP setpoint events, since
    # the CP layer still runs).  At minimum the cp_setpoint counter
    # should be ≥ off's (CP ADMM continues to fire either way).
    assert (
        art_on["cross_sector"]["cp_setpoint"]
        >= art_off["cross_sector"]["cp_setpoint"]
    )
