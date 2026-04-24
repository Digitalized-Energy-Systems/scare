"""Scenario test: full multi-energy restoration with real monee network.

This test exercises the entire pipeline: world creation, failure injection,
gossip-based negotiation, constraint monitoring, and generation control.
It verifies that the system responds to failures — not that it perfectly
resolves them (that depends on convergence tuning and network topology).

The GEKKO solver is stubbed out so these tests don't require a numerical
solver installation.
"""

import pytest

from mango_energy_environments import fetch_example_net
import mango_energy_environments.environments.restoration.multi_energy_monee as _restoration_mod

from scare.base.util import create_failures
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)


class _FakeEnergyFlowResult:
    """Pass-through stub: returns original network so observer closures work."""

    def __init__(self, net):
        self.network = net


@pytest.fixture(autouse=True)
def _stub_energyflow(monkeypatch):
    """Replace energyflow with a pass-through stub to avoid GEKKO dependency."""
    monkeypatch.setattr(
        _restoration_mod,
        "energyflow",
        lambda net: _FakeEnergyFlowResult(net),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_restoration_scenario_runs_without_error():
    """Full scenario: create world, inject failure, run 10s of simulation."""
    net = fetch_example_net()

    world = create_restoration_scenario_world(
        net, simulation_duration_s=10.0
    )

    failures = create_failures(net, "branch", num_failures=1, delay_s_max=1.0)
    assert len(failures) >= 1

    # The main check: the simulation runs to completion without raising
    await start_restoration_simulation(world, failures, simulation_duration_s=10.0)

    # Basic sanity: world clock advanced
    assert world.clock.time > 0


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_restoration_scenario_has_recordings():
    """After simulation, world should contain recording data."""
    net = fetch_example_net()

    world = create_restoration_scenario_world(
        net, simulation_duration_s=5.0
    )

    failures = create_failures(net, "branch", num_failures=1, delay_s_max=0.5)
    await start_restoration_simulation(world, failures, simulation_duration_s=5.0)

    # The world should have accumulated some agent recording data
    assert len(world.data_agent_collections) > 0
