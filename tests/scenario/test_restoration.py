"""Scenario test: full multi-energy restoration with a real monee network.

Exercises the whole pipeline (world creation, failure injection, gossip
negotiation, constraint monitoring, generation control) and checks the
system runs and responds to failures — not that it perfectly resolves them.

GEKKO is stubbed out so no numerical solver install is required.
"""

import mango_energy_environments.environments.restoration.multi_energy_monee as _restoration_mod
import pytest
from mango_energy_environments import fetch_example_net

from scare.scenario.failure_sampling import create_failures
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)


class _FakeEnergyFlowResult:
    """Pass-through stub: returns original network so observer closures work."""

    def __init__(self, net):
        self.network = net


class _FakeStepResult:
    def __init__(self, net):
        self.failed = False
        self.error = None
        self.result = _FakeEnergyFlowResult(net)


class _FakeStepper:
    """Pass-through stepper: every step 'solves' to the live net unchanged."""

    def __init__(self, net):
        self._net = net
        self.changes = []

    def step(self, dt_h, **kwargs):
        return _FakeStepResult(self._net)

    def changes_df(self):
        return None


@pytest.fixture(autouse=True)
def _stub_energyflow(monkeypatch):
    """Replace the physics stepper with a pass-through stub — no solver needed."""
    monkeypatch.setattr(
        _restoration_mod,
        "create_physics_stepper",
        lambda net, **kwargs: _FakeStepper(net),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_restoration_scenario_runs_without_error():
    """Create world, inject a failure, run 10 s of simulation to completion."""
    net = fetch_example_net()

    world = create_restoration_scenario_world(net, simulation_duration_s=10.0)

    failures = create_failures(net, "branch", num_failures=1, delay_s_max=1.0)
    assert len(failures) >= 1

    await start_restoration_simulation(world, failures, simulation_duration_s=10.0)

    assert world.clock.time > 0


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_restoration_scenario_has_recordings():
    """Simulation accumulates agent recording data."""
    net = fetch_example_net()

    world = create_restoration_scenario_world(net, simulation_duration_s=5.0)

    failures = create_failures(net, "branch", num_failures=1, delay_s_max=0.5)
    await start_restoration_simulation(world, failures, simulation_duration_s=5.0)

    assert len(world.data_agent_collections) > 0
