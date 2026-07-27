"""The forced-republish watchdogs must be configurable, not pinned to 30 s.

``HolonSummaryRole``, ``HolonicCommunityRole`` and ``CPPriorityAdmmRole`` each
run a periodic task that forces a publish / re-form / re-balance through the
delta gate that otherwise suppresses them while nothing moves. Every one of
them defaulted to a hardcoded ``watchdog_s=30.0`` that no caller overrode, and
``simulation_duration_s`` is 30.0 in the shipped campaign configs — so the
watchdogs fired at most once, at the horizon, and the recovery path they exist
to provide was inert for a whole campaign. A deficit allocation computed once on
a stale observation then stayed frozen for the rest of the run.
"""

from __future__ import annotations

import mango_energy_environments.environments.restoration.multi_energy_monee as _restoration_mod
import pytest
from mango_energy_environments import fetch_example_net

from scare.base.config import RestorationConfiguration
from scare.community.holonic import HolonicCommunityRole
from scare.community.summary import HolonSummaryRole
from scare.scenario.restoration import create_restoration_scenario_world
from scare.service.coupling.cp_priority_admm_role import CPPriorityAdmmRole

_WATCHDOG_ROLES = (HolonSummaryRole, HolonicCommunityRole, CPPriorityAdmmRole)


class _FakeStepResult:
    def __init__(self, net):
        self.failed = False
        self.error = None
        self.result = type("_R", (), {"network": net})()


class _FakeStepper:
    def __init__(self, net):
        self._net = net
        self.changes = []

    def step(self, dt_h, **kwargs):
        return _FakeStepResult(self._net)

    def changes_df(self):
        return None


@pytest.fixture(autouse=True)
def _stub_energyflow(monkeypatch):
    monkeypatch.setattr(
        _restoration_mod,
        "create_physics_stepper",
        lambda net, **kwargs: _FakeStepper(net),
    )


def _watchdog_roles(world):
    agents = world.agents
    # ``world.agents`` is a dict keyed by aid; iterating it directly would walk
    # the keys and silently find nothing.
    agents = agents.values() if hasattr(agents, "values") else agents
    return [
        role
        for agent in agents
        for role in getattr(agent, "roles", []) or []
        if isinstance(role, _WATCHDOG_ROLES)
    ]


def test_default_watchdog_still_matches_the_historical_hardcoded_value():
    """Default must stay 30.0 so prior campaigns remain byte-reproducible."""
    assert RestorationConfiguration().holon_watchdog_s == 30.0
    for cls in _WATCHDOG_ROLES:
        import inspect

        default = inspect.signature(cls.__init__).parameters["watchdog_s"].default
        assert default == 30.0, f"{cls.__name__} watchdog default drifted"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_watchdog_period_is_plumbed_from_config_to_every_role():
    net = fetch_example_net()
    world = create_restoration_scenario_world(
        net,
        simulation_duration_s=10.0,
        config=RestorationConfiguration(holon_watchdog_s=1.5),
    )
    roles = _watchdog_roles(world)
    assert roles, "no watchdog-bearing role was built — test is vacuous"
    stuck = [type(r).__name__ for r in roles if r.watchdog_s != 1.5]
    assert not stuck, f"roles ignored holon_watchdog_s: {sorted(set(stuck))}"


def test_watchdog_at_or_above_the_horizon_never_fires_a_second_time():
    """Guards the actual defect: a period >= the run length arms nothing.

    Not a style check — with ``holon_watchdog_s == simulation_duration_s`` the
    periodic task has at most one firing opportunity, at the horizon, so the
    delta gate can never be broken mid-run.
    """
    duration_s = 30.0
    cfg = RestorationConfiguration()
    assert cfg.holon_watchdog_s >= duration_s, (
        "default no longer matches the shipped horizon; update this test and "
        "re-baseline, the change is not byte-identical"
    )
    n_firings = int(duration_s // cfg.holon_watchdog_s)
    assert n_firings <= 1, "expected the inert-watchdog condition at the default"

    armed = RestorationConfiguration(holon_watchdog_s=2.0)
    assert int(duration_s // armed.holon_watchdog_s) >= 10
