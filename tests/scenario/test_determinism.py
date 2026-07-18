"""Reproducibility lock: the same scenario, run twice, produces the same trace.

The refactor roadmap is gated on "the default arm reproduces HEAD byte-for-byte
(id-normalized)", but ``experiment/eval/parity.py`` only supplies the comparison
half (``normalize_ids``/``diff_normalized``) -- nothing executed a scenario twice
and checked. This is that half, in-process and small enough to run every commit.

It compares the ORDERED action + event ledgers, not just totals: ordering is what
catches message-arrival and dispatch-order drift, which is exactly what a
refactor of the coordination layer risks perturbing.

Determinism here depends on seeding the process-global RNG (failure sampling
draws from it) -- ``tests/conftest.py`` does that autouse. GEKKO is stubbed, so
this exercises coordination, not physics.
"""

from __future__ import annotations

import random

import mango_energy_environments.environments.restoration.multi_energy_monee as _restoration_mod
import numpy as np
import pytest
from mango_energy_environments import fetch_example_net

from scare.base.runtime import diagnostics
from scare.scenario.failure_sampling import create_failures
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)

_SIM_S = 8.0
_SEED = 0


class _FakeEnergyFlowResult:
    def __init__(self, net):
        self.network = net


class _FakeStepResult:
    def __init__(self, net):
        self.failed = False
        self.error = None
        self.result = _FakeEnergyFlowResult(net)


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


def _trace() -> tuple[tuple, ...]:
    """Ordered (kind, aid, t, detail) tuples from one scenario run."""
    actions = tuple(
        (
            r.aid,
            round(float(r.t), 9),
            r.reason,
            round(float(getattr(r, "value", 0.0)), 9),
        )
        for r in diagnostics.action_log()
    )
    events = tuple(
        (r.aid, round(float(r.t), 9), r.kind, r.sector) for r in diagnostics.event_log()
    )
    return actions + (("--",),) + events


async def _run_once() -> tuple[tuple, ...]:
    random.seed(_SEED)
    np.random.seed(_SEED)
    diagnostics.arm()

    net = fetch_example_net()
    world = create_restoration_scenario_world(net, simulation_duration_s=_SIM_S)
    failures = create_failures(net, "branch", num_failures=1, delay_s_max=1.0)
    await start_restoration_simulation(world, failures, simulation_duration_s=_SIM_S)
    return _trace()


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_same_seed_reproduces_the_same_ordered_trace():
    first = await _run_once()
    second = await _run_once()

    assert first == second, (
        "Two runs of the same seeded scenario diverged. Something on the "
        "coordination path is not reproducible (unseeded RNG, identity-ordered "
        "iteration, or off-clock scheduling). The parity gate the refactor "
        "roadmap depends on cannot hold while this fails."
    )
    # Guard against the assertion passing vacuously on an empty ledger.
    assert len(first) > 1, "scenario produced no diagnostics to compare"


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_failure_sampling_is_seed_determined():
    """The sampled scenario -- the experiments' independent variable -- must
    follow from the seed alone."""
    net = fetch_example_net()

    random.seed(_SEED)
    a = [
        (tuple(f.branch_ids), round(f.delay_s, 9))
        for f in create_failures(net, "branch", num_failures=2, delay_s_max=1.0)
    ]
    random.seed(_SEED)
    b = [
        (tuple(f.branch_ids), round(f.delay_s, 9))
        for f in create_failures(net, "branch", num_failures=2, delay_s_max=1.0)
    ]

    assert a == b
    random.seed(_SEED + 1)
    c = [
        (tuple(f.branch_ids), round(f.delay_s, 9))
        for f in create_failures(net, "branch", num_failures=2, delay_s_max=1.0)
    ]
    assert a != c, "a different seed must give a different scenario"
