"""Shared fixtures for SCARE tests."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

# Failure sampling draws from the process-global RNG (see
# scare.scenario.failure_sampling), so an unseeded test gets a different failure
# scenario every run -- different branches cut, different delays. Eval campaigns
# seed via experiment.hpc.runner._seed_everything; the suite had no equivalent,
# which made outcomes that depend on the sampled scenario flap run-to-run.
_TEST_SEED = 0


@pytest.fixture(autouse=True)
def _seed_rngs():
    random.seed(_TEST_SEED)
    np.random.seed(_TEST_SEED)


# ---------------------------------------------------------------------------
# MockBehavior — stubs RestorationEnvironmentBehavior
# ---------------------------------------------------------------------------


class MockBehavior:
    """Test double for ``RestorationEnvironmentBehavior``.

    Stores observations by agent-id and records every ``act`` call so
    tests can assert on control actions taken by roles.
    """

    def __init__(self) -> None:
        self._observations: dict[str, dict] = {}
        self._actions: dict[str, set[str]] = {}
        self.action_log: list[tuple[str, str, tuple, dict]] = []

    # --- Setup helpers ---

    def set_obs(self, aid: str, obs: dict) -> None:
        self._observations[aid] = obs

    def add_action(self, aid: str, action: str) -> None:
        self._actions.setdefault(aid, set()).add(action)

    # --- Interface used by roles ---

    def observe(self, agent_id: str) -> dict:
        return self._observations.get(agent_id, {})

    def act(self, agent_id: str, action: str, *args: Any, **kwargs: Any) -> None:
        self.action_log.append((agent_id, action, args, kwargs))

    def has_action(self, agent_id: str, action: str) -> bool:
        return action in self._actions.get(agent_id, set())

    def install(self, agent: Any, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Observation dict factories
# ---------------------------------------------------------------------------


def make_electricity_gen(
    p_mw: float = -5.0,
    regulation: float = 1.0,
    vm_pu: float = 1.0,
) -> dict:
    return {"p_mw": p_mw, "regulation": regulation, "vm_pu": vm_pu}


def make_electricity_load(
    p_mw: float = 3.0,
    regulation: float = 0.0,
    vm_pu: float = 1.0,
    priority: int = 1,
) -> dict:
    return {
        "p_mw": p_mw,
        "regulation": regulation,
        "vm_pu": vm_pu,
        "priority": priority,
    }
