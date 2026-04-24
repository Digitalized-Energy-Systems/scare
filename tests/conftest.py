"""Shared fixtures for SCARE tests."""

from __future__ import annotations

from typing import Any

import pytest

from scare.base.model import Sector


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


@pytest.fixture()
def mock_behavior() -> MockBehavior:
    return MockBehavior()


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
    return {"p_mw": p_mw, "regulation": regulation, "vm_pu": vm_pu, "priority": priority}


def make_gas_obs(
    mass_flow: float = 0.5,
    regulation: float = 0.5,
    pressure_pu: float = 1.0,
) -> dict:
    return {"mass_flow": mass_flow, "regulation": regulation, "pressure_pu": pressure_pu}


def make_heat_obs(
    q_w_set: float = 1000.0,
    regulation: float = 0.5,
    t_k: float = 363.15,
) -> dict:
    return {"q_w_set": q_w_set, "regulation": regulation, "t_k": t_k}


def make_heat_mass_flow_obs(
    mass_flow: float = 0.3,
    regulation: float = 0.5,
    t_k: float = 363.15,
) -> dict:
    """Heat junction that carries mass_flow (disambiguated from gas by t_k)."""
    return {"mass_flow": mass_flow, "regulation": regulation, "t_k": t_k}
