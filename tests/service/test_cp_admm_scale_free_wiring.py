"""``cp_admm_scale_free`` must reach both L3 solver paths.

The flag switches the lexicographic cascade into a non-dimensional frame with
a ridge relative to each CP's own ``||c_i||^2``.  Without it, on LV-scale CPs
every ADMM residual sits below ``admm_abs_tol`` from the first iteration of
each tier and the cascade commits its ``r = 0`` initialisation while reporting
convergence.  These tests pin the plumbing -- the numerics are covered by
DRO's ``test_scale_invariance``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from distributed_resource_optimization import SectorDemand

from scare.base.config import RestorationConfiguration
from scare.base.model import Sector
from scare.service.coupling.cp_priority_admm_role import CPPriorityAdmmRole
from tests.service.test_cp_priority_admm_wiring import _build_behavior_for

_LV_CAPS = {"electricity": 0.0050, "heat": -0.0036}


def _role(**kwargs: Any) -> CPPriorityAdmmRole:
    return CPPriorityAdmmRole(
        behavior=_build_behavior_for("cp-1"),
        cp_id="cp-1",
        capacity_by_sector=_LV_CAPS,
        bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        **kwargs,
    )


def test_role_defaults_to_legacy_behaviour():
    """Byte-parity guard: constructing a role without the flag must not opt in."""
    assert _role().scale_free is False


def test_role_accepts_the_flag():
    assert _role(scale_free=True).scale_free is True


def test_config_default_is_on():
    """The fix ships enabled; False is the A/B counterfactual."""
    assert RestorationConfiguration().cp_admm_scale_free is True


class _StubParticipant:
    def __init__(self) -> None:
        self.started: list[Any] = []

    def is_round_active(self) -> bool:
        return False

    async def on_exchange_message(self, carrier: Any, msg: Any, meta: Any) -> None:
        self.started.append(msg)


@pytest.mark.parametrize("scale_free", [False, True])
@pytest.mark.asyncio
async def test_flag_reaches_the_real_gossip_start(monkeypatch, scale_free):
    """Drive the role's own ``_run_gossip_round`` and read the emitted Start."""
    from scare.base.model import Sector as S

    role = _role(scale_free=scale_free, algorithm="gossip")
    stub = _StubParticipant()
    role._gossip_participant = stub  # type: ignore[assignment]
    role._gossip_carrier = object()  # type: ignore[assignment]
    monkeypatch.setattr(role, "_reachable_node_set", lambda: None)
    monkeypatch.setattr(role, "_am_gossip_initiator", lambda _n: True)
    monkeypatch.setattr(role, "_reachable_peer_cp_ids", lambda _n=None: set())
    monkeypatch.setattr(
        role,
        "_build_demands",
        lambda _n: [
            SectorDemand(
                sector=S.HEAT.value,
                demand_by_tier={1: np.array([0.008])},
                base_supply=np.array([0.0]),
            )
        ],
    )

    await role._run_gossip_round()

    assert len(stub.started) == 1, "role did not emit a gossip Start"
    start = stub.started[0]
    assert start.normalize is scale_free
    assert start.r_regularization_relative is scale_free
    assert start.minimize_usage is scale_free


def test_restoration_threads_config_flag_into_the_role():
    """``_attach_cp_role`` must forward ``cp_admm_scale_free``, not drop it."""
    import inspect

    from scare.scenario import restoration

    src = inspect.getsource(restoration)
    assert "scale_free=config.cp_admm_scale_free" in src
