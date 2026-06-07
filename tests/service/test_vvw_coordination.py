"""Tests for coordinated Volt-VAR-Watt reactive support in the Q(U) droop.

With ``enable_vvw_coordination`` the inverter uses its full apparent-power
capability circle (``q_max = √(S_n²−p²)``) under voltage support instead of the
tighter VDE cos-φ displacement cap (``q_max = p·tan φ_min``). Once active power
is curtailed this frees apparent capacity for extra reactive absorption — the
key to clearing over-voltage without heavy active curtailment.
"""

from __future__ import annotations

import math

import pytest

from scare.base.config import RestorationConfiguration
from scare.service.control.voltage_droop import ReactivePowerDroopRole
from tests.conftest import MockBehavior


class _FakeCtx:
    def __init__(self, aid: str) -> None:
        self.aid = aid
        self.current_timestamp = 0.0


def _make_droop(
    *, s_nom_mva: float, vvw: bool
) -> tuple[ReactivePowerDroopRole, MockBehavior]:
    b = MockBehavior()
    b.add_action("pv", "set_q")
    b._scare_config = RestorationConfiguration(enable_vvw_coordination=vvw)
    role = ReactivePowerDroopRole(b, s_nom_mva=s_nom_mva)
    role._context = _FakeCtx("pv")  # type: ignore[attr-defined]
    return role, b


def _last_set_q(b: MockBehavior) -> float:
    calls = [c for c in b.action_log if c[1] == "set_q"]
    assert calls, "no set_q command recorded"
    return calls[-1][2][0]


# S_n = p_rated/0.9 = 0.1148 for a 0.1033 MW PV; at a CURTAILED p=0.09 the
# capability circle (0.0713) exceeds the cos-φ cap (0.0436).
_S_NOM = 0.1033 / 0.9
_P_CURTAILED = 0.09


@pytest.mark.asyncio
async def test_vvw_uses_full_capability_circle_when_curtailed():
    role, b = _make_droop(s_nom_mva=_S_NOM, vvw=True)
    # Over-voltage (v >= 1.05) -> droop absorbs +q_max; p curtailed below rating.
    b.set_obs("pv", {"vm_pu": 1.07, "p_mw": -_P_CURTAILED, "regulation": 1.0})
    await role._step()
    circle_q = math.sqrt(_S_NOM**2 - _P_CURTAILED**2)
    assert _last_set_q(b) == pytest.approx(circle_q, abs=1e-4)


@pytest.mark.asyncio
async def test_cos_phi_cap_applies_without_vvw():
    role, b = _make_droop(s_nom_mva=_S_NOM, vvw=False)
    b.set_obs("pv", {"vm_pu": 1.07, "p_mw": -_P_CURTAILED, "regulation": 1.0})
    await role._step()
    cos = 0.90  # S_n > 13.8 kVA
    cos_phi_q = _P_CURTAILED * math.sqrt(1 - cos**2) / cos
    assert _last_set_q(b) == pytest.approx(cos_phi_q, abs=1e-4)


@pytest.mark.asyncio
async def test_vvw_gives_more_reactive_than_cos_phi_when_curtailed():
    on, b_on = _make_droop(s_nom_mva=_S_NOM, vvw=True)
    off, b_off = _make_droop(s_nom_mva=_S_NOM, vvw=False)
    obs = {"vm_pu": 1.07, "p_mw": -_P_CURTAILED, "regulation": 1.0}
    b_on.set_obs("pv", obs)
    b_off.set_obs("pv", dict(obs))
    await on._step()
    await off._step()
    assert _last_set_q(b_on) > _last_set_q(b_off)


@pytest.mark.asyncio
async def test_vvw_and_cos_phi_coincide_at_full_power():
    # At full active output S_n = p/cos φ_min, so circle and cos-φ caps match:
    # VVW only bites once p is curtailed.
    p_full = 0.1033
    s_nom = p_full / 0.9
    on, b_on = _make_droop(s_nom_mva=s_nom, vvw=True)
    off, b_off = _make_droop(s_nom_mva=s_nom, vvw=False)
    obs = {"vm_pu": 1.07, "p_mw": -p_full, "regulation": 1.0}
    b_on.set_obs("pv", obs)
    b_off.set_obs("pv", dict(obs))
    await on._step()
    await off._step()
    assert _last_set_q(b_on) == pytest.approx(_last_set_q(b_off), abs=1e-4)
