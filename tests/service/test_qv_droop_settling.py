"""Tests for the VDE-AR-N 4105 §5.7.2 settling lag on the Q(U) droop.

Q(U) is proportional-only and the whole inverter fleet closes the loop through
one shared bus voltage, so on a weak feeder the aggregate loop gain
``|dV/dQ| · Σq_max / (V_HIGH − V_DEADBAND_HIGH)`` exceeds 1 and the droop
limit-cycles bang-bang between the deadband edge and saturation. The lag scales
that gain by ``dt/tau``. Measured on LV-S (0.91 MVar of inverter circle behind a
160 kVA substation): gain 4.25 without the lag, 0.71 at tau=3 s.
"""

from __future__ import annotations

import math

import pytest

from scare.base.config import RestorationConfiguration
from scare.service.control.voltage_droop import (
    VDE_V_DEADBAND_HIGH,
    VDE_V_HIGH,
    ReactivePowerDroopRole,
)
from tests.conftest import MockBehavior

_S_NOM = 0.1033 / 0.9
_POLL = 0.5


class _FakeCtx:
    def __init__(self, aid: str) -> None:
        self.aid = aid
        self.current_timestamp = 0.0


def _make_droop(
    *, tau: float, attack: float | None = None, vvw: bool = True
) -> tuple[ReactivePowerDroopRole, MockBehavior, _FakeCtx]:
    b = MockBehavior()
    b.add_action("pv", "set_q")
    b._scare_config = RestorationConfiguration(enable_vvw_coordination=vvw)
    role = ReactivePowerDroopRole(
        b, s_nom_mva=_S_NOM, settling_tau_s=tau, attack_tau_s=attack
    )
    ctx = _FakeCtx("pv")
    role._context = ctx  # type: ignore[attr-defined]
    return role, b, ctx


def _last_set_q(b: MockBehavior) -> float:
    calls = [c for c in b.action_log if c[1] == "set_q"]
    assert calls, "no set_q command recorded"
    return calls[-1][2][0]


async def _tick(role, ctx, *, v: float, p: float = 0.0, dt: float = _POLL) -> None:
    role.behavior.set_obs("pv", {"vm_pu": v, "p_mw": -p, "regulation": 1.0})
    await role._step()
    ctx.current_timestamp += dt


@pytest.mark.asyncio
async def test_tau_zero_commits_instantly():
    role, b, ctx = _make_droop(tau=0.0)
    await _tick(role, ctx, v=1.07)
    assert _last_set_q(b) == pytest.approx(_S_NOM, abs=1e-4)


@pytest.mark.asyncio
async def test_first_tick_delivers_only_one_alpha_step():
    role, b, ctx = _make_droop(tau=3.0)
    await _tick(role, ctx, v=1.07)
    assert _last_set_q(b) == pytest.approx(_S_NOM * (_POLL / 3.0), rel=1e-6)


@pytest.mark.asyncio
async def test_ramps_to_target_on_a_static_plateau():
    """The cache gate must not freeze the ramp: inputs stop moving after the
    first tick, but the committed Q has to keep closing on the target."""
    role, b, ctx = _make_droop(tau=3.0)
    for _ in range(60):
        await _tick(role, ctx, v=1.07)
    assert _last_set_q(b) == pytest.approx(_S_NOM, rel=1e-3)


@pytest.mark.asyncio
async def test_lag_damps_the_limit_cycle():
    """Closed loop against a linear plant stiff enough to be unstable without
    the lag: v = v_open − k·Q with a loop gain of 4.25, LV-S's measured value."""
    band = VDE_V_HIGH - VDE_V_DEADBAND_HIGH
    v_open = 1.0372
    gain = 4.25
    k = gain * band / _S_NOM  # so k·q_max/band == gain

    async def _run(tau: float) -> float:
        role, b, ctx = _make_droop(tau=tau)
        v = v_open
        seen = []
        for _ in range(80):
            await _tick(role, ctx, v=v)
            v = v_open - k * role._q_filt
            seen.append(v)
        tail = seen[-20:]
        return max(tail) - min(tail)

    assert await _run(0.0) > 0.02, "unlagged droop should limit-cycle"
    assert await _run(3.0) < 1e-3, "tau=3 s should settle the loop"


@pytest.mark.asyncio
async def test_settles_on_the_analytic_equilibrium():
    band = VDE_V_HIGH - VDE_V_DEADBAND_HIGH
    v_open = 1.0372
    gain = 4.25
    k = gain * band / _S_NOM
    role, b, ctx = _make_droop(tau=3.0)
    v = v_open
    for _ in range(200):
        await _tick(role, ctx, v=v)
        v = v_open - k * role._q_filt
    # v = v_open − k·q_max·(v − DB)/band  =>  v(1+gain) = v_open + gain·DB
    v_star = (v_open + gain * VDE_V_DEADBAND_HIGH) / (1.0 + gain)
    assert v == pytest.approx(v_star, abs=1e-4)


@pytest.mark.asyncio
async def test_plateau_does_not_bank_dt():
    """A long cache-gated stretch must not let the next moving tick jump the
    whole ramp — the lag clock advances even on skipped ticks."""
    role, b, ctx = _make_droop(tau=3.0)
    await _tick(role, ctx, v=1.0)  # deadband, q stays 0, then plateau
    for _ in range(40):
        await _tick(role, ctx, v=1.0)
    await _tick(role, ctx, v=1.07)
    assert _last_set_q(b) == pytest.approx(_S_NOM * (_POLL / 3.0), rel=1e-6)


@pytest.mark.asyncio
async def test_fast_attack_clears_over_voltage_in_one_tick():
    role, b, ctx = _make_droop(tau=3.0, attack=0.0)
    await _tick(role, ctx, v=1.07)
    assert _last_set_q(b) == pytest.approx(_S_NOM, abs=1e-4)


@pytest.mark.asyncio
async def test_release_still_lagged_under_fast_attack():
    role, b, ctx = _make_droop(tau=3.0, attack=0.0)
    await _tick(role, ctx, v=1.07)  # attack to q_max
    await _tick(role, ctx, v=1.00)  # deadband: target 0, release is lagged
    assert _last_set_q(b) == pytest.approx(_S_NOM * (1.0 - _POLL / 3.0), rel=1e-6)


@pytest.mark.asyncio
async def test_sign_flip_counts_as_attack():
    """Absorption -> injection reverses the support direction; that is fresh
    demand, so it must not crawl out through the slow release path."""
    role, b, ctx = _make_droop(tau=3.0, attack=0.0)
    await _tick(role, ctx, v=1.07)  # +q_max (absorb)
    await _tick(role, ctx, v=0.90)  # -q_max (inject)
    assert _last_set_q(b) == pytest.approx(-_S_NOM, abs=1e-4)


@pytest.mark.asyncio
async def test_fast_attack_still_kills_the_limit_cycle():
    """The destabilising half is the release, so attack=0 must stay stable at
    the same loop gain that bang-bangs without any lag."""
    band = VDE_V_HIGH - VDE_V_DEADBAND_HIGH
    v_open = 1.0372
    gain = 4.25
    k = gain * band / _S_NOM

    async def _run(tau: float, attack: float | None) -> float:
        role, b, ctx = _make_droop(tau=tau, attack=attack)
        v = v_open
        seen = []
        for _ in range(120):
            await _tick(role, ctx, v=v)
            v = v_open - k * role._q_filt
            seen.append(v)
        tail = seen[-20:]
        return max(tail) - min(tail)

    assert await _run(0.0, None) > 0.02, "unlagged droop should limit-cycle"
    assert await _run(3.0, 0.0) < 1e-3, "fast attack + lagged release is stable"


@pytest.mark.asyncio
async def test_rejects_negative_attack_tau():
    b = MockBehavior()
    with pytest.raises(ValueError, match="attack_tau_s"):
        ReactivePowerDroopRole(b, s_nom_mva=_S_NOM, attack_tau_s=-1.0)


@pytest.mark.asyncio
async def test_rejects_negative_tau():
    b = MockBehavior()
    with pytest.raises(ValueError, match="settling_tau_s"):
        ReactivePowerDroopRole(b, s_nom_mva=_S_NOM, settling_tau_s=-1.0)
    with pytest.raises(ValueError, match="settling_tau_s"):
        ReactivePowerDroopRole(b, s_nom_mva=_S_NOM, settling_tau_s=math.inf)
