"""Tests for the coordinated Q(U)-droop / curtailment-auction hand-off
(``enable_qv_auction_coordination``).

The droop publishes the reactive voltage-relief its unused capability can still
deliver; the auction sheds active power only for the residual (Mechanism A) and
the gen over-voltage curtail-lock hands back once reactive can back-stop a
restore (Mechanism B). See ``project_pv_overvoltage_levers`` memory for why the
naive VVW flag stacked the two levers instead.
"""

from __future__ import annotations

import pytest

from scare.base import runtime  # noqa: F401  (ensure package import)
from scare.base.config import RestorationConfiguration
from scare.base.model import Sector
from scare.base.runtime import diagnostics
from scare.base.util import (
    _QV_LOCK_RESTORE_STEP,
    _gen_curtail_lock_store,
    _last_regulate_store,
    apply_regulate,
    publish_qv_relief,
    qv_relief_avail,
)
from scare.service.control.constraints import GridConstraintMonitor
from scare.service.control.voltage_droop import ReactivePowerDroopRole
from tests.conftest import MockBehavior


class _FakeCtx:
    def __init__(self, aid: str) -> None:
        self.aid = aid
        self.addr = aid
        self.current_timestamp = 1.0


# ---------------------------------------------------------------------------
# Shared relief ledger
# ---------------------------------------------------------------------------


def test_qv_relief_publish_read_and_ttl():
    b = MockBehavior()
    publish_qv_relief(b, "pv", 0.004, now=1.0)
    assert qv_relief_avail(b, "pv", now=1.0) == pytest.approx(0.004)
    # Within TTL.
    assert qv_relief_avail(b, "pv", now=2.5) == pytest.approx(0.004)
    # Past TTL (2.0 s) -> stale -> 0.
    assert qv_relief_avail(b, "pv", now=3.1) == 0.0
    # Unknown aid -> 0.
    assert qv_relief_avail(b, "other", now=1.0) == 0.0
    # Negatives are floored at 0.
    publish_qv_relief(b, "pv", -1.0, now=1.0)
    assert qv_relief_avail(b, "pv", now=1.0) == 0.0


# ---------------------------------------------------------------------------
# Droop publishes relief (Mechanism A / B source)
# ---------------------------------------------------------------------------


def _make_droop(*, vvw: bool, coord: bool):
    b = MockBehavior()
    b.add_action("pv", "set_q")
    b._scare_config = RestorationConfiguration(
        enable_vvw_coordination=vvw, enable_qv_auction_coordination=coord
    )
    role = ReactivePowerDroopRole(b, s_nom_mva=0.1033 / 0.9)
    role._context = _FakeCtx("pv")  # type: ignore[attr-defined]
    return role, b


async def _calibrate(role, b, voltages):
    """Step the droop through a voltage sequence so its |dV/dQ| EMA accumulates
    real (Δv, Δq) samples (q varies across the ramp band)."""
    for v in voltages:
        b.set_obs("pv", {"vm_pu": v, "p_mw": -0.05, "regulation": 1.0})
        await role._step()


@pytest.mark.asyncio
async def test_droop_relief_gated_until_calibrated_then_published():
    # Confidence gate: before the |dV/dQ| EMA has real samples, the droop
    # advertises NO relief (auction = default-safe).  After it calibrates from a
    # voltage excursion (and with headroom remaining), it advertises relief > 0.
    role, b = _make_droop(vvw=True, coord=True)
    b.set_obs("pv", {"vm_pu": 1.04, "p_mw": -0.05, "regulation": 1.0})
    await role._step()  # first tick: no prior sample yet
    assert qv_relief_avail(b, "pv", now=1.0) == 0.0
    # Ramp through the band to gather >= _DVDQ_MIN_SAMPLES samples, ending where
    # the droop still has headroom (v in the ramp band, q < q_max).
    await _calibrate(role, b, [1.035, 1.045, 1.038, 1.043, 1.040])
    assert qv_relief_avail(b, "pv", now=1.0) > 0.0


@pytest.mark.asyncio
async def test_droop_relief_zero_when_saturated():
    # Over-voltage (v=1.07) saturates the droop at q=q_max, so headroom ~0:
    # the measured voltage already reflects full reactive -> ~no extra relief.
    role, b = _make_droop(vvw=True, coord=True)
    b.set_obs("pv", {"vm_pu": 1.07, "p_mw": -0.05, "regulation": 1.0})
    await role._step()
    assert qv_relief_avail(b, "pv", now=1.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.asyncio
async def test_droop_does_not_publish_when_coordination_off():
    role, b = _make_droop(vvw=True, coord=False)
    b.set_obs("pv", {"vm_pu": 1.00, "p_mw": -0.05, "regulation": 1.0})
    await role._step()
    assert qv_relief_avail(b, "pv", now=1.0) == 0.0


# ---------------------------------------------------------------------------
# Mechanism A — auction defers to reactive when it covers the overshoot
# ---------------------------------------------------------------------------


def _make_monitor(*, coord: bool):
    b = MockBehavior()
    b._scare_config = RestorationConfiguration(enable_qv_auction_coordination=coord)
    m = GridConstraintMonitor(
        b, Sector.ELECTRICITY, enable_qv_auction_coordination=coord
    )
    m._context = _FakeCtx("pv")  # type: ignore[attr-defined]
    return m, b


@pytest.mark.asyncio
async def test_auction_defers_when_reactive_covers_overshoot():
    diagnostics.arm()
    m, b = _make_monitor(coord=True)
    # Overshoot = (1.055 - 1.05)/0.1 = 0.05 span-units == 0.005 p.u.; publish
    # relief that fully covers it.
    publish_qv_relief(b, "pv", 0.01, now=1.0)
    before = len(diagnostics.event_log())
    await m._request_curtailment("vm_pu", value=1.055, lo=0.95, hi=1.05)
    new = diagnostics.event_log()[before:]
    # No auction opened; the in-flight slot is cleared so a later poll retries.
    assert m._open_auctions == {}
    assert "vm_pu" not in m._curtail_inflight
    assert any(
        e.kind == "curtail_deferred_to_qv_relief" and e.aid == "pv" for e in new
    )


@pytest.mark.asyncio
async def test_auction_escalates_when_reactive_stalls():
    # Reactive claims to cover it, but the voltage does NOT drop between polls:
    # the auction must stop deferring (outcome-based) and shed active, rather
    # than trust the prediction indefinitely.
    diagnostics.arm()
    m, b = _make_monitor(coord=True)
    publish_qv_relief(b, "pv", 0.01, now=1.0)
    # First poll defers (no history yet).
    await m._request_curtailment("vm_pu", value=1.055, lo=0.95, hi=1.05)
    before = len(diagnostics.event_log())
    # Same voltage next poll = no progress -> escalate (proceeds into the
    # neighbour lookup that needs a real context -> AttributeError).
    with pytest.raises(AttributeError):
        await m._request_curtailment("vm_pu", value=1.055, lo=0.95, hi=1.05)
    new = diagnostics.event_log()[before:]
    assert any(e.kind == "curtail_qv_defer_escalated" and e.aid == "pv" for e in new)


@pytest.mark.asyncio
async def test_auction_keeps_deferring_while_voltage_drops():
    # While the droop is measurably pulling the voltage down, the auction keeps
    # deferring (no premature active shed).
    diagnostics.arm()
    m, b = _make_monitor(coord=True)
    publish_qv_relief(b, "pv", 0.01, now=1.0)
    for v in (1.058, 1.056, 1.054, 1.052):
        before = len(diagnostics.event_log())
        await m._request_curtailment("vm_pu", value=v, lo=0.95, hi=1.05)
        new = diagnostics.event_log()[before:]
        assert any(e.kind == "curtail_deferred_to_qv_relief" for e in new)
        assert not any(e.kind == "curtail_qv_defer_escalated" for e in new)


@pytest.mark.asyncio
async def test_auction_does_not_defer_without_coordination():
    diagnostics.arm()
    m, b = _make_monitor(coord=False)
    publish_qv_relief(b, "pv", 0.01, now=1.0)
    before = len(diagnostics.event_log())
    # With the flag off the discount is skipped, so the auction proceeds past it
    # into the neighbour lookup (which needs a real mango context here, hence the
    # AttributeError).  Reaching that point proves it did NOT take the reactive
    # hand-off.
    with pytest.raises(AttributeError):
        await m._request_curtailment("vm_pu", value=1.055, lo=0.95, hi=1.05)
    new = diagnostics.event_log()[before:]
    assert not any(e.kind == "curtail_deferred_to_qv_relief" for e in new)


# ---------------------------------------------------------------------------
# Mechanism B — gen curtail-lock hands back when reactive has headroom
# ---------------------------------------------------------------------------


def _locked_behavior(*, coord: bool) -> MockBehavior:
    b = MockBehavior()
    b.add_action("pv", "regulate")
    b._scare_config = RestorationConfiguration(
        enable_curtail_ramp_interlock=True,
        enable_qv_auction_coordination=coord,
    )
    # Auction holds a fresh over-voltage curtail-lock on the generator.
    _gen_curtail_lock_store(b)["pv"] = 1.0
    return b


def test_gen_lock_hands_back_one_bounded_step_when_in_band():
    b = _locked_behavior(coord=True)
    # Reactive holding the node genuinely in-band (v <= 1.03) with headroom.
    publish_qv_relief(b, "pv", 0.005, now=1.5, v_pu=1.02)
    applied = apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="self_local_gen", timestamp=1.5
    )
    assert applied is True
    # Bounded ramp step, NOT a jump to 1.0 (current was 0.0).
    assert _last_regulate_store(b)["pv"] == pytest.approx(_QV_LOCK_RESTORE_STEP)
    # Lock kept fresh so the closed-loop ramp continues next cycle.
    assert "pv" in _gen_curtail_lock_store(b)


def test_gen_lock_fully_released_when_step_reaches_full():
    b = _locked_behavior(coord=True)
    _last_regulate_store(b)["pv"] = 0.95  # already near full
    publish_qv_relief(b, "pv", 0.005, now=1.5, v_pu=1.02)
    applied = apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="self_local_gen", timestamp=1.5
    )
    assert applied is True
    assert _last_regulate_store(b)["pv"] == pytest.approx(1.0)
    # Fully restored -> lock dropped.
    assert "pv" not in _gen_curtail_lock_store(b)


def test_gen_lock_holds_when_voltage_still_elevated():
    # Headroom exists but the node still sits in the Q(U) ramp band (v=1.045):
    # restoring active there risks re-breaching -> keep deferring.
    b = _locked_behavior(coord=True)
    publish_qv_relief(b, "pv", 0.005, now=1.5, v_pu=1.045)
    applied = apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="self_local_gen", timestamp=1.5
    )
    assert applied is False
    assert "pv" in _gen_curtail_lock_store(b)


def test_gen_lock_holds_when_reactive_saturated():
    b = _locked_behavior(coord=True)
    publish_qv_relief(b, "pv", 0.0, now=1.5)  # no headroom -> keep deferring
    applied = apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="self_local_gen", timestamp=1.5
    )
    assert applied is False
    assert "pv" in _gen_curtail_lock_store(b)


def test_gen_lock_holds_without_coordination():
    b = _locked_behavior(coord=False)
    publish_qv_relief(b, "pv", 0.05, now=1.5)  # plenty of relief, but flag off
    applied = apply_regulate(
        b, "pv", 1.0, sector="electricity", reason="self_local_gen", timestamp=1.5
    )
    assert applied is False
    assert "pv" in _gen_curtail_lock_store(b)
