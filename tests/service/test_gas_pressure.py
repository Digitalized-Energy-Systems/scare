"""Unit tests for the layer-0 gas pressure regulator control law."""

import asyncio

import pytest

from scare.base.model import SECTOR_CONSTRAINTS, Sector
from scare.base.runtime import diagnostics
from scare.base.util import lookup_slack_pressure
from scare.service.control.gas_pressure import GasPressureRegulator

LO, HI = SECTOR_CONSTRAINTS[Sector.GAS]["pressure_pu"]


class _FakeBehavior:
    """Minimal behavior: observe returns the slack node pressure (= setpoint),
    ``set_pressure`` re-pins it, mirroring ``ExtHydrGrid.overwrite``."""

    def __init__(self, own_pressure: float) -> None:
        self.own_pressure = float(own_pressure)
        self.acted: list[tuple[str, float]] = []

    def observe(self, aid):
        return {"pressure_pu": self.own_pressure}

    def has_action(self, aid, name):
        return name == "set_pressure"

    def act(self, aid, name, value):
        self.acted.append((name, value))
        self.own_pressure = float(value)


class _FakeContext:
    def __init__(self, aid="child-gasslack", t=10.0) -> None:
        self.aid = aid
        self.current_timestamp = t


def _make(own_p: float, reports: dict[str, float], gain: float = 0.5):
    beh = _FakeBehavior(own_p)
    reg = GasPressureRegulator(beh, Sector.GAS, gain=gain)
    reg._context = _FakeContext()
    reg._freshness_s = 100.0
    # Inject downstream mesh reports as if freshly received.
    reg._reports = {k: (v, reg.context.current_timestamp) for k, v in reports.items()}
    return reg, beh


def _run(reg):
    asyncio.run(reg._control())


def test_underpressure_raises_setpoint_within_band():
    # Worst downstream node at 0.80 (below LO=0.85); source/top at 1.00.
    reg, beh = _make(own_p=1.00, reports={"n1": 0.80, "n2": 0.95})
    _run(reg)
    sp = lookup_slack_pressure(beh, reg.context.aid)
    assert sp is not None and sp > 1.00  # raised
    # need = LO-0.80 = 0.05, headroom = HI-1.00 = 0.25 -> delta=0.05, gain 0.5.
    assert sp == pytest.approx(1.00 + 0.5 * (LO - 0.80))
    assert beh.acted and beh.acted[-1][0] == "set_pressure"


def test_overpressure_lowers_setpoint():
    # Low demand lifted the whole profile above HI=1.25; regulator lowers the
    # setpoint back toward the band. need=0.05, headroom=p_min-LO large, so the
    # feedback step 1.30-0.025=1.275 clamps to the band ceiling HI.
    reg, beh = _make(own_p=1.30, reports={"n1": 1.30, "n2": 1.10})
    _run(reg)
    sp = lookup_slack_pressure(beh, reg.context.aid)
    assert sp is not None and sp < 1.30
    assert sp == pytest.approx(HI)  # clamped to band ceiling


def test_overpressure_gain_bounded_when_unclamped():
    # Source at 1.20 (in band) with a downstream node at 1.28 (> HI): lower by
    # the feedback-gained, headroom-bounded delta without hitting the clamp.
    reg, beh = _make(own_p=1.20, reports={"n1": 1.28})
    _run(reg)
    sp = lookup_slack_pressure(beh, reg.context.aid)
    # need=1.28-HI=0.03, headroom=p_min-LO=0.35 -> delta=-0.03, gain 0.5.
    assert sp == pytest.approx(1.20 - 0.5 * (1.28 - HI))


def test_spread_exceeds_band_saturates_and_flags():
    diagnostics.arm()
    try:
        # spread from 0.80 (min) to source 1.25 = 0.45 > band width (HI-LO=0.40).
        reg, beh = _make(own_p=1.25, reports={"n1": 0.80})
        _run(reg)
        kinds = {e.kind for e in diagnostics.event_log()}
        assert "gas_pressure_setpoint_saturated" in kinds
        # Still moves the setpoint by the (zero) top headroom-bounded amount:
        # headroom = HI-1.25 = 0 -> no raise, but saturation surfaced.
        sp = lookup_slack_pressure(beh, reg.context.aid)
        assert sp is None or sp == pytest.approx(1.25)
    finally:
        diagnostics._RECORDER._armed = False


def test_in_band_relaxes_toward_nominal():
    # Comfortably in band, setpoint elevated at 1.20 -> relax down toward 1.0.
    reg, beh = _make(own_p=1.20, reports={"n1": 1.05, "n2": 1.08})
    # own_p 1.20 would be p_max; ensure it's comfortably below HI with margin.
    _run(reg)
    sp = lookup_slack_pressure(beh, reg.context.aid)
    assert sp is not None and sp < 1.20  # relaxed downward toward nominal 1.0


def test_deenergised_reports_ignored():
    # A source-isolated node at ~0 must not trigger a raise.
    reg, beh = _make(own_p=1.00, reports={"dead": 0.0, "n2": 1.00})
    _run(reg)
    # p_min over energised = 1.00 (in band), no actuation.
    assert not beh.acted


def test_no_reports_no_action():
    reg, beh = _make(own_p=1.00, reports={})
    _run(reg)
    assert not beh.acted


def test_stale_reports_dropped():
    # A severe under-pressure report that is older than the freshness window
    # must be ignored (post-failure topology may have orphaned its origin).
    reg, beh = _make(own_p=1.00, reports={"n2": 1.00})
    reg._freshness_s = 5.0
    # Stamp a 0.70-pu report 100 s in the past — well outside the window.
    reg._reports["stale"] = (0.70, reg.context.current_timestamp - 100.0)
    _run(reg)
    assert not beh.acted  # only the fresh in-band 1.00 counts -> no raise


def test_sqrt3_phantom_overpressure_ignored():
    # A zero-flow / P2G junction can saturate monee's relaxed-Weymouth
    # pressure_squared_pu box, reading pressure_pu ~ sqrt(3) ~ 1.732. The
    # regulator must treat it as a de-energised artifact (like the ~0 collapse)
    # and NOT chase it down — otherwise it walks the whole profile to the floor.
    reg, beh = _make(own_p=1.00, reports={"p2g": 3.0**0.5, "n2": 0.95})
    _run(reg)
    # Energised profile = {1.00, 0.95}: in band -> no over-pressure walk-down.
    assert not beh.acted


def test_real_overpressure_below_artifact_still_acts():
    # A genuine over-pressure (1.60 < the sqrt(3) artifact threshold) must still
    # be acted on — the artifact guard only drops the saturated ~1.732 reading.
    reg, beh = _make(own_p=1.20, reports={"n1": 1.60})
    _run(reg)
    sp = lookup_slack_pressure(beh, reg.context.aid)
    assert sp is not None and sp < 1.20  # lowered toward band


def test_overpressure_trap_not_blamed_on_shedding():
    # Spread exceeds band on the HIGH side: p_max=1.30>HI with p_min already at
    # LO. The saturated event must NOT claim shedding clears it.
    diagnostics.arm()
    try:
        reg, beh = _make(own_p=1.30, reports={"n1": 1.30, "n2": 0.85})
        _run(reg)
        traps = [
            e
            for e in diagnostics.event_log()
            if e.kind == "gas_pressure_overpressure_trap"
        ]
        assert traps, "over-pressure trap event expected"
        assert "not relievable by shedding" in traps[-1].detail
    finally:
        diagnostics._RECORDER._armed = False
