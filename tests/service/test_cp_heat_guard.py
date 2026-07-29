"""Unit tests for the CP heat-outlet guard: AIMD control law, born-state
self-actuation, and the apply_regulate ceiling chokepoint."""

import asyncio
from types import SimpleNamespace

import pytest

from scare.base.model import SECTOR_CONSTRAINTS, Sector
from scare.base.util import (
    apply_regulate,
    cp_heat_ceiling,
    publish_cp_heat_ceiling,
)
from scare.service.control.cp_heat_guard import (
    _CEILING_FLOOR,
    _DECREASE_FACTOR,
    _HI_MARGIN_K,
    _RECOVER_BAND_K,
    _RECOVER_STEP,
    CPHeatOutletGuard,
)

_LO, _HI = SECTOR_CONSTRAINTS[Sector.HEAT]["t_k"]
CP_AID = "branch-371-114"
OUTLET_AID = "node-371"

HOT = _HI + 50.0  # deep over-temperature (born-hot CP outlet)
WARM = _HI - _HI_MARGIN_K - 2.0  # inside the hysteresis band
COOL = _HI - _RECOVER_BAND_K - 5.0  # clear of the band -> recovery


class _FakeBehavior:
    """observe() serves the outlet t_k and the CP's own regulation; act()
    records regulates and feeds them back as the standing regulation."""

    def __init__(self, t_k: float, regulation: float = 1.0) -> None:
        self.t_k = float(t_k)
        self.regulation = float(regulation)
        self.acted: list[float] = []

    def observe(self, aid):
        if aid == OUTLET_AID:
            return {"t_k": self.t_k}
        if aid == CP_AID:
            return {"regulation": self.regulation}
        return {}

    def has_action(self, aid, name):
        return name == "regulate"

    def act(self, aid, name, value):
        assert name == "regulate"
        self.acted.append(float(value))
        self.regulation = float(value)


class _FakeContext:
    def __init__(self, aid=CP_AID, t=0.0) -> None:
        self.aid = aid
        self.current_timestamp = t


def _make(t_k: float, regulation: float = 1.0):
    beh = _FakeBehavior(t_k, regulation)
    guard = CPHeatOutletGuard(beh, outlet_aid=OUTLET_AID)
    guard._context = _FakeContext()
    return guard, beh


def _tick(guard, n: int = 1):
    for _ in range(n):
        guard.context.current_timestamp += 1.0
        asyncio.run(guard._control())


def test_born_hot_outlet_wound_down_geometrically():
    guard, beh = _make(t_k=HOT, regulation=1.0)
    _tick(guard)
    now = guard.context.current_timestamp
    assert cp_heat_ceiling(beh, CP_AID, now) == pytest.approx(_DECREASE_FACTOR)
    # Born state: no kernel commit exists, the guard itself actuated.
    assert beh.acted == [pytest.approx(_DECREASE_FACTOR)]
    _tick(guard)
    assert beh.acted[-1] == pytest.approx(_DECREASE_FACTOR**2)
    # Persistently hot -> geometric decrease snaps to a full cut at the floor.
    _tick(guard, 10)
    assert beh.acted[-1] == 0.0
    assert cp_heat_ceiling(beh, CP_AID, guard.context.current_timestamp) == 0.0


def test_hysteresis_band_holds_and_keeps_entry_fresh():
    guard, beh = _make(t_k=HOT)
    _tick(guard)  # ceiling -> 0.5
    beh.t_k = WARM  # inside (hi - RECOVER_BAND, hi - HI_MARGIN): hold
    _tick(guard, 6)  # > TTL worth of ticks; republish must keep it fresh
    now = guard.context.current_timestamp
    assert cp_heat_ceiling(beh, CP_AID, now) == pytest.approx(_DECREASE_FACTOR)


def test_recovery_is_additive_and_releases_at_full():
    guard, beh = _make(t_k=HOT)
    _tick(guard, 2)  # ceiling -> 0.25, guard self-actuated the plant to match
    beh.t_k = COOL
    _tick(guard)
    now = guard.context.current_timestamp
    assert cp_heat_ceiling(beh, CP_AID, now) == pytest.approx(
        _DECREASE_FACTOR**2 + _RECOVER_STEP
    )
    # The plant follows the released ceiling (a kernel commit landing at the
    # cap), so each step is observed before the next is granted.
    for _ in range(20):
        beh.regulation = min(1.0, guard._ceiling)
        _tick(guard)
    now = guard.context.current_timestamp
    assert cp_heat_ceiling(beh, CP_AID, now) is None


def test_recovery_holds_while_plant_parked_below_ceiling():
    """The regression: a cool outlet is NOT evidence the ceiling is safe when
    another writer holds the CP below it. Recovery must not run open-loop."""
    guard, beh = _make(t_k=HOT)
    _tick(guard, 2)  # ceiling -> 0.25
    beh.t_k = COOL
    beh.regulation = 0.02  # L3 commits ~0; the guard never writes UP
    _tick(guard, 20)
    now = guard.context.current_timestamp
    # Frozen where the wind-down left it: reg + step sits below the standing
    # ceiling, and the never-tighten clamp holds it there.
    assert cp_heat_ceiling(beh, CP_AID, now) == pytest.approx(_DECREASE_FACTOR**2)


def test_plant_tracking_off_restores_open_loop_ramp():
    beh = _FakeBehavior(t_k=HOT, regulation=1.0)
    guard = CPHeatOutletGuard(beh, outlet_aid=OUTLET_AID, plant_tracking=False)
    guard._context = _FakeContext()
    _tick(guard, 2)
    beh.t_k = COOL
    beh.regulation = 0.02
    _tick(guard, 20)
    now = guard.context.current_timestamp
    assert cp_heat_ceiling(beh, CP_AID, now) is None


def test_plant_tracking_never_tightens_the_ceiling():
    """A parked plant degrades recovery to a HOLD; only the hot branch cuts."""
    guard, beh = _make(t_k=HOT)
    _tick(guard)  # ceiling -> 0.5
    beh.t_k = COOL
    beh.regulation = 0.0
    _tick(guard, 5)
    now = guard.context.current_timestamp
    assert cp_heat_ceiling(beh, CP_AID, now) == pytest.approx(_DECREASE_FACTOR)


def test_deenergised_outlet_holds_ceiling_without_tightening():
    guard, beh = _make(t_k=HOT)
    _tick(guard)  # ceiling -> 0.5
    beh.t_k = 0.0  # de-energised artifact reading
    _tick(guard, 6)
    now = guard.context.current_timestamp
    # Neither wound further down nor TTL-released: held and refreshed.
    assert cp_heat_ceiling(beh, CP_AID, now) == pytest.approx(_DECREASE_FACTOR)
    assert beh.acted == [pytest.approx(_DECREASE_FACTOR)]


def test_no_self_actuation_below_ceiling():
    guard, beh = _make(t_k=HOT, regulation=0.1)
    _tick(guard)  # ceiling 0.5 > standing 0.1
    assert beh.acted == []


def test_apply_regulate_clamps_cp_writes_to_ceiling():
    beh = _FakeBehavior(t_k=HOT)
    publish_cp_heat_ceiling(beh, CP_AID, 0.3, now=10.0)
    applied = apply_regulate(
        beh, CP_AID, 1.0, sector="cp", reason="cp_priority_admm", timestamp=10.5
    )
    assert applied and beh.acted[-1] == pytest.approx(0.3)
    # Ratchet fight: the kernel re-commits 1.0; the clamp holds, dedup absorbs.
    applied = apply_regulate(
        beh, CP_AID, 1.0, sector="cp", reason="cp_priority_admm", timestamp=11.0
    )
    assert not applied
    assert beh.acted == [pytest.approx(0.3)]


def test_apply_regulate_ignores_stale_ceiling():
    beh = _FakeBehavior(t_k=HOT)
    publish_cp_heat_ceiling(beh, CP_AID, 0.3, now=0.0)
    applied = apply_regulate(
        beh, CP_AID, 1.0, sector="cp", reason="cp_priority_admm", timestamp=20.0
    )
    assert applied and beh.acted[-1] == pytest.approx(1.0)


def test_apply_regulate_non_cp_sector_not_clamped():
    beh = _FakeBehavior(t_k=HOT)
    publish_cp_heat_ceiling(beh, CP_AID, 0.3, now=10.0)
    applied = apply_regulate(
        beh,
        CP_AID,
        1.0,
        sector="electricity",
        reason="holon_supply_priority",
        timestamp=10.5,
    )
    assert applied and beh.acted[-1] == pytest.approx(1.0)


def _slew_step() -> float:
    from scare.base.config import RestorationConfiguration

    return RestorationConfiguration().cp_regulate_slew_step


def test_cp_kernel_commit_is_slew_limited_symmetrically():
    beh = _FakeBehavior(t_k=COOL)
    step = _slew_step()
    # Cold start passes unbounded: a born CP must take its first dispatch.
    apply_regulate(
        beh, CP_AID, 0.5, sector="cp", reason="cp_priority_admm", timestamp=1.0
    )
    assert beh.acted[-1] == pytest.approx(0.5)
    # Up and down are bounded by the same step — a one-sided bound would
    # ratchet the CP to zero under the kernel's alternating commits.
    apply_regulate(
        beh, CP_AID, 1.0, sector="cp", reason="cp_priority_admm", timestamp=2.0
    )
    assert beh.acted[-1] == pytest.approx(0.5 + step)
    apply_regulate(
        beh, CP_AID, 0.0, sector="cp", reason="cp_priority_admm", timestamp=3.0
    )
    assert beh.acted[-1] == pytest.approx(0.5)


def test_slew_limit_preserves_the_fixed_point():
    """Repeated identical commits still land exactly on the request (an EMA
    would only approach it, breaking the >= 1.0 ceiling release)."""
    beh = _FakeBehavior(t_k=COOL)
    for i in range(12):
        apply_regulate(
            beh, CP_AID, 1.0, sector="cp", reason="cp_priority_admm", timestamp=float(i)
        )
    assert beh.acted[-1] == pytest.approx(1.0)


def test_guard_relief_bypasses_the_slew_limit():
    """A hot outlet needs the full geometric cut on the poll that sees it."""
    beh = _FakeBehavior(t_k=HOT)
    apply_regulate(
        beh, CP_AID, 1.0, sector="cp", reason="cp_priority_admm", timestamp=1.0
    )
    apply_regulate(
        beh, CP_AID, 0.0, sector="cp", reason="cp_heat_outlet_relief", timestamp=2.0
    )
    assert beh.acted[-1] == pytest.approx(0.0)


def test_floor_constant_below_first_geometric_steps():
    # Guard invariant: the snap floor must sit below DECREASE^2 so the first
    # two wind-down steps are genuinely geometric, not an immediate full cut.
    assert _CEILING_FLOOR < _DECREASE_FACTOR**2


def test_heat_outlet_aid_resolution():
    from scare.scenario.restoration import _heat_outlet_aid_for_node

    class PowerToHeatControlNode:  # name matched via _model_type_name
        pass

    sub = SimpleNamespace()
    chphg_model = SimpleNamespace(_sub_hg=sub)
    net = SimpleNamespace(childs=[SimpleNamespace(model=sub, node_id=42)])

    chp_node = SimpleNamespace(model=chphg_model, id=7)
    assert _heat_outlet_aid_for_node(chp_node, net) == "node-42"

    hx_node = SimpleNamespace(model=PowerToHeatControlNode(), id=9)
    assert _heat_outlet_aid_for_node(hx_node, net) == "node-9"

    class Bus:
        pass

    plain = SimpleNamespace(model=Bus(), id=3)
    assert _heat_outlet_aid_for_node(plain, net) is None
