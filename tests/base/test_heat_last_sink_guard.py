"""Unit tests for the R5 heat last-sink guard.

A junction's sole HeatLoad must keep absorbing the co-located fixed
HeatGenerator injection; ``apply_regulate`` clamps any shed below the
registered floor (``enable_heat_last_sink_guard``). See the node-365
over-temperature root cause: shedding the only sink removed the junction's
cooling draw while the non-actuatable generator kept injecting.
"""

from __future__ import annotations

from scare.base.util import (
    apply_regulate,
    heat_last_sink_floor,
    register_heat_last_sink_floor,
)


class _Behavior:
    def __init__(self, config=None):
        self._scare_config = config
        self.acted: list[tuple[str, float]] = []
        self._net_results = object()

    def observe(self, aid):
        return {"q_mw_heat": 0.005, "regulation": 1.0}

    def has_action(self, aid, action):
        return True

    def act(self, aid, action, factor):
        self.acted.append((aid, factor))


class _Cfg:
    def __init__(self, guard=True):
        self.enable_heat_last_sink_guard = guard
        self.enable_heat_curtail_lock = False
        self.cooldown_s = 0.0


def test_registry_roundtrip_and_clamp01():
    b = _Behavior(_Cfg())
    register_heat_last_sink_floor(b, "child-462", 1.7)
    assert heat_last_sink_floor(b, "child-462") == 1.0
    register_heat_last_sink_floor(b, "child-1", -0.3)
    assert heat_last_sink_floor(b, "child-1") == 0.0
    assert heat_last_sink_floor(b, "child-999") is None


def test_shed_below_floor_is_clamped_up():
    b = _Behavior(_Cfg(guard=True))
    register_heat_last_sink_floor(b, "child-462", 1.0)
    applied = apply_regulate(
        b,
        "child-462",
        0.0,
        sector="heat",
        reason="holon_supply_priority",
        timestamp=1.0,
    )
    assert applied
    assert b.acted == [("child-462", 1.0)]


def test_partial_floor_allows_shed_down_to_floor():
    b = _Behavior(_Cfg(guard=True))
    register_heat_last_sink_floor(b, "child-7", 0.4)
    apply_regulate(
        b,
        "child-7",
        0.1,
        sector="heat",
        reason="holon_supply_priority",
        timestamp=1.0,
    )
    assert b.acted == [("child-7", 0.4)]
    b.acted.clear()
    # At/above the floor passes through untouched.
    apply_regulate(
        b,
        "child-7",
        0.7,
        sector="heat",
        reason="holon_supply_priority",
        timestamp=2.0,
    )
    assert b.acted == [("child-7", 0.7)]


def test_flag_off_is_inert():
    b = _Behavior(_Cfg(guard=False))
    register_heat_last_sink_floor(b, "child-462", 1.0)
    apply_regulate(
        b,
        "child-462",
        0.0,
        sector="heat",
        reason="holon_supply_priority",
        timestamp=1.0,
    )
    assert b.acted == [("child-462", 0.0)]


def test_unregistered_load_unaffected():
    b = _Behavior(_Cfg(guard=True))
    apply_regulate(
        b,
        "child-5",
        0.0,
        sector="heat",
        reason="holon_supply_priority",
        timestamp=1.0,
    )
    assert b.acted == [("child-5", 0.0)]


def test_non_heat_sector_unaffected():
    b = _Behavior(_Cfg(guard=True))
    register_heat_last_sink_floor(b, "child-462", 1.0)
    apply_regulate(
        b,
        "child-462",
        0.0,
        sector="electricity",
        reason="balance",
        timestamp=1.0,
    )
    assert b.acted == [("child-462", 0.0)]


def test_default_config_leaves_guard_off():
    """An attr-less legacy config (and None) must resolve the flag to the
    DECLARED default (False) via cfg_value — pins the dataclass default."""

    class _Bare:
        cooldown_s = 0.0
        enable_heat_curtail_lock = False

    for cfg in (None, _Bare()):
        b = _Behavior(cfg)
        register_heat_last_sink_floor(b, "child-462", 1.0)
        apply_regulate(
            b,
            "child-462",
            0.0,
            sector="heat",
            reason="holon_supply_priority",
            timestamp=1.0,
        )
        assert b.acted == [("child-462", 0.0)], cfg


def test_clamp_runs_before_heat_curtail_lock():
    """The floor clamp must precede the curtail-lock block: a locked load's
    clamped-up factor is an L2 restore the lock DEFERS (return False), never
    a bypass that writes below the floor."""

    class _CfgLock(_Cfg):
        def __init__(self):
            super().__init__(guard=True)
            self.enable_heat_curtail_lock = True

    b = _Behavior(_CfgLock())
    register_heat_last_sink_floor(b, "child-9", 0.8)
    # Auction locks the load down first (curtail reason sets the lock).
    apply_regulate(b, "child-9", 0.3, sector="heat", reason="curtail", timestamp=1.0)
    assert b.acted == [("child-9", 0.8)]  # even the auction respects the floor
    b.acted.clear()
    # An L2 write below the floor is clamped UP to it; the lock sees a
    # non-restore (equal factor) and lets it through or defers — but never
    # below the floor.
    apply_regulate(
        b,
        "child-9",
        0.0,
        sector="heat",
        reason="holon_supply_priority",
        timestamp=2.0,
    )
    assert all(f >= 0.8 for _, f in b.acted), b.acted


def test_l2_reassert_reason_is_l2_allocation_class():
    from scare.base.util.blackboard import L2_ALLOCATION_REASONS

    assert "l2_reassert" in L2_ALLOCATION_REASONS
