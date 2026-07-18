"""Line-relief lock coordinated hand-off (Mechanism B for lines).

The line-relief auction sheds a downstream load to relieve an overloaded
branch and takes a lock so L2 can't immediately re-serve it. Legacy behaviour
hard-deferred every L2 restore until the lock aged out, then let L2 slam the
load back to full — a relaxation limit cycle that leaves loading oscillating.
The hand-off instead ramps the load back in bounded steps, gated each tick on
the branch having fresh loading headroom below the limit.
"""

from dataclasses import replace

from scare.base.config import RestorationConfiguration
from scare.base.model import Sector
from scare.base.util import (
    _LINE_RELIEF_HANDOFF_HEADROOM_PCT,
    _LINE_RELIEF_RESTORE_STEP,
    CURTAIL_AUCTION_REASON,
    apply_regulate,
    last_actuated_factor,
    publish_line_relief_headroom,
)
from tests.conftest import MockBehavior

AID = "load-1"  # not a "child-" id: bypasses slack / heat-sink guards
RESTORE = "holon_tier_alloc"  # an L2_ALLOCATION_REASON


def _behavior(**cfg_overrides):
    b = MockBehavior()
    b.add_action(AID, "regulate")
    base = dict(
        enable_branch_downstream_relief=True,
        enable_l2_priority_floor=False,
        cooldown_s=0.0,
    )
    base.update(cfg_overrides)
    b._scare_config = replace(RestorationConfiguration(), **base)
    return b


def _shed_and_lock(b, t=1.0, factor=0.4):
    """Auction sheds the load and takes the line-relief lock."""
    applied = apply_regulate(
        b,
        AID,
        factor,
        sector=Sector.ELECTRICITY,
        reason=CURTAIL_AUCTION_REASON,
        timestamp=t,
    )
    assert applied
    assert last_actuated_factor(b, AID) == factor


def test_restore_deferred_without_headroom():
    b = _behavior()
    _shed_and_lock(b, t=1.0, factor=0.4)
    # No headroom published → the line has no room; L2 restore must defer.
    applied = apply_regulate(
        b, AID, 1.0, sector=Sector.ELECTRICITY, reason=RESTORE, timestamp=1.2
    )
    assert applied is False
    assert last_actuated_factor(b, AID) == 0.4


def test_restore_deferred_when_headroom_below_margin():
    b = _behavior()
    _shed_and_lock(b, t=1.0, factor=0.4)
    publish_line_relief_headroom(b, AID, _LINE_RELIEF_HANDOFF_HEADROOM_PCT - 1.0, 1.2)
    applied = apply_regulate(
        b, AID, 1.0, sector=Sector.ELECTRICITY, reason=RESTORE, timestamp=1.2
    )
    assert applied is False
    assert last_actuated_factor(b, AID) == 0.4


def test_restore_ramps_in_bounded_steps_with_headroom():
    b = _behavior()
    _shed_and_lock(b, t=1.0, factor=0.4)
    # Ample headroom → hand back one bounded step per call, not a full slam.
    t = 1.1
    expected = 0.4
    for _ in range(3):
        publish_line_relief_headroom(b, AID, 25.0, t)
        applied = apply_regulate(
            b, AID, 1.0, sector=Sector.ELECTRICITY, reason=RESTORE, timestamp=t
        )
        assert applied is True
        expected = min(1.0, expected + _LINE_RELIEF_RESTORE_STEP)
        assert abs(last_actuated_factor(b, AID) - expected) < 1e-9
        t += 0.1
    # Never jumped straight to 1.0.
    assert last_actuated_factor(b, AID) < 1.0


def test_ramp_completes_and_drops_lock():
    b = _behavior()
    _shed_and_lock(b, t=1.0, factor=0.4)
    t = 1.1
    for _ in range(20):
        publish_line_relief_headroom(b, AID, 25.0, t)
        apply_regulate(
            b, AID, 1.0, sector=Sector.ELECTRICITY, reason=RESTORE, timestamp=t
        )
        t += 0.1
    assert abs(last_actuated_factor(b, AID) - 1.0) < 1e-9
    # Lock dropped once fully restored → a later restore is a no-op dedup, not a defer.
    assert "_scare_line_curtail_lock" in vars(b)
    assert AID not in b._scare_line_curtail_lock


def test_further_shed_always_passes_while_locked():
    b = _behavior()
    _shed_and_lock(b, t=1.0, factor=0.4)
    # A deeper shed (below where we sit) is not a claw-back; it must apply even
    # with no headroom, since it only relieves the line further.
    applied = apply_regulate(
        b, AID, 0.2, sector=Sector.ELECTRICITY, reason=RESTORE, timestamp=1.2
    )
    assert applied is True
    assert last_actuated_factor(b, AID) == 0.2
