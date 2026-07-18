"""ConstraintSignalListener proactive-utilization age-out.

``ConstraintWarning`` is emitted only above ``PROACTIVE_WARNING_FRACTION`` and
the recovery path emits nothing, so a recorded entry -- always in [0.85, 1.0] --
throttles the agent for the rest of the run unless it expires. The latch is the
shipped default; ``proactive_util_ttl_s > 0`` opts into clearing it.
"""

from __future__ import annotations

from types import SimpleNamespace

from scare.base.model import PROACTIVE_WARNING_FRACTION, ConstraintWarning, Sector
from scare.service.balance.balance import ConstraintSignalListener


class _Ctx:
    def __init__(self) -> None:
        self.current_timestamp = 0.0
        self.aid = "child-1"

    def get_role(self, _cls):
        return None


def _throttle(ttl: float):
    ctx = _Ctx()
    role = SimpleNamespace(
        sector=Sector.ELECTRICITY,
        constraint_aware=True,
        proactive_util_ttl_s=ttl,
        context=ctx,
    )
    return ConstraintSignalListener(role), ctx


def _warn(util: float) -> ConstraintWarning:
    return ConstraintWarning(
        sector=Sector.ELECTRICITY,
        variable="vm_pu",
        value=0.94,
        bound_low=0.95,
        bound_high=1.05,
        utilization=util,
        node_id=1,
    )


def test_warning_throttles_participation():
    throttle, _ = _throttle(ttl=0.0)
    throttle.record_warning(_warn(0.9))
    assert throttle.compute_participation_scale({}) == 1.0 - 0.9


def test_default_latches_forever():
    """Shipped behaviour: the throttle never lifts, however much time passes."""
    throttle, ctx = _throttle(ttl=0.0)
    throttle.record_warning(_warn(0.95))
    ctx.current_timestamp = 10_000.0
    assert throttle.compute_participation_scale({}) == 1.0 - 0.95


def test_ttl_expires_the_entry():
    throttle, ctx = _throttle(ttl=2.0)
    throttle.record_warning(_warn(0.95))
    ctx.current_timestamp = 1.0
    assert throttle.compute_participation_scale({}) == 1.0 - 0.95, "still inside TTL"
    ctx.current_timestamp = 5.0
    assert throttle.compute_participation_scale({}) == 1.0, "expired -> unthrottled"


def test_every_recorded_warning_is_above_the_threshold():
    """Why the latch bites: no entry can ever be small."""
    assert PROACTIVE_WARNING_FRACTION == 0.85
    throttle, _ = _throttle(ttl=0.0)
    throttle.record_warning(_warn(PROACTIVE_WARNING_FRACTION))
    scale = throttle.compute_participation_scale({})
    assert scale <= 1.0 - PROACTIVE_WARNING_FRACTION
