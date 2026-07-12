"""Phase 2 slack-budget fix: feedback target shift so the settle band's top
edge is B (a tol margin below the compliance claim's B*(1+tol) threshold,
which the old target B put the band edge coincident with)."""

from __future__ import annotations

import asyncio

import pytest

from scare.base.model import Sector
from scare.base.util import lookup_slack_eff_budget
from scare.service.control.slack_budget import (
    _FEEDBACK_TARGET_MARGIN,
    SlackBudgetMonitor,
)

BUDGET = 0.20
OBS_KEY = "p_mw"


class _Behavior:
    """Slack draw source + attribute-store host (set_slack_* write onto self)."""

    def __init__(self, draw: float) -> None:
        self.draw = float(draw)

    def observe(self, aid):
        return {OBS_KEY: self.draw}

    def has_action(self, aid, name):
        return True

    def act(self, aid, name, value):
        pass


class _Ctx:
    def __init__(self, aid="child-118", t=1.0) -> None:
        self.aid = aid
        self.addr = aid
        self.current_timestamp = float(t)
        self.sent: list = []

    async def send_message(self, content, receiver_addr=None):
        self.sent.append((content, receiver_addr))

    def emit_event(self, event):
        raise KeyError("no subscriber")

    def schedule_periodic_task(self, *a, **k):
        pass


def _mon(draw: float):
    beh = _Behavior(draw)
    mon = SlackBudgetMonitor(
        beh, Sector.ELECTRICITY, obs_key=OBS_KEY, budget=BUDGET, tol=0.05
    )
    mon._context = _Ctx()
    return mon, beh


def _tick(mon, beh, draw=None, dt=1.0):
    if draw is not None:
        beh.draw = float(draw)
    mon._context.current_timestamp += dt
    asyncio.run(mon._monitor())


# --- feedback target shift -------------------------------------------------


def test_draw_at_target_is_in_settle_band():
    # A draw sitting at the feedback target (mid-band) must NOT wind the
    # effective budget down — the loop has converged.
    mon, beh = _mon(-(1.0 - _FEEDBACK_TARGET_MARGIN) * BUDGET)  # |draw| == target
    _tick(mon, beh)
    assert lookup_slack_eff_budget(beh, mon.context.aid) is None  # untouched


def test_draw_below_claim_threshold_not_flagged():
    # The settle band top edge is B; a draw there is a full tol below the
    # compliance claim's B*(1+tol) threshold, so it is not a violation.
    mon, beh = _mon(-BUDGET)  # |draw| == B, i.e. inside B*(1+tol)
    _tick(mon, beh)
    assert not mon._violation_active


def test_draw_above_band_winds_eff_down():
    # A 1.04B draw sat in the OLD deadband (err=0.04B<0.05B -> frozen). With the
    # target shifted to 0.95B, err=0.09B>0.05B, so the loop now corrects.
    mon, beh = _mon(-1.04 * BUDGET)
    _tick(mon, beh)
    eff = lookup_slack_eff_budget(beh, mon.context.aid)
    assert eff is not None and eff < BUDGET


def test_margin_matches_tol_so_band_top_is_budget():
    # Design invariant: target + tol*B == B (top edge lands on the budget).
    assert _FEEDBACK_TARGET_MARGIN == pytest.approx(0.05)
