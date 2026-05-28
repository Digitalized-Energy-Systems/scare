"""Regression test for the slack-budget override sign.

eval_full_small_20260527-165650 task-84/85: a gas (``ExtHydrGrid``)
slack that reports *import* as a **positive** ``mass_flow`` received a
positive (add-load) override target and was driven to the 10x LP
envelope (+900% over budget).  The override must always *shed* toward
the budget for an over-importing slack, independent of which raw sign
the slack uses to encode import.  ``ExtPowerGrid`` slacks (import =
negative ``p_mw``) must be unchanged.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scare.base.model import Sector, StartBalanceNegotiation
from scare.service.slack_budget import SlackBudgetMonitor


class _Ctx:
    """Minimal RoleContext stand-in capturing sent messages."""

    def __init__(self, aid="child-slack"):
        self.aid = aid
        self.addr = aid
        self.current_timestamp = 1.0
        self.sent: list = []

    async def send_message(self, content, receiver_addr=None):
        self.sent.append((content, receiver_addr))

    def emit_event(self, event):  # no co-located subscriber in the test
        raise KeyError("no subscriber")

    def schedule_periodic_task(self, *a, **k):
        pass


def _run_monitor(*, obs_key, draw, budget, sector):
    """Drive one _monitor pass with a slack drawing `draw` over `budget`
    and return the override_target sent to the home leader (or None)."""
    mon = SlackBudgetMonitor(
        behavior=SimpleNamespace(observe=lambda aid: {obs_key: draw}),
        sector=sector,
        obs_key=obs_key,
        budget=budget,
        tol=0.05,
        home_leader_addr="leader-0",
    )
    mon._context = _Ctx()
    asyncio.run(mon._monitor())
    for content, _ in mon._context.sent:
        if isinstance(content, StartBalanceNegotiation):
            return content.override_target
    return None


def test_positive_import_slack_sheds_not_adds():
    # Gas slack: import encoded as POSITIVE mass_flow, 10x over budget.
    tgt = _run_monitor(obs_key="mass_flow", draw=0.00135, budget=0.000135,
                       sector=Sector.GAS)
    assert tgt is not None
    assert tgt < 0, f"over-importing slack must shed (negative target), got {tgt}"
    # Magnitude is the over-budget amount.
    assert tgt == pytest.approx(-(0.00135 - 0.000135))


def test_negative_import_slack_unchanged():
    # Electricity slack: import encoded as NEGATIVE p_mw — the historical
    # (working) case; target must be the same -(|val|-budget) shed value.
    tgt = _run_monitor(obs_key="p_mw", draw=-0.19, budget=0.168,
                       sector=Sector.ELECTRICITY)
    assert tgt is not None
    assert tgt < 0
    assert tgt == pytest.approx(-(0.19 - 0.168))


def test_within_budget_no_override():
    # Draw within budget*(1+tol) → no violation, no override.
    tgt = _run_monitor(obs_key="p_mw", draw=-0.17, budget=0.168,
                       sector=Sector.ELECTRICITY)
    assert tgt is None
