"""A slack's published headroom is its budget MINUS what is already flowing.

``HolonSummary.slack_budget_by_sector`` is a cap on total import, while
``served_by_sector_priority`` is a flow already drawing on that cap. The L3 CP
kernel wants "current service plus what more is available", so it needs the
remainder — adding the whole budget to ``served`` books the slack twice and
flips the sign of the deficit (eval_full_v2_20260728 task 004610: 47% of
electricity shed, reported as a +0.068 MW surplus).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from mango.express import topology as mango_topology

from scare.base.model import Sector
from scare.base.util import register_slack
from scare.base.util.blackboard import _slack_eff_budget_store
from scare.community import summary_publish as sp
from scare.community.summary_publish import SummaryPublisher
from scare.service.balance.grid_former import (
    GRID_FORMER_SUPPLY_PROBE_SHARE,
    GridFormerPolicy,
)
from tests.conftest import MockBehavior


class _Ctx:
    def __init__(self) -> None:
        self.aid = "leader"
        self.current_timestamp = 5.0
        self.sent: list[Any] = []

    async def send_message(self, payload: Any, **_kw: Any) -> None:
        self.sent.append(payload)


def _publisher(
    behavior: MockBehavior,
    members: list[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    nominal: bool = True,
) -> SummaryPublisher:
    role = SimpleNamespace(
        sector=Sector.ELECTRICITY,
        behavior=behavior,
        context=_Ctx(),
        cp_budget_nominal=nominal,
        inversion_tol=1e-9,
        _topology_tid="sector",
        _my_node_id=None,
        _version=SimpleNamespace(next=lambda: 1),
        _summary_changed=lambda *_a, **_k: True,
        _grid_former_policy=GridFormerPolicy(
            behavior, probe_share=GRID_FORMER_SUPPLY_PROBE_SHARE
        ),
    )
    monkeypatch.setattr(
        sp,
        "topology_neighbors",
        lambda _role, tid=None: (
            [SimpleNamespace(aid=a) for a in members] if tid == "groups" else ["peer"]
        ),
    )
    monkeypatch.setattr(mango_topology, "topology_characteristic", lambda *_a: None)
    pub = SummaryPublisher(role)
    monkeypatch.setattr(
        sp.CrossSectorChannel, "for_behavior", staticmethod(lambda _b: _NullChannel())
    )
    return pub


class _NullChannel:
    def publish(self, *_a: Any, **_k: Any) -> None:
        pass


def _behavior_with_slack(*, budget: float, flowing: float) -> MockBehavior:
    b = MockBehavior()
    # Load: cap 0.4, fully served.
    b.set_obs("load", {"p_mw": 0.4, "regulation": 1.0})
    # Slack: obs carries the LP operating point (load convention, injecting).
    b.set_obs("slack", {"p_mw": -flowing})
    register_slack(b, "slack", rating_mw=budget)
    _slack_eff_budget_store(b)["slack"] = budget
    return b


@pytest.mark.parametrize(
    ("budget", "flowing", "expected"),
    [
        (0.276287, 0.237774, 0.038513),  # task 004610: nearly all of it in use
        (0.168, 0.0, 0.168),  # idle slack offers its whole budget
        (0.168, 0.168, 0.0),  # maxed slack offers nothing
        (0.168, 0.30, 0.0),  # over budget clamps at zero, never negative
    ],
)
def test_headroom_is_budget_minus_current_import(
    budget: float, flowing: float, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    b = _behavior_with_slack(budget=budget, flowing=flowing)
    pub = _publisher(b, ["load", "slack"], monkeypatch)
    asyncio.run(pub._publish(force=True))

    summary = pub._role.context.sent[-1]
    sec = Sector.ELECTRICITY.value
    assert summary.slack_budget_by_sector.get(sec, 0.0) == pytest.approx(budget)
    assert summary.slack_headroom_by_sector.get(sec, 0.0) == pytest.approx(expected)


def test_headroom_never_exceeds_the_budget_it_is_derived_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the invariant the L3 consumer relies on: swapping budget for
    headroom can only ever lower ``base_supply``, never raise it."""
    import asyncio

    for flowing in (0.0, 0.05, 0.1, 0.2, 0.4):
        b = _behavior_with_slack(budget=0.2, flowing=flowing)
        pub = _publisher(b, ["load", "slack"], monkeypatch)
        asyncio.run(pub._publish(force=True))
        s = pub._role.context.sent[-1]
        sec = Sector.ELECTRICITY.value
        assert (
            0.0
            <= s.slack_headroom_by_sector.get(sec, 0.0)
            <= s.slack_budget_by_sector.get(sec, 0.0)
        )
