"""The L3 -> L2 wake-up edge must have a publisher.

``HolonicCommunityRole._handle_cp_setpoint`` is L2's only reactive trigger for
electricity and gas: ``RebalanceRound.dirty`` clears on the first round and the
periodic ``_try_rebalance`` runs at ``holon_watchdog_s`` (30 s, i.e. a whole
episode). ``EnergyConverterRole`` published the :class:`CPSetpoint` that lifts
it; :class:`CPPriorityAdmmRole` superseded that role (they are ``elif``-exclusive
in ``scenario.restoration``) without carrying the edge over, so the handler had
no publisher at all. Measured on ``simbench_lv_gas_dependent``: gas leaders
allocated exactly twice (t=0.08, t=0.18), both before any P2G had produced, and
that blanket 0.0 stood for the remaining 29.8 s — gas served ≡ 0.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from scare.base.channel import CPSetpoint
from scare.base.model import Sector
from scare.community.holonic import HolonicCommunityRole
from scare.service.coupling.cp_priority_admm_role import CPPriorityAdmmRole


@dataclass
class _MockBehavior:
    actions: dict[str, set[str]] = field(default_factory=dict)
    obs: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_log: list[tuple[str, str, float]] = field(default_factory=list)

    def has_action(self, aid: str, kind: str) -> bool:
        return kind in self.actions.get(aid, set())

    def observe(self, aid: str) -> dict[str, Any] | None:
        return self.obs.get(aid)

    def act(self, aid: str, kind: str, value: float) -> None:
        self.action_log.append((aid, kind, float(value)))


class _Addr:
    def __init__(self, aid: str) -> None:
        self.aid = aid

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Addr({self.aid})"


class _FakeContext:
    def __init__(self, aid: str, t: float = 100.0) -> None:
        self.aid = aid
        self.addr = _Addr(aid)
        self.current_timestamp = t
        self.sent: list[tuple[Any, Any]] = []
        self.scheduled: list[Any] = []

    async def send_message(self, payload: Any, *, receiver_addr: Any) -> None:
        self.sent.append((payload, receiver_addr))

    def schedule_instant_task(self, coro: Any) -> None:
        self.scheduled.append(coro)

    def schedule_periodic_task(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def subscribe_message(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def drain(self) -> None:
        pending, self.scheduled = self.scheduled, []
        for coro in pending:
            await coro


def _make_p2g(
    cp_id: str = "branch-260-2",
    *,
    gas_mw: float = -0.0036,
    el_mw: float = 0.0060,
) -> tuple[CPPriorityAdmmRole, _FakeContext]:
    """A P2G as built on ``simbench_lv_gas_dependent``: produces gas (cap < 0),
    draws electricity, bridges neither heat."""
    behavior = _MockBehavior()
    behavior.actions[cp_id] = {"regulate"}
    behavior.obs[cp_id] = {"regulation": 0.0}
    role = CPPriorityAdmmRole(
        behavior=behavior,
        cp_id=cp_id,
        capacity_by_sector={
            Sector.GAS.value: gas_mw,
            Sector.ELECTRICITY.value: el_mw,
        },
        bridged_sectors=[Sector.ELECTRICITY, Sector.GAS],
        algorithm="gossip",
    )
    ctx = _FakeContext(cp_id)
    role._context = ctx  # type: ignore[attr-defined]
    return role, ctx


def _cache_leader(role: CPPriorityAdmmRole, sector: Sector, aid: str) -> None:
    """Stand in for a HolonSummary arrival: what ``_wake_l2`` addresses."""
    role._leader_addrs.setdefault(sector.value, {})[aid] = _Addr(aid)


def _setpoints(ctx: _FakeContext) -> list[tuple[CPSetpoint, Any]]:
    return [(m, a) for m, a in ctx.sent if isinstance(m, CPSetpoint)]


@pytest.mark.asyncio
async def test_commit_wakes_every_gas_leader_that_fed_us_demand():
    """The regression: without this no gas leader ever re-ran after t=0.18."""
    role, ctx = _make_p2g()
    for aid in ("child-148", "child-160", "child-215"):
        _cache_leader(role, Sector.GAS, aid)

    role._on_gossip_commit(r=np.array([1.0]), converged=True, iterations=4)
    await ctx.drain()

    woken = {addr.aid for _, addr in _setpoints(ctx)}
    assert woken == {"child-148", "child-160", "child-215"}


@pytest.mark.asyncio
async def test_wake_up_is_scoped_to_bridged_sectors():
    """A P2G moves no heat flow, so heat leaders must not be woken."""
    role, ctx = _make_p2g()
    _cache_leader(role, Sector.GAS, "gas-leader")
    _cache_leader(role, Sector.ELECTRICITY, "el-leader")
    role._leader_addrs.setdefault(Sector.HEAT.value, {})["heat-leader"] = _Addr(
        "heat-leader"
    )

    await role._wake_l2(1.0)

    woken = {addr.aid for _, addr in _setpoints(ctx)}
    assert woken == {"gas-leader", "el-leader"}


@pytest.mark.asyncio
async def test_setpoint_carries_capacity_scaled_flows():
    """The receiver gauges the shift from ``sector_flows_mw``, so it has to be
    the real flow, not the factor."""
    role, ctx = _make_p2g(gas_mw=-0.0036, el_mw=0.0060)
    _cache_leader(role, Sector.GAS, "gas-leader")

    await role._wake_l2(0.5)

    msg, _ = _setpoints(ctx)[0]
    assert msg.cp_id == role.cp_id
    assert msg.regulation_factor == pytest.approx(0.5)
    assert msg.sector_flows_mw[Sector.GAS.value] == pytest.approx(-0.0018)
    assert msg.sector_flows_mw[Sector.ELECTRICITY.value] == pytest.approx(0.0030)


@pytest.mark.asyncio
async def test_repeat_commit_at_the_same_factor_is_dead_banded():
    """376 commits/task x 39 gas leaders would be a message storm; only a shift
    the receiver would act on is sent."""
    role, ctx = _make_p2g()
    _cache_leader(role, Sector.GAS, "gas-leader")

    await role._wake_l2(1.0)
    assert len(_setpoints(ctx)) == 1
    await role._wake_l2(1.0)
    assert len(_setpoints(ctx)) == 1
    await role._wake_l2(0.999)  # |cap| * 0.001 is far below the dead band
    assert len(_setpoints(ctx)) == 1


@pytest.mark.asyncio
async def test_a_material_shift_wakes_again():
    role, ctx = _make_p2g(gas_mw=-0.0036, el_mw=0.0060)
    _cache_leader(role, Sector.GAS, "gas-leader")

    await role._wake_l2(1.0)
    await role._wake_l2(0.0)  # full shutdown: 0.0060 MW shift
    assert len(_setpoints(ctx)) == 2


@pytest.mark.asyncio
async def test_no_cached_leader_is_a_silent_noop():
    """A CP that has never received a summary has nobody to wake."""
    role, ctx = _make_p2g()
    await role._wake_l2(1.0)
    assert _setpoints(ctx) == []


def test_send_gate_is_never_coarser_than_the_receivers_predicate():
    """If the send-side dead band exceeded the receiver's, we would silently
    suppress wake-ups the receiver would have acted on."""
    assert (
        CPPriorityAdmmRole._L2_WAKE_DEAD_BAND_MW
        <= HolonicCommunityRole._CP_PREDICATE_DEAD_BAND_MW
    )


@pytest.mark.asyncio
async def test_leader_address_is_learned_from_the_summary_sender():
    """mango's connector list is a locality-free cross-product, so the summary
    senders are the only measured leader set."""
    from scare.base.channel import HolonSummary

    role, ctx = _make_p2g()
    summary = HolonSummary(
        publisher="child-148",
        version=1,
        caused_by={},
        timestamp_s=float(ctx.current_timestamp),
        sector=Sector.GAS,
    )
    meta = {"sender_addr": "tcp://leader:1", "sender_id": "child-148"}
    await role._on_holon_summary(summary, meta=meta)
    assert "child-148" in role._leader_addrs[Sector.GAS.value]
    assert role._leader_addrs[Sector.GAS.value]["child-148"].aid == "child-148"


@pytest.mark.asyncio
async def test_wake_survives_an_unreachable_leader():
    """One dead address must not stop the rest of the fan-out."""
    role, ctx = _make_p2g()
    for aid in ("good-1", "dead", "good-2"):
        _cache_leader(role, Sector.GAS, aid)

    original = ctx.send_message

    async def _flaky(payload: Any, *, receiver_addr: Any) -> None:
        if receiver_addr.aid == "dead":
            raise ConnectionError("agent gone")
        await original(payload, receiver_addr=receiver_addr)

    ctx.send_message = _flaky  # type: ignore[method-assign]
    await role._wake_l2(1.0)

    assert {addr.aid for _, addr in _setpoints(ctx)} == {"good-1", "good-2"}


@pytest.mark.asyncio
async def test_commit_schedules_the_wake_because_on_commit_is_sync():
    """DRO's ``on_commit`` is a plain callback, so the wake has to go through
    ``schedule_instant_task`` or it is never awaited."""
    role, ctx = _make_p2g()
    _cache_leader(role, Sector.GAS, "gas-leader")

    role._on_gossip_commit(r=np.array([0.4]), converged=True, iterations=3)
    assert _setpoints(ctx) == []  # not sent yet — it is a coroutine
    # The commit also schedules a CPSummary republish (peers net our factor out
    # of their base supply), so assert on the wake itself, not the queue length.
    assert any(c.__name__ == "_wake_l2" for c in ctx.scheduled)

    await ctx.drain()
    assert len(_setpoints(ctx)) == 1


@pytest.mark.asyncio
async def test_concurrent_wakes_do_not_duplicate():
    """Two commits resolved in the same tick still send one fan-out each at
    most; the gate is checked before the awaits."""
    role, ctx = _make_p2g()
    _cache_leader(role, Sector.GAS, "gas-leader")

    await asyncio.gather(role._wake_l2(1.0), role._wake_l2(1.0))
    assert len(_setpoints(ctx)) == 1
