"""Contract tests for the L3 priority-ADMM role wiring.

The kernel in :mod:`scare.service.cp_priority_admm` is the compute
side of the L3 redesign; the role
:class:`scare.service.cp_priority_admm_role.CPPriorityAdmmRole` is the
wiring side that drives the kernel from the gossiped peer view and
commits the local CP's regulation factor.

These tests exercise the role directly against a fake mango context
(no agent world spin-up).  They lock in three contract properties:

1. The role publishes its own :class:`CPSummary` on setup so peer CPs
   can include it in their replicated view from the first kernel run.
2. A fresh :class:`HolonSummary` from a leader triggers the kernel
   and produces exactly one ``apply_regulate`` write addressed to the
   CP's own aid — no cross-CP envelope, no leader-side directive.
3. Two roles holding identical replicated views compute the same
   regulation factor, the determinism guarantee that the
   coordinator-free design depends on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from scare.base.channel import CPSummary, HolonSummary
from scare.base.model import Sector
from scare.service.cp_priority_admm_role import CPPriorityAdmmRole
from tests.conftest import MockBehavior


# ---------------------------------------------------------------------------
# Fake mango context
# ---------------------------------------------------------------------------


@dataclass
class _SentMessage:
    payload: Any
    receiver_addr: Any


class _Addr:
    def __init__(self, aid: str) -> None:
        self.aid = aid

    def __repr__(self) -> str:  # pragma: no cover
        return f"<addr {self.aid}>"


class _FakeContext:
    def __init__(self, aid: str, t: float = 100.0) -> None:
        self.aid = aid
        self.addr = _Addr(aid)
        self.current_timestamp = t
        self.sent: list[_SentMessage] = []
        self._scheduled: list[Any] = []

    async def send_message(self, payload: Any, *, receiver_addr: Any) -> None:
        self.sent.append(_SentMessage(payload=payload, receiver_addr=receiver_addr))

    def schedule_instant_task(self, coro: Any) -> None:
        # Tests await the coroutine manually via asyncio.run — record
        # for visibility but do not auto-run.
        self._scheduled.append(coro)

    def schedule_periodic_task(self, *_args: Any, **_kwargs: Any) -> None:
        # No periodic execution in tests; the watchdog tick is
        # exercised by directly invoking the role's helpers.
        pass

    def subscribe_message(self, *_args: Any, **_kwargs: Any) -> None:
        # The tests inject summaries directly into the role's caches,
        # bypassing the subscription dispatch.
        pass


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_behavior_for(aid: str) -> MockBehavior:
    """MockBehavior with the regulate action enabled on ``aid`` so
    ``apply_regulate`` won't bail at the ``has_action`` gate."""
    b = MockBehavior()
    b.add_action(aid, "regulate")
    b.set_obs(aid, {"regulation": 0.0})
    return b


def _make_role(
    cp_id: str,
    *,
    capacity_by_sector: dict[str, float],
    bridged_sectors: list[Sector] | None = None,
    rebalance_min_gap_s: float = 0.0,
) -> tuple[CPPriorityAdmmRole, _FakeContext, MockBehavior]:
    """Construct a role bound to a fake context.  Returns
    ``(role, ctx, behavior)`` so tests can drive ``role`` and observe
    side effects on ``ctx.sent`` and ``behavior.action_log``.
    """
    behavior = _build_behavior_for(cp_id)
    role = CPPriorityAdmmRole(
        behavior=behavior,
        cp_id=cp_id,
        capacity_by_sector=capacity_by_sector,
        bridged_sectors=bridged_sectors or [
            Sector(s) if not isinstance(s, Sector) else s
            for s in (
                # Default: derive bridged sectors from the non-zero
                # capacity entries so tests can pass just the dict.
                k for k, v in capacity_by_sector.items() if v != 0.0
            )
        ],
        rebalance_min_gap_s=rebalance_min_gap_s,
        admm_max_iters=200,
    )
    ctx = _FakeContext(cp_id)
    role._context = ctx  # type: ignore[attr-defined]
    return role, ctx, behavior


def _inject_holon_summary(
    role: CPPriorityAdmmRole,
    *,
    leader_aid: str,
    sector: Sector,
    supply_mw: float,
    demand_by_tier: dict[int, float],
    served_by_tier: dict[int, float] | None = None,
    slack_budget_mw: float = 0.0,
    version: int = 1,
) -> None:
    """Populate the role's leader-summaries cache directly so the
    kernel sees the desired peer view without going through the
    subscription path."""
    served_by_tier = served_by_tier or {t: 0.0 for t in demand_by_tier}
    sv = sector.value
    summary = HolonSummary(
        publisher=leader_aid,
        version=version,
        caused_by={},
        timestamp_s=0.0,
        sector=sector,
        per_tier_served_mw=dict(served_by_tier),
        per_tier_demand_mw=dict(demand_by_tier),
        supply_by_sector={sv: supply_mw} if supply_mw > 0 else {},
        demand_by_sector_priority={sv: dict(demand_by_tier)},
        served_by_sector_priority={sv: dict(served_by_tier)},
        slack_budget_by_sector={sv: slack_budget_mw} if slack_budget_mw > 0 else {},
    )
    role._leader_summaries.setdefault(sv, {})[leader_aid] = summary  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Sanity: config flag exists
# ---------------------------------------------------------------------------


def test_config_flag_is_present_and_on_by_default() -> None:
    """``enable_cp_priority_admm`` is the default L3 path: the replicated
    kernel runs in place of the legacy coordinator-elected path, which the
    install chain in ``scenario.restoration`` shadows when the flag is on.
    Both flags default True; set ``enable_cp_priority_admm=False`` to opt
    back into the legacy coordinator path.
    """
    from scare.base.config import RestorationConfiguration

    cfg = RestorationConfiguration()
    assert hasattr(cfg, "enable_cp_priority_admm")
    assert cfg.enable_cp_priority_admm is True
    assert cfg.enable_cp_admm is True


# ---------------------------------------------------------------------------
# Contract 1: initial publish
# ---------------------------------------------------------------------------


def test_setup_publishes_initial_cp_summary_to_every_peer() -> None:
    """Setup schedules an initial publish to every known peer CP so
    each peer can include this CP in its replicated view from the
    first round.
    """
    role, ctx, _ = _make_role(
        "p2h-A",
        capacity_by_sector={"electricity": 10.0, "heat": -9.5},
    )
    peer_addrs = {
        "p2h-B": _Addr("p2h-B"),
        "chp-C": _Addr("chp-C"),
    }
    role.wire(
        topology_mirror=None,  # type: ignore[arg-type]
        peer_cp_addrs=peer_addrs,
        peer_cp_node_ids={"p2h-B": 1, "chp-C": 2},
    )
    role.setup()
    # The publish coroutine is scheduled — await it.
    while ctx._scheduled:  # type: ignore[attr-defined]
        coro = ctx._scheduled.pop(0)  # type: ignore[attr-defined]
        asyncio.run(coro)

    summaries = [m for m in ctx.sent if isinstance(m.payload, CPSummary)]
    assert len(summaries) == 2, (
        "expected one CPSummary to each known peer; got "
        f"{[type(m.payload).__name__ for m in ctx.sent]}"
    )
    receivers = {m.receiver_addr.aid for m in summaries}
    assert receivers == {"p2h-B", "chp-C"}
    # All published summaries carry the role's signed capacity.
    for m in summaries:
        s: CPSummary = m.payload
        assert s.publisher == "p2h-A"
        assert s.capacity_by_sector == {"electricity": 10.0, "heat": -9.5}


# ---------------------------------------------------------------------------
# Contract 2: kernel run + self-addressed commit
# ---------------------------------------------------------------------------


def test_holon_summary_triggers_kernel_and_self_addressed_apply_regulate() -> None:
    """A fresh leader summary fires the kernel; the role commits its
    own regulation factor via ``apply_regulate`` (recorded as a
    ``regulate`` entry on ``behavior.action_log`` keyed on the CP's
    own aid).  No cross-CP envelope is sent.
    """
    role, ctx, behavior = _make_role(
        "p2h-A",
        capacity_by_sector={"electricity": 10.0, "heat": -9.5},
    )
    role.wire(
        topology_mirror=None,  # type: ignore[arg-type]
        peer_cp_addrs={},  # singleton component
        peer_cp_node_ids={},
    )

    # Heat scarcity: tier-1 demand 5 MW, no base supply on heat side.
    # Electricity side: no demand, 100 MW base supply.  The P2H should
    # ramp up to serve heat.
    _inject_holon_summary(
        role,
        leader_aid="heat-leader",
        sector=Sector.HEAT,
        supply_mw=0.0,
        demand_by_tier={1: 5.0},
    )
    _inject_holon_summary(
        role,
        leader_aid="el-leader",
        sector=Sector.ELECTRICITY,
        supply_mw=100.0,
        demand_by_tier={},
    )

    role._dirty = True  # type: ignore[attr-defined]
    asyncio.run(role._maybe_rebalance())  # type: ignore[attr-defined]

    # Exactly one regulate write on this CP's aid.
    regulates = [
        e for e in behavior.action_log
        if e[0] == "p2h-A" and e[1] == "regulate"
    ]
    assert len(regulates) == 1, (
        f"expected exactly one regulate write on p2h-A; got: {behavior.action_log}"
    )
    # And it should be a non-trivial ramp-up under the heat scarcity.
    factor = float(regulates[0][2][0])
    assert factor > 0.1, (
        f"expected P2H to ramp up under heat-tier-1 scarcity; got factor={factor:.3f}"
    )

    # No cross-CP CPSummary publish was triggered by the kernel run —
    # CPSummary only flows on initial publish / capacity-change / watchdog.
    cp_summary_sends = [m for m in ctx.sent if isinstance(m.payload, CPSummary)]
    assert not cp_summary_sends, (
        f"kernel run should not publish CPSummary; got {len(cp_summary_sends)} sends"
    )


# ---------------------------------------------------------------------------
# Contract 3: determinism across CPs
# ---------------------------------------------------------------------------


def test_two_roles_with_identical_view_commit_the_same_factor() -> None:
    """The replicated-coordinator pattern relies on the kernel being
    a deterministic pure function of its inputs.  Two role instances
    given the same leader summaries AND aware of the same peer set
    must compute (and apply) identical regulation factors.
    """

    def _build(cp_id: str) -> tuple[CPPriorityAdmmRole, _FakeContext, MockBehavior]:
        return _make_role(
            cp_id,
            capacity_by_sector={"electricity": 10.0, "heat": -9.5},
            bridged_sectors=[Sector.ELECTRICITY, Sector.HEAT],
        )

    role_a, _, beh_a = _build("p2h-A")
    role_b, _, beh_b = _build("p2h-B")

    # Each role knows about both CPs as peers (the other is its peer).
    role_a.wire(
        topology_mirror=None,  # type: ignore[arg-type]
        peer_cp_addrs={"p2h-B": _Addr("p2h-B")},
        peer_cp_node_ids={"p2h-B": 1},
    )
    role_b.wire(
        topology_mirror=None,  # type: ignore[arg-type]
        peer_cp_addrs={"p2h-A": _Addr("p2h-A")},
        peer_cp_node_ids={"p2h-A": 1},
    )

    # Each role caches the *other* CP's summary (the kernel internally
    # constructs the self spec from its own capacity).
    peer_summary = CPSummary(
        publisher="__peer__",  # patched per-role below
        version=1,
        caused_by={},
        timestamp_s=0.0,
        capacity_by_sector={"electricity": 10.0, "heat": -9.5},
        home_node_id=1,
    )
    role_a._peer_cps["p2h-B"] = CPSummary(  # type: ignore[attr-defined]
        **{**peer_summary.__dict__, "publisher": "p2h-B"}
    )
    role_b._peer_cps["p2h-A"] = CPSummary(  # type: ignore[attr-defined]
        **{**peer_summary.__dict__, "publisher": "p2h-A"}
    )

    # Identical leader view: 8 MW heat tier-1 unmet, 100 MW el surplus.
    for role in (role_a, role_b):
        _inject_holon_summary(
            role,
            leader_aid="heat-leader",
            sector=Sector.HEAT,
            supply_mw=0.0,
            demand_by_tier={1: 8.0},
        )
        _inject_holon_summary(
            role,
            leader_aid="el-leader",
            sector=Sector.ELECTRICITY,
            supply_mw=100.0,
            demand_by_tier={},
        )
        role._dirty = True  # type: ignore[attr-defined]

    asyncio.run(role_a._maybe_rebalance())  # type: ignore[attr-defined]
    asyncio.run(role_b._maybe_rebalance())  # type: ignore[attr-defined]

    f_a = float([
        e for e in beh_a.action_log if e[0] == "p2h-A" and e[1] == "regulate"
    ][-1][2][0])
    f_b = float([
        e for e in beh_b.action_log if e[0] == "p2h-B" and e[1] == "regulate"
    ][-1][2][0])

    # Determinism: same input view → same output factor.
    assert f_a == pytest.approx(f_b, abs=1e-9), (
        f"replicated kernel must be deterministic; got f_A={f_a}, f_B={f_b}"
    )
    # And it should be a non-trivial ramp-up (the asymmetric heat
    # demand of 8 MW versus combined supply of 19 MW splits roughly
    # evenly between the two CPs).
    assert f_a > 0.1, f"expected non-trivial ramp; got {f_a}"


# ---------------------------------------------------------------------------
# Contract 4: throttle gates back-to-back triggers
# ---------------------------------------------------------------------------


def test_rebalance_throttle_suppresses_back_to_back_triggers() -> None:
    """The minimum-gap throttle prevents a burst of HolonSummary
    arrivals from saturating the CP with kernel runs.  With the
    default gap of 0.5 s and two triggers landing inside that window,
    only the first runs the kernel; the second observes the throttle
    and defers.
    """
    role, _, behavior = _make_role(
        "p2h-A",
        capacity_by_sector={"electricity": 10.0, "heat": -9.5},
        rebalance_min_gap_s=0.5,
    )
    role.wire(
        topology_mirror=None,  # type: ignore[arg-type]
        peer_cp_addrs={},
        peer_cp_node_ids={},
    )
    _inject_holon_summary(
        role,
        leader_aid="heat-leader",
        sector=Sector.HEAT,
        supply_mw=0.0,
        demand_by_tier={1: 5.0},
    )
    _inject_holon_summary(
        role,
        leader_aid="el-leader",
        sector=Sector.ELECTRICITY,
        supply_mw=100.0,
        demand_by_tier={},
    )

    role._dirty = True  # type: ignore[attr-defined]
    asyncio.run(role._maybe_rebalance())  # type: ignore[attr-defined]
    n_after_first = sum(
        1 for e in behavior.action_log if e[1] == "regulate"
    )

    # Second trigger at the same simulated time — throttle should
    # suppress.  ``_dirty`` must be re-flipped to True because the
    # first run cleared it.
    role._dirty = True  # type: ignore[attr-defined]
    asyncio.run(role._maybe_rebalance())  # type: ignore[attr-defined]
    n_after_second = sum(
        1 for e in behavior.action_log if e[1] == "regulate"
    )

    assert n_after_first == 1, (
        f"first trigger should produce one regulate; got {n_after_first}"
    )
    # The second regulate either doesn't fire (throttle) or, even if
    # the kernel runs, the same factor is dedup'd by ``apply_regulate``.
    # Either way the count stays at one.
    assert n_after_second == 1, (
        f"throttle/dedup should keep regulate count at 1; got {n_after_second}"
    )


# ---------------------------------------------------------------------------
# Contract 5: gas supply/demand are converted to MW before the kernel
# ---------------------------------------------------------------------------


def test_build_demands_converts_gas_dimension_to_mw() -> None:
    """The kernel works in MW across every dimension — a CP's
    ``capacity_by_sector`` is MW (gas converted via ``kgps_to_mw`` in
    ``_cp_signed_capacity_by_sector``).  A ``HolonSummary`` carries gas
    supply/demand in native kg/s, so ``_build_demands`` must scale the
    gas dimension by ``kgps_to_mw`` to keep ``supply_net = base_supply −
    Σ r·c`` unit-consistent.  The electricity dimension must pass
    through untouched.
    """
    from scare.base.util import kgps_to_mw

    role, _, _ = _make_role(
        "g2p-A",
        capacity_by_sector={"gas": 5.0, "electricity": -2.0},
        bridged_sectors=[Sector.GAS, Sector.ELECTRICITY],
    )

    _inject_holon_summary(
        role,
        leader_aid="gas-leader",
        sector=Sector.GAS,
        supply_mw=1.0,            # 1.0 kg/s despite the generic param name
        demand_by_tier={1: 2.0},  # 2.0 kg/s
    )
    _inject_holon_summary(
        role,
        leader_aid="el-leader",
        sector=Sector.ELECTRICITY,
        supply_mw=10.0,           # 10 MW
        demand_by_tier={1: 4.0},  # 4 MW
    )

    demands = {d.sector: d for d in role._build_demands()}  # type: ignore[attr-defined]

    gas = demands[Sector.GAS.value]
    assert float(gas.base_supply[0]) == pytest.approx(kgps_to_mw(1.0))
    assert float(gas.demand_by_tier[1][0]) == pytest.approx(kgps_to_mw(2.0))
    # Sanity: the conversion is the ~55× HHV factor, not a no-op.
    assert float(gas.base_supply[0]) > 50.0

    el = demands[Sector.ELECTRICITY.value]
    assert float(el.base_supply[0]) == pytest.approx(10.0)
    assert float(el.demand_by_tier[1][0]) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Heat -> L3 supply link (enable_heat_cp_supply)
# ---------------------------------------------------------------------------


def test_heat_base_supply_uses_delivered_heat_in_deficit_mode():
    """With ``heat_supply_from_deficit`` set, the heat sector's L3 base
    supply is the *delivered* heat (Σ served), not the (unbounded) heat-
    slack budget — so the unmet demand becomes the gap CPs must fill."""
    role, _, _ = _make_role("chp-A", capacity_by_sector={"heat": -0.05, "gas": 0.1})
    role.heat_supply_from_deficit = True
    _inject_holon_summary(
        role,
        leader_aid="heat-leader",
        sector=Sector.HEAT,
        supply_mw=10.0,                 # unbounded-ish slack pool
        demand_by_tier={1: 0.8},        # nominal heat demand
        served_by_tier={1: 0.3},        # only 0.3 delivered (temp-limited)
    )
    heat = {d.sector: d for d in role._build_demands()}[Sector.HEAT.value]
    # base supply == delivered (0.3), NOT the 10.0 slack pool
    assert float(heat.base_supply[0]) == pytest.approx(0.3)
    assert float(heat.demand_by_tier[1][0]) == pytest.approx(0.8)


def test_heat_base_supply_uses_slack_when_flag_off():
    """Default (flag off): heat keeps the slack-budget base supply, so the
    pre-existing behaviour is preserved for ablation."""
    role, _, _ = _make_role("chp-A", capacity_by_sector={"heat": -0.05})
    assert role.heat_supply_from_deficit is False
    _inject_holon_summary(
        role,
        leader_aid="heat-leader",
        sector=Sector.HEAT,
        supply_mw=10.0,
        demand_by_tier={1: 0.8},
        served_by_tier={1: 0.3},
    )
    heat = {d.sector: d for d in role._build_demands()}[Sector.HEAT.value]
    assert float(heat.base_supply[0]) == pytest.approx(10.0)


def test_deficit_mode_caps_electricity_input_at_served_plus_slack_budget():
    """With ``heat_supply_from_deficit`` set, the CP-input sectors
    (electricity, gas) use ``base_supply = Σ served + slack eff_budget``
    instead of the aggregate supply pool — so a CP consuming from that
    sector is bounded by the binding slack's operator budget, not the
    (uncapped) non-slack |cap| sum."""
    role, _, _ = _make_role(
        "p2h-A", capacity_by_sector={"heat": -0.05, "electricity": 0.05},
        bridged_sectors=[Sector.HEAT, Sector.ELECTRICITY],
    )
    role.heat_supply_from_deficit = True
    _inject_holon_summary(
        role, leader_aid="el-leader", sector=Sector.ELECTRICITY,
        supply_mw=10.0,            # aggregate pool (slack + non-slack |cap|)
        slack_budget_mw=0.168,     # binding electricity slack's eff_budget
        demand_by_tier={1: 0.4}, served_by_tier={1: 0.37},
    )
    el = {d.sector: d for d in role._build_demands()}[Sector.ELECTRICITY.value]
    # served (0.37) + slack budget (0.168) = 0.538, NOT the 10.0 pool.
    assert float(el.base_supply[0]) == pytest.approx(0.538)


def test_deficit_mode_caps_gas_input_at_served_plus_slack_budget():
    """Same input-sector cap applies to gas (kg/s converted via
    kgps_to_mw on the way into the kernel)."""
    from scare.base.util import kgps_to_mw

    role, _, _ = _make_role(
        "chp-A",
        capacity_by_sector={"heat": -0.04, "gas": 0.05, "electricity": -0.02},
        bridged_sectors=[Sector.HEAT, Sector.GAS, Sector.ELECTRICITY],
    )
    role.heat_supply_from_deficit = True
    _inject_holon_summary(
        role, leader_aid="gas-leader", sector=Sector.GAS,
        supply_mw=5.0,                # aggregate (kg/s in the summary)
        slack_budget_mw=0.01,         # binding gas slack budget (kg/s)
        demand_by_tier={1: 0.04}, served_by_tier={1: 0.03},
    )
    gas = {d.sector: d for d in role._build_demands()}[Sector.GAS.value]
    # served (0.03) + slack budget (0.01) = 0.04 kg/s, kgps_to_mw'd:
    assert float(gas.base_supply[0]) == pytest.approx(kgps_to_mw(0.04))


def test_input_cap_off_when_flag_off():
    """Default (flag off): electricity keeps slack supply, preserving
    the pre-existing behaviour for ablation."""
    role, _, _ = _make_role(
        "chp-A", capacity_by_sector={"heat": -0.05, "electricity": -0.02},
        bridged_sectors=[Sector.HEAT, Sector.ELECTRICITY],
    )
    assert role.heat_supply_from_deficit is False
    _inject_holon_summary(
        role, leader_aid="el-leader", sector=Sector.ELECTRICITY,
        supply_mw=10.0, slack_budget_mw=0.168,
        demand_by_tier={1: 4.0}, served_by_tier={1: 1.0},
    )
    el = {d.sector: d for d in role._build_demands()}[Sector.ELECTRICITY.value]
    assert float(el.base_supply[0]) == pytest.approx(10.0)
