"""Component tests for GridConstraintMonitor role."""

import pytest
from mango import RoleAgent, create_world
from mango.simulation.world import step_simulation

from scare.base.model import ConstraintViolation, ConstraintWarning, Sector
from scare.service.control.constraints import GridConstraintMonitor
from tests.conftest import MockBehavior


def _make_monitor(
    behavior: MockBehavior,
    aid: str = "agent-0",
    sector: Sector = Sector.ELECTRICITY,
    vm_pu: float = 1.0,
    max_hops: int = 1,
) -> GridConstraintMonitor:
    behavior.set_obs(aid, {"p_mw": 5.0, "vm_pu": vm_pu})
    behavior.add_action(aid, "regulate")
    return GridConstraintMonitor(behavior, sector, node_id=0, max_hops=max_hops)


@pytest.mark.asyncio
async def test_no_event_within_bounds():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)  # center of [0.95, 1.05]

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        await step_simulation(world, step_size_s=1.0)

    assert monitor.is_locally_feasible()


@pytest.mark.asyncio
async def test_violation_emitted_when_out_of_bounds():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.06)  # above 1.05

    violations = []

    class ViolationCapture(RoleAgent):
        def handle_message(self, content, meta):
            pass

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    from mango.agent.role import Role

    class Listener(Role):
        def setup(self):
            self.context.subscribe_event(self, ConstraintViolation, self._on_violation)

        def _on_violation(self, event, src):
            violations.append(event)

    listener = Listener()
    agent.add_role(listener)

    async with world:
        # Advance past the electricity poll period (0.5s).
        await step_simulation(world, step_size_s=1.0)

    assert len(violations) >= 1
    assert violations[0].variable == "vm_pu"
    assert violations[0].value == 1.06
    assert not monitor.is_locally_feasible()


@pytest.mark.asyncio
async def test_warning_emitted_near_bound():
    # vm_pu=1.044 => util = |1.044-1.0|/0.05 = 0.88, above the 0.85
    # PROACTIVE_WARNING_FRACTION but below the 1.0 violation threshold.
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.044)

    warnings = []

    from mango.agent.role import Role

    class Listener(Role):
        def setup(self):
            self.context.subscribe_event(self, ConstraintWarning, self._on_warning)

        def _on_warning(self, event, src):
            warnings.append(event)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    agent.add_role(Listener())

    async with world:
        await step_simulation(world, step_size_s=1.0)

    assert len(warnings) >= 1
    assert warnings[0].variable == "vm_pu"
    assert warnings[0].utilization > 0.85


@pytest.mark.asyncio
async def test_is_locally_feasible_default():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        pass

    assert monitor.is_locally_feasible()


@pytest.mark.asyncio
async def test_local_sensitivity_ema_update():
    """Sensitivity estimator should converge via EMA from repeated (P, V) samples."""
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    initial = monitor.local_sensitivity()

    async with world:
        # First sample establishes baseline; subsequent samples move the EMA.
        for p, v in [(5.0, 1.00), (6.0, 1.02), (7.0, 1.04), (8.0, 1.06)]:
            behavior.set_obs("agent-0", {"p_mw": p, "vm_pu": v})
            await step_simulation(world, step_size_s=1.0)

    assert monitor.local_sensitivity() != initial
    assert monitor.local_sensitivity() > 0.0


@pytest.mark.asyncio
async def test_heat_sensitivity_updates_for_mw_scale_dp():
    """Heat sensitivity must update for MW-scale setpoint deltas.

    Heat ``obs_setpoint`` is ``q_mw_heat`` in MW (~0.0075-0.05), so
    ``_SENSITIVITY_MIN_DP[HEAT]`` must be small enough that a regulation
    step exceeds it; otherwise the estimate stays pinned at its default
    and the curtailment-auction willingness loses its sensitivity term.
    """
    behavior = MockBehavior()
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    initial = monitor.local_sensitivity()

    async with world:
        # setpoint = q_mw_heat * regulation; each step moves P by ~0.02 MW
        # (≫ the 5e-4 MW floor) and t_k by a few K -> the EMA must move.
        for reg, t in [(1.0, 360.0), (0.6, 366.0), (0.3, 372.0)]:
            behavior.set_obs(
                "agent-0", {"q_mw_heat": 0.05, "regulation": reg, "t_k": t}
            )
            await step_simulation(world, step_size_s=6.0)

    assert monitor.local_sensitivity() != initial
    assert monitor.local_sensitivity() > 0.0


@pytest.mark.asyncio
async def test_curtail_willingness_sensitivity_is_bounded():
    """The sensitivity multiplier in the auction willingness is clamped to
    ``[_SENS_MULT_MIN, _SENS_MULT_MAX]`` so it ranks within a tier but can
    never overcome the 1e4 priority tier step (no waterfall inversion)."""
    from scare.service.control.constraint_tuning import _SENS_MULT_MAX, _SENS_MULT_MIN

    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.04, "regulation": 1.0, "t_k": 360.0, "priority": 4},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        obs = behavior.observe("agent-0")
        monitor._sens._value = 1e9  # absurdly high
        w_hi = monitor._own_curtail_willingness(obs)
        monitor._sens._value = 1e-9  # absurdly low
        w_lo = monitor._own_curtail_willingness(obs)

    # Same tier + reducible in both calls, so the ratio is purely the
    # clamped sensitivity-multiplier span (16x), not the raw 1e18.
    assert w_hi / w_lo == pytest.approx(_SENS_MULT_MAX / _SENS_MULT_MIN)


@pytest.mark.asyncio
async def test_heat_frontier_sheds_to_partial_not_zero():
    """A cold heat node is curtailed to a PARTIAL feasible frontier, not
    bang-bang to 0: with a learned dT/dreg sensitivity the frontier
    controller takes a proportional step toward t_k = floor + margin."""
    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 305.0, "priority": 3},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        monitor._sens._value = 660.0  # learned dT/dP (K per MW): frontier ~0.8
        await monitor._heat_frontier_control()

    regs = [a for a in behavior.action_log if a[1] == "regulate"]
    assert regs, "frontier controller should have written a regulate"
    factor = regs[-1][2][0]
    assert 0.0 < factor < 1.0  # partial frontier, not bang-bang 0


@pytest.mark.asyncio
async def test_heat_frontier_applies_to_tier1():
    """Tier-1 heat is NOT exempt from the frontier controller — a critical
    heat load at an infeasible temperature is curtailed to its feasible
    partial (serving >0 beats the barrier crediting 0 at full draw)."""
    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 300.0, "priority": 1},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        monitor._sens._value = 660.0
        await monitor._heat_frontier_control()

    regs = [a for a in behavior.action_log if a[1] == "regulate"]
    assert regs and regs[-1][2][0] < 1.0  # tier-1 heat was curtailed


@pytest.mark.asyncio
async def test_heat_frontier_holds_in_band():
    """A feasible heat node comfortably inside the hold band is not touched."""
    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 318.0, "priority": 3},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(behavior, Sector.HEAT, node_id=0, max_hops=1)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        monitor._sens._value = 660.0
        await monitor._heat_frontier_control()

    assert not [a for a in behavior.action_log if a[1] == "regulate"]


@pytest.mark.asyncio
async def test_worst_neighbour_utilization_default():
    behavior = MockBehavior()
    monitor = _make_monitor(behavior, vm_pu=1.0)

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    async with world:
        pass

    assert monitor.worst_neighbour_utilization() == 0.0


# ---------------------------------------------------------------------------
# Heat priority-waterfall gate
# ---------------------------------------------------------------------------


def _cold_heat_monitor(priority: int, waterfall: bool = True):
    """A cold (t_k=300) heat load that the frontier controller would shed.

    ``enable_heat_frontier=False`` only suppresses the *auto-scheduled*
    periodic task (which would otherwise fire during world startup with an
    empty peer cache); the tests invoke ``_heat_frontier_control`` directly
    to exercise the gate in isolation.
    """
    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 300.0, "priority": priority},
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(
        behavior,
        Sector.HEAT,
        node_id=0,
        max_hops=1,
        # Disable the other auto-scheduled levers (SCADA-poll auction
        # self-curtail, multi-hop propagation) so only the manually-invoked
        # frontier gate writes a regulate.
        enable_curtailment_auction=False,
        enable_multihop_constraint=False,
        enable_heat_frontier=False,
        enable_heat_priority_waterfall=waterfall,
    )
    return behavior, monitor


@pytest.mark.asyncio
async def test_frontier_defers_shed_when_lower_priority_reducible_in_region():
    """A cold high-priority (tier-1) heat load defers its own shed while a
    strictly lower-priority (tier-4) peer still has reducible draw — so the
    waterfall sheds the low-priority load first."""
    behavior, monitor = _cold_heat_monitor(priority=1)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    async with world:
        monitor._sens._value = 660.0
        now = monitor.context.current_timestamp
        monitor._heat_frontier.note_peer_state("peer-4", now, 4, 0.05)  # tier-4
        await monitor._heat_frontier_control()
    assert not [a for a in behavior.action_log if a[1] == "regulate"], (
        "tier-1 load should defer to the lower-priority peer, not self-shed"
    )


@pytest.mark.asyncio
async def test_frontier_sheds_when_only_higher_priority_peers():
    """No strictly-lower-priority reducible peer ⇒ the cold load sheds
    itself (it is the lowest-priority lever available)."""
    behavior, monitor = _cold_heat_monitor(priority=2)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    async with world:
        monitor._sens._value = 660.0
        now = monitor.context.current_timestamp
        monitor._heat_frontier.note_peer_state("peer-1", now, 1, 0.05)  # higher prio
        await monitor._heat_frontier_control()
    regs = [a for a in behavior.action_log if a[1] == "regulate"]
    assert regs and regs[-1][2][0] < 1.0


def test_frontier_defer_decision_surfaces_waterfall():
    """The deferral is an explicit decision (not a silent hold) so the
    monitor can actuate the waterfall, and the request targets come back
    lowest-priority-first."""
    from scare.service.control.heat_frontier import HeatFrontierController

    ctrl = HeatFrontierController(peer_freshness_s=10.0)
    ctrl.note_peer_state("peer-3", 0.0, 3, 0.02)
    ctrl.note_peer_state("peer-4", 0.0, 4, 0.05)
    d = ctrl.decide(
        t=300.0,
        lo=313.15,
        cap=0.05,
        cur=1.0,
        sensitivity=660.0,
        now=0.0,
        my_tier=1,
        has_lock=False,
        waterfall_enabled=True,
    )
    assert d is not None
    assert d.reason == "defer_waterfall"
    assert d.new_reg == 1.0
    assert ctrl.waterfall_request_targets(1, 0.0) == [
        ("peer-4", 4, 0.05),
        ("peer-3", 3, 0.02),
    ]


@pytest.mark.asyncio
async def test_waterfall_peer_shed_request_curtails_peer():
    """The deferring cold tier-1 load actively sheds its lowest-priority
    reducible peer: a bounded CurtailmentRequest reaches the peer, which
    curtails multiplicatively — while the tier-1 load itself stays served."""
    from scare.base.model import ConstraintStateMessage

    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 300.0, "priority": 1},
    )
    behavior.add_action("agent-0", "regulate")
    behavior.set_obs(
        "agent-1",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 330.0, "priority": 4},
    )
    behavior.add_action("agent-1", "regulate")

    m0 = GridConstraintMonitor(
        behavior,
        Sector.HEAT,
        node_id=0,
        max_hops=1,
        enable_curtailment_auction=False,
        enable_multihop_constraint=False,
        enable_heat_frontier=False,
        enable_heat_priority_waterfall=True,
    )
    m1 = GridConstraintMonitor(
        behavior,
        Sector.HEAT,
        node_id=1,
        max_hops=1,
        enable_curtailment_auction=False,
        enable_multihop_constraint=False,
        enable_heat_frontier=False,
    )

    world = create_world()
    a0 = world.register(RoleAgent(), suggested_aid="agent-0")
    a0.add_role(m0)
    a1 = world.register(RoleAgent(), suggested_aid="agent-1")
    a1.add_role(m1)

    async with world:
        m0._sens._value = 660.0
        now = m0.context.current_timestamp
        origin = str(m1.context.addr)
        m0._heat_frontier.note_peer_state(origin, now, 4, 0.05)
        m0._neighbour_state[(origin, "t_k")] = ConstraintStateMessage(
            sector=Sector.HEAT,
            variable="t_k",
            value=330.0,
            utilization=0.1,
            hops_remaining=1,
            origin_addr=m1.context.addr,
            priority_tier=4,
            reducible=0.05,
        )
        await m0._heat_frontier_control()
        await step_simulation(world, step_size_s=1.0)

    peer_regs = [
        a for a in behavior.action_log if a[0] == "agent-1" and a[1] == "regulate"
    ]
    assert peer_regs, "peer never received/applied the waterfall curtail request"
    expected = 1.0 * (1.0 - GridConstraintMonitor._HEAT_WATERFALL_SHED_AMOUNT)
    assert peer_regs[-1][2][0] == pytest.approx(expected)
    own_regs = [
        a for a in behavior.action_log if a[0] == "agent-0" and a[1] == "regulate"
    ]
    assert not own_regs, "the deferring tier-1 load must not shed itself"


@pytest.mark.asyncio
async def test_waterfall_peer_shed_request_cooldown():
    """Repeated polls within the per-peer cooldown send only one request."""
    from scare.base.model import ConstraintStateMessage

    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 300.0, "priority": 1},
    )
    behavior.add_action("agent-0", "regulate")
    behavior.set_obs(
        "agent-1",
        {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 330.0, "priority": 4},
    )
    behavior.add_action("agent-1", "regulate")

    m0 = GridConstraintMonitor(
        behavior,
        Sector.HEAT,
        node_id=0,
        max_hops=1,
        enable_curtailment_auction=False,
        enable_multihop_constraint=False,
        enable_heat_frontier=False,
        enable_heat_priority_waterfall=True,
    )
    m1 = GridConstraintMonitor(
        behavior,
        Sector.HEAT,
        node_id=1,
        max_hops=1,
        enable_curtailment_auction=False,
        enable_multihop_constraint=False,
        enable_heat_frontier=False,
    )

    world = create_world()
    a0 = world.register(RoleAgent(), suggested_aid="agent-0")
    a0.add_role(m0)
    a1 = world.register(RoleAgent(), suggested_aid="agent-1")
    a1.add_role(m1)

    async with world:
        m0._sens._value = 660.0
        now = m0.context.current_timestamp
        origin = str(m1.context.addr)
        m0._heat_frontier.note_peer_state(origin, now, 4, 0.05)
        m0._neighbour_state[(origin, "t_k")] = ConstraintStateMessage(
            sector=Sector.HEAT,
            variable="t_k",
            value=330.0,
            utilization=0.1,
            hops_remaining=1,
            origin_addr=m1.context.addr,
            priority_tier=4,
            reducible=0.05,
        )
        # Two control polls back-to-back — inside the 2 s per-peer cooldown.
        await m0._heat_frontier_control()
        await m0._heat_frontier_control()
        await step_simulation(world, step_size_s=1.0)

    peer_regs = [
        a for a in behavior.action_log if a[0] == "agent-1" and a[1] == "regulate"
    ]
    assert len(peer_regs) == 1  # a single 0.25 step, not two compounded


@pytest.mark.asyncio
async def test_frontier_sheds_when_lower_priority_exhausted():
    """Once the lower-priority peer has shed (reducible ≈ 0) the gate opens
    and the high-priority load finally sheds itself (waterfall terminates)."""
    behavior, monitor = _cold_heat_monitor(priority=1)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    async with world:
        monitor._sens._value = 660.0
        now = monitor.context.current_timestamp
        monitor._heat_frontier.note_peer_state("peer-4", now, 4, 1e-9)  # ~shed
        await monitor._heat_frontier_control()
    regs = [a for a in behavior.action_log if a[1] == "regulate"]
    assert regs and regs[-1][2][0] < 1.0


@pytest.mark.asyncio
async def test_frontier_stale_peer_aged_out():
    """A lower-priority peer whose state is older than the freshness window
    is ignored, so the load is not pinned in a permanent defer."""
    behavior, monitor = _cold_heat_monitor(priority=1)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    async with world:
        monitor._sens._value = 660.0
        now = monitor.context.current_timestamp
        stale = now - monitor._heat_frontier._peer_freshness_s - 1.0
        monitor._heat_frontier.note_peer_state("peer-4", stale, 4, 0.05)
        await monitor._heat_frontier_control()
    regs = [a for a in behavior.action_log if a[1] == "regulate"]
    assert regs and regs[-1][2][0] < 1.0


@pytest.mark.asyncio
async def test_frontier_waterfall_disabled_sheds_tier_blind():
    """With the gate disabled the controller reverts to tier-blind shedding
    even when a lower-priority peer exists."""
    behavior, monitor = _cold_heat_monitor(priority=1, waterfall=False)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    async with world:
        monitor._sens._value = 660.0
        now = monitor.context.current_timestamp
        monitor._heat_frontier.note_peer_state("peer-4", now, 4, 0.05)
        await monitor._heat_frontier_control()
    regs = [a for a in behavior.action_log if a[1] == "regulate"]
    assert regs and regs[-1][2][0] < 1.0


@pytest.mark.asyncio
async def test_handle_constraint_state_populates_heat_peer_cache():
    """A heat t_k ConstraintStateMessage carrying (tier, reducible) lands in
    the priority-waterfall peer cache."""
    from scare.base.model import ConstraintStateMessage

    behavior, monitor = _cold_heat_monitor(priority=1)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    async with world:
        msg = ConstraintStateMessage(
            sector=Sector.HEAT,
            variable="t_k",
            value=300.0,
            utilization=1.2,
            hops_remaining=1,
            origin_addr="peer-9",
            priority_tier=4,
            reducible=0.03,
        )
        await monitor._handle_constraint_state(
            msg, {"sender_addr": "peer-sender", "sender_id": "s0"}
        )
    assert "peer-9" in monitor._heat_frontier._peer_state
    _t, tier, reducible, component = monitor._heat_frontier._peer_state["peer-9"]
    assert tier == 4 and abs(reducible - 0.03) < 1e-9
    assert component is None


# ---------------------------------------------------------------------------
# Curtailment-auction gating (enable_curtail_auction_gating)
# ---------------------------------------------------------------------------


def _gating_monitor(
    gating: bool, sector: Sector = Sector.HEAT, heat_frontier: bool = False
):
    behavior = MockBehavior()
    behavior.set_obs("agent-0", {"q_mw_heat": 0.05, "regulation": 1.0, "t_k": 300.0})
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(
        behavior,
        sector,
        node_id=0,
        max_hops=1,
        enable_curtailment_auction=True,
        enable_curtail_auction_gating=gating,
        enable_multihop_constraint=False,
        enable_heat_frontier=heat_frontier,
    )
    return behavior, monitor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gating,heat_frontier,expect_fire",
    [
        (False, True, False),
        (True, True, False),
        (False, False, True),
        (True, False, True),
    ],
)
async def test_gating_scopes_auction_off_temperature(
    gating, heat_frontier, expect_fire
):
    """``t_k`` is scoped off the auction only while the heat frontier owns it;
    with the frontier disabled the auction is the remaining backstop and must
    fire, independent of the progress-gating flag."""
    behavior, monitor = _gating_monitor(gating=gating, heat_frontier=heat_frontier)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    fired = []

    async def _spy(var, val, lo, hi):
        fired.append(var)

    async with world:
        monitor._request_curtailment = _spy  # type: ignore[assignment]
        await monitor._handle_violation({}, "t_k", 300.0, 313.15, 403.15)

    assert bool(fired) is expect_fire


@pytest.mark.asyncio
async def test_gating_progress_gate_suspends_stalled_rearm():
    """The progress guard (b): once the violation overshoot fails to
    improve for ``_CURTAIL_NO_PROGRESS_LIMIT`` consecutive rounds, the
    auction stops arming.  Driven directly so each call models one round
    (the in-flight guard is cleared between calls as ``_allocate_auction``
    would)."""
    from scare.service.control.constraint_tuning import _CURTAIL_NO_PROGRESS_LIMIT

    behavior, monitor = _gating_monitor(gating=True, sector=Sector.ELECTRICITY)
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    opened = []

    async def _spy_allocate(auction_id):
        opened.append(auction_id)
        monitor._open_auctions.pop(auction_id, None)

    async with world:
        monitor._allocate_auction = _spy_allocate  # type: ignore[assignment]
        # No neighbours ⇒ self-only auction allocates immediately each round.
        # Same un-improving overshoot every round (value pinned at 1.10,
        # hi=1.05): rounds beyond the no-progress limit must be suspended.
        n_rounds = _CURTAIL_NO_PROGRESS_LIMIT + 3
        for _ in range(n_rounds):
            monitor._curtail_inflight.pop("vm_pu", None)  # clear in-flight guard
            await monitor._request_curtailment("vm_pu", 1.10, 0.95, 1.05)

    # First (limit+1) rounds run (the +1 is the initial best-set round),
    # then the gate suspends the rest.
    assert len(opened) <= _CURTAIL_NO_PROGRESS_LIMIT + 1
    assert len(opened) < n_rounds


def test_curtail_proximity_monotonic_in_hops():
    """The targeting proximity multiplier increases with cached
    ``hops_remaining`` (closer to origin) and is neutral with no state."""
    from scare.base.model import ConstraintStateMessage
    from scare.service.control.constraint_tuning import (
        _CURTAIL_PROX_MAX,
        _CURTAIL_PROX_MIN,
    )

    behavior = MockBehavior()
    behavior.set_obs("agent-0", {"p_mw": 5.0, "vm_pu": 1.0})
    monitor = GridConstraintMonitor(
        behavior,
        Sector.ELECTRICITY,
        node_id=0,
        max_hops=3,
        enable_curtail_auction_targeting=True,
    )

    def _cache(hops):
        monitor._neighbour_state[("orig", "vm_pu")] = ConstraintStateMessage(
            sector=Sector.ELECTRICITY,
            variable="vm_pu",
            value=1.07,
            utilization=1.2,
            hops_remaining=hops,
            origin_addr="orig",
        )

    # No cached state ⇒ neutral.
    assert monitor._curtail_proximity("orig", "vm_pu") == 1.0
    _cache(3)  # closest (received with all hops left)
    near = monitor._curtail_proximity("orig", "vm_pu")
    _cache(0)  # farthest
    far = monitor._curtail_proximity("orig", "vm_pu")
    assert near == pytest.approx(_CURTAIL_PROX_MAX)
    assert far == pytest.approx(_CURTAIL_PROX_MIN)
    assert near > far


@pytest.mark.asyncio
async def test_targeting_scales_bid_by_proximity():
    """A bidder close to the violation origin replies with a strictly
    larger willingness than the same bidder when far — so the auctioneer's
    proportional allocation concentrates the shed on near (high-leverage)
    loads."""
    from scare.base.model import ConstraintStateMessage, CurtailmentBid, CurtailmentNeed

    behavior = MockBehavior()
    behavior.set_obs(
        "agent-0", {"q_mw_heat": 0.05, "regulation": 1.0, "vm_pu": 1.0, "priority": 3}
    )
    behavior.add_action("agent-0", "regulate")
    monitor = GridConstraintMonitor(
        behavior,
        Sector.ELECTRICITY,
        node_id=0,
        max_hops=3,
        enable_curtail_auction_targeting=True,
        enable_multihop_constraint=False,
    )
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    bids = []

    async def _spy_send(msg, receiver_addr=None, **kw):
        if isinstance(msg, CurtailmentBid):
            bids.append(msg.willingness)

    need = CurtailmentNeed(
        sector=Sector.ELECTRICITY,
        total_amount=0.3,
        auction_id="a1",
        origin_addr="orig",
        variable="vm_pu",
    )

    async with world:
        monitor.context.send_message = _spy_send  # type: ignore[assignment]
        # Far first (no cached state ⇒ neutral), then near.
        await monitor._handle_curtailment_need(
            need, {"sender_addr": "orig", "sender_id": "o"}
        )
        monitor._neighbour_state[("orig", "vm_pu")] = ConstraintStateMessage(
            sector=Sector.ELECTRICITY,
            variable="vm_pu",
            value=1.07,
            utilization=1.2,
            hops_remaining=3,
            origin_addr="orig",
        )
        await monitor._handle_curtailment_need(
            need, {"sender_addr": "orig", "sender_id": "o"}
        )

    assert len(bids) == 2 and bids[1] > bids[0]


@pytest.mark.asyncio
async def test_line_relief_reassert_cooldown():
    """Iterative line-relief re-asserts the relief while the line stays
    overloaded, but the cooldown suppresses a second send until it
    elapses (or the line clears) — so it never out-paces the gossip."""
    behavior = MockBehavior()
    behavior.set_obs("agent-0", {"vm_pu": 1.0, "p_from_mw": 0.5, "p_to_mw": 0.0})
    monitor = GridConstraintMonitor(
        behavior,
        Sector.ELECTRICITY,
        node_id=0,
        max_hops=1,
        branch_id="branch-1",
        home_leader_addr="leader-0",
        enable_line_relief_reassert=True,
    )
    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)

    sends = []

    async def _spy(obs, val, lo, hi):
        sends.append(val)

    obs = {"p_from_mw": 0.5, "p_to_mw": 0.0}
    async with world:
        monitor._send_line_overload_relief = _spy  # type: ignore[assignment]
        # First poll over the bound ⇒ one relief send.
        await monitor._reassert_line_relief(
            obs, "loading_percent", 116.0, -100.0, 100.0
        )
        # Same timestamp ⇒ inside cooldown ⇒ suppressed.
        await monitor._reassert_line_relief(
            obs, "loading_percent", 116.0, -100.0, 100.0
        )
        assert len(sends) == 1
        # Line returns in-bounds clears the cooldown (as _monitor does) ⇒
        # a later re-breach re-arms.
        monitor._relief_inflight.pop("loading_percent", None)
        await monitor._reassert_line_relief(
            obs, "loading_percent", 112.0, -100.0, 100.0
        )
        assert len(sends) == 2


def _make_waterfall_monitor(behavior):
    return GridConstraintMonitor(
        behavior,
        Sector.ELECTRICITY,
        node_id=0,
        max_hops=1,
        branch_id="branch-1",
        enable_branch_downstream_relief=True,
        enable_line_relief_waterfall=True,
        downstream_load_addrs=["L4", "L3", "L2", "L1"],
    )


async def _run_waterfall_alloc(monitor, bid_meta, total=0.5):
    """Drive ``_allocate_auction`` for a waterfall auction; return list of
    (receiver_addr, amount) the auctioneer dispatched."""
    from scare.base.model import CurtailmentRequest

    world = create_world()
    agent = world.register(RoleAgent(), suggested_aid="agent-0")
    agent.add_role(monitor)
    dispatched: list[tuple] = []

    async def _spy_send(msg, receiver_addr=None, **kw):
        if isinstance(msg, CurtailmentRequest):
            dispatched.append((receiver_addr, msg.amount))

    monitor._open_auctions["a1"] = {
        "bids": {k: 1.0 for k in bid_meta},
        "bidders": {k: k for k in bid_meta},  # addr == key for the test
        "bid_meta": dict(bid_meta),
        "total": total,
        "var": "loading_percent",
        "self_willingness": None,
        "self_addr": None,
        "neighbours_contacted": len(bid_meta),
        "waterfall": True,
    }
    async with world:
        monitor.context.send_message = _spy_send  # type: ignore[assignment]
        await monitor._allocate_auction("a1")
    return dispatched


@pytest.mark.asyncio
async def test_line_relief_waterfall_sheds_lowest_priority_tier_first():
    """With all tiers reducible, only the lowest-priority (tier-4) downstream
    loads are shed; higher tiers (incl. tier-1) are untouched this round."""
    behavior = MockBehavior()
    monitor = _make_waterfall_monitor(behavior)
    # (tier, reducible) per bidder.
    meta = {"L4": (4, 0.02), "L3": (3, 0.10), "L2": (2, 0.05), "L1": (1, 0.03)}
    dispatched = await _run_waterfall_alloc(monitor, meta, total=0.5)
    shed = {addr for addr, _ in dispatched}
    assert shed == {"L4"}, shed
    assert all(amt == pytest.approx(0.5) for _, amt in dispatched)


@pytest.mark.asyncio
async def test_line_relief_waterfall_escalates_when_lower_tier_exhausted():
    """When tier-4 is exhausted (reducible below the threshold), the cascade
    escalates to tier-3 — and never sheds tier-1."""
    behavior = MockBehavior()
    monitor = _make_waterfall_monitor(behavior)
    meta = {"L4": (4, 1e-6), "L3": (3, 0.10), "L2": (2, 0.05), "L1": (1, 0.03)}
    dispatched = await _run_waterfall_alloc(monitor, meta, total=0.5)
    shed = {addr for addr, _ in dispatched}
    assert shed == {"L3"}, shed


@pytest.mark.asyncio
async def test_line_relief_waterfall_protects_tier1_and_flags_residual():
    """When the only reducible downstream load left is tier-1, nothing is shed
    and the tier-1-residual flag is set so the auction stops re-arming."""
    behavior = MockBehavior()
    monitor = _make_waterfall_monitor(behavior)
    meta = {"L4": (4, 1e-6), "L3": (3, 1e-6), "L2": (2, 1e-6), "L1": (1, 0.03)}
    dispatched = await _run_waterfall_alloc(monitor, meta, total=0.5)
    assert dispatched == [], dispatched
    assert monitor._line_relief_tier1_residual.get("loading_percent") is True
