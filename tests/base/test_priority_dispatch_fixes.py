"""Targeted unit tests for the priority-dispatch behaviours: tier-aware
clamp, priority-aware cooldown bypass, stale-observation detection,
heat-sink and slack-class curtailment guards, and dual normalisation
over unsaturated entries only.
"""

from __future__ import annotations

import scare.base.runtime.diagnostics as _diag
from scare.base.model import Sector
from scare.base.runtime.diagnostics import arm
from scare.base.util import (
    apply_regulate,
    clamp_to_constraints,
    obs_priority,
    register_priority,
)


def _drain_events():
    """Snapshot the diagnostics event log and clear it for the next test."""
    events = list(_diag._event_log)
    _diag._event_log.clear()
    return events


def _disarm():
    _diag._armed = False
    _diag._event_log.clear()


# ---------------------------------------------------------------------------
# Tier-aware clamp
# ---------------------------------------------------------------------------


class TestTierAwareClamp:
    def test_critical_tier_skips_clamp_at_moderate_util(self):
        # vm_pu=1.04 → util=0.8.  Tier 1 is immune; tier-4 deadband 0.85 >
        # 0.8 ⇒ no clamp either. Both leave the setpoint unchanged.
        obs = {"p_mw": 10.0, "vm_pu": 1.04}
        assert clamp_to_constraints(5.0, obs, Sector.ELECTRICITY, tier=1) == 5.0
        assert clamp_to_constraints(5.0, obs, Sector.ELECTRICITY, tier=4) == 5.0

    def test_critical_tier_resists_clamp_past_default_deadband(self):
        # UNDER-voltage vm_pu=0.952 → util=0.96 (serving a load worsens the low
        # bound).  Tier 4 deadband 0.85 ⇒ allowed=(1-0.96)/0.15 ≈ 0.267,
        # max_abs≈2.67, so a low-tier 5 MW setpoint is throttled to ~2.67 MW.
        # Tier 1 is immune ⇒ no clamp regardless of util.
        obs = {"p_mw": 10.0, "vm_pu": 0.952}
        low_tier_result = clamp_to_constraints(5.0, obs, Sector.ELECTRICITY, tier=4)
        high_tier_result = clamp_to_constraints(5.0, obs, Sector.ELECTRICITY, tier=1)
        assert abs(low_tier_result) < 5.0  # was throttled
        assert high_tier_result == 5.0  # was not throttled (immune)

    def test_no_tier_preserves_legacy_behaviour(self):
        # No tier arg → legacy 0.85 deadband.  vm_pu=1.04 (util=0.8) ⇒ no clamp.
        obs = {"p_mw": 10.0, "vm_pu": 1.04}
        assert clamp_to_constraints(5.0, obs, Sector.ELECTRICITY) == 5.0

    def test_tier1_immune_even_under_extreme_stress(self):
        # At under-voltage vm_pu=0.9501 (util ≈ 0.998) a tier-2 load clamps to
        # near zero, but tier-1 immunity dominates the soft clamp and passes
        # through unmodified.
        obs = {"p_mw": 10.0, "vm_pu": 0.9501}
        tier1 = clamp_to_constraints(5.0, obs, Sector.ELECTRICITY, tier=1)
        tier2 = clamp_to_constraints(5.0, obs, Sector.ELECTRICITY, tier=2)
        assert tier1 == 5.0  # immune
        assert abs(tier2) < 5.0  # was throttled


# ---------------------------------------------------------------------------
# Priority-aware apply_regulate cooldown
# ---------------------------------------------------------------------------


class _FakeBehavior:
    def __init__(self, cooldown_s: float = 0.0):
        from types import SimpleNamespace

        self.acts: list[tuple[str, str, float]] = []
        self._scare_config = SimpleNamespace(cooldown_s=cooldown_s)
        self._net_results = object()  # opaque non-None sentinel

    def has_action(self, aid: str, action: str) -> bool:
        return True

    def act(self, aid: str, action: str, value: float) -> None:
        self.acts.append((aid, action, value))


class TestCooldownBypass:
    def setup_method(self):
        arm()

    def teardown_method(self):
        _disarm()

    def test_low_tier_suppressed_by_cooldown(self):
        b = _FakeBehavior(cooldown_s=1.0)
        # First write lands.
        assert (
            apply_regulate(
                b,
                "child-4",
                0.5,
                sector="electricity",
                reason="test",
                timestamp=10.0,
                priority_tier=4,
            )
            is True
        )
        # Second write within cooldown is suppressed.
        assert (
            apply_regulate(
                b,
                "child-4",
                0.6,
                sector="electricity",
                reason="test",
                timestamp=10.5,
                priority_tier=4,
            )
            is False
        )
        assert len(b.acts) == 1
        events = [
            e for e in _drain_events() if e.kind == "regulate_suppressed_by_cooldown"
        ]
        assert len(events) == 1
        assert "tier=4" in events[0].detail

    def test_critical_tier_bypasses_cooldown(self):
        b = _FakeBehavior(cooldown_s=1.0)
        assert (
            apply_regulate(
                b,
                "child-1",
                0.5,
                sector="electricity",
                reason="test",
                timestamp=10.0,
                priority_tier=1,
            )
            is True
        )
        # Second write within cooldown DOES land because tier 1 bypasses.
        assert (
            apply_regulate(
                b,
                "child-1",
                0.6,
                sector="electricity",
                reason="test",
                timestamp=10.5,
                priority_tier=1,
            )
            is True
        )
        assert len(b.acts) == 2

    def test_no_tier_falls_back_to_cooldown(self):
        b = _FakeBehavior(cooldown_s=1.0)
        assert (
            apply_regulate(
                b,
                "child-0",
                0.5,
                sector="electricity",
                reason="test",
                timestamp=10.0,
            )
            is True
        )
        # No tier given → not critical → suppressed.
        assert (
            apply_regulate(
                b,
                "child-0",
                0.6,
                sector="electricity",
                reason="test",
                timestamp=10.5,
            )
            is False
        )


# ---------------------------------------------------------------------------
# Stale-observation detector
# ---------------------------------------------------------------------------


class TestStaleObsCounter:
    def setup_method(self):
        arm()

    def teardown_method(self):
        _disarm()

    def test_no_event_when_obs_advances(self):
        b = _FakeBehavior()
        apply_regulate(b, "a1", 0.4, sector="e", reason="r", timestamp=1.0)
        b._net_results = object()  # advance — LP re-solved
        apply_regulate(b, "a2", 0.4, sector="e", reason="r", timestamp=2.0)
        events = [e for e in _drain_events() if e.kind == "regulate_on_stale_obs"]
        assert events == []

    def test_event_fires_when_same_aid_regulates_on_frozen_obs(self):
        b = _FakeBehavior()
        apply_regulate(b, "a1", 0.4, sector="e", reason="r", timestamp=1.0)
        # No _net_results update — a SECOND regulate on the SAME aid is stale.
        # Distinct factor clears the same-value dedup, not the stale check.
        apply_regulate(b, "a1", 0.5, sector="e", reason="r", timestamp=2.0)
        events = [e for e in _drain_events() if e.kind == "regulate_on_stale_obs"]
        assert len(events) == 1

    def test_no_event_for_distinct_aids_on_frozen_obs(self):
        # Batched multi-agent dispatch between two solves: each agent's first
        # write on the snapshot is fresh, so no stale event (the detector is
        # keyed per-aid; a write on a1 does not make a2's obs stale).
        b = _FakeBehavior()
        apply_regulate(b, "a1", 0.4, sector="e", reason="r", timestamp=1.0)
        apply_regulate(b, "a2", 0.4, sector="e", reason="r", timestamp=2.0)
        events = [e for e in _drain_events() if e.kind == "regulate_on_stale_obs"]
        assert events == []


# ---------------------------------------------------------------------------
# Heat-side Sink curtailment guard
# ---------------------------------------------------------------------------
# Heat-sector consumers are modelled in monee as a (HeatLoad, Sink) pair on
# the same junction: HeatLoad withdraws thermal energy, Sink withdraws the
# matching return-line mass flow.  Curtailing the Sink (regulation < 1)
# zeroes mass-flow withdrawal while the upstream pipe still pushes water in,
# breaking the junction's mass-flow balance and presolving the energy-flow
# LP into infeasibility.  The guard blocks such writes; thermal-control
# curtailment must flow through the HeatLoad instead.


class _FakeMonee:
    """Minimal monee-network stand-in for the heat-sink guard test.

    Supports the two lookups ``_is_heat_side_mass_flow_sink`` needs:
    ``child_by_id`` and ``node_by_id``.  The child model class drives the
    Sink-vs-other decision; the grid name on the child's node decides the
    heat/water-vs-gas branch.
    """

    def __init__(self, children: dict[int, object], nodes: dict[int, str]):
        self._children = children
        self._nodes = nodes

    def child_by_id(self, cid: int):
        return self._children[cid]

    def node_by_id(self, nid: int):
        return self._nodes[nid]


class _Node:
    def __init__(self, grid_name: str):
        from types import SimpleNamespace

        self.grid = SimpleNamespace(name=grid_name)


class _BehaviorWithNet(_FakeBehavior):
    def __init__(self, net, cooldown_s: float = 0.0):
        super().__init__(cooldown_s=cooldown_s)
        self._net = net


class TestHeatSinkGuard:
    def setup_method(self):
        arm()

    def teardown_method(self):
        _disarm()

    @staticmethod
    def _build(child_cls_name: str, grid_name: str) -> _BehaviorWithNet:
        from types import SimpleNamespace

        # Use a real monee Sink instance for the truthy case; a stand-in
        # with a non-Sink class for the falsy cases.
        if child_cls_name == "Sink":
            from monee.model.child import Sink

            model = Sink(mass_flow_kgs=0.05)
        else:
            model = SimpleNamespace()  # any non-Sink object
        child = SimpleNamespace(id=42, node_id=7, model=model)
        net = _FakeMonee(children={42: child}, nodes={7: _Node(grid_name)})
        return _BehaviorWithNet(net)

    def test_heat_side_sink_curtailment_is_blocked(self):
        b = self._build("Sink", "water-heat-supply")
        applied = apply_regulate(
            b,
            "child-42",
            0.0,
            sector="heat",
            reason="test",
            timestamp=1.0,
        )
        assert applied is False
        assert b.acts == []
        events = [e for e in _drain_events() if e.kind == "regulate_blocked_heat_sink"]
        assert len(events) == 1

    def test_heat_side_sink_full_passthrough_is_allowed(self):
        b = self._build("Sink", "water-heat-supply")
        applied = apply_regulate(
            b,
            "child-42",
            1.0,
            sector="heat",
            reason="test",
            timestamp=1.0,
        )
        assert applied is True
        assert b.acts == [("child-42", "regulate", 1.0)]

    def test_gas_sink_curtailment_is_allowed(self):
        b = self._build("Sink", "gas-supply")
        applied = apply_regulate(
            b,
            "child-42",
            0.3,
            sector="gas",
            reason="test",
            timestamp=1.0,
        )
        assert applied is True
        assert b.acts == [("child-42", "regulate", 0.3)]

    def test_heat_side_non_sink_curtailment_is_allowed(self):
        # HeatLoad (or any non-Sink class) on the heat grid stays curtailable.
        b = self._build("HeatLoad", "water-heat-supply")
        applied = apply_regulate(
            b,
            "child-42",
            0.4,
            sector="heat",
            reason="test",
            timestamp=1.0,
        )
        assert applied is True
        assert b.acts == [("child-42", "regulate", 0.4)]


# ---------------------------------------------------------------------------
# Slack-class curtailment guard
# ---------------------------------------------------------------------------
# Slack agents (ExtPowerGrid / ExtHydrGrid) have a free p_mw / mass_flow
# Var the LP picks within a wide physical envelope.  Writing
# ``regulation < 1`` clamps the LP's effective slack contribution to a
# fraction of that envelope, and the next solve diagnoses infeasible
# the moment the network needs more headroom than the clamped fraction.
# The guard blocks every such write; the gossip-side _apply_setpoint
# has a parallel class-check because it bypasses ``apply_regulate``
# entirely.


class TestSlackClassGuard:
    def setup_method(self):
        arm()

    def teardown_method(self):
        _disarm()

    @staticmethod
    def _build(child_cls_name: str) -> _BehaviorWithNet:
        from types import SimpleNamespace

        if child_cls_name == "ExtPowerGrid":
            from monee.model.child import ExtPowerGrid

            model = ExtPowerGrid(p_mw=1.0, q_mvar=0.0)
        elif child_cls_name == "ExtHydrGrid":
            from monee.model.child import ExtHydrGrid

            model = ExtHydrGrid()
        else:
            model = SimpleNamespace()
        child = SimpleNamespace(id=42, node_id=7, model=model)
        net = _FakeMonee(children={42: child}, nodes={7: _Node("power")})
        return _BehaviorWithNet(net)

    def test_ext_power_grid_curtailment_blocked(self):
        b = self._build("ExtPowerGrid")
        applied = apply_regulate(
            b,
            "child-42",
            0.3,
            sector="electricity",
            reason="test",
            timestamp=1.0,
        )
        assert applied is False
        assert b.acts == []
        events = [e for e in _drain_events() if e.kind == "regulate_blocked_slack"]
        assert len(events) == 1

    def test_ext_hydr_grid_curtailment_blocked(self):
        # Heat-side ExtHydrGrid is intentionally unbounded — the registry
        # never sees it — but the class check still protects it.
        b = self._build("ExtHydrGrid")
        applied = apply_regulate(
            b,
            "child-42",
            0.0,
            sector="heat",
            reason="test",
            timestamp=1.0,
        )
        assert applied is False
        assert b.acts == []
        events = [e for e in _drain_events() if e.kind == "regulate_blocked_slack"]
        assert len(events) == 1

    def test_ext_power_grid_full_passthrough_allowed(self):
        b = self._build("ExtPowerGrid")
        applied = apply_regulate(
            b,
            "child-42",
            1.0,
            sector="electricity",
            reason="test",
            timestamp=1.0,
        )
        assert applied is True
        assert b.acts == [("child-42", "regulate", 1.0)]


# ---------------------------------------------------------------------------
# World-construction skip for heat-side Sinks
# ---------------------------------------------------------------------------
# The upstream complement to the apply_regulate guard: heat-side Sinks are a
# monee topology artifact, so we never register an EnergyBalanceNegotiator
# or a priority for them.  The dispatcher therefore never sees them as
# members of any holon and cannot attempt a regulation write in the first
# place.


class TestHeatSinkUpstreamSkip:
    def test_predicate_identifies_heat_side_sinks(self):
        from types import SimpleNamespace

        from monee.model.child import HeatLoad, Sink

        from scare.scenario.restoration import _is_heat_side_mass_flow_sink

        class _Net:
            def __init__(self, node_grid_name: str):
                self._grid = SimpleNamespace(name=node_grid_name)

            def node_by_id(self, _nid):
                return SimpleNamespace(grid=self._grid)

        heat_sink = SimpleNamespace(id=1, node_id=10, model=Sink(mass_flow_kgs=0.04))
        gas_sink = SimpleNamespace(id=2, node_id=10, model=Sink(mass_flow_kgs=0.04))
        heat_load = SimpleNamespace(id=3, node_id=10, model=HeatLoad(q_mw=0.05))

        assert _is_heat_side_mass_flow_sink(heat_sink, _Net("water-heat-supply"))
        assert _is_heat_side_mass_flow_sink(heat_sink, _Net("heat-return"))
        assert not _is_heat_side_mass_flow_sink(gas_sink, _Net("gas-supply"))
        assert not _is_heat_side_mass_flow_sink(heat_load, _Net("water-heat-supply"))


# ---------------------------------------------------------------------------
# Default-fallback priority diagnostic
# ---------------------------------------------------------------------------


class TestSaturationFilteredDual:
    """The dual normalisation must divide by Σ a_j over *unsaturated*
    entries only, so saturated agents don't slow the dual step for free
    agents. Replicates EnergyBalanceNegotiator's dual update formula
    without spinning up mango.
    """

    @staticmethod
    def _entry_responsiveness(prio: int, target_sign: int) -> float:
        """Mirrors EnergyBalanceNegotiator._entry_responsiveness on the
        4-tier schedule (delegates to ``tier_priority_weight``)."""
        from scare.base.util import tier_priority_weight

        return tier_priority_weight(prio, regime=target_sign, priority_tiers=4)

    def test_saturated_entries_excluded_from_normaliser(self):
        # Synthetic ledger: five tier-2 saturated entries (huge weight
        # 1e8) plus one tier-4 unsaturated entry (small weight 1). The
        # unfiltered Σ a_j ≈ 5 × 1e8 + 1; filtered to unsaturated it is
        # just 1. Tier 1 is avoided here because its QP weight is 0
        # (hard-locked off-QP), which would trivialise the test.
        memory = {
            "a": (0.0, 10, 2, True),  # saturated tier-2 (weight 1e8)
            "b": (0.0, 10, 2, True),
            "c": (0.0, 10, 2, True),
            "d": (0.0, 10, 2, True),
            "e": (0.0, 10, 2, True),
            "f": (0.001, 10, 4, False),  # active tier-4 (weight 1)
        }
        target_sign = 1

        sum_all = sum(
            self._entry_responsiveness(int(v[2]), target_sign) for v in memory.values()
        )
        sum_free = sum(
            self._entry_responsiveness(int(v[2]), target_sign)
            for v in memory.values()
            if not v[3]
        )

        residual = 0.5  # arbitrary
        gamma = 0.6  # convergence rate
        step_all_entries = gamma * residual / sum_all
        step_free_only = gamma * residual / sum_free

        # The free-only step must be substantially larger — at least
        # 100× — because the saturated agents' weights dominated the
        # all-entries denominator.
        assert step_free_only > step_all_entries * 100

    def test_filter_with_no_saturated_entries_matches_legacy(self):
        # If nobody is saturated, the filtered and unfiltered sums must
        # coincide exactly (the filter is a no-op in the healthy regime).
        memory = {
            "a": (0.0, 10, 2, False),
            "b": (0.0, 10, 3, False),
            "c": (0.0, 10, 4, False),
        }
        sum_all = sum(self._entry_responsiveness(int(v[2]), 1) for v in memory.values())
        sum_free = sum(
            self._entry_responsiveness(int(v[2]), 1)
            for v in memory.values()
            if not v[3]
        )
        assert sum_all == sum_free


class TestPriorityFallbackEvent:
    def setup_method(self):
        arm()

    def teardown_method(self):
        _disarm()

    def test_fires_once_for_unregistered_load(self):
        b = _FakeBehavior()
        obs = {"p_mw": 5.0}  # load (positive cap), no priority registered
        # First call → fallback event.
        obs_priority(obs, behavior=b, aid="orphan", record_default_fallback_t=1.0)
        # Second call same aid → no extra event.
        obs_priority(obs, behavior=b, aid="orphan", record_default_fallback_t=2.0)
        events = [e for e in _drain_events() if e.kind == "priority_default_fallback"]
        assert len(events) == 1
        assert events[0].aid == "orphan"

    def test_no_event_for_generator(self):
        b = _FakeBehavior()
        obs = {"p_mw": -5.0}  # generator, defaults to tier 0 legitimately
        obs_priority(obs, behavior=b, aid="gen", record_default_fallback_t=1.0)
        events = [e for e in _drain_events() if e.kind == "priority_default_fallback"]
        assert events == []

    def test_no_event_when_registered(self):
        b = _FakeBehavior()
        register_priority(b, "ok", 3)
        obs = {"p_mw": 5.0}
        obs_priority(obs, behavior=b, aid="ok", record_default_fallback_t=1.0)
        events = [e for e in _drain_events() if e.kind == "priority_default_fallback"]
        assert events == []
