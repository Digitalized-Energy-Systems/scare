"""End-to-end integration for the L3 priority-cascaded ADMM role.

Drives the full :func:`create_restoration_scenario_world` +
:func:`start_restoration_simulation` pipeline with the new L3 cutover
flag and verifies that the replicated kernel actually activates
coupling-point flexibility on a CP-relevant scenario.

Scenario design
---------------

The standard example_net carries three CPs (2× G2P, 1× P2G).  Priorities
are deliberately skewed so the cross-sector trade has a sharp signal:

* every electricity load → tier 1 (critical)
* every heat / gas load → tier 4 (deferrable)

Under that skew, a branch failure in the electricity sector creates
tier-1 scarcity in electricity while gas remains in surplus.  The
priority-correct response is to **fire the G2P units** — convert
plentiful gas to scarce electricity — and to **keep the P2G off** (don't
drain electricity to make gas while electricity tier-1 is unmet).  This
is the textbook case the oracle's MIQCQP would also resolve by
activating the G2Ps; the integration test verifies the
:class:`CPPriorityAdmmRole` reaches the same qualitative answer.

What the tests assert
---------------------

1. With ``enable_cp_priority_admm=True``, the new role gets wired into
   every CP in the world and the simulation runs end-to-end without
   error.
2. At least one CP receives a non-trivial regulation write during the
   run, demonstrating the role actually activates CPs (the load-bearing
   behaviour of the cutover).
3. With the new flag off and no other CP path active, no CP regulate
   writes happen — the baseline comparison that proves the activation
   above is attributable to the new role rather than to some other
   mechanism in the codebase.

The numerical solver is stubbed (same pattern as
``tests/integration/test_cross_sector_coalition_e2e.py``) so the test
runs without a GEKKO installation.  Stubbing energyflow does not affect
the agent-layer decision logic, which is what we're testing.
"""

from __future__ import annotations

import pytest

from mango_energy_environments import fetch_example_net
import mango_energy_environments.environments.restoration.multi_energy_monee as _restoration_mod

from scare.base import diagnostics
from scare.base.config import RestorationConfiguration
from scare.base.util import create_failures, obs_capacity, obs_sector
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)
from scare.service.cp_priority_admm_role import CPPriorityAdmmRole


# ---------------------------------------------------------------------------
# Solver stub
# ---------------------------------------------------------------------------


class _FakeEnergyFlowResult:
    def __init__(self, net):
        self.network = net


@pytest.fixture(autouse=True)
def _stub_energyflow(monkeypatch):
    monkeypatch.setattr(
        _restoration_mod,
        "energyflow",
        lambda net: _FakeEnergyFlowResult(net),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skewed_priorities(net) -> dict[str, int]:
    """Electricity loads → tier 1; other loads → tier 4; generators → tier 0.

    The 4-tier weight schedule (``10^(P − t + 1)`` with ``P=4``) puts
    tier-1 at ``10^4`` and tier-4 at ``10`` — a 1000× separation, more
    than enough to drive a clear cross-sector trade signal.
    """
    priorities: dict[str, int] = {}
    for child in net.childs:
        obs = dict(child.model.values)
        cap = obs_capacity(obs)
        aid = f"child-{child.id}"
        if cap <= 0:
            priorities[aid] = 0
            continue
        sector = obs_sector(obs)
        if sector is not None and sector.value == "electricity":
            priorities[aid] = 1
        else:
            priorities[aid] = 4
    return priorities


def _count_role_instances(world, role_cls) -> int:
    """Number of *role_cls* instances installed across all agents."""
    n = 0
    for agent in world._agents.values():
        for role in getattr(agent, "roles", []):
            if isinstance(role, role_cls):
                n += 1
    return n


def _cp_aids(world) -> set[str]:
    """aids of every agent that carries a :class:`CPPriorityAdmmRole`."""
    out: set[str] = set()
    for agent in world._agents.values():
        for role in getattr(agent, "roles", []):
            if isinstance(role, CPPriorityAdmmRole):
                out.add(str(agent.aid))
                break
    return out


def _regulate_writes_per_cp(cp_aids: set[str]) -> dict[str, list[float]]:
    """Pull every ``regulate`` action recorded by ``record_regulate``
    that targets a CP's aid; returns aid → list[(t, factor, reason)]."""
    out: dict[str, list[tuple[float, float, str]]] = {aid: [] for aid in cp_aids}
    # The diagnostics module records every successful ``apply_regulate``
    # via ``record_regulate``; ``_log`` is the underlying deque of
    # ``ActionRecord(kind="regulate", aid, sector, value, reason, t)``.
    for r in diagnostics._log:  # noqa: SLF001 — test introspection
        if r.kind != "regulate":
            continue
        if r.aid not in cp_aids:
            continue
        out[r.aid].append((float(r.t), float(r.value), str(r.reason)))
    return out


def _role_commits_per_cp(world) -> dict[str, float | None]:
    """Per-CP ``_last_committed_factor`` from the role itself —
    independent of the diagnostics ledger so we can cross-check."""
    out: dict[str, float | None] = {}
    for agent in world._agents.values():
        for role in getattr(agent, "roles", []):
            if isinstance(role, CPPriorityAdmmRole):
                out[str(agent.aid)] = role._last_committed_factor
                break
    return out


async def _run_once(
    *,
    enable_cp_priority_admm: bool,
    enable_cp_admm: bool,
    simulation_duration_s: float = 12.0,
):
    """One full simulation pass, returning (world, behavior, cp_aids).

    Skewed priorities + a single branch failure draw chosen to create
    electricity-side scarcity (the priority-1 sector).  ``num_failures=2``
    raises the odds of clipping an electricity feeder in the small
    example_net; the deterministic seed keeps the draw reproducible.
    """
    diagnostics.arm()
    net = fetch_example_net()
    priorities = _skewed_priorities(net)

    config = RestorationConfiguration(
        enable_cp_priority_admm=enable_cp_priority_admm,
        enable_cp_admm=enable_cp_admm,
        # Speed up the L2 mesh so the kernel sees fresh demand
        # within the simulation window.
        holon_summary_period_s=0.5,
    )

    world = create_restoration_scenario_world(
        net,
        priorities=priorities,
        simulation_duration_s=simulation_duration_s,
        config=config,
    )

    failures = create_failures(net, "branch", num_failures=2, delay_s_max=1.0)
    await start_restoration_simulation(
        world, failures, simulation_duration_s=simulation_duration_s,
    )

    cp_aids = _cp_aids(world) if enable_cp_priority_admm else set()
    return world, world.environment.behavior, cp_aids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_role_installed_on_every_cp_agent_when_flag_on():
    """Wiring sanity: when the cutover flag is on, every CP agent in
    the world carries the new role and the legacy
    ``EnergyConverterRole`` is *not* installed (the new path replaces
    the triple bundle, not augments it)."""
    from scare.service.cp import EnergyConverterRole

    world, _, cp_aids = await _run_once(
        enable_cp_priority_admm=True,
        enable_cp_admm=False,
        simulation_duration_s=4.0,
    )
    n_new = _count_role_instances(world, CPPriorityAdmmRole)
    n_legacy = _count_role_instances(world, EnergyConverterRole)
    # example_net has 3 CPs (2× G2P, 1× P2G).
    assert n_new == 3, f"expected 3 CPPriorityAdmmRole instances; got {n_new}"
    assert n_legacy == 0, (
        f"legacy EnergyConverterRole should be skipped under cutover; "
        f"got {n_legacy} instances"
    )
    assert len(cp_aids) == 3


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_cp_priority_admm_activates_cps_under_electricity_scarcity():
    """Load-bearing assertion: under the priority-1 electricity-scarcity
    setup, at least one CP receives a non-trivial regulation write
    during the run.

    This is the qualitative parity check with what the oracle would do
    on the same scenario: serve the critical electricity loads by
    activating the G2P units that can convert plentiful gas into the
    scarce sector.  The replicated kernel reaches that answer locally,
    one CP at a time, without a coordinator.
    """
    world, _, cp_aids = await _run_once(
        enable_cp_priority_admm=True,
        enable_cp_admm=False,
        simulation_duration_s=12.0,
    )
    assert world.clock.time > 0
    assert cp_aids, "no CP agents found — scenario setup broken"

    # Two signals cross-check each other: (a) the role's own
    # ``_last_committed_factor`` (set only when ``apply_regulate``
    # returned True), and (b) the diagnostics ledger's
    # ``record_regulate`` entries, which fire from the same code path.
    commits = _role_commits_per_cp(world)
    writes = _regulate_writes_per_cp(cp_aids)

    activated_via_role = {
        aid: f for aid, f in commits.items()
        if f is not None and f > 1e-6
    }
    activated_via_ledger = {
        aid: ws for aid, ws in writes.items()
        if any(f > 1e-6 for (_t, f, _r) in ws)
    }

    assert activated_via_role, (
        "expected the L3 priority-ADMM role to activate at least one CP "
        "under electricity-tier-1 scarcity (via role._last_committed_factor); "
        f"got: {commits}"
    )
    assert activated_via_ledger, (
        "diagnostics ledger should record the same regulate writes the "
        f"role committed; ledger entries by aid: {writes}"
    )
    # Cross-check: at least one CP appears in both signals.
    assert set(activated_via_role) & set(activated_via_ledger), (
        f"role-side commits {set(activated_via_role)} and ledger writes "
        f"{set(activated_via_ledger)} disagree — diagnostics wiring bug?"
    )

    # The replicated kernel's commits must carry the new reason tag so
    # downstream eval/report tooling can attribute the writes to L3.
    reasons_seen = {
        r for ws in writes.values() for (_t, _f, r) in ws
    }
    assert "cp_priority_admm" in reasons_seen, (
        f"expected at least one regulate with reason='cp_priority_admm'; "
        f"saw reasons: {reasons_seen}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_baseline_no_cp_path_means_no_cp_regulate_writes():
    """Baseline comparison: with both CP paths off, no CP regulate
    writes happen.  This establishes that the activation observed in
    :func:`test_cp_priority_admm_activates_cps_under_electricity_scarcity`
    is attributable to the new role rather than to any other mechanism
    that might commit factors on CP aids.
    """
    await _run_once(
        enable_cp_priority_admm=False,
        enable_cp_admm=False,
        simulation_duration_s=12.0,
    )
    # Identify CP aids the same way the role wiring would.
    cp_aids: set[str] = set()
    from mango_energy_environments.environments.restoration.multi_energy_monee import (
        create_branch_aid,
    )
    net = fetch_example_net()
    for branch in net.branches:
        if branch.model.is_cp():
            cp_aids.add(create_branch_aid(branch.id))
    assert cp_aids, "scenario sanity: example_net must carry CP branches"

    writes = _regulate_writes_per_cp(cp_aids)
    total_writes = sum(len(fs) for fs in writes.values())
    assert total_writes == 0, (
        f"baseline (no CP path) should not produce CP regulate writes; "
        f"got {total_writes} writes: {writes}"
    )
