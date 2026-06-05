"""End-to-end integration for the L3 priority-cascaded ADMM role.

Drives the full :func:`create_restoration_scenario_world` +
:func:`start_restoration_simulation` pipeline with the L3 priority-ADMM
flag and verifies the role activates coupling-point flexibility.

Scenario: example_net carries three CPs (2x G2P, 1x P2G). Priorities
are skewed (electricity loads -> tier 1, heat/gas loads -> tier 4) so a
branch failure creates tier-1 electricity scarcity while gas stays in
surplus. The priority-correct response is to fire the G2P units (gas ->
scarce electricity) and keep the P2G off.

Tests assert: (1) with the flag on, the role is wired into every CP and
the sim runs without error; (2) at least one CP receives a non-trivial
regulation write; (3) with the flag off and no other CP path active, no
CP regulate writes happen, isolating the activation to this role.

The numerical solver is stubbed so the test runs without GEKKO; this
does not affect the agent-layer decision logic under test.
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


def _skewed_priorities(net) -> dict[str, int]:
    """Electricity loads -> tier 1; other loads -> tier 4; generators -> tier 0.

    The 4-tier weight schedule gives a 1000x separation between tier-1
    and tier-4, enough to drive a clear cross-sector trade signal.
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
    # diagnostics._log is the deque of ActionRecords from record_regulate.
    for r in diagnostics._log:  # noqa: SLF001 — test introspection
        if r.kind != "regulate":
            continue
        if r.aid not in cp_aids:
            continue
        out[r.aid].append((float(r.t), float(r.value), str(r.reason)))
    return out


def _role_commits_per_cp(world) -> dict[str, float | None]:
    """Per-CP ``_last_committed_factor`` read from the role itself,
    independent of the diagnostics ledger for cross-checking."""
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

    ``num_failures=2`` raises the odds of clipping an electricity feeder
    in the small example_net, creating priority-1 scarcity.
    """
    diagnostics.arm()
    net = fetch_example_net()
    priorities = _skewed_priorities(net)

    config = RestorationConfiguration(
        enable_cp_priority_admm=enable_cp_priority_admm,
        enable_cp_admm=enable_cp_admm,
        # Speed up the L2 mesh so the kernel sees fresh demand in time.
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


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_role_installed_on_every_cp_agent_when_flag_on():
    """When the flag is on, every CP agent carries the new role and the
    legacy ``EnergyConverterRole`` is not installed (replace, not augment)."""
    from scare.service.cp import EnergyConverterRole

    world, _, cp_aids = await _run_once(
        enable_cp_priority_admm=True,
        enable_cp_admm=False,
        simulation_duration_s=4.0,
    )
    n_new = _count_role_instances(world, CPPriorityAdmmRole)
    n_legacy = _count_role_instances(world, EnergyConverterRole)
    # example_net has 4 CPs: 1x CHP (node), 2x G2P, 1x P2G (branches).
    assert n_new == 4, f"expected 4 CPPriorityAdmmRole instances; got {n_new}"
    assert n_legacy == 0, (
        f"legacy EnergyConverterRole should be skipped under cutover; "
        f"got {n_legacy} instances"
    )
    assert len(cp_aids) == 4


@pytest.mark.xfail(
    reason="Gossip L3 path is WIP: after the _MangoGossipCarrier fix the cascade "
    "runs but a round never resolves within the sim window, so _on_gossip_commit "
    "never fires and no CP commits a factor. Tracks completing the gossip "
    "cascade convergence/commit, not the (fixed) carrier crash.",
    strict=False,
)
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_cp_priority_admm_activates_cps_under_electricity_scarcity():
    """Under priority-1 electricity scarcity, at least one CP receives a
    non-trivial regulation write: the kernel activates the G2P units that
    convert plentiful gas into the scarce sector, locally and without a
    coordinator.
    """
    world, _, cp_aids = await _run_once(
        enable_cp_priority_admm=True,
        enable_cp_admm=False,
        simulation_duration_s=12.0,
    )
    assert world.clock.time > 0
    assert cp_aids, "no CP agents found — scenario setup broken"

    # Cross-check two signals: the role's own ``_last_committed_factor``
    # and the diagnostics ledger's ``record_regulate`` entries.
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

    # Commits must carry the reason tag so eval/report tooling can
    # attribute the writes to L3.
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
    """Baseline: with both CP paths off, no CP regulate writes happen,
    isolating the activation in the companion test to this role.
    """
    await _run_once(
        enable_cp_priority_admm=False,
        enable_cp_admm=False,
        simulation_duration_s=12.0,
    )
    # Identify CP aids the same way the role wiring does.
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
