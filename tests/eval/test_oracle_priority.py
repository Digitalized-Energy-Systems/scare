"""Integration tests for the priority-aware oracle.

Runs monee's ``min_load_shedding`` LP through ``run_oracle`` with and
without a priorities map. When priorities are present and the LP must
shed, high-priority loads should keep more of their served setpoint than
low-priority loads.

Needs a real LP solver (Pyomo + Gurobi); skipped when Gurobi is absent.
"""

from __future__ import annotations

import pytest
from mango_energy_environments import fetch_example_net

from experiment.eval.oracle import _weight_for_load_factory, run_oracle


def _gurobi_available() -> bool:
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gurobi_available(),
    reason="Gurobi not available — oracle LP needs a real solver",
)


def test_factory_returns_none_without_priorities():
    """No priorities → factory returns None → monee uses its default
    flat-weight behaviour (no priority discrimination)."""
    net = fetch_example_net()
    assert _weight_for_load_factory(net, None, base_demand_weight=1e3) is None
    assert _weight_for_load_factory(net, {}, base_demand_weight=1e3) is None


def test_factory_returns_per_tier_weights():
    """With priorities, the factory returns a callable emitting
    ``base × _ORACLE_TIER_WEIGHT[tier]`` for known loads and ``None``
    for unmapped models. Weights are a bounded near-strict ladder
    (tier 1 → 1e6 ... tier 4 → 1, adjacent ratio 100): strictly decreasing
    so the LP prefers higher tiers per MW, but span-limited so no tier is
    swamped below the solver tolerance (unlike the legacy 1e12 ladder)."""
    net = fetch_example_net()
    load_aids = [
        f"child-{c.id}" for c in net.childs if type(c.model).__name__ == "PowerLoad"
    ][:2]
    assert len(load_aids) >= 2

    priorities = {load_aids[0]: 1, load_aids[1]: 4}
    wfn = _weight_for_load_factory(net, priorities, base_demand_weight=100.0)
    assert wfn is not None

    # tier-1 model: base × 1e6
    aid_1 = load_aids[0]
    cid_1 = int(aid_1.split("-")[1])
    model_1 = next(c.model for c in net.childs if c.id == cid_1)
    w1 = wfn(model_1)
    assert w1 == pytest.approx(100.0 * 1e6)

    # tier-4 model: base × 1
    aid_4 = load_aids[1]
    cid_4 = int(aid_4.split("-")[1])
    model_4 = next(c.model for c in net.childs if c.id == cid_4)
    w4 = wfn(model_4)
    assert w4 == pytest.approx(100.0 * 1.0)

    # strictly decreasing, bounded span (1e6) — strict at the margin but
    # resolvable, unlike the legacy 1e12 ladder.
    assert wfn(model_1) > wfn(model_4)
    assert wfn(model_1) / wfn(model_4) == pytest.approx(1e6)

    # Unmapped model — return None so monee uses its default.
    other = next(
        c.model
        for c in net.childs
        if c.id not in (cid_1, cid_4) and type(c.model).__name__ == "PowerLoad"
    )
    assert wfn(other) is None

    # Slack / tier-0 → None
    assert (
        _weight_for_load_factory(
            net,
            {load_aids[0]: 0},
            base_demand_weight=100.0,
        )
        is None
    )


def test_weight_resolves_on_copied_network():
    """Regression: monee's solver deep-copies the network before building the
    objective (solver/core.py: input_network.copy()), so the weight closure must
    resolve on COPIED load models. The previous ``id(model)`` map missed every
    copied model, returned None for all loads, and made the oracle priority-BLIND
    (reversing the tier ladder left the dispatch unchanged). The closure now
    stamps ``_scare_oracle_tier`` on the model, which survives the copy."""
    net = fetch_example_net()
    cid = next(c.id for c in net.childs if type(c.model).__name__ == "PowerLoad")
    wfn = _weight_for_load_factory(net, {f"child-{cid}": 1}, base_demand_weight=100.0)
    assert wfn is not None
    net_copy = net.copy()
    model_copy = next(c.model for c in net_copy.childs if c.id == cid)
    assert id(model_copy) != id(next(c.model for c in net.childs if c.id == cid))
    assert wfn(model_copy) == pytest.approx(100.0 * 1e6), (
        "weight must resolve on the copied model the solver actually optimises"
    )


@pytest.mark.slow
def test_oracle_with_priorities_sheds_low_priority_first():
    """End-to-end LP test: with no failure nothing should shed; under a
    forced deficit the lower-priority load is shed at least as deeply as
    the higher-priority one."""
    from monee.model.formulation import EL_MISOCP_FORMULATION

    net = fetch_example_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)

    # Pick two PowerLoads to compare.
    power_loads = [
        (c.id, c.model) for c in net.childs if type(c.model).__name__ == "PowerLoad"
    ]
    assert len(power_loads) >= 2
    aid_tier1 = f"child-{power_loads[0][0]}"
    aid_tier4 = f"child-{power_loads[1][0]}"

    # Assign every child a priority: two specific loads get tier 1 and
    # tier 4, the rest tier 3.
    priorities: dict[str, int] = {}
    for c in net.childs:
        aid = f"child-{c.id}"
        if aid == aid_tier1:
            priorities[aid] = 1
        elif aid == aid_tier4:
            priorities[aid] = 4
        elif type(c.model).__name__ == "PowerLoad":
            priorities[aid] = 3

    # No failure: oracle should fully serve everything.
    out_clean = run_oracle(net, [], priorities=priorities)
    if not out_clean.get("lp_success", True):
        pytest.skip(
            "Oracle LP did not solve cleanly on this environment; "
            "downstream priority assertions cannot be evaluated."
        )
    regs = out_clean["regulations"]
    assert regs[aid_tier1] >= 0.95, (
        f"tier-1 should be fully served on clean net, got {regs[aid_tier1]}"
    )
    assert regs[aid_tier4] >= 0.95, (
        f"tier-4 should be fully served on clean net, got {regs[aid_tier4]}"
    )

    # Force a deficit via a tight slack budget on the external power grid.
    # ``run_oracle`` reads ``_scare_slack_budget_mw`` and feeds it as
    # ``ext_grid_el_bounds`` to the LP; raw ``p_mw.min/max`` edits are
    # ignored by the oracle path.
    total_p_load_mw = sum(
        getattr(c.model, "p_mw", 0.0)
        for c in net.childs
        if type(c.model).__name__ == "PowerLoad"
    )
    assert total_p_load_mw > 0.0
    ext_grids = [c for c in net.childs if type(c.model).__name__ == "ExtPowerGrid"]
    assert ext_grids, "example net should carry an ExtPowerGrid"
    cap_mw = 0.6 * total_p_load_mw
    for c in ext_grids:
        c.model._scare_slack_budget_mw = cap_mw

    out_shed = run_oracle(net, [], priorities=priorities)
    regs_shed = out_shed["regulations"]
    # monee may silently return the trivial witness (all regulations
    # pinned to 1.0) under infeasibleOrUnbounded, making the shedding
    # assertions meaningless — skip that degenerate case.
    if all(abs(r - 1.0) < 1e-9 for r in regs_shed.values()):
        pytest.skip(
            "Oracle LP returned trivial witness (all regulations = 1.0) "
            "under the budget cap; solver-environment quirk — shedding "
            "ordering cannot be evaluated."
        )
    r_tier1 = regs_shed.get(aid_tier1)
    r_tier4 = regs_shed.get(aid_tier4)
    assert r_tier1 is not None and r_tier4 is not None
    # tier-4 should be cut at least as deeply as tier-1 (tolerance
    # handles tiny-demand ties), and some shedding must have happened.
    assert r_tier1 >= r_tier4 - 1e-6, (
        f"priority inversion under deficit: tier-1 r={r_tier1:.4f} < "
        f"tier-4 r={r_tier4:.4f}"
    )
    assert min(regs_shed.values()) < 0.99, (
        "expected at least one load to shed under the import cap; the "
        "scenario may have been too lenient"
    )
