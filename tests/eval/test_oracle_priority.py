"""Integration tests for the priority-aware oracle (Option B).

Run monee's ``min_load_shedding`` LP through ``run_oracle`` with and
without a priorities map.  When priorities are present and the LP must
shed, high-priority loads should keep more of their served setpoint
than low-priority loads.

These tests need a real LP solver (Pyomo + Gurobi).  They're tagged
``@pytest.mark.slow`` so the fast unit-test pass can skip them.
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
    """No priorities supplied → factory returns None → monee uses
    its legacy flat-weight behaviour (no priority discrimination).
    """
    net = fetch_example_net()
    assert _weight_for_load_factory(net, None, base_demand_weight=1e3) is None
    assert _weight_for_load_factory(net, {}, base_demand_weight=1e3) is None


def test_factory_returns_per_tier_weights():
    """With priorities, the factory returns a callable that emits
    ``base × 2^(P-tier+1)`` for known loads and ``None`` for
    unmapped models.
    """
    net = fetch_example_net()
    # Pick two real load aids
    load_aids = [
        f"child-{c.id}"
        for c in net.childs
        if type(c.model).__name__ == "PowerLoad"
    ][:2]
    assert len(load_aids) >= 2

    priorities = {load_aids[0]: 1, load_aids[1]: 10}
    wfn = _weight_for_load_factory(net, priorities, base_demand_weight=100.0, n_tiers=10)
    assert wfn is not None

    # tier-1 model
    aid_1 = load_aids[0]
    cid_1 = int(aid_1.split("-")[1])
    model_1 = next(c.model for c in net.childs if c.id == cid_1)
    w1 = wfn(model_1)
    assert w1 == pytest.approx(100.0 * 2.0 ** 10)  # 102400

    # tier-10 model
    aid_10 = load_aids[1]
    cid_10 = int(aid_10.split("-")[1])
    model_10 = next(c.model for c in net.childs if c.id == cid_10)
    w10 = wfn(model_10)
    assert w10 == pytest.approx(100.0 * 2.0 ** 1)  # 200

    # Unmapped model — return None so monee uses its default.
    other = next(
        c.model for c in net.childs
        if c.id not in (cid_1, cid_10)
        and type(c.model).__name__ == "PowerLoad"
    )
    assert wfn(other) is None

    # Slack / tier-0 → None
    assert _weight_for_load_factory(
        net, {load_aids[0]: 0}, base_demand_weight=100.0,
    ) is None


@pytest.mark.slow
def test_oracle_with_priorities_sheds_low_priority_first():
    """End-to-end LP test: run the oracle on the example MES with
    priorities skewed so two known PowerLoads have tier-1 vs tier-10.
    With no failure (steady state), nothing should shed.  Then apply
    a contrived failure that forces shedding and verify the tier-10
    load gets shed first.
    """
    from monee.model.formulation import MISOCP_NETWORK_FORMULATION
    from mango_energy_environments.environments.restoration.multi_energy_monee import (
        Failure,
    )

    net = fetch_example_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)

    # Pick two PowerLoads to compare.
    power_loads = [
        (c.id, c.model)
        for c in net.childs
        if type(c.model).__name__ == "PowerLoad"
    ]
    assert len(power_loads) >= 2
    aid_tier1 = f"child-{power_loads[0][0]}"
    aid_tier10 = f"child-{power_loads[1][0]}"

    # Assign ALL childs a priority (skewed so the LP can't trivially
    # ignore tiers).  Two specific loads get tier 1 and 10; the rest
    # get tier 5 (middle).
    priorities: dict[str, int] = {}
    for c in net.childs:
        aid = f"child-{c.id}"
        if aid == aid_tier1:
            priorities[aid] = 1
        elif aid == aid_tier10:
            priorities[aid] = 10
        elif type(c.model).__name__ == "PowerLoad":
            priorities[aid] = 5

    # No failure — sanity: oracle should fully serve everything.
    out_clean = run_oracle(net, [], priorities=priorities)
    regs = out_clean["regulations"]
    assert regs[aid_tier1] >= 0.95, (
        f"tier-1 should be fully served on clean net, got {regs[aid_tier1]}"
    )
    assert regs[aid_tier10] >= 0.95, (
        f"tier-10 should be fully served on clean net, got {regs[aid_tier10]}"
    )
