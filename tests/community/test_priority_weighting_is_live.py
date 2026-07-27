"""``enable_priority_weighting=False`` must actually change the allocation.

It did not. The waterfall visits cells via
``np.argsort(-priorities, kind="stable")``, and cells are laid out
tier-ascending (``_flat_idx = sec_idx*n_tier + tier_idx`` over a *sorted*
``tiers`` list). With weighting on, the strictly-monotone weight orders the
visit tier 1 -> P. With weighting off every weight is 1.0, so the **stable**
tie-break falls back to cell index — which is the same tier 1 -> P order. Both
settings produced byte-identical allocations, which is why every
``enable_priority_holon_allocation=False`` arm in eval_full_v2 came back
bit-identical to its baseline despite the mechanism issuing ~2584 regulates per
task in the same experiment.

The priority-blind counterfactual is pro-rata: one common service fraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from scare.community.supply_priority_admm import (
    _prorata_target,
    _waterfall_target,
    allocate_supply_priority,
)


def test_flat_weights_reproduce_the_priority_order_in_the_waterfall():
    """The defect itself, pinned so it cannot silently return."""
    demand = np.array([1.0, 1.0, 1.0, 1.0])
    monotone = np.array([4.0, 3.0, 2.0, 1.0])  # tier 1 -> 4
    flat = np.ones(4)
    supply = 2.5
    assert np.allclose(
        _waterfall_target(demand, monotone, supply),
        _waterfall_target(demand, flat, supply),
    ), "flattening the weights must not be mistaken for disabling priority"


def test_prorata_is_priority_blind():
    demand = np.array([1.0, 1.0, 1.0, 1.0])
    out = _prorata_target(demand, 2.0)
    assert np.allclose(out, [0.5, 0.5, 0.5, 0.5])
    assert out.sum() == pytest.approx(2.0)


def test_prorata_never_over_serves_when_supply_exceeds_demand():
    demand = np.array([1.0, 2.0])
    assert np.allclose(_prorata_target(demand, 99.0), demand)


def test_prorata_degenerates_safely():
    assert np.allclose(_prorata_target(np.zeros(3), 5.0), np.zeros(3))
    assert np.allclose(_prorata_target(np.array([1.0]), 0.0), np.zeros(1))


@pytest.mark.asyncio
async def test_disabling_priority_weighting_changes_the_service_fractions():
    """End-to-end through the short-circuit path component scope always takes."""
    kwargs = dict(
        sectors=["electricity"],
        tiers=[1, 2, 3, 4],
        actor_supplies=[{"electricity": 2.0}],
        actor_demands=[{"electricity": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}}],
        priority_tiers=4,
    )
    prio, _, meta = await allocate_supply_priority(
        **kwargs, enable_priority_weighting=True
    )
    blind, _, _ = await allocate_supply_priority(
        **kwargs, enable_priority_weighting=False
    )
    assert meta["short_circuit"] == "waterfall", "expected the short-circuit path"

    p, b = prio["electricity"], blind["electricity"]
    assert p != b, "the flag must change the allocation"
    # Priority-aware: strictly serves tier 1 before tier 4 under a 2-of-4 deficit.
    assert p[1] == pytest.approx(1.0) and p[4] == pytest.approx(0.0)
    # Priority-blind: one common fraction across every tier.
    assert len({round(v, 9) for v in b.values()}) == 1
    assert b[1] == pytest.approx(0.5)
    # Both allocate the whole pool.
    assert sum(p.values()) == pytest.approx(sum(b.values()))
