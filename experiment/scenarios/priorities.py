"""Per-load priority-tier assignment for the 4-tier restoration model."""

import random

from scare.base.util import obs_capacity


def assign_load_priorities(
    monee_net: "object",
    *,
    seed: int = 0,
    distribution: str = "skewed",
) -> dict[str, int]:
    """Assign per-load priority tiers under the 4-tier model.

    Tier 1 = critical (hard-locked at the L1 leader pre-step);
    tier 2 = high, tier 3 = medium, tier 4 = sheddable (QP-weighted).

    Returns a ``priorities`` dict keyed by ``child-{id}`` for
    ``create_restoration_scenario_world(priorities=...)``. Generators
    (cap < 0) and CPs are skipped (default to tier 0 in ``obs_priority``).

    ``distribution`` knobs:

    - ``"uniform"``  — uniform over [1, 4]; maximally diverse.
    - ``"skewed"``    — realistic 10/30/40/20 % across tiers 1-4
      (default). The 10 % tier-1 share keeps the per-community supply
      pool able to cover hard-locked demand while leaving enough
      QP-weighted demand to discriminate L2 allocation.
    - ``"by_capacity"`` — large loads to tier 1, small to tier 4
      ("feed the big hospitals first").
    - ``"all_one"``  — everyone tier 1; hard-locks every load
      (typically infeasible, triggers pro-rata branch). Ablation knob.

    ``seed`` makes assignments deterministic.
    """
    rng = random.Random(seed * 7919 + 31)
    P = 4
    out: dict[str, int] = {}

    for child in monee_net.childs:
        # Capacity straight from the model (no behavior yet).
        obs = dict(child.model.values)
        cap = obs_capacity(obs)
        if cap <= 0:
            continue  # generators / unknown default to tier 0
        aid = f"child-{child.id}"
        if distribution == "uniform":
            out[aid] = rng.randint(1, P)
        elif distribution == "skewed":
            r = rng.random()
            if r < 0.10:
                out[aid] = 1  # critical (hard-locked)
            elif r < 0.40:
                out[aid] = 2  # high
            elif r < 0.80:
                out[aid] = 3  # medium
            else:
                out[aid] = 4  # sheddable
        elif distribution == "by_capacity":
            # Resolved in the second pass (needs the global distribution).
            out[aid] = -1  # sentinel
        elif distribution == "all_one":
            out[aid] = 1
        else:
            raise ValueError(f"unknown priority distribution: {distribution}")

    if distribution == "by_capacity":
        # Bin by capacity quartile: top to tier 1, bottom to tier 4.
        items = []
        for child in monee_net.childs:
            obs = dict(child.model.values)
            cap = obs_capacity(obs)
            if cap > 0:
                items.append((cap, f"child-{child.id}"))
        items.sort(reverse=True)  # largest first
        n = len(items)
        for rank, (_cap, aid) in enumerate(items):
            tier = 1 + int(rank * P / max(n, 1))
            out[aid] = max(1, min(P, tier))

    return out
