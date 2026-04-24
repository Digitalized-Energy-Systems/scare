from __future__ import annotations

import random
from typing import Any

import numpy as np
from mango_energy_environments import Failure

from scare.base.model import Sector

HHV: float = 15.3  # MW / (kg/s) for natural gas

_CAPACITY_KEYS = (
    "p_mw",
    "q_w_set",
    "mass_flow",
    "p_kw",
    "q_mvar",
    "p_mw_capacity",
    "mass_flow_capacity",
)


def mw_to_kgps(value: float) -> float:
    return value / (3.6 * HHV)


def kgps_to_mw(value: float) -> float:
    return value * 3.6 * HHV


def obs_capacity(obs: dict) -> float:
    for key in _CAPACITY_KEYS:
        if key in obs:
            return float(obs[key])
    return 0.0


def obs_setpoint(obs: dict) -> float:
    return obs_capacity(obs) * float(obs.get("regulation", 1.0))


def obs_min_max(obs: dict) -> tuple[float, float]:
    """Return (delta_min, delta_max) relative to current setpoint."""
    cap = obs_capacity(obs)
    sp = obs_setpoint(obs)
    if cap < 0:
        return (cap - sp, -sp)
    else:
        return (-sp, cap - sp)


def sector_from_grid(grid: Any) -> Sector | None:
    """Resolve a Sector from a monee grid object via its .name attribute.

    Returns None for multi-grid nodes (e.g. CHPControlNode) because they
    straddle sectors and the sector has to be chosen explicitly by
    context.
    """
    if grid is None or isinstance(grid, (list, tuple)):
        return None
    name = str(getattr(grid, "name", "")).lower()
    if "power" in name:
        return Sector.ELECTRICITY
    if "gas" in name:
        return Sector.GAS
    if "water" in name or "heat" in name:
        return Sector.HEAT
    return None


def _sector_store(behavior: Any) -> dict[str, Sector]:
    store = getattr(behavior, "_scare_sectors", None)
    if store is None:
        store = {}
        behavior._scare_sectors = store
    return store


def register_sector(behavior: Any, aid: str, sector: Sector | None) -> None:
    if sector is not None:
        _sector_store(behavior)[aid] = sector


def lookup_sector(behavior: Any, aid: str) -> Sector | None:
    return _sector_store(behavior).get(aid)


def obs_sector(
    obs: dict,
    *,
    behavior: Any = None,
    aid: str | None = None,
) -> Sector | None:
    """Resolve the energy sector an observation belongs to.

    Preferred path: look up the (behavior, aid) pair in the sector
    registry populated at world-construction time.  The obs-key
    heuristic is retained only as a last-resort fallback — monee
    junction obs dicts are shape-identical between gas and water, so
    any inference from keys alone is unreliable.
    """
    if behavior is not None and aid is not None:
        found = lookup_sector(behavior, aid)
        if found is not None:
            return found
    if "p_mw" in obs or "p_kw" in obs or "p_mw_capacity" in obs:
        return Sector.ELECTRICITY
    if "q_w_set" in obs:
        return Sector.HEAT
    if "q_mvar" in obs and "p_mw" not in obs:
        return Sector.HEAT
    return None


def create_branch_aid(branch_id: tuple) -> str:
    a, b = branch_id[0], branch_id[1]
    hi, lo = (a, b) if a > b else (b, a)
    return f"branch-{hi}-{lo}"


def get_by_branch_id(centrality: dict, branch_id: tuple) -> float:
    if branch_id in centrality:
        return centrality[branch_id]
    rev = (branch_id[1], branch_id[0]) + branch_id[2:]
    return centrality.get(rev, 0.0)


def create_failures(
    monee_net: Any,
    failure_type: str = "branch",
    *,
    num_failures: int = 1,
    delay_s_max: float = 5.0,
) -> list[Failure]:
    if failure_type == "branch":
        candidates = [b for b in monee_net.branches if not b.model.is_cp()]
        selected = random.sample(candidates, min(num_failures, len(candidates)))
        return [
            Failure(delay_s=random.uniform(0.0, delay_s_max), branch_ids=[b.id])
            for b in selected
        ]
    return []


def efficiency_vector(eta_el: float, eta_heat: float, eta_gas: float) -> np.ndarray:
    return np.array([eta_el, eta_heat, eta_gas], dtype=float)


def create_chp_admm_flex_actor(chp_obs: dict, priority: int):
    """CHP: produces electricity + heat from gas."""
    from distributed_resource_optimization import create_admm_flex_actor_one_to_many

    cap = kgps_to_mw(float(chp_obs.get("gas_kgps", obs_capacity(chp_obs))))
    eta = efficiency_vector(
        chp_obs.get("eta_el", 0.35), chp_obs.get("eta_heat", 0.45), -1.0
    )
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def create_p2g_admm_flex_actor(p2g_obs: dict, priority: int):
    """P2G: consumes electricity, produces gas."""
    from distributed_resource_optimization import create_admm_flex_actor_one_to_many

    cap = float(p2g_obs.get("el_mw", obs_capacity(p2g_obs)))
    eta = efficiency_vector(-1.0, 0.0, p2g_obs.get("eta_gas", 0.6))
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def create_g2p_admm_flex_actor(g2p_obs: dict, priority: int):
    """G2P: consumes gas, produces electricity."""
    from distributed_resource_optimization import create_admm_flex_actor_one_to_many

    cap = kgps_to_mw(float(g2p_obs.get("gas_kgps", obs_capacity(g2p_obs))))
    eta = efficiency_vector(g2p_obs.get("eta_el", 0.45), 0.0, -1.0)
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def sector_color(sector: Sector) -> str:
    return {Sector.GAS: "green", Sector.HEAT: "red", Sector.ELECTRICITY: "orange"}[
        sector
    ]


# ---------------------------------------------------------------------------
# Grid-constraint observation helpers
# ---------------------------------------------------------------------------

# Keys in observation dicts that carry constraint-relevant quantities.
# These must match the keys returned by monee model.values, which are
# in per-unit / SI (Kelvin) — *not* in engineering units (bar, °C).
_CONSTRAINT_OBS_KEYS: dict[Sector, dict[str, str]] = {
    Sector.ELECTRICITY: {
        "vm_pu": "vm_pu",              # from Bus model
    },
    Sector.GAS: {
        "pressure_pu": "pressure_pu",  # from Junction model
    },
    Sector.HEAT: {
        "t_k": "t_k",                  # from Junction model (Kelvin)
    },
}


def obs_constraint_values(obs: dict, sector: Sector) -> dict[str, float]:
    """Extract grid-constraint measurements from an observation dict."""
    keys = _CONSTRAINT_OBS_KEYS.get(sector, {})
    result: dict[str, float] = {}
    for var, obs_key in keys.items():
        if obs_key in obs:
            result[var] = float(obs[obs_key])
    return result


def constraint_utilization(
    value: float, bound_low: float, bound_high: float
) -> float:
    """Return 0..1 indicating how close *value* is to violating a bound.

    0.0 = at the centre of the feasible range.
    1.0 = at or beyond a bound.
    """
    span = bound_high - bound_low
    if span <= 0:
        return 1.0
    mid = (bound_low + bound_high) / 2.0
    return min(1.0, abs(value - mid) / (span / 2.0))


def obs_priority(obs: dict) -> int:
    """Read an explicit priority value from an observation dict.

    Falls back to inferring priority from the agent type:
    generators (delta_min < 0) get priority 0, loads get 1.
    """
    if "priority" in obs:
        return int(obs["priority"])
    dmin, _ = obs_min_max(obs)
    return 0 if dmin < 0 else 1


def compute_priority_weighted_shares(
    demand_by_priority_per_group: list[dict[int, float]],
    served_by_priority_per_group: list[dict[int, float]],
    total_available: float,
) -> list[float]:
    """Compute each group's share of *total_available* via waterfall allocation.

    Starting from the highest-priority tier (lowest number), allocate
    proportionally to unserved demand within each tier until the budget
    is exhausted.  This guarantees that critical loads across all groups
    are served before any low-priority load receives resources.

    Returns a list of shares (one per group), summing to at most
    *total_available*.
    """
    n = len(demand_by_priority_per_group)
    shares = [0.0] * n
    if total_available <= 0 or n == 0:
        return shares

    all_tiers = sorted(
        {t for d in demand_by_priority_per_group for t in d}
    )
    remaining = total_available

    for tier in all_tiers:
        if remaining <= 1e-9:
            break
        tier_unserved = []
        for i in range(n):
            demand = demand_by_priority_per_group[i].get(tier, 0.0)
            served = served_by_priority_per_group[i].get(tier, 0.0)
            tier_unserved.append(max(0.0, demand - served))

        total_tier = sum(tier_unserved)
        if total_tier <= 1e-9:
            continue

        allocatable = min(remaining, total_tier)
        for i in range(n):
            share = allocatable * (tier_unserved[i] / total_tier)
            shares[i] += share
        remaining -= allocatable

    return shares


def aggregate_priority_weight(
    demand_by_priority: dict[int, float],
    served_by_priority: dict[int, float],
) -> float:
    """Compute a scalar urgency weight from priority-tier demand breakdown.

    Higher-priority tiers contribute exponentially more weight per unit of
    unserved demand.  This is used as the ADMM S parameter to pull
    allocation toward groups with critical unserved loads.
    """
    weight = 0.0
    for tier, demand in demand_by_priority.items():
        served = served_by_priority.get(tier, 0.0)
        unserved = max(0.0, demand - served)
        # Exponential weighting: tier 1 → 2^9=512, tier 10 → 2^0=1
        tier_weight = 2.0 ** max(0, 10 - tier)
        weight += unserved * tier_weight
    return weight


def clamp_to_constraints(
    setpoint: float,
    obs: dict,
    sector: Sector,
) -> float:
    """Clamp a proposed setpoint so it stays within local constraint bounds.

    This implements the "Conservative feasibility margins" MUST-requirement
    from improvements.txt §5.

    The clamping factor is derived from the constraint utilization:
    ``allowed_fraction = 1 - utilization``.  When a constraint variable is
    at the centre of its feasible range (utilization = 0), the full
    capacity is available.  As the variable approaches a bound
    (utilization → 1), the allowed fraction shrinks linearly to zero.
    This gives a smooth, monotonically decreasing response that
    degrades gracefully rather than switching between discrete steps.
    """
    from scare.base.model import SECTOR_CONSTRAINTS

    bounds = SECTOR_CONSTRAINTS.get(sector, {})
    cap = obs_capacity(obs)
    if cap == 0.0:
        return setpoint

    # Determine the tightest constraint across all local variables.
    tightest_fraction = 1.0
    for var, (lo, hi) in bounds.items():
        if var not in obs:
            continue
        val = float(obs[var])
        util = constraint_utilization(val, lo, hi)
        allowed = max(0.0, 1.0 - util)
        tightest_fraction = min(tightest_fraction, allowed)

    if tightest_fraction < 1.0:
        max_abs = tightest_fraction * abs(cap)
        setpoint = max(-max_abs, min(max_abs, setpoint))

    return setpoint
