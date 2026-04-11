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


def obs_sector(obs: dict) -> Sector | None:
    if "p_mw" in obs or "p_kw" in obs or "p_mw_capacity" in obs:
        return Sector.ELECTRICITY
    if "mass_flow" in obs or "mass_flow_capacity" in obs:
        return Sector.GAS
    if "q_w_set" in obs or "q_mvar" in obs:
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
