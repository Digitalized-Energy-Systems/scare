"""Thin wrappers around ``distributed_resource_optimization.create_admm_flex_actor_one_to_many``
for each cross-sector coupling plant type (CHP / P2G / G2P / P2H).

Each wrapper resolves the device's capacity in MW (or kg/s converted to
MW) and packages it with a sector-efficiency vector + uniform priority.
"""

from __future__ import annotations

import numpy as np
from distributed_resource_optimization import create_admm_flex_actor_one_to_many

from scare.base.util import efficiency_vector, kgps_to_mw, obs_capacity


def create_chp_admm_flex_actor(chp_obs: dict, priority: int):
    """CHP: gas → electricity + heat."""
    cap = kgps_to_mw(float(chp_obs.get("gas_kgps", obs_capacity(chp_obs))))
    eta = efficiency_vector(
        chp_obs.get("eta_el", 0.35), chp_obs.get("eta_heat", 0.45), -1.0
    )
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def create_p2g_admm_flex_actor(p2g_obs: dict, priority: int):
    """P2G: electricity → gas."""
    cap = float(p2g_obs.get("el_mw", obs_capacity(p2g_obs)))
    eta = efficiency_vector(-1.0, 0.0, p2g_obs.get("eta_gas", 0.6))
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def create_g2p_admm_flex_actor(g2p_obs: dict, priority: int):
    """G2P: gas → electricity."""
    cap = kgps_to_mw(float(g2p_obs.get("gas_kgps", obs_capacity(g2p_obs))))
    eta = efficiency_vector(g2p_obs.get("eta_el", 0.45), 0.0, -1.0)
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def create_p2h_admm_flex_actor(p2h_obs: dict, priority: int):
    """P2H: electricity → heat (high- or low-grade)."""
    cap = float(p2h_obs.get("el_mw", obs_capacity(p2h_obs)))
    eta = efficiency_vector(-1.0, p2h_obs.get("eta_heat", 0.9), 0.0)
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))
