"""Unit conversions and sector name/colour resolution."""

from __future__ import annotations

from typing import Any

import numpy as np

from scare.base.model import Sector

# Natural gas HHV. MW/(kg/s) factor is 3.6*HHV, not HHV itself.
# Must match the fluid of the simulated grids: all benchmark nets are built
# with gas_type="lgas" (monee model/grid.py), not hgas (15.3).
HHV: float = 11.79011  # kWh/kg (lgas)


def mw_to_kgps(value: float) -> float:
    return value / (3.6 * HHV)


def kgps_to_mw(value: float) -> float:
    return value * 3.6 * HHV


def efficiency_vector(eta_el: float, eta_heat: float, eta_gas: float) -> np.ndarray:
    return np.array([eta_el, eta_heat, eta_gas], dtype=float)


def sector_color(sector: Sector) -> str:
    return {Sector.GAS: "green", Sector.HEAT: "red", Sector.ELECTRICITY: "orange"}[
        sector
    ]


def sector_from_grid(grid: Any) -> Sector | None:
    """Resolve a Sector from a monee grid object via its .name attribute.

    Returns None for multi-grid nodes (e.g. CHPControlNode) which straddle
    sectors and must be resolved by context.
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
