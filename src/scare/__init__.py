"""SCARE – Scalable Community-based Adaptive Resilience for Energy Systems.

A distributed multi-agent system for resilient multi-energy restoration, built
on mango-agents, mango-energy-environments, and distributed-resource-optimization.
"""

from scare.base.model import (
    SECTOR_CONSTRAINTS,
    SECTOR_TIMESCALE,
    CommunityAssignment,
    ConstraintViolation,
    ConstraintWarning,
    EnergyData,
    HolonicAssignment,
    LocalGenerationRequest,
    NegotiationFinishedEvent,
    Sector,
    SystemStrategy,
)
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)

__all__ = [
    "Sector",
    "SystemStrategy",
    "EnergyData",
    "NegotiationFinishedEvent",
    "CommunityAssignment",
    "ConstraintViolation",
    "ConstraintWarning",
    "HolonicAssignment",
    "LocalGenerationRequest",
    "SECTOR_CONSTRAINTS",
    "SECTOR_TIMESCALE",
    "create_restoration_scenario_world",
    "start_restoration_simulation",
]
