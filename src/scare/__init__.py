"""SCARE – Scalable Community-based Adaptive Resilience for Energy Systems.

A distributed multi-agent system for resilient multi-energy restoration
built on top of mango-agents, mango-energy-environments, and
distributed-resource-optimization.

Key capabilities:
- Grid constraint enforcement (voltage, pressure, temperature)
- Priority-aware intra-sector load restoration
- Holonic (multi-level) community formation for scalability
- Sector-specific time-scale awareness
- Proactive curtailment signaling and multi-hop constraint propagation
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
    ResultService,
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
    "ResultService",
    "ConstraintViolation",
    "ConstraintWarning",
    "HolonicAssignment",
    "LocalGenerationRequest",
    "SECTOR_CONSTRAINTS",
    "SECTOR_TIMESCALE",
    "create_restoration_scenario_world",
    "start_restoration_simulation",
]
