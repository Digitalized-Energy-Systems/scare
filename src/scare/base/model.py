from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import UUID


class Sector(str, Enum):
    ELECTRICITY = "electricity"
    HEAT = "heat"
    GAS = "gas"


class SystemStrategy(Enum):
    CP_TO_GROUP = auto()
    GROUP_TO_CP = auto()
    SIMULTANEOUSLY = auto()


@dataclass
class EnergyData:
    electricity: dict[str, float] = field(default_factory=dict)
    gas: dict[str, float] = field(default_factory=dict)
    heat: dict[str, float] = field(default_factory=dict)

    def get_sector(self, sector: Sector) -> dict[str, float]:
        match sector:
            case Sector.ELECTRICITY:
                return self.electricity
            case Sector.GAS:
                return self.gas
            case Sector.HEAT:
                return self.heat
        raise ValueError(f"Unknown sector: {sector}")


@dataclass
class CommunityAssignment:
    community_id: UUID | None = None
    neighbors: list[Any] = field(default_factory=list)


@dataclass
class ResultService:
    aid_to_result: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class NegotiationFinishedEvent:
    new_setpoint: float
    sector: Sector


@dataclass(frozen=True)
class OptimizationFinishedLocalEvent:
    result: list[float]  # [el_mw, heat_w, gas_kgps]


@dataclass
class LineFailure:
    source_node_id: Any
    target_node_id: Any
    branch_id: tuple


@dataclass
class AskEnergyMessage:
    negotiation_id: str
    sector: Sector


@dataclass
class ResponseEnergyMessage:
    negotiation_id: str
    setpoint: float
    available: float
    name: str = ""


@dataclass
class EnergyNegotiationMessage:
    negotiation_id: str
    sector: Sector
    negotiation_target: float
    current_delta: float
    counter: int
    negotiation_memory: dict[str, tuple[float, int]]


@dataclass
class GridPathMessage:
    source_addr: Any
    target_addr: Any
    path: list[Any]
    asked_agents: list[Any]
    uncertain_connections: list[tuple[Any, Any]]


@dataclass
class GridPathResult:
    path: list[Any]
    uncertain_connections: list[tuple[Any, Any]]


@dataclass
class AskForAvailableFlex:
    include_connectors: bool = False


@dataclass
class AvailableFlexAnswer:
    flex: float
    balance: float
    shedded: float
    sector: Sector


@dataclass
class StartBalanceNegotiation:
    pass


@dataclass
class BalanceNegotiationStart:
    community_id: UUID
    target: list[float]


@dataclass
class NegotiationResult:
    flexibility: float
    control_setpoint: float


@dataclass
class BalanceProblem:
    sector: Sector
    imbalance: float


@dataclass
class CHSJoinRequest:
    group_id: UUID
    group_size: int


@dataclass
class CHSJoinRequestAnswer:
    group_id: UUID
    accept: bool
