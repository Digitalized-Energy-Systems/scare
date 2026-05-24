"""Unit tests for scare.base.model — data models and constants."""

import pytest

from scare.base.model import (
    ConstraintViolation,
    EnergyData,
    EnergyNegotiationMessage,
    LocalGenerationRequest,
    NegotiationFinishedEvent,
    SECTOR_CONSTRAINTS,
    SECTOR_TIMESCALE,
    Sector,
)


class TestSectorEnum:
    def test_values(self):
        assert Sector.ELECTRICITY.value == "electricity"
        assert Sector.HEAT.value == "heat"
        assert Sector.GAS.value == "gas"

    def test_is_str(self):
        assert isinstance(Sector.ELECTRICITY, str)
        assert Sector.GAS == "gas"


class TestSectorConstraints:
    def test_all_sectors_present(self):
        assert set(SECTOR_CONSTRAINTS.keys()) == {
            Sector.ELECTRICITY, Sector.GAS, Sector.HEAT,
        }

    def test_electricity_bounds(self):
        lo, hi = SECTOR_CONSTRAINTS[Sector.ELECTRICITY]["vm_pu"]
        assert lo == 0.95 and hi == 1.05

    def test_gas_bounds(self):
        lo, hi = SECTOR_CONSTRAINTS[Sector.GAS]["pressure_pu"]
        assert lo == 0.90 and hi == 1.10

    def test_heat_bounds(self):
        lo, hi = SECTOR_CONSTRAINTS[Sector.HEAT]["t_k"]
        assert lo == 283.15 and hi == 403.15


class TestSectorTimescale:
    def test_all_sectors_present(self):
        assert set(SECTOR_TIMESCALE.keys()) == {
            Sector.ELECTRICITY, Sector.GAS, Sector.HEAT,
        }

    def test_electricity_fastest(self):
        assert SECTOR_TIMESCALE[Sector.ELECTRICITY]["poll_period_s"] < \
               SECTOR_TIMESCALE[Sector.GAS]["poll_period_s"] < \
               SECTOR_TIMESCALE[Sector.HEAT]["poll_period_s"]


class TestEnergyData:
    def test_get_sector(self):
        ed = EnergyData(
            electricity={"a": 1.0},
            gas={"b": 2.0},
            heat={"c": 3.0},
        )
        assert ed.get_sector(Sector.ELECTRICITY) == {"a": 1.0}
        assert ed.get_sector(Sector.GAS) == {"b": 2.0}
        assert ed.get_sector(Sector.HEAT) == {"c": 3.0}


class TestNegotiationMessage:
    def test_memory_default_empty(self):
        msg = EnergyNegotiationMessage(
            negotiation_id="x",
            sector=Sector.ELECTRICITY,
            negotiation_target=1.0,
            current_delta=0.0,
            counter=0,
        )
        assert msg.memory == {}

    def test_memory_custom(self):
        ledger = {"agent-1": (0.5, 2, 0), "agent-2": (-0.3, 1, 1)}
        msg = EnergyNegotiationMessage(
            negotiation_id="x",
            sector=Sector.ELECTRICITY,
            negotiation_target=1.0,
            current_delta=0.0,
            counter=0,
            memory=ledger,
        )
        assert msg.memory == ledger


class TestFrozenDataclasses:
    def test_negotiation_finished_immutable(self):
        event = NegotiationFinishedEvent(new_setpoint=1.0, sector=Sector.ELECTRICITY)
        with pytest.raises(AttributeError):
            event.new_setpoint = 2.0

    def test_constraint_violation_immutable(self):
        cv = ConstraintViolation(
            sector=Sector.ELECTRICITY,
            variable="vm_pu",
            value=1.06,
            bound_low=0.95,
            bound_high=1.05,
        )
        with pytest.raises(AttributeError):
            cv.value = 1.0


class TestLocalGenerationRequest:
    def test_construction(self):
        req = LocalGenerationRequest(sector=Sector.ELECTRICITY, residual_deficit=2.5)
        assert req.sector == Sector.ELECTRICITY
        assert req.residual_deficit == 2.5
