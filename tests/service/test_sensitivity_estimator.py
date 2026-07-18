"""Characterization tests for SensitivityEstimator, the |dV/dP| EMA extracted
from GridConstraintMonitor (cluster G)."""

from __future__ import annotations

from scare.base.model import Sector
from scare.service.control import constraints as C


def _patch_obs(monkeypatch):
    # Isolate the EMA: capacity fixed non-zero (so p = sp), sp read from obs.
    monkeypatch.setattr(C, "obs_capacity", lambda obs, behavior, aid: 1.0)
    monkeypatch.setattr(C, "obs_setpoint", lambda obs, behavior, aid: obs["sp"])


def test_first_sample_only_seeds_and_default_prior(monkeypatch):
    _patch_obs(monkeypatch)
    est = C.SensitivityEstimator(Sector.ELECTRICITY)
    default = est.value
    var = C._SECTOR_PRIMARY_VAR[Sector.ELECTRICITY]
    est.update({var: 1.0, "sp": 0.0}, behavior=None, aid="a")
    assert est.value == default  # nothing to differentiate against yet


def test_ema_moves_toward_observed_sample(monkeypatch):
    _patch_obs(monkeypatch)
    est = C.SensitivityEstimator(Sector.ELECTRICITY)
    default = est.value
    var = C._SECTOR_PRIMARY_VAR[Sector.ELECTRICITY]
    est.update({var: 1.0, "sp": 0.0}, behavior=None, aid="a")
    est.update({var: 1.1, "sp": 1.0}, behavior=None, aid="a")  # dp=1.0, dv=0.1
    assert est.value != default
    assert min(default, 0.1) <= est.value <= max(default, 0.1)


def test_subthreshold_dp_is_ignored(monkeypatch):
    _patch_obs(monkeypatch)
    est = C.SensitivityEstimator(Sector.ELECTRICITY)
    default = est.value
    var = C._SECTOR_PRIMARY_VAR[Sector.ELECTRICITY]
    est.update({var: 1.0, "sp": 0.5}, behavior=None, aid="a")
    est.update({var: 2.0, "sp": 0.5}, behavior=None, aid="a")  # dp=0 < min_dp
    assert est.value == default


def test_missing_primary_var_is_noop(monkeypatch):
    _patch_obs(monkeypatch)
    est = C.SensitivityEstimator(Sector.ELECTRICITY)
    default = est.value
    est.update({"unrelated": 1.0, "sp": 0.0}, behavior=None, aid="a")
    est.update({"unrelated": 2.0, "sp": 1.0}, behavior=None, aid="a")
    assert est.value == default
