"""Lock for the shared runtime-default constants in config.py: one source flows
to both CampaignConfig and RuntimePlan, and the fatal_claims container types
(list vs tuple) are preserved across asdict()/from_config_json."""

from __future__ import annotations

import dataclasses as dc
import json

from experiment.hpc.config import (
    FATAL_CLAIMS_DEFAULT,
    CampaignConfig,
    RuntimePlan,
)


def test_fatal_claims_container_types_preserved():
    assert isinstance(CampaignConfig(name="x").fatal_claims, list)
    assert isinstance(RuntimePlan().fatal_claims, tuple)
    d = dc.asdict(CampaignConfig(name="x"))
    assert d["fatal_claims"] == list(FATAL_CLAIMS_DEFAULT)
    assert isinstance(d["fatal_claims"], list)


def test_shared_scalar_defaults_flow_to_both():
    cc, rp = CampaignConfig(name="x"), RuntimePlan()
    assert cc.simulation_duration_s == rp.simulation_duration_s
    assert cc.task_timeout_s == rp.task_timeout_s
    assert cc.failure_delay_s_max == rp.failure_delay_s_max
    assert cc.write_timeseries == rp.write_timeseries
    assert cc.write_trajectories == rp.write_trajectories
    assert list(cc.fatal_claims) == list(rp.fatal_claims)


def test_from_config_json_falls_back_to_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"name": "x"}))  # no runtime keys
    assert RuntimePlan.from_config_json(p) == RuntimePlan()
