"""An ablation arm that only restates the defaults must fail planning.

Such an arm is a bit-identical duplicate of its baseline, so the published
``delta`` is exactly 0 by construction and reads as "this component does not
matter". eval_full_v2 shipped four of them; the fingerprint sweep that found
them also showed the mechanisms were demonstrably live in the baseline (e.g.
``holon_supply_priority`` issuing 2584 regulates/task), so the null was an
authoring artefact, not a result.

The failure mode is not only authoring: ``ab_heat_priority_v2`` was written when
``enable_heat_l2_dispatch`` defaulted to False, and the arm silently became a
duplicate when that default later flipped to True. Nothing detected it.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

from experiment.hpc.config import CampaignConfig
from experiment.hpc.plan import _validate_config_overrides, build_tasks


def _cfg(**exp):
    base = {"name": "e", "grids": ["simbench_lv_small"], "n_seeds": 1}
    return CampaignConfig.from_dict(
        {"name": "c", "experiments": [{**base, **exp}]},
    )


def test_ablation_arm_restating_every_default_is_rejected():
    # enable_monotonic_floor defaults to False, so an arm setting it False
    # changes nothing.
    cfg = _cfg(ablations=[{}, {"enable_monotonic_floor": False}])
    with pytest.raises(ValueError, match="restates the default"):
        _validate_config_overrides(cfg)


def test_flipping_to_the_non_default_side_is_accepted():
    cfg = _cfg(ablations=[{}, {"enable_monotonic_floor": True}])
    _validate_config_overrides(cfg)


def test_arm_with_one_real_flip_among_restated_keys_is_accepted():
    """Only a *fully* redundant arm is rejected — a compound arm that also
    moves something real still produces a distinct configuration."""
    cfg = _cfg(
        ablations=[
            {},
            {"enable_monotonic_floor": False, "enable_holon_summary": False},
        ]
    )
    _validate_config_overrides(cfg)


def test_sweeps_may_include_the_default_as_an_anchor_point():
    """Sweeps are exempt: the default is the natural anchor of a sweep and the
    neighbouring points still vary."""
    cfg = _cfg(
        sweeps=[
            {"holon_admm_max_iters": 25},
            {"holon_admm_max_iters": 50},  # the default
            {"holon_admm_max_iters": 100},
        ]
    )
    _validate_config_overrides(cfg)


def test_explicit_annotation_allows_a_deliberate_duplicate_control():
    cfg = _cfg(
        ablations=[
            {},
            {
                "enable_monotonic_floor": False,
                "$allow_default_valued": "negative control, must be bit-identical",
            },
        ]
    )
    _validate_config_overrides(cfg)


@pytest.mark.parametrize(
    "path",
    sorted(
        p
        for p in glob.glob("experiment/configs/*.json")
        if os.path.basename(p) != "display_aliases.json"
    ),
)
def test_shipped_campaign_configs_have_no_no_op_ablation_arms(path):
    """Whole-repo sweep: every shipped campaign must plan cleanly.

    ``ab_cp_slack_debit`` is a known pre-existing break on a *different* check
    (it references ``enable_cp_slack_reserve_debit``, which is not a
    RestorationConfiguration field), so it is xfailed rather than silently
    excluded.
    """
    if os.path.basename(path) == "ab_cp_slack_debit.json":
        pytest.xfail("references a removed config field; predates this guard")
    build_tasks(CampaignConfig.from_json(Path(path)))
