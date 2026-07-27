import pandas as pd

from experiment.eval.population import (
    all_variant_experiments,
    default_arm_mask,
    matched_frame,
)

VARIANTS = ("scare", "oracle", "component_level", "single_level")


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "experiment",
            "grid",
            "scenario",
            "seed",
            "variant",
            "ablation",
            "sweep",
            "status",
        ],
    )


def _cell(
    exp, seed, variants=VARIANTS, ablation="default", sweep="default", status="ok"
):
    return [(exp, "g1", "s1", seed, v, ablation, sweep, status) for v in variants]


def test_default_arm_mask_excludes_ablated_and_swept():
    df = _frame(
        _cell("e1", 1)
        + _cell("e1", 2, ablation="no_holonic")
        + _cell("e1", 3, sweep="latency=50")
    )
    assert int(default_arm_mask(df).sum()) == 4


def test_default_arm_mask_treats_missing_as_default():
    df = _frame(_cell("e1", 1)).drop(columns=["ablation", "sweep"])
    assert default_arm_mask(df).all()


def test_all_variant_experiments_counts_error_rows():
    """The oracle's MILP crashes on the hardest cells. Deriving the matched set
    from completed rows would drop those cells for every variant at once."""
    df = _frame(
        _cell("e_full", 1)
        + _cell("e_crash", 1, variants=("scare", "component_level", "single_level"))
        + [("e_crash", "g1", "s1", 1, "oracle", "default", "default", "error")]
        + _cell("e_partial", 1, variants=("scare", "oracle"))
    )
    assert all_variant_experiments(df) == ["e_crash", "e_full"]


def test_matched_frame_cells_and_provenance():
    df = _frame(
        _cell("e1", 1)
        + _cell("e1", 2)
        + _cell("e1", 3, ablation="no_holonic")
        + _cell("e2", 1, variants=("scare", "oracle"))
    )
    out, prov = matched_frame(df)
    assert prov["n_experiments"] == 1
    assert prov["n_cells"] == 2
    assert prov["n_rows"] == 8
    assert prov["rows_by_variant"] == dict.fromkeys(VARIANTS, 2)
    assert set(out["experiment"]) == {"e1"}


def test_matched_frame_has_no_duplicate_cell_variant_pairs():
    df = _frame(_cell("e1", 1) + _cell("e1", 2))
    out, _ = matched_frame(df)
    assert not out.duplicated(
        subset=["experiment", "grid", "scenario", "seed", "variant"]
    ).any()
