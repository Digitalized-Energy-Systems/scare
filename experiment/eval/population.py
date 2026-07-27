"""Comparable-population selection for cross-variant tables and figures.

Every variant-grouped aggregate in the campaign pooled whatever experiment mix
each variant happened to run: SCARE contributed 4195 rows of which 56% are
deliberately-ablated or degraded-comms arms, against baselines that are 1-8%
ablated and an oracle that is 0%. Comparing those populations is not a
comparison of the controllers.

This module defines the two restrictions that make a cross-variant number mean
what it says:

``default_arm_mask``
    drop the arms a variant was deliberately degraded in.

``all_variant_experiments``
    keep only experiments every variant actually ran.

Neither is a bug fix — the aggregation code does what it was written to do. They
are a stated selection rule, and because applying them moves SCARE's numbers up
on most grids, the matched frame must be published *beside* the pooled one with
the decomposition, never as a silent replacement.
"""

from __future__ import annotations

import pandas as pd

CELL_KEY = ("experiment", "grid", "scenario", "seed")
DEFAULT_ARM = "default"


def default_arm_mask(df: pd.DataFrame) -> pd.Series:
    """Rows on the unablated, unswept arm of their experiment."""
    mask = pd.Series(True, index=df.index)
    for col in ("ablation", "sweep"):
        if col in df.columns:
            mask &= df[col].fillna(DEFAULT_ARM).astype(str) == DEFAULT_ARM
    return mask


def all_variant_experiments(df: pd.DataFrame, n_variants: int = 4) -> list[str]:
    """Experiments in which every variant was planned.

    Built from ALL rows regardless of ``status``. Deriving it from completed
    rows would delete exactly the cells where the oracle's MILP crashed — the
    hardest reconfiguration cells — for all four variants at once, manufacturing
    a survivorship bias in the direction that flatters every controller.
    """
    if "experiment" not in df.columns or "variant" not in df.columns:
        return []
    counts = df.groupby("experiment")["variant"].nunique()
    return sorted(counts[counts >= n_variants].index)


def matched_frame(df: pd.DataFrame, n_variants: int = 4) -> tuple[pd.DataFrame, dict]:
    """Restrict to default arms of experiments every variant ran.

    Returns the restricted frame and a provenance dict for the caption: a
    matched number that does not carry its own selection rule is not auditable.
    """
    exps = all_variant_experiments(df, n_variants=n_variants)
    mask = default_arm_mask(df)
    if exps:
        mask &= df["experiment"].isin(exps)
    out = df[mask]
    key = [c for c in CELL_KEY if c in out.columns]
    prov = {
        "experiments": exps,
        "n_experiments": len(exps),
        "n_cells": int(out.groupby(key).ngroups) if key and not out.empty else 0,
        "n_rows": int(len(out)),
        "rows_by_variant": (
            out["variant"].value_counts().to_dict() if "variant" in out.columns else {}
        ),
        "dropped_rows": int(len(df) - len(out)),
    }
    return out, prov


def population_note(prov: dict) -> str:
    """One-line provenance caption for a matched table or figure."""
    n = prov["n_experiments"]
    noun = "experiment" if n == 1 else "experiments"
    return (
        f"Matched population: default arms of the {n} {noun} all variants ran "
        f"({prov['n_cells']} cells, {prov['n_rows']} rows; "
        f"{prov['dropped_rows']} rows outside it). "
        "Published beside the pooled figure, not in place of it."
    )
