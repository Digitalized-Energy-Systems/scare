"""Shared compliance gate + CI statistics for the eval / aggregation layer.

A run is *compliant* only when it passed BOTH the operator slack budget AND
end-of-sim grid feasibility (no voltage / pressure / temperature / line-loading
violation). A variant can inflate priority-weighted-served two ways the oracle
cannot — draw the slack past the operator envelope, or credit load served
through an overloaded line / at an infeasible voltage or temperature — so
served-fraction aggregates must restrict to the compliant subset to stay
comparable to the constraint-respecting oracle. This module is the single
source of that gate; both ``eval.plots`` and ``hpc.aggregate`` consume it.
"""

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

SLACK_COMPLIANCE_COL = "claims__slack_budget_compliance__passed"
CONSTRAINT_COMPLIANCE_COL = "claims__constraint_compliance__passed"
COMPLIANCE_COLS = (SLACK_COMPLIANCE_COL, CONSTRAINT_COMPLIANCE_COL)


def compliant_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series over ``df.index`` selecting rows that pass every
    available compliance claim. Missing per-row data counts as failure (drop
    rather than reward unverifiable rows). All-True when no compliance column
    is present at all.
    """
    present = [c for c in COMPLIANCE_COLS if c in df.columns]
    if not present:
        return pd.Series(True, index=df.index)
    mask = pd.Series(True, index=df.index)
    for col in present:
        mask &= df[col].fillna(False).astype(bool)
    return mask


def compliance_rate(df: pd.DataFrame) -> float | None:
    """Fraction of rows passing all compliance claims. ``None`` when no
    compliance column is present, so callers can suppress the annotation.
    """
    present = [c for c in COMPLIANCE_COLS if c in df.columns]
    if not present or df.empty:
        return None
    return float(compliant_mask(df).sum() / len(df))


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    """Sample mean and 95% CI half-width via a coarse t-table (no scipy
    dependency, robust to small n). Accepts any iterable / Series; ``None`` and
    NaN entries are dropped. Returns ``(mean, half_width)`` — display as
    ``mean ± half_width``; ``(nan, 0.0)`` for empty input.
    """
    arr = np.asarray(
        [
            v
            for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ],
        dtype=float,
    )
    n = arr.size
    if n == 0:
        return float("nan"), 0.0
    if n == 1:
        return float(arr[0]), 0.0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    t = 1.96 if n > 30 else 2.262 if n > 10 else 2.776
    return mean, t * se
