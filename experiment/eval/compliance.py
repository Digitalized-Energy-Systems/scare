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


# Two-sided 95% Student-t critical values t_{0.975, df} for df = 1..30 (no scipy
# dependency). Above df=30 the value descends slowly toward the normal 1.96, so a
# few coarse bands suffice. The previous 3-bucket table (2.776/2.262/1.96) was
# anti-conservative at very small n (used 2.776 for df=2 where t=4.303) and used
# the normal z=1.96 for all n>30 (t=2.042 at df=30), understating every CI there.
_T_975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def _t_crit_95(df: int) -> float:
    """Two-sided 95% t critical value for ``df`` degrees of freedom."""
    if df <= 0:
        return float("inf")
    if df <= 30:
        return _T_975[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.96


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    """Sample mean and 95% CI half-width via a Student-t table (no scipy
    dependency, robust to small n). Accepts any iterable / Series; ``None`` and
    NaN entries are dropped. Returns ``(mean, half_width)`` — display as
    ``mean ± half_width``. For ``n < 2`` the half-width is NaN (a single sample
    carries no spread information; 0.0 would render as "±0.0000", i.e. false
    certainty) — formatters should show it as missing.
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
        return float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), float("nan")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    return mean, _t_crit_95(n - 1) * se
