"""Shared compliance gate + CI statistics for the eval / aggregation layer.

A run is *compliant* only when it passed BOTH the operator slack budget AND
end-of-sim grid feasibility (no voltage / pressure / temperature / line-loading
violation). A variant can inflate priority-weighted-served two ways the oracle
cannot — draw the slack past the operator envelope, or credit load served
through an overloaded line / at an infeasible voltage or temperature — so
served-fraction aggregates must restrict to the compliant subset to stay
comparable to the constraint-respecting oracle. This module is the single
source of that gate; both ``eval.plots`` and ``hpc.aggregate`` consume it.

Two ways a *passing* verdict can still be empty: it may have been read off a
state the simulator never produced (:func:`unsolved_mask`), or there may have
been nothing to check (:func:`slack_gate_vacuous_mask`). Only the first is a
reason to drop the row; the second is reported so a structurally-unbudgeted
pillar does not read as verified.
"""

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

SLACK_COMPLIANCE_COL = "claims__slack_budget_compliance__passed"
CONSTRAINT_COMPLIANCE_COL = "claims__constraint_compliance__passed"
COMPLIANCE_COLS = (SLACK_COMPLIANCE_COL, CONSTRAINT_COMPLIANCE_COL)

_SLACK_UTIL_PREFIX = "claims__slack_budget_compliance__detail__per_slack__"

_VAR_CHECKED_TMPL = "claims__constraint_compliance__detail__by_variable__{}__n_checked"
_TEMP_CHECKED_COL = _VAR_CHECKED_TMPL.format("temperature")
_OTHER_CHECKED_COLS = tuple(
    _VAR_CHECKED_TMPL.format(v) for v in ("voltage", "pressure", "line_load")
)
_SOLVES_FAILED_COL = "outcomes__physics_solves__failed"
_SOLVES_OK_COL = "outcomes__physics_solves__ok"
_COMPLETED_STATUS = ("ok", "claims_failed")

STALE_PHYSICS_FRACTION = 0.25

# The published campaign graded slack at a one-sided +5% tolerance.
SLACK_TOL_PUBLISHED = 0.05
# Utilisation is stored already rounded; without a slop term a value recorded as
# exactly 1.0 fails a tol=0 gate on the last bit. Moves the tol=0 rate 45.51% ->
# 45.60% (SCARE), so it must be stated wherever a tol=0 number is published.
SLACK_UTIL_EPS = 1e-12


def _completed(df: pd.DataFrame) -> pd.Series:
    if "status" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["status"].isin(_COMPLETED_STATUS)


def frozen_net_mask(df: pd.DataFrame) -> pd.Series:
    """Rows graded off a network the physics never solved.

    ``physics_final_solve_ok`` was never persisted, so a null temperature
    ``n_checked`` alongside a populated voltage / pressure / line-load scan is
    the available detector: ``vm_pu`` and ``pressure_pu`` carry constructor
    defaults and are emitted by the scan whether or not a solve ran, but ``t_k``
    is solver-populated only, so a pristine net yields exactly that signature.
    Reproduced directly — ``constraint_rows`` on an unsolved ``simbench_lv_small``
    emits 15 ``vm_pu`` + 15 ``pressure_pu`` + 14 ``loading_percent`` rows and
    **zero** ``t_k``; after a successful solve the 15 ``t_k`` rows appear.

    All 33 such rows in eval_full_v2 carry the matching fingerprint on disk:
    ``vm_pu`` takes only {1.0, 1.025}, ``pressure_pu`` is uniformly 1.0 and
    ``loading_percent`` is uniformly 100, against 127 distinct voltages spanning
    0.965-1.025 on a solved row of the same grid. A nonzero
    ``physics_solves__ok`` does not contradict this: the env hands back the
    unsolved constructor net when the *final* solve fails, and observers then
    read regulation 1.0 defaults — which is also why such a row can post a
    plausible-looking served fraction. Its compliance verdict says nothing about
    the controller and must not enter a compliant subset.

    The other-variable requirement additionally keeps a grid with no heat sector
    from being dropped on a signal that there means the opposite.
    """
    if _TEMP_CHECKED_COL not in df.columns:
        return pd.Series(False, index=df.index)
    others = [c for c in _OTHER_CHECKED_COLS if c in df.columns]
    if not others:
        return pd.Series(False, index=df.index)
    scanned_something = df[others].notna().any(axis=1)
    return df[_TEMP_CHECKED_COL].isna() & scanned_something & _completed(df)


def stale_physics_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where more than ``STALE_PHYSICS_FRACTION`` of physics solves failed.

    The end state is then largely extrapolated from the last feasible solve, so
    the graded readings are not the state the controller actually produced.
    """
    if _SOLVES_FAILED_COL not in df.columns or _SOLVES_OK_COL not in df.columns:
        return pd.Series(False, index=df.index)
    failed = df[_SOLVES_FAILED_COL].astype(float)
    ok = df[_SOLVES_OK_COL].astype(float)
    total = failed + ok
    frac = failed.where(total > 0).div(total.where(total > 0))
    return frac.gt(STALE_PHYSICS_FRACTION).fillna(False) & _completed(df)


def unsolved_mask(df: pd.DataFrame) -> pd.Series:
    """Union of the two "graded off a state the simulator did not produce" gates.

    On eval_full_v2 they overlap on 25 of 33 rows, so the frozen-net detector
    adds 8 beyond :func:`stale_physics_mask` alone. Neither catches a run whose
    *first* steps fail and later recover — the failure fraction stays under
    ``STALE_PHYSICS_FRACTION`` and the final solve succeeds — so this is not a
    complete "was the controller acting on real physics?" test.
    """
    return frozen_net_mask(df) | stale_physics_mask(df)


def compliant_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series over ``df.index`` selecting rows that pass every
    available compliance claim AND were graded off a state the physics
    actually solved (see :func:`unsolved_mask`). Missing per-row data counts as
    failure (drop rather than reward unverifiable rows). All-True when no
    compliance column is present at all.
    """
    present = [c for c in COMPLIANCE_COLS if c in df.columns]
    if not present:
        return pd.Series(True, index=df.index)
    mask = pd.Series(True, index=df.index)
    for col in present:
        mask &= df[col].fillna(False).astype(bool)
    return mask & ~unsolved_mask(df)


def slack_gate_vacuous_mask(df: pd.DataFrame) -> pd.Series:
    """Completed rows whose slack claim passed with no budget to check.

    A scenario declaring no ``slack_budget_pct`` never has ``apply_slack_budget``
    run against it (``runner.py``), so ``_load_slack_budgets`` returns nothing,
    the grader falls back to its legacy event path and the claim passes
    vacuously. That is correct — there is no operator policy to violate — but it
    is indistinguishable from a measured pass in a compliance rate, and it is not
    evenly spread: on eval_full_v2 all 363 such rows are the ``simbench_lv_small``
    voltage pillar. Report it beside any compliance number drawn from those arms
    rather than letting a structurally-unbudgeted pillar read as verified.

    "Measured" must be tested against ANY per-slack field, not ``utilization``:
    the oracle emits a different schema (``budget_mw`` / ``draw_kgs`` / ...) with
    no ``utilization`` at all, so keying off that column alone marks its whole
    arm vacuous — precisely inverting the truth, since ``enforced_at_lp`` means
    the budget is an LP bound the solve cannot violate. Both guards are applied.
    """
    per_slack = [c for c in df.columns if c.startswith(_SLACK_UTIL_PREFIX)]
    if not per_slack:
        return pd.Series(False, index=df.index)
    measured = df[per_slack].notna().any(axis=1)
    lp_col = "claims__slack_budget_compliance__detail__enforced_at_lp"
    if lp_col in df.columns:
        measured |= df[lp_col].fillna(False).astype(bool)
    passed = (
        df[SLACK_COMPLIANCE_COL].fillna(False).astype(bool)
        if SLACK_COMPLIANCE_COL in df.columns
        else pd.Series(False, index=df.index)
    )
    return ~measured & passed & _completed(df)


def compliance_rate(df: pd.DataFrame) -> float | None:
    """Fraction of rows passing all compliance claims. ``None`` when no
    compliance column is present, so callers can suppress the annotation.
    """
    present = [c for c in COMPLIANCE_COLS if c in df.columns]
    if not present or df.empty:
        return None
    return float(compliant_mask(df).sum() / len(df))


# Two-sided 95% Student-t criticals t_{0.975,df} for df=1..30; exact via scipy
# above. Replaces a 3-bucket table that was anti-conservative: used 2.776 at
# df=2 (true 4.303) and z=1.96 for all n>30 (true 2.042 at df=30).
_T_975: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_crit_95(df: int) -> float:
    """Two-sided 95% t critical value for ``df`` degrees of freedom.

    Exact via scipy above the tabulated range. The previous four coarse bands
    each returned the value at their *ceiling*, so every published CI with
    df > 30 was up to 1.0% too narrow, one-directionally.
    """
    if df <= 0:
        return float("inf")
    if df <= 30:
        return _T_975[df]
    return float(stats.t.ppf(0.975, df))


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    """Sample mean and 95% CI half-width via a Student-t critical (robust to
    small n). Accepts any iterable / Series; ``None`` and
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


def cluster_ci95(values: Iterable[float], clusters: Iterable) -> tuple[float, float]:
    """Mean and cluster-robust (CR1) 95% CI half-width over ``(grid, scenario,
    seed)`` replicates of one failure draw. Degrees of freedom are G-1 in the
    number of clusters, not n-1; reduces exactly to :func:`mean_ci95` when every
    cluster has size 1.

    Deliberately NOT wired into the variant tables. Within a single variant the
    only rows sharing a draw are that variant's own ablation / sweep arms, so
    clustering and the pooled-population defect are the same thing measured
    twice: pooled, SCARE's interval widens 2.20x (+-0.0048 -> +-0.0105) while the
    oracle's is untouched at 1.00x — but restricted to default arms EVERY
    variant has mean cluster size exactly 1.00 and this function is a no-op.
    Widening the interval would leave the biased point estimate in place and
    dress it as caution; :func:`eval.population.matched_frame` is the fix.
    Kept for a figure that pools arms by design and must say so.
    """
    pairs = [
        (float(v), c)
        for v, c in zip(values, clusters, strict=False)
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not pairs:
        return float("nan"), float("nan")
    arr = np.asarray([p[0] for p in pairs], dtype=float)
    keys = [p[1] for p in pairs]
    n = arr.size
    mean = float(arr.mean())
    groups: dict[object, list[float]] = {}
    for v, k in zip(arr, keys, strict=True):
        groups.setdefault(k, []).append(float(v))
    g = len(groups)
    if n < 2 or g < 2:
        return mean, float("nan")
    # CR1: sum of squared within-cluster residual sums, with the finite-cluster
    # correction G/(G-1). Identical to the iid SE when every cluster is size 1.
    meat = sum(math.fsum(v - mean for v in vs) ** 2 for vs in groups.values())
    var = (g / (g - 1)) * meat / (n**2)
    if not (var > 0):
        return mean, 0.0
    return mean, _t_crit_95(g - 1) * math.sqrt(var)


def slack_utilisation(df: pd.DataFrame) -> pd.DataFrame:
    """Per-slack utilisation (draw / budget) columns, as floats."""
    cols = [
        c
        for c in df.columns
        if c.startswith(_SLACK_UTIL_PREFIX) and c.endswith("__utilization")
    ]
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].astype(float)


def slack_pass_mask(
    df: pd.DataFrame,
    tol: float = SLACK_TOL_PUBLISHED,
    eps: float = SLACK_UTIL_EPS,
) -> pd.Series:
    """Rows drawing no slack past ``1 + tol``.

    Vacuous-pass semantics: a row with no recorded utilisation cannot exceed the
    bound and passes, reproducing the shipped grader. Expressed as
    ``~(U > thr).any()`` rather than ``U.max() <= thr`` because the latter
    reverses that convention for all-NaN rows.
    """
    util = slack_utilisation(df)
    if util.empty:
        return pd.Series(True, index=df.index)
    return ~(util > (1.0 + tol + eps)).any(axis=1)


def compliant_mask_at_tol(
    df: pd.DataFrame,
    tol: float = SLACK_TOL_PUBLISHED,
    eps: float = SLACK_UTIL_EPS,
) -> pd.Series:
    """:func:`compliant_mask` with the slack claim re-graded at ``tol``.

    Re-grading at ``tol=0`` is a *selection* change, not a pure re-score: it
    preferentially drops higher-scoring runs, so the resulting means are not
    comparable to the published ones without stating that.
    """
    mask = pd.Series(True, index=df.index)
    if CONSTRAINT_COMPLIANCE_COL in df.columns:
        mask &= df[CONSTRAINT_COMPLIANCE_COL].fillna(False).astype(bool)
    return mask & slack_pass_mask(df, tol=tol, eps=eps) & ~unsolved_mask(df)


def freshness(df: pd.DataFrame) -> pd.Series:
    """Seconds between the last feasible physics solve and the graded end state.

    Variant-asymmetric by ~23x on a 30 s horizon, so it belongs beside every
    compliance number rather than in a footnote.
    """
    final_col = "sim_time_final"
    last_col = "legacy_metrics__last_feasible_solve_t__last"
    if final_col not in df.columns or last_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[final_col].astype(float) - df[last_col].astype(float)
