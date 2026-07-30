"""Aggregate per-task outputs into a single CSV (+ a Markdown summary).

CLI:
    python -m experiment.hpc.aggregate --campaign-dir runs/restoration_2026-05-05

Always writes:
  - summary.csv : one row per task (manifest + status + result + exception)
  - summary.md  : compact human-readable per-grid roll-up

Crashed tasks still appear in both with empty metric columns and
``status != ok`` so failures stay visible.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.eval.aliases import (
    alias_ablation,
    alias_experiment,
    alias_grid,
    alias_scenario,
    alias_sweep,
    alias_variant,
)
from experiment.eval.compliance import (
    COMPLIANCE_COLS,
    compliance_rate,
    compliant_mask,
    mean_ci95,
)
from experiment.hpc.config import CAMPAIGN_LAYOUT
from experiment.hpc.plan import read_manifest

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None


def aggregate(campaign_dir: Path) -> pd.DataFrame:
    tasks_root = campaign_dir / CAMPAIGN_LAYOUT["tasks"]
    rows: list[dict[str, Any]] = []
    for task in read_manifest(campaign_dir):
        td = tasks_root / f"{task.task_id:06d}"
        status = _load_json(td / "status.json") or {"status": "missing"}
        result = _load_json(td / "result.json") or {}
        exception = _load_json(td / "exception.json")
        # Flatten the structured eval payload into top-level columns so
        # the Markdown / pandas comparisons can group on them.
        flat = _flatten(result)
        rows.append(
            {
                "task_id": task.task_id,
                "grid": task.grid,
                "seed": task.seed,
                "n_failures": task.n_failures,
                "variant": getattr(task, "variant", "scare"),
                "experiment": getattr(task, "experiment", ""),
                "ablation": _key_of(getattr(task, "ablation", {}) or {}),
                "sweep": _key_of(getattr(task, "sweep", {}) or {}),
                "scenario": _key_of(getattr(task, "scenario", {}) or {}),
                "status": status.get("status", "missing"),
                "duration_s": status.get("duration_s"),
                "solver_failures": status.get("solver_failures"),
                # Split solver-failure counts (LP infeasibility vs non-OK
                # warnings) so per-variant plots can show the split.
                "solver_infeasibilities": status.get("solver_infeasibilities"),
                "solver_warnings": status.get("solver_warnings"),
                # first_failed_step == 0 separates a born-infeasible scenario
                # from a control-induced one; without it both land in
                # solver_infeasibilities and read as the same defect.
                "first_failed_step": status.get("first_failed_step"),
                "n_failed_steps": status.get("n_failed_steps"),
                "physics_solves_ok": status.get("physics_solves_ok"),
                "physics_solves_failed": status.get("physics_solves_failed"),
                "exception_type": (exception or {}).get("type"),
                "exception_message": (exception or {}).get("message"),
                **flat,
            }
        )
    return pd.DataFrame(rows)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into ``a__b__c`` keys. Lists stay in a single
    cell; the main consumers (per-tier served, per-sector violation
    integrals) are dicts, not lists."""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}__{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _key_of(d) -> str:
    """Stable string grouping key for an ablation / sweep / scenario.

    Sorted keys make it reproducible regardless of dict insertion order.
    A dict is joined, a string passed through, anything else repr'd;
    empty / null maps to ``"default"`` so the column has no NaNs.
    """
    if d is None or d == {} or d == "":
        return "default"
    if isinstance(d, dict):
        return ";".join(f"{k}={d[k]}" for k in sorted(d))
    if isinstance(d, str):
        return d
    return repr(d)


_INTERESTING_METRIC_SUFFIXES = ("__last", "__mean", "__min")


def _interesting_metric_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if any(col.endswith(s) for s in _INTERESTING_METRIC_SUFFIXES):
            out.append(col)
    return sorted(out)[:8]  # cap to keep the table readable


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _format_md(
    df: pd.DataFrame, targets: dict[str, tuple[list[str], str]] | None = None
) -> str:
    parts: list[str] = ["# Campaign summary", ""]

    n = len(df)
    parts.append(f"- Tasks: **{n}**")
    if n:
        for s in ("ok", "claims_failed", "error", "timeout", "killed", "missing"):
            c = int((df["status"] == s).sum())
            parts.append(f"  - {s}: {c} ({100.0 * c / n:.1f}%)")
        if "duration_s" in df.columns and df["duration_s"].notna().any():
            dur = df["duration_s"].dropna()
            if not dur.empty:
                parts.append(
                    f"- Mean duration: {dur.mean():.1f}s "
                    f"(p95 {dur.quantile(0.95):.1f}s)"
                )
        if "solver_failures" in df.columns:
            parts.append(
                f"- Total solver-failure warnings: "
                f"{int(df['solver_failures'].fillna(0).sum())}"
            )

    parts.append("")
    parts.append("## Per grid")
    if not n:
        parts.append("_(empty)_")
    else:
        agg = df.groupby("grid").agg(
            total=("task_id", "count"),
            ok=("status", lambda s: int((s == "ok").sum())),
            claims_failed=("status", lambda s: int((s == "claims_failed").sum())),
            error=("status", lambda s: int((s == "error").sum())),
            timeout=("status", lambda s: int((s == "timeout").sum())),
            killed=("status", lambda s: int((s == "killed").sum())),
            missing=("status", lambda s: int((s == "missing").sum())),
            mean_duration_s=("duration_s", "mean"),
            mean_solver_failures=("solver_failures", "mean"),
        )
        rows = []
        for grid, r in agg.iterrows():
            rows.append(
                [
                    alias_grid(grid),
                    int(r["total"]),
                    int(r["ok"]),
                    int(r["claims_failed"]),
                    int(r["error"]),
                    int(r["timeout"]),
                    int(r["killed"]),
                    int(r["missing"]),
                    f"{r['mean_duration_s']:.1f}"
                    if pd.notna(r["mean_duration_s"])
                    else "—",
                    f"{r['mean_solver_failures']:.2f}"
                    if pd.notna(r["mean_solver_failures"])
                    else "—",
                ]
            )
        parts.append(
            _markdown_table(
                [
                    "grid",
                    "total",
                    "ok",
                    "claims_failed",
                    "error",
                    "timeout",
                    "killed",
                    "missing",
                    "mean dur (s)",
                    "mean solver fails",
                ],
                rows,
            )
        )

    metric_cols = _interesting_metric_columns(df) if n else []
    if metric_cols:
        parts.append("")
        parts.append(
            "## Per-grid metrics (mean over completed runs: ok + claims_failed)"
        )
        # Include ``claims_failed``: the sim completed and the metrics are
        # real — a claim failure flags an invariant violation, not a
        # missing measurement.
        ok_df = df[df["status"].isin(("ok", "claims_failed"))]
        if ok_df.empty:
            parts.append("_(no successful runs)_")
        else:
            agg = ok_df.groupby("grid")[metric_cols].mean()
            rows = []
            for grid, r in agg.iterrows():
                rows.append(
                    [
                        alias_grid(grid),
                        *[
                            f"{r[c]:.4g}" if pd.notna(r[c]) else "—"
                            for c in metric_cols
                        ],
                    ]
                )
            parts.append(_markdown_table(["grid", *metric_cols], rows))

    if "exception_type" in df.columns and df["exception_type"].notna().any():
        # Count exceptions only for non-``ok`` tasks; a stale
        # ``exception.json`` from a since-rerun task would inflate it.
        parts.append("")
        parts.append("## Exception breakdown")
        failed = df[df["status"] != "ok"]
        counts = failed["exception_type"].dropna().value_counts()
        if counts.empty:
            parts.append("_(no failures)_")
        else:
            rows = [[t, int(c)] for t, c in counts.items()]
            parts.append(_markdown_table(["type", "count"], rows))

    parts.extend(_format_eval_sections(df, targets))
    return "\n".join(parts) + "\n"


_PRIMARY_OUTCOME = "outcomes__priority_weighted_fraction"
_TIME_TO_STABILISE = "outcomes__time_to_stabilise_s"
_REGULATES_TOTAL = "outcomes__regulates_total"

# Gurobi reported an optimal solve (status 2) rather than hitting the time limit
# (status 9). A time-limited oracle is not a reference point: on eval_full_v2
# 469 of 1435 oracle runs (32.7%) were time-limited — 82.8% on lv_reconfig and
# 56.7% in ``optimality_gap`` itself — and every variant's beats-oracle rate is
# ~22x higher there than on certified pairs.
_ORACLE_OPTIMAL_COL = "outcomes__oracle_solve_optimal"


def _subset_cols(v_num, o_num, mask) -> list[str]:
    """``[n, Δ, wins/n]`` over the ``mask`` subset of a paired comparison."""
    n = int(mask.sum())
    if n == 0:
        return ["0", "—", "—"]
    v, o = v_num[mask], o_num[mask]
    return [str(n), f"{v.mean() - o.mean():+.4f}", f"{int((v > o + 1e-9).sum())}/{n}"]


def _as_bool(value: object) -> bool:
    """Truthiness for a CSV round-tripped boolean.

    ``read_csv`` yields real bools for a clean column but ``"True"``/``"False"``
    strings once any row is missing, and ``bool("False")`` is True — so the
    string form must be handled explicitly or a time-limited oracle would be
    counted as certified.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "1.0", "yes")
    if value is None:
        return False
    try:
        return bool(value) and value == value  # NaN-safe
    except (TypeError, ValueError):
        return False


# Short labels for the justification table; fall back to the last two
# ``__``-segments for anything not listed.
_METRIC_ALIAS = {
    "outcomes__priority_weighted_fraction": "PWSF",
    "outcomes__priority_weighted_fraction_by_sector__electricity": "PWSF(el)",
    "outcomes__priority_weighted_fraction_by_sector__heat": "PWSF(heat)",
    "outcomes__priority_weighted_fraction_by_sector__gas": "PWSF(gas)",
    "outcomes__constraint_violation_integral__electricity": "viol∫(el)",
    "outcomes__constraint_violation_integral__gas": "viol∫(gas)",
    "outcomes__constraint_violation_integral__heat": "viol∫(heat)",
    "claims__constraint_compliance__passed": "feas-ok",
    "claims__slack_budget_compliance__passed": "slack-ok",
    "claims__priority_invariant__passed": "prio-inv",
    "claims__constraint_compliance__detail__by_variable__voltage__n_violations": "volt n_viol",
    "claims__constraint_compliance__detail__by_variable__pressure__n_violations": "press n_viol",
    "claims__constraint_compliance__detail__by_variable__line_load__n_violations": "line n_viol",
    "diary__abandoned": "abandoned",
    "outcomes__regulates_total": "regulates",
}


def _short_metric(col: str) -> str:
    if col in _METRIC_ALIAS:
        return _METRIC_ALIAS[col]
    seg = col.split("__")
    return "·".join(seg[-2:]) if len(seg) >= 2 else col


def _load_experiment_targets(campaign_dir: Path) -> dict[str, tuple[list[str], str]]:
    """Read each experiment's ``target_metrics`` / ``population`` from the
    resolved ``config.json``. Missing file / fields => empty map, and the
    summary falls back to the legacy PWSF tables (eval_full stays unchanged).
    """
    data = _load_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
    out: dict[str, tuple[list[str], str]] = {}
    if not data:
        return out
    for e in data.get("experiments", []) or []:
        if not isinstance(e, dict):
            continue
        metrics = e.get("target_metrics") or []
        if not metrics:
            continue
        out[e.get("name", "")] = (list(metrics), e.get("population") or "compliant")
    return out


def _metric_stats(
    g: pd.DataFrame, col: str, population: str
) -> tuple[float, float, int] | None:
    """``(mean, ci95_halfwidth, n)`` for ``col`` over the chosen row
    population, or ``None`` if the column is absent / empty. Booleans (claim
    ``__passed`` flags) coerce to 0/1 so the mean is a pass-rate.

    A ``__passed`` compliance flag is ALWAYS measured over all runs: a pass-rate
    restricted to the compliant subset is a degenerate 1.0, so the
    feasibility / slack-budget columns the campaign reports for every lever stay
    meaningful even when the experiment's served metric is on the compliant
    population.
    """
    if col not in g.columns:
        return None
    pop = "all" if col.endswith("__passed") else population
    sub = g if pop == "all" else g[compliant_mask(g)]
    s = pd.to_numeric(sub[col], errors="coerce").dropna()
    if s.empty:
        return None
    mean, ci = mean_ci95(s)
    return mean, ci, int(len(s))


_PAIR_KEYS = ("grid", "scenario", "seed")


def _paired_delta(
    base: pd.DataFrame, row: pd.DataFrame, col: str, population: str
) -> tuple[float, float, int] | None:
    """Per-seed paired mean-difference of ``col`` (row − baseline) with 95% CI.

    plan.py derives seed from (experiment, grid, run) only, so ablation rows share
    the failure draw; pairing on (grid, scenario, seed) removes the between-seed
    difficulty variance that dominates the marginal CIs. ``None`` if <2 shared
    non-NaN pairs. ``__passed`` flags pair over all runs (compliant-only is
    degenerate); other metrics honour ``population``, so paired n can be below the
    per-group n and is reported separately.
    """
    pop = "all" if col.endswith("__passed") else population
    b = base if pop == "all" else base[compliant_mask(base)]
    r = row if pop == "all" else row[compliant_mask(row)]
    keys = [k for k in _PAIR_KEYS if k in b.columns and k in r.columns]
    if not keys:
        return None
    bs = b[keys + [col]].copy()
    rs = r[keys + [col]].copy()
    bs[col] = pd.to_numeric(bs[col], errors="coerce").astype(float)
    rs[col] = pd.to_numeric(rs[col], errors="coerce").astype(float)

    # The merge fans out if a side has several rows per pair key (an
    # experiment varying both ablation AND sweep would); collapse to the
    # per-key mean so pairing stays 1:1.
    def _dedupe(side: pd.DataFrame, label: str) -> pd.DataFrame:
        n_dup = int(side.duplicated(keys).sum())
        if not n_dup:
            return side
        logger.warning(
            "_paired_delta: %d duplicate %s rows on %s for %r — averaging "
            "within key to keep the pairing 1:1",
            n_dup,
            label,
            keys,
            col,
        )
        return side.groupby(keys, as_index=False)[col].mean()

    bs = _dedupe(bs, "baseline")
    rs = _dedupe(rs, "candidate")
    j = bs.merge(rs, on=keys, suffixes=("_b", "_r")).dropna(
        subset=[f"{col}_b", f"{col}_r"]
    )
    if len(j) < 2:
        return None
    d = j[f"{col}_r"] - j[f"{col}_b"]
    mean_d, ci_d = mean_ci95(d)
    return mean_d, ci_d, int(len(j))


def _format_justification_sections(
    ok: pd.DataFrame, targets: dict[str, tuple[list[str], str]]
) -> list[str]:
    """Per-experiment justification tables: for each experiment that declares
    ``target_metrics``, paired-delta every target metric (over its declared
    population) against the in-experiment ``default`` baseline, and screen
    whether ANY of them moved beyond noise. Owns the target-mapped experiments so
    the legacy PWSF tables skip them."""
    body: list[str] = []
    min_paired_n = None  # smallest paired n seen, surfaced in the caption
    for exp_name in sorted(targets):
        metrics, population = targets[exp_name]
        g_exp = ok[ok["experiment"] == exp_name]
        if g_exp.empty:
            continue
        if "ablation" in g_exp.columns and g_exp["ablation"].nunique() > 1:
            axis, baseline_key = "ablation", "default"
        elif "sweep" in g_exp.columns and g_exp["sweep"].nunique() > 1:
            axis, baseline_key = "sweep", "default"
        elif "variant" in g_exp.columns and g_exp["variant"].nunique() > 1:
            # Architectural comparison: scare vs single_level vs component_level
            # on a shared scenario; variants share seeds, so they pair too.
            axis, baseline_key = "variant", "scare"
        else:
            continue
        # Ablation/sweep buckets and the paired `default` baseline must not pool
        # other-variant rows (e.g. oracle): pin to the SCARE variant, as the
        # legacy ablation path does (variant IS the axis on the variant branch).
        if axis in ("ablation", "sweep") and "variant" in g_exp.columns:
            g_exp = g_exp[g_exp["variant"] == "scare"]
            if g_exp.empty:
                continue
        has_default = bool((g_exp[axis] == baseline_key).any())
        base = g_exp[g_exp[axis] == baseline_key] if has_default else None
        base_stats = (
            {m: _metric_stats(base, m, population) for m in metrics}
            if base is not None
            else {}
        )

        header = [axis, "n", "compliance"]
        for m in metrics:
            header.append(_short_metric(m))
            if has_default:
                header.append("Δ")
        if has_default:
            header.append("effect?")

        rows: list[list[str]] = []
        for key, gk in g_exp.groupby(axis):
            rate = compliance_rate(gk)
            row = [
                str(key),
                str(len(gk)),
                "—" if rate is None else f"{rate * 100:.0f}%",
            ]
            sig: dict[str, bool] = {}
            for m in metrics:
                st = _metric_stats(gk, m, population)
                row.append("—" if st is None else f"{st[0]:.4g}")
                if not has_default:
                    continue
                bst = base_stats.get(m)
                if key == baseline_key or bst is None or base is None:
                    row.append("—")
                    continue
                pd_ = _paired_delta(base, gk, m, population)
                if pd_ is None:
                    row.append("—")
                    continue
                mean_d, ci_d, pn = pd_
                min_paired_n = pn if min_paired_n is None else min(min_paired_n, pn)
                # Magnitude-relative floor so FP noise on a constant metric
                # (CI=0 on both sides) can't trip "yes".
                scale = abs(bst[0]) + (abs(st[0]) if st else 0.0)
                tol = 1e-9 * scale + 1e-12
                row.append(f"{0.0 if abs(mean_d) <= tol else mean_d:+.3g}")
                sig[m] = abs(mean_d) > tol and abs(mean_d) > ci_d
            if has_default:
                # Fire if ANY declared target metric moves — every metric in the
                # list was chosen as relevant to the lever (priority-inversion /
                # abandonment / compliance), not just the primary.
                row.append(
                    "—"
                    if key == baseline_key
                    else ("yes" if any(sig.values()) else "ns")
                )
            rows.append(row)
        rows.sort(key=lambda r: (r[0] != baseline_key, r[0]))
        aliaser = {
            "ablation": alias_ablation,
            "sweep": alias_sweep,
            "variant": alias_variant,
        }[axis]
        for r in rows:
            r[0] = aliaser(r[0])
        pop_note = "all runs" if population == "all" else "compliant runs"
        body.append("")
        body.append(
            f"### {exp_name} — {', '.join(_short_metric(m) for m in metrics)} "
            f"over {pop_note}"
        )
        body.append(_markdown_table(header, rows))

    if not body:
        return []
    paired_note = (
        f" Smallest paired n across rows: {min_paired_n}."
        if min_paired_n is not None
        else ""
    )
    return [
        "",
        "## Config justification (per-experiment target metrics)",
        "",
        "_`Δ` is the per-seed paired mean‑difference vs the in‑experiment "
        "`default` (paired on grid·scenario·seed); `effect?`=yes when ANY target "
        "metric's |Δ| exceeds its paired 95% CI (a screen, not a confirmatory "
        "test — read the sign against the expected direction in "
        "CONFIG_JUSTIFICATION.md). `__passed` compliance flags are rates over "
        "all runs; other metrics use the population in the heading, so their "
        "paired n can be below the `n` column." + paired_note + "_",
        *body,
    ]


def _compliant_split(g: pd.DataFrame) -> tuple[pd.Series, float, int, int]:
    """Split a group into ``(pwsf_compliant, compliance_rate, n_compliant, n_total)``.

    ``pwsf_compliant`` is PWSF restricted to tasks passing every compliance claim
    (slack budget AND feasibility); ``rate`` is ``nan`` when no compliance column
    exists so the caller suppresses it rather than reporting a fake 100%. Gating
    is required because a variant can inflate PWSF two ways the oracle cannot —
    overdrawing slack or crediting load delivered out of bounds — making the mean
    non-comparable to the constraint-respecting oracle otherwise.
    """
    full = (
        g[_PRIMARY_OUTCOME].dropna()
        if _PRIMARY_OUTCOME in g.columns
        else pd.Series(dtype=float)
    )
    n_total = int(len(full))
    present = [c for c in COMPLIANCE_COLS if c in g.columns]
    if not present:
        # No compliance columns: can't verify, so report every task and
        # suppress the rate column.
        return full, float("nan"), n_total, n_total
    # Conjunction across the available compliance flags, restricted to the
    # PWSF-defined rows (NaN => False, handled by ``compliant_mask``).
    passed_bool = compliant_mask(g).loc[full.index]
    n_compliant = int(passed_bool.sum())
    pwsf_compliant = full[passed_bool]
    rate = float(n_compliant / n_total) if n_total else float("nan")
    return pwsf_compliant, rate, n_compliant, n_total


def _paired_vs_oracle_section(ok_vc: pd.DataFrame) -> list[str]:
    """Per-grid PWSF of each non-oracle variant vs the oracle, PAIRED on
    identical task identity (every key but ``variant``) with BOTH sides
    compliant and PWSF-defined.

    The unpaired ``Variant comparison`` table above takes each variant's own
    compliant-subset mean, which can manufacture inversions: a variant scored
    on an easier surviving subset than the oracle (e.g. the oracle silently
    dropping MILP-crash tasks, status=error => PWSF NaN) is not comparable.
    Pairing on task identity and intersecting the compliant sets removes both
    the unequal-task-set and unequal-compliant-subset confounds, so ``Δ vs
    oracle`` is the honest optimality gap (negative => variant below oracle).
    """
    if ok_vc.empty or "variant" not in ok_vc.columns:
        return []
    if "oracle" not in set(ok_vc["variant"]) or _PRIMARY_OUTCOME not in ok_vc.columns:
        return []
    key = [
        c
        for c in ("seed", "scenario", "experiment", "ablation", "sweep")
        if c in ok_vc.columns
    ]
    if not key:
        return []
    rows: list[list[str]] = []
    for grid, g in ok_vc.groupby("grid"):
        gc = g[compliant_mask(g) & g[_PRIMARY_OUTCOME].notna()]
        og = gc[gc["variant"] == "oracle"].drop_duplicates(key)
        orc = og.set_index(key)[_PRIMARY_OUTCOME]
        if orc.empty:
            continue
        # Oracle runs that hit the solver time limit are not a reference: a
        # variant "beating" one is measuring the time limit, not the ladder.
        if _ORACLE_OPTIMAL_COL in og.columns:
            certified = og.set_index(key)[_ORACLE_OPTIMAL_COL].map(_as_bool)
        else:
            certified = None
        for variant, gv in gc.groupby("variant"):
            if variant == "oracle":
                continue
            vv = gv.drop_duplicates(key).set_index(key)[_PRIMARY_OUTCOME]
            common = vv.index.intersection(orc.index)
            if len(common) == 0:
                continue
            v_arr = vv.loc[common]
            o_arr = orc.loc[common]
            vm, _ = mean_ci95(v_arr)
            om, _ = mean_ci95(o_arr)
            v_num = v_arr.to_numpy()
            o_num = o_arr.to_numpy()
            beats = v_num > o_num + 1e-9
            wins = int(beats.sum())
            if certified is None:
                cert_cols = ["—", "—", "—"]
                usable_cols = ["—", "—", "—"]
            else:
                cert_mask = certified.reindex(common).fillna(False).to_numpy()
                cert_cols = _subset_cols(v_num, o_num, cert_mask)
                # Three-way verdict. The oracle's PWSF comes from a FEASIBLE
                # incumbent, so on an uncertified pair where the oracle already
                # matches or beats the variant, the true optimum is at least as
                # good and "variant trails oracle" still holds — only the
                # reverse case is indeterminate. This recovers the pairs the
                # binary certified filter discards, and discarding them is not
                # neutral: certified ⊂ shallow-deficit (on eval_full_v2_
                # 20260727 uncertified oracle tasks carry ~2x the weighted shed,
                # 18x on lv_reconfig, and certification falls monotonically with
                # failure count), so the certified subset is the easy tail.
                usable_mask = cert_mask | ~beats
                usable_cols = _subset_cols(v_num, o_num, usable_mask)
                usable_cols[-1] = str(int((~usable_mask).sum()))
            rows.append(
                [
                    alias_grid(grid),
                    alias_variant(variant),
                    str(len(common)),
                    f"{vm:.4f}",
                    f"{om:.4f}",
                    f"{vm - om:+.4f}",
                    f"{wins}/{len(common)}",
                    *cert_cols,
                    *usable_cols,
                ]
            )
    if not rows:
        return []
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        "",
        "## Variant vs oracle (paired on task identity, both compliant)",
        "",
        "_Each non-oracle variant against the oracle on the SAME tasks where "
        "both are compliant and have a defined PWSF. `Δ vs oracle` < 0 means the "
        "variant trails the oracle. Differs from the unpaired table above, which "
        "can fabricate inversions when the oracle's compliant/surviving subset "
        "differs from the variant's._\n\n"
        "_The oracle is **not** a PWSF upper bound: it maximises a "
        "near-lexicographic tier ladder (`oracle._ORACLE_TIER_WEIGHT` "
        "1e6/1e4/1e2/1) and is scored on PWSF's 8:4:2:1, so `variant>oracle` can "
        "be honest. It is also not a bound when the solve hit the time limit. "
        "The `certified` columns restrict to pairs where the oracle reported "
        "`solve_optimal`: on eval_full_v2 SCARE beat the oracle in 5.08% of "
        "time-limited pairs vs **0.23%** of certified ones, a 22x enrichment._\n\n"
        "_**Read the `usable` columns as the result**, not `certified`. Every "
        "uncertified oracle solve is a wall-clock time-out with a FEASIBLE "
        "incumbent (eval_full_v2_20260727: 862 `OPTIMAL` / 483 `TIME_LIMIT` / 5 "
        "`INFEASIBLE`, and no uncertified row lacks an incumbent), so on a pair "
        "where the oracle already matches or beats the variant the true optimum "
        "is at least as good and the conclusion holds regardless of convergence. "
        "Only `indeterminate` pairs — uncertified AND variant-ahead — are "
        "unusable. This matters because `certified` is a BIASED filter: "
        "uncertified oracle tasks carry ~2x the weighted shed (18x on "
        "lv_reconfig) and certification falls monotonically with failure count, "
        "so certified ⊂ the easy tail, and on eval_full_v2_20260727 restricting "
        "to it flipped the sign of the backup-branches row. Caveat: the "
        "monotone argument is rigorous in the oracle's own tier ladder and only "
        "heuristic in PWSF, and on `usable` pairs `Δ` is a LOWER bound on the "
        "variant's true shortfall._",
        _markdown_table(
            [
                "grid",
                "variant",
                "n_paired",
                "mean PWSF",
                "oracle PWSF",
                "Δ vs oracle",
                "variant>oracle",
                "n_certified",
                "Δ (certified)",
                "variant>oracle (certified)",
                "n_usable",
                "Δ (usable)",
                "n_indeterminate",
            ],
            rows,
        ),
    ]


def _format_eval_sections(
    df: pd.DataFrame, targets: dict[str, tuple[list[str], str]] | None = None
) -> list[str]:
    parts: list[str] = []
    targets = targets or {}
    mapped = set(targets)
    if df.empty or "variant" not in df.columns:
        return parts
    # Include ``claims_failed`` runs: PWSF comes from a completed sim; the
    # claim flagged a priority inversion but the served value is valid.
    ok = df[df["status"].isin(("ok", "claims_failed"))]
    if ok.empty or _PRIMARY_OUTCOME not in ok.columns:
        return parts

    # Variant comparison per grid. PWSF mean restricted to compliant
    # tasks (see ``_compliant_split``); ``compliance`` reports the
    # fraction honouring the operator's slack-budget policy.
    parts.append("")
    # Mapped experiments own their variant rows in the justification section.
    ok_vc = ok[~ok["experiment"].isin(mapped)] if "experiment" in ok.columns else ok
    parts.append(
        "## Variant comparison (priority-weighted served on compliant runs, mean ± 95% CI)"
    )
    # Count tasks with no defined PWSF (error/timeout/killed/missing) per
    # (grid, variant) from the FULL df, so silently-dropped runs — notably the
    # oracle's MILP crashes (e.g. LV-recfg drops 65/105) — are disclosed in the
    # ``dropped`` column instead of vanishing from an n that only sees completed
    # runs.
    df_vc = df[~df["experiment"].isin(mapped)] if "experiment" in df.columns else df
    _DROPPED = ("error", "timeout", "killed", "missing")
    dropped_counts = (
        df_vc[df_vc["status"].isin(_DROPPED)]
        .groupby(["grid", "variant"])
        .size()
        .to_dict()
    )
    rows = []
    for (grid, variant), g in ok_vc.groupby(["grid", "variant"]):
        pwsf_c, rate, n_c, n_t = _compliant_split(g)
        if n_t == 0:
            continue
        if pwsf_c.empty:
            mean_str, ci_str = "—", "—"
        else:
            mean, ci = mean_ci95(pwsf_c)
            mean_str = f"{mean:.4f}"
            ci_str = "—" if pd.isna(ci) else f"±{ci:.4f}"
        rate_str = "—" if pd.isna(rate) else f"{rate * 100:.0f}%"
        rows.append(
            [
                alias_grid(grid),
                alias_variant(variant),
                f"{n_c}/{n_t}",
                rate_str,
                str(dropped_counts.get((grid, variant), 0)),
                mean_str,
                ci_str,
            ]
        )
    if rows:
        rows.sort(key=lambda r: (r[0], r[1]))
        parts.append(
            _markdown_table(
                [
                    "grid",
                    "variant",
                    "n_compliant/n_total",
                    "compliance",
                    "dropped",
                    "mean PWSF",
                    "95% CI",
                ],
                rows,
            )
        )

    parts.extend(_paired_vs_oracle_section(ok_vc))

    # Ablation impact: only experiments that define an ablation matrix
    # (else the "default" baseline pools across non-ablation experiments
    # and washes out the Δ). Restrict to scare-variant rows whose
    # experiment has >1 distinct ablation key, baselined on its own
    # in-experiment ``default``.
    # Target-mapped experiments are owned by the justification section below;
    # exclude them here so they aren't also reported on the PWSF headline.
    abl_all = ok[(ok["variant"] == "scare") & (~ok["experiment"].isin(mapped))]
    abl_experiments = []
    if "experiment" in abl_all.columns and "ablation" in abl_all.columns:
        for exp_name, g in abl_all.groupby("experiment"):
            if g["ablation"].nunique() > 1:
                abl_experiments.append(exp_name)
    abl = (
        abl_all[abl_all["experiment"].isin(abl_experiments)]
        if abl_experiments
        else abl_all.iloc[0:0]
    )
    if "ablation" in abl.columns and abl["ablation"].nunique() > 1:
        parts.append("")
        parts.append("## Ablation impact (SCARE variant only, compliant runs)")
        parts.append(
            "_Each ablation arm vs its OWN in-experiment baseline — the "
            "unablated `default` arm, shown as *full system* — "
            "(same experiment + grid + scenario set). Pooling defaults across "
            "experiments/grids — as a single combined table did — diffs an arm "
            "against an unrelated population and manufactures spurious Δ. Read Δ "
            "together with `n_compliant`: a Δ on a tiny compliant subset (e.g. "
            "1/15) is a single-sample artefact, not an effect._"
        )
        # One table PER ablation experiment; the ``default`` baseline is scoped to
        # the SAME (experiment, grid) as each arm. Most ablation experiments are
        # single-grid, but some (e.g. line_stress) span two grids with arms on
        # different grids, so pooling the default across grids — even within one
        # experiment — would still diff an arm against an unrelated baseline.
        for exp_name in sorted(abl_experiments):
            g_exp = abl[abl["experiment"] == exp_name]
            if g_exp["ablation"].nunique() <= 1:
                continue
            multi_grid = g_exp["grid"].nunique() > 1
            rows = []
            for grid in sorted(g_exp["grid"].unique()):
                g_grid = g_exp[g_exp["grid"] == grid]
                if g_grid["ablation"].nunique() <= 1:
                    continue
                baseline_pwsf_c, _, _, _ = _compliant_split(
                    g_grid[g_grid["ablation"] == "default"]
                )
                b_mean = (
                    float(baseline_pwsf_c.mean())
                    if not baseline_pwsf_c.empty
                    else float("nan")
                )
                for ablation_key, g in g_grid.groupby("ablation"):
                    pwsf_c, rate, n_c, n_t = _compliant_split(g)
                    if n_t == 0:
                        continue
                    mean = float(pwsf_c.mean()) if not pwsf_c.empty else float("nan")
                    diff = (
                        mean - b_mean
                        if not (pd.isna(b_mean) or pd.isna(mean))
                        else float("nan")
                    )
                    rate_str = "—" if pd.isna(rate) else f"{rate * 100:.0f}%"
                    row = [
                        ablation_key,
                        f"{n_c}/{n_t}",
                        rate_str,
                        "—" if pd.isna(mean) else f"{mean:.4f}",
                        "—" if pd.isna(diff) else f"{diff:+.4f}",
                    ]
                    if multi_grid:
                        row.insert(0, alias_grid(grid))
                    rows.append(row)
            if not rows:
                continue
            # Pin each grid's ``default`` row first within its grid block;
            # alias to display names only after the raw-key sort.
            rows.sort(
                key=lambda r: (
                    (r[0], r[1] != "default", r[1])
                    if multi_grid
                    else (r[0] != "default", r[0])
                )
            )
            key_idx = 1 if multi_grid else 0
            for r in rows:
                r[key_idx] = alias_ablation(r[key_idx])
            header = [
                "ablation",
                "n_compliant/n_total",
                "compliance",
                "mean PWSF",
                "Δ vs full system",
            ]
            if multi_grid:
                header = ["grid"] + header
            parts.append("")
            parts.append(f"### {alias_experiment(exp_name)}")
            parts.append(_markdown_table(header, rows))

    # Sweep curves (target-mapped experiments owned by the justification
    # section below).
    ok_sw = ok[~ok["experiment"].isin(mapped)] if "experiment" in ok.columns else ok
    if "sweep" in ok_sw.columns and ok_sw["sweep"].nunique() > 1:
        parts.append("")
        parts.append(
            "## Sensitivity sweeps (priority-weighted served on compliant runs)"
        )
        parts.append("")
        parts.append(
            "_Read `mean regulates` beside `mean PWSF`: a degradation sweep can "
            "hold PWSF up simply because the controller stopped acting. On "
            "eval_full_v2's packet-loss arms `regulates_total` fell monotonically "
            "4181 -> 1352 -> 899 -> 660 -> 274 across 0/5/10/20/50% loss while "
            "PWSF was non-monotone (20% scored 0.784, 50% scored 0.889, paired "
            "p=8.6e-13) — the curve tracks the controller falling silent, not "
            "graceful degradation. An arm whose PWSF holds while its regulate "
            "count collapses is not evidence of robustness._"
        )
        rows = []
        for sweep_key, g in ok_sw.groupby("sweep"):
            pwsf_c, rate, n_c, n_t = _compliant_split(g)
            if n_t == 0:
                continue
            t = g[_TIME_TO_STABILISE].dropna()
            r = g[_REGULATES_TOTAL].dropna()
            mean_str = "—" if pwsf_c.empty else f"{pwsf_c.mean():.4f}"
            rate_str = "—" if pd.isna(rate) else f"{rate * 100:.0f}%"
            rows.append(
                [
                    sweep_key,
                    f"{n_c}/{n_t}",
                    rate_str,
                    mean_str,
                    f"{t.mean():.2f}" if not t.empty else "—",
                    f"{int(r.mean())}" if not r.empty else "—",
                ]
            )
        rows.sort(key=lambda r: (r[0] != "default", r[0]))
        for r in rows:
            r[0] = alias_sweep(r[0])
        parts.append(
            _markdown_table(
                [
                    "sweep",
                    "n_compliant/n_total",
                    "compliance",
                    "mean PWSF",
                    "mean t_stab (s)",
                    "mean regulates",
                ],
                rows,
            )
        )

    # Scenarios — robustness + cascading.
    if "scenario" in ok.columns and ok["scenario"].nunique() > 1:
        parts.append("")
        parts.append("## Scenario comparison (compliant runs)")
        rows = []
        for scenario_key, g in ok.groupby("scenario"):
            pwsf_c, rate, n_c, n_t = _compliant_split(g)
            if n_t == 0:
                continue
            mean_str = "—" if pwsf_c.empty else f"{pwsf_c.mean():.4f}"
            rate_str = "—" if pd.isna(rate) else f"{rate * 100:.0f}%"
            rows.append(
                [
                    scenario_key,
                    f"{n_c}/{n_t}",
                    rate_str,
                    mean_str,
                ]
            )
        rows.sort(key=lambda r: r[0])
        for r in rows:
            r[0] = alias_scenario(r[0])
        parts.append(
            _markdown_table(
                ["scenario", "n_compliant/n_total", "compliance", "mean PWSF"],
                rows,
            )
        )

    # Restoration vs no-failure baseline — surface absolute load lost
    # alongside the priority-weighted view, so "PWSF 99%" vs "10 MW of
    # low-priority load dropped" isn't hidden in CSV columns.
    base_col = "outcomes__restoration__total_served_baseline_mw"
    post_col = "outcomes__restoration__total_served_post_mw"
    drop_col = "outcomes__restoration__absolute_load_dropped_mw"
    raw_col = "outcomes__restoration__raw_restoration_ratio"
    pwsf_col = "outcomes__restoration__pwsf_restoration_ratio"
    if base_col in ok.columns and post_col in ok.columns:
        rest = ok[ok["variant"] == "scare"] if "scare" in ok["variant"].unique() else ok
        rest = rest.dropna(subset=[base_col, post_col])
        if not rest.empty:
            parts.append("")
            parts.append("## Restoration vs no-failure baseline (SCARE, mean)")
            rows = []
            for grid, g in rest.groupby("grid"):
                base_mw = float(g[base_col].mean())
                post_mw = float(g[post_col].mean())
                drop_mw = (
                    float(g[drop_col].mean())
                    if drop_col in g.columns
                    else max(0.0, base_mw - post_mw)
                )
                raw_r = (
                    float(g[raw_col].mean())
                    if raw_col in g.columns
                    else (post_mw / base_mw if base_mw else 1.0)
                )
                pwsf_r = (
                    float(g[pwsf_col].mean()) if pwsf_col in g.columns else float("nan")
                )
                pct = drop_mw / base_mw if base_mw else 0.0
                rows.append(
                    [
                        alias_grid(grid),
                        len(g),
                        f"{base_mw:.3f}",
                        f"{post_mw:.3f}",
                        f"{drop_mw:.3f}",
                        f"{pct * 100:.1f}%",
                        f"{raw_r:.3f}",
                        f"{pwsf_r:.3f}" if not (pwsf_r != pwsf_r) else "—",
                    ]
                )
            rows.sort(key=lambda r: -float(r[4]))  # biggest drop first
            parts.append(
                _markdown_table(
                    [
                        "grid",
                        "n",
                        "base MW",
                        "post MW",
                        "dropped MW",
                        "% of base",
                        "raw ratio",
                        "PWSF ratio",
                    ],
                    rows,
                )
            )

    # Diary invariant rate per variant — flag where the protocol drifts.
    inv_col = "diary__invariant_holds"
    if inv_col in df.columns:
        parts.append("")
        parts.append("## Diary invariant compliance")
        # The oracle is an LP: it runs no negotiations, so it has no diary and
        # the invariant holds over an empty one. Reporting that beside a MAS
        # variant's rate reads as "equally correct" when the oracle is not in
        # the comparison at all — so count the rows that actually had a diary.
        diary_counters = [
            c for c in df.columns if c.startswith("diary__") and c != inv_col
        ]
        rows = []
        for variant, g in df.groupby("variant"):
            s = g[inv_col].dropna()
            if s.empty:
                continue
            if diary_counters:
                observed = int(g.loc[s.index, diary_counters].notna().any(axis=1).sum())
            else:
                observed = len(s)
            ok_pct = 100.0 * float(s.astype(bool).sum()) / len(s)
            rows.append(
                [
                    alias_variant(variant),
                    len(s),
                    f"{ok_pct:.1f}%" if observed else "n/a (no diary)",
                    str(len(s) - observed),
                ]
            )
        parts.append(
            _markdown_table(
                ["variant", "n", "invariant holds", "n with no diary"],
                rows,
            )
        )
        parts.append("")
        parts.append(
            "_`n with no diary` counts rows where the invariant was evaluated "
            "against no recorded negotiation, so it holds vacuously. The oracle "
            "solves an LP and never negotiates, so its whole column is vacuous "
            "and is shown as `n/a` rather than a perfect score._"
        )

    parts.extend(_format_justification_sections(ok, targets))
    return parts


def write_summary(campaign_dir: Path) -> tuple[Path, Path]:
    df = aggregate(campaign_dir)
    targets = _load_experiment_targets(campaign_dir)
    csv_path = campaign_dir / CAMPAIGN_LAYOUT["summary_csv"]
    md_path = campaign_dir / CAMPAIGN_LAYOUT["summary_md"]
    df.to_csv(csv_path, index=False)
    md_path.write_text(_format_md(df, targets), encoding="utf-8")

    n = len(df)
    by_status = df["status"].value_counts().to_dict() if n else {}
    logger.info("Aggregated %d task(s) → %s", n, csv_path)
    for s in ("ok", "claims_failed", "error", "timeout", "killed", "missing"):
        if s in by_status:
            logger.info("  %-14s %d", s, by_status[s])
    if "solver_failures" in df.columns and df["solver_failures"].notna().any():
        logger.info(
            "  total solver-failure warnings: %d",
            int(df["solver_failures"].fillna(0).sum()),
        )
    logger.info("Markdown report → %s", md_path)
    return csv_path, md_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--campaign-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
    args = _parse_args()
    write_summary(args.campaign_dir.resolve())


if __name__ == "__main__":
    main()
