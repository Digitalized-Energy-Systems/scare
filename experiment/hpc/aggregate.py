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
        # Flatten the structured eval payload (outcomes / diary /
        # events / messages) into top-level columns so downstream
        # comparisons in the Markdown / pandas can group on them.
        flat = _flatten(result)
        rows.append({
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
            "exception_type": (exception or {}).get("type"),
            "exception_message": (exception or {}).get("message"),
            **flat,
        })
    return pd.DataFrame(rows)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into ``a__b__c`` style keys.  Lists keep
    a single column with their JSON repr — pandas isn't great at
    list-typed cells but the aggregator's main consumers (per-tier
    served, per-sector violation integrals) are dicts, not lists."""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}__{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _key_of(d) -> str:
    """Stable string key for an ablation / sweep / scenario value.
    Used as a grouping column.  Sorted keys → reproducible across
    runs even if dict insertion order differs.

    Robust to upstream variations: a dict gets joined; a string is
    passed through; anything else is repr'd.  Empty / null inputs
    map to ``"default"`` so the grouping column has no NaNs.
    """
    if d is None or d == {} or d == "":
        return "default"
    if isinstance(d, dict):
        return ";".join(f"{k}={d[k]}" for k in sorted(d))
    if isinstance(d, str):
        return d
    return repr(d)


# ---- Markdown report --------------------------------------------------------


_INTERESTING_METRIC_SUFFIXES = ("__last", "__mean", "__min")


def _interesting_metric_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if any(col.endswith(s) for s in _INTERESTING_METRIC_SUFFIXES):
            out.append(col)
    return sorted(out)[:8]  # cap so the table stays readable


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _format_md(df: pd.DataFrame) -> str:
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
                parts.append(f"- Mean duration: {dur.mean():.1f}s "
                             f"(p95 {dur.quantile(0.95):.1f}s)")
        if "solver_failures" in df.columns:
            parts.append(f"- Total solver-failure warnings: "
                         f"{int(df['solver_failures'].fillna(0).sum())}")

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
        try:
            from experiment.eval.aliases import alias_grid as _alias_grid
        except Exception:  # pragma: no cover
            def _alias_grid(g):
                return g
        rows = []
        for grid, r in agg.iterrows():
            rows.append([
                _alias_grid(grid), int(r["total"]), int(r["ok"]),
                int(r["claims_failed"]), int(r["error"]),
                int(r["timeout"]), int(r["killed"]), int(r["missing"]),
                f"{r['mean_duration_s']:.1f}" if pd.notna(r["mean_duration_s"]) else "—",
                f"{r['mean_solver_failures']:.2f}" if pd.notna(r["mean_solver_failures"]) else "—",
            ])
        parts.append(_markdown_table(
            ["grid", "total", "ok", "claims_failed", "error", "timeout",
             "killed", "missing", "mean dur (s)", "mean solver fails"],
            rows,
        ))

    metric_cols = _interesting_metric_columns(df) if n else []
    if metric_cols:
        parts.append("")
        parts.append("## Per-grid metrics (mean over OK runs)")
        # Include ``claims_failed`` here: the simulation ran to completion
        # and the metrics are real — claims failure flags a chapter-level
        # invariant violation, not a missing measurement.
        ok_df = df[df["status"].isin(("ok", "claims_failed"))]
        if ok_df.empty:
            parts.append("_(no successful runs)_")
        else:
            agg = ok_df.groupby("grid")[metric_cols].mean()
            try:
                from experiment.eval.aliases import alias_grid as _alias_grid_m
            except Exception:  # pragma: no cover
                def _alias_grid_m(g):
                    return g
            rows = []
            for grid, r in agg.iterrows():
                rows.append([_alias_grid_m(grid), *[f"{r[c]:.4g}" if pd.notna(r[c]) else "—" for c in metric_cols]])
            parts.append(_markdown_table(["grid", *metric_cols], rows))

    if "exception_type" in df.columns and df["exception_type"].notna().any():
        # Only count exceptions for tasks whose final status is not ``ok``.
        # Stale ``exception.json`` files from a failed run that was later
        # re-run successfully would otherwise inflate the breakdown.
        parts.append("")
        parts.append("## Exception breakdown")
        failed = df[df["status"] != "ok"]
        counts = failed["exception_type"].dropna().value_counts()
        if counts.empty:
            parts.append("_(no failures)_")
        else:
            rows = [[t, int(c)] for t, c in counts.items()]
            parts.append(_markdown_table(["type", "count"], rows))

    parts.extend(_format_eval_sections(df))
    return "\n".join(parts) + "\n"


# ---- Eval-specific sections ------------------------------------------------


_PRIMARY_OUTCOME = "outcomes__priority_weighted_fraction"
_TIME_TO_STABILISE = "outcomes__time_to_stabilise_s"
_REGULATES_TOTAL = "outcomes__regulates_total"


def _format_eval_sections(df: pd.DataFrame) -> list[str]:
    parts: list[str] = []
    if df.empty or "variant" not in df.columns:
        return parts
    # Include ``claims_failed`` runs: the headline ``priority_weighted_fraction``
    # is computed from a completed simulation; the claim flagged a
    # priority inversion but the served value itself is valid.
    ok = df[df["status"].isin(("ok", "claims_failed"))]
    if ok.empty or _PRIMARY_OUTCOME not in ok.columns:
        return parts

    # Variant comparison: scare vs single_level vs oracle, per grid.
    try:
        from experiment.eval.aliases import alias_grid as _ag, alias_variant as _av
    except Exception:  # pragma: no cover
        def _ag(x): return x
        def _av(x): return x
    parts.append("")
    parts.append("## Variant comparison (priority-weighted served, mean ± 95% CI)")
    rows = []
    for (grid, variant), g in ok.groupby(["grid", "variant"]):
        s = g[_PRIMARY_OUTCOME].dropna()
        if s.empty:
            continue
        mean, ci = _mean_ci95(s)
        rows.append([_ag(grid), _av(variant), len(s), f"{mean:.4f}", f"±{ci:.4f}"])
    if rows:
        rows.sort(key=lambda r: (r[0], r[1]))
        parts.append(_markdown_table(["grid", "variant", "n", "mean", "95% CI"], rows))

    # Ablation impact: only rows belonging to an experiment that
    # actually defines an ablation matrix (otherwise the "default"
    # baseline pools across non-ablation experiments and washes out
    # the ablation Δ).  We restrict to scare-variant rows whose
    # experiment has more than one distinct ablation key and use the
    # in-experiment ``default`` as the baseline.
    abl_all = ok[ok["variant"] == "scare"]
    abl_experiments = []
    if "experiment" in abl_all.columns and "ablation" in abl_all.columns:
        for exp_name, g in abl_all.groupby("experiment"):
            if g["ablation"].nunique() > 1:
                abl_experiments.append(exp_name)
    abl = abl_all[abl_all["experiment"].isin(abl_experiments)] if abl_experiments else abl_all.iloc[0:0]
    if "ablation" in abl.columns and abl["ablation"].nunique() > 1:
        parts.append("")
        parts.append("## Ablation impact (scare variant only)")
        baseline_key = "default"
        baseline = abl[abl["ablation"] == baseline_key][_PRIMARY_OUTCOME]
        b_mean = baseline.mean() if not baseline.empty else float("nan")
        rows = []
        for ablation_key, g in abl.groupby("ablation"):
            s = g[_PRIMARY_OUTCOME].dropna()
            if s.empty:
                continue
            mean = float(s.mean())
            diff = mean - b_mean if not pd.isna(b_mean) else float("nan")
            rows.append([
                ablation_key, len(s),
                f"{mean:.4f}",
                "—" if pd.isna(diff) else f"{diff:+.4f}",
            ])
        rows.sort(key=lambda r: (r[0] != "default", r[0]))
        parts.append(_markdown_table(
            ["ablation", "n", "mean", "Δ vs default"],
            rows,
        ))

    # Sweep curves.
    if "sweep" in ok.columns and ok["sweep"].nunique() > 1:
        parts.append("")
        parts.append("## Sensitivity sweeps (priority-weighted served, mean)")
        rows = []
        for sweep_key, g in ok.groupby("sweep"):
            s = g[_PRIMARY_OUTCOME].dropna()
            if s.empty:
                continue
            t = g[_TIME_TO_STABILISE].dropna()
            r = g[_REGULATES_TOTAL].dropna()
            rows.append([
                sweep_key, len(s),
                f"{s.mean():.4f}",
                f"{t.mean():.2f}" if not t.empty else "—",
                f"{int(r.mean())}" if not r.empty else "—",
            ])
        rows.sort(key=lambda r: (r[0] != "default", r[0]))
        parts.append(_markdown_table(
            ["sweep", "n", "mean served", "mean t_stab (s)", "mean regulates"],
            rows,
        ))

    # Scenarios — robustness + cascading.
    if "scenario" in ok.columns and ok["scenario"].nunique() > 1:
        parts.append("")
        parts.append("## Scenario comparison")
        rows = []
        for scenario_key, g in ok.groupby("scenario"):
            s = g[_PRIMARY_OUTCOME].dropna()
            if s.empty:
                continue
            rows.append([scenario_key, len(s), f"{s.mean():.4f}"])
        rows.sort(key=lambda r: r[0])
        parts.append(_markdown_table(
            ["scenario", "n", "mean served"], rows
        ))

    # Restoration vs no-failure baseline — surface absolute load lost
    # alongside the priority-weighted view so the contrast between
    # "PWSF says 99%" and "10 MW of low-priority load actually
    # dropped" isn't hidden in CSV columns.
    base_col = "outcomes__restoration__total_served_baseline_mw"
    post_col = "outcomes__restoration__total_served_post_mw"
    drop_col = "outcomes__restoration__absolute_load_dropped_mw"
    raw_col  = "outcomes__restoration__raw_restoration_ratio"
    pwsf_col = "outcomes__restoration__pwsf_restoration_ratio"
    if base_col in ok.columns and post_col in ok.columns:
        rest = ok[ok["variant"] == "scare"] if "scare" in ok["variant"].unique() else ok
        rest = rest.dropna(subset=[base_col, post_col])
        if not rest.empty:
            parts.append("")
            parts.append("## Restoration vs no-failure baseline (scare, mean)")
            rows = []
            for grid, g in rest.groupby("grid"):
                base_mw = float(g[base_col].mean())
                post_mw = float(g[post_col].mean())
                drop_mw = float(g[drop_col].mean()) if drop_col in g.columns else max(0.0, base_mw - post_mw)
                raw_r   = float(g[raw_col].mean())  if raw_col  in g.columns else (post_mw / base_mw if base_mw else 1.0)
                pwsf_r  = float(g[pwsf_col].mean()) if pwsf_col in g.columns else float("nan")
                pct     = drop_mw / base_mw if base_mw else 0.0
                rows.append([
                    _ag(grid), len(g),
                    f"{base_mw:.3f}", f"{post_mw:.3f}",
                    f"{drop_mw:.3f}", f"{pct*100:.1f}%",
                    f"{raw_r:.3f}",
                    f"{pwsf_r:.3f}" if not (pwsf_r != pwsf_r) else "—",
                ])
            rows.sort(key=lambda r: -float(r[4]))  # biggest absolute drop first
            parts.append(_markdown_table(
                ["grid", "n", "base MW", "post MW", "dropped MW",
                 "% of base", "raw ratio", "PWSF ratio"],
                rows,
            ))

    # Diary invariant rate per variant — flag where the protocol drifts.
    inv_col = "diary__invariant_holds"
    if inv_col in df.columns:
        parts.append("")
        parts.append("## Diary invariant compliance")
        rows = []
        for variant, g in df.groupby("variant"):
            s = g[inv_col].dropna()
            if s.empty:
                continue
            ok_pct = 100.0 * float(s.astype(bool).sum()) / len(s)
            rows.append([_av(variant), len(s), f"{ok_pct:.1f}%"])
        parts.append(_markdown_table(
            ["variant", "n", "invariant holds"], rows,
        ))

    return parts


def _mean_ci95(s: pd.Series) -> tuple[float, float]:
    """Sample mean and 95 % CI half-width via t-distribution (robust to
    small n).  ``ci`` is the half-width — display as ``mean ± ci``.
    """
    import math

    n = len(s)
    if n <= 1:
        return float(s.mean()) if n else 0.0, 0.0
    mean = float(s.mean())
    sd = float(s.std(ddof=1))
    se = sd / math.sqrt(n)
    # t critical for 95 %, 2-sided, df = n−1.  Use 1.96 as a portable
    # default — exact t-table not worth the dependency for n ≥ 30.
    t = 1.96 if n > 30 else 2.262 if n > 10 else 2.776
    return mean, t * se


# ---- Entry point ------------------------------------------------------------


def write_summary(campaign_dir: Path) -> tuple[Path, Path]:
    df = aggregate(campaign_dir)
    csv_path = campaign_dir / CAMPAIGN_LAYOUT["summary_csv"]
    md_path = campaign_dir / CAMPAIGN_LAYOUT["summary_md"]
    df.to_csv(csv_path, index=False)
    md_path.write_text(_format_md(df))

    n = len(df)
    by_status = df["status"].value_counts().to_dict() if n else {}
    logger.info("Aggregated %d task(s) → %s", n, csv_path)
    for s in ("ok", "claims_failed", "error", "timeout", "killed", "missing"):
        if s in by_status:
            logger.info("  %-14s %d", s, by_status[s])
    if "solver_failures" in df.columns and df["solver_failures"].notna().any():
        logger.info("  total solver-failure warnings: %d",
                    int(df["solver_failures"].fillna(0).sum()))
    logger.info("Markdown report → %s", md_path)
    return csv_path, md_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--campaign-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    args = _parse_args()
    write_summary(args.campaign_dir.resolve())


if __name__ == "__main__":
    main()
