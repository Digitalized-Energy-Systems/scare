"""Post-aggregate analysis for the cross-sector coalition campaign.

Reads the aggregator's ``summary.csv`` plus each task's ``events.csv``
and writes:

* ``cp_coalition_findings.md`` — per-experiment / per-grid breakdown of
  the with- vs without-cross-sector arms (PWSF, restoration ratio,
  cross-sector event-count roll-up).
* ``plots/`` — every figure from :mod:`experiment.eval.cp_plots`,
  rendered against the representative task's ledger and the
  flag-on/flag-off comparison summary.

Usage:
    python -m experiment.eval.cp_campaign_analysis \\
        --campaign-dir experiment/_runs/eval/cp_coalition_eval

Defaults to the most recent ``cp_coalition_eval*`` directory under
``experiment/_runs/eval/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.eval import cp_plots
from experiment.hpc.config import CAMPAIGN_LAYOUT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


XS_EVENT_KINDS: tuple[str, ...] = (
    "cross_sector_inversion_detected",
    "cross_sector_coalition_allocation",
    "cp_envelope_set",
    "cp_envelope_clamp",
    "cp_setpoint",
    "cp_admm_skipped_same_sign",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _campaign_dir_default(base: Path = Path("experiment/_runs/eval")) -> Path | None:
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("cp_coalition_eval*"))
    return candidates[-1] if candidates else None


def _load_summary(campaign_dir: Path) -> pd.DataFrame:
    summary_csv = campaign_dir / CAMPAIGN_LAYOUT["summary_csv"]
    if not summary_csv.exists():
        raise SystemExit(
            f"summary.csv missing — run `python -m experiment.hpc.aggregate "
            f"--campaign-dir {campaign_dir}` first."
        )
    return pd.read_csv(summary_csv)


def _events_for_task(campaign_dir: Path, task_id: int) -> pd.DataFrame:
    path = (
        campaign_dir / CAMPAIGN_LAYOUT["tasks"]
        / f"{int(task_id):06d}" / "events.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _events_json_for_task(campaign_dir: Path, task_id: int) -> Path:
    """Materialise the task's events.csv as the events.json layout the
    cp_plots helpers read.
    """
    csv_path = (
        campaign_dir / CAMPAIGN_LAYOUT["tasks"]
        / f"{int(task_id):06d}" / "events.csv"
    )
    json_path = csv_path.with_name("events.json")
    if csv_path.exists() and not json_path.exists():
        df = pd.read_csv(csv_path)
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            records.append({
                "t": float(row.get("t", 0.0)),
                "kind": str(row.get("kind", "")),
                "aid": str(row.get("aid", "")),
                "sector": str(row.get("sector", "")),
                "detail": str(row.get("detail", "")),
            })
        json_path.write_text(json.dumps(records, indent=2))
    return json_path


def _summary_json_for_run(
    campaign_dir: Path, task_ids: list[int], out_path: Path,
) -> Path:
    """Aggregate a set of tasks' event counts into the summary.json
    layout the flag_on_off_comparison helper reads.
    """
    all_counts: dict[str, int] = defaultdict(int)
    for tid in task_ids:
        df = _events_for_task(campaign_dir, tid)
        if df.empty:
            continue
        for kind, n in df.groupby("kind").size().items():
            all_counts[kind] += int(n)
    xs_counts = {k: all_counts.get(k, 0) for k in XS_EVENT_KINDS}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"all": dict(all_counts), "cross_sector": xs_counts},
        indent=2, sort_keys=True,
    ))
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LABEL_RE = re.compile(r"\$label=([^;]+)")


def _label_from_ablation_key(key: str) -> str:
    """Pull ``$label`` out of the sorted ``k=v`` ablation key, else
    return the full key.
    """
    m = _LABEL_RE.search(key)
    return m.group(1) if m else key


def _xs_event_counts_for_task(
    campaign_dir: Path, task_id: int,
) -> dict[str, int]:
    df = _events_for_task(campaign_dir, task_id)
    counts = {k: 0 for k in XS_EVENT_KINDS}
    if df.empty:
        return counts
    for kind, n in df.groupby("kind").size().items():
        if kind in counts:
            counts[kind] = int(n)
    return counts


# ---------------------------------------------------------------------------
# Findings report
# ---------------------------------------------------------------------------


def _format_metric(values: pd.Series) -> str:
    valid = values.dropna()
    if valid.empty:
        return "(n/a)"
    if len(valid) == 1:
        return f"{float(valid.iloc[0]):.3f}  (n=1)"
    return f"{float(valid.mean()):.3f} ± {float(valid.std()):.3f}  (n={len(valid)})"


def _format_int_counts(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "0"
    total = int(valid.sum())
    if len(valid) == 1:
        return f"{total}"
    return f"{total} (μ={valid.mean():.1f}/task)"


def write_findings_report(
    campaign_dir: Path,
    summary: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Per-experiment / per-grid markdown: with/without cross-sector
    deltas plus the event-count roll-up.
    """
    pwsf = "outcomes__priority_weighted_fraction"
    restoration_col = "outcomes__restoration"
    served_col = "outcomes__served_fraction"

    # Attach per-task event-count columns alongside the metric deltas.
    xs_counts_per_task: dict[int, dict[str, int]] = {}
    for tid in summary["task_id"].astype(int):
        xs_counts_per_task[int(tid)] = _xs_event_counts_for_task(
            campaign_dir, int(tid)
        )
    for kind in XS_EVENT_KINDS:
        summary[f"xs__{kind}"] = summary["task_id"].map(
            lambda t: xs_counts_per_task.get(int(t), {}).get(kind, 0)
        )

    summary["arm"] = summary["ablation"].apply(_label_from_ablation_key)

    lines: list[str] = []
    lines.append("# Cross-sector coalition campaign — findings")
    lines.append("")
    lines.append(
        f"Source: `{campaign_dir.name}` "
        f"({len(summary)} tasks, {int((summary['status'] == 'ok').sum())} ok)."
    )
    lines.append("")
    lines.append(
        "Metrics shown per cell (mean ± std across seeds; n in parens):\n"
        "* **PWSF** — priority-weighted served fraction (the headline metric)\n"
        "* **restoration** — fraction of pre-failure served load recovered\n"
        "* **served_fraction** — total served / total demand\n"
        "* **xs_inv** — `cross_sector_inversion_detected` event total\n"
        "* **xs_alloc** — `cross_sector_coalition_allocation` event total\n"
        "* **env_set** — `cp_envelope_set` (commitments dispatched)\n"
        "* **env_clamp** — `cp_envelope_clamp` (CP ADMM overridden)\n"
    )

    for exp_name, exp_df in summary.groupby("experiment"):
        if not exp_name:
            continue
        lines.append(f"\n## {exp_name}")
        for grid, grid_df in exp_df.groupby("grid"):
            lines.append(f"\n### {grid}")
            lines.append("")
            lines.append(
                "| arm | n_ok | PWSF | restoration | served | "
                "xs_inv | xs_alloc | env_set | env_clamp |"
            )
            lines.append(
                "|---|---:|---|---|---|---:|---:|---:|---:|"
            )
            ok_df = grid_df[grid_df["status"] == "ok"]
            for arm, arm_df in sorted(
                ok_df.groupby("arm"), key=lambda kv: kv[0],
            ):
                lines.append(
                    f"| **{arm}** | {len(arm_df)} | "
                    f"{_format_metric(arm_df.get(pwsf, pd.Series([], dtype=float)))} | "
                    f"{_format_metric(arm_df.get(restoration_col, pd.Series([], dtype=float)))} | "
                    f"{_format_metric(arm_df.get(served_col, pd.Series([], dtype=float)))} | "
                    f"{_format_int_counts(arm_df['xs__cross_sector_inversion_detected'])} | "
                    f"{_format_int_counts(arm_df['xs__cross_sector_coalition_allocation'])} | "
                    f"{_format_int_counts(arm_df['xs__cp_envelope_set'])} | "
                    f"{_format_int_counts(arm_df['xs__cp_envelope_clamp'])} |"
                )

            # Delta row between the with/without arms.
            arms = sorted(ok_df["arm"].unique())
            if len(arms) == 2 and pwsf in ok_df.columns:
                v0 = ok_df[ok_df["arm"] == arms[0]][pwsf].dropna()
                v1 = ok_df[ok_df["arm"] == arms[1]][pwsf].dropna()
                if len(v0) and len(v1):
                    delta = v1.mean() - v0.mean()
                    lines.append("")
                    lines.append(
                        f"**Δ PWSF** ({arms[1]} − {arms[0]}) = "
                        f"`{delta:+.3f}`"
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------


def _pick_representative_task(
    summary: pd.DataFrame, *, prefer_label: str = "with_xs"
) -> int | None:
    """Pick the most cross-sector-active task: largest
    ``cp_envelope_set``, then largest
    ``cross_sector_coalition_allocation``, restricted to
    ``arm == prefer_label``.
    """
    if summary.empty:
        return None
    df = summary[summary["status"] == "ok"]
    if df.empty:
        df = summary
    df = df.assign(arm=df["ablation"].apply(_label_from_ablation_key))
    df = df[df["arm"] == prefer_label]
    if df.empty:
        return None
    if "xs__cp_envelope_set" in df.columns:
        df = df.sort_values(
            ["xs__cp_envelope_set", "xs__cross_sector_coalition_allocation"],
            ascending=False,
        )
    return int(df["task_id"].iloc[0])


def render_campaign_plots(
    campaign_dir: Path,
    summary: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """Render every CP-focused figure off the campaign output.

    * ``representative_run/`` — figures off the most cross-sector-active
      task in the ``with_xs`` arm.
    * ``flag_comparison/`` — with-vs-without bar chart aggregated across
      the ``xs_coalition_ablation`` experiment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    rep_task_id = _pick_representative_task(summary, prefer_label="with_xs")
    if rep_task_id is not None:
        rep_dir = (
            campaign_dir / CAMPAIGN_LAYOUT["tasks"] / f"{rep_task_id:06d}"
        )
        events_json = _events_json_for_task(campaign_dir, rep_task_id)
        rep_out = out_dir / "representative_run"
        rep_out.mkdir(parents=True, exist_ok=True)
        # Render so paths land under out_dir, not the task dir.
        written.update({
            "cp_setpoint_timeline": cp_plots.cp_setpoint_timeline(
                events_json, rep_out / "cp_setpoint_timeline",
            ),
            "coalition_lifecycle_gantt": cp_plots.coalition_lifecycle_gantt(
                events_json, rep_out / "coalition_lifecycle_gantt",
            ),
            "envelope_clamp_arrows": cp_plots.envelope_clamp_arrows(
                events_json, rep_out / "envelope_clamp_arrows",
            ),
            "cross_sector_transfer_distribution": (
                cp_plots.cross_sector_transfer_distribution(
                    events_json,
                    rep_out / "cross_sector_transfer_distribution",
                )
            ),
        })
        logger.info(
            "Representative task = %06d (CP-active run from with_xs arm)",
            rep_task_id,
        )

    # Aggregate flag-comparison summaries over the ablation experiment only.
    ablation = summary[summary["experiment"] == "xs_coalition_ablation"].copy()
    if not ablation.empty:
        ablation["arm"] = ablation["ablation"].apply(_label_from_ablation_key)
        comp_dir = out_dir / "flag_comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        on_ids = ablation[ablation["arm"] == "with_xs"]["task_id"].astype(int).tolist()
        off_ids = ablation[ablation["arm"] == "without_xs"]["task_id"].astype(int).tolist()
        on_summary = _summary_json_for_run(
            campaign_dir, on_ids, comp_dir / "on" / "summary.json",
        )
        off_summary = _summary_json_for_run(
            campaign_dir, off_ids, comp_dir / "off" / "summary.json",
        )
        written["flag_on_off_comparison"] = cp_plots.flag_on_off_comparison(
            off_summary, on_summary, comp_dir / "flag_on_off_comparison",
        )

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    campaign_dir = args.campaign_dir or _campaign_dir_default()
    if campaign_dir is None or not campaign_dir.is_dir():
        raise SystemExit(
            "no campaign directory found; pass --campaign-dir explicitly."
        )
    campaign_dir = campaign_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else campaign_dir / "analysis"
    )

    summary = _load_summary(campaign_dir)
    logger.info("Loaded %d tasks from %s", len(summary), campaign_dir)

    findings = write_findings_report(
        campaign_dir, summary,
        out_dir / "cp_coalition_findings.md",
    )
    logger.info("Wrote findings: %s", findings)

    plots = render_campaign_plots(
        campaign_dir, summary, out_dir / "plots",
    )
    for name, stem in plots.items():
        logger.info("  %s -> %s.html", name, stem)


if __name__ == "__main__":
    main()
