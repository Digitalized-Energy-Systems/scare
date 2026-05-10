"""Generate the per-campaign evaluation report.

Reads a campaign directory (must contain ``summary.csv`` from
``experiment.hpc.aggregate``), produces:

- ``plots/<experiment>/*.png``  — one PNG per relevant plot
- ``REPORT.md``                  — Markdown stitching the figures and
                                   an at-a-glance numeric summary

CLI:
    python -m experiment.eval.report --campaign-dir <path>

Idempotent: re-runs overwrite plots in place.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from experiment.eval import plots
from experiment.eval.loader import CampaignData, TaskArtefacts, load_campaign

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-experiment dispatchers
# ---------------------------------------------------------------------------


def _functional_baseline(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Restoration-quality baselines: variant comparison + a representative
    trajectory + a per-tier served bar chart for the representative task."""
    sub = campaign.by_experiment("functional_baseline")
    figs: list[str] = []
    if sub.empty:
        return figs

    figs.append(str(plots.variant_comparison_bar(
        sub[sub["variant"] == "scare"],
        out_dir / "served_per_grid.png",
        title="Priority-weighted served, scare baseline by grid",
    )))

    rep = campaign.representative_task("functional_baseline", "scare")
    if rep is not None:
        figs.append(str(plots.served_by_tier(
            rep.served, out_dir / "representative_served_by_tier.png",
            title=f"Served fraction by tier — task {rep.task_id} ({rep.grid})",
        )))
        figs.append(str(plots.restoration_trajectory(
            rep.timeseries, rep.events,
            out_dir / "representative_trajectory.png",
            title=f"Restoration trajectory — task {rep.task_id} ({rep.grid})",
            failure_t=rep.first_failure_time(),
        )))
    return figs


def _optimality_gap(campaign: CampaignData, out_dir: Path) -> list[str]:
    sub = campaign.by_experiment("optimality_gap")
    if sub.empty:
        return []
    sub = sub[sub["status"] == "ok"]
    return [
        str(plots.optimality_gap_scatter(sub, out_dir / "scatter.png")),
        str(plots.optimality_gap_box(sub, out_dir / "box.png")),
    ]


def _variant_comparison(campaign: CampaignData, out_dir: Path) -> list[str]:
    sub = campaign.by_experiment("variant_comparison")
    if sub.empty:
        return []
    sub = sub[sub["status"] == "ok"]
    return [
        str(plots.variant_comparison_bar(
            sub, out_dir / "served_by_variant.png",
            title="Priority-weighted served by variant",
        )),
        str(plots.diary_outcomes_bar(sub, out_dir / "diary_outcomes.png")),
        str(plots.claims_pass_rate(sub, out_dir / "claims_pass_rate.png")),
    ]


def _ablation(campaign: CampaignData, out_dir: Path) -> list[str]:
    sub = campaign.by_experiment("ablation")
    if sub.empty:
        return []
    sub = sub[sub["status"] == "ok"]
    return [
        str(plots.ablation_impact_bar(sub, out_dir / "ablation_impact.png")),
    ]


def _robustness(campaign: CampaignData, out_dir: Path) -> list[str]:
    figs: list[str] = []
    pl = campaign.by_experiment("robustness_packet_loss")
    if not pl.empty:
        figs.append(str(plots.robustness_curve(
            pl[pl["status"] == "ok"],
            out_dir / "packet_loss.png",
            sweep_param="comms_packet_loss_pct",
            x_label="packet loss (%)",
            title="Robustness — packet loss",
        )))
    lat = campaign.by_experiment("robustness_latency")
    if not lat.empty:
        figs.append(str(plots.robustness_curve(
            lat[lat["status"] == "ok"],
            out_dir / "latency_jitter.png",
            sweep_param="comms_latency_jitter_ms",
            x_label="latency jitter (ms)",
            title="Robustness — latency jitter",
        )))
    return figs


def _cascading(campaign: CampaignData, out_dir: Path) -> list[str]:
    sub = campaign.by_experiment("cascading")
    if sub.empty:
        return []
    return [
        str(plots.cascading_curve(
            sub[sub["status"] == "ok"], out_dir / "n_failures.png"
        )),
    ]


def _sweeps(campaign: CampaignData, out_dir: Path) -> list[str]:
    figs: list[str] = []
    for exp_name, param, label, title in (
        ("cooldown_sweep", "cooldown_s",
         "cooldown (s)", "Cooldown sweep — served + wallclock"),
        ("ttl_sweep", "ttl_hops",
         "FailureNotice TTL (hops)", "TTL sweep — served + wallclock"),
        ("holon_size_sweep", "holon_max_size",
         "max holon size", "Holon-size sweep — served + wallclock"),
    ):
        sub = campaign.by_experiment(exp_name)
        if sub.empty:
            continue
        figs.append(str(plots.sweep_curve_dual(
            sub[sub["status"] == "ok"],
            out_dir / f"{exp_name}.png",
            sweep_param=param,
            x_label=label,
            title=title,
        )))
    return figs


def _claims(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide claims roll-up across every variant + experiment."""
    df = campaign.summary
    if df.empty:
        return []
    return [str(plots.claims_pass_rate(
        df[df["status"] == "ok"], out_dir / "claims_overall.png"
    ))]


def _restoration(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide restoration view: pre-failure baseline vs
    post-restoration absolute MW + per-tier ratio.  Only emits figures
    when the campaign carried the new ``outcomes.restoration.*`` block
    (older campaigns silently fall back to an empty placeholder).
    """
    df = campaign.summary
    if df.empty:
        return []
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return []
    figs = [
        str(plots.restoration_vs_baseline_bar(
            ok, out_dir / "absolute_vs_baseline.png",
        )),
        str(plots.absolute_load_lost_bar(
            ok, out_dir / "absolute_load_lost.png",
        )),
        str(plots.restoration_by_tier_bar(
            ok, out_dir / "by_tier.png",
        )),
    ]
    return figs


# ---------------------------------------------------------------------------
# Markdown stitch
# ---------------------------------------------------------------------------


def _table_status(campaign: CampaignData) -> str:
    df = campaign.summary
    if df.empty:
        return "_(empty campaign)_"
    by_status = df["status"].value_counts().to_dict()
    parts = [f"- Total tasks: **{len(df)}**"]
    for k in ("ok", "error", "timeout", "missing"):
        if k in by_status:
            parts.append(
                f"  - {k}: {by_status[k]} ({100.0 * by_status[k] / len(df):.1f}%)"
            )
    return "\n".join(parts)


def _table_variant_means(campaign: CampaignData) -> str:
    df = campaign.summary
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns:
        return "_(no priority-weighted served column)_"
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return "_(no successful runs)_"
    rows = ok.groupby("variant")[metric].agg(["mean", "std", "count"])
    if rows.empty:
        return "_(no variant data)_"
    lines = ["| variant | n | mean served | std |", "|---|---|---|---|"]
    for variant, r in rows.iterrows():
        lines.append(
            f"| {variant} | {int(r['count'])} | {r['mean']:.4f} | {r['std']:.4f} |"
        )
    return "\n".join(lines)


def _table_claims(campaign: CampaignData) -> str:
    df = campaign.summary
    if df.empty:
        return ""
    cols = [c for c in df.columns if c.startswith("claims__") and c.endswith("__passed")]
    if not cols:
        return ""
    lines = ["", "## Claims compliance", "", "| claim | n | pass rate |", "|---|---|---|"]
    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        rate = float(s.astype(bool).sum()) / len(s)
        claim = col[len("claims__"):-len("__passed")]
        lines.append(f"| {claim} | {len(s)} | {100.0 * rate:.1f}% |")
    return "\n".join(lines)


def _todo_section(campaign: CampaignData) -> str:
    """Surface the TODO experiments listed in metadata so reviewers see
    what isn't yet measured."""
    md = campaign.metadata or {}
    cfg = md.get("campaign_config", {}) or {}
    exps = cfg.get("experiments", []) or []
    todos = [e for e in exps if e.get("name", "").endswith("_TODO") or not e.get("grids")]
    if not todos:
        return ""
    lines = ["", "## TODO — experiments not run", ""]
    for e in todos:
        lines.append(f"- **{e.get('name', '?')}** — {e.get('notes', '(no notes)')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def generate_report(campaign_dir: Path) -> Path:
    """Generate plots + REPORT.md for *campaign_dir*.  Returns the
    Markdown path."""
    campaign = load_campaign(campaign_dir)
    plots_root = campaign_dir / "plots"
    plots_root.mkdir(exist_ok=True)

    sections: list[tuple[str, list[str]]] = []
    for label, fn, sub in (
        ("Functional baseline", _functional_baseline, "functional_baseline"),
        ("Optimality gap", _optimality_gap, "optimality_gap"),
        ("Variant comparison", _variant_comparison, "variant_comparison"),
        ("Ablation impact", _ablation, "ablation"),
        ("Robustness", _robustness, "robustness"),
        ("Cascading", _cascading, "cascading"),
        ("Sensitivity sweeps", _sweeps, "sweeps"),
        ("Restoration vs baseline", _restoration, "restoration"),
        ("Claims (overall)", _claims, "claims"),
    ):
        out_dir = plots_root / sub
        try:
            figs = fn(campaign, out_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Section %r failed: %s — skipping", label, exc
            )
            figs = []
        if figs:
            sections.append((label, figs))

    md = _stitch(campaign, sections)
    report_path = campaign_dir / "REPORT.md"
    report_path.write_text(md)
    logger.info("Wrote %s (%d sections)", report_path, len(sections))
    return report_path


def _stitch(
    campaign: CampaignData, sections: list[tuple[str, list[str]]]
) -> str:
    parts = [f"# Evaluation report — {campaign.campaign_dir.name}", ""]
    parts.append("## Status")
    parts.append("")
    parts.append(_table_status(campaign))
    parts.append("")
    parts.append("## Variant means")
    parts.append("")
    parts.append(_table_variant_means(campaign))
    parts.append(_table_claims(campaign))

    for label, figs in sections:
        parts.append("")
        parts.append(f"## {label}")
        parts.append("")
        for fig in figs:
            stem = Path(fig)
            png = stem.with_suffix(".png")
            html = stem.with_suffix(".html")
            png_rel = png.relative_to(campaign.campaign_dir)
            parts.append(f"![{stem.stem}]({png_rel})")
            if html.exists():
                html_rel = html.relative_to(campaign.campaign_dir)
                parts.append(f"_[interactive ↗]({html_rel})_")
            parts.append("")

    parts.append(_todo_section(campaign))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    generate_report(args.campaign_dir.resolve())


if __name__ == "__main__":
    main()
