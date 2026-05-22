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
from experiment.eval.overview import write_overview

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
        # Constraint-envelope overlay for the same task — directly
        # surfaces whether voltage / pressure / temperature ever left
        # the operating band during the recovery.
        figs.append(str(plots.constraint_envelope_trajectory(
            rep.timeseries, rep.events,
            out_dir / "representative_constraint_envelope.png",
            title=(
                f"Constraint envelopes — task {rep.task_id} ({rep.grid})"
            ),
            failure_t=rep.first_failure_time(),
            solver_failures=rep.solver_failures(),
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
        # "How fast does each variant settle?" — restoration-time view
        # that mirrors the served-fraction headline metric.
        str(plots.time_to_stabilise_box(
            sub, out_dir / "time_to_stabilise.png",
        )),
        # "Which control layer actually fires under each variant?" —
        # exposes the regulate trigger mix so ablations stand out.
        str(plots.regulates_by_reason_bar(
            sub, out_dir / "regulates_by_reason.png",
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


def _solver_health(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide solver-health view.  Surfaces mean infeasibility /
    warning counts per task split by variant so regressions in the
    energy-flow LP (e.g. the failure-mode the run-log audit chased
    down) show up at a glance instead of needing a grep over run.log.
    """
    df = campaign.summary
    if df.empty:
        return []
    return [str(plots.solver_health_bar(
        df, out_dir / "solver_health.png",
    ))]


def _validity(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Validity plots — verify the multi-level controller behaved as
    the architecture chapter claims.

    Four traces, all drawn for the ``functional_baseline`` scare
    representative task:

    - **System balance** — per-sector ``Σ regulation`` (does the global
      controller settle after the failure?).
    - **Coalition balances** — one line per Level-1 community per
      sector (do groups individually converge?).
    - **Holon balances** — one line per Level-2 chunk per sector (does
      the holonic ADMM smooth the per-coalition signal?).
    - **Per-child regulation** — every active child agent's factor
      trace (are devices actually being modulated, or sitting flat?).
    """
    df = campaign.summary
    if df.empty:
        return []
    ok = df[(df["status"] == "ok") & (df.get("variant") == "scare")]
    if ok.empty:
        return []

    # Validity plots are most informative when the per-coalition /
    # per-holon balance recordings are present in ``timeseries.csv``
    # (introduced by the validity-plot landing).  Walk OK scare tasks
    # in ``functional_baseline``-first order and pick the first one
    # whose timeseries actually carries the new columns, falling back
    # to ``functional_baseline``'s representative if no task has them
    # yet (so the system_balance subplot still renders on legacy data).
    fb_first = pd.concat([
        ok[ok["experiment"] == "functional_baseline"],
        ok[ok["experiment"] != "functional_baseline"],
    ])
    rep = None
    for tid in fb_first["task_id"].astype(int).tolist():
        candidate = campaign.task(int(tid))
        cols = list(candidate.timeseries.columns)
        if any(c.startswith("coalition_balance__") for c in cols):
            rep = candidate
            break
    if rep is None:
        rep = campaign.representative_task("functional_baseline", "scare")
    if rep is None:
        rep = campaign.task(int(ok["task_id"].iloc[0]))

    figs: list[str] = []
    failure_t = rep.first_failure_time()
    figs.append(str(plots.system_balance_trajectory(
        rep.timeseries, rep.events,
        out_dir / "system_balance.png",
        title=(
            f"System balance — task {rep.task_id} ({rep.grid})"
        ),
        failure_t=failure_t,
    )))
    figs.append(str(plots.coalition_balance_lines(
        rep.timeseries,
        out_dir / "coalition_balance.png",
        title=(
            f"Coalition balances (Level-1) — task {rep.task_id} ({rep.grid})"
        ),
    )))
    figs.append(str(plots.holon_balance_lines(
        rep.timeseries,
        out_dir / "holon_balance.png",
        title=(
            f"Holon balances (Level-2) — task {rep.task_id} ({rep.grid})"
        ),
    )))
    figs.append(str(plots.regulation_per_child_lines(
        rep.trajectories,
        out_dir / "regulation_per_child.png",
        title=(
            f"Per-child regulation — task {rep.task_id} ({rep.grid})"
        ),
    )))
    return figs


def _constraints(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide constraint-handling view.  Surfaces the per-sector
    violation integral (``∫ max(0, util-1) dt``) split by variant so
    "did the constraint layer keep the network inside its envelope"
    is answered in one bar.  The constraint envelope trajectories
    (per-task voltage / pressure / temperature with shaded bands)
    live next to each ``functional_baseline`` representative task and
    each per-experiment trajectory pair; the dedicated
    ``overview_constraints.html`` page collates them all in one view.
    """
    df = campaign.summary
    if df.empty:
        return []
    return [str(plots.constraint_violation_integral_bar(
        df, out_dir / "violation_integral.png",
    ))]


def _per_experiment_trajectories(
    campaign: CampaignData, plots_root: Path,
) -> list[tuple[str, list[str]]]:
    """One trajectory + constraint-envelope per (experiment, variant).

    Previously only ``functional_baseline`` got a representative task
    drawn — every other experiment had the same per-task artefacts but
    no figure.  Iterate over the (experiment × variant) cells of the
    summary and emit a trajectory + envelope for the lowest-id OK task
    in each.  Skips ``functional_baseline`` since the dedicated
    dispatcher already covers it.
    """
    df = campaign.summary
    if df.empty or "experiment" not in df.columns or "variant" not in df.columns:
        return []
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return []

    out: list[tuple[str, list[str]]] = []
    from experiment.eval.aliases import alias_experiment, alias_variant
    seen_pairs: set[tuple[str, str]] = set()
    for exp_name in sorted(ok["experiment"].dropna().unique()):
        if exp_name == "functional_baseline" or not exp_name:
            continue
        for variant in sorted(ok[ok["experiment"] == exp_name]["variant"].dropna().unique()):
            pair = (str(exp_name), str(variant))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            rep = campaign.representative_task(str(exp_name), str(variant))
            if rep is None:
                continue
            out_dir = plots_root / "trajectories" / str(exp_name) / str(variant)
            label = f"Trajectory — {alias_experiment(exp_name)} / {alias_variant(variant)}"
            figs: list[str] = []
            try:
                figs.append(str(plots.restoration_trajectory(
                    rep.timeseries, rep.events,
                    out_dir / "trajectory.png",
                    title=(
                        f"Restoration trajectory — task {rep.task_id} "
                        f"({rep.grid}, {alias_variant(variant)})"
                    ),
                    failure_t=rep.first_failure_time(),
                )))
                figs.append(str(plots.constraint_envelope_trajectory(
                    rep.timeseries, rep.events,
                    out_dir / "constraint_envelope.png",
                    title=(
                        f"Constraint envelopes — task {rep.task_id} "
                        f"({rep.grid}, {alias_variant(variant)})"
                    ),
                    failure_t=rep.first_failure_time(),
                    solver_failures=rep.solver_failures(),
                )))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Trajectory for (%s, %s) failed: %s — skipping",
                    exp_name, variant, exc,
                )
            if figs:
                out.append((label, figs))
    return out


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
        str(plots.restoration_ratio_by_variant_bar(
            ok, out_dir / "ratio_by_variant.png",
        )),
        str(plots.absolute_load_lost_bar(
            ok, out_dir / "absolute_load_lost.png",
        )),
        str(plots.restoration_by_tier_bar(
            ok, out_dir / "by_tier.png",
        )),
        # Per-sector mirror of the per-tier ratio bar — uses the
        # outcomes__restoration__by_sector__<sec>__ratio columns the
        # aggregator already flattens.
        str(plots.restoration_by_sector_bar(
            ok, out_dir / "by_sector.png",
        )),
        # Split per-tier loss into priority-blind (physical disconnect)
        # vs priority-aware (agent-shed) — the chapter's tier waterfall
        # claim applies only to the latter.
        str(plots.restoration_loss_split_by_tier_bar(
            ok, out_dir / "loss_split_by_tier.png",
        )),
        str(plots.agent_only_ratio_by_tier_bar(
            ok, out_dir / "agent_only_ratio_by_tier.png",
        )),
    ]
    return figs


def _missing_experiment_sections(
    campaign: CampaignData, plots_root: Path,
) -> list[tuple[str, list[str]]]:
    """Per-experiment served-by-variant bars for every experiment that
    doesn't have a dedicated dispatcher above.  Closes the gap where
    cp_flexibility / cp_size_sweep / cold_day_stress /
    concentrated_imbalance / generator_failure / scaling had data in
    summary.csv but no figure in the report.
    """
    df = campaign.summary
    if df.empty or "experiment" not in df.columns:
        return []
    handled = {
        "functional_baseline", "optimality_gap", "variant_comparison",
        "ablation", "robustness_packet_loss", "robustness_latency",
        "cascading", "cooldown_sweep", "ttl_sweep", "holon_size_sweep",
    }
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return []
    out: list[tuple[str, list[str]]] = []
    for exp_name in sorted(ok["experiment"].dropna().unique()):
        if exp_name in handled or not exp_name:
            continue
        sub = ok[ok["experiment"] == exp_name]
        if sub.empty:
            continue
        from experiment.eval.aliases import alias_experiment
        out_dir = plots_root / exp_name
        figs = [
            str(plots.variant_comparison_bar(
                sub,
                out_dir / "served_by_variant.png",
                title=f"PWSF by variant — {alias_experiment(exp_name)}",
            ))
        ]
        out.append((f"{alias_experiment(exp_name)} (auto)", figs))
    return out


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
    from experiment.eval.aliases import alias_variant

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
            f"| {alias_variant(str(variant))} | {int(r['count'])} | "
            f"{r['mean']:.4f} | {r['std']:.4f} |"
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
        ("Solver health", _solver_health, "solver_health"),
        ("Constraints", _constraints, "constraints"),
        ("Validity (multi-level balances)", _validity, "validity"),
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

    # Auto-dispatched per-experiment sections — closes the gap where
    # experiments like cp_flexibility / cp_size_sweep / cold_day_stress
    # /concentrated_imbalance / generator_failure / scaling have data
    # in summary.csv but no dedicated curve.
    try:
        for label, figs in _missing_experiment_sections(campaign, plots_root):
            if figs:
                sections.append((label, figs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto-dispatch failed: %s — skipping", exc)

    # Representative trajectory + constraint envelope per (experiment,
    # variant).  ``_functional_baseline`` already covers its own slot;
    # this dispatcher handles every other experiment so the per-task
    # artefacts that were collected stop going to waste.
    try:
        for label, figs in _per_experiment_trajectories(campaign, plots_root):
            if figs:
                sections.append((label, figs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Per-experiment trajectories failed: %s — skipping", exc)

    md = _stitch(campaign, sections)
    report_path = campaign_dir / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    logger.info("Wrote %s (%d sections)", report_path, len(sections))

    # Generate the multi-plot HTML overviews — non-fatal if it fails,
    # the per-figure HTML/PDF is still on disk for inclusion.
    try:
        overview_path = write_overview(campaign)
        logger.info("Wrote %s", overview_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Overview generation failed: %s — skipping", exc)

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
