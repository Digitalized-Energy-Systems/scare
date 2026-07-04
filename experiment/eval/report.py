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
from experiment.eval.aliases import alias_experiment, alias_grid, alias_variant
from experiment.eval.compliance import compliant_mask, mean_ci95
from experiment.eval.loader import CampaignData, load_campaign
from experiment.eval.overview import write_overview

logger = logging.getLogger(__name__)


def _completed_sims(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose sim completed with valid metrics. ``claims_failed`` = a
    fatal claim failed, not a missing measurement — figures must include it
    like the tables do, else the population is variant-asymmetric (oracle ~0%
    claims_failed vs single_level ~78%) and weak baselines look inflated."""
    if "status" not in df.columns:
        return df
    return df[df["status"].isin(("ok", "claims_failed"))]


# Per-experiment dispatchers


def _functional_baseline(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Restoration-quality baselines: variant comparison + a representative
    trajectory + a per-tier served bar chart for the representative task."""
    sub = _completed_sims(campaign.by_experiment("functional_baseline"))
    figs: list[str] = []
    if sub.empty:
        return figs

    figs.append(
        str(
            plots.variant_comparison_bar(
                sub[sub["variant"] == "scare"],
                out_dir / "served_per_grid.png",
                title="Priority-weighted served, scare baseline by grid",
            )
        )
    )

    rep = campaign.representative_task("functional_baseline", "scare")
    if rep is not None:
        figs.append(
            str(
                plots.served_by_tier(
                    rep.served,
                    out_dir / "representative_served_by_tier.png",
                    title=f"Served fraction by tier — task {rep.task_id} ({rep.grid})",
                )
            )
        )
        figs.append(
            str(
                plots.restoration_trajectory(
                    rep.timeseries,
                    rep.events,
                    out_dir / "representative_trajectory.png",
                    title=f"Restoration trajectory — task {rep.task_id} ({rep.grid})",
                    failure_t=rep.first_failure_time(),
                )
            )
        )
        # Constraint-envelope overlay: whether voltage / pressure /
        # temperature ever left the operating band during recovery.
        figs.append(
            str(
                plots.constraint_envelope_trajectory(
                    rep.timeseries,
                    rep.events,
                    out_dir / "representative_constraint_envelope.png",
                    title=(f"Constraint envelopes — task {rep.task_id} ({rep.grid})"),
                    failure_t=rep.first_failure_time(),
                    solver_failures=rep.solver_failures(),
                )
            )
        )
    return figs


def _optimality_gap(campaign: CampaignData, out_dir: Path) -> list[str]:
    # Pool EVERY completed run, not just the dedicated ``optimality_gap``
    # experiment: the scatter/box pair scare vs oracle on task identity
    # (experiment, grid, seed, scenario) and drop unpaired rows, so pooling
    # all experiments that ran both variants integrates every grid family into
    # one figure instead of the single S1 optimality-gap sweep.
    sub = _completed_sims(campaign.summary)
    if sub.empty:
        return []
    return [
        str(plots.optimality_gap_scatter(sub, out_dir / "scatter.png")),
        str(plots.optimality_gap_box(sub, out_dir / "box.png")),
    ]


def _variant_comparison(campaign: CampaignData, out_dir: Path) -> list[str]:
    sub = _completed_sims(campaign.by_experiment("variant_comparison"))
    if sub.empty:
        return []
    return [
        str(
            plots.variant_comparison_bar(
                sub,
                out_dir / "served_by_variant.png",
                title="Priority-weighted served by variant",
            )
        ),
        # Per-sector breakdown of the headline PWSF (gas is excluded from the
        # aggregate above as it is in mass-flow units; here it shows per sector).
        str(
            plots.pwsf_by_sector_bar(
                sub,
                out_dir / "pwsf_by_sector.png",
            )
        ),
        # Regulate trigger mix per variant — which control layer fires.
        str(
            plots.regulates_by_reason_bar(
                sub,
                out_dir / "regulates_by_reason.png",
            )
        ),
        str(plots.diary_outcomes_bar(sub, out_dir / "diary_outcomes.png")),
        str(plots.claims_pass_rate(sub, out_dir / "claims_pass_rate.png")),
    ]


def _cp_influence(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Coupling-point optimization influence, pooled over EVERY completed
    task (all experiments) so each (grid, variant) cell has real sample
    mass — the basic-result view of how much the L3 cross-sector ADMM
    actually steers each grid."""
    done = _completed_sims(campaign.summary)
    if done.empty:
        return []
    return [
        str(
            plots.cp_influence_bar(
                done,
                out_dir / "cp_influence.png",
            )
        )
    ]


def _restoration_time(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide time-to-stabilise box plot.

    Pools every completed task across all experiments so the per-variant box
    has enough samples to be meaningful (the ``variant_comparison`` slice
    alone collapses to n=1 per variant in small campaigns).
    """
    done = _completed_sims(campaign.summary)
    if done.empty:
        return []
    return [
        str(
            plots.time_to_stabilise_box(
                done,
                out_dir / "time_to_stabilise.png",
            )
        )
    ]


def _ablation_experiments(campaign: CampaignData) -> list[str]:
    """Every experiment whose name marks it as an ablation matrix. The
    eval_full campaign splits the matrix by theme (``ablation_core``,
    ``ablation_voltage``, …) rather than a single ``ablation`` experiment."""
    return [e for e in campaign.experiments() if e == "ablation" or e.startswith("ablation_")]


def _ablation(
    campaign: CampaignData,
    plots_root: Path,
) -> list[tuple[str, list[str]]]:
    """One ablation-impact bar per ablation experiment, each in its own
    ``plots/<experiment>/`` folder. Compares the per-flag PWSF against the
    in-block ``default`` baseline — the actual ablation matrix, not the
    one-bar served-by-variant fallback.

    Scare rows only (the title claims "scare variant"; pooling variants
    diffed arms against a mixed population), and one figure per grid on
    multi-grid experiments so each arm is baselined against its own
    (experiment, grid) default — matching ``hpc.aggregate``'s methodology.
    """
    out: list[tuple[str, list[str]]] = []
    for exp_name in _ablation_experiments(campaign):
        sub = _completed_sims(campaign.by_experiment(exp_name))
        if "variant" in sub.columns:
            sub = sub[sub["variant"] == "scare"]
        if sub.empty:
            continue
        grids = (
            sorted(sub["grid"].dropna().unique()) if "grid" in sub.columns else []
        )
        figs: list[str] = []
        if len(grids) > 1:
            for grid in grids:
                figs.append(
                    str(
                        plots.ablation_impact_bar(
                            sub[sub["grid"] == grid],
                            plots_root / exp_name / f"ablation_impact_{grid}.png",
                            title=(
                                f"Ablation impact — {alias_grid(grid)} "
                                "(scare variant, compliant runs)"
                            ),
                        )
                    )
                )
        else:
            figs.append(
                str(
                    plots.ablation_impact_bar(
                        sub, plots_root / exp_name / "ablation_impact.png"
                    )
                )
            )
        out.append((f"Ablation impact — {alias_experiment(exp_name)}", figs))
    return out


def _robustness(campaign: CampaignData, out_dir: Path) -> list[str]:
    figs: list[str] = []
    pl = campaign.by_experiment("robustness_packet_loss")
    if not pl.empty:
        figs.append(
            str(
                plots.robustness_curve(
                    _completed_sims(pl),
                    out_dir / "packet_loss.png",
                    sweep_param="comms_packet_loss_pct",
                    x_label="packet loss (%)",
                    title="Robustness — packet loss",
                )
            )
        )
    lat = campaign.by_experiment("robustness_latency")
    if not lat.empty:
        figs.append(
            str(
                plots.robustness_curve(
                    _completed_sims(lat),
                    out_dir / "latency_jitter.png",
                    sweep_param="comms_latency_jitter_ms",
                    x_label="latency jitter (ms)",
                    title="Robustness — latency jitter",
                )
            )
        )
    return figs


def _cascading(campaign: CampaignData, out_dir: Path) -> list[str]:
    sub = campaign.by_experiment("cascading")
    if sub.empty:
        return []
    return [
        str(
            plots.cascading_curve(
                _completed_sims(sub), out_dir / "n_failures.png"
            )
        ),
    ]


def _sweeps(campaign: CampaignData, out_dir: Path) -> list[str]:
    figs: list[str] = []
    for exp_name, param, label, title in (
        (
            "cooldown_sweep",
            "cooldown_s",
            "cooldown (s)",
            "Cooldown sweep — served + wallclock",
        ),
        (
            "ttl_sweep",
            "ttl_hops",
            "FailureNotice TTL (hops)",
            "TTL sweep — served + wallclock",
        ),
        (
            "holon_size_sweep",
            "holon_max_size",
            "max holon size",
            "Holon-size sweep — served + wallclock",
        ),
    ):
        sub = campaign.by_experiment(exp_name)
        if sub.empty:
            continue
        figs.append(
            str(
                plots.sweep_curve_dual(
                    _completed_sims(sub),
                    out_dir / f"{exp_name}.png",
                    sweep_param=param,
                    x_label=label,
                    title=title,
                )
            )
        )
    return figs


def _extensions(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Extension campaign: paired A/B PWSF per (grid, seed) for the islanding
    MILP (clean vs microgrid) and the temporal physics extensions (off vs
    linepack+LTC), plus the per-task evidence trajectories (de-energised
    nodes / islanding events; linepack + LTC temperature over physical
    hours with the oracle's per-step linepack as reference)."""
    if not any(e.startswith("extension_") for e in campaign.experiments()):
        return []
    figs: list[str] = []

    isl = _completed_sims(campaign.by_experiment("extension_islanding"))
    if not isl.empty:
        figs.append(
            str(plots.extension_islanding_ab(isl, out_dir / "islanding_ab.png"))
        )
        micro = isl[
            (isl["variant"] == "scare")
            & isl["scenario"].str.contains("kind=microgrid", na=False)
        ]
        tasks = []
        for _, row in micro.sort_values("seed").iterrows():
            task = campaign.task(int(row["task_id"]))
            tasks.append((f"seed {task.seed}", task.timeseries))
        figs.append(
            str(
                plots.extension_islanding_events(
                    tasks, out_dir / "islanding_events.png"
                )
            )
        )

    tmp = _completed_sims(campaign.by_experiment("extension_temporal"))
    if not tmp.empty:
        figs.append(
            str(plots.extension_temporal_ab(tmp, out_dir / "temporal_ab.png"))
        )
        b_arm = tmp[tmp["scenario"].str.contains("linepack=True", na=False)]
        tasks = []
        for _, row in b_arm[b_arm["variant"] == "scare"].sort_values("seed").iterrows():
            task = campaign.task(int(row["task_id"]))
            tasks.append((f"seed {task.seed}", task.timeseries, task.scenario))
        oracle_series = []
        for _, row in b_arm[b_arm["variant"] == "oracle"].sort_values("seed").iterrows():
            task = campaign.task(int(row["task_id"]))
            series = (
                (task.result.get("outcomes") or {}).get("oracle_temporal") or {}
            ).get("series") or []
            if series:
                oracle_series.append((f"seed {task.seed}", series))
        figs.append(
            str(
                plots.extension_temporal_trajectories(
                    tasks,
                    out_dir / "temporal_trajectories.png",
                    oracle_series=oracle_series,
                )
            )
        )
    return figs


def _claims(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide claims roll-up across every variant + experiment."""
    df = campaign.summary
    if df.empty:
        return []
    return [
        str(
            plots.claims_pass_rate(
                _completed_sims(df), out_dir / "claims_overall.png"
            )
        )
    ]


def _solver_health(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide solver-health view: mean infeasibility / warning counts
    per task split by variant, so energy-flow LP regressions show up without
    a grep over run.log.
    """
    df = campaign.summary
    if df.empty:
        return []
    return [
        str(
            plots.solver_health_bar(
                df,
                out_dir / "solver_health.png",
            )
        )
    ]


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

    # Pick a representative OK scare task whose ``timeseries.csv`` carries the
    # richest recordings.  Walk ``functional_baseline``-first, preferring tasks
    # with slack columns, then coalition_balance, then the FB representative or
    # first OK task.  Each required prefix is a superset of the next, so the
    # newest match gives every subplot data.
    fb_first = pd.concat(
        [
            ok[ok["experiment"] == "functional_baseline"],
            ok[ok["experiment"] != "functional_baseline"],
        ]
    )
    rep = None
    for required in ("slack__", "coalition_balance__"):
        for tid in fb_first["task_id"].astype(int).tolist():
            candidate = campaign.task(int(tid))
            cols = list(candidate.timeseries.columns)
            if any(c.startswith(required) for c in cols):
                rep = candidate
                break
        if rep is not None:
            break
    if rep is None:
        rep = campaign.representative_task("functional_baseline", "scare")
    if rep is None:
        rep = campaign.task(int(ok["task_id"].iloc[0]))

    figs: list[str] = []
    failure_t = rep.first_failure_time()
    figs.append(
        str(
            plots.system_balance_trajectory(
                rep.timeseries,
                rep.events,
                out_dir / "system_balance.png",
                title=(f"System balance — task {rep.task_id} ({rep.grid})"),
                failure_t=failure_t,
            )
        )
    )
    figs.append(
        str(
            plots.coalition_balance_lines(
                rep.timeseries,
                out_dir / "coalition_balance.png",
                title=(
                    f"Coalition balances (Level-1) — task {rep.task_id} ({rep.grid})"
                ),
            )
        )
    )
    figs.append(
        str(
            plots.holon_balance_lines(
                rep.timeseries,
                out_dir / "holon_balance.png",
                title=(f"Holon balances (Level-2) — task {rep.task_id} ({rep.grid})"),
            )
        )
    )
    figs.append(
        str(
            plots.regulation_per_child_lines(
                rep.trajectories,
                out_dir / "regulation_per_child.png",
                title=(f"Per-child regulation — task {rep.task_id} ({rep.grid})"),
            )
        )
    )
    figs.append(
        str(
            plots.slack_trajectory(
                rep.timeseries,
                out_dir / "slack_trajectory.png",
                title=(f"External-grid slack — task {rep.task_id} ({rep.grid})"),
                failure_t=failure_t,
                slack_meta=rep.slack_meta,
            )
        )
    )
    figs.append(
        str(
            plots.gas_slack_pressure_trajectory(
                rep.timeseries,
                out_dir / "gas_slack_pressure_trajectory.png",
                title=(
                    f"Gas ext-grid regulator — task {rep.task_id} ({rep.grid})"
                ),
                failure_t=failure_t,
            )
        )
    )
    return figs


def _constraints(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide constraint-handling view: the per-sector violation
    integral (``int max(0, util-1) dt``) split by variant — did the constraint
    layer keep the network inside its envelope — plus the per-variable-type
    violation count (voltage / pressure / line load / slack / temperature) that
    accompanies the compliance gate.  Per-task envelope trajectories live
    alongside each representative task; ``overview_constraints.html`` collates
    them.
    """
    df = campaign.summary
    if df.empty:
        return []
    return [
        str(
            plots.constraint_violation_integral_bar(
                df,
                out_dir / "violation_integral.png",
            )
        ),
        # Companion count to the integral: how many bounds each variant
        # breaches at end-of-sim, split by variable type (voltage / pressure /
        # line load / slack / temperature).
        str(
            plots.constraint_violations_by_variable_bar(
                df,
                out_dir / "violations_by_variable.png",
            )
        ),
    ]


def _per_experiment_trajectories(
    campaign: CampaignData,
    plots_root: Path,
) -> list[tuple[str, list[str]]]:
    """One trajectory + constraint-envelope per (experiment, variant).

    Emits a trajectory + envelope for the representative OK task in each
    (experiment x variant) cell.  Skips ``functional_baseline``, covered by
    its own dispatcher.
    """
    df = campaign.summary
    if df.empty or "experiment" not in df.columns or "variant" not in df.columns:
        return []
    # ok-only on purpose: this picks one representative per-task artefact and
    # ``CampaignData.representative_task`` only returns OK tasks.
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return []

    out: list[tuple[str, list[str]]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for exp_name in sorted(ok["experiment"].dropna().unique()):
        if exp_name == "functional_baseline" or not exp_name:
            continue
        for variant in sorted(
            ok[ok["experiment"] == exp_name]["variant"].dropna().unique()
        ):
            pair = (str(exp_name), str(variant))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            rep = campaign.representative_task(str(exp_name), str(variant))
            if rep is None:
                continue
            out_dir = plots_root / "trajectories" / str(exp_name) / str(variant)
            label = (
                f"Trajectory — {alias_experiment(exp_name)} / {alias_variant(variant)}"
            )
            figs: list[str] = []
            try:
                figs.append(
                    str(
                        plots.restoration_trajectory(
                            rep.timeseries,
                            rep.events,
                            out_dir / "trajectory.png",
                            title=(
                                f"Restoration trajectory — task {rep.task_id} "
                                f"({rep.grid}, {alias_variant(variant)})"
                            ),
                            failure_t=rep.first_failure_time(),
                        )
                    )
                )
                figs.append(
                    str(
                        plots.constraint_envelope_trajectory(
                            rep.timeseries,
                            rep.events,
                            out_dir / "constraint_envelope.png",
                            title=(
                                f"Constraint envelopes — task {rep.task_id} "
                                f"({rep.grid}, {alias_variant(variant)})"
                            ),
                            failure_t=rep.first_failure_time(),
                            solver_failures=rep.solver_failures(),
                        )
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Trajectory for (%s, %s) failed: %s — skipping",
                    exp_name,
                    variant,
                    exc,
                )
            if figs:
                out.append((label, figs))
    return out


def _per_task_overviews(
    campaign: CampaignData,
    out_dir: Path,
) -> list[tuple[str, list[str]]]:
    """Render :func:`plots.system_state_overview` for every completed task
    (ok + claims_failed — the failed-claim tasks are the ones worth
    inspecting).

    One stacked-subplots figure per task at
    ``plots/per_task_overview/<task_id>.html`` — slack injection,
    control-variable envelopes, line-loading aggregates, and per-tier
    demand fulfilment (fraction + MW) on shared simulation-time axes.
    Grouped per (experiment, variant) so the report can hyperlink
    section-by-section without a thousand inline images.
    """
    df = campaign.summary
    if df.empty or "task_id" not in df.columns:
        return []
    done = _completed_sims(df)
    if done.empty:
        return []

    grouped: dict[tuple[str, str], list[str]] = {}
    for _, row in done.iterrows():
        try:
            tid = int(row["task_id"])
        except (TypeError, ValueError):
            continue
        task = campaign.task(tid)
        try:
            fig_path = str(
                plots.system_state_overview(
                    task.timeseries,
                    task.events,
                    out_dir / f"{tid:06d}.png",
                    title=(
                        f"System-state overview — task {tid} "
                        f"({task.grid}, {alias_variant(task.variant)})"
                    ),
                    failure_t=task.first_failure_time(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "system_state_overview task %d failed: %s — skipping",
                tid,
                exc,
            )
            continue
        key = (str(row.get("experiment", "")), str(row.get("variant", "")))
        grouped.setdefault(key, []).append(fig_path)

    sections: list[tuple[str, list[str]]] = []
    for (exp_name, variant), figs in sorted(grouped.items()):
        label = (
            f"Per-task overview — {alias_experiment(exp_name)} / "
            f"{alias_variant(variant)}"
        )
        sections.append((label, figs))
    return sections


def _restoration(campaign: CampaignData, out_dir: Path) -> list[str]:
    """Campaign-wide restoration view: pre-failure baseline vs post-restoration
    absolute MW + per-tier ratio.  Emits figures only when the campaign carried
    the ``outcomes.restoration.*`` block (else an empty placeholder).
    """
    df = campaign.summary
    if df.empty:
        return []
    done = _completed_sims(df)
    if done.empty:
        return []
    figs = [
        str(
            plots.restoration_vs_baseline_bar(
                done,
                out_dir / "absolute_vs_baseline.png",
            )
        ),
        str(
            plots.restoration_ratio_by_variant_bar(
                done,
                out_dir / "ratio_by_variant.png",
            )
        ),
        str(
            plots.absolute_load_lost_bar(
                done,
                out_dir / "absolute_load_lost.png",
            )
        ),
        str(
            plots.restoration_by_tier_bar(
                done,
                out_dir / "by_tier.png",
            )
        ),
        # Per-sector mirror of the per-tier ratio bar.
        str(
            plots.restoration_by_sector_bar(
                done,
                out_dir / "by_sector.png",
            )
        ),
        # Split per-tier loss into priority-blind (physical disconnect) vs
        # priority-aware (agent-shed); the tier-waterfall claim covers the latter.
        str(
            plots.restoration_loss_split_by_tier_bar(
                done,
                out_dir / "loss_split_by_tier.png",
            )
        ),
        str(
            plots.agent_only_ratio_by_tier_bar(
                done,
                out_dir / "agent_only_ratio_by_tier.png",
            )
        ),
    ]
    return figs


def _missing_experiment_sections(
    campaign: CampaignData,
    plots_root: Path,
) -> list[tuple[str, list[str]]]:
    """Per-experiment served-by-variant bars for every experiment lacking a
    dedicated dispatcher above, so experiments with data in summary.csv still
    get a figure.
    """
    df = campaign.summary
    if df.empty or "experiment" not in df.columns:
        return []
    handled = {
        "functional_baseline",
        "optimality_gap",
        "variant_comparison",
        "robustness_packet_loss",
        "robustness_latency",
        "cascading",
        "cooldown_sweep",
        "ttl_sweep",
        "holon_size_sweep",
        "extension_islanding",
        "extension_temporal",
    }
    ablation_exps = set(_ablation_experiments(campaign))
    done = _completed_sims(df)
    if done.empty:
        return []
    out: list[tuple[str, list[str]]] = []
    for exp_name in sorted(done["experiment"].dropna().unique()):
        if exp_name in handled or exp_name in ablation_exps or not exp_name:
            continue
        sub = done[done["experiment"] == exp_name]
        if sub.empty:
            continue
        out_dir = plots_root / exp_name
        figs = [
            str(
                plots.variant_comparison_bar(
                    sub,
                    out_dir / "served_by_variant.png",
                    title=f"PWSF by variant — {alias_experiment(exp_name)}",
                )
            )
        ]
        out.append((f"{alias_experiment(exp_name)} (auto)", figs))
    return out


# Markdown stitch


def _table_status(campaign: CampaignData) -> str:
    df = campaign.summary
    if df.empty:
        return "_(empty campaign)_"
    by_status = df["status"].value_counts().to_dict()
    parts = [f"- Total tasks: **{len(df)}**"]
    # Include claims_failed (a large cohort — runs that completed but failed a
    # fatal claim) and killed; omitting them understated the total accounted-for.
    for k in ("ok", "claims_failed", "error", "timeout", "killed", "missing"):
        if k in by_status:
            parts.append(
                f"  - {k}: {by_status[k]} ({100.0 * by_status[k] / len(df):.1f}%)"
            )
    return "\n".join(parts)


def _table_variant_means(campaign: CampaignData) -> str:
    """Per-variant PWSF on the COMPLIANT subset with 95% CI, matching the
    campaign's primary methodology (compliance gate + CI; see
    ``hpc.aggregate``). The earlier version reported an unpaired, un-gated,
    no-CI mean over ``status=='ok'`` only, which both dropped the claims_failed
    cohort asymmetrically (oracle is never claims_failed) and presented
    non-comparable per-variant subsets as if comparable. This table is still
    UNPAIRED and pooled across grids — for the apples-to-apples comparison see
    the paired ``Variant vs oracle`` table in ``summary.md``. PWSF is
    electricity+heat only (gas is in mass-flow units, reported per-sector).
    """
    df = campaign.summary
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns:
        return "_(no priority-weighted served column)_"
    # PWSF is valid for any completed run (ok or claims_failed); error/timeout
    # have no defined PWSF.
    done = df[df["status"].isin(("ok", "claims_failed"))]
    done = done[done[metric].notna()]
    if done.empty:
        return "_(no successful runs)_"
    lines = [
        "_Compliant-subset PWSF (slack + grid feasibility), unpaired, pooled "
        "across grids AND experiments (each variant is averaged over whatever "
        "experiment mix it ran in); electricity+heat only. Paired comparison: "
        "see `summary.md`._",
        "",
        "| variant | n_compliant/n_total | compliance | mean PWSF | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for variant, g in done.groupby("variant"):
        n_total = len(g)
        comp = g[compliant_mask(g)]
        n_comp = len(comp)
        rate = 100.0 * n_comp / n_total if n_total else float("nan")
        if n_comp == 0:
            mean_str, ci_str = "—", "—"
        else:
            mean, ci = mean_ci95(comp[metric])
            mean_str = f"{mean:.4f}"
            ci_str = "—" if pd.isna(ci) else f"±{ci:.4f}"
        lines.append(
            f"| {alias_variant(str(variant))} | {n_comp}/{n_total} | "
            f"{rate:.0f}% | {mean_str} | {ci_str} |"
        )
    return "\n".join(lines)


def _table_claims(campaign: CampaignData) -> str:
    df = campaign.summary
    if df.empty:
        return ""
    cols = [
        c for c in df.columns if c.startswith("claims__") and c.endswith("__passed")
    ]
    if not cols:
        return ""
    variants = sorted(df["variant"].dropna().unique()) if "variant" in df else []
    if not variants:
        return ""
    # Per-variant pass rate (over GRADED rows only): pooling all variants
    # conflated the system-under-test with the weak baselines, and the oracle is
    # exempt from several claims (shown as "—", not counted as passing).
    header = "| claim | " + " | ".join(alias_variant(v) for v in variants) + " |"
    sep = "|---|" + "|".join(["---"] * len(variants)) + "|"
    lines = ["", "## Claims compliance (pass rate by variant, graded rows)", "", header, sep]
    for col in cols:
        claim = col[len("claims__") : -len("__passed")]
        cells = []
        for v in variants:
            s = df.loc[df["variant"] == v, col].dropna()
            if s.empty:
                cells.append("—")
            else:
                rate = 100.0 * float(s.astype(bool).sum()) / len(s)
                cells.append(f"{rate:.0f}% (n={len(s)})")
        lines.append(f"| {claim} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _todo_section(campaign: CampaignData) -> str:
    """Surface the TODO experiments listed in metadata (not yet measured)."""
    md = campaign.metadata or {}
    cfg = md.get("campaign_config", {}) or {}
    exps = cfg.get("experiments", []) or []
    todos = [
        e for e in exps if e.get("name", "").endswith("_TODO") or not e.get("grids")
    ]
    if not todos:
        return ""
    lines = ["", "## TODO — experiments not run", ""]
    for e in todos:
        lines.append(f"- **{e.get('name', '?')}** — {e.get('notes', '(no notes)')}")
    return "\n".join(lines)


# Top-level entry point


def generate_report(
    campaign_dir: Path,
    *,
    per_task_overviews: bool = False,
) -> Path:
    """Generate plots + REPORT.md for *campaign_dir*.  Returns the Markdown
    path.

    ``per_task_overviews`` (default ``False``) renders the
    one-figure-per-OK-task ``system_state_overview`` panels — hundreds of files
    on a large campaign.  Representative-task plots (one per experiment-variant)
    are always rendered regardless.
    """
    campaign = load_campaign(campaign_dir)
    plots_root = campaign_dir / "plots"
    plots_root.mkdir(exist_ok=True)

    sections: list[tuple[str, list[str]]] = []
    for label, fn, sub in (
        ("Functional baseline", _functional_baseline, "functional_baseline"),
        ("Optimality gap", _optimality_gap, "optimality_gap"),
        ("Variant comparison", _variant_comparison, "variant_comparison"),
        ("Coupling-point influence", _cp_influence, "cp_influence"),
        ("Time to stabilise", _restoration_time, "restoration_time"),
        ("Robustness", _robustness, "robustness"),
        ("Cascading", _cascading, "cascading"),
        ("Sensitivity sweeps", _sweeps, "sweeps"),
        ("Restoration vs baseline", _restoration, "restoration"),
        (
            "Extensions — islanding + temporal physics (paired A/B)",
            _extensions,
            "extensions",
        ),
        ("Claims (overall)", _claims, "claims"),
        ("Solver health", _solver_health, "solver_health"),
        ("Constraints", _constraints, "constraints"),
        ("Validity (multi-level balances)", _validity, "validity"),
    ):
        out_dir = plots_root / sub
        try:
            figs = fn(campaign, out_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Section %r failed: %s — skipping", label, exc)
            figs = []
        if figs:
            sections.append((label, figs))

    # Ablation matrices — one impact bar per themed ablation experiment.
    try:
        for label, figs in _ablation(campaign, plots_root):
            if figs:
                sections.append((label, figs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ablation section failed: %s — skipping", exc)

    # Auto-dispatched per-experiment sections for experiments with data in
    # summary.csv but no dedicated curve.
    try:
        for label, figs in _missing_experiment_sections(campaign, plots_root):
            if figs:
                sections.append((label, figs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto-dispatch failed: %s — skipping", exc)

    # Representative trajectory + constraint envelope per (experiment,
    # variant) for every experiment except functional_baseline (own slot).
    try:
        for label, figs in _per_experiment_trajectories(campaign, plots_root):
            if figs:
                sections.append((label, figs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Per-experiment trajectories failed: %s — skipping", exc)

    # One system-state overview HTML per OK task (slack + control vars + line
    # loading + per-tier fulfilment).  Opt-in via ``--per-task-overviews``.
    if per_task_overviews:
        try:
            for label, figs in _per_task_overviews(
                campaign,
                plots_root / "per_task_overview",
            ):
                if figs:
                    sections.append((label, figs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Per-task overviews failed: %s — skipping", exc)

    md = _stitch(campaign, sections)
    report_path = campaign_dir / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    logger.info("Wrote %s (%d sections)", report_path, len(sections))

    # Multi-plot HTML overviews — non-fatal; per-figure HTML/PDF is on disk.
    try:
        overview_path = write_overview(campaign)
        logger.info("Wrote %s", overview_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Overview generation failed: %s — skipping", exc)

    return report_path


def _stitch(campaign: CampaignData, sections: list[tuple[str, list[str]]]) -> str:
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--campaign-dir", required=True, type=Path)
    p.add_argument(
        "--per-task-overviews",
        action="store_true",
        help=(
            "Also render one system_state_overview figure per OK task "
            "(skipped by default — hundreds of files on a large campaign)."
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
    args = _parse_args()
    generate_report(
        args.campaign_dir.resolve(),
        per_task_overviews=args.per_task_overviews,
    )


if __name__ == "__main__":
    main()
