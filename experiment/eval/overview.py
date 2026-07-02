"""Multi-plot HTML overview pages for quick result browsing.

Stitches the per-section interactive figures the report writes into
side-by-side overview HTMLs so reviewers can scan the headline result
of a campaign without opening one PDF at a time.

Two flavours of overview are produced under ``<campaign>/plots/``:

- ``overview.html``                 — campaign-wide headlines (status
  table + the key restoration / solver / variant plots).
- ``overview_<experiment>.html``   — every plot the report generated
  for that experiment, in one page.

Each page reuses the already-rendered ``.html`` files written by the
plot primitives in :mod:`experiment.eval.plots` (we slice out their
``<div>`` + ``<script>`` blocks so the figures stay fully
interactive).  Plotly is loaded once from CDN per page.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from experiment.eval.aliases import alias_variant
from experiment.eval.compliance import compliant_mask, mean_ci95
from experiment.eval.loader import CampaignData

logger = logging.getLogger(__name__)


_PLOTLY_CDN = (
    '<script src="https://cdn.plot.ly/plotly-3.5.0.min.js" charset="utf-8"></script>'
)

# Shared fonts with the figures plus a responsive grid that drops to one
# column under ~640px.
_STYLE = """
<style>
  :root {
    --fg:    #1A1A1A;
    --muted: #666666;
    --bg:    #FFFFFF;
    --grid:  #ECECEC;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 1.5rem 2rem 4rem;
    font-family: Inter, -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--fg);
    background: var(--bg);
    font-size: 14px;
    line-height: 1.45;
  }
  h1 {
    font-family: Charter, Georgia, "Times New Roman", serif;
    font-size: 1.6rem;
    margin: 0 0 .4rem;
  }
  h2 {
    font-family: Charter, Georgia, "Times New Roman", serif;
    font-size: 1.15rem;
    margin: 2rem 0 .8rem;
    padding-bottom: .25rem;
    border-bottom: 1px solid var(--grid);
  }
  p.lead { color: var(--muted); margin: 0 0 1.5rem; }
  .meta { color: var(--muted); font-size: .85rem; margin-bottom: 1rem; }
  .meta code { background: #F5F5F5; padding: 0 .3em; border-radius: 3px; }
  .status-table { border-collapse: collapse; margin: .5rem 0 1rem; }
  .status-table th, .status-table td {
    text-align: left;
    padding: .25rem .8rem .25rem 0;
    font-variant-numeric: tabular-nums;
  }
  .status-table th { color: var(--muted); font-weight: 500; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 1.25rem;
    margin: .5rem 0;
  }
  .plot {
    border: 1px solid var(--grid);
    border-radius: 6px;
    padding: .25rem;
    background: var(--bg);
    overflow: auto;
  }
  /* Plotly inserts its own width attr; let the card decide instead. */
  .plot .plotly-graph-div { width: 100% !important; }
  nav.toc { margin: 1rem 0 2rem; }
  nav.toc a {
    display: inline-block;
    margin-right: 1rem;
    margin-bottom: .25rem;
    color: #1F4E96;
    text-decoration: none;
  }
  nav.toc a:hover { text-decoration: underline; }
  footer { color: var(--muted); margin-top: 3rem; font-size: .8rem; }
</style>
"""


# Plotly's saved-HTML payload is the `<div class="plotly-graph-div">` plus the
# `<script>…Plotly.newPlot…</script>` that follows; non-greedy to `</script>`.
_PLOT_BLOCK_RE = re.compile(
    r'(<div id="[0-9a-f-]+" class="plotly-graph-div"[\s\S]*?</script>)',
    re.IGNORECASE,
)


def _extract_plot_block(html_path: Path) -> str | None:
    """Return the embeddable ``<div>+<script>`` for *html_path* or None
    when the file is missing or doesn't look like a Plotly figure.
    """
    if not html_path.exists():
        return None
    try:
        txt = html_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("Could not read %s: %s", html_path, exc)
        return None
    m = _PLOT_BLOCK_RE.search(txt)
    return m.group(1) if m else None


def _status_table(campaign: CampaignData) -> str:
    df = campaign.summary
    if df.empty:
        return "<p><em>(empty campaign)</em></p>"
    by_status = df["status"].value_counts().to_dict()
    rows = [f"<tr><td>Total</td><td><b>{len(df)}</b></td></tr>"]
    for key in ("ok", "claims_failed", "error", "timeout", "killed", "missing"):
        if key in by_status:
            pct = 100.0 * by_status[key] / max(1, len(df))
            rows.append(
                f"<tr><td>{key}</td>"
                f"<td>{by_status[key]} <span style='color:#888'>"
                f"({pct:.1f}%)</span></td></tr>"
            )
    extras: list[str] = []
    for col, label in (
        ("solver_infeasibilities", "solver infeasibilities (total)"),
        ("solver_warnings", "solver warnings (total)"),
        ("duration_s", "mean duration"),
    ):
        if col in df.columns and df[col].notna().any():
            if col == "duration_s":
                val = f"{df[col].mean():.1f} s"
            else:
                val = f"{int(df[col].fillna(0).sum())}"
            extras.append(f"<tr><td>{label}</td><td>{val}</td></tr>")
    return "<table class='status-table'>" + "".join(rows + extras) + "</table>"


def _variant_means_table(campaign: CampaignData) -> str:
    """Compliant-subset PWSF over completed sims (ok + claims_failed) with a
    95% CI — the same gate + population as the REPORT.md "Variant means"
    table, so the landing page cannot reverse the report's variant ranking."""
    df = campaign.summary
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns or "variant" not in df.columns:
        return ""
    done = df
    if "status" in done.columns:
        done = done[done["status"].isin(("ok", "claims_failed"))]
    done = done[done[metric].notna()]
    if done.empty:
        return ""
    rows: list[tuple[str, int, int, float, float]] = []
    for variant, g in done.groupby("variant"):
        n_total = len(g)
        comp = g[compliant_mask(g)]
        if len(comp):
            mean, ci = mean_ci95(comp[metric])
        else:
            mean, ci = float("nan"), float("nan")
        rows.append((str(variant), len(comp), n_total, mean, ci))
    rows.sort(key=lambda r: (pd.isna(r[3]), -(r[3] if pd.notna(r[3]) else 0.0)))
    head = (
        "<tr><th>variant</th><th>PWSF (compliant mean)</th><th>95% CI</th>"
        "<th>n_compliant/n_total</th></tr>"
    )
    body = "".join(
        f"<tr><td>{alias_variant(v)}</td>"
        f"<td>{'—' if pd.isna(mean) else f'{mean:.4f}'}</td>"
        f"<td>{'—' if pd.isna(ci) else f'±{ci:.4f}'}</td>"
        f"<td>{n_c}/{n_t}</td></tr>"
        for v, n_c, n_t, mean, ci in rows
    )
    caption = (
        "<tr><td colspan='4' style='color:#888'>compliant subset, "
        "completed sims (ok + claims_failed), unpaired, pooled across "
        "grids and experiments — see REPORT.md</td></tr>"
    )
    return "<table class='status-table'>" + head + body + caption + "</table>"


def _plot_card(html_path: Path) -> str | None:
    block = _extract_plot_block(html_path)
    if block is None:
        return None
    return f'<div class="plot">{block}</div>'


def _section(title: str, html_paths: list[Path]) -> str:
    cards = [c for c in (_plot_card(p) for p in html_paths) if c is not None]
    if not cards:
        return ""
    return f"<h2>{title}</h2>" + '<div class="grid">' + "\n".join(cards) + "</div>"


def _page(title: str, *, lead: str, meta: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head>"
        f"<meta charset='utf-8'><title>{title}</title>"
        f"{_PLOTLY_CDN}{_STYLE}"
        "</head><body>"
        f"<h1>{title}</h1>"
        f"<p class='lead'>{lead}</p>"
        f"<p class='meta'>{meta}</p>"
        f"{body}"
        "<footer>Generated by <code>experiment.eval.overview</code>."
        "  Hover any plot for details; the underlying PDFs sit next to each "
        "HTML for inclusion in chapters.</footer>"
        "</body></html>"
    )


# Top-level / per-experiment generators


def _top_level_sections(plots_root: Path) -> list[tuple[str, list[Path]]]:
    """Curated headline view — only the plots that answer "did the
    chapter's core claims survive this run?".
    """
    p = plots_root
    return [
        (
            "Restoration headline",
            [
                p / "functional_baseline" / "served_per_grid.html",
                p / "restoration" / "absolute_vs_baseline.html",
                p / "restoration" / "ratio_by_variant.html",
                p / "restoration" / "absolute_load_lost.html",
            ],
        ),
        (
            "Per-tier / per-sector breakdown",
            [
                p / "restoration" / "by_tier.html",
                p / "restoration" / "by_sector.html",
                p / "restoration" / "loss_split_by_tier.html",
                p / "restoration" / "agent_only_ratio_by_tier.html",
            ],
        ),
        (
            "Variant comparison",
            [
                p / "variant_comparison" / "served_by_variant.html",
                p / "variant_comparison" / "pwsf_by_sector.html",
                p / "restoration_time" / "time_to_stabilise.html",
                p / "variant_comparison" / "regulates_by_reason.html",
                p / "variant_comparison" / "diary_outcomes.html",
            ],
        ),
        (
            "Diagnostics",
            [
                p / "solver_health" / "solver_health.html",
                p / "constraints" / "violation_integral.html",
                p / "claims" / "claims_overall.html",
                p / "variant_comparison" / "claims_pass_rate.html",
            ],
        ),
        (
            "Representative trajectory (functional baseline)",
            [
                p / "functional_baseline" / "representative_trajectory.html",
                p / "functional_baseline" / "representative_constraint_envelope.html",
                p / "functional_baseline" / "representative_served_by_tier.html",
            ],
        ),
    ]


def _constraints_sections(
    plots_root: Path,
    experiments: list[str],
) -> list[tuple[str, list[Path]]]:
    """Constraint-handling overview: did the network stay inside its operating
    envelope?  Bundles the campaign-wide per-sector violation-integral bar, the
    functional-baseline representative envelope trajectory, and every
    per-(experiment, variant) envelope trajectory under
    ``plots/trajectories/<exp>/<variant>/``.
    """
    p = plots_root
    sections: list[tuple[str, list[Path]]] = []

    # Campaign-wide integral bar.
    head: list[Path] = [p / "constraints" / "violation_integral.html"]
    if (p / "functional_baseline" / "representative_constraint_envelope.html").exists():
        head.append(
            p / "functional_baseline" / "representative_constraint_envelope.html"
        )
    sections.append(("Campaign + representative", head))

    # Per-(experiment × variant) envelope trajectories.
    traj_root = p / "trajectories"
    if traj_root.exists():
        for exp in experiments:
            if exp == "functional_baseline":
                continue
            exp_dir = traj_root / exp
            if not exp_dir.exists():
                continue
            envelopes: list[Path] = []
            for variant_dir in sorted(d for d in exp_dir.iterdir() if d.is_dir()):
                env = variant_dir / "constraint_envelope.html"
                if env.exists():
                    envelopes.append(env)
            if envelopes:
                sections.append((f"{exp} — per-variant envelopes", envelopes))
    return sections


def _experiment_sections(
    plots_root: Path,
    experiment: str,
) -> list[tuple[str, list[Path]]]:
    """All ``.html`` figures rendered under ``plots/<experiment>/`` plus
    any per-variant trajectory pair under ``plots/trajectories/<experiment>/``.
    """
    sections: list[tuple[str, list[Path]]] = []
    exp_dir = plots_root / experiment
    if exp_dir.exists():
        figs = sorted(exp_dir.glob("*.html"))
        if figs:
            sections.append(("Figures", figs))
    traj_dir = plots_root / "trajectories" / experiment
    if traj_dir.exists():
        for variant_dir in sorted(p for p in traj_dir.iterdir() if p.is_dir()):
            figs = sorted(variant_dir.glob("*.html"))
            if figs:
                sections.append((f"Trajectory — {variant_dir.name}", figs))
    return sections


def write_overview(campaign: CampaignData) -> Path:
    """Write ``plots/overview.html`` and one ``overview_<experiment>.html``
    per experiment.  Returns the path to the top-level overview.
    """
    plots_root = campaign.campaign_dir / "plots"
    plots_root.mkdir(exist_ok=True)

    # Top-level
    lead = (
        "Headline view of campaign "
        f"<code>{campaign.campaign_dir.name}</code>. "
        "Each card is a fully interactive Plotly figure — hover for values, "
        "drag to zoom, double-click to reset."
    )
    meta_bits = [_status_table(campaign), _variant_means_table(campaign)]
    body_sections = _top_level_sections(plots_root)
    body = "".join(_section(t, p) for t, p in body_sections)

    # Per-experiment TOC links, so the headline page links to each
    # per-experiment overview before they are materialised below.
    experiments = campaign.experiments()
    toc_links: list[str] = [
        '<a href="overview_validity.html">validity</a>',
        '<a href="overview_constraints.html">constraints</a>',
    ]
    for exp in experiments:
        toc_links.append(f'<a href="overview_{exp}.html">{exp}</a>')
    toc = (
        (
            "<nav class='toc'><b>Dedicated overviews:</b><br>"
            + " ".join(toc_links)
            + "</nav>"
        )
        if toc_links
        else ""
    )

    out_path = plots_root / "overview.html"
    out_path.write_text(
        _page(
            title=f"Overview — {campaign.campaign_dir.name}",
            lead=lead,
            meta="\n".join(meta_bits),
            body=toc + body,
        ),
        encoding="utf-8",
    )

    # Validity overview (dedicated)
    validity_dir = plots_root / "validity"
    if validity_dir.exists():
        validity_figs = [
            validity_dir / "system_balance.html",
            validity_dir / "coalition_balance.html",
            validity_dir / "holon_balance.html",
            validity_dir / "regulation_per_child.html",
        ]
        validity_body = _section(
            "Multi-level balances + per-child regulation",
            validity_figs,
        )
        if validity_body:
            (plots_root / "overview_validity.html").write_text(
                _page(
                    title=(f"Validity overview — {campaign.campaign_dir.name}"),
                    lead=(
                        "Did the multi-level controller behave as the "
                        "architecture chapter claims?  The four traces "
                        "show progressively wider views — full system, "
                        "Level-2 holons, Level-1 coalitions, individual "
                        "device regulation — all for the "
                        "<code>functional_baseline</code> representative "
                        "task.  Coalitions / holons that stay oscillating "
                        "after the rest have settled point at a "
                        "coordination gap; flat per-child regulation "
                        "means the device wasn't asked to do anything."
                    ),
                    meta=(
                        "Source: <code>plots/validity/*.html</code>.  "
                        'Top-level overview: <a href="overview.html">overview.html</a>.'
                    ),
                    body=validity_body,
                ),
                encoding="utf-8",
            )

    # Constraints overview (dedicated)
    constraints_sections = _constraints_sections(plots_root, experiments)
    if any(figs for _, figs in constraints_sections):
        constraints_body = "".join(
            _section(t, paths) for t, paths in constraints_sections
        )
        n_env = sum(
            len([p for p in paths if "constraint_envelope" in p.name])
            for _, paths in constraints_sections
        )
        (plots_root / "overview_constraints.html").write_text(
            _page(
                title=f"Constraints overview — {campaign.campaign_dir.name}",
                lead=(
                    "Did the network stay inside its operating envelope?  "
                    "The integral bar quantifies average overshoot per "
                    "variant; the envelope trajectories show one task per "
                    "(experiment × variant) so excursion patterns can be "
                    "compared side-by-side."
                ),
                meta=(
                    f"<code>{len(experiments)}</code> experiment(s), "
                    f"<code>{n_env}</code> envelope trajectory plot(s) collated. "
                    'Top-level overview: <a href="overview.html">overview.html</a>.'
                ),
                body=constraints_body,
            ),
            encoding="utf-8",
        )

    # Per experiment
    for exp in experiments:
        sections = _experiment_sections(plots_root, exp)
        if not sections:
            continue
        sub = campaign.by_experiment(exp)
        ok = sub[sub["status"] == "ok"] if "status" in sub.columns else sub
        meta = (
            f"<code>{exp}</code> — "
            f"{len(sub)} task(s), {len(ok)} OK, "
            f"{sub['variant'].nunique() if 'variant' in sub.columns else 0} variants, "
            f"{sub['grid'].nunique() if 'grid' in sub.columns else 0} grid(s)."
        )
        page_body = "".join(_section(t, p) for t, p in sections)
        (plots_root / f"overview_{exp}.html").write_text(
            _page(
                title=f"Overview — {exp}",
                lead=(
                    "All figures rendered for this experiment, in one place. "
                    'Top-level overview: <a href="overview.html">overview.html</a>.'
                ),
                meta=meta,
                body=page_body,
            ),
            encoding="utf-8",
        )

    logger.info(
        "Wrote overview.html + %d per-experiment overview(s) under %s",
        sum(1 for e in experiments if (plots_root / f"overview_{e}.html").exists()),
        plots_root,
    )
    return out_path
