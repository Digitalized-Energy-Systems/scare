"""Plotly-based plot primitives for the dissertation evaluation.

Each function takes a slice of the summary DataFrame (or a per-task
artefact) and writes one figure to disk in two formats:

- ``<name>.html`` — interactive Plotly figure for exploration.
- ``<name>.png``  — high-DPI static image for inclusion in chapters.

Style is consistent across all figures via the ``_apply_theme`` helper:
serif title, sans-serif body, light gridlines, scientific colour
palette, no chart junk.  Variants get fixed colours so the eye learns
them across the report.

Each primitive returns the *base* path stem (``out_path`` without a
suffix) so the caller can reference both the .html and .png alongside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Style — consistent palette + layout across the whole report
# ---------------------------------------------------------------------------


# Variant palette — picked for a) good contrast on white, b) print-friendly,
# c) consistent meaning across every figure that splits by variant.
_VARIANT_COLOR = {
    "oracle": "#2E7D32",         # green — upper bound
    "scare": "#1F4E96",          # deep blue — the protagonist
    "single_level": "#E07A1F",   # warm orange — the strawman
}

# Sector palette — used in trajectory + per-tier views.
_SECTOR_COLOR = {
    "electricity": "#1F77B4",
    "gas": "#2CA02C",
    "heat": "#D62728",
}

# Qualitative palette for ablations / sweeps / scenarios — colour-blind safe.
_QUAL_PALETTE = [
    "#1F4E96", "#2E7D32", "#E07A1F", "#9467BD", "#8C564B",
    "#17BECF", "#BCBD22", "#7F7F7F", "#E377C2", "#AEC7E8",
]

_FONT_FAMILY = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
_TITLE_FONT_FAMILY = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

_DEFAULT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=_FONT_FAMILY, size=13, color="#1A1A1A"),
    title=dict(
        font=dict(family=_TITLE_FONT_FAMILY, size=18, color="#1A1A1A"),
        x=0.02,
        xanchor="left",
        y=0.97,
        yanchor="top",
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=70, r=30, t=70, b=60),
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#DCDCDC",
        borderwidth=1,
        font=dict(size=12),
    ),
    xaxis=dict(
        gridcolor="#EAEAEA",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="#1A1A1A",
        linewidth=1,
        ticks="outside",
        tickcolor="#1A1A1A",
        ticklen=4,
    ),
    yaxis=dict(
        gridcolor="#EAEAEA",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="#1A1A1A",
        linewidth=1,
        ticks="outside",
        tickcolor="#1A1A1A",
        ticklen=4,
    ),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#1A1A1A",
        font=dict(family=_FONT_FAMILY, size=12),
    ),
)


def _apply_theme(fig: go.Figure, *, title: str, height: int = 460) -> go.Figure:
    fig.update_layout(_DEFAULT_LAYOUT)
    fig.update_layout(title=dict(text=title, **_DEFAULT_LAYOUT["title"]),
                      height=height)
    return fig


def _save(fig: go.Figure, out_path: Path, *, png_scale: int = 2) -> Path:
    """Write both HTML and PNG; return the directory-relative stem path
    (no suffix)."""
    out_path = Path(out_path).with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_path = out_path.with_suffix(".html")
    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True, "toImageButtonOptions": {"scale": png_scale}},
    )
    png_path = out_path.with_suffix(".png")
    try:
        fig.write_image(png_path, scale=png_scale)
    except Exception:
        # Kaleido failure shouldn't kill the whole report — HTML is
        # the canonical format; PNG is for static inclusion only.
        pass
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))])
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


def _empty_fig(message: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family=_FONT_FAMILY, size=14, color="#888888"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply_theme(fig, title=title)


def _variant_color(variant: str, fallback: str | None = None) -> str:
    return _VARIANT_COLOR.get(variant, fallback or _QUAL_PALETTE[hash(variant) % len(_QUAL_PALETTE)])


# ---------------------------------------------------------------------------
# Variant comparison (Pillar 3)
# ---------------------------------------------------------------------------


def variant_comparison_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    metric_col: str = "outcomes__priority_weighted_fraction",
    title: str = "Priority-weighted served fraction by variant",
) -> Path:
    if df.empty or metric_col not in df.columns:
        return _save(_empty_fig("no data", title), out_path)

    grouped = df.groupby(["grid", "variant"])[metric_col].apply(list)
    grids = sorted({k[0] for k in grouped.index})
    variants = sorted({k[1] for k in grouped.index})

    fig = go.Figure()
    for variant in variants:
        means: list[float] = []
        cis: list[float] = []
        ns: list[int] = []
        for grid in grids:
            vals = grouped.get((grid, variant), [])
            mean, ci = _mean_ci(vals)
            means.append(mean)
            cis.append(ci)
            ns.append(len(vals))
        hover = [
            f"<b>{variant}</b><br>grid: {g}<br>mean: {m:.4f}<br>95% CI: ±{c:.4f}<br>n: {n}"
            for g, m, c, n in zip(grids, means, cis, ns)
        ]
        fig.add_trace(go.Bar(
            name=variant,
            x=grids,
            y=means,
            error_y=dict(type="data", array=cis, visible=True, thickness=1.5, color="#1A1A1A"),
            marker=dict(color=_variant_color(variant), line=dict(width=0)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

    fig.update_layout(barmode="group", bargap=0.18, bargroupgap=0.06)
    fig.update_yaxes(range=[0, 1.05], title="priority-weighted served fraction")
    fig.update_xaxes(title="grid")
    return _save(_apply_theme(fig, title=title), out_path)


# ---------------------------------------------------------------------------
# Optimality gap (Pillar 2)
# ---------------------------------------------------------------------------


def optimality_gap_scatter(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Optimality gap: scare vs centralised oracle"
    if df.empty:
        return _save(_empty_fig("no data", title), out_path)
    metric = "outcomes__priority_weighted_fraction"
    pivot = df.pivot_table(index=["grid", "seed"], columns="variant", values=metric)
    if "scare" not in pivot.columns or "oracle" not in pivot.columns:
        return _save(_empty_fig("need both 'scare' and 'oracle' variants", title), out_path)
    pivot = pivot.dropna(subset=["scare", "oracle"], how="any")
    if pivot.empty:
        return _save(_empty_fig("no scare/oracle pairs", title), out_path)

    fig = go.Figure()
    # Parity line first so points draw on top.
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#999999", dash="dash", width=1.2),
        name="parity (oracle = scare)",
        hoverinfo="skip",
    ))
    grids = sorted(pivot.index.get_level_values("grid").unique())
    for i, grid in enumerate(grids):
        sub = pivot.xs(grid, level="grid")
        seeds = list(sub.index)
        fig.add_trace(go.Scatter(
            x=sub["oracle"], y=sub["scare"],
            mode="markers",
            name=grid,
            marker=dict(
                size=10,
                color=_QUAL_PALETTE[i % len(_QUAL_PALETTE)],
                line=dict(width=1.2, color="white"),
                opacity=0.85,
            ),
            customdata=[
                f"grid: {grid}<br>seed: {s}<br>oracle: {sub.loc[s,'oracle']:.4f}<br>scare: {sub.loc[s,'scare']:.4f}<br>gap: {(sub.loc[s,'oracle']-sub.loc[s,'scare']):.4f}"
                for s in seeds
            ],
            hovertemplate="%{customdata}<extra></extra>",
        ))

    # Mean gap annotation.
    pivot["gap"] = pivot["oracle"] - pivot["scare"]
    mean_gap = float(pivot["gap"].mean())
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#1A1A1A", borderwidth=1, borderpad=6,
        text=f"<b>mean gap</b>: {mean_gap:.4f}<br><b>n</b>: {len(pivot)}",
        font=dict(family=_FONT_FAMILY, size=12),
    )

    fig.update_xaxes(title="oracle priority-weighted served", range=[0, 1.05])
    fig.update_yaxes(title="scare priority-weighted served", range=[0, 1.05])
    return _save(_apply_theme(fig, title=title), out_path)


def optimality_gap_box(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Optimality gap distribution by grid"
    if df.empty:
        return _save(_empty_fig("no data", title), out_path)
    metric = "outcomes__priority_weighted_fraction"
    pivot = df.pivot_table(index=["grid", "seed"], columns="variant", values=metric)
    if "scare" not in pivot.columns or "oracle" not in pivot.columns:
        return _save(_empty_fig("need both 'scare' and 'oracle' variants", title), out_path)
    pivot = pivot.dropna(subset=["scare", "oracle"], how="any")
    if pivot.empty:
        return _save(_empty_fig("no scare/oracle pairs", title), out_path)
    pivot["gap"] = (pivot["oracle"] - pivot["scare"]) / pivot["oracle"].replace(0, np.nan)
    pivot = pivot.dropna(subset=["gap"])
    if pivot.empty:
        return _save(_empty_fig("no positive-oracle rows", title), out_path)

    fig = go.Figure()
    grids = sorted(pivot.index.get_level_values("grid").unique())
    for i, grid in enumerate(grids):
        gaps = pivot.xs(grid, level="grid")["gap"].values
        fig.add_trace(go.Box(
            y=gaps,
            name=grid,
            marker=dict(color=_QUAL_PALETTE[i % len(_QUAL_PALETTE)], outliercolor="#D62728"),
            boxmean=True,
            boxpoints="outliers",
            line=dict(width=1.2),
            hovertemplate="grid: %{x}<br>gap: %{y:.4f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="#999999", dash="dash", width=1))
    fig.update_yaxes(title="relative gap (oracle − scare) / oracle")
    fig.update_xaxes(title="grid")
    fig.update_layout(showlegend=False)
    return _save(_apply_theme(fig, title=title), out_path)


# ---------------------------------------------------------------------------
# Ablation impact (Pillar 4)
# ---------------------------------------------------------------------------


def ablation_impact_bar(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Ablation impact (scare variant)"
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    grouped = df.groupby("ablation")[metric].agg(mean="mean", count="count", std="std")
    grouped["ci"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"]) * 1.96
    grouped = grouped.sort_values("mean", ascending=True)

    baseline_mean = grouped.loc["default", "mean"] if "default" in grouped.index else None
    colors = ["#7F7F7F" if k == "default" else _VARIANT_COLOR["scare"] for k in grouped.index]

    hover = [
        f"<b>{k}</b><br>mean: {m:.4f}<br>95% CI: ±{c:.4f}<br>n: {n}<br>"
        f"Δ vs default: {(m - baseline_mean):+.4f}" if baseline_mean is not None else
        f"<b>{k}</b><br>mean: {m:.4f}<br>95% CI: ±{c:.4f}<br>n: {n}"
        for k, m, c, n in zip(grouped.index, grouped["mean"], grouped["ci"], grouped["count"])
    ]

    fig = go.Figure(go.Bar(
        x=grouped["mean"],
        y=grouped.index,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        error_x=dict(type="data", array=grouped["ci"], thickness=1.5, color="#1A1A1A"),
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    if baseline_mean is not None:
        fig.add_vline(
            x=baseline_mean,
            line=dict(color="#D62728", dash="dash", width=1.2),
            annotation_text="baseline",
            annotation_position="top right",
            annotation_font=dict(color="#D62728", size=12),
        )
    fig.update_xaxes(title="mean priority-weighted served fraction", range=[0, 1.05])
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False, height=max(360, 50 * len(grouped) + 80))
    return _save(_apply_theme(fig, title=title, height=max(360, 50 * len(grouped) + 80)), out_path)


# ---------------------------------------------------------------------------
# Robustness curves (Pillar 5)
# ---------------------------------------------------------------------------


def robustness_curve(
    df: pd.DataFrame,
    out_path: Path,
    *,
    sweep_param: str,
    x_label: str,
    title: str,
) -> Path:
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns:
        return _save(_empty_fig("no data", title), out_path)

    parsed = df.copy()
    parsed["x"] = parsed["sweep"].apply(lambda s: _extract_sweep_value(s, sweep_param))
    parsed = parsed.dropna(subset=["x"])
    if parsed.empty:
        return _save(_empty_fig(f"no {sweep_param} sweep", title), out_path)

    grouped = parsed.groupby("x")[metric].agg(["mean", "std", "count"])
    grouped["ci"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"]) * 1.96
    grouped = grouped.sort_index()

    x = grouped.index.tolist()
    y = grouped["mean"].tolist()
    upper = (grouped["mean"] + grouped["ci"]).tolist()
    lower = (grouped["mean"] - grouped["ci"]).tolist()

    fig = go.Figure()
    # CI ribbon — fill between upper and lower with a transparent band.
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(31, 78, 150, 0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color=_VARIANT_COLOR["scare"], width=2.2),
        marker=dict(size=9, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1.2)),
        customdata=[
            f"x={xv}<br>mean: {m:.4f}<br>95% CI: ±{c:.4f}<br>n: {int(n)}"
            for xv, m, c, n in zip(x, grouped["mean"], grouped["ci"], grouped["count"])
        ],
        hovertemplate="%{customdata}<extra></extra>",
        name="scare",
    ))
    fig.update_yaxes(title="priority-weighted served fraction", range=[0, 1.05])
    fig.update_xaxes(title=x_label)
    fig.update_layout(showlegend=False)
    return _save(_apply_theme(fig, title=title), out_path)


def _extract_sweep_value(sweep_key: str, param: str) -> float | None:
    if not sweep_key or sweep_key == "default":
        return 0.0
    for tok in str(sweep_key).split(";"):
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k.strip() == param:
            try:
                return float(v.strip())
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Cascading (Pillar 6)
# ---------------------------------------------------------------------------


def cascading_curve(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Restoration quality vs simultaneous failure count"
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns or "n_failures" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)

    grouped = df.groupby("n_failures")[metric].agg(["mean", "std", "count"])
    grouped["ci"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"]) * 1.96
    grouped = grouped.sort_index()

    x = grouped.index.tolist()
    y = grouped["mean"].tolist()
    upper = (grouped["mean"] + grouped["ci"]).tolist()
    lower = (grouped["mean"] - grouped["ci"]).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(31, 78, 150, 0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color=_VARIANT_COLOR["scare"], width=2.2),
        marker=dict(size=10, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1.2)),
        customdata=[
            f"failures: {xv}<br>mean served: {m:.4f}<br>95% CI: ±{c:.4f}<br>n: {int(n)}"
            for xv, m, c, n in zip(x, grouped["mean"], grouped["ci"], grouped["count"])
        ],
        hovertemplate="%{customdata}<extra></extra>",
        name="scare",
    ))
    fig.update_yaxes(title="priority-weighted served fraction", range=[0, 1.05])
    fig.update_xaxes(title="number of simultaneous failures", dtick=1)
    fig.update_layout(showlegend=False)
    return _save(_apply_theme(fig, title=title), out_path)


# ---------------------------------------------------------------------------
# Sensitivity sweeps (Pillar 7)
# ---------------------------------------------------------------------------


def sweep_curve_dual(
    df: pd.DataFrame,
    out_path: Path,
    *,
    sweep_param: str,
    x_label: str,
    title: str,
) -> Path:
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns:
        return _save(_empty_fig("no data", title), out_path)

    parsed = df.copy()
    parsed["x"] = parsed["sweep"].apply(lambda s: _extract_sweep_value(s, sweep_param))
    parsed = parsed.dropna(subset=["x"])
    if parsed.empty:
        return _save(_empty_fig(f"no {sweep_param} sweep", title), out_path)

    served = parsed.groupby("x")[metric].mean().sort_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=served.index, y=served.values,
        mode="lines+markers",
        name="served",
        line=dict(color=_VARIANT_COLOR["scare"], width=2.2),
        marker=dict(size=9, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1.2)),
        hovertemplate=f"{sweep_param}: %{{x}}<br>served: %{{y:.4f}}<extra></extra>",
    ), secondary_y=False)

    if "duration_s" in parsed.columns:
        wall = parsed.groupby("x")["duration_s"].mean().sort_index()
        fig.add_trace(go.Scatter(
            x=wall.index, y=wall.values,
            mode="lines+markers",
            name="wallclock (s)",
            line=dict(color=_VARIANT_COLOR["single_level"], width=2.2, dash="dot"),
            marker=dict(size=9, color=_VARIANT_COLOR["single_level"], symbol="square",
                        line=dict(color="white", width=1.2)),
            hovertemplate=f"{sweep_param}: %{{x}}<br>wallclock: %{{y:.0f}}s<extra></extra>",
        ), secondary_y=True)
        fig.update_yaxes(title="wallclock (s)", color=_VARIANT_COLOR["single_level"], secondary_y=True,
                         showgrid=False, gridcolor="#EAEAEA")

    fig.update_yaxes(title="priority-weighted served fraction", range=[0, 1.05],
                     color=_VARIANT_COLOR["scare"], secondary_y=False)
    fig.update_xaxes(title=x_label)
    fig.update_layout(legend=dict(x=0.5, y=-0.25, orientation="h", xanchor="center"))
    return _save(_apply_theme(fig, title=title), out_path)


# ---------------------------------------------------------------------------
# Per-tier served (Pillars 1, 4)
# ---------------------------------------------------------------------------


def served_by_tier(
    served: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Served fraction by priority tier",
) -> Path:
    if served.empty or {"sector", "tier", "fraction"} - set(served.columns):
        return _save(_empty_fig("no served.csv", title), out_path)
    pivot = served.pivot(index="tier", columns="sector", values="fraction").sort_index()
    if pivot.empty:
        return _save(_empty_fig("empty served data", title), out_path)

    fig = go.Figure()
    for sec in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index.astype(str),
            y=pivot[sec].values,
            name=sec,
            marker=dict(color=_SECTOR_COLOR.get(sec, "#888888"), line=dict(width=0)),
            hovertemplate=f"<b>{sec}</b><br>tier: %{{x}}<br>served: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.18, bargroupgap=0.08)
    fig.update_xaxes(title="priority tier (1 = most critical)")
    fig.update_yaxes(title="served fraction", range=[0, 1.05])
    return _save(_apply_theme(fig, title=title), out_path)


# ---------------------------------------------------------------------------
# Restoration trajectory (per task)
# ---------------------------------------------------------------------------


def restoration_trajectory(
    timeseries: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Restoration trajectory",
    failure_t: float | None = None,
) -> Path:
    if timeseries.empty or "time_s" not in timeseries.columns:
        return _save(_empty_fig("no timeseries", title), out_path)

    fig = go.Figure()
    sectors = [
        ("electrical_balance", "electricity"),
        ("gas_balance", "gas"),
        ("heat_balance", "heat"),
    ]
    for col, sec in sectors:
        if col in timeseries.columns:
            fig.add_trace(go.Scatter(
                x=timeseries["time_s"], y=timeseries[col],
                mode="lines",
                name=sec,
                line=dict(color=_SECTOR_COLOR[sec], width=2),
                hovertemplate=f"<b>{sec}</b><br>t: %{{x:.2f}}s<br>balance: %{{y:.4f}}<extra></extra>",
            ))

    # Event markers — distinct shapes / colours so they read at a glance.
    if not events.empty and {"t", "kind"}.issubset(events.columns):
        event_styles = {
            "line_failure": dict(color="#1A1A1A", dash="dash"),
            "reconfiguration_completed": dict(color="#9467BD", dash="dot"),
            "islanding_request": dict(color="#E07A1F", dash="dot"),
            "islanding_covered": dict(color="#2E7D32", dash="dot"),
            "constraint_violation": dict(color="#D62728", dash="dot"),
        }
        for kind, style in event_styles.items():
            sub = events[events["kind"] == kind]
            if sub.empty:
                continue
            for tx in sub["t"].astype(float).unique():
                fig.add_vline(
                    x=float(tx),
                    line=dict(color=style["color"], dash=style["dash"], width=1),
                    opacity=0.7,
                )
            # Add a sentinel scatter so it appears in the legend.
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color=style["color"], dash=style["dash"], width=2),
                name=kind,
                showlegend=True,
            ))
    elif failure_t is not None:
        fig.add_vline(x=failure_t, line=dict(color="#1A1A1A", dash="dash", width=1))

    fig.update_xaxes(title="simulation time (s)")
    fig.update_yaxes(title="Σ regulation per sector")
    fig.update_layout(legend=dict(x=1.02, y=1.0, xanchor="left"))
    return _save(_apply_theme(fig, title=title), out_path)


# ---------------------------------------------------------------------------
# Claims pass-rate (Pillar 8)
# ---------------------------------------------------------------------------


def claims_pass_rate(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Claims validation pass rate by variant"
    if df.empty:
        return _save(_empty_fig("no data", title), out_path)
    claim_cols = [c for c in df.columns if c.startswith("claims__") and c.endswith("__passed")]
    if not claim_cols:
        return _save(_empty_fig("no claims data", title), out_path)

    rows = []
    for col in claim_cols:
        claim_name = col[len("claims__"):-len("__passed")]
        for variant, g in df.groupby("variant"):
            n = g[col].dropna().shape[0]
            if n == 0:
                continue
            rate = float(g[col].astype(bool).sum()) / n
            rows.append((claim_name, str(variant), rate, n))
    if not rows:
        return _save(_empty_fig("no claims data", title), out_path)

    rdf = pd.DataFrame(rows, columns=["claim", "variant", "pass_rate", "n"])
    pivot = rdf.pivot(index="claim", columns="variant", values="pass_rate").fillna(np.nan)
    n_pivot = rdf.pivot(index="claim", columns="variant", values="n").fillna(0)

    fig = go.Figure()
    for variant in pivot.columns:
        fig.add_trace(go.Bar(
            y=pivot.index,
            x=pivot[variant].values,
            name=variant,
            orientation="h",
            marker=dict(color=_variant_color(variant), line=dict(width=0)),
            customdata=[
                f"<b>{c}</b><br>variant: {variant}<br>pass rate: {p:.1%}<br>n: {int(n_pivot.loc[c, variant])}"
                for c, p in zip(pivot.index, pivot[variant].values)
            ],
            hovertemplate="%{customdata}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.06)
    fig.update_xaxes(title="pass rate", range=[0, 1.05], tickformat=".0%")
    fig.update_yaxes(title="")
    return _save(_apply_theme(fig, title=title, height=max(360, 60 * len(pivot) + 80)), out_path)


# ---------------------------------------------------------------------------
# Diary distribution (Pillars 1, 8)
# ---------------------------------------------------------------------------


def diary_outcomes_bar(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Negotiation lifecycle outcomes by variant"
    cols = [
        ("diary__finished", "finished", "#2E7D32"),
        ("diary__stalled", "stalled", "#17BECF"),
        ("diary__cancelled", "cancelled", "#E07A1F"),
        ("diary__timed_out", "timed_out", "#D62728"),
        ("diary__abandoned", "abandoned", "#7F7F7F"),
        ("diary__skipped_balanced", "skipped_balanced", "#1F4E96"),
        ("diary__skipped_singleton", "skipped_singleton", "#9467BD"),
    ]
    cols = [c for c in cols if c[0] in df.columns]
    if df.empty or not cols:
        return _save(_empty_fig("no diary data", title), out_path)

    by_variant = df.groupby("variant")[[c[0] for c in cols]].sum()
    if by_variant.empty:
        return _save(_empty_fig("no diary data", title), out_path)

    fig = go.Figure()
    for col, label, color in cols:
        fig.add_trace(go.Bar(
            x=by_variant.index,
            y=by_variant[col].values,
            name=label,
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate=f"<b>{label}</b><br>variant: %{{x}}<br>count: %{{y}}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.18)
    fig.update_xaxes(title="variant")
    fig.update_yaxes(title="count")
    return _save(_apply_theme(fig, title=title), out_path)
