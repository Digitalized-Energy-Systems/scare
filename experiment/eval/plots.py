"""Plotly plot primitives for the evaluation.

Each function takes a slice of the summary DataFrame (or a per-task
artefact) and writes one figure as ``<name>.html`` (interactive) and
``<name>.pdf`` (vector, for chapter inclusion). Style is shared via
``_apply_theme``; variants get fixed colours for cross-figure consistency.
Returns the base path stem (``out_path`` without suffix).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiment.eval.compliance import (
    compliance_rate as _compliance_rate,
)
from experiment.eval.compliance import (
    compliant_mask as _compliant_mask,
)
from experiment.eval.compliance import (
    mean_ci95 as _mean_ci,
)

# Style — consistent palette + layout across the whole report


# Variant palette — fixed colour per variant across every figure.
_VARIANT_COLOR = {
    "oracle": "#2E7D32",  # green — upper bound
    "scare": "#1F4E96",  # deep blue
    "single_level": "#E07A1F",  # warm orange — baseline
    "component_level": "#9467BD",  # purple — component-scoped variant
}

# Sector palette — used in trajectory + per-tier views.
_SECTOR_COLOR = {
    "electricity": "#1F77B4",
    "gas": "#2CA02C",
    "heat": "#D62728",
}

# Constraint-variable-type palette + canonical order for the per-variable
# violation tally. ``temperature`` is non-gating (diagnostic only — see
# ``claims.NON_GATING_CONSTRAINT_VARIABLES``); rendered with a hatch so it
# reads as a companion stat, not a compliance failure.
_CONSTRAINT_VARIABLE_COLOR = {
    "voltage": "#1F77B4",
    "pressure": "#2CA02C",
    "line_load": "#9467BD",
    "slack": "#E07A1F",
    "temperature": "#D62728",
}
_CONSTRAINT_VARIABLE_ORDER = (
    "voltage",
    "pressure",
    "line_load",
    "slack",
    "temperature",
)
_NONGATING_VARIABLE_TYPES = frozenset({"temperature"})

# Qualitative palette for ablations / sweeps / scenarios — colourblind-safe.
_QUAL_PALETTE = [
    "#1F4E96",
    "#2E7D32",
    "#E07A1F",
    "#9467BD",
    "#8C564B",
    "#17BECF",
    "#BCBD22",
    "#7F7F7F",
    "#E377C2",
    "#AEC7E8",
]

_FONT_FAMILY = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
_TITLE_FONT_FAMILY = "Charter, Georgia, 'Times New Roman', serif"

# Full-width landscape figures with generous label breathing room.
_FIG_WIDTH = 1000
_FIG_HEIGHT = 440

# Font sizes tuned to read at full-width PDF without further scaling.
_BASE_FONT_SIZE = 17
_TITLE_FONT_SIZE = 22
_AXIS_TITLE_FONT_SIZE = 18
_TICK_FONT_SIZE = 16
_LEGEND_FONT_SIZE = 16
_ANNOTATION_FONT_SIZE = 14

_GRID_COLOR = "#ECECEC"
_AXIS_COLOR = "#1A1A1A"
_MUTED_COLOR = "#666666"

# Horizontal gridlines only, no axis spine or enclosing box. X-axis
# overrides below to hide grid lines entirely.
_AXIS_STYLE = dict(
    gridcolor=_GRID_COLOR,
    gridwidth=0.8,
    zeroline=False,
    showline=False,
    mirror=False,
    # No outside tick marks — gridlines + labels carry the scale; this also
    # keeps every subplot panel consistent (no stray ticks on some rows).
    ticks="",
    ticklen=0,
    tickcolor=_AXIS_COLOR,
    tickwidth=0.9,
    tickfont=dict(size=_TICK_FONT_SIZE),
    title=dict(font=dict(size=_AXIS_TITLE_FONT_SIZE), standoff=10),
    automargin=True,
)
_X_AXIS_STYLE = {**_AXIS_STYLE, "showgrid": False}
_Y_AXIS_STYLE = {**_AXIS_STYLE, "showgrid": True}

_DEFAULT_LAYOUT = dict(
    template="plotly_white",
    width=_FIG_WIDTH,
    height=_FIG_HEIGHT,
    font=dict(family=_FONT_FAMILY, size=_BASE_FONT_SIZE, color=_AXIS_COLOR),
    # Centered title.
    title=dict(
        font=dict(family=_TITLE_FONT_FAMILY, size=_TITLE_FONT_SIZE, color=_AXIS_COLOR),
        x=0.5,
        xanchor="center",
        y=0.96,
        yanchor="top",
        pad=dict(t=6, b=6),
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=84, r=160, t=72, b=72),
    # Vertical legend on the right — keeps the full plot area for the data.
    legend=dict(
        bgcolor="rgba(255,255,255,0)",
        bordercolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=_LEGEND_FONT_SIZE),
        orientation="v",
        x=1.02,
        xanchor="left",
        y=1.0,
        yanchor="top",
        itemsizing="constant",
        tracegroupgap=4,
    ),
    xaxis=_X_AXIS_STYLE,
    yaxis=_Y_AXIS_STYLE,
    hoverlabel=dict(
        bgcolor="white",
        bordercolor=_AXIS_COLOR,
        font=dict(family=_FONT_FAMILY, size=13),
    ),
    # Solid bar fills, no white separator stroke.
    bargap=0.2,
    bargroupgap=0.06,
)


def _apply_theme(
    fig: go.Figure,
    *,
    title: str,
    height: int = _FIG_HEIGHT,
    width: int = _FIG_WIDTH,
    font_bump: int = 0,
    legend_top: bool = False,
    no_legend: bool = False
) -> go.Figure:
    """Apply the shared figure theme. ``font_bump`` adds the same delta
    (pt) to every text element, for figures shown as large panels.
    ``legend_top`` moves the legend to a horizontal strip above the plot
    (used for bar charts) and reclaims the right margin.
    """
    fig.update_layout(_DEFAULT_LAYOUT)
    fig.update_layout(
        title=dict(text=title, **_DEFAULT_LAYOUT["title"]),
        height=height,
        width=width,
    )
    if no_legend:
        fig.update_layout(
            margin=dict(r=50)
        )
    # ``update_layout`` only styles axis #1 (``xaxis``/``yaxis``); on a
    # make_subplots figure every other panel's axes keep plotly defaults, so
    # tick marks and gridlines render inconsistently from panel to panel.
    # Broadcast the shared axis theme to every subplot axis. Secondary
    # (overlay) y-axes keep their grid off so a dual-axis plot doesn't draw
    # two clashing horizontal grids.
    for axis in fig.select_xaxes():
        axis.update(_X_AXIS_STYLE)
    for axis in fig.select_yaxes():
        if axis.overlaying:
            axis.update({**_Y_AXIS_STYLE, "showgrid": False})
        else:
            axis.update(_Y_AXIS_STYLE)
    if font_bump:
        fig.update_layout(
            font=dict(size=_BASE_FONT_SIZE + font_bump),
            title=dict(
                text=title,
                **{
                    **_DEFAULT_LAYOUT["title"],
                    "font": dict(
                        family=_TITLE_FONT_FAMILY,
                        size=_TITLE_FONT_SIZE + font_bump,
                        color=_AXIS_COLOR,
                    ),
                },
            ),
            legend=dict(
                **{
                    **_DEFAULT_LAYOUT["legend"],
                    "font": dict(size=_LEGEND_FONT_SIZE + font_bump),
                },
            ),
        )
        # Apply to every axis (covers subplots too).
        fig.update_xaxes(
            tickfont=dict(size=_TICK_FONT_SIZE + font_bump),
            title=dict(font=dict(size=_AXIS_TITLE_FONT_SIZE + font_bump), standoff=8),
        )
        fig.update_yaxes(
            tickfont=dict(size=_TICK_FONT_SIZE + font_bump),
            title=dict(font=dict(size=_AXIS_TITLE_FONT_SIZE + font_bump), standoff=8),
        )
        # Subplot titles (annotations attached to the figure layout).
        for ann in fig.layout.annotations or ():
            new_size = (ann.font.size or _AXIS_TITLE_FONT_SIZE) + font_bump
            ann.font.size = new_size
    # Force every category tick on the bar's category axis. Plotly thins
    # categorical labels when the plot area is short (e.g. a 3-bar chart under
    # a tall legend), silently dropping middle labels; ``dtick=1`` pins them.
    bar_orients = {
        getattr(tr, "orientation", None) or "v"
        for tr in fig.data
        if getattr(tr, "type", None) == "bar"
    }
    if "h" in bar_orients:
        fig.update_yaxes(dtick=1, tick0=0)
    elif bar_orients:  # vertical bars → category axis is x
        fig.update_xaxes(dtick=1, tick0=0)

    if legend_top:
        # Horizontal legend strip stacked *below* the title and *above* the
        # plot, with the right margin reclaimed from the old vertical legend.
        # The top margin grows with the number of (wrapped) legend rows so a
        # many-series legend never collides with the title. Applied last so
        # it survives the font_bump legend reset above.
        legend_font = _LEGEND_FONT_SIZE + font_bump
        title_font = _TITLE_FONT_SIZE + font_bump
        # Cap the title font so the longest title line fits the (narrow) bar
        # figure width — a bumped 24pt title overflows a 720px canvas on the
        # longer titles (e.g. with a compliance subtitle line).
        longest_line = max(
            (_visible_len(seg) for seg in str(title).split("<br>")), default=1
        )
        fit_font = int(width * 0.92 / max(1, longest_line) / 0.52)
        title_font = max(13, min(title_font, fit_font))
        labels = [
            str(tr.name)
            for tr in fig.data
            if getattr(tr, "showlegend", None) is not False
            and getattr(tr, "name", None) not in (None, "")
        ]
        # Greedily pack legend entries left-to-right to mirror plotly's own
        # wrapping, so the reserved top margin matches the rows actually drawn.
        avail = max(1.0, width - 84)
        char_px = 0.56 * legend_font
        rows, cur = 1, 0.0
        for lab in labels:
            item_w = 34 + len(lab) * char_px + 18
            if cur > 0 and cur + item_w > avail:
                rows += 1
                cur = item_w
            else:
                cur += item_w
        # Titles may carry a second ``<br>`` subtitle line (e.g. the
        # compliance-rate note) — reserve a line per ``<br>`` (matching the
        # title's own top offset) so the top line is never clipped and the
        # legend clears the full title block.
        n_title_lines = 1 + str(title).count("<br>")
        # ``title_off`` is the gap from the container top to the title's first
        # line; plotly's container-ref title renders higher than yanchor="top"
        # nominally implies, so ~1.15·font is needed to avoid top clipping.
        title_off = title_font * 1.15
        title_px = int(title_off + n_title_lines * title_font * 1.25 + 6)
        row_px = legend_font + 12
        top_margin = int(title_px + rows * row_px + 18)
        # Grow the figure for multi-row legends rather than squeezing the
        # plot area (the passed height budgets for a single legend row).
        height = height + max(0, rows - 1) * row_px
        fig.update_layout(
            height=height,
            title=dict(
                text=title,
                font=dict(
                    family=_TITLE_FONT_FAMILY, size=title_font, color=_AXIS_COLOR
                ),
                x=0.5,
                xanchor="center",
                yref="container",
                yanchor="top",
                y=1.0 - title_off / height,
                pad=dict(t=0, b=0),
            ),
            legend=dict(
                orientation="h",
                xref="container",
                yref="container",
                yanchor="top",
                y=1.0 - (title_px + 4) / height,
                xanchor="left",
                x=84 / width,
                bgcolor="rgba(255,255,255,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=legend_font),
                itemsizing="constant",
                tracegroupgap=6,
            ),
            margin=dict(l=84, r=40, t=top_margin, b=64),
        )
    return fig


def _save(fig: go.Figure, out_path: Path) -> Path:
    """Write both HTML and PDF; return the directory-relative stem path
    (no suffix)."""
    out_path = Path(out_path).with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_path = out_path.with_suffix(".html")
    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displayModeBar": True,
            "responsive": True,
            "toImageButtonOptions": {"format": "pdf"},
        },
    )
    pdf_path = out_path.with_suffix(".pdf")
    try:
        fig.write_image(pdf_path, format="pdf")
    except Exception:
        # PDF is for static inclusion only; HTML is canonical, so a
        # Kaleido failure shouldn't kill the report.
        pass
    return out_path


def _empty_fig(message: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(family=_FONT_FAMILY, size=_ANNOTATION_FONT_SIZE, color=_MUTED_COLOR),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply_theme(fig, title=title)


def _compliance_subtitle(rate: float | None, n_compliant: int, n_total: int) -> str:
    """Build the ``"compliant runs: M/N (X%)"`` subtitle line. Empty
    when rate is None (no compliance column)."""
    if rate is None:
        return ""
    if n_total == 0:
        return ""
    return (
        f"<span style='font-size:11px;color:{_MUTED_COLOR}'>"
        f"compliant runs: {n_compliant}/{n_total} ({rate * 100:.0f}%)"
        f"</span>"
    )


def _variant_color(variant: str, fallback: str | None = None) -> str:
    return _VARIANT_COLOR.get(
        variant, fallback or _QUAL_PALETTE[hash(variant) % len(_QUAL_PALETTE)]
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


# Bar styling — shared across every bar figure


# Dark edge on every bar so adjacent fills separate cleanly in print.
_BAR_LINE_COLOR = "#2A2A2A"
_BAR_LINE_WIDTH = 0.8

# Colourblind-safe hatch shapes. Categorical multi-series bars overlay a
# distinct hatch per series on top of the (kept) hue, so series stay
# separable in greyscale and for colour-vision-deficient readers. An empty
# entry leaves the primary series solid.
_PATTERN_SHAPES = ("", "/", "\\", "x", "-", "+", "|", ".")

# Fixed hatch per variant / sector for cross-figure consistency. Sector
# hues are kept (see ``_SECTOR_COLOR``); the hatch is the redundant CVD
# channel layered on top. The primary "scare" variant stays solid.
_VARIANT_PATTERN = {
    "scare": "",
    "oracle": "/",
    "single_level": "\\",
    "component_level": "x",
}
_SECTOR_PATTERN = {
    "electricity": "",
    "gas": "/",
    "heat": "\\",
}
_CONSTRAINT_VARIABLE_PATTERN = {
    "voltage": "",
    "pressure": "/",
    "line_load": "\\",
    "slack": "+",
    "temperature": "x",
}


def _visible_len(s: str) -> int:
    """Character count of ``s`` ignoring HTML tags (so a ``<span>`` subtitle
    or ``<br>`` doesn't inflate the measured title width)."""
    out, depth = 0, 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out += 1
    return out


def _bar_pattern(shape: str | None, *, fg: str = "white") -> dict | None:
    """Build a subtle hatch ``marker.pattern`` for a bar, or ``None`` when
    ``shape`` is empty/None (solid fill)."""
    if not shape:
        return None
    return dict(shape=shape, solidity=0.30, size=7, fgcolor=fg, fgopacity=0.55)


def _bar_marker(
    color: str,
    *,
    pattern_shape: str | None = None,
    pattern_fg: str = "white",
) -> dict:
    """Bar marker with the shared dark outline and an optional colourblind
    hatch overlaid on the solid fill."""
    marker: dict[str, Any] = dict(
        color=color,
        line=dict(color=_BAR_LINE_COLOR, width=_BAR_LINE_WIDTH),
    )
    pattern = _bar_pattern(pattern_shape, fg=pattern_fg)
    if pattern is not None:
        marker["pattern"] = pattern
    return marker


# Compact figure footprint for bar charts (the full-width default is for
# trajectories). Horizontal bars size their height by category count so
# bars stay slim rather than ballooning to fill a fixed canvas.
_BAR_FIG_WIDTH = 720

# Box plots: fraction of the category slot a box occupies (keeps boxes slim
# regardless of canvas width).
_BOX_WIDTH = 0.5


def _box_fig_width(n_boxes: int) -> int:
    """Compact width for a box figure so a few boxes don't sprawl across the
    full canvas. Scales with box count, capped at the default width."""
    return int(max(340, 130 * max(1, n_boxes) + 100))


def _add_box(
    fig: go.Figure,
    values: Any,
    *,
    name: str,
    color: str,
    hovertemplate: str,
) -> None:
    """Add a consistently-styled box: tinted fill, coloured border, a dark
    hairline on the median/whiskers for crispness, dashed mean, and outliers
    flagged in red."""
    fig.add_trace(
        go.Box(
            y=values,
            name=name,
            width=_BOX_WIDTH,
            whiskerwidth=0.5,
            fillcolor=_hex_to_rgba(color, 0.22),
            line=dict(width=1.6, color=color),
            marker=dict(
                color=color,
                outliercolor="#D62728",
                size=4,
                line=dict(width=0.7, color="#D62728"),
            ),
            boxmean=True,
            boxpoints="outliers",
            hovertemplate=hovertemplate,
        )
    )


def _tier_ramp_color(tier: int, n_tiers: int) -> str:
    """Sequential dark-crimson → pale-grey luminance ramp across the tier
    range. Tiers are an *ordinal* series, so a single-hue lightness ramp is
    the CVD-safe encoding (distinguished by luminance, not hue) and avoids
    overlaying 10 illegible hatches on a grouped bar."""
    t = 0.0 if n_tiers <= 1 else (tier - 1) / (n_tiers - 1)
    r0, g0, b0 = 0x8B, 0x1A, 0x1A  # critical (tier 1)
    r1, g1, b1 = 0xCF, 0xCF, 0xCF  # low priority
    r = round(r0 + (r1 - r0) * t)
    g = round(g0 + (g1 - g0) * t)
    b = round(b0 + (b1 - b0) * t)
    return f"rgb({r}, {g}, {b})"


def _hbar_height(n_groups: int, n_series: int = 1) -> int:
    """Compact height for a horizontal bar figure: enough per-category room
    to keep bars slim, plus headroom for the title + top legend + x-axis."""
    if n_series <= 1:
        per_row = 30
    else:
        per_row = 15 * n_series + 10
    return int(min(640, max(230, per_row * n_groups + 150)))


# Display-only alias helpers: map canonical summary.csv names to the
# short figure labels in ``experiment/configs/display_aliases.json``.
from experiment.eval.aliases import (  # noqa: E402
    alias_grid,
    alias_variant,
)


def _grids_display(grids: list[str]) -> list[str]:
    return [alias_grid(g) for g in grids]


def _variants_display(variants: list[str]) -> list[str]:
    return [alias_variant(v) for v in variants]


# Variant comparison (Pillar 3)


def variant_comparison_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    metric_col: str = "outcomes__priority_weighted_fraction",
    title: str = "Priority-weighted served fraction by variant (compliant runs)",
) -> Path:
    if df.empty or metric_col not in df.columns:
        return _save(_empty_fig("no data", title), out_path)

    # Restrict the PWSF mean to compliant rows (see ``_compliant_mask``).
    # Compliance rate is reported per-(grid, variant) in hover + subtitle.
    compliant = df[_compliant_mask(df)]
    grouped_c = compliant.groupby(["grid", "variant"])[metric_col].apply(list)
    # Full counts so the hover can report ``n_compliant/n_total``.
    grouped_full = df.groupby(["grid", "variant"])[metric_col].apply(list)
    grids = sorted({k[0] for k in grouped_full.index})
    variants = sorted({k[1] for k in grouped_full.index})
    grids_lbl = _grids_display(grids)

    fig = go.Figure()
    for variant in variants:
        means: list[float] = []
        cis: list[float] = []
        n_c: list[int] = []
        n_t: list[int] = []
        for grid in grids:
            vals_c = grouped_c.get((grid, variant), [])
            vals_t = grouped_full.get((grid, variant), [])
            mean, ci = _mean_ci(vals_c)
            means.append(mean)
            cis.append(ci)
            n_c.append(len(vals_c))
            n_t.append(len(vals_t))
        hover = [
            f"<b>{alias_variant(variant)}</b><br>grid: {g}<br>"
            f"mean PWSF (compliant): {m:.4f}<br>95% CI: ±{c:.4f}<br>"
            f"compliant: {nc}/{nt}" + (f" ({nc / nt * 100:.0f}%)" if nt else "")
            for g, m, c, nc, nt in zip(grids, means, cis, n_c, n_t)
        ]
        fig.add_trace(
            go.Bar(
                name=alias_variant(variant),
                # Horizontal: long grid labels read cleanly down the y-axis
                # and variants compare left-to-right within each grid block.
                y=grids_lbl,
                x=means,
                orientation="h",
                error_x=dict(
                    type="data",
                    array=cis,
                    visible=True,
                    thickness=1.2,
                    width=4,
                    color=_MUTED_COLOR,
                ),
                marker=_bar_marker(
                    _variant_color(variant),
                    pattern_shape=_VARIANT_PATTERN.get(variant, ""),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            )
        )

    rate = _compliance_rate(df)
    if rate is not None:
        n_c_total = int(_compliant_mask(df).sum())
        title = f"{title}<br>{_compliance_subtitle(rate, n_c_total, len(df))}"

    fig.update_layout(barmode="group", 
                      bargap=0.32, 
                      bargroupgap=0.12
    )
    fig.update_xaxes(
        range=[0, 1.05], title="priority-weighted served fraction", tickformat=".2f"
    )
    fig.update_yaxes(title="grid")
    height = _hbar_height(len(grids), len(variants))
    return _save(
        _apply_theme(
            fig,
            title=title,
            height=height,
            width=_BAR_FIG_WIDTH,
            font_bump=2,
            legend_top=True,
        ),
        out_path,
    )


# Optimality gap (Pillar 2)


def optimality_gap_scatter(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Optimality gap: scare vs centralised oracle (compliant pairs)"
    if df.empty:
        return _save(_empty_fig("no data", title), out_path)
    metric = "outcomes__priority_weighted_fraction"
    # Keep only rows where both variants are compliant — else an
    # over-drawing scare reports a misleading "negative gap" against a
    # compliant oracle.
    df = df[_compliant_mask(df)]
    pivot = df.pivot_table(index=["grid", "seed"], columns="variant", values=metric)
    if "scare" not in pivot.columns or "oracle" not in pivot.columns:
        return _save(
            _empty_fig("need both 'scare' and 'oracle' variants", title), out_path
        )
    pivot = pivot.dropna(subset=["scare", "oracle"], how="any")
    if pivot.empty:
        return _save(_empty_fig("no compliant scare/oracle pairs", title), out_path)

    fig = go.Figure()
    # Parity line first so points draw on top.
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#BBBBBB", dash="dash", width=1),
            name="parity",
            hoverinfo="skip",
        )
    )
    grids = sorted(pivot.index.get_level_values("grid").unique())
    for i, grid in enumerate(grids):
        sub = pivot.xs(grid, level="grid")
        seeds = list(sub.index)
        fig.add_trace(
            go.Scatter(
                x=sub["oracle"],
                y=sub["scare"],
                mode="markers",
                name=alias_grid(grid),
                marker=dict(
                    size=9,
                    color=_QUAL_PALETTE[i % len(_QUAL_PALETTE)],
                    line=dict(width=1, color="white"),
                    opacity=0.9,
                ),
                customdata=[
                    f"grid: {alias_grid(grid)}<br>seed: {s}<br>oracle: {sub.loc[s, 'oracle']:.4f}<br>scare: {sub.loc[s, 'scare']:.4f}<br>gap: {(sub.loc[s, 'oracle'] - sub.loc[s, 'scare']):.4f}"
                    for s in seeds
                ],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    # Mean gap annotation.
    pivot["gap"] = pivot["oracle"] - pivot["scare"]
    mean_gap = float(pivot["gap"].mean())
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.97,
        y=0.05,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#CCCCCC",
        borderwidth=0.6,
        borderpad=5,
        text=f"<b>mean gap</b> {mean_gap:+.4f}  ·  <b>n</b>={len(pivot)}",
        font=dict(family=_FONT_FAMILY, size=_ANNOTATION_FONT_SIZE, color=_AXIS_COLOR),
    )

    fig.update_xaxes(
        title="oracle priority-weighted served", range=[0, 1.05], tickformat=".2f"
    )
    fig.update_yaxes(
        title="scare priority-weighted served", range=[0, 1.05], tickformat=".2f"
    )
    return _save(_apply_theme(fig, title=title, font_bump=4), out_path)


def optimality_gap_box(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Optimality gap by grid (compliant pairs)"
    if df.empty:
        return _save(_empty_fig("no data", title), out_path)
    metric = "outcomes__priority_weighted_fraction"
    # Same compliance filter as ``optimality_gap_scatter``.
    df = df[_compliant_mask(df)]
    pivot = df.pivot_table(index=["grid", "seed"], columns="variant", values=metric)
    if "scare" not in pivot.columns or "oracle" not in pivot.columns:
        return _save(
            _empty_fig("need both 'scare' and 'oracle' variants", title), out_path
        )
    pivot = pivot.dropna(subset=["scare", "oracle"], how="any")
    if pivot.empty:
        return _save(_empty_fig("no compliant scare/oracle pairs", title), out_path)
    pivot["gap"] = (pivot["oracle"] - pivot["scare"]) / pivot["oracle"].replace(
        0, np.nan
    )
    pivot = pivot.dropna(subset=["gap"])
    if pivot.empty:
        return _save(_empty_fig("no positive-oracle rows", title), out_path)

    fig = go.Figure()
    grids = sorted(pivot.index.get_level_values("grid").unique())
    for i, grid in enumerate(grids):
        gaps = pivot.xs(grid, level="grid")["gap"].values
        _add_box(
            fig,
            gaps,
            name=alias_grid(grid),
            color=_QUAL_PALETTE[i % len(_QUAL_PALETTE)],
            hovertemplate="grid: %{x}<br>gap: %{y:.4f}<extra></extra>",
        )
    fig.add_hline(y=0, line=dict(color="#BBBBBB", dash="dash", width=1))
    fig.update_yaxes(
        title="relative gap (oracle − scare) / oracle", tickformat=".2f", zeroline=False
    )
    fig.update_xaxes(title="grid")
    fig.update_layout(showlegend=False, boxgap=0.45, boxgroupgap=0.2)
    return _save(
        _apply_theme(
            fig, title=title, height=380, width=_box_fig_width(len(grids)), no_legend=True
        ),
        out_path,
    )


# Ablation impact (Pillar 4)


def ablation_impact_bar(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Ablation impact (scare variant, compliant runs)"
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    # Pre-filter counts so the hover can show ``n_compliant/n_total``
    # and flag ablations that break budget compliance.
    full_counts = df.groupby("ablation").size()
    df_c = df[_compliant_mask(df)]
    grouped = df_c.groupby("ablation")[metric].agg(
        mean="mean", count="count", std="std"
    )
    if grouped.empty:
        return _save(_empty_fig("no compliant ablation rows", title), out_path)
    grouped["ci"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"]) * 1.96
    grouped = grouped.sort_values("mean", ascending=True)

    baseline_mean = (
        grouped.loc["default", "mean"] if "default" in grouped.index else None
    )
    colors = [
        "#7F7F7F" if k == "default" else _VARIANT_COLOR["scare"] for k in grouped.index
    ]

    hover = []
    for k, m, c, n in zip(
        grouped.index, grouped["mean"], grouped["ci"], grouped["count"]
    ):
        n_total = int(full_counts.get(k, n))
        rate_str = f" ({n / n_total * 100:.0f}%)" if n_total else ""
        base = (
            f"<b>{k}</b><br>mean PWSF (compliant): {m:.4f}<br>"
            f"95% CI: ±{c:.4f}<br>compliant: {n}/{n_total}{rate_str}"
        )
        if baseline_mean is not None:
            base += f"<br>Δ vs default: {(m - baseline_mean):+.4f}"
        hover.append(base)

    fig = go.Figure(
        go.Bar(
            x=grouped["mean"],
            y=grouped.index,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color=_BAR_LINE_COLOR, width=_BAR_LINE_WIDTH),
            ),
            error_x=dict(
                type="data",
                array=grouped["ci"],
                thickness=1.2,
                width=4,
                color=_MUTED_COLOR,
            ),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    if baseline_mean is not None:
        fig.add_vline(
            x=baseline_mean,
            line=dict(color="#701E96", dash="dash", width=1.1),
            annotation_text="baseline",
            annotation_position="top",
            annotation_font=dict(color="#701E96", size=_ANNOTATION_FONT_SIZE),
        )
    fig.update_xaxes(
        title="mean priority-weighted served fraction",
        range=[0, 1.05],
        tickformat=".2f",
    )
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False, bargap=0.45)
    height = _hbar_height(len(grouped))
    fig = _apply_theme(fig, title=title, height=height, width=_BAR_FIG_WIDTH)
    fig.update_layout(margin=dict(l=84, r=48, t=72, b=64))
    return _save(fig, out_path)


# Robustness curves (Pillar 5)


def _compliant_share_by_x(parsed: pd.DataFrame) -> pd.Series | None:
    """Per-``x`` fraction of runs passing the compliance gate (sorted by x),
    or ``None`` when no compliance column is present so callers can suppress
    the second curve."""
    if _compliance_rate(parsed) is None:
        return None
    return _compliant_mask(parsed).groupby(parsed["x"]).mean().sort_index()


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

    # PWSF over the compliant subset, per sweep point. A sweep value
    # that pushes the variant out of compliance still shows its
    # compliance rate even when its mean PWSF is empty.
    full_counts = parsed.groupby("x").size()
    parsed_c = parsed[_compliant_mask(parsed)]
    grouped = parsed_c.groupby("x")[metric].agg(["mean", "std", "count"])
    grouped["ci"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"]) * 1.96
    grouped = grouped.sort_index()
    if grouped.empty:
        return _save(_empty_fig(f"no compliant {sweep_param} rows", title), out_path)

    x = grouped.index.tolist()
    y = grouped["mean"].tolist()
    upper = (grouped["mean"] + grouped["ci"]).tolist()
    lower = (grouped["mean"] - grouped["ci"]).tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # CI ribbon.
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=_hex_to_rgba(_VARIANT_COLOR["scare"], 0.16),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=_VARIANT_COLOR["scare"], width=2.0),
            marker=dict(
                size=8, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1)
            ),
            customdata=[
                f"x={xv}<br>mean PWSF (compliant): {m:.4f}<br>95% CI: ±{c:.4f}<br>"
                f"compliant: {int(n)}/{int(full_counts.get(xv, n))}"
                for xv, m, c, n in zip(
                    x, grouped["mean"], grouped["ci"], grouped["count"]
                )
            ],
            hovertemplate="%{customdata}<extra></extra>",
            name="served (compliant)",
        ),
        secondary_y=False,
    )

    # Share of compliant runs per sweep point — right-hand axis. A sweep
    # value that pushes the variant out of compliance shows here even where
    # the (compliant-only) PWSF line drops out for lack of compliant rows.
    share = _compliant_share_by_x(parsed)
    if share is not None:
        fig.add_trace(
            go.Scatter(
                x=share.index.tolist(),
                y=(share.values * 100).tolist(),
                mode="lines+markers",
                name="compliant runs",
                line=dict(color=_VARIANT_COLOR["oracle"], width=2.0, dash="dot"),
                marker=dict(
                    size=8,
                    color=_VARIANT_COLOR["oracle"],
                    symbol="square",
                    line=dict(color="white", width=1),
                ),
                hovertemplate=(
                    f"{sweep_param}: %{{x}}<br>compliant runs: %{{y:.0f}}%"
                    "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title="compliant runs (%)",
            range=[0, 105],
            color=_VARIANT_COLOR["oracle"],
            secondary_y=True,
            showgrid=False,
            ticksuffix="%",
            tickfont=dict(size=_TICK_FONT_SIZE),
            title_font=dict(size=_AXIS_TITLE_FONT_SIZE),
        )

    rate = _compliance_rate(parsed)
    if rate is not None:
        n_c_total = int(_compliant_mask(parsed).sum())
        title = f"{title}<br>{_compliance_subtitle(rate, n_c_total, len(parsed))}"
    fig.update_yaxes(
        title="priority-weighted served fraction (compliant)",
        range=[0, 1.05],
        color=_VARIANT_COLOR["scare"],
        secondary_y=False,
        tickformat=".2f",
    )
    fig.update_xaxes(title=x_label)
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


# Cascading (Pillar 6)


def cascading_curve(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Restoration quality vs simultaneous failure count (compliant runs)"
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns or "n_failures" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)

    # PWSF over the compliant subset per failure count, so the cliff at
    # high n_failures reflects lost compliance, not slack overdraw.
    full_counts = df.groupby("n_failures").size()
    df_c = df[_compliant_mask(df)]
    grouped = df_c.groupby("n_failures")[metric].agg(["mean", "std", "count"])
    grouped["ci"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"]) * 1.96
    grouped = grouped.sort_index()
    if grouped.empty:
        return _save(_empty_fig("no compliant rows", title), out_path)

    x = grouped.index.tolist()
    y = grouped["mean"].tolist()
    upper = (grouped["mean"] + grouped["ci"]).tolist()
    lower = (grouped["mean"] - grouped["ci"]).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=_hex_to_rgba(_VARIANT_COLOR["scare"], 0.16),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=_VARIANT_COLOR["scare"], width=2.0),
            marker=dict(
                size=9, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1)
            ),
            customdata=[
                f"failures: {xv}<br>mean PWSF (compliant): {m:.4f}<br>"
                f"95% CI: ±{c:.4f}<br>compliant: {int(n)}/{int(full_counts.get(xv, n))}"
                for xv, m, c, n in zip(
                    x, grouped["mean"], grouped["ci"], grouped["count"]
                )
            ],
            hovertemplate="%{customdata}<extra></extra>",
            name="scare",
        )
    )
    rate = _compliance_rate(df)
    if rate is not None:
        n_c_total = int(_compliant_mask(df).sum())
        title = f"{title}<br>{_compliance_subtitle(rate, n_c_total, len(df))}"
    fig.update_yaxes(
        title="priority-weighted served fraction (compliant)",
        range=[0, 1.05],
        tickformat=".2f",
    )
    fig.update_xaxes(title="number of simultaneous failures", dtick=1)
    fig.update_layout(showlegend=False)
    return _save(_apply_theme(fig, title=title), out_path)


# Sensitivity sweeps (Pillar 7)


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

    # Served axis uses the compliant subset; wallclock axis uses all
    # rows (timing is a system property, not metric-validity).
    parsed_c = parsed[_compliant_mask(parsed)]
    if parsed_c.empty:
        return _save(_empty_fig(f"no compliant {sweep_param} rows", title), out_path)
    served = parsed_c.groupby("x")[metric].mean().sort_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=served.index,
            y=served.values,
            mode="lines+markers",
            name="served (compliant)",
            line=dict(color=_VARIANT_COLOR["scare"], width=2.0),
            marker=dict(
                size=8, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1)
            ),
            hovertemplate=f"{sweep_param}: %{{x}}<br>served (compliant): %{{y:.4f}}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Share of compliant runs per sweep point — shares the left (fraction)
    # axis with the served line, since the right axis already carries
    # wallclock.
    share = _compliant_share_by_x(parsed)
    if share is not None:
        fig.add_trace(
            go.Scatter(
                x=share.index.tolist(),
                y=share.values.tolist(),
                mode="lines+markers",
                name="compliant share",
                line=dict(color=_VARIANT_COLOR["oracle"], width=2.0, dash="dot"),
                marker=dict(
                    size=8,
                    color=_VARIANT_COLOR["oracle"],
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                hovertemplate=f"{sweep_param}: %{{x}}<br>compliant share: %{{y:.0%}}<extra></extra>",
            ),
            secondary_y=False,
        )

    if "duration_s" in parsed.columns:
        wall = parsed.groupby("x")["duration_s"].mean().sort_index()
        fig.add_trace(
            go.Scatter(
                x=wall.index,
                y=wall.values,
                mode="lines+markers",
                name="wallclock (s)",
                line=dict(color=_VARIANT_COLOR["single_level"], width=2.0, dash="dot"),
                marker=dict(
                    size=8,
                    color=_VARIANT_COLOR["single_level"],
                    symbol="square",
                    line=dict(color="white", width=1),
                ),
                hovertemplate=f"{sweep_param}: %{{x}}<br>wallclock: %{{y:.0f}}s<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title="wallclock (s)",
            color=_VARIANT_COLOR["single_level"],
            secondary_y=True,
            showgrid=False,
            tickfont=dict(size=_TICK_FONT_SIZE),
            title_font=dict(size=_AXIS_TITLE_FONT_SIZE),
        )

    fig.update_yaxes(
        title="priority-weighted served fraction (compliant)",
        range=[0, 1.05],
        color=_VARIANT_COLOR["scare"],
        secondary_y=False,
        tickformat=".2f",
    )
    fig.update_xaxes(title=x_label)
    rate = _compliance_rate(parsed)
    if rate is not None:
        n_c_total = int(_compliant_mask(parsed).sum())
        title = f"{title}<br>{_compliance_subtitle(rate, n_c_total, len(parsed))}"
        # Re-apply so the updated subtitle shows.
        fig.update_layout(title=dict(text=title))
    return _save(_apply_theme(fig, title=title), out_path)


# Per-tier served (Pillars 1, 4)


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

    # Kept vertical: priority tier is an ordinal axis that reads naturally
    # left-to-right (1 = most critical). Sectors keep their hues + a hatch.
    fig = go.Figure()
    for sec in pivot.columns:
        fig.add_trace(
            go.Bar(
                x=pivot.index.astype(str),
                y=pivot[sec].values,
                name=sec,
                marker=_bar_marker(
                    _SECTOR_COLOR.get(sec, "#888888"),
                    pattern_shape=_SECTOR_PATTERN.get(sec, ""),
                ),
                hovertemplate=f"<b>{sec}</b><br>tier: %{{x}}<br>served: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(title="priority tier (1 = most critical)")
    fig.update_yaxes(title="served fraction", range=[0, 1.05], tickformat=".2f")
    return _save(
        _apply_theme(
            fig, title=title, height=360, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


# Restoration trajectory (per task)


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
            fig.add_trace(
                go.Scatter(
                    x=timeseries["time_s"],
                    y=timeseries[col],
                    mode="lines",
                    name=sec,
                    line=dict(color=_SECTOR_COLOR[sec], width=2),
                    hovertemplate=f"<b>{sec}</b><br>t: %{{x:.2f}}s<br>balance: %{{y:.4f}}<extra></extra>",
                )
            )

    # Event markers — distinct colours/dashes per kind.
    if not events.empty and {"t", "kind"}.issubset(events.columns):
        event_styles = {
            "line_failure": dict(color="#1A1A1A", dash="dash"),
            "reconfiguration_completed": dict(color="#9467BD", dash="dot"),
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
            # Sentinel scatter so the kind appears in the legend.
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=style["color"], dash=style["dash"], width=2),
                    name=kind,
                    showlegend=True,
                )
            )
    elif failure_t is not None:
        fig.add_vline(x=failure_t, line=dict(color="#1A1A1A", dash="dash", width=1))

    fig.update_xaxes(title="simulation time (s)")
    fig.update_yaxes(title="Σ regulation per sector")
    return _save(_apply_theme(fig, title=title, height=360), out_path)


# Claims pass-rate (Pillar 8)


def claims_pass_rate(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Claims validation pass rate by variant"
    if df.empty:
        return _save(_empty_fig("no data", title), out_path)
    claim_cols = [
        c for c in df.columns if c.startswith("claims__") and c.endswith("__passed")
    ]
    if not claim_cols:
        return _save(_empty_fig("no claims data", title), out_path)

    rows = []
    for col in claim_cols:
        claim_name = col[len("claims__") : -len("__passed")]
        for variant, g in df.groupby("variant"):
            n = g[col].dropna().shape[0]
            if n == 0:
                continue
            rate = float(g[col].astype(bool).sum()) / n
            rows.append((claim_name, str(variant), rate, n))
    if not rows:
        return _save(_empty_fig("no claims data", title), out_path)

    rdf = pd.DataFrame(rows, columns=["claim", "variant", "pass_rate", "n"])
    pivot = rdf.pivot(index="claim", columns="variant", values="pass_rate").fillna(
        np.nan
    )
    n_pivot = rdf.pivot(index="claim", columns="variant", values="n").fillna(0)

    fig = go.Figure()
    for variant in pivot.columns:
        fig.add_trace(
            go.Bar(
                y=pivot.index,
                x=pivot[variant].values,
                name=alias_variant(variant),
                orientation="h",
                marker=_bar_marker(
                    _variant_color(variant),
                    pattern_shape=_VARIANT_PATTERN.get(variant, ""),
                ),
                customdata=[
                    f"<b>{c}</b><br>variant: {alias_variant(variant)}<br>pass rate: {p:.1%}<br>n: {int(n_pivot.loc[c, variant])}"
                    for c, p in zip(pivot.index, pivot[variant].values)
                ],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.34, bargroupgap=0.12)
    fig.update_xaxes(title="pass rate", range=[0, 1.05], tickformat=".0%")
    fig.update_yaxes(title="")
    height = _hbar_height(len(pivot), len(pivot.columns))
    return _save(
        _apply_theme(
            fig,
            title=title,
            height=height,
            width=_BAR_FIG_WIDTH,
            font_bump=2,
            legend_top=True,
        ),
        out_path,
    )


# Diary distribution (Pillars 1, 8)


# Restoration vs baseline (raw load + per-tier)


def restoration_vs_baseline_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Absolute restoration vs no-failure baseline",
) -> Path:
    """Per-grid grouped bars: pre-failure (baseline) vs post-restoration
    served, in raw MW (unweighted). PWSF ratio overlaid so absolute loss
    can be compared against the priority-weighted view. Reads the
    ``total_served_baseline_mw`` / ``total_served_post_mw`` columns;
    empty figure if absent.
    """
    base_col = "outcomes__restoration__total_served_baseline_mw"
    post_col = "outcomes__restoration__total_served_post_mw"
    raw_col = "outcomes__restoration__raw_restoration_ratio"
    pwsf_col = "outcomes__restoration__pwsf_restoration_ratio"
    if df.empty or base_col not in df.columns or post_col not in df.columns:
        return _save(
            _empty_fig("no restoration data — re-run campaign", title), out_path
        )

    sub = df[df["variant"] == "scare"] if "scare" in df["variant"].unique() else df
    sub = sub.dropna(subset=[base_col, post_col])
    if sub.empty:
        return _save(_empty_fig("no scare baseline rows", title), out_path)

    grouped = (
        sub.groupby("grid")
        .agg(
            baseline_mw=(base_col, "mean"),
            post_mw=(post_col, "mean"),
            raw_ratio=(raw_col, "mean")
            if raw_col in sub.columns
            else (post_col, "mean"),
            pwsf_ratio=(pwsf_col, "mean")
            if pwsf_col in sub.columns
            else (post_col, "mean"),
            n=(base_col, "count"),
        )
        .sort_values("baseline_mw", ascending=False)
    )
    grids = grouped.index.tolist()
    grids_lbl = _grids_display(grids)

    # Kept vertical: the overlaid restoration-ratio markers ride a secondary
    # y-axis, which only exists in the vertical orientation.
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=grids_lbl,
            y=grouped["baseline_mw"].values,
            name="baseline (no failure)",
            # Hatched so the baseline overhang behind the post bar still
            # reads as the reference envelope.
            marker=_bar_marker("#BFBFBF", pattern_shape="/", pattern_fg="#7F7F7F"),
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>baseline: %{y:.3f} MW<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=grids_lbl,
            y=grouped["post_mw"].values,
            name="post-restoration",
            marker=_bar_marker(_VARIANT_COLOR["scare"]),
            hovertemplate="<b>%{x}</b><br>post: %{y:.3f} MW<extra></extra>",
        ),
        secondary_y=False,
    )
    if raw_col in sub.columns:
        fig.add_trace(
            go.Scatter(
                x=grids_lbl,
                y=grouped["raw_ratio"].values,
                mode="markers",
                name="raw ratio",
                marker=dict(
                    color=_AXIS_COLOR,
                    size=9,
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                hovertemplate="<b>%{x}</b><br>raw ratio: %{y:.3f}<extra></extra>",
            ),
            secondary_y=True,
        )
    if pwsf_col in sub.columns:
        fig.add_trace(
            go.Scatter(
                x=grids_lbl,
                y=grouped["pwsf_ratio"].values,
                mode="markers",
                name="PWSF ratio",
                marker=dict(
                    color="#D62728",
                    size=9,
                    symbol="triangle-up",
                    line=dict(color="white", width=1),
                ),
                hovertemplate="<b>%{x}</b><br>PWSF ratio: %{y:.3f}<extra></extra>",
            ),
            secondary_y=True,
        )
    fig.add_hline(
        y=1.0, line=dict(color="#BBBBBB", dash="dash", width=1), secondary_y=True
    )

    fig.update_layout(barmode="overlay", bargap=0.34)
    fig.update_yaxes(
        title="served (MW, unweighted)",
        secondary_y=False,
        rangemode="tozero",
        tickformat=".2f",
    )
    fig.update_yaxes(
        title="restoration ratio",
        secondary_y=True,
        range=[0, 1.05],
        showgrid=False,
        tickformat=".2f",
        tickfont=dict(size=_TICK_FONT_SIZE),
        title_font=dict(size=_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_xaxes(title="grid")
    fig = _apply_theme(
        fig, title=title, height=360, width=_BAR_FIG_WIDTH, legend_top=True
    )
    # Restore right margin so the secondary-axis title/ticks aren't clipped
    # (legend_top otherwise reclaims it). Only ``r`` is overridden — the
    # computed top margin for the legend strip is preserved.
    fig.update_layout(margin_r=78)
    return _save(fig, out_path)


def restoration_by_tier_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Per-tier restoration ratio (post / baseline served, MW)",
) -> Path:
    """Grouped bars per grid × tier: fraction of each tier's pre-failure
    served load that survived restoration. Tier 1 (most critical) sitting
    below 1.0 means the protocol is failing the loads that matter most.
    """
    if df.empty or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    sub = df[df["variant"] == "scare"] if "scare" in df["variant"].unique() else df

    tier_cols: dict[int, str] = {}
    for col in sub.columns:
        m = col
        prefix = "outcomes__restoration__by_tier__"
        suffix = "__ratio"
        if m.startswith(prefix) and m.endswith(suffix):
            try:
                tier = int(m[len(prefix) : -len(suffix)])
            except ValueError:
                continue
            tier_cols[tier] = m
    if not tier_cols:
        return _save(_empty_fig("no per-tier restoration data", title), out_path)

    tiers = sorted(tier_cols)
    grouped = sub.groupby("grid")[[tier_cols[t] for t in tiers]].mean()
    if grouped.empty:
        return _save(_empty_fig("empty per-tier table", title), out_path)
    grids = grouped.index.tolist()
    grids_lbl = _grids_display(grids)

    # Kept vertical: a 10-tier grouped bar laid horizontally would stack 10
    # bars per grid block and run very tall. Tiers are encoded by a CVD-safe
    # luminance ramp (see ``_tier_ramp_color``) instead of hatches.
    n_tiers = len(tiers)
    fig = go.Figure()
    for tier in tiers:
        col = tier_cols[tier]
        fig.add_trace(
            go.Bar(
                name=f"tier {tier}",
                x=grids_lbl,
                y=grouped[col].values,
                marker=_bar_marker(_tier_ramp_color(tier, n_tiers)),
                hovertemplate=f"<b>tier {tier}</b><br>grid: %{{x}}<br>ratio: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line=dict(color="#BBBBBB", dash="dash", width=1))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(
        title="restoration ratio (post / baseline served)",
        range=[0, 1.05],
        tickformat=".2f",
    )
    fig.update_xaxes(title="grid")
    return _save(
        _apply_theme(
            fig, title=title, height=360, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


def restoration_loss_split_by_tier_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Per-tier loss split: physical disconnect vs agent-shed (MW)",
) -> Path:
    """Stacked bars per tier attributing each tier's restoration loss:

    - ``disconnect_lost`` (priority-blind): demand whose node had no
      active path to a grid-forming source — physics, unsavable by agents.
    - ``agent_shed`` (priority-aware): load the QP / ADMM layers chose to
      drop — the only contribution priority weighting controls.

    The "tier 1 protected, tier 10 sheds first" claim applies to
    ``agent_shed`` only: a tier-1 bar dominated by ``disconnect_lost`` is
    a topology limit, not a priority-machinery failure.
    """
    if df.empty or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    sub = df[df["variant"] == "scare"] if "scare" in df["variant"].unique() else df

    # Discover tiers from the per-tier disconnect columns.
    disc_pat = "outcomes__restoration__by_tier__"
    disc_suf = "__disconnect_lost_mw"
    agt_suf = "__agent_shed_mw"
    tiers: list[int] = []
    for col in sub.columns:
        if col.startswith(disc_pat) and col.endswith(disc_suf):
            try:
                tiers.append(int(col[len(disc_pat) : -len(disc_suf)]))
            except ValueError:
                continue
    tiers = sorted(set(tiers))
    if not tiers:
        return _save(
            _empty_fig(
                "no disconnect/agent split data — re-run with metric update", title
            ),
            out_path,
        )

    disc_means: list[float] = []
    agt_means: list[float] = []
    for t in tiers:
        d_col = f"{disc_pat}{t}{disc_suf}"
        a_col = f"{disc_pat}{t}{agt_suf}"
        d = sub[d_col].dropna() if d_col in sub.columns else pd.Series(dtype=float)
        a = sub[a_col].dropna() if a_col in sub.columns else pd.Series(dtype=float)
        disc_means.append(float(d.mean()) if len(d) else 0.0)
        agt_means.append(float(a.mean()) if len(a) else 0.0)

    tier_labels = [f"tier {t}" for t in tiers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="physical disconnect (priority-blind)",
            y=tier_labels,
            x=disc_means,
            orientation="h",
            # Hatch the priority-blind loss so it reads apart from the
            # agent-controlled share in greyscale too.
            marker=_bar_marker("#999999", pattern_shape="x"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "disconnect lost (mean per task): %{x:.3f} MW<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            name="agent-shed (priority-aware)",
            y=tier_labels,
            x=agt_means,
            orientation="h",
            marker=_bar_marker(_VARIANT_COLOR.get("scare", "#1F4E96")),
            hovertemplate=(
                "<b>%{y}</b><br>agent shed (mean per task): %{x:.3f} MW<extra></extra>"
            ),
        )
    )
    fig.update_layout(barmode="stack", bargap=0.36)
    fig.update_xaxes(
        title="loss per task (MW, mean over scare tasks)",
        rangemode="tozero",
        tickformat=".3f",
    )
    # Tier 1 (most critical) on top.
    fig.update_yaxes(title="priority tier (1 = most critical)", autorange="reversed")
    height = _hbar_height(len(tier_labels))
    return _save(
        _apply_theme(
            fig, title=title, height=height, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


def agent_only_ratio_by_tier_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Per-tier restoration ratio (agent-shed only — disconnect excluded)",
) -> Path:
    """Per-tier bars of ``agent_only_ratio`` — the share each tier kept
    after removing physically-disconnected load from the denominator.
    Isolates the priority signal from topology noise: if the claim holds,
    tier 1 sits near 1.0 and the curve slopes down monotonically.
    """
    if df.empty or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    sub = df[df["variant"] == "scare"] if "scare" in df["variant"].unique() else df

    pat = "outcomes__restoration__by_tier__"
    suf = "__agent_only_ratio"
    tier_cols: dict[int, str] = {}
    for col in sub.columns:
        if col.startswith(pat) and col.endswith(suf):
            try:
                tier_cols[int(col[len(pat) : -len(suf)])] = col
            except ValueError:
                continue
    if not tier_cols:
        return _save(
            _empty_fig("no agent_only_ratio data — re-run with metric update", title),
            out_path,
        )
    tiers = sorted(tier_cols)
    grouped = sub.groupby("grid")[[tier_cols[t] for t in tiers]].mean()
    if grouped.empty:
        return _save(_empty_fig("empty per-tier table", title), out_path)
    grids = grouped.index.tolist()
    grids_lbl = _grids_display(grids)

    # Vertical for the same reason as ``restoration_by_tier_bar``; CVD-safe
    # tier luminance ramp instead of per-series hatches.
    n_tiers = len(tiers)
    fig = go.Figure()
    for tier in tiers:
        col = tier_cols[tier]
        fig.add_trace(
            go.Bar(
                name=f"tier {tier}",
                x=grids_lbl,
                y=grouped[col].values,
                marker=_bar_marker(_tier_ramp_color(tier, n_tiers)),
                hovertemplate=f"<b>tier {tier}</b><br>grid: %{{x}}<br>agent-only ratio: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line=dict(color="#BBBBBB", dash="dash", width=1))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(
        title="agent-only ratio",
        range=[0, 1.05],
        tickformat=".2f",
    )
    fig.update_xaxes(title="grid")
    return _save(
        _apply_theme(
            fig, title=title, height=360, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


def restoration_ratio_by_variant_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Raw restoration ratio (post / baseline) by grid × variant",
) -> Path:
    """Per-grid grouped bars comparing every variant on
    raw_restoration_ratio against the no-failure baseline, in one figure
    (``restoration_vs_baseline_bar`` plots one variant;
    ``optimality_gap_scatter`` only scare vs oracle).
    """
    col = "outcomes__restoration__raw_restoration_ratio"
    if df.empty or col not in df.columns or "variant" not in df.columns:
        return _save(_empty_fig("no restoration ratio data", title), out_path)
    sub = df.dropna(subset=[col])
    if sub.empty:
        return _save(_empty_fig("no restoration ratio data", title), out_path)

    grouped = sub.groupby(["grid", "variant"])[col].mean().unstack("variant")
    if grouped.empty:
        return _save(_empty_fig("empty grouping", title), out_path)

    grids = grouped.index.tolist()
    grids_lbl = _grids_display(grids)
    variants = sorted(grouped.columns)

    fig = go.Figure()
    for variant in variants:
        ys = grouped[variant].values
        fig.add_trace(
            go.Bar(
                y=grids_lbl,
                x=ys,
                orientation="h",
                name=alias_variant(variant),
                marker=_bar_marker(
                    _variant_color(variant),
                    pattern_shape=_VARIANT_PATTERN.get(variant, ""),
                ),
                hovertemplate=(
                    f"<b>{alias_variant(variant)}</b>"
                    "<br>grid: %{y}<br>raw ratio: %{x:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=1.0, line=dict(color="#BBBBBB", dash="dash", width=1))
    fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.12)
    fig.update_xaxes(title="raw restoration ratio", range=[0, 1.05], tickformat=".2f")
    fig.update_yaxes(title="grid")
    height = _hbar_height(len(grids), len(variants))
    return _save(
        _apply_theme(
            fig, title=title, height=height, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


def absolute_load_lost_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Absolute load lost despite restoration (MW, unweighted)",
) -> Path:
    """Per-grid ``absolute_load_dropped_mw`` — served load lost vs the
    no-failure baseline. The unweighted shortfall shows whether a high
    PWSF hides a large absolute loss on low-priority loads.
    """
    col = "outcomes__restoration__absolute_load_dropped_mw"
    base_col = "outcomes__restoration__total_served_baseline_mw"
    if df.empty or col not in df.columns:
        return _save(
            _empty_fig("no restoration data — re-run campaign", title), out_path
        )

    sub = df[df["variant"] == "scare"] if "scare" in df["variant"].unique() else df
    sub = sub.dropna(subset=[col])
    if sub.empty:
        return _save(_empty_fig("no scare rows", title), out_path)

    grouped = (
        sub.groupby("grid")
        .agg(
            dropped_mw=(col, "mean"),
            baseline_mw=(base_col, "mean") if base_col in sub.columns else (col, "max"),
            n=(col, "count"),
        )
        .sort_values("dropped_mw", ascending=False)
    )

    pct = (grouped["dropped_mw"] / grouped["baseline_mw"]).where(
        grouped["baseline_mw"] > 0, 0.0
    )

    fig = go.Figure(
        go.Bar(
            y=_grids_display(list(grouped.index)),
            x=grouped["dropped_mw"].values,
            orientation="h",
            marker=_bar_marker("#701E96"),
            text=[f"{p * 100:.1f}%" for p in pct],
            textposition="outside",
            textfont=dict(size=_ANNOTATION_FONT_SIZE),
            cliponaxis=False,
            customdata=[
                f"<b>{alias_grid(g)}</b><br>dropped: {d:.3f} MW<br>baseline: {b:.3f} MW<br>"
                f"share: {p * 100:.1f}%<br>n: {int(n)}"
                for g, d, b, p, n in zip(
                    grouped.index,
                    grouped["dropped_mw"],
                    grouped["baseline_mw"],
                    pct,
                    grouped["n"],
                )
            ],
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_yaxes(title="grid", autorange="reversed")
    # Pad the value axis so the outside "% of baseline" labels have room.
    x_max = float(grouped["dropped_mw"].max()) if len(grouped) else 1.0
    fig.update_xaxes(
        title="absolute load dropped vs baseline (MW)",
        range=[0, x_max * 1.18],
        tickformat=".2f",
    )
    fig.update_layout(showlegend=False, bargap=0.45)
    height = _hbar_height(len(grouped))
    fig = _apply_theme(fig, title=title, height=height, width=_BAR_FIG_WIDTH)
    fig.update_layout(margin=dict(l=84, r=60, t=72, b=64))
    return _save(fig, out_path)


# Diary distribution (Pillars 1, 8)


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

    # Oracle is a snapshot LP solve with no negotiation, so every diary
    # counter is zero — drop it to avoid an empty stack.
    if "variant" in df.columns:
        df = df[df["variant"] != "oracle"]
    if df.empty:
        return _save(_empty_fig("no non-oracle diary data", title), out_path)

    by_variant = df.groupby("variant")[[c[0] for c in cols]].sum()
    if by_variant.empty:
        return _save(_empty_fig("no diary data", title), out_path)

    fig = go.Figure()
    variants_lbl = _variants_display(list(by_variant.index))
    for i, (col, label, color) in enumerate(cols):
        fig.add_trace(
            go.Bar(
                y=variants_lbl,
                x=by_variant[col].values,
                orientation="h",
                name=label,
                marker=_bar_marker(
                    color, pattern_shape=_PATTERN_SHAPES[i % len(_PATTERN_SHAPES)]
                ),
                hovertemplate=f"<b>{label}</b><br>variant: %{{y}}<br>count: %{{x}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack", bargap=0.42)
    fig.update_yaxes(title="variant")
    fig.update_xaxes(title="count", rangemode="tozero")
    height = _hbar_height(len(variants_lbl)) + 30
    return _save(
        _apply_theme(
            fig,
            title=title,
            height=height,
            width=_BAR_FIG_WIDTH,
            font_bump=2,
            legend_top=True,
        ),
        out_path,
    )


# Restoration time — the chapter's first-asked metric


def time_to_stabilise_box(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Time to stabilize by variant",
) -> Path:
    """Box-plot of ``outcomes.time_to_stabilise_s`` per variant — sim
    time to reach steady-state after the failure. NaN (never stabilised)
    rows are dropped. Oracle is excluded: its snapshot solve hard-codes
    the metric to 0.0, which would crush the other distributions at y=0.
    """
    col = "outcomes__time_to_stabilise_s"
    if df.empty or col not in df.columns:
        return _save(_empty_fig("no time_to_stabilise_s column", title), out_path)
    sub = df.dropna(subset=[col, "variant"])
    sub = sub[sub["variant"] != "oracle"]
    if sub.empty:
        return _save(_empty_fig("no stabilisation samples", title), out_path)

    variants = sorted(sub["variant"].unique())
    fig = go.Figure()
    n_boxes = 0
    for v in variants:
        values = sub[sub["variant"] == v][col].astype(float).values
        if values.size == 0:
            continue
        n_boxes += 1
        _add_box(
            fig,
            values,
            name=alias_variant(str(v)),
            color=_variant_color(str(v)),
            hovertemplate=(
                f"<b>{alias_variant(str(v))}</b><br>"
                "time: %{y:.2f} s<extra></extra>"
            ),
        )
    fig.update_yaxes(
        title="time to stabilize (s)", rangemode="tozero", tickformat=".2f"
    )
    fig.update_xaxes(title="variant")
    fig.update_layout(showlegend=False, boxgap=0.45, boxgroupgap=0.2)
    return _save(
        _apply_theme(fig, title=title, height=380, width=_box_fig_width(n_boxes), no_legend=True),
        out_path,
    )


# Solver health — regression watch


def solver_health_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Solver health by variant",
) -> Path:
    """Per-variant stacked bars of mean solver-failure counts per task,
    split into ``solver_infeasibilities`` (LP infeasibilities) and
    ``solver_warnings`` (non-OK terminations like gap / time limit).
    Plotting the components surfaces a regression in either pathway.
    """
    inf_col = "solver_infeasibilities"
    warn_col = "solver_warnings"
    if df.empty or inf_col not in df.columns:
        return _save(_empty_fig("no solver-health columns", title), out_path)
    sub = df.dropna(subset=["variant"]).copy()
    if sub.empty:
        return _save(_empty_fig("no rows with a variant", title), out_path)

    sub[inf_col] = sub[inf_col].fillna(0).astype(float)
    if warn_col in sub.columns:
        sub[warn_col] = sub[warn_col].fillna(0).astype(float)
    else:
        sub[warn_col] = 0.0

    grouped = sub.groupby("variant").agg(
        inf_mean=(inf_col, "mean"),
        warn_mean=(warn_col, "mean"),
        n=(inf_col, "count"),
    )
    if grouped.empty:
        return _save(_empty_fig("no grouped rows", title), out_path)

    variants = list(grouped.index)
    variants_lbl = _variants_display(variants)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="infeasibilities",
            y=variants_lbl,
            x=grouped["inf_mean"].values,
            orientation="h",
            marker=_bar_marker("#D62728"),
            hovertemplate=(
                "<b>%{y}</b><br>mean infeasibilities/task: %{x:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            name="other warnings",
            y=variants_lbl,
            x=grouped["warn_mean"].values,
            orientation="h",
            # Red vs orange is a deuteranopia clash — hatch the warnings.
            marker=_bar_marker("#E07A1F", pattern_shape="/"),
            hovertemplate=(
                "<b>%{y}</b><br>mean warnings/task: %{x:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(barmode="stack", bargap=0.42)
    fig.update_yaxes(title="variant")
    fig.update_xaxes(
        title="solver events per task (mean)", rangemode="tozero", tickformat=".2f"
    )
    height = _hbar_height(len(variants_lbl))
    return _save(
        _apply_theme(
            fig, title=title, height=height, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


# Regulation actions by reason — which layer actually fired?


# Label + colour per regulate reason (from the ``record_regulate(reason=)``
# call sites in ``scare/service/*.py``). Unknown reasons fall back to the
# qualitative palette.
_REGULATE_REASON_LABELS: dict[str, tuple[str, str]] = {
    # Balance / gossip layer
    "balance": ("balance gossip", "#1F4E96"),
    "self_local_gen": ("balance self local-gen", "#3F6FBD"),
    # Holonic ADMM
    "holon_supply_priority": ("holon supply-priority", "#2E7D32"),
    "holon_tier_alloc": ("holon tier-alloc", "#56A656"),
    # Cross-sector CP ADMM
    "cp_admm": ("CP ADMM", "#17BECF"),
    # Constraint-driven local actions
    "curtail": ("curtailment auction", "#E07A1F"),
    "heat_recovery": ("heat recovery", "#FFA959"),
    # Stability / local-gen fallback
    "stability": ("generation control", "#9467BD"),
    "local_gen_fallback": ("local-gen fallback", "#D62728"),
}


def regulates_by_reason_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Regulation actions by trigger (per variant)",
) -> Path:
    """Stacked bar of mean ``outcomes.regulates_by_reason`` counts per
    variant — "did each layer actually fire?". An ablation that disables
    the holonic ADMM should show zero ``holon_*`` actions, etc. Reasons
    are discovered dynamically from the columns; unknown ones use the raw
    key and a fallback colour.
    """
    prefix = "outcomes__regulates_by_reason__"
    reason_keys = sorted({c[len(prefix) :] for c in df.columns if c.startswith(prefix)})
    if df.empty or not reason_keys:
        return _save(_empty_fig("no regulates_by_reason columns", title), out_path)
    sub = df.dropna(subset=["variant"]).copy()
    # Oracle never fires the regulate path — drop its empty bar.
    sub = sub[sub["variant"] != "oracle"]
    if sub.empty:
        return _save(_empty_fig("no non-oracle rows with a variant", title), out_path)

    cols: list[tuple[str, str, str]] = []
    for i, key in enumerate(reason_keys):
        label, color = _REGULATE_REASON_LABELS.get(
            key,
            (key, _QUAL_PALETTE[i % len(_QUAL_PALETTE)]),
        )
        col = f"{prefix}{key}"
        sub[col] = sub[col].fillna(0).astype(float)
        cols.append((col, label, color))

    grouped = sub.groupby("variant")[[c[0] for c in cols]].mean()
    if grouped.empty:
        return _save(_empty_fig("no grouped data", title), out_path)
    variants_lbl = _variants_display(list(grouped.index))

    fig = go.Figure()
    for i, (col, label, color) in enumerate(cols):
        fig.add_trace(
            go.Bar(
                name=label,
                y=variants_lbl,
                x=grouped[col].values,
                orientation="h",
                marker=_bar_marker(
                    color, pattern_shape=_PATTERN_SHAPES[i % len(_PATTERN_SHAPES)]
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>variant: %{{y}}<br>"
                    "mean count/task: %{x:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(barmode="stack", bargap=0.42)
    fig.update_yaxes(title="variant")
    fig.update_xaxes(
        title="mean regulation actions per task", rangemode="tozero", tickformat=".1f"
    )
    height = _hbar_height(len(variants_lbl)) + 40
    return _save(
        _apply_theme(
            fig,
            title=title,
            height=height,
            width=_BAR_FIG_WIDTH,
            font_bump=2,
            legend_top=True,
        ),
        out_path,
    )


# Per-sector restoration ratio (mirrors restoration_by_tier_bar)


def restoration_by_sector_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Per-sector restoration ratio (post / baseline served, MW)",
) -> Path:
    """:func:`restoration_by_tier_bar` along the sector axis: one bar per
    (grid × sector) showing each carrier's pre-failure served load that
    survived restoration. Reads the
    ``outcomes__restoration__by_sector__<sector>__ratio`` columns.
    """
    if df.empty or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    sub = df[df["variant"] == "scare"] if "scare" in df["variant"].unique() else df

    sector_cols: dict[str, str] = {}
    prefix = "outcomes__restoration__by_sector__"
    suffix = "__ratio"
    for col in sub.columns:
        if col.startswith(prefix) and col.endswith(suffix):
            sector_cols[col[len(prefix) : -len(suffix)]] = col
    if not sector_cols:
        return _save(
            _empty_fig("no per-sector restoration columns", title),
            out_path,
        )

    sectors = sorted(sector_cols)
    grouped = sub.groupby("grid")[[sector_cols[s] for s in sectors]].mean()
    if grouped.empty:
        return _save(_empty_fig("empty per-sector table", title), out_path)
    grids = grouped.index.tolist()
    grids_lbl = _grids_display(grids)

    fig = go.Figure()
    for sec in sectors:
        col = sector_cols[sec]
        color = _SECTOR_COLOR.get(sec, "#888888")
        fig.add_trace(
            go.Bar(
                name=sec,
                y=grids_lbl,
                x=grouped[col].values,
                orientation="h",
                marker=_bar_marker(
                    color, pattern_shape=_SECTOR_PATTERN.get(sec, "")
                ),
                hovertemplate=(
                    f"<b>{sec}</b><br>grid: %{{y}}<br>ratio: %{{x:.3f}}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=1.0, line=dict(color="#BBBBBB", dash="dash", width=1))
    fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.12)
    fig.update_xaxes(
        title="restoration ratio (post / baseline served)",
        range=[0, 1.05],
        tickformat=".2f",
    )
    fig.update_yaxes(title="grid")
    height = _hbar_height(len(grids), len(sectors))
    return _save(
        _apply_theme(
            fig, title=title, height=height, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


# Constraint-envelope trajectory (voltage / pressure / temperature)


# Operating envelopes from scare's ``SECTOR_CONSTRAINTS`` so the shaded
# band matches the bounds ``GridConstraintMonitor`` fires on. Lazy import
# keeps this module usable without scare on PYTHONPATH; on failure, fall
# back to the relaxed LP bounds.
def _sector_envelope_bounds() -> dict[str, tuple[float, float]]:
    try:
        from scare.base.model import SECTOR_CONSTRAINTS, Sector
    except Exception:  # pragma: no cover — only on import-path issues
        # Relaxed LP bounds (see monee.solve_load_shedding_*).
        return {
            "avg_vm_pu": (0.90, 1.10),
            "avg_pressure_pu": (0.90, 1.10),
            "avg_t_k": (313.15, 403.15),
        }
    return {
        "avg_vm_pu": SECTOR_CONSTRAINTS[Sector.ELECTRICITY]["vm_pu"],
        "avg_pressure_pu": SECTOR_CONSTRAINTS[Sector.GAS]["pressure_pu"],
        "avg_t_k": SECTOR_CONSTRAINTS[Sector.HEAT]["t_k"],
    }


# Envelope band drawn per row in low-alpha sector colour; the single
# legend entry is neutral grey (covers all three sectors).
_ENVELOPE_GRAY = "#7F7F7F"
_ENVELOPE_FILL_ALPHA = 0.10
_ENVELOPE_FILL_RGBA = "rgba(127, 127, 127, 0.18)"
_SPREAD_FILL_ALPHA = 0.20

# Dash styles for avg / min / max — distinguishable in greyscale and
# for colourblind readers.
_AVG_DASH = "solid"
_MIN_DASH = "longdash"
_MAX_DASH = "dashdot"


def _stale_data_segment(
    timeseries: pd.DataFrame,
) -> tuple[float | None, int]:
    """Return ``(stale_from_t, repeated_samples)`` from the
    ``last_feasible_solve_t`` column.

    When an energyflow recompute returns infeasible the previous
    ``_net_results`` is kept, so every observation-based metric freezes
    at the last-feasible state. ``last_feasible_solve_t`` is the most
    recent successful refresh; rows past it repeat that snapshot.
    ``stale_from_t`` is the earliest such ``time_s`` (``None`` when the
    column is absent or the trace is fresh); the count of repeated
    samples doubles as a solver-failure proxy.
    """
    if "last_feasible_solve_t" not in timeseries.columns:
        return None, 0
    if "time_s" not in timeseries.columns:
        return None, 0
    t = timeseries["time_s"].astype(float).values
    lfs = timeseries["last_feasible_solve_t"].astype(float).values
    if len(t) == 0:
        return None, 0
    stale_mask = t > (lfs + 1e-9)
    if not stale_mask.any():
        return None, 0
    first_stale_idx = int(stale_mask.argmax())
    return float(t[first_stale_idx]), int(stale_mask.sum())


def _mask_deenergised(avg_col: str, vals: np.ndarray | None) -> np.ndarray | None:
    """Mask de-energised solver-bound artefacts in a recorded min/max series.

    Gas junctions cut off from supply read ``pressure_pu`` at a solver bound
    (~0 or ~sqrt(3)~1.732); isolated heat junctions read ``t_k~0``. These are
    not physical extremes, so blank them (NaN) — they were dropped at record
    time for new runs, but older runs baked them into the aggregate. The true
    energised extreme at those ticks was never persisted, so masked ticks gap.
    """
    if vals is None:
        return None
    from scare.base.model import (
        DEENERGISED_PRESSURE_HIGH_PU,
        DEENERGISED_PRESSURE_PU,
    )

    out = np.asarray(vals, dtype=float).copy()
    if avg_col.endswith("pressure_pu"):
        out[(out <= DEENERGISED_PRESSURE_PU) | (out >= DEENERGISED_PRESSURE_HIGH_PU)] = (
            np.nan
        )
    elif avg_col.endswith("t_k"):
        out[out <= 0.0] = np.nan
    return out


def constraint_envelope_trajectory(
    timeseries: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Constraint envelopes — voltage / pressure / temperature",
    failure_t: float | None = None,
    solver_failures: int | None = None,
) -> Path:
    """Voltage / pressure / temperature trajectories with operating
    envelopes shaded. One stacked subplot per sector; missing sectors
    are skipped.

    Each sector panel draws four series: average (solid, node mean), min
    (long-dash), max (dash-dot), and the min–max node spread (translucent
    fill). The envelope is a sector-tinted band with dotted boundaries;
    the legend folds all sectors into one neutral-grey "constraint
    envelope" entry, and min/max/avg/spread are legended once each.
    Failure / reconfiguration events are vertical guides on every row.

    Stale-data handling: an infeasible energyflow recompute freezes every
    observation-based metric at the last-feasible state. lfs lagging
    ``time_s`` alone is NOT staleness (the recording tick can be finer
    than the solve schedule), so a stretch is marked stale only when
    ``solver_failures > 0``; older runs fall back to lfs-based detection.
    """
    if timeseries.empty or "time_s" not in timeseries.columns:
        return _save(_empty_fig("no timeseries", title), out_path)
    # Trust staleness only when the runner reported real infeasibilities;
    # lfs-vs-time lag without failures is normal inter-solve sampling.
    lfs_from_t, lfs_sample_count = _stale_data_segment(timeseries)
    if solver_failures is not None and solver_failures > 0:
        stale_from_t = lfs_from_t
    else:
        stale_from_t = None
    if solver_failures and solver_failures > 0:
        banner = (
            f" — ⚠ {solver_failures} solver infeasibility(ies); data past "
            f"t≈{stale_from_t:.2f}s is the last-feasible snapshot held over"
            if stale_from_t is not None
            else f" — ⚠ {solver_failures} solver infeasibility(ies)"
        )
        title = title + banner

    envelopes = _sector_envelope_bounds()
    # (avg_col, min_col, max_col, panel title, envelope bounds, sector colour)
    rows = [
        (
            "avg_vm_pu",
            "min_vm_pu",
            "max_vm_pu",
            "voltage (p.u.)",
            envelopes["avg_vm_pu"],
            _SECTOR_COLOR["electricity"],
        ),
        (
            "avg_pressure_pu",
            "min_pressure_pu",
            "max_pressure_pu",
            "pressure (p.u.)",
            envelopes["avg_pressure_pu"],
            _SECTOR_COLOR["gas"],
        ),
        (
            "avg_t_k",
            "min_t_k",
            "max_t_k",
            "temperature (K)",
            envelopes["avg_t_k"],
            _SECTOR_COLOR["heat"],
        ),
    ]
    present = [r for r in rows if r[0] in timeseries.columns]
    if not present:
        return _save(_empty_fig("no constraint-state recordings", title), out_path)

    fig = make_subplots(
        rows=len(present),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[r[3] for r in present],
    )

    # Event marker styles — applied as vlines on every row.
    event_styles = {
        "line_failure": dict(color="#1A1A1A", dash="dash"),
        "branch_failure": dict(color="#1A1A1A", dash="dash"),
        "reconfiguration_completed": dict(color="#9467BD", dash="dot"),
        "constraint_violation": dict(color="#D62728", dash="dot"),
    }
    seen_kinds: set[str] = set()
    if not events.empty and {"t", "kind"}.issubset(events.columns):
        seen_kinds = {str(k) for k in events["kind"].unique() if str(k) in event_styles}

    # Single-entry legend groups — each is shown once across all rows.
    LG_ENVELOPE = "constraint_envelope"
    LG_SPREAD = "node_spread"
    LG_AVG = "node_avg"
    LG_MIN = "node_min"
    LG_MAX = "node_max"
    legend_emitted: set[str] = set()

    def _once(group: str) -> bool:
        if group in legend_emitted:
            return False
        legend_emitted.add(group)
        return True

    # Neutral-grey sentinel legend entry for the constraint envelope
    # (per-row fills are sector-tinted; the legend stays context-grey).
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                size=14,
                color=_ENVELOPE_FILL_RGBA,
                line=dict(color=_ENVELOPE_GRAY, width=1),
            ),
            name="constraint envelope",
            legendgroup=LG_ENVELOPE,
            showlegend=_once(LG_ENVELOPE),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    x = timeseries["time_s"].values
    x_list = list(x)
    x_band = x_list + x_list[::-1]

    for row_idx, (avg_col, min_col, max_col, y_label, bounds, line_color) in enumerate(
        present, start=1
    ):
        lo, hi = bounds

        # Operating envelope (sector-tinted band + dotted boundaries)
        fig.add_trace(
            go.Scatter(
                x=x_band,
                y=[hi] * len(x_list) + [lo] * len(x_list),
                fill="toself",
                fillcolor=_hex_to_rgba(line_color, _ENVELOPE_FILL_ALPHA),
                line=dict(color="rgba(0,0,0,0)"),
                name="constraint envelope",
                legendgroup=LG_ENVELOPE,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row_idx,
            col=1,
        )
        fig.add_hline(
            y=lo, line=dict(color=line_color, dash="dot", width=1), row=row_idx, col=1
        )
        fig.add_hline(
            y=hi, line=dict(color=line_color, dash="dot", width=1), row=row_idx, col=1
        )

        # Stale-snapshot overlay
        if stale_from_t is not None:
            x_max = float(x[-1])
            if x_max > stale_from_t:
                fig.add_vrect(
                    x0=stale_from_t,
                    x1=x_max,
                    fillcolor="rgba(150, 150, 150, 0.20)",
                    line=dict(width=0),
                    layer="below",
                    row=row_idx,
                    col=1,
                )

        avg_vals = timeseries[avg_col].astype(float).values
        have_min = min_col in timeseries.columns
        have_max = max_col in timeseries.columns
        min_vals = timeseries[min_col].astype(float).values if have_min else None
        max_vals = timeseries[max_col].astype(float).values if have_max else None
        # Post-process: drop de-energised solver-bound artefacts so regenerated
        # plots of runs recorded before the record-time filter don't draw
        # spurious min/max lines. A gas region cut off from supply reads
        # pressure_pu~0 or saturates to ~sqrt(3)~1.732, and an isolated heat
        # junction reads t_k~0. The true energised extreme at those ticks was
        # never persisted (only the aggregate), so masked points become gaps.
        min_vals = _mask_deenergised(avg_col, min_vals)
        max_vals = _mask_deenergised(avg_col, max_vals)

        # Min–max spread (translucent fill between min and max)
        if have_min and have_max:
            fig.add_trace(
                go.Scatter(
                    x=x_list + x_list[::-1],
                    y=list(max_vals) + list(min_vals[::-1]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(line_color, _SPREAD_FILL_ALPHA),
                    line=dict(color="rgba(0,0,0,0)"),
                    name="min–max spread (across nodes)",
                    legendgroup=LG_SPREAD,
                    showlegend=_once(LG_SPREAD),
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=1,
            )

        # min / max / avg series
        def _add_series(
            y_vals,
            *,
            dash: str,
            width: float,
            legend_group: str,
            legend_name: str,
            hover_label: str,
        ) -> None:
            # Solid up to the freshness boundary, dotted afterwards to
            # demote the held-over snapshot; split segments share a
            # legend group.
            if stale_from_t is not None:
                fresh_mask = x <= stale_from_t
                stale_mask = x >= stale_from_t
                if fresh_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=x[fresh_mask],
                            y=y_vals[fresh_mask],
                            mode="lines",
                            line=dict(color=line_color, width=width, dash=dash),
                            name=legend_name,
                            legendgroup=legend_group,
                            showlegend=_once(legend_group),
                            hovertemplate=(
                                f"<b>{y_label} — {hover_label}</b><br>"
                                "t: %{x:.2f}s<br>value: %{y:.4f}<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=1,
                    )
                if stale_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=x[stale_mask],
                            y=y_vals[stale_mask],
                            mode="lines",
                            line=dict(color=line_color, width=width, dash="dot"),
                            opacity=0.55,
                            name=legend_name,
                            legendgroup=legend_group,
                            showlegend=_once(legend_group),
                            hovertemplate=(
                                f"<b>{y_label} — {hover_label} (held over)</b><br>"
                                "t: %{x:.2f}s<br>value: %{y:.4f}<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=1,
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_vals,
                        mode="lines",
                        line=dict(color=line_color, width=width, dash=dash),
                        name=legend_name,
                        legendgroup=legend_group,
                        showlegend=_once(legend_group),
                        hovertemplate=(
                            f"<b>{y_label} — {hover_label}</b><br>"
                            "t: %{x:.2f}s<br>value: %{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=1,
                )

        # Order matters for z-stacking: min/max below, avg on top.
        if have_min:
            _add_series(
                min_vals,
                dash=_MIN_DASH,
                width=1.6,
                legend_group=LG_MIN,
                legend_name="min (across nodes)",
                hover_label="min",
            )
        if have_max:
            _add_series(
                max_vals,
                dash=_MAX_DASH,
                width=1.6,
                legend_group=LG_MAX,
                legend_name="max (across nodes)",
                hover_label="max",
            )
        _add_series(
            avg_vals,
            dash=_AVG_DASH,
            width=2.4,
            legend_group=LG_AVG,
            legend_name="avg (across nodes)",
            hover_label="avg",
        )

        # Event vlines (same on every row for a synced read)
        if not events.empty and {"t", "kind"}.issubset(events.columns):
            for kind, style in event_styles.items():
                ev = events[events["kind"] == kind]
                if ev.empty:
                    continue
                for tx in ev["t"].astype(float).unique():
                    fig.add_vline(
                        x=float(tx),
                        line=dict(color=style["color"], dash=style["dash"], width=1),
                        opacity=0.6,
                        row=row_idx,
                        col=1,
                    )
        elif failure_t is not None:
            fig.add_vline(
                x=failure_t,
                line=dict(color="#1A1A1A", dash="dash", width=1),
                opacity=0.6,
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(title=y_label, row=row_idx, col=1)

    # Sentinel scatters for the event-kind legend.
    for kind in sorted(seen_kinds):
        style = event_styles[kind]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=style["color"], dash=style["dash"], width=2),
                name=kind,
                legendgroup=f"event_{kind}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title="simulation time (s)", row=len(present), col=1)
    height = max(_FIG_HEIGHT, 200 * len(present) + 100)
    return _save(_apply_theme(fig, title=title, height=height), out_path)


# Constraint-violation integral by sector × variant


def constraint_violation_integral_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Constraint-violation integral by sector",
) -> Path:
    """Grouped bars: per-variant mean of the per-sector violation
    integral ``∫ max(0, util(t) − 1) dt`` (from
    :func:`metrics.constraint_violation_integral`). Zero ⇔ the sector
    never left its envelope on average; larger ⇔ longer/deeper excursions.
    """
    sectors = ["electricity", "gas", "heat"]
    cols = {s: f"outcomes__constraint_violation_integral__{s}" for s in sectors}
    present = [s for s in sectors if cols[s] in df.columns]
    if df.empty or not present:
        return _save(
            _empty_fig("no constraint_violation_integral columns", title), out_path
        )
    sub = df.dropna(subset=["variant"]).copy()
    if sub.empty:
        return _save(_empty_fig("no rows with a variant", title), out_path)
    for s in present:
        sub[cols[s]] = sub[cols[s]].fillna(0).astype(float)

    grouped = sub.groupby("variant")[[cols[s] for s in present]].mean()
    if grouped.empty:
        return _save(_empty_fig("no grouped data", title), out_path)
    variants = list(grouped.index)
    variants_lbl = _variants_display(variants)

    fig = go.Figure()
    for sec in present:
        fig.add_trace(
            go.Bar(
                name=sec,
                y=variants_lbl,
                x=grouped[cols[sec]].values,
                orientation="h",
                marker=_bar_marker(
                    _SECTOR_COLOR.get(sec, "#888888"),
                    pattern_shape=_SECTOR_PATTERN.get(sec, ""),
                ),
                hovertemplate=(
                    f"<b>{sec}</b><br>variant: %{{y}}<br>"
                    "mean integral: %{x:.4g}<extra></extra>"
                ),
            )
        )
    fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.12)
    fig.update_yaxes(title="variant")
    fig.update_xaxes(
        title="∫ max(0, util(t) − 1) dt  (mean per task)",
        rangemode="tozero",
        tickformat=".3g",
    )
    height = _hbar_height(len(variants), len(present))
    return _save(
        _apply_theme(
            fig, title=title, height=height, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


# Per-variable-type violation columns in summary.csv (flattened claim detail).
# Voltage / pressure / temperature / line_load come from the constraint
# compliance claim's ``by_variable`` tally; slack from the slack-budget claim's
# steady-state breach count (legacy ``n_violations`` as a fallback).
_VIOLATION_VAR_COLS = {
    "voltage": [
        "claims__constraint_compliance__detail__by_variable__voltage__n_violations"
    ],
    "pressure": [
        "claims__constraint_compliance__detail__by_variable__pressure__n_violations"
    ],
    "line_load": [
        "claims__constraint_compliance__detail__by_variable__line_load__n_violations"
    ],
    "temperature": [
        "claims__constraint_compliance__detail__by_variable__temperature__n_violations"
    ],
    "slack": [
        "claims__slack_budget_compliance__detail__n_steady_breaches",
        "claims__slack_budget_compliance__detail__n_violations",
    ],
}


def constraint_violations_by_variable_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Constraint violations by variable type (mean per task)",
) -> Path:
    """Grouped bars: per-variant mean number of end-of-sim constraint
    violations, split by variable type (voltage, pressure, line load, slack,
    temperature). A compliance-accompanying count — *how many* bounds a variant
    breaches, complementing the binary ``constraint_compliance`` /
    ``slack_budget_compliance`` pass/fail and the per-sector violation integral.

    ``temperature`` (heat ``t_k``) is non-gating: a temperature-infeasible node
    already serves no load, so it is penalised via the served metric, not the
    compliance gate. It is shown hatched as a diagnostic, not a failure.

    Reads the flattened claim-detail columns (see ``_VIOLATION_VAR_COLS``);
    missing columns count as zero so a campaign that never recorded a given
    variable simply shows an empty bar.
    """
    if df.empty or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    sub = df.dropna(subset=["variant"]).copy()
    if sub.empty:
        return _save(_empty_fig("no rows with a variant", title), out_path)

    # Resolve each variable type to its first present column; build a per-type
    # numeric series (absent -> 0).
    var_series: dict[str, pd.Series] = {}
    for var, candidates in _VIOLATION_VAR_COLS.items():
        col = next((c for c in candidates if c in sub.columns), None)
        if col is None:
            continue
        var_series[var] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
    if not var_series:
        return _save(
            _empty_fig("no per-variable violation columns — re-run campaign", title),
            out_path,
        )

    work = sub[["variant"]].copy()
    for var, s in var_series.items():
        work[var] = s.values
    present = [v for v in _CONSTRAINT_VARIABLE_ORDER if v in var_series]
    grouped = work.groupby("variant")[present].mean()
    if grouped.empty:
        return _save(_empty_fig("no grouped data", title), out_path)
    variants = list(grouped.index)
    variants_lbl = _variants_display(variants)

    fig = go.Figure()
    for var in present:
        nongating = var in _NONGATING_VARIABLE_TYPES
        label = f"{var} (non-gating)" if nongating else var
        # Each variable type gets a distinct hatch (CVD channel) on top of
        # its hue; the non-gating temperature stays hatched as before.
        fig.add_trace(
            go.Bar(
                name=label,
                y=variants_lbl,
                x=grouped[var].values,
                orientation="h",
                marker=_bar_marker(
                    _CONSTRAINT_VARIABLE_COLOR.get(var, "#888888"),
                    pattern_shape=_CONSTRAINT_VARIABLE_PATTERN.get(var, ""),
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>variant: %{{y}}<br>"
                    "mean violations / task: %{x:.3g}<extra></extra>"
                ),
            )
        )
    fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.10)
    fig.update_yaxes(title="variant")
    fig.update_xaxes(
        title="mean number of violations per task",
        rangemode="tozero",
        tickformat=".3g",
    )
    height = _hbar_height(len(variants), len(present))
    return _save(
        _apply_theme(
            fig, title=title, height=height, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


# Validity plots — verify the multi-level controller behaves as claimed


def system_balance_trajectory(
    timeseries: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "System balance — Σ regulation per sector",
    failure_t: float | None = None,
) -> Path:
    """Per-sector ``Σ regulation`` over time (full-system view). Same
    series as ``restoration_trajectory`` but with an explicit
    "did the global controller settle?" framing.
    """
    return restoration_trajectory(
        timeseries,
        events,
        out_path,
        title=title,
        failure_t=failure_t,
    )


def _group_balance_lines(
    timeseries: pd.DataFrame,
    out_path: Path,
    *,
    column_prefix: str,
    series_label: str,
    title: str,
) -> Path:
    """Shared body for ``coalition_balance_lines`` / ``holon_balance_lines``.
    Plots every ``<column_prefix>__<sector>__<idx>`` column — one faded
    line per group, one subplot row per sector.
    """
    if timeseries.empty or "time_s" not in timeseries.columns:
        return _save(_empty_fig("no timeseries", title), out_path)

    by_sector: dict[str, list[str]] = {}
    for col in timeseries.columns:
        if not col.startswith(f"{column_prefix}__"):
            continue
        parts = col.split("__")
        if len(parts) < 3:
            continue
        sec = parts[1]
        by_sector.setdefault(sec, []).append(col)

    if not by_sector:
        return _save(_empty_fig(f"no {column_prefix}__* columns", title), out_path)

    sectors = [s for s in ("electricity", "gas", "heat") if s in by_sector]
    if not sectors:
        sectors = sorted(by_sector)

    fig = make_subplots(
        rows=len(sectors),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[s for s in sectors],
    )
    x = timeseries["time_s"].values
    for row_idx, sec in enumerate(sectors, start=1):
        cols = sorted(by_sector[sec])
        base = _SECTOR_COLOR.get(sec, "#888888")
        for i, col in enumerate(cols):
            # Fade opacity when many groups overlap.
            opacity = 0.55 if len(cols) > 6 else 0.85
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=timeseries[col].astype(float).values,
                    mode="lines",
                    line=dict(color=base, width=2),
                    opacity=opacity,
                    name=col.split("__")[-1],
                    legendgroup=sec,
                    legendgrouptitle_text=sec,
                    showlegend=(row_idx == 1 and i < 10),  # avoid legend explosion
                    hovertemplate=(
                        f"<b>{series_label} {sec}/{col.split('__')[-1]}</b><br>"
                        "t: %{x:.2f}s<br>Σ regulation: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )
        fig.update_yaxes(title=f"Σ reg. ({sec})", row=row_idx, col=1)

    fig.update_xaxes(title="simulation time (s)", row=len(sectors), col=1)
    height = max(_FIG_HEIGHT, 170 * len(sectors) + 80)
    return _save(_apply_theme(fig, title=title, height=height), out_path)


def slack_trajectory(
    timeseries: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "External-grid slack — operating point over time",
    failure_t: float | None = None,
    slack_meta: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Per-slack-child trajectory — one line per ExtPowerGrid /
    ExtHydrGrid child, one subplot per sector. Reads the
    ``slack__<sector>__<aid>`` columns from ``_register_recordings`` in
    ``scare.scenario.restoration``: electricity carries ``p_mw``, gas /
    heat carry ``mass_flow``. A dashed vline marks the first failure.

    With ``slack_meta`` (from ``slack_meta.json``), overlays per-child
    ``±budget`` (solid, operator-policy target) and ``±lp_envelope``
    (dotted, the LP Var bound) so "did the MAS stay inside its budget
    envelope?" reads at a glance. Unbudgeted children get no overlay.
    """
    if timeseries.empty or "time_s" not in timeseries.columns:
        return _save(_empty_fig("no timeseries", title), out_path)

    by_sector: dict[str, list[str]] = {}
    for col in timeseries.columns:
        if not col.startswith("slack__"):
            continue
        parts = col.split("__")
        if len(parts) < 3:
            continue
        by_sector.setdefault(parts[1], []).append(col)

    if not by_sector:
        return _save(_empty_fig("no slack__* columns", title), out_path)

    sectors = [s for s in ("electricity", "gas", "heat") if s in by_sector]
    if not sectors:
        sectors = sorted(by_sector)

    fig = make_subplots(
        rows=len(sectors),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[s for s in sectors],
    )
    x = timeseries["time_s"].values
    x_span = [float(x[0]), float(x[-1])] if len(x) else [0.0, 1.0]
    _y_unit = {
        "electricity": "p_mw",
        "gas": "mass_flow (kg/s)",
        "heat": "mass_flow (kg/s)",
    }
    meta = slack_meta or {}
    for row_idx, sec in enumerate(sectors, start=1):
        cols = sorted(by_sector[sec])
        base = _SECTOR_COLOR.get(sec, "#888888")
        for i, col in enumerate(cols):
            opacity = 0.55 if len(cols) > 6 else 0.85
            aid = col.split("__", 2)[-1]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=timeseries[col].astype(float).values,
                    mode="lines",
                    line=dict(color=base, width=1.4),
                    opacity=opacity,
                    name=aid,
                    legendgroup=sec,
                    legendgrouptitle_text=sec,
                    showlegend=(row_idx == 1 and i < 10),
                    hovertemplate=(
                        f"<b>slack {sec}/{aid}</b><br>"
                        "t: %{x:.2f}s<br>value: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

            # Overlay ±budget and ±LP-envelope when ``slack_meta`` has
            # them for this aid; legend collapses onto each sector's
            # first child to stay compact.
            child_meta = meta.get(aid) or {}
            budget = child_meta.get("budget")
            envelope = child_meta.get("lp_envelope")
            show_legend_overlay = row_idx == 1 and i == 0
            if isinstance(budget, (int, float)):
                for sign, label in ((1.0, "+budget"), (-1.0, "−budget")):
                    fig.add_trace(
                        go.Scatter(
                            x=x_span,
                            y=[sign * float(budget), sign * float(budget)],
                            mode="lines",
                            line=dict(color="#444444", dash="solid", width=1.2),
                            name=f"budget ({sec})",
                            legendgroup=f"budget_{sec}",
                            showlegend=(show_legend_overlay and sign > 0),
                            hovertemplate=(
                                f"<b>{label} {sec}/{aid}</b><br>"
                                f"value: {sign * float(budget):.4f}<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=1,
                    )
            if isinstance(envelope, (int, float)):
                for sign in (1.0, -1.0):
                    fig.add_trace(
                        go.Scatter(
                            x=x_span,
                            y=[sign * float(envelope), sign * float(envelope)],
                            mode="lines",
                            line=dict(color="#999999", dash="dot", width=1.0),
                            name=f"LP envelope ({sec})",
                            legendgroup=f"envelope_{sec}",
                            showlegend=(show_legend_overlay and sign > 0),
                            hovertemplate=(
                                f"<b>LP envelope {sec}/{aid}</b><br>"
                                f"value: {sign * float(envelope):.4f}<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=1,
                    )

        fig.update_yaxes(title=_y_unit.get(sec, "value"), row=row_idx, col=1)
        if failure_t is not None:
            fig.add_vline(
                x=float(failure_t),
                line=dict(color="#888888", dash="dash", width=1),
                row=row_idx,
                col=1,
            )

    fig.update_xaxes(title="simulation time (s)", row=len(sectors), col=1)
    height = max(_FIG_HEIGHT, 170 * len(sectors) + 80)
    return _save(_apply_theme(fig, title=title, height=height), out_path)


def gas_slack_pressure_trajectory(
    timeseries: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Gas ext-grid regulator — slack pressure setpoint over time",
    failure_t: float | None = None,
) -> Path:
    """Single-panel trajectory of the gas slack pressure setpoint — one line
    per gas ``ExtHydrGrid``, the ``GasPressureRegulator``'s layer-0 lever.
    Reads the ``slack_pressure__gas__<aid>`` columns from
    ``_register_recordings``. The operating band (``SECTOR_CONSTRAINTS`` gas
    ``pressure_pu``) is shaded with dotted boundaries; a dashed vline marks the
    first failure. Style mirrors ``slack_trajectory``.
    """
    if timeseries.empty or "time_s" not in timeseries.columns:
        return _save(_empty_fig("no timeseries", title), out_path)

    cols = sorted(
        c for c in timeseries.columns if c.startswith("slack_pressure__gas__")
    )
    if not cols:
        return _save(
            _empty_fig("no slack_pressure__gas__* columns", title), out_path
        )

    x = timeseries["time_s"].values
    x_span = [float(x[0]), float(x[-1])] if len(x) else [0.0, 1.0]
    lo, hi = _sector_envelope_bounds()["avg_pressure_pu"]
    base = _SECTOR_COLOR.get("gas", "#2CA02C")

    fig = go.Figure()
    # Operating band: shaded envelope + dotted boundaries (neutral grey).
    fig.add_trace(
        go.Scatter(
            x=x_span + x_span[::-1],
            y=[hi, hi, lo, lo],
            fill="toself",
            fillcolor=_ENVELOPE_FILL_RGBA,
            line=dict(color=_ENVELOPE_GRAY, width=0),
            name="operating band",
            hoverinfo="skip",
        )
    )
    for b in (lo, hi):
        fig.add_trace(
            go.Scatter(
                x=x_span,
                y=[b, b],
                mode="lines",
                line=dict(color=_ENVELOPE_GRAY, dash="dot", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    opacity = 0.55 if len(cols) > 6 else 0.85
    for i, col in enumerate(cols):
        aid = col.split("__", 2)[-1]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=timeseries[col].astype(float).values,
                mode="lines",
                line=dict(color=base, width=1.4),
                opacity=opacity,
                name=aid,
                showlegend=(i < 10),
                hovertemplate=(
                    f"<b>gas slack {aid}</b><br>"
                    "t: %{x:.2f}s<br>setpoint: %{y:.4f} pu<extra></extra>"
                ),
            )
        )

    if failure_t is not None:
        fig.add_vline(
            x=float(failure_t),
            line=dict(color="#888888", dash="dash", width=1),
        )
    fig.update_yaxes(title="slack pressure setpoint (p.u.)")
    fig.update_xaxes(title="simulation time (s)")
    return _save(_apply_theme(fig, title=title), out_path)


def coalition_balance_lines(
    timeseries: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Coalition balances (Level-1) over time",
) -> Path:
    """Per-coalition ``Σ regulation`` lines — one trace per Level-1
    community, subplots by sector. Reads ``coalition_balance__<sec>__<idx>``.
    Each coalition should converge to a flat track after its gossip round;
    persistent oscillation flags a Level-2 coordination gap.
    """
    return _group_balance_lines(
        timeseries,
        out_path,
        column_prefix="coalition_balance",
        series_label="coalition",
        title=title,
    )


def holon_balance_lines(
    timeseries: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Holon balances (Level-2) over time",
) -> Path:
    """Per-holon ``Σ regulation`` lines — one trace per Level-2 chunk,
    one subplot per sector. Reads ``holon_balance__<sec>__<idx>``.
    A holon should smooth out faster than its member coalitions (ADMM
    spreads the burden). With ``enable_holonic=False`` the columns are
    absent and the plot shows a "no data" placeholder.
    """
    return _group_balance_lines(
        timeseries,
        out_path,
        column_prefix="holon_balance",
        series_label="holon",
        title=title,
    )


def regulation_per_child_lines(
    trajectories: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Per-child regulation factor over time",
    max_lines: int = 60,
) -> Path:
    """One line per child aid showing the applied regulation factor over
    sim time. Reads the wide ``trajectories.csv`` from
    ``write_trajectories_csv`` (event-driven, forward-filled).

    Cardinality is high, so lines are thin/transparent and at most
    ``max_lines`` aids are shown, prioritising those whose factor moves
    (std > 0); truncation is footnoted in the title.
    """
    if trajectories.empty or "time_s" not in trajectories.columns:
        return _save(_empty_fig("no trajectories.csv", title), out_path)

    aid_cols = [c for c in trajectories.columns if c != "time_s"]
    if not aid_cols:
        return _save(_empty_fig("no per-aid columns", title), out_path)

    # Pick the most active aids by variance; constant series (unmodulated
    # loads) are filtered so the plot isn't flat lines at 1.0.
    arr = trajectories[aid_cols].astype(float).fillna(method="ffill")
    stds = arr.std(axis=0).fillna(0.0)
    active = stds[stds > 1e-9].sort_values(ascending=False)
    truncated = False
    if len(active) > max_lines:
        active = active.head(max_lines)
        truncated = True
    show_cols = list(active.index) if not active.empty else aid_cols[:max_lines]

    fig = go.Figure()
    x = trajectories["time_s"].values
    for i, aid in enumerate(show_cols):
        color = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=arr[aid].values,
                mode="lines",
                line=dict(color=color, width=1.0),
                opacity=0.55,
                name=aid,
                showlegend=False,
                hovertemplate=(
                    f"<b>{aid}</b><br>"
                    "t: %{x:.2f}s<br>factor: %{y:.3f}<extra></extra>"
                ),
            )
        )

    if truncated:
        subtitle = (
            f"  ·  showing {len(show_cols)} of {len(aid_cols)} aids (highest-variance)"
        )
    else:
        subtitle = ""
    fig.update_xaxes(title="simulation time (s)")
    fig.update_yaxes(title="regulation factor", range=[-0.05, 1.5], tickformat=".2f")
    return _save(_apply_theme(fig, title=title + subtitle, height=380), out_path)


# Per-task system-state overview (slack + control vars + lines + tiers)


_TIER_PALETTE = [
    "#D62728",
    "#E55A4E",
    "#E07A1F",
    "#FFA959",
    "#BCBD22",
    "#2E7D32",
    "#56A656",
    "#17BECF",
    "#1F4E96",
    "#7F7F7F",
]


def _tier_color(tier: int) -> str:
    # Tier 1 = red (critical), fades through the palette to grey (low).
    if 1 <= tier <= len(_TIER_PALETTE):
        return _TIER_PALETTE[tier - 1]
    return _QUAL_PALETTE[tier % len(_QUAL_PALETTE)]


def system_state_overview(
    timeseries: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "System-state overview",
    failure_t: float | None = None,
) -> Path:
    """Per-task stacked-subplots view of the system state over time.

    Five vertically-stacked panels share the simulation-time x-axis:

    1. **External-grid slack** — one trace per slack child, coloured by
       sector (read from ``slack__<sector>__<aid>``).
    2. **Control variables** — ``avg_vm_pu`` / ``avg_pressure_pu`` /
       ``avg_t_k`` with the operating envelopes shaded.
    3. **Line loading** — ``max / p95 / avg`` aggregate of every
       electricity-branch ``loading_percent`` observation, with a 100 %
       reference line.
    4. **Per-tier served fraction** — ``tier_served_mw__<i>`` /
       ``tier_demand_mw__<i>`` per tier, clamped to ``[0, 1]``.
    5. **Per-tier served MW** — absolute served-load timeseries per
       tier so the absolute loss is visible alongside the fraction view.

    Reads all signals from the per-task ``timeseries.csv``. Missing
    columns degrade gracefully (panel drawn but tagged "(no data)").
    Failure / reconfiguration events are synced vlines across every panel.
    """
    if timeseries.empty or "time_s" not in timeseries.columns:
        return _save(_empty_fig("no timeseries", title), out_path)

    x = timeseries["time_s"].astype(float).values

    # Panel inventory
    slack_cols_by_sector: dict[str, list[str]] = {}
    for col in timeseries.columns:
        if not col.startswith("slack__"):
            continue
        parts = col.split("__")
        if len(parts) >= 3:
            slack_cols_by_sector.setdefault(parts[1], []).append(col)

    ctrl_rows: list[tuple[str, str, tuple[float, float], str]] = []
    envelopes = _sector_envelope_bounds()
    for col, label, sector_key in (
        ("avg_vm_pu", "vm (p.u.)", "electricity"),
        ("avg_pressure_pu", "pressure (p.u.)", "gas"),
        ("avg_t_k", "t (K)", "heat"),
    ):
        if col in timeseries.columns:
            ctrl_rows.append((col, label, envelopes[col], _SECTOR_COLOR[sector_key]))

    line_cols = [
        c
        for c in (
            "max_line_loading_percent",
            "p95_line_loading_percent",
            "avg_line_loading_percent",
        )
        if c in timeseries.columns
    ]

    tier_demand_cols: dict[int, str] = {}
    tier_served_cols: dict[int, str] = {}
    for col in timeseries.columns:
        if col.startswith("tier_demand_mw__"):
            try:
                tier_demand_cols[int(col.split("__")[-1])] = col
            except ValueError:
                continue
        elif col.startswith("tier_served_mw__"):
            try:
                tier_served_cols[int(col.split("__")[-1])] = col
            except ValueError:
                continue
    tiers = sorted(set(tier_demand_cols) & set(tier_served_cols))

    panel_specs = [
        ("Slack injection", bool(slack_cols_by_sector)),
        ("Control variables (vm / pressure / t)", bool(ctrl_rows)),
        ("Line loading (%)", bool(line_cols)),
        ("Demand fulfilment by tier (fraction)", bool(tiers)),
        ("Demand fulfilment by tier (MW)", bool(tiers)),
    ]
    rows = len(panel_specs)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[t for t, _ in panel_specs],
    )

    # Panel 1: slack
    panel_idx = 1
    if slack_cols_by_sector:
        # One trace per slack child, coloured and legend-grouped by sector.
        legend_seen: set[str] = set()
        for sec in sorted(slack_cols_by_sector):
            base = _SECTOR_COLOR.get(sec, "#888888")
            for col in sorted(slack_cols_by_sector[sec]):
                aid = col.split("__", 2)[-1]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=timeseries[col].astype(float).values,
                        mode="lines",
                        line=dict(color=base, width=1.4),
                        opacity=0.85 if len(slack_cols_by_sector[sec]) <= 4 else 0.55,
                        name=sec if sec not in legend_seen else None,
                        legendgroup=sec,
                        showlegend=sec not in legend_seen,
                        hovertemplate=(
                            f"<b>slack {sec}/{aid}</b><br>"
                            "t: %{x:.2f}s<br>value: %{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=panel_idx,
                    col=1,
                )
                legend_seen.add(sec)
        fig.update_yaxes(title="MW / kg·s⁻¹", row=panel_idx, col=1)
    panel_idx += 1

    # Panel 2: control variables
    if ctrl_rows:
        # vm_pu (≈1) and t_k (≈300+) would collapse on a shared scale, so
        # normalise each to its envelope band (0=lo, 1=hi) and plot that;
        # hover reports the raw value via customdata.
        for col, label, bounds, color in ctrl_rows:
            lo, hi = bounds
            raw = timeseries[col].astype(float).values
            denom = hi - lo if hi != lo else 1.0
            norm = (raw - lo) / denom
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=norm,
                    mode="lines",
                    line=dict(color=color, width=1.6),
                    name=label,
                    customdata=raw,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "t: %{x:.2f}s<br>"
                        "raw: %{customdata:.4f}<br>"
                        "normalised: %{y:.3f}"
                        "<extra></extra>"
                    ),
                    legendgroup="ctrl",
                    showlegend=True,
                ),
                row=panel_idx,
                col=1,
            )
        # Envelope band at [0, 1] (values are normalised).
        fig.add_hrect(
            y0=0.0,
            y1=1.0,
            fillcolor="rgba(150,150,150,0.10)",
            line=dict(width=0),
            layer="below",
            row=panel_idx,
            col=1,
        )
        fig.add_hline(
            y=0.0,
            line=dict(color=_MUTED_COLOR, dash="dot", width=1),
            row=panel_idx,
            col=1,
        )
        fig.add_hline(
            y=1.0,
            line=dict(color=_MUTED_COLOR, dash="dot", width=1),
            row=panel_idx,
            col=1,
        )
        fig.update_yaxes(title="normalised to envelope", row=panel_idx, col=1)
    panel_idx += 1

    # Panel 3: line loading
    if line_cols:
        # max red, p95 orange, avg blue; 100% reference line for overloads.
        colors = {
            "max_line_loading_percent": ("#D62728", "max"),
            "p95_line_loading_percent": ("#E07A1F", "p95"),
            "avg_line_loading_percent": ("#1F4E96", "avg"),
        }
        for col in line_cols:
            c, label = colors.get(col, ("#888888", col))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=timeseries[col].astype(float).values,
                    mode="lines",
                    line=dict(color=c, width=1.6),
                    name=label,
                    legendgroup="line_loading",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{label} loading</b><br>"
                        "t: %{x:.2f}s<br>%{y:.1f} %%"
                        "<extra></extra>"
                    ),
                ),
                row=panel_idx,
                col=1,
            )
        fig.add_hline(
            y=100.0,
            line=dict(color="#D62728", dash="dash", width=1),
            row=panel_idx,
            col=1,
        )
        fig.update_yaxes(title="loading (%)", row=panel_idx, col=1, rangemode="tozero")
    panel_idx += 1

    # Panel 4: per-tier fraction
    if tiers:
        for tier in tiers:
            demand = timeseries[tier_demand_cols[tier]].astype(float).values
            served = timeseries[tier_served_cols[tier]].astype(float).values
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(demand > 1e-9, served / demand, 1.0)
            frac = np.clip(frac, 0.0, 1.05)
            color = _tier_color(tier)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=frac,
                    mode="lines",
                    line=dict(color=color, width=1.6),
                    name=f"tier {tier}",
                    legendgroup="tiers",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>tier {tier}</b><br>"
                        "t: %{x:.2f}s<br>fraction: %{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=panel_idx,
                col=1,
            )
        fig.update_yaxes(
            title="served / demand",
            row=panel_idx,
            col=1,
            range=[-0.05, 1.10],
            tickformat=".2f",
        )
    panel_idx += 1

    # Panel 5: per-tier MW
    if tiers:
        for tier in tiers:
            served = timeseries[tier_served_cols[tier]].astype(float).values
            color = _tier_color(tier)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=served,
                    mode="lines",
                    line=dict(color=color, width=1.4),
                    name=f"tier {tier} MW",
                    legendgroup="tiers_mw",
                    showlegend=False,  # legend already shows tiers in panel 4
                    hovertemplate=(
                        f"<b>tier {tier}</b><br>"
                        "t: %{x:.2f}s<br>served: %{y:.4f} MW"
                        "<extra></extra>"
                    ),
                ),
                row=panel_idx,
                col=1,
            )
        fig.update_yaxes(title="MW served", row=panel_idx, col=1, rangemode="tozero")

    # Synced event markers across every panel
    event_styles = {
        "line_failure": dict(color="#1A1A1A", dash="dash"),
        "branch_failure": dict(color="#1A1A1A", dash="dash"),
        "reconfiguration_completed": dict(color="#9467BD", dash="dot"),
        "constraint_violation": dict(color="#D62728", dash="dot"),
    }
    seen_kinds: set[str] = set()
    if not events.empty and {"t", "kind"}.issubset(events.columns):
        for kind, style in event_styles.items():
            sub = events[events["kind"] == kind]
            if sub.empty:
                continue
            seen_kinds.add(kind)
            for tx in sub["t"].astype(float).unique():
                for r in range(1, rows + 1):
                    fig.add_vline(
                        x=float(tx),
                        line=dict(color=style["color"], dash=style["dash"], width=1),
                        opacity=0.55,
                        row=r,
                        col=1,
                    )
    elif failure_t is not None:
        for r in range(1, rows + 1):
            fig.add_vline(
                x=float(failure_t),
                line=dict(color="#1A1A1A", dash="dash", width=1),
                opacity=0.55,
                row=r,
                col=1,
            )

    # Sentinel scatters for the event-kind legend (one per kind).
    for kind in sorted(seen_kinds):
        style = event_styles[kind]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=style["color"], dash=style["dash"], width=2),
                name=kind,
                legendgroup="events",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title="simulation time (s)", row=rows, col=1)
    height = 180 * rows + 100
    return _save(
        _apply_theme(fig, title=title, height=height, width=int(_FIG_WIDTH * 1.4)),
        out_path,
    )
