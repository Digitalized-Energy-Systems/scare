"""Plotly plot primitives for the evaluation.

Each function takes a slice of the summary DataFrame (or a per-task
artefact) and writes one figure as ``<name>.html`` (interactive) and
``<name>.pdf`` (vector, for chapter inclusion). Style is shared via
``_apply_theme``; variants get fixed colours for cross-figure consistency.
Returns the base path stem (``out_path`` without suffix).
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# Statuses whose sim completed with valid metrics. ``claims_failed`` means a
# fatal claim failed, not a missing measurement — excluding it is
# variant-asymmetric (the oracle is almost never claims_failed) and inflates
# weak baselines.
_COMPLETED_STATUSES = ("ok", "claims_failed")


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return df
    return df[df["status"].isin(_COMPLETED_STATUSES)]


def _ci_label(ci: float) -> str:
    """Hover/table text for a CI half-width; NaN (n=1, undefined) renders as
    an em-dash rather than ``±nan``."""
    return "—" if pd.isna(ci) else f"±{ci:.4f}"


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
# violation tally. All variable types gate (see
# ``claims.NON_GATING_CONSTRAINT_VARIABLES``, now empty), temperature included.
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
_NONGATING_VARIABLE_TYPES: frozenset[str] = frozenset()

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

# Font sizes tuned to read when the figure is scaled down to a two-column
# (half-text-width) slot: bumped ~4pt over the full-width baseline so labels
# stay legible after LaTeX shrinks the canvas.
_BASE_FONT_SIZE = 25
_TITLE_FONT_SIZE = 30
_AXIS_TITLE_FONT_SIZE = 26
_TICK_FONT_SIZE = 24
_LEGEND_FONT_SIZE = 24
_ANNOTATION_FONT_SIZE = 22

# Multi-panel trajectory / constraint-envelope figures are shown small and
# dense, so they take an extra font bump on top of the base sizes above.
_TRAJ_FONT_BUMP = 4

# Data-series line thickness and marker size for the line/scatter figures
# (sweeps, cascading, robustness, optimality). Kept well above plotly's
# defaults so a single series reads clearly at two-column scale. Reference
# lines (parity, bounds, gridlines) keep their own thin widths.
_DATA_LINE_WIDTH = 7.0
_MARKER_SIZE = 17
# Slightly thinner than ``_DATA_LINE_WIDTH`` because trajectory/envelope
# panels carry several overlaid series; still well above the old width=2.
_TRAJ_LINE_WIDTH = 3.6

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
            # Bottom margin must clear the x-axis tick labels + axis title,
            # both of which grow with the (bumped) font sizes; a fixed 64 px
            # clipped the x-title once the fonts were enlarged.
            margin=dict(
                l=84,
                r=40,
                t=top_margin,
                b=int(_TICK_FONT_SIZE + _AXIS_TITLE_FONT_SIZE + 2 * font_bump + 42),
            ),
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


def _time_axis(t_seconds: Any) -> tuple[Any, str, float]:
    """Pick a readable time unit for a trajectory x-axis from its span.

    Returns ``(scaled_values, axis_label, scale)``. ``scale`` multiplies any
    other quantity in seconds (event/failure times, stale-region bounds) so
    they stay aligned with the rescaled series. Long agent-time runs (the
    temporal experiments reach ~6 h of sim time) become unreadable in
    seconds; short runs keep seconds.
    """
    t = np.asarray(t_seconds, dtype=float)
    tmax = float(t.max()) if t.size else 0.0
    if tmax >= 7200.0:  # >= 2 h
        return t / 3600.0, "simulation time (h)", 1.0 / 3600.0
    if tmax >= 600.0:  # >= 10 min
        return t / 60.0, "simulation time (min)", 1.0 / 60.0
    return t, "simulation time (s)", 1.0


def _decimate(n: int, max_points: int = 4000) -> Any:
    """Index selector that caps a trace at ``max_points`` via a uniform stride
    (the last sample is always kept). Long temporal runs record ~50k ticks per
    task; plotting them raw makes multi-MB HTML. A uniform stride preserves the
    staircase shape of the mostly-flat inter-solve segments. Returns
    ``slice(None)`` when no decimation is needed."""
    if n <= max_points:
        return slice(None)
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


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
        f"compliant runs: {n_compliant}/{n_total} completed ({rate * 100:.0f}%)"
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
    """Height for a horizontal bar figure: enough per-category room to keep
    bars readable, plus headroom for the title + top legend + x-axis. The
    per-category allowance (and the floor/cap) is doubled over the original
    compact sizing so the bars read at twice the thickness at two-column
    scale; the fixed header allowance is kept as-is."""
    if n_series <= 1:
        per_row = 60
    else:
        per_row = 30 * n_series + 20
    return int(min(1280, max(460, per_row * n_groups + 150)))


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
    height: int | None = None,
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
            f"mean PWSF (compliant): {m:.4f}<br>95% CI: {_ci_label(c)}<br>"
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


    fig.update_layout(barmode="group", 
                      bargap=0.32, 
                      bargroupgap=0.12
    )
    fig.update_xaxes(
        range=[0, 1.05], title="priority-weighted served fraction", tickformat=".2f"
    )
    fig.update_yaxes(title="grid")
    if height is None:
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


def pwsf_by_sector_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    col_prefix: str = "outcomes__priority_weighted_fraction_by_sector__",
    title: str = "Per-sector PWSF by variant (compliant runs)",
) -> Path:
    """Grouped per-sector PWSF: one bar group per variant, a bar per sector.

    Companion to the aggregate ``variant_comparison_bar`` (the single headline
    electricity+heat+gas PWSF). This breaks the metric out by sector so each
    carrier's served fraction — including gas — is visible on its own.
    The segments are INDEPENDENT fractions in [0, 1] (no cross-sector unit
    mixing), so they are shown GROUPED (not stacked): stacking three independent
    [0,1] fractions produced a meaningless y-axis reaching ~2.3 that visually
    overstated the variant. Each bar carries its 95% CI; read each on its own.
    """
    sectors = [
        s for s in ("electricity", "gas", "heat") if f"{col_prefix}{s}" in df.columns
    ]
    if df.empty or not sectors or "variant" not in df.columns:
        return _save(_empty_fig("no per-sector PWSF data", title), out_path)

    compliant = df[_compliant_mask(df)]
    variants = sorted(compliant["variant"].dropna().unique())
    if not variants:
        return _save(_empty_fig("no compliant runs", title), out_path)

    fig = go.Figure()
    for sec in sectors:
        col = f"{col_prefix}{sec}"
        means: list[float] = []
        cis: list[float] = []
        hover: list[str] = []
        for v in variants:
            vals = compliant.loc[compliant["variant"] == v, col].dropna().tolist()
            m, ci = _mean_ci(vals)
            means.append(m)
            cis.append(ci)
            hover.append(
                f"<b>{sec}</b><br>variant: {alias_variant(v)}<br>"
                f"mean PWSF: {m:.4f}<br>95% CI: {_ci_label(ci)}<br>n={len(vals)}"
            )
        fig.add_trace(
            go.Bar(
                name=sec,
                x=[alias_variant(v) for v in variants],
                y=means,
                error_y={"type": "data", "array": cis, "visible": True},
                marker=_bar_marker(
                    _SECTOR_COLOR.get(sec, "#888888"),
                    pattern_shape=_SECTOR_PATTERN.get(sec, ""),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            )
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
    fig.update_xaxes(title="variant")
    fig.update_yaxes(
        title="per-sector PWSF",
        tickformat=".2f",
        # Headroom above 1.0 so a mean+CI near full service is not silently
        # clipped at the frame edge (per-sector PWSF is in [0, 1]).
        range=[0, 1.1],
    )
    return _save(
        _apply_theme(
            fig, title=title, height=380, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


# Combined stress-class comparison (Pillar 5)


# Canonical order + display labels for the hardest scenario classes, combined
# into a single grouped bar instead of one panel per class.
_STRESS_CLASS_ORDER: list[tuple[str, str]] = [
    ("generator_failure", "generator outage"),
    ("line_stress", "line stress"),
    ("concentrated_imbalance", "concentrated"),
    ("cold_day_stress", "cold-day"),
    ("voltage_stress", "pv-peak"),
    ("reconfiguration", "reconfiguration"),
]


# Compliance-composition categories: a run is ``valid`` when it passes both
# gates, else labelled by the sole violated *gating* constraint, or ``multi``
# when more than one gating constraint is violated. Temperature gates like the
# other envelope bounds, so it is a failure reason. (key, display label, colour,
# hatch). Colours/patterns sit alongside the variant palette without colliding
# with it (oracle green / scare blue / single-level orange / component purple),
# and every violation category carries a hatch so the stack reads at
# two-column scale; ``valid`` stays solid as the clean baseline.
_COMPLIANCE_CATS: list[tuple[str, str, str, str]] = [
    ("valid", "valid", "#7FB069", ""),
    ("slack", "slack budget", "#C44E52", "x"),
    ("voltage", "voltage", "#5B8FB0", "/"),
    ("pressure", "pressure", "#A16BB0", "\\"),
    ("temperature", "temperature", "#D62728", "."),
    ("line_load", "line loading", "#E1A44C", "-"),
    ("multi", "multi", "#555555", "+"),
]


def _compliance_category(df: pd.DataFrame) -> pd.Series:
    """Per-row compliance category over ``_COMPLIANCE_CATS`` keys."""
    slack_ok = (
        df["claims__slack_budget_compliance__passed"].fillna(False).astype(bool)
    )
    cc_ok = df["claims__constraint_compliance__passed"].fillna(False).astype(bool)

    def nv(v: str) -> pd.Series:
        c = f"claims__constraint_compliance__detail__by_variable__{v}__n_violations"
        return (df[c].fillna(0) > 0) if c in df.columns else pd.Series(
            False, index=df.index
        )

    reasons = {
        "slack": ~slack_ok,
        "voltage": nv("voltage"),
        "pressure": nv("pressure"),
        "temperature": nv("temperature"),
        "line_load": nv("line_load"),
    }
    n_reasons = sum(m.astype(int) for m in reasons.values())
    compliant = slack_ok & cc_ok
    cat = pd.Series("valid", index=df.index, dtype=object)
    cat[~compliant] = "multi"  # non-compliant with >1 (or 0) reasons
    single = (~compliant) & (n_reasons == 1)
    for key, mask in reasons.items():
        cat[single & mask] = key
    return cat


def variant_pwsf_compliance_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    groups: list[tuple[str, str]],
    group_col: str,
    group_label: str,
    metric_col: str = "outcomes__priority_weighted_fraction",
    title: str = "Served fraction and compliance by variant",
) -> Path:
    """Two-panel figure sharing a category (grid / stress class) y-axis:

    - Left: grouped bars of compliant-subset \\ac{PWSF} per variant.
    - Right: a stacked bar per variant showing the compliance composition of
      *all* completed runs -- the ``valid`` (compliant) share plus the share
      that failed on each gating constraint (or ``multi``).

    Carries two horizontal legends (variants, compliance categories) on top.
    """
    if df.empty or metric_col not in df.columns or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    have = set(df[group_col].dropna().unique())
    present = [(k, lbl) for k, lbl in groups if k in have]
    if not present:
        return _save(_empty_fig("no groups present", title), out_path)
    labels = [lbl for _, lbl in present]

    work = df.copy()
    work["_cat"] = _compliance_category(work)
    compliant = work[_compliant_mask(work)]
    variants = sorted(work["variant"].dropna().unique())

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        column_widths=[0.55, 0.45],
    )
    # Left — PWSF grouped bars.
    for variant in variants:
        means: list[float] = []
        cis: list[float] = []
        for key, _ in present:
            vals = (
                compliant[
                    (compliant[group_col] == key) & (compliant["variant"] == variant)
                ][metric_col]
                .dropna()
                .tolist()
            )
            m, ci = _mean_ci(vals) if vals else (float("nan"), 0.0)
            means.append(m)
            cis.append(0.0 if ci != ci else ci)
        fig.add_trace(
            go.Bar(
                y=labels,
                x=means,
                orientation="h",
                offsetgroup=variant,
                name=alias_variant(variant),
                legend="legend",
                marker=_bar_marker(
                    _variant_color(variant),
                    pattern_shape=_VARIANT_PATTERN.get(variant, ""),
                ),
                error_x=dict(
                    type="data",
                    array=cis,
                    visible=True,
                    thickness=1.2,
                    width=4,
                    color=_MUTED_COLOR,
                ),
                hovertemplate=(
                    f"<b>{alias_variant(variant)}</b><br>{group_label}: "
                    "%{y}<br>PWSF: %{x:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    # Right — compliance composition, stacked per variant. Only categories that
    # actually occur in this campaign get a bar + legend entry (e.g. pressure
    # never fails here), so the legend stays compact.
    present_cats = set(work["_cat"].unique())
    cats = [c for c in _COMPLIANCE_CATS if c[0] in present_cats]
    for key, lbl, color, hatch in cats:
        for vi, variant in enumerate(variants):
            fracs: list[float] = []
            for gkey, _ in present:
                sub = work[
                    (work[group_col] == gkey) & (work["variant"] == variant)
                ]
                fracs.append((sub["_cat"] == key).mean() if len(sub) else float("nan"))
            fig.add_trace(
                go.Bar(
                    y=labels,
                    x=fracs,
                    orientation="h",
                    offsetgroup=variant,
                    name=lbl,
                    legendgroup=key,
                    showlegend=(vi == 0),
                    legend="legend2",
                    marker=_bar_marker(color, pattern_shape=hatch),
                    hovertemplate=(
                        f"<b>{lbl}</b> · {alias_variant(variant)}<br>{group_label}: "
                        "%{y}<br>share: %{x:.0%}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )

    fig.update_layout(barmode="relative", bargap=0.28, bargroupgap=0.06)
    fig.update_xaxes(range=[0, 1.05], title="PWSF", tickformat=".2f", row=1, col=1)
    fig.update_xaxes(
        range=[0, 1.001], title="share of runs", tickformat=".0%", row=1, col=2
    )
    fig.update_yaxes(title=group_label, row=1, col=1)

    base_h = _hbar_height(len(labels), len(variants))
    height = base_h + 90  # top room for title + one shared legend row
    width = 1400  # wide enough for both legends side by side on one row
    themed = _apply_theme(fig, title=title, height=height, width=width, font_bump=2)
    lf = _LEGEND_FONT_SIZE
    # Two legends on the SAME horizontal row: variants left-anchored, compliance
    # categories right-anchored, no legend titles.
    legend_y = 1.0 - 62 / height
    themed.update_layout(
        title=dict(
            text=title,
            yref="container",
            yanchor="top",
            y=0.985,
            x=0.5,
            xanchor="center",
            font=dict(
                family=_TITLE_FONT_FAMILY, size=_TITLE_FONT_SIZE + 2, color=_AXIS_COLOR
            ),
        ),
        legend=dict(
            orientation="h",
            yref="container",
            xref="container",
            y=legend_y,
            x=0.01,
            xanchor="left",
            yanchor="top",
            font=dict(size=lf),
        ),
        legend2=dict(
            orientation="h",
            yref="container",
            xref="container",
            y=legend_y,
            x=0.99,
            xanchor="right",
            yanchor="top",
            font=dict(size=lf),
        ),
        margin=dict(l=90, r=40, t=130, b=80),
    )
    return _save(themed, out_path)


def stress_class_variant_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    classes: list[tuple[str, str]] | None = None,
    metric_col: str = "outcomes__priority_weighted_fraction",
    title: str = "Priority-weighted served fraction by variant across stress classes",
) -> Path:
    """One grouped horizontal bar per stress scenario class, a bar per variant.

    Combines the per-class served-by-variant panels into a single figure so the
    hardest scenarios read side by side. Compliant-subset PWSF mean with a 95%
    CI whisker; a class that ran only a subset of the variants simply omits the
    missing bars (e.g. the pv-peak class runs \\textsc{Scare} only).
    """
    if classes is None:
        classes = _STRESS_CLASS_ORDER
    if df.empty or metric_col not in df.columns or "experiment" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    have = set(df["experiment"].dropna().unique())
    present = [(e, lbl) for e, lbl in classes if e in have]
    if not present:
        return _save(_empty_fig("no stress-class experiments", title), out_path)

    compliant = df[_compliant_mask(df)]
    exps = [e for e, _ in present]
    variants = sorted(
        compliant[compliant["experiment"].isin(exps)]["variant"].dropna().unique()
    )
    labels = [lbl for _, lbl in present]

    fig = go.Figure()
    for variant in variants:
        means: list[float] = []
        cis: list[float] = []
        hover: list[str] = []
        for exp, label in present:
            vals = (
                compliant[
                    (compliant["experiment"] == exp)
                    & (compliant["variant"] == variant)
                ][metric_col]
                .dropna()
                .tolist()
            )
            if vals:
                m, ci = _mean_ci(vals)
            else:
                m, ci = float("nan"), 0.0
            means.append(m)
            cis.append(0.0 if ci != ci else ci)  # NaN CI -> 0 (single sample)
            hover.append(
                f"<b>{alias_variant(variant)}</b><br>class: {label}<br>"
                f"mean PWSF (compliant): {m:.4f}<br>95% CI: {_ci_label(ci)}<br>"
                f"n={len(vals)}"
            )
        fig.add_trace(
            go.Bar(
                name=alias_variant(variant),
                y=labels,
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
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(
        range=[0, 1.05], title="priority-weighted served fraction", tickformat=".2f"
    )
    fig.update_yaxes(title="stress scenario class")
    height = _hbar_height(len(labels), len(variants))
    # 30% wider than the standard bar canvas so the four-variant legend fits a
    # single horizontal row above the plot.
    return _save(
        _apply_theme(
            fig,
            title=title,
            height=height,
            width=int(_BAR_FIG_WIDTH * 1.3),
            font_bump=2,
            legend_top=True,
        ),
        out_path,
    )


# Optimality gap (Pillar 2)


def _gap_pair_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per single run pair: pivot on (experiment, grid, seed,
    scenario) — not (grid, seed), which silently MEANS over the scenarios
    sharing a seed and, with the compliance filter applied per row first,
    mixes asymmetric scenario subsets between the two sides. ``experiment`` is
    included so pooling several experiments (all grids) keeps each scare/oracle
    pair matched within its own experiment rather than averaging the same
    (grid, seed, scenario) across experiments; ``ablation``/``sweep`` join the
    identity for the same reason — multi-arm experiments would otherwise mean
    deliberately-degraded arms into one scare cell. Cells where either side is
    missing or non-compliant are dropped."""
    df = df[_compliant_mask(df)]
    idx = (
        (["experiment"] if "experiment" in df.columns else [])
        + ["grid", "seed"]
        + [c for c in ("scenario", "ablation", "sweep") if c in df.columns]
    )
    return df.pivot_table(index=idx, columns="variant", values=metric)


def _pair_key_label(key: Any) -> str:
    if isinstance(key, tuple):
        return f"seed: {key[0]}<br>scenario: {key[1]}"
    return f"seed: {key}"


def optimality_gap_scatter(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Optimality gap: scare vs centralised oracle (compliant pairs)"
    if df.empty:
        return _save(_empty_fig("no data", title), out_path)
    metric = "outcomes__priority_weighted_fraction"
    # Keep only run pairs where both variants are compliant — else an
    # over-drawing scare reports a misleading "negative gap" against a
    # compliant oracle.
    pivot = _gap_pair_pivot(df, metric)
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
        fig.add_trace(
            go.Scatter(
                x=sub["oracle"],
                y=sub["scare"],
                mode="markers",
                name=alias_grid(grid),
                marker=dict(
                    size=_MARKER_SIZE,
                    color=_QUAL_PALETTE[i % len(_QUAL_PALETTE)],
                    line=dict(width=1, color="white"),
                    opacity=0.9,
                ),
                customdata=[
                    f"<b>single run pair</b><br>grid: {alias_grid(grid)}<br>"
                    f"{_pair_key_label(k)}<br>oracle: {r['oracle']:.4f}<br>"
                    f"scare: {r['scare']:.4f}<br>"
                    f"gap: {(r['oracle'] - r['scare']):.4f}"
                    for k, r in sub.iterrows()
                ],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    # Mean gap annotation.
    pivot["gap"] = pivot["oracle"] - pivot["scare"]
    # Surface SCARE-beats-oracle pairs: the oracle is the constraint-respecting
    # optimum, so a negative gap signals an oracle bug or scare credited above
    # feasibility. (The scalar metrics.optimality_gap clips these to 0; this is
    # the actual reporting site, so the warning belongs here too.)
    n_neg = int((pivot["gap"] < -1e-9).sum())
    if n_neg:
        logger.warning(
            "optimality_gap_scatter: %d/%d compliant scare/oracle pairs have "
            "scare > oracle (negative gap, min %.4f) — oracle should dominate.",
            n_neg,
            len(pivot),
            float(pivot["gap"].min()),
        )
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
        text=f"<b>mean gap</b> {mean_gap:+.4f}  ·  <b>n</b>={len(pivot)} run pairs",
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
    # Same per-run pairing + compliance intersection as
    # ``optimality_gap_scatter``.
    pivot = _gap_pair_pivot(df, metric)
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
            hovertemplate="grid: %{x}<br>gap (single run pair): %{y:.4f}<extra></extra>",
        )
    fig.add_hline(y=0, line=dict(color="#BBBBBB", dash="dash", width=1))
    fig.update_yaxes(
        title="relative gap (oracle − scare) / oracle", tickformat=".2f", zeroline=False
    )
    fig.update_xaxes(title="grid")
    fig.update_layout(showlegend=False, boxgap=0.45, boxgroupgap=0.2)
    return _save(
        _apply_theme(
            fig,
            title=title,
            height=440,
            width=_box_fig_width(len(grids)),
            no_legend=True,
            font_bump=6,
        ),
        out_path,
    )


# Ablation impact (Pillar 4)


def ablation_impact_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Ablation impact (scare variant, compliant runs)",
) -> Path:
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    # Pre-filter counts so the hover can show ``n_compliant/n_total``
    # and flag ablations that break budget compliance.
    full_counts = df.groupby("ablation").size()
    df_c = df[_compliant_mask(df)]
    grouped = df_c.groupby("ablation")[metric].agg(mean="mean", count="count")
    if grouped.empty:
        return _save(_empty_fig("no compliant ablation rows", title), out_path)
    grouped["ci"] = df_c.groupby("ablation")[metric].apply(lambda s: _mean_ci(s)[1])
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
            f"95% CI: {_ci_label(c)}<br>compliant: {n}/{n_total}{rate_str}"
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
    grouped = parsed_c.groupby("x")[metric].agg(["mean", "count"])
    grouped["ci"] = parsed_c.groupby("x")[metric].apply(lambda s: _mean_ci(s)[1])
    grouped = grouped.sort_index()
    if grouped.empty:
        return _save(_empty_fig(f"no compliant {sweep_param} rows", title), out_path)

    x = grouped.index.tolist()
    y = grouped["mean"].tolist()
    # NaN CI (n=1) collapses the ribbon to the line at that point.
    band = grouped["ci"].fillna(0)
    upper = (grouped["mean"] + band).tolist()
    lower = (grouped["mean"] - band).tolist()

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
            line=dict(color=_VARIANT_COLOR["scare"], width=_DATA_LINE_WIDTH),
            marker=dict(
                size=_MARKER_SIZE, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1)
            ),
            customdata=[
                f"x={xv}<br>mean PWSF (compliant): {m:.4f}<br>95% CI: {_ci_label(c)}<br>"
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
                line=dict(color=_VARIANT_COLOR["oracle"], width=_DATA_LINE_WIDTH, dash="dot"),
                marker=dict(
                    size=_MARKER_SIZE,
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
            title="compliant (%)",
            range=[0, 105],
            color=_VARIANT_COLOR["oracle"],
            secondary_y=True,
            showgrid=False,
            ticksuffix="%",
            tickfont=dict(size=_TICK_FONT_SIZE),
            title_font=dict(size=_AXIS_TITLE_FONT_SIZE),
        )

    fig.update_yaxes(
        title="PWSF (compliant)",
        range=[0, 1.05],
        color=_VARIANT_COLOR["scare"],
        secondary_y=False,
        tickformat=".2f",
    )
    fig.update_xaxes(title=x_label)
    themed = _apply_theme(fig, title=title, font_bump=6, legend_top=True)
    # ``legend_top`` reclaims the right margin (the legend moved up top), but
    # this figure keeps a right-hand secondary axis (compliant %), so give that
    # axis its ticks + title back.
    themed.update_layout(margin_r=150)
    return _save(themed, out_path)


# RestorationConfiguration defaults (src/scare/base/config.py) — the
# ``default`` sweep arm must resolve to the parameter's actual default
# (ttl_hops=3, holon_max_size=4), not a blanket 0 that merges it into an
# explicit 0-valued arm. Unknown parameters resolve to None (row dropped).
_SWEEP_PARAM_DEFAULTS: dict[str, float] = {
    "cooldown_s": 0.0,
    "ttl_hops": 3.0,
    "cp_bridge_cost": 2.0,
    "holon_max_size": 4.0,
    "community_label_propagation_radius": 2.0,
    "comms_packet_loss_pct": 0.0,
    "comms_latency_jitter_ms": 0.0,
}


def _extract_sweep_value(sweep_key: str, param: str) -> float | None:
    if not sweep_key or sweep_key == "default":
        return _SWEEP_PARAM_DEFAULTS.get(param)
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
    grouped = df_c.groupby("n_failures")[metric].agg(["mean", "count"])
    grouped["ci"] = df_c.groupby("n_failures")[metric].apply(lambda s: _mean_ci(s)[1])
    grouped = grouped.sort_index()
    if grouped.empty:
        return _save(_empty_fig("no compliant rows", title), out_path)

    x = grouped.index.tolist()
    y = grouped["mean"].tolist()
    # NaN CI (n=1) collapses the ribbon to the line at that point.
    band = grouped["ci"].fillna(0)
    upper = (grouped["mean"] + band).tolist()
    lower = (grouped["mean"] - band).tolist()

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
            line=dict(color=_VARIANT_COLOR["scare"], width=_DATA_LINE_WIDTH),
            marker=dict(
                size=_MARKER_SIZE, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1)
            ),
            customdata=[
                f"failures: {xv}<br>mean PWSF (compliant): {m:.4f}<br>"
                f"95% CI: {_ci_label(c)}<br>compliant: {int(n)}/{int(full_counts.get(xv, n))}"
                for xv, m, c, n in zip(
                    x, grouped["mean"], grouped["ci"], grouped["count"]
                )
            ],
            hovertemplate="%{customdata}<extra></extra>",
            name="scare",
        )
    )
    fig.update_yaxes(
        title="priority-weighted served fraction (compliant)",
        range=[0, 1.05],
        tickformat=".2f",
    )
    fig.update_xaxes(title="number of simultaneous failures", dtick=1)
    fig.update_layout(showlegend=False)
    return _save(_apply_theme(fig, title=title), out_path)


# Scaling (Pillar 7) — metric vs grid size

# MES node counts for runs that predate ``outcomes.n_net_nodes`` (older
# result.json files); current runs carry the real count per task.
_GRID_MES_NODES: dict[str, int] = {
    "simbench_lv_small": 47,
    "simbench_lv_medium": 134,
    "simbench_lv": 393,
    "simbench_mvlv": 778,
}


def scaling_curve(
    df: pd.DataFrame,
    out_path: Path,
    *,
    metric: str,
    y_label: str,
    title: str,
    variants: tuple[str, ...] = ("scare", "oracle"),
) -> Path:
    """Metric vs MES node count, one mean±95%-CI ribbon line per variant —
    the sweep-style scaling read (wallclock, time-to-stabilise, ...)."""
    if df.empty or metric not in df.columns:
        return _save(_empty_fig("no data", title), out_path)

    parsed = df.copy()
    if "outcomes__n_net_nodes" in parsed.columns:
        parsed["x"] = parsed["outcomes__n_net_nodes"]
    else:
        parsed["x"] = pd.NA
    parsed["x"] = parsed["x"].fillna(parsed["grid"].map(_GRID_MES_NODES))
    parsed = parsed.dropna(subset=["x", metric])
    if parsed.empty:
        return _save(_empty_fig("no node-count data", title), out_path)

    fig = go.Figure()
    for variant in variants:
        sub = parsed[parsed["variant"] == variant]
        if sub.empty:
            continue
        grouped = sub.groupby("x")[metric].agg(["mean", "count"])
        grouped["ci"] = sub.groupby("x")[metric].apply(lambda s: _mean_ci(s)[1])
        grouped = grouped.sort_index()
        x = grouped.index.tolist()
        band = grouped["ci"].fillna(0)
        upper = (grouped["mean"] + band).tolist()
        lower = (grouped["mean"] - band).tolist()
        color = _VARIANT_COLOR.get(variant, _VARIANT_COLOR["scare"])
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.16),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=grouped["mean"].tolist(),
                mode="lines+markers",
                name=variant,
                line=dict(color=color, width=_DATA_LINE_WIDTH),
                marker=dict(
                    size=_MARKER_SIZE, color=color, line=dict(color="white", width=1)
                ),
                customdata=[
                    f"nodes: {int(xv)}<br>{y_label}: {m:.2f}<br>"
                    f"95% CI: {_ci_label(c)}<br>n: {int(n)}"
                    for xv, m, c, n in zip(
                        x, grouped["mean"], grouped["ci"], grouped["count"]
                    )
                ],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
    fig.update_yaxes(title=y_label, rangemode="tozero")
    fig.update_xaxes(title="MES nodes")
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
            line=dict(color=_VARIANT_COLOR["scare"], width=_DATA_LINE_WIDTH),
            marker=dict(
                size=_MARKER_SIZE, color=_VARIANT_COLOR["scare"], line=dict(color="white", width=1)
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
                line=dict(color=_VARIANT_COLOR["oracle"], width=_DATA_LINE_WIDTH, dash="dot"),
                marker=dict(
                    size=_MARKER_SIZE,
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
                line=dict(color=_VARIANT_COLOR["single_level"], width=_DATA_LINE_WIDTH, dash="dot"),
                marker=dict(
                    size=_MARKER_SIZE,
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
        title="PWSF (compliant)",
        range=[0, 1.05],
        color=_VARIANT_COLOR["scare"],
        secondary_y=False,
        tickformat=".2f",
    )
    fig.update_xaxes(title=x_label)
    themed = _apply_theme(fig, title=title, font_bump=4, legend_top=True)
    # ``legend_top`` reclaims the right margin, but this figure keeps a
    # right-hand secondary axis (wallclock), so give that axis its room back.
    themed.update_layout(margin_r=150)
    return _save(themed, out_path)


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
            fig, title=title, height=240, width=_BAR_FIG_WIDTH, legend_top=True
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
    x_vals, x_title, x_scale = _time_axis(timeseries["time_s"].values)
    x_hover = x_title.split("(")[-1].rstrip(")")
    sectors = [
        ("electrical_balance", "electricity"),
        ("gas_balance", "gas"),
        ("heat_balance", "heat"),
    ]
    for col, sec in sectors:
        if col in timeseries.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=timeseries[col],
                    mode="lines",
                    name=sec,
                    line=dict(color=_SECTOR_COLOR[sec], width=_TRAJ_LINE_WIDTH),
                    hovertemplate=f"<b>{sec}</b><br>t: %{{x:.2f}}{x_hover}<br>balance: %{{y:.4f}}<extra></extra>",
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
                    x=float(tx) * x_scale,
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
        fig.add_vline(
            x=failure_t * x_scale, line=dict(color="#1A1A1A", dash="dash", width=1)
        )

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Σ regulation per sector")
    return _save(
        _apply_theme(
            fig, title=title, height=360, legend_top=True, font_bump=_TRAJ_FONT_BUMP
        ),
        out_path,
    )


# Claims pass-rate (Pillar 8)


def claims_pass_rate(df: pd.DataFrame, out_path: Path) -> Path:
    title = "Claims validation pass rate by variant"
    if df.empty:
        return _save(_empty_fig("no data", title), out_path)
    # Oracle is a one-shot LP with no MAS dispatch trajectory, so it emits only
    # a partial claim set (no priority/monotonic/diary invariants). Its blank
    # cells read as "0% fail" rather than "not applicable"; drop it entirely.
    if "variant" in df.columns:
        df = df[df["variant"] != "oracle"]
    claim_cols = [
        c for c in df.columns if c.startswith("claims__") and c.endswith("__passed")
    ]
    if not claim_cols:
        return _save(_empty_fig("no claims data", title), out_path)

    rows = []
    for col in claim_cols:
        claim_name = col[len("claims__") : -len("__passed")]
        for variant, g in df.groupby("variant"):
            vals = g[col].dropna()
            n = vals.shape[0]
            if n == 0:
                continue
            # Count passes only among GRADED rows. ``astype(bool)`` on the raw
            # column maps NaN (claim not evaluated, e.g. oracle) to True, which
            # inflates the rate and can exceed 1.0; drop NaN first.
            rate = float(vals.astype(bool).sum()) / n
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

    # Ratio overlays only when their columns exist — substituting the raw
    # post-MW mean would put megawatts on the ratio axis.
    agg_spec: dict[str, tuple[str, str]] = {
        "baseline_mw": (base_col, "mean"),
        "post_mw": (post_col, "mean"),
        "n": (base_col, "count"),
    }
    if raw_col in sub.columns:
        agg_spec["raw_ratio"] = (raw_col, "mean")
    if pwsf_col in sub.columns:
        agg_spec["pwsf_ratio"] = (pwsf_col, "mean")
    grouped = sub.groupby("grid").agg(**agg_spec).sort_values(
        "baseline_mw", ascending=False
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
    title: str = "Per-tier served fraction (post / demand, MW)",
) -> Path:
    """Grouped bars per grid × tier: fraction of each tier's demand served
    post-restoration. Tier 1 (most critical) sitting below the others means
    the protocol is failing the loads that matter most.

    Uses ``post_fraction`` (absolute, against demand). The legacy
    baseline-relative ``ratio`` divides by a priority-aware, slack-budget-
    bounded baseline that already sheds low tiers, saturating them at 1.0 and
    fabricating an apparent tier inversion; it remains only as a fallback for
    campaigns that predate ``post_fraction``.
    """
    if df.empty or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    sub = df[df["variant"] == "scare"] if "scare" in df["variant"].unique() else df

    prefix = "outcomes__restoration__by_tier__"
    tier_cols: dict[int, str] = {}
    y_title = "served fraction (post / demand)"
    for suffix in ("__post_fraction", "__ratio"):
        for col in sub.columns:
            if col.startswith(prefix) and col.endswith(suffix):
                try:
                    tier = int(col[len(prefix) : -len(suffix)])
                except ValueError:
                    continue
                tier_cols[tier] = col
        if tier_cols:
            if suffix == "__ratio":
                title = "Per-tier restoration ratio (post / baseline served, MW)"
                y_title = "restoration ratio (post / baseline served)"
            break
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
                hovertemplate=f"<b>tier {tier}</b><br>grid: %{{x}}<br>served: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line=dict(color="#BBBBBB", dash="dash", width=1))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(
        title=y_title,
        range=[0, 1.05],
        tickformat=".2f",
    )
    fig.update_xaxes(title="grid")
    return _save(
        _apply_theme(
            fig, title=title, height=480, width=_BAR_FIG_WIDTH, legend_top=True
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
    - ``constraint_lost`` (priority-blind): served the agents dispatched but
      the eval feasibility gates removed (heat t_k out of bounds, line
      overload de-rate) — physics-throttled, no agent lever.
    - ``agent_shed`` (priority-aware): load the QP / ADMM layers chose to
      drop — the only contribution priority weighting controls.

    The "tier 1 protected, tier 10 sheds first" claim applies to
    ``agent_shed`` only: a tier-1 bar dominated by the priority-blind
    shares is a topology / physics limit, not a priority-machinery failure.
    """
    if df.empty or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    sub = df[df["variant"] == "scare"] if "scare" in df["variant"].unique() else df

    # Discover tiers from the per-tier disconnect columns.
    disc_pat = "outcomes__restoration__by_tier__"
    disc_suf = "__disconnect_lost_mw"
    cap_suf = "__constraint_lost_mw"
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

    def _tier_means(suffix: str) -> list[float]:
        means: list[float] = []
        for t in tiers:
            col = f"{disc_pat}{t}{suffix}"
            v = sub[col].dropna() if col in sub.columns else pd.Series(dtype=float)
            means.append(float(v.mean()) if len(v) else 0.0)
        return means

    disc_means = _tier_means(disc_suf)
    cap_means = _tier_means(cap_suf)
    agt_means = _tier_means(agt_suf)

    tier_labels = [f"tier {t}" for t in tiers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="physical disconnect (priority-blind)",
            y=tier_labels,
            x=disc_means,
            orientation="h",
            # Hatch the priority-blind losses so they read apart from the
            # agent-controlled share in greyscale too.
            marker=_bar_marker("#999999", pattern_shape="x"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "disconnect lost (mean per task): %{x:.3f} MW<extra></extra>"
            ),
        )
    )
    if any(v > 0 for v in cap_means):
        fig.add_trace(
            go.Bar(
                name="physics-throttled (feasibility gate)",
                y=tier_labels,
                x=cap_means,
                orientation="h",
                marker=_bar_marker("#C4A35A", pattern_shape="/"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "constraint lost (mean per task): %{x:.3f} MW<extra></extra>"
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
    height = int(_hbar_height(len(tier_labels)) * 0.7)  # -30%: compact next to the taller vertical panels
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
    title: str = (
        "Per-tier restoration ratio (agent-shed only — disconnect + gate excluded)"
    ),
) -> Path:
    """Per-tier bars of ``agent_only_ratio`` — the share of controllable
    baseline each tier kept after removing physically-disconnected and
    feasibility-gate-throttled load from the denominator. Isolates the
    priority signal from topology/physics noise: if the claim holds, tier 1
    sits near 1.0 and the curve slopes down monotonically. Note: still
    baseline-relative (see ``restoration_by_tier_bar`` for the absolute view).
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
            fig, title=title, height=480, width=_BAR_FIG_WIDTH, legend_top=True
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
    # Ratio can exceed 1 (post-restoration served above the no-failure baseline
    # for a grid·variant mean, observed up to ~1.06); cap above 1.1 so such bars
    # are not silently clipped at the frame edge.
    fig.update_xaxes(title="raw restoration ratio", range=[0, 1.15], tickformat=".2f")
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

    # The "% of baseline" labels need the real baseline column; faking it
    # from max(dropped) put a wrong share on the figure.
    has_base = base_col in sub.columns
    agg_spec: dict[str, tuple[str, str]] = {
        "dropped_mw": (col, "mean"),
        "n": (col, "count"),
    }
    if has_base:
        agg_spec["baseline_mw"] = (base_col, "mean")
    grouped = sub.groupby("grid").agg(**agg_spec).sort_values(
        "dropped_mw", ascending=False
    )

    if has_base:
        pct = (grouped["dropped_mw"] / grouped["baseline_mw"]).where(
            grouped["baseline_mw"] > 0, 0.0
        )
        text = [f"{p * 100:.1f}%" for p in pct]
        hover = [
            f"<b>{alias_grid(g)}</b><br>dropped: {d:.3f} MW<br>baseline: {b:.3f} MW<br>"
            f"share: {p * 100:.1f}%<br>n: {int(n)}"
            for g, d, b, p, n in zip(
                grouped.index,
                grouped["dropped_mw"],
                grouped["baseline_mw"],
                pct,
                grouped["n"],
            )
        ]
    else:
        text = None
        hover = [
            f"<b>{alias_grid(g)}</b><br>dropped: {d:.3f} MW<br>n: {int(n)}"
            for g, d, n in zip(grouped.index, grouped["dropped_mw"], grouped["n"])
        ]

    fig = go.Figure(
        go.Bar(
            y=_grids_display(list(grouped.index)),
            x=grouped["dropped_mw"].values,
            orientation="h",
            marker=_bar_marker("#701E96"),
            text=text,
            textposition="outside",
            textfont=dict(size=_ANNOTATION_FONT_SIZE),
            cliponaxis=False,
            customdata=hover,
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

    # Normalise to per-task averages: variants have unequal task counts (~2x in
    # the variant-comparison run, more in ablation campaigns), so raw count SUMS
    # are not comparable (a variant with more tasks looks busier regardless of
    # behaviour). Dividing by each variant's task count gives the mean number of
    # negotiations of each outcome PER TASK, which is comparable.
    n_tasks = df.groupby("variant").size()
    by_variant = (
        df.groupby("variant")[[c[0] for c in cols]].sum().div(n_tasks, axis=0)
    )
    if by_variant.empty:
        return _save(_empty_fig("no diary data", title), out_path)

    fig = go.Figure()
    variants_lbl = _variants_display(list(by_variant.index))
    n_by_variant = [int(n_tasks.loc[v]) for v in by_variant.index]
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
                customdata=n_by_variant,
                hovertemplate=(
                    f"<b>{label}</b><br>variant: %{{y}}<br>"
                    "per-task: %{x:.2f}<br>n_tasks=%{customdata}<extra></extra>"
                ),
            )
        )
    fig.update_layout(barmode="stack", bargap=0.42)
    fig.update_yaxes(title="variant")
    fig.update_xaxes(title="negotiations per task", rangemode="tozero")
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
    # Completed sims only: a crashed/timed-out task has no meaningful solver
    # tally, and fillna(0) would count it as a clean zero-event run.
    sub = _completed(df).dropna(subset=["variant"]).copy()
    if sub.empty:
        return _save(_empty_fig("no completed rows with a variant", title), out_path)

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


# Coupling-point optimization influence — how much the L3 cross-sector ADMM
# actually steers each run, next to what that steering buys in restored load.


def cp_influence_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = (
        "Coupling-point optimization — activity and restored load "
        "(all runs, by grid)"
    ),
) -> Path:
    """Two panels, shared grid axis, pooled over every completed run.

    LEFT — CP contribution: mean delivered coupling-point converter output
    per task in MW, from ``outcomes.cp_generation`` (end-of-sim solved net;
    el + heat + gas-as-energy across every CHP / P2H / G2P / P2G / G2H unit),
    STACKED BY CARRIER (gas at the base, then el, then heat). Colour keys the
    method (as in the right panel); the hatch + stack position key the carrier.
    Splitting by carrier is deliberate: the raw oracle>scare total is ~97% gas
    (PowerToGas), and that gas block is NOT a served-load signal — the oracle
    solves converter ``regulation`` as a free variable that carries no weight in
    the min-load-shedding objective (``include_coupling_points`` is left False),
    and gas Sinks are absent from served accounting, so the oracle's gas
    dispatch is closer to an unpriced free-variable resting point than a usage
    ceiling. The only load-relevant carrier is heat (small MW, but the piece
    that tracks the PWSF gap); reading it against the gas block it sits on keeps
    "SCARE under-uses its CPs" from being over-read off the total. On campaigns
    recorded before ``cp_generation`` existed the panel falls back to a single
    CP-ADMM regulate-action-count bar from ``outcomes.regulates_by_reason``
    (``cp*`` reasons; oracle dropped there — it never fires the regulate path).

    RIGHT — restored load: compliant-mean PWSF for the same (grid, variant)
    cells, so a grid steered heavily through its coupling points can be read
    against what that steering buys. The scare hover carries the Δ vs each
    L3-less baseline; NOTE both baselines (``component_level``,
    ``single_level``) disable the holonic layer TOO, so the gap is the joint
    L2+L3 coordination lift — an upper bound on, not an isolation of, the CP
    contribution (no variant toggles CP alone; see the
    ``enable_cp_admm=False`` ablation for the scare-internal toggle).
    """
    prefix = "outcomes__regulates_by_reason__"
    cp_cols = [
        c
        for c in df.columns
        if c.startswith(prefix) and c[len(prefix) :].startswith("cp")
    ]
    mw_col = "outcomes__cp_generation__total_mw"
    use_mw = mw_col in df.columns and bool(
        pd.to_numeric(df[mw_col], errors="coerce").notna().any()
    )
    pwsf_col = "outcomes__priority_weighted_fraction"
    if df.empty or (not cp_cols and not use_mw) or "grid" not in df.columns:
        return _save(_empty_fig("no CP generation / regulate data", title), out_path)

    sub = df.dropna(subset=["variant"]).copy()
    if not use_mw:
        # Action-count fallback: the oracle never fires the regulate path, so
        # its bar would be an empty artefact. With MW recorded it stays in as
        # the CP-usage ceiling.
        sub = sub[sub["variant"] != "oracle"]
    if sub.empty:
        return _save(_empty_fig("no rows with a variant", title), out_path)

    if use_mw:
        sub["_cp_actions"] = pd.to_numeric(sub[mw_col], errors="coerce")
    else:
        sub["_cp_actions"] = sub[cp_cols].fillna(0).astype(float).sum(axis=1)
    total_col = "outcomes__regulates_total"
    if total_col in sub.columns:
        sub["_reg_total"] = sub[total_col].fillna(0).astype(float)
    else:
        sub["_reg_total"] = float("nan")

    grouped = sub.groupby(["grid", "variant"])
    cp_vals = grouped["_cp_actions"].apply(lambda s: list(s.dropna()))
    tot_sum = grouped["_reg_total"].sum()
    # Per-carrier breakdown + active-unit counts for the MW-mode hover.
    bd_means: dict[str, Any] = {}
    if use_mw:
        for key in ("el_mw", "heat_mw", "gas_mw", "n_active", "n_cp"):
            col = f"outcomes__cp_generation__{key}"
            if col in sub.columns:
                bd_means[key] = grouped[col].apply(
                    lambda s: pd.to_numeric(s, errors="coerce").mean()
                )
    # Restored load on the compliant subset — same gate as the variant tables.
    have_pwsf = pwsf_col in sub.columns
    if have_pwsf:
        comp = sub[_compliant_mask(sub)]
        pwsf_vals = comp.groupby(["grid", "variant"])[pwsf_col].apply(
            lambda s: list(s.dropna())
        )
    grids = sorted({k[0] for k in cp_vals.index})
    variants = sorted({k[1] for k in cp_vals.index})
    grids_lbl = _grids_display(grids)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=(
            (
                "CP generation delivered (MW) / task — stacked by carrier"
                if use_mw
                else "CP-ADMM converter actions / task"
            ),
            "restored load — PWSF (compliant mean)",
        ),
    )

    # Pre-compute compliant PWSF means so the scare hover can quote deltas.
    pwsf_mean: dict[tuple[str, str], float] = {}
    if have_pwsf:
        for (grid, variant), vals in pwsf_vals.items():
            if vals:
                pwsf_mean[(grid, variant)] = float(pd.Series(vals).mean())

    # LEFT panel stacks the delivered MW BY CARRIER. Stack order puts the
    # PWSF-neutral gas block at the base with the load-relevant heat/el on top:
    # the oracle>scare CP-MW gap is ~97% gas (a converter free variable the shed
    # LP never prices, so the oracle's gas dispatch is not a served-load optimum
    # — see the module docstring), and gas Sinks are absent from served
    # accounting, so a single total bar reads as "scare under-uses CPs" when the
    # only load-relevant gap (heat) is small. Colour stays keyed to method (as in
    # the right panel); carrier is the hatch + stack channel.
    carrier_stack = (("gas", "gas_mw"), ("el", "el_mw"), ("heat", "heat_mw"))
    carrier_label = {"gas": "gas (P2G)", "el": "electricity", "heat": "heat"}
    carrier_sector = {"gas": "gas", "el": "electricity", "heat": "heat"}
    # Same method hue for every segment (variant is the colour, as in the right
    # panel); carrier separates by fill opacity + hatch. The gradient is
    # semantic: the PWSF-neutral gas base is faded, the load-relevant heat cap is
    # solid, so the eye is drawn to the carrier that actually tracks restored
    # load rather than to the (usually larger) gas block.
    carrier_alpha = {"gas": 0.5, "el": 0.72, "heat": 1.0}

    def _nz(v: Any) -> float:
        return 0.0 if v is None or v != v else float(v)

    for variant in variants:
        cp_means: list[float] = []
        cp_cis: list[float] = []
        shares: list[float] = []
        ns: list[int] = []
        hover_cp: list[str] = []
        carrier_hover: dict[str, list[str]] = {c: [] for c, _ in carrier_stack}
        for grid, g_lbl in zip(grids, grids_lbl):
            vals = cp_vals.get((grid, variant), [])
            mean, ci = _mean_ci(vals)
            cp_means.append(mean)
            cp_cis.append(ci)
            total = float(tot_sum.get((grid, variant), 0.0))
            cp_total = float(sum(vals))
            shares.append(cp_total / total if total > 0 else float("nan"))
            ns.append(len(vals))
            if use_mw:
                by_c = {
                    "el": _nz(bd_means.get("el_mw", {}).get((grid, variant))),
                    "heat": _nz(bd_means.get("heat_mw", {}).get((grid, variant))),
                    "gas": _nz(bd_means.get("gas_mw", {}).get((grid, variant))),
                }
                n_act = _nz(bd_means.get("n_active", {}).get((grid, variant)))
                n_cp = _nz(bd_means.get("n_cp", {}).get((grid, variant)))
                units = (
                    f"active CP units: {n_act:.1f} of {n_cp:.0f}<br>" if n_cp else ""
                )
                for c, _mk in carrier_stack:
                    carrier_hover[c].append(
                        f"<b>{alias_variant(variant)}</b> — {carrier_label[c]}<br>"
                        f"grid: {g_lbl}<br>"
                        f"{carrier_label[c]}: {by_c[c]:.3f} MW/task<br>"
                        f"all-carrier total: {mean:.3f} MW/task "
                        f"(95% CI {_ci_label(ci)})<br>" + units + f"n = {len(vals)}"
                    )
            else:
                share = shares[-1]
                hover_cp.append(
                    f"<b>{alias_variant(variant)}</b><br>grid: {g_lbl}<br>"
                    f"mean CP-ADMM actions/task: {mean:.1f}<br>"
                    f"95% CI: {_ci_label(ci)}<br>"
                    + (
                        f"share of all regulate actions: {share * 100:.1f}%<br>"
                        if share == share  # NaN-safe
                        else ""
                    )
                    + f"n = {len(vals)}"
                )
        if use_mw:
            for c, mw_key in carrier_stack:
                xs = [
                    _nz(bd_means.get(mw_key, {}).get((grid, variant)))
                    for grid in grids
                ]
                fig.add_trace(
                    go.Bar(
                        name=alias_variant(variant),
                        legendgroup=variant,
                        offsetgroup=variant,
                        showlegend=False,
                        y=grids_lbl,
                        x=xs,
                        orientation="h",
                        marker=_bar_marker(
                            _hex_to_rgba(
                                _variant_color(variant), carrier_alpha[c]
                            ),
                            pattern_shape=_SECTOR_PATTERN.get(
                                carrier_sector[c], ""
                            ),
                        ),
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=carrier_hover[c],
                    ),
                    row=1,
                    col=1,
                )
        else:
            fig.add_trace(
                go.Bar(
                    name=alias_variant(variant),
                    legendgroup=variant,
                    y=grids_lbl,
                    x=cp_means,
                    orientation="h",
                    error_x=dict(
                        type="data",
                        array=cp_cis,
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
                    customdata=hover_cp,
                ),
                row=1,
                col=1,
            )

        if not have_pwsf:
            continue
        p_means: list[float] = []
        p_cis: list[float] = []
        p_ns: list[int] = []
        hover_p: list[str] = []
        for grid, g_lbl in zip(grids, grids_lbl):
            vals = pwsf_vals.get((grid, variant), []) if have_pwsf else []
            mean, ci = _mean_ci(vals)
            p_means.append(mean)
            p_cis.append(ci)
            p_ns.append(len(vals))
            deltas = ""
            if variant == "scare":
                for other in variants:
                    if other == variant:
                        continue
                    o_mean = pwsf_mean.get((grid, other))
                    s_mean = pwsf_mean.get((grid, variant))
                    if o_mean is not None and s_mean is not None:
                        deltas += (
                            f"Δ vs {alias_variant(other)}: "
                            f"{s_mean - o_mean:+.4f}<br>"
                        )
                if deltas:
                    deltas = (
                        "<i>joint L2+L3 lift (baselines also drop the "
                        "holonic layer):</i><br>" + deltas
                    )
            hover_p.append(
                f"<b>{alias_variant(variant)}</b><br>grid: {g_lbl}<br>"
                f"mean PWSF (compliant): {mean:.4f}<br>"
                f"95% CI: {_ci_label(ci)}<br>n = {len(vals)}<br>" + deltas
            )
        fig.add_trace(
            go.Bar(
                name=alias_variant(variant),
                legendgroup=variant,
                offsetgroup=variant,
                # In MW mode the left panel's bars are carrier-stacked and carry
                # no legend, so the variant key is sourced here.
                showlegend=use_mw,
                y=grids_lbl,
                x=p_means,
                orientation="h",
                error_x=dict(
                    type="data",
                    array=p_cis,
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
                customdata=hover_p,
            ),
            row=1,
            col=2,
        )

    if use_mw:
        # Carrier key for the stacked left panel (grey swatches, hatch = carrier).
        # Purely informational — real bars stay grouped by method for toggling.
        for c, _mk in carrier_stack:
            fig.add_trace(
                go.Bar(
                    name=f"{carrier_label[c]} (fill)",
                    legendgroup="__carrier__",
                    y=[None],
                    x=[None],
                    orientation="h",
                    marker=_bar_marker(
                        _hex_to_rgba("#777777", carrier_alpha[c]),
                        pattern_shape=_SECTOR_PATTERN.get(carrier_sector[c], ""),
                        pattern_fg="#333333",
                    ),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        barmode="stack" if use_mw else "group", bargap=0.32, bargroupgap=0.12
    )
    fig.update_xaxes(
        title_text=(
            "mean delivered MW per task" if use_mw else "mean actions per task"
        ),
        rangemode="tozero",
        tickformat=".2f" if use_mw else ".0f",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="PWSF (compliant mean)",
        range=[0, 1.05],
        tickformat=".2f",
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="grid", row=1, col=1)
    height = _hbar_height(len(grids), len(variants)) + 30
    return _save(
        _apply_theme(
            fig,
            title=title,
            height=height,
            width=int(_BAR_FIG_WIDTH * 1.55),
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
    height = int(_hbar_height(len(grids), len(sectors)) * 0.7)  # -30%: compact next to the taller vertical panels
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
        DEENERGISED_VM_PU,
    )

    out = np.asarray(vals, dtype=float).copy()
    if avg_col.endswith("pressure_pu"):
        out[(out <= DEENERGISED_PRESSURE_PU) | (out >= DEENERGISED_PRESSURE_HIGH_PU)] = (
            np.nan
        )
    elif avg_col.endswith("t_k"):
        out[out <= 0.0] = np.nan
    elif avg_col.endswith("vm_pu"):
        # An electricity node cut off from its slack collapses to vm_pu~0; that
        # is de-energisation, not an under-voltage violation, so blank it rather
        # than plot a spurious dive through the operating band.
        out[out <= DEENERGISED_VM_PU] = np.nan
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

    # Decimate long runs (stale detection above already used the full trace).
    _dec = _decimate(len(timeseries))
    if not isinstance(_dec, slice):
        timeseries = timeseries.iloc[_dec].reset_index(drop=True)
    x, x_title, x_scale = _time_axis(timeseries["time_s"].values)
    x_list = list(x)
    x_band = x_list + x_list[::-1]
    # Stale boundary in the (possibly rescaled) x-axis unit.
    stale_x = None if stale_from_t is None else stale_from_t * x_scale
    x_hover = x_title.split("(")[-1].rstrip(")")

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
        if stale_x is not None:
            x_max = float(x[-1])
            if x_max > stale_x:
                fig.add_vrect(
                    x0=stale_x,
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
        # Also mask the AVG line: a de-energised network collapses avg_vm_pu to
        # ~0 etc., which otherwise draws a spurious dive through the operating
        # band (the same de-energisation the violation integral now excludes).
        avg_vals = _mask_deenergised(avg_col, avg_vals)

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
            if stale_x is not None:
                fresh_mask = x <= stale_x
                stale_mask = x >= stale_x
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
                                f"t: %{{x:.2f}}{x_hover}<br>value: %{{y:.4f}}<extra></extra>"
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
                                f"t: %{{x:.2f}}{x_hover}<br>value: %{{y:.4f}}<extra></extra>"
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
                            f"t: %{{x:.2f}}{x_hover}<br>value: %{{y:.4f}}<extra></extra>"
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
                width=2.4,
                legend_group=LG_MIN,
                legend_name="min (across nodes)",
                hover_label="min",
            )
        if have_max:
            _add_series(
                max_vals,
                dash=_MAX_DASH,
                width=2.4,
                legend_group=LG_MAX,
                legend_name="max (across nodes)",
                hover_label="max",
            )
        _add_series(
            avg_vals,
            dash=_AVG_DASH,
            width=_TRAJ_LINE_WIDTH,
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
                        x=float(tx) * x_scale,
                        line=dict(color=style["color"], dash=style["dash"], width=1),
                        opacity=0.6,
                        row=row_idx,
                        col=1,
                    )
        elif failure_t is not None:
            fig.add_vline(
                x=failure_t * x_scale,
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

    fig.update_xaxes(title=x_title, row=len(present), col=1)
    height = max(_FIG_HEIGHT, 200 * len(present) + 100)
    return _save(
        _apply_theme(
            fig, title=title, height=height, legend_top=True, font_bump=_TRAJ_FONT_BUMP
        ),
        out_path,
    )


# Constraint-violation integral by sector × variant


def constraint_violation_integral_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Constraint-violation integral by sector",
) -> Path:
    """Grouped bars: per-variant mean of the per-sector violation
    integral ``∫ max(0, util(t) − 1) dt`` (from
    :func:`metrics.constraint_violation_integral`, de-energised samples
    excluded). Zero ⇔ the sector never left its envelope on average; larger ⇔
    longer/deeper excursions.

    The ORACLE is excluded: it is a single static LP solve with no time series,
    so its integral is a hardcoded 0 (never computed) — plotting it would imply a
    perfect score and make the time-evolving MAS variants look spuriously worse.
    """
    sectors = ["electricity", "gas", "heat"]
    cols = {s: f"outcomes__constraint_violation_integral__{s}" for s in sectors}
    present = [s for s in sectors if cols[s] in df.columns]
    if df.empty or not present:
        return _save(
            _empty_fig("no constraint_violation_integral columns", title), out_path
        )
    # Completed sims only — a crashed task has no integral; fillna(0) would
    # count it as a zero-violation run.
    sub = _completed(df).dropna(subset=["variant"]).copy()
    # Drop rows whose integral is vacuous. Preferred: an explicit validity
    # flag (newer payloads); fallback: the oracle, whose integral is a
    # hardcoded 0 (single static LP solve, no time series).
    valid_col = next(
        (
            c
            for c in (
                "outcomes__constraint_violation_integral_valid",
                "outcomes__constraint_violation_integral__valid",
            )
            if c in sub.columns
        ),
        None,
    )
    had_oracle = bool((sub["variant"] == "oracle").any())
    if valid_col is not None:
        sub = sub[sub[valid_col].fillna(False).astype(bool)]
    else:
        sub = sub[sub["variant"] != "oracle"]
    oracle_dropped = had_oracle and not bool((sub["variant"] == "oracle").any())
    if sub.empty:
        return _save(_empty_fig("no rows with a valid integral", title), out_path)
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
    if oracle_dropped:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.99,
            y=-0.28,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            text=(
                "oracle omitted: its integral is never computed "
                "(static LP, no time series) — absence ≠ zero violations"
            ),
            font=dict(
                family=_FONT_FAMILY, size=_ANNOTATION_FONT_SIZE - 2,
                color=_MUTED_COLOR,
            ),
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

    ``temperature`` (heat ``t_k``) gates like the other envelope bounds: a
    temperature-infeasible node is a compliance failure, counted here alongside
    voltage, pressure, and line loading.

    Reads the flattened claim-detail columns (see ``_VIOLATION_VAR_COLS``);
    missing columns count as zero so a campaign that never recorded a given
    variable simply shows an empty bar.
    """
    if df.empty or "variant" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    # Completed sims only — a crashed task never got graded; fillna(0) would
    # count it as a zero-violation run.
    sub = _completed(df).dropna(subset=["variant"]).copy()
    if sub.empty:
        return _save(_empty_fig("no completed rows with a variant", title), out_path)

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
    x, x_title, _ = _time_axis(timeseries["time_s"].values)
    x_hover = x_title.split("(")[-1].rstrip(")")
    _dec = _decimate(len(x))
    x = x[_dec]
    for row_idx, sec in enumerate(sectors, start=1):
        cols = sorted(by_sector[sec])
        base = _SECTOR_COLOR.get(sec, "#888888")
        for i, col in enumerate(cols):
            # Fade opacity when many groups overlap.
            opacity = 0.55 if len(cols) > 6 else 0.85
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=timeseries[col].astype(float).values[_dec],
                    mode="lines",
                    line=dict(color=base, width=_TRAJ_LINE_WIDTH),
                    opacity=opacity,
                    name=col.split("__")[-1],
                    legendgroup=sec,
                    legendgrouptitle_text=sec,
                    showlegend=(row_idx == 1 and i < 10),  # avoid legend explosion
                    hovertemplate=(
                        f"<b>{series_label} {sec}/{col.split('__')[-1]}</b><br>"
                        f"t: %{{x:.2f}}{x_hover}<br>Σ regulation: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )
        fig.update_yaxes(title=f"Σ reg. ({sec})", row=row_idx, col=1)

    fig.update_xaxes(title=x_title, row=len(sectors), col=1)
    height = max(_FIG_HEIGHT, 170 * len(sectors) + 80)
    return _save(_apply_theme(fig, title=title, height=height, legend_top=True, font_bump=_TRAJ_FONT_BUMP), out_path)


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
    x, x_title, x_scale = _time_axis(timeseries["time_s"].values)
    x_hover = x_title.split("(")[-1].rstrip(")")
    _dec = _decimate(len(x))
    x = x[_dec]
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
                    y=timeseries[col].astype(float).values[_dec],
                    mode="lines",
                    line=dict(color=base, width=_TRAJ_LINE_WIDTH),
                    opacity=opacity,
                    name=aid,
                    legendgroup=sec,
                    legendgrouptitle_text=sec,
                    showlegend=(row_idx == 1 and i < 10),
                    hovertemplate=(
                        f"<b>slack {sec}/{aid}</b><br>"
                        f"t: %{{x:.2f}}{x_hover}<br>value: %{{y:.4f}}<extra></extra>"
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
                x=float(failure_t) * x_scale,
                line=dict(color="#888888", dash="dash", width=1),
                row=row_idx,
                col=1,
            )

    fig.update_xaxes(title=x_title, row=len(sectors), col=1)
    height = max(_FIG_HEIGHT, 170 * len(sectors) + 80)
    return _save(_apply_theme(fig, title=title, height=height, legend_top=True, font_bump=_TRAJ_FONT_BUMP), out_path)


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

    x, x_title, x_scale = _time_axis(timeseries["time_s"].values)
    x_hover = x_title.split("(")[-1].rstrip(")")
    _dec = _decimate(len(x))
    x = x[_dec]
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
                y=timeseries[col].astype(float).values[_dec],
                mode="lines",
                line=dict(color=base, width=_TRAJ_LINE_WIDTH),
                opacity=opacity,
                name=aid,
                showlegend=(i < 10),
                hovertemplate=(
                    f"<b>gas slack {aid}</b><br>"
                    f"t: %{{x:.2f}}{x_hover}<br>setpoint: %{{y:.4f}} pu<extra></extra>"
                ),
            )
        )

    if failure_t is not None:
        fig.add_vline(
            x=float(failure_t) * x_scale,
            line=dict(color="#888888", dash="dash", width=1),
        )
    fig.update_yaxes(title="slack pressure setpoint (p.u.)")
    fig.update_xaxes(title=x_title)
    return _save(_apply_theme(fig, title=title, legend_top=True), out_path)


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
    x, x_title, _ = _time_axis(trajectories["time_s"].values)
    x_hover = x_title.split("(")[-1].rstrip(")")
    _dec = _decimate(len(x))
    x = x[_dec]
    for i, aid in enumerate(show_cols):
        color = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=arr[aid].values[_dec],
                mode="lines",
                line=dict(color=color, width=1.0),
                opacity=0.55,
                name=aid,
                showlegend=False,
                hovertemplate=(
                    f"<b>{aid}</b><br>"
                    f"t: %{{x:.2f}}{x_hover}<br>factor: %{{y:.3f}}<extra></extra>"
                ),
            )
        )

    if truncated:
        subtitle = (
            f"  ·  showing {len(show_cols)} of {len(aid_cols)} aids (highest-variance)"
        )
    else:
        subtitle = ""
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="regulation factor", range=[-0.05, 1.5], tickformat=".2f")
    return _save(
        _apply_theme(fig, title=title + subtitle, height=380, legend_top=True, font_bump=_TRAJ_FONT_BUMP), out_path
    )


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

    x_full, x_title, x_scale = _time_axis(timeseries["time_s"].astype(float).values)
    _dec = _decimate(len(x_full))
    x = x_full[_dec]
    if not isinstance(_dec, slice):
        timeseries = timeseries.iloc[_dec].reset_index(drop=True)

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
                        x=float(tx) * x_scale,
                        line=dict(color=style["color"], dash=style["dash"], width=1),
                        opacity=0.55,
                        row=r,
                        col=1,
                    )
    elif failure_t is not None:
        for r in range(1, rows + 1):
            fig.add_vline(
                x=float(failure_t) * x_scale,
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

    fig.update_xaxes(title=x_title, row=rows, col=1)
    height = 180 * rows + 100
    return _save(
        _apply_theme(
            fig,
            title=title,
            height=height,
            width=int(_FIG_WIDTH * 1.4),
            legend_top=True,
        ),
        out_path,
    )


# Extension experiments — islanding MILP + temporal physics (linepack / LTC)


def _scenario_value(scenario_key: str, field: str) -> str | None:
    """Value of ``field`` in an aggregated ``k=v;k=v`` scenario key (the
    format of ``hpc.aggregate._key_of``); ``None`` when absent."""
    for tok in str(scenario_key).split(";"):
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k.strip() == field:
            return v.strip()
    return None


def _extension_ab_bar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    arm_of,
    label_a: str,
    label_b: str,
    title: str,
    oracle_b_col: str | None = None,
) -> Path:
    """Paired A/B PWSF bars (scare variant), one bar pair per (grid, seed),
    with oracle reference markers. ``arm_of`` maps a scenario key to
    "A"/"B"/None; pairs missing either arm are dropped. ``oracle_b_col``
    substitutes a dedicated oracle outcome column for the B arm when present
    (e.g. the receding-horizon temporal oracle's mean PWSF)."""
    metric = "outcomes__priority_weighted_fraction"
    if df.empty or metric not in df.columns or "scenario" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)

    d = _completed(df).copy()
    d["arm"] = d["scenario"].map(arm_of)
    d = d[d["arm"].notna()]
    scare = d[d["variant"] == "scare"]
    if scare.empty:
        return _save(_empty_fig("no scare rows", title), out_path)
    pivot = scare.pivot_table(index=["grid", "seed"], columns="arm", values=metric)
    if pivot.empty or {"A", "B"} - set(pivot.columns):
        return _save(_empty_fig("need both A and B arms (scare)", title), out_path)
    pivot = pivot.dropna(subset=["A", "B"]).sort_index()
    if pivot.empty:
        return _save(_empty_fig("no complete scare A/B pairs", title), out_path)

    multi_grid = pivot.index.get_level_values("grid").nunique() > 1
    x = [
        f"{alias_grid(g)} · s{s}" if multi_grid else f"seed {s}"
        for g, s in pivot.index
    ]
    delta = pivot["B"] - pivot["A"]

    fig = go.Figure()
    for arm, name, color in (
        ("A", label_a, "#7F7F7F"),
        ("B", label_b, _VARIANT_COLOR["scare"]),
    ):
        hover = [
            f"<b>{name}</b><br>{lbl}<br>PWSF: {v:.4f}<br>Δ (B−A): {dl:+.4f}"
            for lbl, v, dl in zip(x, pivot[arm], delta)
        ]
        fig.add_trace(
            go.Bar(
                name=name,
                x=x,
                y=pivot[arm].values,
                marker=_bar_marker(color),
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    orc = d[d["variant"] == "oracle"].copy()
    if not orc.empty:
        orc["_ref"] = orc[metric]
        if oracle_b_col and oracle_b_col in orc.columns:
            b_rows = orc["arm"] == "B"
            orc.loc[b_rows, "_ref"] = orc.loc[b_rows, oracle_b_col].fillna(
                orc.loc[b_rows, metric]
            )
        opivot = orc.pivot_table(
            index=["grid", "seed"], columns="arm", values="_ref"
        ).reindex(pivot.index)
        for arm, name, symbol in (
            ("A", f"oracle · {label_a}", "circle"),
            ("B", f"oracle · {label_b}", "diamond"),
        ):
            if arm not in opivot.columns or opivot[arm].dropna().empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=opivot[arm].values,
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=_MARKER_SIZE,
                        symbol=symbol,
                        color=_VARIANT_COLOR["oracle"],
                        line=dict(width=1.2, color="white"),
                    ),
                    hovertemplate=(
                        f"<b>{name}</b><br>%{{x}}<br>PWSF: %{{y:.4f}}"
                        "<extra></extra>"
                    ),
                )
            )

    mean_delta = float(delta.mean())
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.97,
        y=0.97,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#CCCCCC",
        borderwidth=0.6,
        borderpad=5,
        text=(
            f"<b>mean Δ (B−A)</b> {mean_delta:+.4f}  ·  "
            f"<b>n</b>={len(pivot)} pairs"
        ),
        font=dict(family=_FONT_FAMILY, size=_ANNOTATION_FONT_SIZE, color=_AXIS_COLOR),
    )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
    fig.update_xaxes(title="paired run (same grid + seed)")
    fig.update_yaxes(
        title="priority-weighted served fraction", range=[0, 1.05], tickformat=".2f"
    )
    return _save(
        _apply_theme(
            fig, title=title, height=520, width=_BAR_FIG_WIDTH, legend_top=True
        ),
        out_path,
    )


def extension_islanding_ab(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Islanding extension — paired PWSF, clean vs microgrid (scare)",
) -> Path:
    def arm_of(scenario: str) -> str | None:
        return {"clean": "A", "microgrid": "B"}.get(
            _scenario_value(scenario, "kind") or ""
        )

    return _extension_ab_bar(
        df,
        out_path,
        arm_of=arm_of,
        label_a="clean",
        label_b="microgrid",
        title=title,
    )


def extension_temporal_ab(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Temporal extensions — paired PWSF, off vs linepack+LTC (scare)",
) -> Path:
    def arm_of(scenario: str) -> str | None:
        on = _scenario_value(scenario, "linepack") in ("True", "true", "1")
        return "B" if on else "A"

    return _extension_ab_bar(
        df,
        out_path,
        arm_of=arm_of,
        label_a="no extensions",
        label_b="linepack+LTC",
        title=title,
        oracle_b_col="outcomes__oracle_temporal__priority_weighted_fraction_mean",
    )


def extension_islanding_events(
    tasks: list[tuple[str, pd.DataFrame]],
    out_path: Path,
    *,
    title: str = (
        "Islanding evidence — de-energised nodes + islanding events "
        "(microgrid arm, scare)"
    ),
) -> Path:
    """One line per (label, timeseries) task: ``nodes_deenergised`` (top) and
    ``islanded_events_cum`` (bottom) over simulation time."""
    panels = (
        (1, "nodes_deenergised", "de-energised nodes"),
        (2, "islanded_events_cum", "cumulative islanding events"),
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[p[2] for p in panels],
    )
    # Unit from the widest span across tasks so every trace shares one axis.
    _tmax = max(
        (float(ts["time_s"].astype(float).max())
         for _, ts in tasks
         if not ts.empty and "time_s" in ts.columns),
        default=0.0,
    )
    _, x_title, x_scale = _time_axis(np.array([_tmax]))
    x_hover = x_title.split("(")[-1].rstrip(")")
    plotted = False
    for i, (label, ts) in enumerate(tasks):
        if ts.empty or "time_s" not in ts.columns:
            continue
        color = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
        idx = _decimate(len(ts))
        for row_idx, col, panel_lbl in panels:
            if col not in ts.columns:
                continue
            plotted = True
            fig.add_trace(
                go.Scatter(
                    x=ts["time_s"].astype(float).values[idx] * x_scale,
                    y=ts[col].astype(float).values[idx],
                    mode="lines",
                    # Counts change stepwise at solve boundaries.
                    line=dict(color=color, width=2, shape="hv"),
                    name=label,
                    legendgroup=label,
                    showlegend=(row_idx == 1),
                    hovertemplate=(
                        f"<b>{label}</b><br>{panel_lbl}: %{{y:.0f}}<br>"
                        f"t: %{{x:.2f}}{x_hover}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )
    if not plotted:
        return _save(_empty_fig("no islanding timeseries columns", title), out_path)

    fig.update_yaxes(title="nodes", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title="events", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title=x_title, row=2, col=1)
    return _save(_apply_theme(fig, title=title, height=620, legend_top=True), out_path)


def extension_temporal_trajectories(
    tasks: list[tuple[str, pd.DataFrame, str]],
    out_path: Path,
    *,
    oracle_series: list[tuple[str, list[dict[str, Any]]]] | None = None,
    title: str = (
        "Temporal extensions — linepack + LTC temperature over physical time "
        "(scare, B arm)"
    ),
) -> Path:
    """Per-task ``linepack_total_kg`` (top) and LTC junction temperature
    mean/min (bottom) against PHYSICAL hours: x = time_s ·
    physics_time_scale / 3600, with the scale read from each task's scenario
    key. ``oracle_series`` overlays the receding-horizon oracle's per-step
    linepack (already in hours) as dashed reference lines."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=["gas linepack", "LTC junction temperature"],
    )
    plotted = False
    for i, (label, ts, scenario) in enumerate(tasks):
        if ts.empty or "time_s" not in ts.columns:
            continue
        try:
            scale = float(_scenario_value(scenario, "physics_time_scale") or 1.0)
        except ValueError:
            scale = 1.0
        idx = _decimate(len(ts))
        x = (ts["time_s"].astype(float) * scale / 3600.0).values[idx]
        color = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
        if "linepack_total_kg" in ts.columns:
            plotted = True
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ts["linepack_total_kg"].astype(float).values[idx],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=label,
                    legendgroup=label,
                    hovertemplate=(
                        f"<b>{label}</b><br>linepack: %{{y:.2f}} kg<br>"
                        "t: %{x:.2f}h<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        for col, dash, sub_lbl in (
            ("ltc_junction_t_mean_k", "solid", "mean"),
            ("ltc_junction_t_min_k", "dot", "min"),
        ):
            if col not in ts.columns:
                continue
            plotted = True
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ts[col].astype(float).values[idx],
                    mode="lines",
                    line=dict(color=color, width=2, dash=dash),
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label}</b><br>T {sub_lbl}: %{{y:.2f}} K<br>"
                        "t: %{x:.2f}h<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
    if not plotted:
        return _save(_empty_fig("no temporal timeseries columns", title), out_path)

    for j, (label, series) in enumerate(oracle_series or []):
        pts = [
            (p.get("t_h"), p.get("linepack_total_kg"))
            for p in series
            if isinstance(p, dict)
        ]
        pts = [(t, v) for t, v in pts if t is not None and v is not None]
        if not pts:
            continue
        fig.add_trace(
            go.Scatter(
                x=[t for t, _ in pts],
                y=[v for _, v in pts],
                mode="lines",
                line=dict(color=_VARIANT_COLOR["oracle"], width=1.6, dash="dash"),
                opacity=0.8,
                name="oracle linepack",
                legendgroup="oracle",
                showlegend=(j == 0),
                hovertemplate=(
                    f"<b>oracle {label}</b><br>linepack: %{{y:.2f}} kg<br>"
                    "t: %{x:.2f}h<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Sentinel entries so the mean/min dash coding is legible from the legend.
    for dash, name in (("solid", "T mean"), ("dot", "T min")):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="#666666", width=2, dash=dash),
                name=name,
                legendgroup="linestyle",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title="linepack (kg)", row=1, col=1)
    fig.update_yaxes(title="temperature (K)", row=2, col=1)
    fig.update_xaxes(title="physical time (h)", row=2, col=1)
    return _save(_apply_theme(fig, title=title, height=620, legend_top=True), out_path)


def _physical_hours(ts: pd.DataFrame, scenario: str) -> Any:
    try:
        scale = float(_scenario_value(scenario, "physics_time_scale") or 1.0)
    except (ValueError, TypeError):
        scale = 1.0
    return (ts["time_s"].astype(float) * scale / 3600.0).values


def extension_temporal_thermal_inertia(
    pairs: list[tuple[str, pd.DataFrame, pd.DataFrame, str]],
    out_path: Path,
    *,
    title: str = (
        "Thermal inertia — junction temperature, no-LTC baseline vs LTC "
        "(same seed)"
    ),
) -> Path:
    """The temporal experiment's defining property: without thermal
    capacitance every solve is an independent steady state, so the junction
    snaps from its start temperature to the operating point in a single step;
    the LTC extension replaces that discontinuity with a physical relaxation.

    ``pairs`` = ``[(label, off_ts, on_ts, scenario), ...]`` — the max-junction
    temperature (``max_t_k``) of the off (dashed) and on (solid) arm of the
    same seed, over physical hours.
    """
    fig = go.Figure()
    plotted = False
    for i, (label, off_ts, on_ts, scenario) in enumerate(pairs):
        color = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
        for ts, dash, arm in ((off_ts, "dash", "no LTC"), (on_ts, "solid", "LTC")):
            if ts is None or ts.empty or "max_t_k" not in ts.columns:
                continue
            if "time_s" not in ts.columns:
                continue
            plotted = True
            idx = _decimate(len(ts))
            fig.add_trace(
                go.Scatter(
                    x=_physical_hours(ts, scenario)[idx],
                    y=ts["max_t_k"].astype(float).values[idx],
                    mode="lines",
                    line=dict(color=color, width=2.2, dash=dash),
                    name=f"{label} · {arm}",
                    legendgroup=label,
                    hovertemplate=(
                        f"<b>{label} · {arm}</b><br>max T: %{{y:.2f}} K<br>"
                        "t: %{x:.2f}h<extra></extra>"
                    ),
                )
            )
    if not plotted:
        return _save(_empty_fig("no max_t_k in either arm", title), out_path)

    for dash, name in (("dash", "no LTC (steady snap)"), ("solid", "LTC (relaxation)")):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="#666666", width=2.2, dash=dash),
                name=name,
                legendgroup="arm_style",
                showlegend=True,
            )
        )
    fig.update_yaxes(title="max junction temperature (K)")
    fig.update_xaxes(title="physical time (h)")
    return _save(_apply_theme(fig, title=title, height=560, legend_top=True), out_path)


def _served_fraction_series(ts: pd.DataFrame) -> Any:
    """Per-tick total served fraction from the ``tier_served_mw__*`` /
    ``tier_demand_mw__*`` recordings (Σ served / Σ demand across tiers)."""
    served_cols = [c for c in ts.columns if c.startswith("tier_served_mw__")]
    demand_cols = [c for c in ts.columns if c.startswith("tier_demand_mw__")]
    if not served_cols or not demand_cols:
        return None
    served = ts[served_cols].astype(float).sum(axis=1)
    demand = ts[demand_cols].astype(float).sum(axis=1)
    return (served / demand.where(demand > 0)).values


def extension_temporal_ride_through(
    pairs: list[tuple[str, pd.DataFrame, pd.DataFrame, str]],
    out_path: Path,
    *,
    title: str = (
        "Deficit ride-through — served fraction, no-extension vs linepack+LTC "
        "(same seed)"
    ),
) -> Path:
    """Served fraction over physical time, off (dashed) vs on (solid) arm of
    the same seed. The temporal extensions' stored energy (gas linepack draw-
    down, thermal inertia) lets the on arm sustain higher service through the
    post-failure deficit window."""
    fig = go.Figure()
    plotted = False
    for i, (label, off_ts, on_ts, scenario) in enumerate(pairs):
        color = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
        for ts, dash, arm in ((off_ts, "dash", "off"), (on_ts, "solid", "on")):
            if ts is None or ts.empty or "time_s" not in ts.columns:
                continue
            frac = _served_fraction_series(ts)
            if frac is None:
                continue
            plotted = True
            idx = _decimate(len(ts))
            fig.add_trace(
                go.Scatter(
                    x=_physical_hours(ts, scenario)[idx],
                    y=frac[idx],
                    mode="lines",
                    line=dict(color=color, width=2.2, dash=dash),
                    name=f"{label} · {arm}",
                    legendgroup=label,
                    hovertemplate=(
                        f"<b>{label} · {arm}</b><br>served: %{{y:.3f}}<br>"
                        "t: %{x:.2f}h<extra></extra>"
                    ),
                )
            )
    if not plotted:
        return _save(_empty_fig("no tier served/demand recordings", title), out_path)

    for dash, name in (("dash", "no extensions"), ("solid", "linepack+LTC")):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="#666666", width=2.2, dash=dash),
                name=name,
                legendgroup="arm_style",
                showlegend=True,
            )
        )
    fig.update_yaxes(title="served fraction", range=[0, 1.05], tickformat=".2f")
    fig.update_xaxes(title="physical time (h)")
    return _save(_apply_theme(fig, title=title, height=560, legend_top=True), out_path)


def extension_islanding_recovery(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = (
        "Islanding recovery — served load by sector, clean vs microgrid (scare)"
    ),
) -> Path:
    """Where the islanding extension saves load: mean served fraction per
    sector for the clean vs microgrid arm (scare variant). Grid-forming
    promotion lets severed sub-islands keep serving their local demand, so the
    microgrid bars sit above clean wherever islands self-anchored."""
    if df.empty or "scenario" not in df.columns:
        return _save(_empty_fig("no data", title), out_path)
    sectors = ("electricity", "gas", "heat")
    frac_col = {s: f"outcomes__served_by_sector__{s}__fraction" for s in sectors}
    present = [s for s in sectors if frac_col[s] in df.columns]
    if not present:
        return _save(_empty_fig("no served_by_sector fractions", title), out_path)

    d = _completed(df).copy()
    d = d[d["variant"] == "scare"]
    d["arm"] = d["scenario"].map(
        lambda s: {"clean": "clean", "microgrid": "microgrid"}.get(
            _scenario_value(s, "kind") or ""
        )
    )
    d = d[d["arm"].notna()]
    if d.empty:
        return _save(_empty_fig("no clean/microgrid scare rows", title), out_path)

    fig = go.Figure()
    for arm, color in (("clean", "#7F7F7F"), ("microgrid", _VARIANT_COLOR["scare"])):
        sub = d[d["arm"] == arm]
        if sub.empty:
            continue
        means = [float(sub[frac_col[s]].astype(float).mean()) for s in present]
        fig.add_trace(
            go.Bar(
                name=arm,
                x=[s for s in present],
                y=means,
                marker=_bar_marker(color),
                hovertemplate=(
                    f"<b>{arm}</b><br>%{{x}}: %{{y:.3f}} served<extra></extra>"
                ),
            )
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
    fig.update_yaxes(title="mean served fraction", range=[0, 1.05], tickformat=".2f")
    fig.update_xaxes(title="sector")
    return _save(
        _apply_theme(fig, title=title, height=520, width=_BAR_FIG_WIDTH, legend_top=True),
        out_path,
    )
