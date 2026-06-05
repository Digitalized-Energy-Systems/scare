"""Plots for the L2.5 cross-sector coalition pathway.

Consume an event ledger (from the e2e test or any campaign run) and
render cross-sector CP behaviour:

* :func:`cp_setpoint_timeline` — per-CP per-sector flow over time, with
  active coalition envelopes overlaid as shaded TTL windows.
* :func:`coalition_lifecycle_gantt` — one horizontal bar per coalition,
  spanning ``issued_at -> issued_at + ttl_s``, coloured by committed CP.
* :func:`envelope_clamp_arrows` — before/after arrows showing how the CP
  envelope re-routed each ADMM-decided sector flow.
* :func:`flag_on_off_comparison` — side-by-side event counts for the
  ablation.
* :func:`cross_sector_transfer_distribution` — histogram of committed
  transfer magnitudes (by sector) across a campaign.

Input format:

* ``events.json``  — list of EventRecord dicts (``t, kind, aid,
  sector, detail``).
* ``summary.json`` — ``{"all": {kind: count}, "cross_sector":
  {kind: count}}``.

Each function returns the figure's directory-relative stem (no
extension) — same contract as :mod:`experiment.eval.plots`.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reuse the shared theme to match the rest of the plot pack.
from experiment.eval.plots import (
    _AXIS_COLOR,
    _DEFAULT_LAYOUT,
    _FIG_HEIGHT,
    _FIG_WIDTH,
    _FONT_FAMILY,
    _MUTED_COLOR,
    _QUAL_PALETTE,
    _SECTOR_COLOR,
    _apply_theme,
    _empty_fig,
    _hex_to_rgba,
    _save,
)


# ---------------------------------------------------------------------------
# Parsing — ledger detail strings into structured fields
# ---------------------------------------------------------------------------


# ``flows={electricity: 0.5000, heat: -1.0000}`` (cp_setpoint,
# cp_envelope_set, cp_envelope_clamp).
_FLOWS_RE = re.compile(r"flows=\{([^}]*)\}")
_COALITION_RE = re.compile(r"coalition=(\S+)")
_TTL_RE = re.compile(r"ttl=([0-9.]+)")
_ENV_ACTIVE_RE = re.compile(r"envelope_active=(True|False)")
_TRANSFER_OUT_RE = re.compile(r"transfer_out=([0-9.]+)")
_TRANSFER_IN_RE = re.compile(r"transfer_in=([0-9.]+)")
_CP_RE = re.compile(r"\bcp=(\S+)")
# Cross-sector inversion, with high/low tiers.
_CSI_RE = re.compile(
    r"cp=(\S+).*?own_sec=(\S+)\s+tier_high=(\d+).*?peer_sec=(\S+)\s+tier_low=(\d+)"
)


@dataclass(frozen=True)
class _CPSetpointRow:
    t: float
    cp_aid: str
    flows: dict[str, float]
    regulation: float
    envelope_active: bool


@dataclass(frozen=True)
class _CoalitionRow:
    t: float
    initiator_aid: str
    cp_aid: str
    transfer_out: float
    transfer_in: float


@dataclass(frozen=True)
class _EnvelopeRow:
    t: float
    cp_aid: str
    coalition_id: str
    ttl_s: float
    flows: dict[str, float]


def _parse_flows(detail: str) -> dict[str, float]:
    m = _FLOWS_RE.search(detail)
    if not m:
        return {}
    out: dict[str, float] = {}
    inner = m.group(1)
    for piece in inner.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            continue
        sec, val = piece.split(":", 1)
        try:
            out[sec.strip()] = float(val.strip())
        except ValueError:
            continue
    return out


def _parse_pre_post(detail: str) -> tuple[list[float], list[float]] | None:
    """Pull ``pre=[..]`` / ``post=[..]`` floats out of a clamp record."""
    m_pre = re.search(r"pre=(\[[^\]]+\])", detail)
    m_post = re.search(r"post=(\[[^\]]+\])", detail)
    if not (m_pre and m_post):
        return None
    try:
        pre = ast.literal_eval(m_pre.group(1))
        post = ast.literal_eval(m_post.group(1))
    except (SyntaxError, ValueError):
        return None
    if not (isinstance(pre, list) and isinstance(post, list)):
        return None
    return [float(x) for x in pre], [float(x) for x in post]


def _load_events(events_json: Path) -> list[dict[str, Any]]:
    try:
        text = Path(events_json).read_text()
    except FileNotFoundError:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _cp_setpoint_rows(events: list[dict[str, Any]]) -> list[_CPSetpointRow]:
    out: list[_CPSetpointRow] = []
    for e in events:
        if e.get("kind") != "cp_setpoint":
            continue
        detail = e.get("detail", "")
        env_m = _ENV_ACTIVE_RE.search(detail)
        reg_m = re.search(r"reg=([0-9.]+)", detail)
        out.append(_CPSetpointRow(
            t=float(e.get("t", 0.0)),
            cp_aid=str(e.get("aid", "")),
            flows=_parse_flows(detail),
            regulation=float(reg_m.group(1)) if reg_m else float("nan"),
            envelope_active=(env_m.group(1) == "True") if env_m else False,
        ))
    return out


def _coalition_rows(events: list[dict[str, Any]]) -> list[_CoalitionRow]:
    out: list[_CoalitionRow] = []
    for e in events:
        if e.get("kind") != "cross_sector_coalition_allocation":
            continue
        detail = e.get("detail", "")
        cp_m = _CP_RE.search(detail)
        to_m = _TRANSFER_OUT_RE.search(detail)
        ti_m = _TRANSFER_IN_RE.search(detail)
        out.append(_CoalitionRow(
            t=float(e.get("t", 0.0)),
            initiator_aid=str(e.get("aid", "")),
            cp_aid=cp_m.group(1) if cp_m else "",
            transfer_out=float(to_m.group(1)) if to_m else 0.0,
            transfer_in=float(ti_m.group(1)) if ti_m else 0.0,
        ))
    return out


def _envelope_rows(events: list[dict[str, Any]]) -> list[_EnvelopeRow]:
    out: list[_EnvelopeRow] = []
    for e in events:
        if e.get("kind") != "cp_envelope_set":
            continue
        detail = e.get("detail", "")
        cid_m = _COALITION_RE.search(detail)
        ttl_m = _TTL_RE.search(detail)
        out.append(_EnvelopeRow(
            t=float(e.get("t", 0.0)),
            cp_aid=str(e.get("aid", "")),
            coalition_id=cid_m.group(1) if cid_m else "",
            ttl_s=float(ttl_m.group(1)) if ttl_m else 0.0,
            flows=_parse_flows(detail),
        ))
    return out


def _all_cps(rows: Iterable[_CPSetpointRow | _EnvelopeRow]) -> list[str]:
    return sorted({r.cp_aid for r in rows if r.cp_aid})


# ---------------------------------------------------------------------------
# Plot 1 — per-CP per-sector setpoint timeline with coalition overlay
# ---------------------------------------------------------------------------


def cp_setpoint_timeline(
    events_json: Path,
    out_path: Path,
    *,
    title: str = "CP sector flows with cross-sector coalition windows",
) -> Path:
    """One subplot per CP; per-sector flow as a line.  Each active
    coalition envelope is a translucent band over
    ``[issued_at, issued_at + ttl_s]``, showing when L2.5 held the CP
    and how its flow moved during the window.
    """
    events = _load_events(events_json)
    setpoints = _cp_setpoint_rows(events)
    envelopes = _envelope_rows(events)
    if not setpoints:
        return _save(_empty_fig("no CP setpoint events", title), out_path)

    cps = _all_cps(setpoints)
    n = len(cps)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=cps,
    )

    for i, cp in enumerate(cps, start=1):
        cp_rows = sorted(
            (r for r in setpoints if r.cp_aid == cp),
            key=lambda r: r.t,
        )
        sectors_seen = set()
        for r in cp_rows:
            sectors_seen.update(r.flows.keys())
        for sec in sorted(sectors_seen):
            xs = [r.t for r in cp_rows]
            ys = [r.flows.get(sec, math.nan) for r in cp_rows]
            color = _SECTOR_COLOR.get(sec, _QUAL_PALETTE[i % len(_QUAL_PALETTE)])
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name=sec if i == 1 else None,  # legend entry once
                    showlegend=(i == 1),
                    line=dict(color=color, width=2.0),
                    marker=dict(size=5, color=color, line=dict(width=0.5, color="white")),
                    hovertemplate=(
                        f"<b>{cp}</b><br>"
                        f"sector: {sec}<br>"
                        "t: %{x:.2f} s<br>"
                        "flow: %{y:.4f} MW"
                        "<extra></extra>"
                    ),
                ),
                row=i, col=1,
            )

        # One translucent band per coalition; palette cycles so
        # overlapping coalitions stay separable.
        cp_envs = [e for e in envelopes if e.cp_aid == cp]
        for j, env in enumerate(cp_envs):
            band_color = _hex_to_rgba(
                _QUAL_PALETTE[j % len(_QUAL_PALETTE)], 0.12,
            )
            fig.add_vrect(
                x0=env.t,
                x1=env.t + env.ttl_s,
                fillcolor=band_color,
                line_width=0,
                layer="below",
                row=i, col=1,
            )

        fig.update_yaxes(
            title_text="MW" if i == n else "",
            row=i, col=1,
            zeroline=True, zerolinecolor=_MUTED_COLOR, zerolinewidth=0.8,
        )

    fig.update_xaxes(title_text="simulation time (s)", row=n, col=1)
    fig.update_layout(_DEFAULT_LAYOUT)
    fig.update_layout(
        title=dict(text=title, **_DEFAULT_LAYOUT["title"]),
        height=max(280, 200 * n),
        width=_FIG_WIDTH * 1.4,
    )
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 2 — coalition lifecycle Gantt
# ---------------------------------------------------------------------------


def coalition_lifecycle_gantt(
    events_json: Path,
    out_path: Path,
    *,
    title: str = "Cross-sector coalition lifecycles",
    default_ttl_s: float = 4.0,
) -> Path:
    """One horizontal bar per coalition, spanning the TTL window.

    Bar y-position is the committed CP; hover shows the initiator and
    in/out transfer magnitudes.
    """
    events = _load_events(events_json)
    coalitions = _coalition_rows(events)
    envelopes = _envelope_rows(events)
    if not coalitions and not envelopes:
        return _save(_empty_fig("no cross-sector coalitions recorded", title), out_path)

    # Index envelopes by (cp_aid, coalition_id) for TTL lookup.
    env_index: dict[tuple[str, str], _EnvelopeRow] = {
        (e.cp_aid, e.coalition_id): e for e in envelopes
    }

    cps = sorted({c.cp_aid for c in coalitions} | {e.cp_aid for e in envelopes})
    cp_to_y = {cp: i for i, cp in enumerate(cps)}

    fig = go.Figure()
    # Background row per CP swim lane.
    for cp, y in cp_to_y.items():
        fig.add_shape(
            type="rect",
            x0=0, x1=1, xref="paper",
            y0=y - 0.45, y1=y + 0.45, yref="y",
            fillcolor="#FAFAFA",
            line=dict(width=0),
            layer="below",
        )

    for j, c in enumerate(coalitions):
        env = env_index.get((c.cp_aid, ""))  # alloc records often lack coalition_id
        # Fallback: any envelope on this CP near the same time.
        if env is None:
            candidate = [
                e for e in envelopes
                if e.cp_aid == c.cp_aid and abs(e.t - c.t) < 1e-3
            ]
            env = candidate[0] if candidate else None
        ttl = env.ttl_s if env else default_ttl_s
        coalition_color = _QUAL_PALETTE[j % len(_QUAL_PALETTE)]
        y = cp_to_y.get(c.cp_aid, 0)

        # Bar spanning [t, t+ttl].
        fig.add_trace(go.Bar(
            x=[ttl],
            y=[c.cp_aid],
            base=[c.t],
            orientation="h",
            marker=dict(
                color=_hex_to_rgba(coalition_color, 0.65),
                line=dict(color=coalition_color, width=1.2),
            ),
            width=0.55,
            name=f"coalition #{j + 1}",
            showlegend=False,
            hovertemplate=(
                f"<b>{c.cp_aid}</b><br>"
                f"initiator: {c.initiator_aid}<br>"
                "t_start: %{base:.2f} s<br>"
                f"ttl: {ttl:.2f} s<br>"
                f"transfer_out: {c.transfer_out:.4f} MW<br>"
                f"transfer_in: {c.transfer_in:.4f} MW<br>"
                "<extra></extra>"
            ),
        ))
        # Start marker for visual anchoring.
        fig.add_trace(go.Scatter(
            x=[c.t], y=[c.cp_aid],
            mode="markers",
            marker=dict(
                color=coalition_color,
                size=10,
                symbol="diamond",
                line=dict(width=1, color="white"),
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_xaxes(title="simulation time (s)")
    fig.update_yaxes(
        title="",
        categoryorder="array",
        categoryarray=cps,
    )
    fig.update_layout(barmode="overlay")
    return _save(_apply_theme(
        fig, title=title,
        height=max(220, 70 * len(cps) + 130),
        width=int(_FIG_WIDTH * 1.4),
    ), out_path)


# ---------------------------------------------------------------------------
# Plot 3 — envelope clamp before/after
# ---------------------------------------------------------------------------


def envelope_clamp_arrows(
    events_json: Path,
    out_path: Path,
    *,
    title: str = "CP ADMM output clamped by cross-sector envelope",
    sector_order: tuple[str, ...] = ("electricity", "heat", "gas"),
) -> Path:
    """Slope chart: per clamp event, an arrow from the pre-clamp ADMM
    result to the post-clamp committed value.

    An L2.5 override of L3's choice shows as a divergent arrow; an
    in-envelope L3 result gives a short/vertical arrow.
    """
    events = _load_events(events_json)
    rows: list[tuple[float, str, list[float], list[float]]] = []
    for e in events:
        if e.get("kind") != "cp_envelope_clamp":
            continue
        pre_post = _parse_pre_post(e.get("detail", ""))
        if pre_post is None:
            continue
        rows.append((
            float(e.get("t", 0.0)),
            str(e.get("aid", "")),
            pre_post[0],
            pre_post[1],
        ))
    if not rows:
        return _save(_empty_fig("no clamp events recorded", title), out_path)

    fig = go.Figure()
    # One evenly-spaced column per sector.
    x_positions = {sec: i for i, sec in enumerate(sector_order)}

    for idx, (t, cp, pre, post) in enumerate(rows):
        for sec_idx, sec in enumerate(sector_order):
            if sec_idx >= len(pre) or sec_idx >= len(post):
                continue
            color = _SECTOR_COLOR.get(sec, _QUAL_PALETTE[idx % len(_QUAL_PALETTE)])
            x = x_positions[sec]
            # Thin segment with a marker cap at the post-clamp value.
            fig.add_trace(go.Scatter(
                x=[x - 0.18, x + 0.18],
                y=[pre[sec_idx], post[sec_idx]],
                mode="lines+markers",
                line=dict(color=color, width=1.6),
                marker=dict(
                    size=[7, 11],
                    color=color,
                    symbol=["circle", "triangle-right"],
                    line=dict(width=0.6, color="white"),
                ),
                opacity=0.55,
                showlegend=False,
                hovertemplate=(
                    f"<b>{cp}</b><br>"
                    f"sector: {sec}<br>"
                    f"t: {t:.2f} s<br>"
                    "pre-clamp: %{y:.4f} MW<br>"
                    "<extra></extra>"
                ),
            ))

    # Sector legend (one entry per sector).
    for sec in sector_order:
        color = _SECTOR_COLOR.get(sec, _MUTED_COLOR)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers+lines",
            line=dict(color=color, width=1.6),
            marker=dict(color=color, size=8),
            name=sec,
            showlegend=True,
        ))

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_positions.values()),
        ticktext=list(x_positions.keys()),
        title="sector",
    )
    fig.update_yaxes(title="ADMM result MW (pre → post)")
    return _save(_apply_theme(fig, title=title, height=380), out_path)


# ---------------------------------------------------------------------------
# Plot 4 — flag-on vs flag-off comparison
# ---------------------------------------------------------------------------


def flag_on_off_comparison(
    summary_off: Path,
    summary_on: Path,
    out_path: Path,
    *,
    title: str = "Cross-sector pathway: flag on vs off",
    kinds: tuple[str, ...] = (
        "cross_sector_inversion_detected",
        "cross_sector_coalition_allocation",
        "cp_envelope_set",
        "cp_envelope_clamp",
        "cp_setpoint",
    ),
) -> Path:
    """Grouped bar chart of selected event-kind counts from two
    ``summary.json`` files; the flag-on/flag-off asymmetry is the
    headline ablation visual.
    """
    def _read(p: Path) -> dict[str, int]:
        try:
            text = Path(p).read_text()
        except FileNotFoundError:
            return {}
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(obj, dict):
            return {}
        return dict(obj.get("cross_sector", obj.get("all", {})))

    off = _read(summary_off)
    on = _read(summary_on)
    if not off and not on:
        return _save(_empty_fig("no summary data", title), out_path)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="flag off",
        x=list(kinds),
        y=[off.get(k, 0) for k in kinds],
        marker=dict(color="#9E9E9E", line=dict(width=0.5, color="white")),
        hovertemplate="<b>%{x}</b><br>flag off: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="flag on",
        x=list(kinds),
        y=[on.get(k, 0) for k in kinds],
        marker=dict(color="#1F4E96", line=dict(width=0.5, color="white")),
        hovertemplate="<b>%{x}</b><br>flag on: %{y}<extra></extra>",
    ))
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.06)
    fig.update_yaxes(title="event count", rangemode="tozero")
    fig.update_xaxes(title="")
    fig.update_layout(xaxis=dict(tickangle=-20))
    return _save(_apply_theme(fig, title=title, width=int(_FIG_WIDTH * 1.4)), out_path)


# ---------------------------------------------------------------------------
# Plot 5 — transfer-magnitude distribution
# ---------------------------------------------------------------------------


def cross_sector_transfer_distribution(
    events_json: Path,
    out_path: Path,
    *,
    title: str = "Cross-sector coalition transfer magnitudes",
) -> Path:
    """Two-panel histogram of ``transfer_out`` and ``transfer_in``
    magnitudes across all coalition allocations; surfaces the typical
    scale of the L2.5 contribution.
    """
    events = _load_events(events_json)
    coalitions = _coalition_rows(events)
    if not coalitions:
        return _save(_empty_fig("no coalition allocations recorded", title), out_path)

    out_vals = [c.transfer_out for c in coalitions if c.transfer_out > 0]
    in_vals = [c.transfer_in for c in coalitions if c.transfer_in > 0]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("transfer_out (sink-side)", "transfer_in (source-side)"),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Histogram(
            x=out_vals,
            marker=dict(
                color=_hex_to_rgba(_SECTOR_COLOR["electricity"], 0.75),
                line=dict(color=_SECTOR_COLOR["electricity"], width=0.8),
            ),
            nbinsx=20,
            showlegend=False,
            hovertemplate="MW: %{x:.4f}<br>count: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=in_vals,
            marker=dict(
                color=_hex_to_rgba(_SECTOR_COLOR["heat"], 0.75),
                line=dict(color=_SECTOR_COLOR["heat"], width=0.8),
            ),
            nbinsx=20,
            showlegend=False,
            hovertemplate="MW: %{x:.4f}<br>count: %{y}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title="MW", row=1, col=1)
    fig.update_xaxes(title="MW", row=1, col=2)
    fig.update_yaxes(title="count", row=1, col=1)
    return _save(_apply_theme(
        fig, title=title, height=320, width=int(_FIG_WIDTH * 1.4),
    ), out_path)


# ---------------------------------------------------------------------------
# One-call bundle
# ---------------------------------------------------------------------------


def render_all(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Render every CP-specific figure for one run directory.

    ``run_dir`` must contain ``events.json`` and ``summary.json``.
    Returns a name -> output-stem mapping for building an index page.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "plots"
    events_json = run_dir / "events.json"
    return {
        "cp_setpoint_timeline": cp_setpoint_timeline(
            events_json, out_dir / "cp_setpoint_timeline"
        ),
        "coalition_lifecycle_gantt": coalition_lifecycle_gantt(
            events_json, out_dir / "coalition_lifecycle_gantt"
        ),
        "envelope_clamp_arrows": envelope_clamp_arrows(
            events_json, out_dir / "envelope_clamp_arrows"
        ),
        "cross_sector_transfer_distribution": cross_sector_transfer_distribution(
            events_json, out_dir / "cross_sector_transfer_distribution"
        ),
    }


def render_comparison(
    off_dir: Path,
    on_dir: Path,
    out_dir: Path,
) -> Path:
    """Render the flag-on-vs-off bar chart from two run directories."""
    return flag_on_off_comparison(
        Path(off_dir) / "summary.json",
        Path(on_dir) / "summary.json",
        Path(out_dir) / "flag_on_off_comparison",
    )
