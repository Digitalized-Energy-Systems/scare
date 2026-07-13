from __future__ import annotations

import logging
from typing import Any

from scare.base.model import Sector
from scare.base.util import sector_color

logger = logging.getLogger(__name__)

_FONT_FAMILY = "Libertinus Sans, sans-serif"


def _emit_figure(
    fig: Any, write_to: str | None, show: bool, *, log: bool = False
) -> None:
    fig.update_layout(font=dict(family=_FONT_FAMILY))
    if write_to:
        if write_to.endswith(".html"):
            fig.write_html(write_to)
        else:
            fig.write_image(write_to)
        if log:
            logger.info("Results written to %s", write_to)
    if show:
        fig.show()


def visualize_results(
    world: Any,
    *,
    write_to: str | None = "results.html",
    show: bool = False,
) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    collections = world.data_collections
    agent_collections = world.data_agent_collections
    all_keys = list(collections.keys()) + list(agent_collections.keys())

    if not all_keys:
        logger.warning("visualize_results: no recordings found in world")
        return go.Figure()

    n_plots = len(all_keys)
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        subplot_titles=all_keys,
        vertical_spacing=0.05,
    )

    for row, key in enumerate(all_keys, start=1):
        if key in collections:
            rec = collections[key]
            fig.add_trace(
                go.Scatter(x=rec.time, y=rec.timeseries, name=key, mode="lines"),
                row=row,
                col=1,
            )
        else:
            rec = agent_collections[key]
            for aid, series in rec.timeseries.items():
                fig.add_trace(
                    go.Scatter(x=rec.time, y=series, name=f"{key}/{aid}", mode="lines"),
                    row=row,
                    col=1,
                )

    fig.update_layout(
        height=300 * n_plots,
        title_text="SCARE Simulation Results",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Time [s]", row=n_plots, col=1)

    _emit_figure(fig, write_to, show, log=True)
    return fig


def visualize_network(
    monee_net: Any,
    *,
    title: str = "Energy Network",
    write_to: str | None = None,
    show: bool = True,
) -> Any:
    import networkx as nx
    import plotly.graph_objects as go

    g: nx.Graph = monee_net.graph

    try:
        pos = {n: (g.nodes[n].get("x", 0.0), g.nodes[n].get("y", 0.0)) for n in g.nodes}
        if all(v == (0.0, 0.0) for v in pos.values()):
            raise ValueError
    except Exception:
        logger.debug(
            "visualize_network: no node positions; falling back to spring layout",
            exc_info=True,
        )
        pos = nx.spring_layout(g, seed=42)

    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    sector_to_color = {
        Sector.ELECTRICITY: "orange",
        Sector.GAS: "green",
        Sector.HEAT: "red",
    }
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in monee_net.nodes:
        x, y = pos.get(node.id, (0.0, 0.0))
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node.id))
        grid = str(node.grid) if not isinstance(node.grid, str) else node.grid
        color = next(
            (sector_to_color[s] for s in Sector if s.value in grid.lower()), "grey"
        )
        node_color.append(color)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="#888"),
                hoverinfo="none",
                name="branches",
            ),
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                text=node_text,
                textposition="top center",
                marker=dict(
                    size=10, color=node_color, line=dict(width=1, color="white")
                ),
                name="nodes",
            ),
        ],
        layout=go.Layout(
            title=title,
            template="plotly_white",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    _emit_figure(fig, write_to, show)
    return fig


def plot_performance_comparison(
    scenarios: list[str],
    performance_values: list[float],
    robustness_values: list[float],
    *,
    title: str = "Performance Comparison",
    write_to: str | None = None,
    show: bool = True,
) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Bar(
                name="L_perf",
                x=scenarios,
                y=performance_values,
                marker_color="steelblue",
            ),
            go.Bar(name="R", x=scenarios, y=robustness_values, marker_color="coral"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Scenario",
        yaxis_title="Metric",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    _emit_figure(fig, write_to, show)
    return fig


def plot_message_counts(
    scenarios: list[str],
    message_counts: list[int],
    *,
    title: str = "Message Count per Scenario",
    write_to: str | None = None,
    show: bool = True,
) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=scenarios, y=message_counts, marker_color="teal")])
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title="Messages",
        template="plotly_white",
    )

    _emit_figure(fig, write_to, show)
    return fig


def plot_sector_timeseries(
    world: Any,
    sectors: list[str] | None = None,
    *,
    write_to: str | None = None,
    show: bool = True,
) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if sectors is None:
        sectors = [s.value for s in Sector]

    matching = {
        s: {k: v for k, v in world.data_collections.items() if s in k.lower()}
        for s in sectors
    }
    matching = {s: d for s, d in matching.items() if d}

    if not matching:
        logger.warning("plot_sector_timeseries: no sector recordings found")
        return go.Figure()

    fig = make_subplots(
        rows=len(matching),
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(matching.keys()),
    )

    for row, (sector_name, data) in enumerate(matching.items(), start=1):
        color = (
            sector_color(Sector(sector_name))
            if sector_name in [s.value for s in Sector]
            else "blue"
        )
        for key, rec in data.items():
            fig.add_trace(
                go.Scatter(
                    x=rec.time,
                    y=rec.timeseries,
                    name=key,
                    mode="lines",
                    line=dict(color=color),
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        height=300 * len(matching),
        title_text="Sector Balance Time-Series",
        template="plotly_white",
    )

    _emit_figure(fig, write_to, show)
    return fig
