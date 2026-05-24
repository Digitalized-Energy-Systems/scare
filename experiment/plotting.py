from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from scare.base.visu import (
    plot_message_counts,
    plot_performance_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def _ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)


def load_results() -> pd.DataFrame:
    records = []
    for p in RESULTS_DIR.glob("*.json"):
        try:
            records.append(json.loads(p.read_text()))
        except Exception as exc:
            logger.warning("Could not load %s: %s", p, exc)

    if not records:
        logger.warning("No result files found in %s", RESULTS_DIR)
        return pd.DataFrame()

    return pd.DataFrame(records)


def plot_side_by_side_comparison(df: pd.DataFrame) -> None:
    if df.empty:
        return
    _ensure_plots_dir()
    scenarios = df["scenario"].tolist()
    perf = df.get("performance", pd.Series([1.0] * len(df))).tolist()
    rob = [1.0 / p if p > 0 else 0.0 for p in perf]

    plot_performance_comparison(
        scenarios,
        perf,
        rob,
        title="Performance vs Robustness by Scenario",
        write_to=str(PLOTS_DIR / "comparison.html"),
        show=False,
    )
    logger.info("Wrote comparison plot to %s", PLOTS_DIR / "comparison.html")


def plot_message_comparison(df: pd.DataFrame) -> None:
    if df.empty or "n_messages" not in df.columns:
        return
    _ensure_plots_dir()
    plot_message_counts(
        df["scenario"].tolist(),
        df["n_messages"].tolist(),
        title="Message Count per Scenario",
        write_to=str(PLOTS_DIR / "messages.html"),
        show=False,
    )
    logger.info("Wrote message count plot to %s", PLOTS_DIR / "messages.html")


def plot_aggregated(df: pd.DataFrame) -> None:
    if df.empty or "performance" not in df.columns:
        return
    _ensure_plots_dir()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["performance"], nbinsx=20, name="performance"))
    fig.update_layout(
        title="Performance Distribution Across Scenarios",
        xaxis_title="Performance Ratio (baseline/MAS)",
        yaxis_title="Count",
        template="plotly_white",
    )
    out = str(PLOTS_DIR / "aggregated_hist.html")
    fig.write_html(out)
    logger.info("Wrote aggregated histogram to %s", out)


def main() -> None:
    df = load_results()
    if df.empty:
        logger.error("No results to plot. Run experiment.evaluation first.")
        return

    logger.info("Loaded %d scenario results", len(df))
    plot_side_by_side_comparison(df)
    plot_message_comparison(df)
    plot_aggregated(df)
    logger.info("All plots written to %s/", PLOTS_DIR)


if __name__ == "__main__":
    main()
