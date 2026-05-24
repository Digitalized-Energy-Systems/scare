"""Load a campaign's results — both the aggregated ``summary.csv`` for
cross-task comparisons and the per-task artefacts (served, diary,
events, timeseries) for per-trajectory plots.

Decoupled from ``aggregate.py`` on purpose: aggregator runs once at
the end of a campaign to materialise summary.csv; the loader lives on
the analysis side and only reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.hpc.config import CAMPAIGN_LAYOUT


# ---------------------------------------------------------------------------
# Per-task wrapper
# ---------------------------------------------------------------------------


@dataclass
class TaskArtefacts:
    """Lazy-loaded view of one task directory.

    Each property is read on first access so a campaign with thousands
    of tasks doesn't pay the IO cost up front when only a small subset
    is plotted (e.g. a single representative trajectory).
    """
    task_dir: Path
    task_id: int
    grid: str
    seed: int
    variant: str
    experiment: str
    ablation: str
    sweep: str
    scenario: str

    # ---- Lazy file accessors ----------------------------------------

    @cached_property
    def result(self) -> dict[str, Any]:
        return _read_json(self.task_dir / "result.json", default={})

    @cached_property
    def status(self) -> dict[str, Any]:
        return _read_json(self.task_dir / "status.json", default={})

    @cached_property
    def failures(self) -> list[dict[str, Any]]:
        data = _read_json(self.task_dir / "failures.json", default=[])
        return data if isinstance(data, list) else []

    @cached_property
    def diary(self) -> pd.DataFrame:
        return _read_csv(self.task_dir / "diary.csv")

    @cached_property
    def events(self) -> pd.DataFrame:
        return _read_csv(self.task_dir / "events.csv")

    @cached_property
    def served(self) -> pd.DataFrame:
        return _read_csv(self.task_dir / "served.csv")

    @cached_property
    def timeseries(self) -> pd.DataFrame:
        return _read_csv(self.task_dir / "timeseries.csv")

    @cached_property
    def trajectories(self) -> pd.DataFrame:
        """Wide per-aid regulation trajectory CSV (forward-filled).

        Only present when the campaign config sets
        ``write_trajectories: true``.  Returns an empty DataFrame when
        the file isn't on disk, so plot helpers can drop straight into
        their empty-fig placeholder.
        """
        return _read_csv(self.task_dir / "trajectories.csv")

    @cached_property
    def slack_meta(self) -> dict[str, dict[str, Any]]:
        """Per-slack-child metadata (``budget``, ``lp_envelope``,
        ``sector``, ``obs_key``, ``node_id``) keyed by aid.

        Written by the runner via ``write_slack_meta`` when the task
        finishes.  Empty dict when the file is absent (legacy tasks or
        a no-slack grid), so plot helpers fall back to drawing the
        trajectory without overlays.
        """
        data = _read_json(self.task_dir / "slack_meta.json", default={})
        return data if isinstance(data, dict) else {}

    # ---- Derived helpers --------------------------------------------

    def is_ok(self) -> bool:
        return self.status.get("status") == "ok"

    def first_failure_time(self) -> float | None:
        delays = [float(f.get("delay_s", 0.0)) for f in self.failures]
        return min(delays) if delays else None

    def solver_failures(self) -> int:
        """Count of energyflow solves that returned infeasible during the
        task run.  Surfaced by the runner from ``_InfeasibilityCounter``
        and saved into ``status.json``.  Used by trajectory plots to
        annotate when the observation pipeline froze on a held-over
        ``_net_results`` snapshot.
        """
        try:
            return int(self.status.get("solver_failures") or 0)
        except (TypeError, ValueError):
            return 0


# ---------------------------------------------------------------------------
# Campaign wrapper
# ---------------------------------------------------------------------------


@dataclass
class CampaignData:
    campaign_dir: Path
    summary: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def task_dir(self, task_id: int) -> Path:
        return self.campaign_dir / CAMPAIGN_LAYOUT["tasks"] / f"{task_id:06d}"

    def task(self, task_id: int) -> TaskArtefacts:
        row = self.summary[self.summary["task_id"] == task_id]
        if row.empty:
            raise KeyError(f"task_id {task_id} not in summary")
        r = row.iloc[0]
        return TaskArtefacts(
            task_dir=self.task_dir(int(task_id)),
            task_id=int(task_id),
            grid=str(r.get("grid", "")),
            seed=int(r.get("seed", 0)),
            variant=str(r.get("variant", "scare")),
            experiment=str(r.get("experiment", "")),
            ablation=str(r.get("ablation", "default")),
            sweep=str(r.get("sweep", "default")),
            scenario=str(r.get("scenario", "default")),
        )

    def ok(self) -> pd.DataFrame:
        """Subset of summary for tasks with status == 'ok'."""
        return self.summary[self.summary["status"] == "ok"].copy()

    def by_experiment(self, name: str) -> pd.DataFrame:
        if "experiment" not in self.summary.columns:
            return self.summary.iloc[0:0].copy()
        return self.summary[self.summary["experiment"] == name].copy()

    def experiments(self) -> list[str]:
        if "experiment" not in self.summary.columns:
            return []
        return sorted({e for e in self.summary["experiment"].dropna() if e})

    def representative_task(
        self, experiment: str, variant: str = "scare"
    ) -> TaskArtefacts | None:
        """Pick one OK task in the given experiment for trajectory plots."""
        df = self.by_experiment(experiment)
        df = df[df["variant"] == variant]
        df = df[df["status"] == "ok"]
        if df.empty:
            return None
        # Pick the lowest task_id for stability.
        return self.task(int(df["task_id"].iloc[0]))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def load_campaign(campaign_dir: Path) -> CampaignData:
    """Load a campaign's summary.csv + metadata.json.  Per-task
    artefacts are loaded lazily via ``CampaignData.task(...)``.
    """
    campaign_dir = Path(campaign_dir).resolve()
    summary_path = campaign_dir / CAMPAIGN_LAYOUT["summary_csv"]
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary.csv missing at {summary_path}; run "
            f"`python -m experiment.hpc.aggregate --campaign-dir {campaign_dir}` "
            f"first."
        )
    summary = pd.read_csv(summary_path)
    metadata = _read_json(
        campaign_dir / CAMPAIGN_LAYOUT["metadata"], default={}
    )
    return CampaignData(
        campaign_dir=campaign_dir,
        summary=summary,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path, *, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        return default


def _read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
