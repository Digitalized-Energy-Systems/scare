"""Typed configuration for an HPC evaluation campaign.

The user-facing surface is one JSON file (``CampaignConfig``); everything
else (manifest, Slurm flags, runner runtime params) is derived from it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    """One reproducible run; everything else is derived from ``seed``.

    The four evaluation axes (variant / ablation / sweep / scenario)
    default to "the established baseline" so legacy task lists stay
    valid: an old ``manifest.jsonl`` written by the pre-eval planner is
    a strict subset of the new format and the planner / runner read
    extras with ``getattr(..., default)``.
    """

    task_id: int
    grid: str
    seed: int
    n_failures: int
    # ---- Evaluation axes (all optional for backward compat) --------
    variant: str = "scare"                 # "scare" | "single_level" | "oracle"
    experiment: str = ""                    # campaign-internal label
    ablation: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, Any] = field(default_factory=dict)
    scenario: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GridSpec:
    """A grid in the manifest; ``None`` overrides fall back to defaults."""

    name: str
    runs_per_grid: int | None = None
    failure_lambda: float | None = None
    max_failures: int | None = None

    @classmethod
    def parse(cls, item: Any) -> "GridSpec":
        if isinstance(item, str):
            return cls(name=item)
        if isinstance(item, dict):
            return cls(**item)
        raise TypeError(f"grids[] entries must be str or object, got {type(item).__name__}")


@dataclass
class GridDefaults:
    runs_per_grid: int = 32
    # n_failures = clip(1 + Poisson(failure_lambda), 1, max_failures).
    # 0.6 ≈ 55% single, 33% double, 10% triple, 2% quadruple-or-more failures.
    failure_lambda: float = 0.6
    max_failures: int = 5


@dataclass
class SlurmConfig:
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodelist: str | None = None
    exclude: str | None = None
    time: str = "00:30:00"
    mem: str = "4G"
    cpus: int = 1
    max_concurrent: int = 32
    aggregate: bool = True
    python_bin: str | None = None
    job_name: str | None = None
    extra_sbatch_args: list[str] = field(default_factory=list)


@dataclass
class ExperimentSpec:
    """One experiment in a campaign — expands into TaskSpecs via
    Cartesian product of (grids × seeds × variants × ablations × sweeps × scenarios).

    Each list defaults to a single trivial entry so an experiment with
    no axes set is just "run baseline scare on these grids and seeds".
    Empty ``grids`` marks the experiment as a TODO placeholder
    (skipped at expansion, surfaced in metadata).
    """
    name: str
    grids: list[GridSpec] = field(default_factory=list)
    n_seeds: int = 0                          # 0 ⇒ defer to campaign default
    variants: list[str] = field(default_factory=lambda: ["scare"])
    ablations: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    sweeps: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    scenarios: list[dict[str, Any]] = field(
        default_factory=lambda: [{"kind": "clean"}]
    )
    notes: str = ""

    @classmethod
    def parse(cls, item: Any) -> "ExperimentSpec":
        if not isinstance(item, dict):
            raise TypeError(
                f"experiments[] entries must be objects, got {type(item).__name__}"
            )
        item = {k: v for k, v in item.items() if not k.startswith("$")}
        grids_raw = item.pop("grids", [])
        return cls(
            grids=[GridSpec.parse(g) for g in grids_raw],
            **item,
        )


@dataclass
class CampaignConfig:
    name: str
    grids: list[GridSpec] = field(default_factory=list)
    experiments: list[ExperimentSpec] = field(default_factory=list)
    out_root: str = "experiment/_runs"
    base_seed: int = 0
    simulation_duration_s: float = 30.0
    task_timeout_s: float = 1500.0
    failure_delay_s_max: float = 2.0
    write_timeseries: bool = True
    timestamp_dir: bool = True
    notes: str = ""
    defaults: GridDefaults = field(default_factory=GridDefaults)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # ---- I/O -----------------------------------------------------------

    @classmethod
    def from_json(cls, path: Path) -> "CampaignConfig":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "CampaignConfig":
        # Strip any unknown top-level keys (e.g. "$schema") so users can
        # add comments-via-keys without breaking the loader.
        data = {k: v for k, v in data.items() if not k.startswith("$")}
        grids_raw = data.pop("grids", [])
        experiments_raw = data.pop("experiments", [])
        if not grids_raw and not experiments_raw:
            raise ValueError(
                "config must define either a top-level 'grids' list "
                "(legacy single-experiment campaign) or 'experiments' "
                "(multi-experiment evaluation campaign)"
            )
        defaults_raw = data.pop("defaults", {})
        slurm_raw = data.pop("slurm", {})
        try:
            return cls(
                grids=[GridSpec.parse(g) for g in grids_raw],
                experiments=[ExperimentSpec.parse(e) for e in experiments_raw],
                defaults=GridDefaults(**defaults_raw),
                slurm=SlurmConfig(**slurm_raw),
                **data,
            )
        except TypeError as exc:
            raise ValueError(f"invalid config: {exc}") from exc

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    # ---- Derived ------------------------------------------------------

    def resolve_grid(self, g: GridSpec) -> tuple[int, float, int]:
        """Apply per-grid overrides over campaign defaults."""
        return (
            g.runs_per_grid if g.runs_per_grid is not None else self.defaults.runs_per_grid,
            g.failure_lambda if g.failure_lambda is not None else self.defaults.failure_lambda,
            g.max_failures if g.max_failures is not None else self.defaults.max_failures,
        )


@dataclass
class RuntimePlan:
    """Subset of the config needed inside the runner — kept narrow so the
    runner doesn't depend on slurm/grid metadata."""

    simulation_duration_s: float = 30.0
    task_timeout_s: float = 1500.0
    failure_delay_s_max: float = 2.0
    write_timeseries: bool = True

    @classmethod
    def from_config_json(cls, path: Path) -> "RuntimePlan":
        data = json.loads(Path(path).read_text())
        return cls(
            simulation_duration_s=float(data.get("simulation_duration_s", 30.0)),
            task_timeout_s=float(data.get("task_timeout_s", 1500.0)),
            failure_delay_s_max=float(data.get("failure_delay_s_max", 2.0)),
            write_timeseries=bool(data.get("write_timeseries", True)),
        )


CAMPAIGN_LAYOUT = {
    "config": "config.json",        # the resolved CampaignConfig (source of truth)
    "config_source": "config.source.json",  # original config file as user passed it
    "manifest": "manifest.jsonl",
    "metadata": "metadata.json",
    "tasks": "tasks",
    "summary_csv": "summary.csv",
    "summary_md": "summary.md",
    "slurm_logs": "slurm_logs",
}


def task_dir(campaign_dir: Path, task_id: int) -> Path:
    return Path(campaign_dir) / CAMPAIGN_LAYOUT["tasks"] / f"{task_id:06d}"
