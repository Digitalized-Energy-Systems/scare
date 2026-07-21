"""Build a campaign directory from a JSON config.

Usage:
    python -m experiment.hpc.plan path/to/config.json [--prebuild] [--no-timestamp]

Layout produced under ``<config.out_root>/<config.name>[_<UTC-stamp>]/``:

    config.json         resolved CampaignConfig (source of truth for runner / submit)
    config.source.json  the file the user passed in (for diffing later)
    manifest.jsonl      one TaskSpec per line
    metadata.json       git, host, python, slurm-job (if invoked under Slurm)
    tasks/              filled in by the runner

Also exposes :func:`read_manifest` and :func:`filter_task_ids` used by
the runner, aggregator, submit, and local-driver modules.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from experiment.hpc.config import (
    CAMPAIGN_LAYOUT,
    CampaignConfig,
    TaskSpec,
    task_dir,
)
from experiment.scenarios import GRIDS  # noqa: PLC0415  heavy import

logger = logging.getLogger(__name__)

FilterMode = Literal["all", "ok", "failed", "timeout", "missing", "incomplete"]


def derive_n_failures(seed: int, failure_lambda: float, max_failures: int) -> int:
    rng = np.random.default_rng(seed)
    n = 1 + int(rng.poisson(failure_lambda))
    return max(1, min(max_failures, n))


def build_tasks(cfg: CampaignConfig) -> list[TaskSpec]:
    """Expand the campaign config into a flat task list.

    Two modes:

    1. **Legacy** — top-level ``grids`` list, no ``experiments``: each
       grid produces ``runs_per_grid`` scare tasks with no axis
       customisation. Byte-compatible with the pre-eval manifest.

    2. **Eval** — ``experiments`` list, each contributing the Cartesian
       product across (grids x seeds x variants x ablations x sweeps x
       scenarios). Empty ``grids`` marks a TODO placeholder, skipped but
       noted in metadata so the reviewer sees the gap.
    """
    if cfg.experiments:
        return _build_eval_tasks(cfg)
    return _build_legacy_tasks(cfg)


def _build_legacy_tasks(cfg: CampaignConfig) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for g_idx, grid in enumerate(cfg.grids):
        runs, lam, max_f = cfg.resolve_grid(grid)
        for run_idx in range(runs):
            seed = cfg.base_seed + g_idx * 1_000_000 + run_idx
            tasks.append(
                TaskSpec(
                    task_id=len(tasks),
                    grid=grid.name,
                    seed=seed,
                    n_failures=derive_n_failures(seed, lam, max_f),
                )
            )
    return tasks


def _validate_config_overrides(cfg: CampaignConfig) -> None:
    """Fail planning on ablation/sweep keys that are not
    ``RestorationConfiguration`` fields — the runner would otherwise silently
    ignore them, degrading the arm to baseline."""
    from scare.base.config import RestorationConfiguration

    valid = {f.name for f in fields(RestorationConfiguration)}
    for exp in cfg.experiments:
        for axis, dicts in (("ablations", exp.ablations), ("sweeps", exp.sweeps)):
            for d in dicts:
                # $-prefixed keys are annotations (e.g. $label drives the
                # aggregate arm labels), stripped by the config layer.
                unknown = sorted(k for k in set(d) - valid if not k.startswith("$"))
                if unknown:
                    raise ValueError(
                        f"experiment {exp.name!r}: unknown "
                        f"RestorationConfiguration field(s) in {axis}: {unknown}"
                    )


def _build_eval_tasks(cfg: CampaignConfig) -> list[TaskSpec]:
    _validate_config_overrides(cfg)
    tasks: list[TaskSpec] = []
    for e_idx, exp in enumerate(cfg.experiments):
        if not exp.grids:
            logger.info(
                "experiment %r: no grids — TODO placeholder, skipping expansion",
                exp.name,
            )
            continue
        for g_idx, grid in enumerate(exp.grids):
            runs, lam, max_f = cfg.resolve_grid(grid)
            n_seeds = exp.n_seeds or runs
            for variant in exp.variants or ["scare"]:
                for ablation in exp.ablations or [{}]:
                    # Ablation/sweep keys are RestorationConfiguration fields —
                    # MAS-side only. The oracle solve reads grid/scenario/seed/
                    # priorities and never the config, so an oracle arm under a
                    # non-empty ablation is a bit-identical duplicate of the
                    # oracle control (measured: 25 dead tasks in eval_full_v2).
                    if variant == "oracle" and ablation:
                        continue
                    for sweep in exp.sweeps or [{}]:
                        if variant == "oracle" and sweep:
                            continue
                        for scenario in exp.scenarios or [{"kind": "clean"}]:
                            for run_idx in range(n_seeds):
                                # Seed mixes experiment, grid, run so
                                # different ablations on the same
                                # (grid, run) share the same failure draw.
                                seed = (
                                    cfg.base_seed
                                    + e_idx * 100_000_000
                                    + g_idx * 1_000_000
                                    + run_idx
                                )
                                # Per-scenario failure-count overrides:
                                # ``n_failures``     exact count, bypasses sampling
                                # ``failure_lambda`` override Poisson lambda
                                # ``max_failures``   override per-grid cap
                                sc_max = scenario.get("max_failures", max_f)
                                if "n_failures" in scenario:
                                    n_fail = max(1, int(scenario["n_failures"]))
                                else:
                                    sc_lam = float(scenario.get("failure_lambda", lam))
                                    n_fail = derive_n_failures(seed, sc_lam, sc_max)
                                tasks.append(
                                    TaskSpec(
                                        task_id=len(tasks),
                                        grid=grid.name,
                                        seed=seed,
                                        n_failures=n_fail,
                                        variant=variant,
                                        experiment=exp.name,
                                        ablation=dict(ablation),
                                        sweep=dict(sweep),
                                        scenario=dict(scenario),
                                    )
                                )
    return tasks


def create_campaign(
    cfg: CampaignConfig,
    *,
    source_path: Path | None = None,
    timestamp_dir: bool | None = None,
) -> Path:
    use_ts = cfg.timestamp_dir if timestamp_dir is None else timestamp_dir
    name = cfg.name
    if use_ts:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{name}_{stamp}"

    campaign_dir = (Path(cfg.out_root) / name).resolve()
    if campaign_dir.exists():
        raise FileExistsError(
            f"campaign dir already exists: {campaign_dir} "
            f"(remove it, or set timestamp_dir=true to auto-disambiguate)"
        )
    campaign_dir.mkdir(parents=True)
    (campaign_dir / CAMPAIGN_LAYOUT["tasks"]).mkdir()

    cfg.to_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
    if source_path is not None and Path(source_path).resolve() != (
        campaign_dir / CAMPAIGN_LAYOUT["config"]
    ):
        (campaign_dir / CAMPAIGN_LAYOUT["config_source"]).write_text(
            Path(source_path).read_text(encoding="utf-8"), encoding="utf-8"
        )

    tasks = build_tasks(cfg)
    with (campaign_dir / CAMPAIGN_LAYOUT["manifest"]).open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), sort_keys=True) + "\n")

    _write_metadata(campaign_dir, cfg, len(tasks))
    _log_breakdown(tasks)
    logger.info("Campaign ready: %s", campaign_dir)
    return campaign_dir


def _log_breakdown(tasks: list[TaskSpec]) -> None:
    # cfg.grids is empty in eval mode; count the grids the tasks actually use.
    logger.info(
        "Wrote %d task(s) across %d grid(s)",
        len(tasks),
        len({t.grid for t in tasks}),
    )
    by_grid: dict[str, dict[int, int]] = {}
    for t in tasks:
        by_grid.setdefault(t.grid, {}).setdefault(t.n_failures, 0)
        by_grid[t.grid][t.n_failures] += 1
    for g, dist in by_grid.items():
        breakdown = ", ".join(f"{n}f={c}" for n, c in sorted(dist.items()))
        logger.info("  %-25s %s", g, breakdown)


# Manifest reading + filtering


def read_manifest(campaign_dir: Path) -> list[TaskSpec]:
    path = Path(campaign_dir) / CAMPAIGN_LAYOUT["manifest"]
    out: list[TaskSpec] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(TaskSpec(**json.loads(line)))
    return out


def task_status(campaign_dir: Path, task: TaskSpec) -> str:
    """Return ``ok|claims_failed|error|timeout|killed|missing`` for a task on disk."""
    f = task_dir(campaign_dir, task.task_id) / "status.json"
    if not f.exists():
        return "missing"
    try:
        return str(json.loads(f.read_text(encoding="utf-8")).get("status", "missing"))
    except json.JSONDecodeError:
        return "missing"


def filter_task_ids(
    campaign_dir: Path,
    tasks: Iterable[TaskSpec],
    mode,
) -> list[int]:
    """Pick task IDs whose on-disk status matches ``mode``.

    ``failed`` = error|killed; ``incomplete`` = anything not ``ok``. OOM-kills write
    no status.json and read ``missing``; pick up via ``missing``/``incomplete`` ONLY
    after the Slurm array drains (a queued task also reads ``missing``, so resubmitting double-runs it via mutual artifact scrubbing).
    """
    if mode == "all":
        return [t.task_id for t in tasks]

    # Both ``ok`` and ``claims_failed`` completed the sim; treat both as
    # "done" so a claim failure doesn't trigger a wholesale recompute.
    completed = ("ok", "claims_failed")
    out: list[int] = []
    for t in tasks:
        s = task_status(campaign_dir, t)
        if mode == "missing" and s == "missing":
            out.append(t.task_id)
        elif mode == "failed" and s in ("error", "killed"):
            out.append(t.task_id)
        elif mode == "timeout" and s == "timeout":
            out.append(t.task_id)
        elif mode == "ok" and s in completed:
            out.append(t.task_id)
        elif mode == "incomplete" and s not in completed:
            out.append(t.task_id)
    return out


def compress_ranges(ids: Iterable[int]) -> str:
    """Render a sorted task-id list as a Slurm array spec like ``0-3,7,10-12``."""
    ordered = sorted(set(ids))
    if not ordered:
        return ""
    parts: list[str] = []
    start = end = ordered[0]
    for x in ordered[1:]:
        if x == end + 1:
            end = x
        else:
            parts.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = x
    parts.append(f"{start}-{end}" if start != end else f"{start}")
    return ",".join(parts)


# Metadata


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None


def _write_metadata(campaign_dir: Path, cfg: CampaignConfig, n_tasks: int) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "python": sys.version,
        "platform": platform.platform(),
        "repo_root": str(repo_root),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(repo_root, "status", "--porcelain") or ""),
        "n_tasks": n_tasks,
        "campaign_config": asdict(cfg),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (campaign_dir / CAMPAIGN_LAYOUT["metadata"]).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


# Optional cache warm-up


def prebuild_grids(grid_names: list[str]) -> None:
    """Build each grid factory once to warm caches before submission,
    so array tasks don't race on a first-time simbench download.
    """

    for g in grid_names:
        if g not in GRIDS:
            raise SystemExit(f"Unknown grid {g!r}; available: {sorted(GRIDS)}")
        logger.info("Pre-building grid %s …", g)
        GRIDS[g]()
        logger.info("  ok")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a campaign directory from a JSON config.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("config", type=Path, help="Path to a campaign JSON config")
    p.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use the bare config name as the dir; default appends a UTC stamp",
    )
    p.add_argument(
        "--prebuild",
        action="store_true",
        help="Build every grid once locally to warm caches before submission",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
    args = _parse_args()
    cfg = CampaignConfig.from_json(args.config)
    timestamp_dir = False if args.no_timestamp else None  # None → use cfg.timestamp_dir
    if args.prebuild:
        # cfg.grids is empty in eval mode (grids live under experiments[].grids),
        # so union both or --prebuild warms nothing and array tasks race the build.
        grid_names = {g.name for g in cfg.grids}
        grid_names |= {g.name for e in cfg.experiments for g in e.grids}
        prebuild_grids(sorted(grid_names))
    campaign_dir = create_campaign(
        cfg, source_path=args.config, timestamp_dir=timestamp_dir
    )
    print(campaign_dir)


if __name__ == "__main__":
    main()
