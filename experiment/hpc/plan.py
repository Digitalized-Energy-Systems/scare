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
from typing import Literal
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment.hpc.config import (
    CAMPAIGN_LAYOUT,
    CampaignConfig,
    TaskSpec,
    task_dir,
)
from experiment.restoration import GRIDS  # noqa: PLC0415  — heavy import

logger = logging.getLogger(__name__)

FilterMode = Literal["all", "ok", "failed", "timeout", "missing", "incomplete"]


# ---- Manifest construction --------------------------------------------------


def derive_n_failures(seed: int, failure_lambda: float, max_failures: int) -> int:
    rng = np.random.default_rng(seed)
    n = 1 + int(rng.poisson(failure_lambda))
    return max(1, min(max_failures, n))


def build_tasks(cfg: CampaignConfig) -> list[TaskSpec]:
    """Expand the campaign config into a flat task list.

    Two modes:

    1. **Legacy** — top-level ``grids`` list, no ``experiments``: each
       grid produces ``runs_per_grid`` tasks of variant ``scare`` with
       no ablation / sweep / scenario customisation.  Identical to the
       pre-eval behaviour, byte-compatible manifest.

    2. **Eval** — ``experiments`` list, each entry contributes a
       Cartesian product across (grids × seeds × variants × ablations
       × sweeps × scenarios).  Empty ``grids`` on an experiment marks
       it as a TODO placeholder and emits a metadata note instead of
       expanding it — this lets the campaign config carry the missing
       grids' shape so the reviewer sees the gap.
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


def _build_eval_tasks(cfg: CampaignConfig) -> list[TaskSpec]:
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
                    for sweep in exp.sweeps or [{}]:
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
                                # Per-scenario overrides for the
                                # failure-count machinery.  Targeted
                                # scenarios (e.g.\ ``concentrated``)
                                # often need an exact count rather
                                # than a Poisson draw, so we accept:
                                #   - ``n_failures``     exact count, bypasses sampling
                                #   - ``failure_lambda`` override Poisson lambda
                                #   - ``max_failures``   override per-grid cap
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
    if source_path is not None and Path(source_path).resolve() != (campaign_dir / CAMPAIGN_LAYOUT["config"]):
        (campaign_dir / CAMPAIGN_LAYOUT["config_source"]).write_text(Path(source_path).read_text())

    tasks = build_tasks(cfg)
    with (campaign_dir / CAMPAIGN_LAYOUT["manifest"]).open("w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), sort_keys=True) + "\n")

    _write_metadata(campaign_dir, cfg, len(tasks))
    _log_breakdown(cfg, tasks)
    logger.info("Campaign ready: %s", campaign_dir)
    return campaign_dir


def _log_breakdown(cfg: CampaignConfig, tasks: list[TaskSpec]) -> None:
    logger.info("Wrote %d task(s) across %d grid(s)", len(tasks), len(cfg.grids))
    by_grid: dict[str, dict[int, int]] = {}
    for t in tasks:
        by_grid.setdefault(t.grid, {}).setdefault(t.n_failures, 0)
        by_grid[t.grid][t.n_failures] += 1
    for g, dist in by_grid.items():
        breakdown = ", ".join(f"{n}f={c}" for n, c in sorted(dist.items()))
        logger.info("  %-25s %s", g, breakdown)


# ---- Manifest reading + filtering ------------------------------------------


def read_manifest(campaign_dir: Path) -> list[TaskSpec]:
    path = Path(campaign_dir) / CAMPAIGN_LAYOUT["manifest"]
    out: list[TaskSpec] = []
    with path.open() as f:
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
        return str(json.loads(f.read_text()).get("status", "missing"))
    except json.JSONDecodeError:
        return "missing"


def filter_task_ids(
    campaign_dir: Path,
    tasks: Iterable[TaskSpec],
    mode,
) -> list[int]:
    """Pick task IDs whose on-disk status matches ``mode``.

    ``incomplete`` covers both crashed/timed-out and never-started tasks
    — i.e. anything that isn't ``ok`` — which is the right default for
    "resubmit what didn't finish".
    """
    if mode == "all":
        return [t.task_id for t in tasks]

    # ``claims_failed`` and ``ok`` both completed the simulation; only
    # ``claims_failed`` carries a chapter-claim violation flag.  Treat
    # both as "done" for re-run filtering so a claim regression doesn't
    # cause a wholesale recompute.
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


# ---- Metadata --------------------------------------------------------------


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
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
        json.dumps(metadata, indent=2, sort_keys=True)
    )


# ---- Optional cache warm-up ------------------------------------------------


def prebuild_grids(grid_names: list[str]) -> None:
    """Build each grid factory once locally to warm caches before submission.

    Avoids 32 array tasks racing on a first-time simbench download.
    """

    for g in grid_names:
        if g not in GRIDS:
            raise SystemExit(f"Unknown grid {g!r}; available: {sorted(GRIDS)}")
        logger.info("Pre-building grid %s …", g)
        GRIDS[g]()
        logger.info("  ok")


# ---- CLI -------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a campaign directory from a JSON config.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("config", type=Path, help="Path to a campaign JSON config")
    p.add_argument(
        "--no-timestamp", action="store_true",
        help="Use the bare config name as the dir; default appends a UTC stamp",
    )
    p.add_argument(
        "--prebuild", action="store_true",
        help="Build every grid once locally to warm caches before submission",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    args = _parse_args()
    cfg = CampaignConfig.from_json(args.config)
    timestamp_dir = False if args.no_timestamp else None  # None → use cfg.timestamp_dir
    if args.prebuild:
        prebuild_grids([g.name for g in cfg.grids])
    campaign_dir = create_campaign(cfg, source_path=args.config, timestamp_dir=timestamp_dir)
    print(campaign_dir)


if __name__ == "__main__":
    main()
