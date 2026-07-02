"""Submit a prepared campaign to Slurm using settings from its config.json.

Usage:
    python -m experiment.hpc.submit <campaign_dir> [--only MODE] [--dry-run]

``--only`` lets you re-submit just a subset:
    all          (default) every task in manifest
    failed       tasks whose status.json shows error/killed
    timeout      only tasks that timed out
    missing      only tasks that never started / never wrote status.json
                 (covers OOM-killed tasks, which never write status.json)
    incomplete   anything that isn't ok (covers failed + timeout + missing)

``missing`` / ``incomplete`` must only be used after the array has drained:
a still-queued task also reads as missing, and resubmitting it double-runs
the task (the two runs scrub each other's artifacts).
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

from experiment.hpc.config import CAMPAIGN_LAYOUT, CampaignConfig, SlurmConfig
from experiment.hpc.plan import (
    FilterMode,
    compress_ranges,
    filter_task_ids,
    read_manifest,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _python_bin(slurm: SlurmConfig) -> str:
    if slurm.python_bin:
        return slurm.python_bin
    venv = _REPO_ROOT / "venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _slurm_flags(slurm: SlurmConfig) -> list[str]:
    flags = [
        f"--cpus-per-task={slurm.cpus}",
        f"--mem={slurm.mem}",
        f"--time={slurm.time}",
    ]
    if slurm.partition:
        flags.append(f"--partition={slurm.partition}")
    if slurm.account:
        flags.append(f"--account={slurm.account}")
    if slurm.qos:
        flags.append(f"--qos={slurm.qos}")
    if slurm.nodelist:
        flags.append(f"--nodelist={slurm.nodelist}")
    if slurm.exclude:
        flags.append(f"--exclude={slurm.exclude}")
    flags.extend(slurm.extra_sbatch_args)
    return flags


def _array_command(
    campaign_dir: Path, slurm: SlurmConfig, array_spec: str, log_dir: Path
) -> list[str]:
    job_name = slurm.job_name or f"scare-restore-{campaign_dir.name}"
    py = _python_bin(slurm)
    wrap = (
        f"cd {shlex.quote(str(_REPO_ROOT))} && "
        f"exec {shlex.quote(py)} -m experiment.hpc.runner "
        f"--campaign-dir {shlex.quote(str(campaign_dir))}"
    )
    return [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--array={array_spec}",
        f"--output={log_dir}/slurm-%A_%a.out",
        f"--error={log_dir}/slurm-%A_%a.err",
        *_slurm_flags(slurm),
        f"--wrap={wrap}",
    ]


def _campaign_eval_slurm(campaign_dir: Path, fallback: SlurmConfig) -> SlurmConfig:
    """Effective eval-job sizing from the campaign's config.json; falls back
    to the given per-task SlurmConfig if the config can't be read (callers may
    only hold the SlurmConfig, not the full CampaignConfig)."""
    try:
        cfg = CampaignConfig.from_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
        return cfg.effective_eval_slurm()
    except (OSError, ValueError):
        return fallback


def _aggregator_command(
    campaign_dir: Path, slurm: SlurmConfig, after_job: str, log_dir: Path
) -> list[str]:
    job_name = (slurm.job_name or f"scare-restore-{campaign_dir.name}") + "-agg"
    py = _python_bin(slurm)
    wrap = (
        f"cd {shlex.quote(str(_REPO_ROOT))} && "
        f"exec {shlex.quote(py)} -m experiment.hpc.aggregate "
        f"--campaign-dir {shlex.quote(str(campaign_dir))}"
    )
    # Size from the config's slurm_eval overlay (like submit_plot) instead of
    # hardcoding 2G/10min, which OOM'd/timed out on large campaigns.
    eval_slurm = _campaign_eval_slurm(campaign_dir, slurm)
    light_flags = [
        f"--cpus-per-task={eval_slurm.cpus}",
        f"--mem={eval_slurm.mem}",
        f"--time={eval_slurm.time}",
    ]
    for k in ("partition", "account", "qos", "nodelist", "exclude"):
        v = getattr(eval_slurm, k)
        if v:
            light_flags.append(f"--{k}={v}")
    return [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--dependency=afterany:{after_job}",
        "--kill-on-invalid-dep=yes",
        f"--output={log_dir}/aggregate-%j.out",
        f"--error={log_dir}/aggregate-%j.err",
        *light_flags,
        f"--wrap={wrap}",
    ]


def _run_sbatch(cmd: list[str], dry_run: bool) -> str:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    if dry_run:
        print(pretty)
        return "<dry-run>"
    logger.debug("$ %s", pretty)
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"sbatch failed (exit {res.returncode})")
    return res.stdout.strip().split(";", 1)[0]


def submit(campaign_dir: Path, only: FilterMode, dry_run: bool) -> int:
    cfg = CampaignConfig.from_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
    tasks = read_manifest(campaign_dir)
    selected = filter_task_ids(campaign_dir, tasks, only)

    if not selected:
        logger.warning("No tasks selected (mode=%s); nothing to submit.", only)
        return 0

    array_spec = f"{compress_ranges(selected)}%{cfg.slurm.max_concurrent}"
    log_dir = campaign_dir / CAMPAIGN_LAYOUT["slurm_logs"]
    log_dir.mkdir(exist_ok=True)

    py = _python_bin(cfg.slurm)
    if not Path(py).exists():
        raise SystemExit(f"python interpreter not found: {py}")

    logger.info("Submitting %d/%d task(s) (mode=%s)", len(selected), len(tasks), only)
    logger.info("  campaign:    %s", campaign_dir)
    logger.info("  array:       %s", array_spec)
    logger.info("  python:      %s", py)
    logger.info("  partition:   %s", cfg.slurm.partition or "<default>")
    logger.info("  nodelist:    %s", cfg.slurm.nodelist or "<any>")
    logger.info(
        "  time/mem:    %s / %s (cpus=%d)",
        cfg.slurm.time,
        cfg.slurm.mem,
        cfg.slurm.cpus,
    )

    array_job = _run_sbatch(
        _array_command(campaign_dir, cfg.slurm, array_spec, log_dir), dry_run
    )
    logger.info("  array job:   %s", array_job)

    if cfg.slurm.aggregate:
        agg_job = _run_sbatch(
            _aggregator_command(campaign_dir, cfg.slurm, array_job, log_dir), dry_run
        )
        logger.info("  aggregator:  %s (afterany:%s)", agg_job, array_job)

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("campaign_dir", type=Path)
    p.add_argument(
        "--only",
        default="all",
        choices=["all", "failed", "timeout", "missing", "incomplete", "ok"],
        help="Filter task subset to (re-)submit (default: all); "
        "'failed' covers error/killed; use 'missing'/'incomplete' only after "
        "the array has drained (a queued task also reads as missing)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command(s) instead of running them",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
    args = _parse_args()
    sys.exit(submit(args.campaign_dir.resolve(), args.only, args.dry_run))


if __name__ == "__main__":
    main()
