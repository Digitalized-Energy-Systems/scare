"""One-shot end-to-end evaluation pipeline.

Single entry point that takes only the evaluation-config path and:

1. Plans the campaign         — ``experiment.hpc.plan.create_campaign``
2. Executes every task        — local process pool *or* SLURM array
3. Aggregates the results     — ``experiment.hpc.aggregate.write_summary``
4. Generates plots + report   — ``experiment.eval.report.generate_report``

Usage::

    python -m experiment.eval.run experiment/configs/eval_quick.json
    python -m experiment.eval.run experiment/configs/eval_full.json --mode slurm

Mode auto-detection: if SLURM (``sbatch`` on PATH) is available the
default is ``slurm``, otherwise ``local``.  Override with ``--mode``.
``--workers`` controls the local process pool (default: ``cpu_count − 1``).

Local mode: blocks until every task finishes, then aggregates + reports
inline.  Returns the path to the generated ``REPORT.md``.

SLURM mode: submits the array job, an ``afterany``-aggregator job, and
a ``afterok``-report job as a dependency chain, then returns
immediately.  When the chain completes, ``REPORT.md`` appears in the
campaign directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
from pathlib import Path

from experiment.eval.report import generate_report
from experiment.hpc.aggregate import write_summary
from experiment.hpc.config import CAMPAIGN_LAYOUT, CampaignConfig
from experiment.hpc.plan import (
    compress_ranges,
    create_campaign,
    filter_task_ids,
    read_manifest,
)
from experiment.hpc.run_local import run_campaign
from experiment.hpc.submit import (
    _aggregator_command,
    _array_command,
    _python_bin,
    _run_sbatch,
)

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]


# Mode detection


def _resolve_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    if shutil.which("sbatch") is not None:
        logger.info("auto-detected SLURM (sbatch on PATH) — using --mode slurm")
        return "slurm"
    return "local"


# Local pipeline


def _run_local(campaign_dir: Path, workers: int) -> Path:
    tasks = read_manifest(campaign_dir)
    if not tasks:
        raise SystemExit("Campaign has no tasks — nothing to run.")

    task_ids = [t.task_id for t in tasks]
    logger.info("Local execution: %d task(s) with %d worker(s)", len(task_ids), workers)
    failures = run_campaign(campaign_dir, task_ids, workers)
    if failures:
        logger.warning("%d task(s) exited non-zero — aggregating anyway", failures)

    logger.info("Aggregating results …")
    write_summary(campaign_dir)

    logger.info("Generating plots + REPORT.md …")
    report_path = generate_report(campaign_dir)
    return report_path


# SLURM pipeline


def _run_slurm(campaign_dir: Path) -> Path:
    """Submit the array → aggregator → report dependency chain.  Returns
    the path the report *will* land at once the chain finishes.
    """
    cfg = CampaignConfig.from_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
    tasks = read_manifest(campaign_dir)
    selected = filter_task_ids(campaign_dir, tasks, "all")
    if not selected:
        raise SystemExit("Campaign has no tasks — nothing to submit.")

    array_spec = f"{compress_ranges(selected)}%{cfg.slurm.max_concurrent}"
    log_dir = campaign_dir / CAMPAIGN_LAYOUT["slurm_logs"]
    log_dir.mkdir(exist_ok=True)

    py = _python_bin(cfg.slurm)
    if not Path(py).exists():
        raise SystemExit(f"python interpreter not found: {py}")

    logger.info("Submitting %d task(s) to SLURM", len(selected))
    logger.info("  campaign:  %s", campaign_dir)
    logger.info("  array:     %s", array_spec)
    logger.info("  python:    %s", py)

    array_job = _run_sbatch(
        _array_command(campaign_dir, cfg.slurm, array_spec, log_dir), False
    )
    logger.info("  array job: %s", array_job)

    agg_job = _run_sbatch(
        _aggregator_command(campaign_dir, cfg.slurm, array_job, log_dir), False
    )
    logger.info("  aggregator: %s (afterany:%s)", agg_job, array_job)

    report_job = _run_sbatch(
        _report_command(campaign_dir, cfg.slurm, agg_job, log_dir, py), False
    )
    logger.info("  report:    %s (afterok:%s)", report_job, agg_job)

    return campaign_dir / "REPORT.md"


def _report_command(
    campaign_dir: Path,
    slurm,
    after_job: str,
    log_dir: Path,
    py: str,
) -> list[str]:
    """sbatch command for the plot+report generation step.  Light-weight
    single-task dependency on the aggregator's success.
    """
    job_name = (slurm.job_name or f"scare-restore-{campaign_dir.name}") + "-report"
    wrap = (
        f"cd {shlex.quote(str(_REPO_ROOT))} && "
        f"exec {shlex.quote(py)} -m experiment.eval.report "
        f"--campaign-dir {shlex.quote(str(campaign_dir))}"
    )
    light_flags = ["--cpus-per-task=1", "--mem=2G", "--time=00:15:00"]
    for k in ("partition", "account", "qos", "nodelist", "exclude"):
        v = getattr(slurm, k)
        if v:
            light_flags.append(f"--{k}={v}")
    return [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--dependency=afterok:{after_job}",
        "--kill-on-invalid-dep=yes",
        f"--output={log_dir}/report-%j.out",
        f"--error={log_dir}/report-%j.err",
        *light_flags,
        f"--wrap={wrap}",
    ]


# Top-level orchestration


def run(config_path: Path, *, mode: str = "auto", workers: int | None = None) -> Path:
    """Plan + execute + aggregate + report.  Returns the path the
    ``REPORT.md`` is at (or will be at, once SLURM finishes).
    """
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    cfg = CampaignConfig.from_json(config_path)

    logger.info("Planning campaign %r from %s", cfg.name, config_path)
    campaign_dir = create_campaign(cfg, source_path=config_path)

    resolved_mode = _resolve_mode(mode)
    if resolved_mode == "local":
        return _run_local(campaign_dir, workers or _default_workers())
    if resolved_mode == "slurm":
        return _run_slurm(campaign_dir)
    raise SystemExit(f"Unknown mode: {resolved_mode!r}")


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("config", type=Path, help="Path to a campaign JSON config.")
    p.add_argument(
        "--mode",
        choices=("auto", "local", "slurm"),
        default="auto",
        help="Execution backend.  ``auto`` = SLURM if sbatch on PATH, else local.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Local-mode parallelism (default: cpu_count − 1).  Ignored for SLURM.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
    args = _parse_args()
    report_path = run(args.config.resolve(), mode=args.mode, workers=args.workers)
    print(report_path)


if __name__ == "__main__":
    main()
