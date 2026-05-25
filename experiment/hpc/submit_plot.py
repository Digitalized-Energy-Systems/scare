"""Submit a plotting / report job for a finished campaign to Slurm.

Wraps ``scripts/plot.sh`` in an sbatch call that inherits partition /
nodelist / account from the campaign's ``config.json``. The job runs
aggregation + report rendering, producing ``summary.csv``, ``REPORT.md``,
and ``plots/`` inside the campaign dir.

Usage:
    python -m experiment.hpc.submit_plot <campaign_dir> [--skip-aggregate] [--dry-run]

Notes:
    - ``submit.py`` already auto-fires an aggregator after the array
      (when ``slurm.aggregate=true``); pass ``--skip-aggregate`` to
      avoid redoing that step.
    - Defaults to 4G / 30 min — override via ``MEM`` / ``TIME`` env vars
      or extend the ``SlurmConfig`` if you need a permanent change.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

from experiment.hpc.config import CAMPAIGN_LAYOUT, CampaignConfig, SlurmConfig

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _python_bin(slurm: SlurmConfig) -> str:
    if slurm.python_bin:
        return slurm.python_bin
    venv = _REPO_ROOT / "venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _plot_command(campaign_dir: Path, slurm: SlurmConfig, skip_aggregate: bool, log_dir: Path) -> list[str]:
    job_name = (slurm.job_name or f"scare-restore-{campaign_dir.name}") + "-plot"
    plot_sh = _REPO_ROOT / "scripts" / "plot.sh"
    env_prefix = "SKIP_AGGREGATE=1 " if skip_aggregate else ""
    wrap = (
        f"cd {shlex.quote(str(_REPO_ROOT))} && "
        f"{env_prefix}PYTHON_BIN={shlex.quote(_python_bin(slurm))} "
        f"exec bash {shlex.quote(str(plot_sh))} {shlex.quote(str(campaign_dir))}"
    )
    # Plotting is single-task: bypass heavy slurm flags but keep
    # partition / account / nodelist so it lands on the same slice the
    # user is allowed to use.
    mem = os.environ.get("MEM", "4G")
    time = os.environ.get("TIME", "00:30:00")
    flags = [f"--cpus-per-task=1", f"--mem={mem}", f"--time={time}"]
    for k in ("partition", "account", "qos", "nodelist", "exclude"):
        v = getattr(slurm, k)
        if v:
            flags.append(f"--{k}={v}")
    return [
        "sbatch", "--parsable",
        f"--job-name={job_name}",
        f"--output={log_dir}/plot-%j.out",
        f"--error={log_dir}/plot-%j.err",
        *flags,
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


def submit_plot(campaign_dir: Path, skip_aggregate: bool, dry_run: bool) -> int:
    cfg = CampaignConfig.from_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
    log_dir = campaign_dir / CAMPAIGN_LAYOUT["slurm_logs"]
    log_dir.mkdir(exist_ok=True)

    logger.info("Submitting plot job")
    logger.info("  campaign:    %s", campaign_dir)
    logger.info("  partition:   %s", cfg.slurm.partition or "<default>")
    logger.info("  nodelist:    %s", cfg.slurm.nodelist or "<any>")
    logger.info("  aggregate:   %s", "skip" if skip_aggregate else "run")

    job = _run_sbatch(_plot_command(campaign_dir, cfg.slurm, skip_aggregate, log_dir), dry_run)
    logger.info("  plot job:    %s", job)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("campaign_dir", type=Path)
    p.add_argument("--skip-aggregate", action="store_true",
                   help="Skip the aggregation step (summary.csv already current)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the sbatch command instead of running it")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    args = _parse_args()
    sys.exit(submit_plot(args.campaign_dir.resolve(), args.skip_aggregate, args.dry_run))


if __name__ == "__main__":
    main()
