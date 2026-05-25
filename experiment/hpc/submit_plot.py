"""Submit a plotting / report job for a finished campaign to Slurm.

Wraps ``scripts/plot.sh`` in an sbatch call that inherits its Slurm
settings from the campaign's ``config.json``. The job runs aggregation
+ report rendering, producing ``summary.csv``, ``REPORT.md``, and
``plots/`` inside the campaign dir.

Sizing precedence (highest first):
    1. ``slurm_eval`` block in config.json (partial overrides on top
       of ``slurm``) — typical use: bump mem / time for large campaigns
       without affecting per-task array sizing.
    2. ``slurm`` block in config.json.

Usage:
    python -m experiment.hpc.submit_plot <campaign_dir> [--skip-aggregate] [--dry-run]

Note: ``submit.py`` already auto-fires an aggregator after the array
(when ``slurm.aggregate=true``); pass ``--skip-aggregate`` to avoid
redoing that step.
"""

from __future__ import annotations

import argparse
import logging
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
    flags = [
        f"--cpus-per-task={slurm.cpus}",
        f"--mem={slurm.mem}",
        f"--time={slurm.time}",
    ]
    for k in ("partition", "account", "qos", "nodelist", "exclude"):
        v = getattr(slurm, k)
        if v:
            flags.append(f"--{k}={v}")
    flags.extend(slurm.extra_sbatch_args)
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
    slurm = cfg.effective_eval_slurm()
    log_dir = campaign_dir / CAMPAIGN_LAYOUT["slurm_logs"]
    log_dir.mkdir(exist_ok=True)

    logger.info("Submitting plot job")
    logger.info("  campaign:    %s", campaign_dir)
    logger.info("  partition:   %s", slurm.partition or "<default>")
    logger.info("  nodelist:    %s", slurm.nodelist or "<any>")
    logger.info("  time/mem:    %s / %s (cpus=%d)", slurm.time, slurm.mem, slurm.cpus)
    logger.info("  overrides:   %s", cfg.slurm_eval or "<none — using slurm block>")
    logger.info("  aggregate:   %s", "skip" if skip_aggregate else "run")

    job = _run_sbatch(_plot_command(campaign_dir, slurm, skip_aggregate, log_dir), dry_run)
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
