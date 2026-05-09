"""Run a campaign locally via a process pool — same on-disk layout as Slurm.

Useful for development on a workstation, smoke-tests, or replaying a
single failing task. Supports the same ``--only`` filter as the Slurm
submit script so you can locally re-run just the failed/timeout/missing
subset.

CLI:
    python -m experiment.hpc.run_local --campaign-dir runs/foo --workers 8
    python -m experiment.hpc.run_local --campaign-dir runs/foo --only failed
    python -m experiment.hpc.run_local --campaign-dir runs/foo --task-ids 3 7 12
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiment.hpc.plan import filter_task_ids, read_manifest

logger = logging.getLogger(__name__)


def _worker(campaign_dir_str: str, task_id: int) -> tuple[int, int]:
    # Imported lazily so child processes don't pay import cost up front.
    from experiment.hpc.runner import run_task

    code = run_task(Path(campaign_dir_str), task_id)
    return task_id, code


def run_campaign(campaign_dir: Path, task_ids: list[int], workers: int) -> int:
    failures = 0
    if workers <= 1:
        for tid in task_ids:
            _, code = _worker(str(campaign_dir), tid)
            failures += int(code != 0)
            logger.info("Task %d → exit=%d", tid, code)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_worker, str(campaign_dir), tid): tid for tid in task_ids}
            done = 0
            for fut in as_completed(futures):
                tid, code = fut.result()
                done += 1
                failures += int(code != 0)
                logger.info("[%d/%d] Task %d → exit=%d", done, len(task_ids), tid, code)
    return failures


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--campaign-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--task-ids", type=int, nargs="*",
                   help="Run only this subset (overrides --only)")
    p.add_argument("--only", default="all",
                   choices=["all", "failed", "timeout", "missing", "incomplete", "ok"],
                   help="Filter task subset by on-disk status (default: all)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    args = _parse_args()
    campaign_dir = args.campaign_dir.resolve()
    tasks = read_manifest(campaign_dir)

    if args.task_ids:
        selected = list(args.task_ids)
    else:
        selected = filter_task_ids(campaign_dir, tasks, args.only)

    if not selected:
        logger.warning("Nothing to run (mode=%s).", args.only)
        sys.exit(0)

    logger.info("Running %d/%d task(s) with %d worker(s)",
                len(selected), len(tasks), args.workers)
    failures = run_campaign(campaign_dir, selected, args.workers)
    if failures:
        logger.warning("%d task(s) exited non-zero", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
