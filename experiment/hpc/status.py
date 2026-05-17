"""At-a-glance status of an in-flight or completed campaign.

Usage:
    python -m experiment.hpc.status <campaign_dir> [--show-failed N]

Prints overall counts, a per-grid breakdown, and the most recent failed
tasks (with exception type) so you can decide whether to ``--only failed``
resubmit or chase a bug.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from experiment.hpc.config import CAMPAIGN_LAYOUT, task_dir
from experiment.hpc.plan import read_manifest, task_status

logger = logging.getLogger(__name__)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "  (no rows)"
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    sep = "  "
    out = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        out.append(sep.join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def report(campaign_dir: Path, show_failed: int) -> None:
    tasks = read_manifest(campaign_dir)
    statuses = {t.task_id: task_status(campaign_dir, t) for t in tasks}

    overall = Counter(statuses.values())
    n = len(tasks)
    pct = lambda c: f"{100.0 * c / n:5.1f}%" if n else "  n/a"  # noqa: E731

    print(f"Campaign: {campaign_dir}")
    print(f"Tasks:    {n}")
    print()
    print(_format_table(
        ["status", "count", "%"],
        [[s, str(overall.get(s, 0)), pct(overall.get(s, 0))]
         for s in ("ok", "claims_failed", "error", "timeout", "killed", "missing")],
    ))

    by_grid: dict[str, Counter] = defaultdict(Counter)
    for t in tasks:
        by_grid[t.grid][statuses[t.task_id]] += 1

    print()
    print("Per grid:")
    rows = []
    for g, counts in sorted(by_grid.items()):
        total = sum(counts.values())
        rows.append([
            g, str(total),
            str(counts.get("ok", 0)),
            str(counts.get("claims_failed", 0)),
            str(counts.get("error", 0)),
            str(counts.get("timeout", 0)),
            str(counts.get("killed", 0)),
            str(counts.get("missing", 0)),
        ])
    print(_format_table(
        ["grid", "total", "ok", "claims_failed", "error",
         "timeout", "killed", "missing"], rows,
    ))

    if show_failed > 0:
        failed = [t for t in tasks
                  if statuses[t.task_id] in ("error", "timeout", "killed")]
        if failed:
            print()
            print(f"Last {min(show_failed, len(failed))} failed task(s):")
            rows = []
            for t in failed[-show_failed:]:
                td = task_dir(campaign_dir, t.task_id)
                exc_type = ""
                exc_msg = ""
                exc_file = td / "exception.json"
                if exc_file.exists():
                    try:
                        d = json.loads(exc_file.read_text())
                        exc_type = str(d.get("type", ""))
                        exc_msg = str(d.get("message", ""))[:80]
                    except json.JSONDecodeError:
                        pass
                rows.append([
                    str(t.task_id), t.grid, str(t.seed), str(t.n_failures),
                    statuses[t.task_id], exc_type, exc_msg,
                ])
            print(_format_table(
                ["task", "grid", "seed", "fails", "status", "exception", "message"], rows,
            ))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("campaign_dir", type=Path)
    p.add_argument("--show-failed", type=int, default=10,
                   help="How many recent failed tasks to list (0 = none)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    campaign_dir = args.campaign_dir.resolve()
    if not (campaign_dir / CAMPAIGN_LAYOUT["manifest"]).exists():
        raise SystemExit(f"Not a campaign dir (no manifest.jsonl): {campaign_dir}")
    report(campaign_dir, args.show_failed)


if __name__ == "__main__":
    main()
