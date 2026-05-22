"""Re-evaluate claims on an existing campaign without re-running tasks.

Useful when the claim definition changes (e.g. tighter filters) — lets
you measure the new pass rate against the same raw artefacts.

Usage:
    python scripts/reeval_claims.py <campaign_dir>
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiment.eval.claims import evaluate_task


def main():
    campaign = Path(sys.argv[1])
    pass_count: Counter = Counter()
    fail_count: Counter = Counter()
    n_with_artefacts = 0
    for task_dir in sorted((campaign / "tasks").glob("*")):
        if not task_dir.is_dir():
            continue
        if not (task_dir / "timeseries.csv").exists() and not (task_dir / "diary.csv").exists():
            continue
        n_with_artefacts += 1
        claims = evaluate_task(task_dir)
        for cname, c in claims.items():
            if c.get("passed"):
                pass_count[cname] += 1
            else:
                fail_count[cname] += 1
    print(f"# {campaign.name}  (n_tasks_with_artefacts={n_with_artefacts})")
    print()
    for cname in ("diary_invariant", "monotonic_progress", "priority_invariant"):
        p, f = pass_count[cname], fail_count[cname]
        tot = p + f
        rate = (p / tot * 100) if tot else 0.0
        print(f"  {cname:24s}  pass={p:4d}  fail={f:4d}  ({rate:5.1f}%)")


if __name__ == "__main__":
    main()
