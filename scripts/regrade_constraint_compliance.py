#!/usr/bin/env python
"""Re-grade ``constraint_compliance`` on a completed campaign, in place.

Rewrites only ``result.json['claims']['constraint_compliance']`` from each
task's persisted ``constraints_final.csv``. Everything else in the payload is
left byte-for-byte alone.

It deliberately does NOT call ``evaluate_task``: that recomputes all nine claims
from artefacts and would silently overwrite the oracle arm's payloads, which are
produced in memory during the solve and cannot be reconstructed from disk.

Guarded by ``experiment.eval.guards`` — refuses a run of record or any target on
a network share, and requires ``--i-have-a-copy``. ``--dry-run`` writes nothing
and is exempt from both, so the run of record can be inspected in place.

    python scripts/regrade_constraint_compliance.py \\
        --campaign-dir F:/scare_work/<campaign>_work --i-have-a-copy [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment.eval.claims import _check_constraint_compliance  # noqa: E402
from experiment.eval.guards import (  # noqa: E402
    ProtectedCampaignError,
    add_regrade_arguments,
    assert_regradable,
)

CLAIM = "constraint_compliance"


def regrade(campaign_dir: Path, *, dry_run: bool) -> dict[str, int]:
    tasks = campaign_dir / "tasks"
    stats = {
        "seen": 0,
        "no_csv": 0,
        "no_result": 0,
        "lp_enforced": 0,
        "changed": 0,
        "flipped": 0,
    }
    flips: list[tuple[str, bool, bool]] = []
    for task in sorted(tasks.iterdir()):
        if not task.is_dir():
            continue
        stats["seen"] += 1
        result_path = task / "result.json"
        csv_path = task / "constraints_final.csv"
        if not result_path.exists():
            stats["no_result"] += 1
            continue
        if not csv_path.exists():
            stats["no_csv"] += 1
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        claims = payload.get("claims")
        if not isinstance(claims, dict) or CLAIM not in claims:
            continue
        before = claims[CLAIM]
        # The oracle's payload is emitted by the solve with an ``enforced_at_lp``
        # provenance marker and is not derivable from constraints_final.csv.
        # Re-deriving it would erase that marker on 1435 tasks — and the marker
        # is the evidence that oracle compliance is feasible-by-construction
        # rather than measured.
        detail = before.get("detail")
        if isinstance(detail, dict) and "enforced_at_lp" in detail:
            stats["lp_enforced"] += 1
            continue
        after = _check_constraint_compliance(csv_path)
        if after == before:
            continue
        stats["changed"] += 1
        was, now = bool(before.get("passed")), bool(after.get("passed"))
        if was != now:
            stats["flipped"] += 1
            flips.append((task.name, was, now))
        if not dry_run:
            claims[CLAIM] = after
            tmp = result_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, result_path)  # atomic; never a half-written payload
    for name, was, now in flips:
        print(f"  FLIP {name}: passed {was} -> {now}")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", required=True, type=Path)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    add_regrade_arguments(p)
    args = p.parse_args()
    try:
        campaign = assert_regradable(
            args.campaign_dir,
            acknowledged=args.acknowledged,
            read_only=args.dry_run,
        )
    except ProtectedCampaignError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    mode = "DRY RUN" if args.dry_run else "WRITING"
    print(f"[{mode}] {campaign}")
    stats = regrade(campaign, dry_run=args.dry_run)
    for k, v in stats.items():
        print(f"  {k:12} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
