"""For each task that failed monotonic_progress, find the time of the
biggest drop in each sector's *_balance series and compare it to the
time of the first failure injection.

If most drops happen BEFORE any failure, they reflect initial dispatch
toward MAS equilibrium (not a restoration regression).

Usage:
    python scripts/analyze_monotonic_drops.py <campaign_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    campaign = Path(sys.argv[1])
    pre_failure = post_failure = 0
    drops = []
    for result_path in sorted((campaign / "tasks").glob("*/result.json")):
        try:
            res = json.loads(result_path.read_text())
        except Exception:
            continue
        mp = res.get("claims", {}).get("monotonic_progress", {})
        if mp.get("passed", True):
            continue
        task_dir = result_path.parent
        ts_path = task_dir / "timeseries.csv"
        if not ts_path.exists():
            continue
        failures_path = task_dir / "failures.json"
        first_failure_t = float("inf")
        if failures_path.exists():
            for f in json.loads(failures_path.read_text()):
                first_failure_t = min(first_failure_t, float(f.get("delay_s", float("inf"))))

        with open(ts_path) as fh:
            rows = list(csv.DictReader(fh))
        for sector_col in ("electrical_balance", "gas_balance", "heat_balance"):
            if sector_col not in rows[0]:
                continue
            biggest = (0.0, 0.0, None, None, None)
            prev = None
            for r in rows:
                try:
                    t = float(r["time_s"])
                    v = float(r[sector_col])
                except Exception:
                    continue
                if prev is not None:
                    drop = prev[1] - v
                    if drop > biggest[0]:
                        biggest = (drop, prev[0], t, prev[1], v)
                prev = (t, v)
            if biggest[0] > 0:
                t_drop = biggest[1]
                drops.append((task_dir.name, sector_col, t_drop, first_failure_t, biggest[0]))
                if t_drop < first_failure_t:
                    pre_failure += 1
                else:
                    post_failure += 1
    print(f"drops pre-failure: {pre_failure}")
    print(f"drops at-or-post-failure: {post_failure}")
    print()
    # bucketed by t<1s, 1-3s, 3+
    print("drops by time-of-occurrence:")
    buckets = Counter()
    for _, _, t, _, _ in drops:
        if t < 0.6:
            buckets["t<0.6"] += 1
        elif t < 1.5:
            buckets["0.6-1.5"] += 1
        elif t < 5:
            buckets["1.5-5"] += 1
        else:
            buckets["5+"] += 1
    for k, v in buckets.most_common():
        print(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
