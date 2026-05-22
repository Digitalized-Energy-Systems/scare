"""Aggregate claim-failure details across an evaluated campaign.

Usage:
    python scripts/analyze_claims.py <campaign_dir>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    campaign = Path(sys.argv[1])
    tasks = sorted((campaign / "tasks").glob("*/result.json"))
    print(f"# {campaign.name}  (n_tasks_with_result={len(tasks)})\n")

    n_with_claims = 0
    pass_count: Counter = Counter()
    fail_count: Counter = Counter()

    mp_drops_per_sec: dict[str, list[float]] = defaultdict(list)
    pi_inversion_counts: list[int] = []
    pi_examples_by_sector: dict[str, list[dict]] = defaultdict(list)
    pi_components_skipped: list[int] = []
    pi_components_checked: list[int] = []

    per_experiment_fails: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for result_path in tasks:
        try:
            res = json.loads(result_path.read_text())
        except Exception:
            continue
        claims = res.get("claims")
        if not claims:
            continue
        n_with_claims += 1
        task_dir = result_path.parent
        cfg = json.loads((task_dir / "config.json").read_text())
        exp = cfg.get("experiment", "?")

        for cname, c in claims.items():
            ok = bool(c.get("passed"))
            (pass_count if ok else fail_count)[cname] += 1
            if not ok:
                per_experiment_fails[exp][cname] += 1
            detail = c.get("detail") or {}
            if cname == "monotonic_progress":
                drops = detail.get("per_sector_relative_drop") or {}
                for sec, d in drops.items():
                    if not ok:
                        mp_drops_per_sec[sec].append(float(d))
            if cname == "priority_invariant":
                pi_inversion_counts.append(int(detail.get("n_inversions") or 0))
                pi_components_checked.append(int(detail.get("n_components_checked") or 0))
                pi_components_skipped.append(int(detail.get("n_components_skipped_no_deficit") or 0))
                for inv in detail.get("inversions", []) or []:
                    pi_examples_by_sector[inv.get("sector", "?")].append(inv)

    print(f"tasks with claims block: {n_with_claims}\n")
    for cname in ("diary_invariant", "monotonic_progress", "priority_invariant"):
        p, f = pass_count[cname], fail_count[cname]
        tot = p + f
        rate = (p / tot * 100) if tot else 0.0
        print(f"  {cname:24s}  pass={p:4d}  fail={f:4d}  ({rate:5.1f}%)")

    print()
    print("## monotonic_progress relative drops (across FAILED tasks)")
    print("(tol=0.05; drop is largest mid-restoration drop in sector's *_balance series)")
    for sec, ds in mp_drops_per_sec.items():
        if not ds:
            continue
        ds_over = [d for d in ds if d > 0.05]
        print(f"  {sec:12s}  n={len(ds):3d}  median={median(ds):.3f}  "
              f"mean={mean(ds):.3f}  max={max(ds):.3f}  n_over_tol={len(ds_over)}")

    print()
    print("## priority_invariant")
    print(f"  components checked  median={median(pi_components_checked or [0])}  "
          f"sum={sum(pi_components_checked)}")
    print(f"  components skipped (no deficit)  median={median(pi_components_skipped or [0])}  "
          f"sum={sum(pi_components_skipped)}")
    print(f"  inversion counts  median={median(pi_inversion_counts or [0])}  "
          f"max={max(pi_inversion_counts or [0])}  sum={sum(pi_inversion_counts)}")
    print()
    print("## priority_invariant: example inversions by sector (first 3 each)")
    for sec, invs in pi_examples_by_sector.items():
        print(f"  sector={sec}  total_examples_seen={len(invs)}")
        for inv in invs[:3]:
            print(f"    tier_{inv['tier_prev']}->{inv['tier_cur']}: "
                  f"frac {inv['frac_prev']:.4f} -> {inv['frac_cur']:.4f}  "
                  f"(component={inv.get('component')})")

    print()
    print("## Per-experiment claim failures")
    print(f"  {'experiment':40s}  {'mono':>5s}  {'prio':>5s}")
    for exp in sorted(per_experiment_fails):
        d = per_experiment_fails[exp]
        print(f"  {exp:40s}  {d['monotonic_progress']:>5d}  {d['priority_invariant']:>5d}")


if __name__ == "__main__":
    main()
