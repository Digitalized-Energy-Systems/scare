"""Compare two priority_dispatch_probe campaigns and print a diff.

Usage:
    python scripts/compare_priority_probes.py BASELINE_DIR NEW_DIR

For each task pair (matched by (experiment, ablation)), reports status/duration,
solver infeasibilities, priority_invariant pass/fail + inversion count, gossip
saturated-vs-free ratio, diagnostic event counts, and per-tier served fraction.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _load_manifest(campaign_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(campaign_dir / "manifest.jsonl") as fh:
        for line in fh:
            spec = json.loads(line)
            out[int(spec["task_id"])] = spec
    return out


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _count_gossip_saturation(run_log: Path) -> tuple[int, int]:
    """Return (saturated_count, free_count) by scanning gossip ledger lines in
    run.log, counting the ``True)`` / ``False)`` literals in stringified tuples.
    """
    if not run_log.exists():
        return 0, 0
    sat, free = 0, 0
    pat_t = re.compile(r", True\)")
    pat_f = re.compile(r", False\)")
    with run_log.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            sat += len(pat_t.findall(line))
            free += len(pat_f.findall(line))
    return sat, free


def _count_events(events_csv: Path) -> dict[str, int]:
    if not events_csv.exists():
        return {}
    counts: dict[str, int] = defaultdict(int)
    with events_csv.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            kind = row.get("kind", "")
            counts[kind] += 1
    return dict(counts)


def _load_served(served_csv: Path) -> dict[tuple[str, int], float]:
    if not served_csv.exists():
        return {}
    out: dict[tuple[str, int], float] = {}
    with served_csv.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                out[(row["sector"], int(row["tier"]))] = float(row["fraction"])
            except (KeyError, ValueError):
                continue
    return out


def _ablation_key(spec: dict) -> str:
    abl = spec.get("ablation") or {}
    if not abl:
        return "default"
    return ",".join(f"{k}={abl[k]}" for k in sorted(abl))


def summarize_task(campaign_dir: Path, task_id: int) -> dict:
    td = campaign_dir / "tasks" / f"{task_id:06d}"
    status = _load_json(td / "status.json") or {}
    result = _load_json(td / "result.json") or {}
    claims = (result.get("claims") or {})
    sat, free = _count_gossip_saturation(td / "run.log")
    events = _count_events(td / "events.csv")
    served = _load_served(td / "served.csv")
    pinv = claims.get("priority_invariant", {})
    return {
        "status": status.get("status", "missing"),
        "duration_s": status.get("duration_s"),
        "solver_inf": status.get("solver_infeasibilities", 0),
        "priority_invariant_passed": pinv.get("passed"),
        "n_inversions": (pinv.get("detail") or {}).get("n_inversions", 0),
        "gossip_saturated": sat,
        "gossip_free": free,
        "events": events,
        "served": served,
    }


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: compare_priority_probes.py BASELINE_DIR NEW_DIR")
    base = Path(sys.argv[1])
    new = Path(sys.argv[2])
    base_man = _load_manifest(base)
    new_man = _load_manifest(new)
    # Index by (experiment, ablation_key) to compare matched tasks.
    base_idx = {(s["experiment"], _ablation_key(s)): t for t, s in base_man.items()}
    new_idx = {(s["experiment"], _ablation_key(s)): t for t, s in new_man.items()}
    keys = sorted(set(base_idx) | set(new_idx))

    print(f"{'experiment / ablation':<60} {'baseline':>40}   {'after fixes':>40}")
    print("-" * 150)
    for key in keys:
        exp, abl = key
        b_t = base_idx.get(key)
        n_t = new_idx.get(key)
        b = summarize_task(base, b_t) if b_t is not None else None
        n = summarize_task(new, n_t) if n_t is not None else None

        def _fmt(s):
            if s is None:
                return "  (missing)"
            ratio = (
                s["gossip_saturated"] / max(1, s["gossip_saturated"] + s["gossip_free"])
            )
            return (
                f"{s['status'][:3]} "
                f"dur={s['duration_s'] or 0:5.0f}s "
                f"inf={s['solver_inf']:>3} "
                f"sat={ratio:.0%} "
                f"inv={s['n_inversions']:>2}"
            )

        print(f"{exp + ' / ' + abl:<60} {_fmt(b):>40}   {_fmt(n):>40}")

    # Roll up diagnostic-event counts across the new campaign only.
    print("\nNew diagnostic events on the post-fix campaign:")
    for kind in (
        "regulate_on_stale_obs",
        "regulate_suppressed_by_cooldown",
        "priority_default_fallback",
    ):
        n_total = sum(
            summarize_task(new, t)["events"].get(kind, 0) for t in new_man
        )
        print(f"  {kind:<40} {n_total}")

    # Detailed served-by-tier diff for matching tasks.
    print("\nServed-fraction-by-tier diff (heat sector only):")
    for key in keys:
        exp, abl = key
        b_t = base_idx.get(key)
        n_t = new_idx.get(key)
        if b_t is None or n_t is None:
            continue
        b = summarize_task(base, b_t)
        n = summarize_task(new, n_t)
        print(f"\n  {exp} / {abl}:")
        tiers = sorted({t for (sec, t) in (b["served"].keys() | n["served"].keys()) if sec == "heat"})
        for t in tiers:
            bv = b["served"].get(("heat", t))
            nv = n["served"].get(("heat", t))
            bv_s = "  -    " if bv is None else f"{bv:.3f}  "
            nv_s = "  -    " if nv is None else f"{nv:.3f}  "
            print(f"    tier {t:>2}: baseline={bv_s} after={nv_s}")


if __name__ == "__main__":
    main()
