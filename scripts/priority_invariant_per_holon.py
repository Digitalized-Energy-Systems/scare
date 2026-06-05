"""Re-evaluate the priority_invariant claim using per-holon aggregation
(SCARE's design intent) instead of per-(sector, component), which pools across
holons in the same connected component.

Reads each holon leader's last ``holon_priority_allocation`` event and checks
that per-tier service fractions inside the holon are non-increasing in tier.
Reports the per-holon pass rate alongside the per-component rate.

Usage:
    python scripts/priority_invariant_per_holon.py <campaign_dir>
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


_FRAC_PATTERN = re.compile(
    r"'([a-z]+):tier(\d+)'\s*:\s*\{[^}]*'service_frac'\s*:\s*([0-9.e+-]+)"
)


def _per_holon_allocations(task_dir: Path) -> dict[str, dict[str, dict[int, float]]]:
    """Return ``{leader_aid: {sector: {tier: fraction}}}`` using the
    *last* ``holon_priority_allocation`` per leader+sector.
    """
    out: dict[str, dict[str, dict[int, float]]] = {}
    events_path = task_dir / "events.csv"
    if not events_path.exists():
        return out
    for r in csv.DictReader(open(events_path)):
        if r["kind"] != "holon_priority_allocation":
            continue
        leader = r["aid"]
        per_sec: dict[str, dict[int, float]] = {}
        for m in _FRAC_PATTERN.finditer(r["detail"]):
            sec, tier, frac = m.group(1), int(m.group(2)), float(m.group(3))
            per_sec.setdefault(sec, {})[tier] = frac
        if per_sec:
            out[leader] = per_sec
    return out


def _holon_inversions(allocs: dict[str, dict[str, dict[int, float]]]) -> int:
    """Count per-holon inversions across all leaders + sectors."""
    n_inv = 0
    for _, per_sec in allocs.items():
        for _, fracs in per_sec.items():
            tiers = sorted(fracs)
            for i in range(1, len(tiers)):
                t_prev, t_cur = tiers[i - 1], tiers[i]
                if fracs[t_cur] > fracs[t_prev] + 1e-3:
                    n_inv += 1
    return n_inv


def main():
    campaign = Path(sys.argv[1])
    n_tasks = 0
    n_per_comp_pass = 0
    n_per_holon_pass = 0
    n_per_holon_inv = 0
    n_per_comp_inv = 0
    for task_dir in sorted((campaign / "tasks").glob("*")):
        if not task_dir.is_dir():
            continue
        result_path = task_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            r = json.loads(result_path.read_text())
        except Exception:
            continue
        pi = r.get("claims", {}).get("priority_invariant")
        if pi is None:
            continue
        n_tasks += 1
        if pi.get("passed"):
            n_per_comp_pass += 1
        per_comp_inv = int(pi.get("detail", {}).get("n_inversions", 0))
        n_per_comp_inv += per_comp_inv

        allocs = _per_holon_allocations(task_dir)
        holon_inv = _holon_inversions(allocs)
        n_per_holon_inv += holon_inv
        if holon_inv == 0:
            n_per_holon_pass += 1
    print(f"# {campaign.name}  n_tasks={n_tasks}\n")
    print(f"per-COMPONENT pass (current claim): {n_per_comp_pass}/{n_tasks} "
          f"({100*n_per_comp_pass/n_tasks:.1f}%)")
    print(f"  total inversions (per-component): {n_per_comp_inv}")
    print(f"per-HOLON     pass (SCARE design):  {n_per_holon_pass}/{n_tasks} "
          f"({100*n_per_holon_pass/n_tasks:.1f}%)")
    print(f"  total inversions (per-holon):    {n_per_holon_inv}")


if __name__ == "__main__":
    main()
