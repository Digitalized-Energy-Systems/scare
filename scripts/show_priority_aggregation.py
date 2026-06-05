"""Side-by-side: claim's per-component aggregation vs SCARE's per-holon priority
decisions. Shows when an apparent component-level "priority inversion" is really
two holons each making priority-correct decisions internally.

Usage:
    python scripts/show_priority_aggregation.py <task_dir>
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def _parse_holon_alloc(detail: str) -> dict[str, dict[int, float]]:
    """Parse the ``holon_priority_allocation`` detail string into
    ``{sector: {tier: service_fraction}}``.

    Detail looks like:
    ``{'heat:tier4': {'T': 0.012, 'weight': 128, 'sum_x': 0.012, 'service_frac': 0.457}, ...}``
    """
    out: dict[str, dict[int, float]] = defaultdict(dict)
    pat = re.compile(
        r"'([a-z]+):tier(\d+)'\s*:\s*\{[^}]*'service_frac'\s*:\s*([0-9.e+-]+)"
    )
    for m in pat.finditer(detail):
        sec, tier, frac = m.group(1), int(m.group(2)), float(m.group(3))
        out[sec][tier] = frac
    return out


def main():
    task_dir = Path(sys.argv[1])
    sector = sys.argv[2] if len(sys.argv) > 2 else "heat"

    # Per-component view (what the claim sees)
    print(f"=== {task_dir.name}  sector={sector}  ===\n")
    print("## Per-component aggregate (priority_invariant claim's view)\n")
    by_comp_tier: dict[tuple[str, int], dict[str, float]] = {}
    for r in csv.DictReader(open(task_dir / "served_by_load.csv")):
        if r["sector"] != sector or r["disconnected"] == "1":
            continue
        k = (r["component"], int(r["tier"]))
        e = by_comp_tier.setdefault(k, {"demand": 0.0, "served": 0.0, "n": 0})
        e["demand"] += float(r["demand"])
        e["served"] += float(r["served"])
        e["n"] += 1
    by_comp: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for (comp, tier), e in by_comp_tier.items():
        by_comp[comp][tier] = e
    for comp, tiermap in by_comp.items():
        print(f"  component={comp}")
        prev_frac = None
        prev_tier = None
        for tier in sorted(tiermap):
            e = tiermap[tier]
            frac = e["served"] / e["demand"] if e["demand"] > 0 else 1.0
            inv = ""
            if prev_frac is not None and frac > prev_frac + 1e-3:
                inv = f"  <- INVERSION vs tier {prev_tier} ({prev_frac:.3f})"
            print(
                f"    tier {tier:2d}: n={e['n']:3d}  "
                f"demand={e['demand']:.4f}  served={e['served']:.4f}  "
                f"frac={frac:.4f}{inv}"
            )
            prev_frac, prev_tier = frac, tier

    # Per-holon view (what SCARE designed for)
    print()
    print("## Per-holon allocations (SCARE's design — priority enforced PER holon)\n")
    leaders_seen: set[str] = set()
    rows_holon_alloc = []
    for r in csv.DictReader(open(task_dir / "events.csv")):
        if r["kind"] != "holon_priority_allocation" or r["sector"] != sector:
            continue
        rows_holon_alloc.append(r)
    # Last allocation per leader is the steady-state allocation
    latest_by_leader: dict[str, dict[str, dict[int, float]]] = {}
    for r in rows_holon_alloc:
        latest_by_leader[r["aid"]] = _parse_holon_alloc(r["detail"])
    print(f"  ({len(latest_by_leader)} holons in {sector} sector)\n")
    any_inversion = False
    for leader in sorted(latest_by_leader):
        by_sec = latest_by_leader[leader]
        fracs = by_sec.get(sector, {})
        tiers = sorted(fracs)
        if len(tiers) < 2:
            continue
        prev_frac = None
        prev_tier = None
        violations = []
        for tier in tiers:
            frac = fracs[tier]
            if prev_frac is not None and frac > prev_frac + 1e-3:
                violations.append((prev_tier, prev_frac, tier, frac))
                any_inversion = True
            prev_frac, prev_tier = frac, tier
        fracs_str = ", ".join(f"t{t}={f:.3f}" for t, f in sorted(fracs.items()))
        marker = "  <- INVERSION" if violations else "  OK monotonic"
        print(f"  {leader:14s}  {{{fracs_str}}}{marker}")
        for tp, fp, tc, fc in violations:
            print(f"    inversion: tier {tp} ({fp:.3f}) < tier {tc} ({fc:.3f})")
    print()
    if not any_inversion:
        print("  => EVERY holon makes priority-correct decisions internally.")
        print("  => The per-component inversion is a cross-holon spatial accident,")
        print("    not a SCARE priority bug.")


if __name__ == "__main__":
    main()
