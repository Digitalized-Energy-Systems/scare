"""Replay the LP that goes infeasible in priority_dispatch_probe at sim_t=0.1.

Reads the captured infeasibility_snapshot.json from a campaign task, rebuilds
the network the same way the runner does (line_stress + slack_budget),
deactivates the same branches, applies the same regulation factors, and
re-runs energyflow.  Then bisects: failure-only, regulations-only, both.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logging.getLogger("pyomo").setLevel(logging.WARNING)
log = logging.getLogger("repro")


def build_net(scenario: dict):
    from experiment.restoration import (
        GRIDS,
        apply_line_stress,
        apply_slack_budget,
    )

    net = GRIDS["simbench_lv"]()
    kind = scenario.get("kind")
    if kind == "line_stress":
        kwargs = {k: scenario[k] for k in ("load_scale", "ampacity_scale") if k in scenario}
        apply_line_stress(net, **kwargs)
    pct = scenario.get("slack_budget_pct")
    if pct is not None:
        apply_slack_budget(net, float(pct))
    return net


def deactivate_branches(net, branch_ids):
    """Match by id tuple (or list-of-3 from JSON)."""
    targets = {tuple(b) if isinstance(b, list) else b for b in branch_ids}
    hit = 0
    for br in net.branches:
        bid = tuple(br.id) if isinstance(br.id, tuple) else br.id
        if bid in targets:
            br.active = False
            hit += 1
    log.info("deactivated %d branches (targets=%s)", hit, list(targets))
    return hit


def apply_regulations(net, regs: dict[str, float]):
    """regs keyed by 'child-{id}'."""
    aid_to_factor = {}
    for k, v in regs.items():
        if k.startswith("child-"):
            aid_to_factor[k[len("child-"):]] = float(v)
    hit = 0
    for child in net.childs:
        cid = str(child.id)
        if cid in aid_to_factor:
            child.model.regulation = aid_to_factor[cid]
            hit += 1
    log.info("applied %d regulation factors", hit)
    return hit


def solve(net, label: str):
    from mango_energy_environments.base.monee import energyflow
    res = energyflow(net)
    ok = bool(getattr(res, "success", False))
    obj = getattr(res, "objective", None)
    tc = getattr(res, "termination_condition", "")
    log.info("[%s] success=%s  objective=%s  tc=%s", label, ok, obj, tc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", help="path to tasks/000000/")
    ap.add_argument("--bisect", action="store_true")
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    snap = json.loads((task_dir / "infeasibility_snapshot.json").read_text())
    cfg = json.loads((task_dir / "config.json").read_text())
    scenario = cfg["scenario"]

    inactive = snap["net"]["inactive_branches"]
    regs = snap["net"]["nondefault_regulations"]
    log.info("snapshot: %d inactive branches, %d non-default regulations at sim_t=%s",
             len(inactive), len(regs), snap["sim_time_s"])

    # baseline: pristine network, no failures, no regulations
    net = build_net(scenario)
    solve(net, "baseline (no failure, no regulations)")

    if not args.bisect:
        # full reproduction
        net = build_net(scenario)
        deactivate_branches(net, inactive)
        apply_regulations(net, regs)
        solve(net, "FULL repro (failure + regulations)")

        # repeat with the SCARE heat-Sink guard applied: simulate the
        # dispatcher dropping any regulation < 1 that targets a heat-side Sink.
        net = build_net(scenario)
        deactivate_branches(net, inactive)
        from monee.model.child import Sink

        guarded = {}
        skipped = 0
        for k, v in regs.items():
            if not k.startswith("child-"):
                guarded[k] = v
                continue
            try:
                cid = int(k[len("child-"):])
            except ValueError:
                guarded[k] = v
                continue
            try:
                child = net.child_by_id(cid)
                grid_name = str(getattr(net.node_by_id(child.node_id).grid, "name", "")).lower()
            except Exception:
                guarded[k] = v
                continue
            is_heat_sink = isinstance(child.model, Sink) and ("water" in grid_name or "heat" in grid_name)
            if is_heat_sink and v < 1.0 - 1e-3:
                skipped += 1
                continue
            guarded[k] = v
        log.info("guard: skipped %d heat-side Sink curtailments (kept %d/%d)",
                 skipped, len(guarded), len(regs))
        apply_regulations(net, guarded)
        solve(net, "GUARDED repro (failure + regulations, heat-Sinks unchanged)")
        return

    # bisection
    net = build_net(scenario)
    deactivate_branches(net, inactive)
    solve(net, "failure ONLY")

    net = build_net(scenario)
    apply_regulations(net, regs)
    solve(net, "regulations ONLY")

    # regulations rounded to {0,1}
    net = build_net(scenario)
    deactivate_branches(net, inactive)
    rounded = {k: (1.0 if v >= 0.5 else 0.0) for k, v in regs.items()}
    apply_regulations(net, rounded)
    solve(net, "failure + regulations rounded to {0,1}")

    # only zero-out the regulations that are < 1e-3 (the "numerical residue" ones)
    net = build_net(scenario)
    deactivate_branches(net, inactive)
    snapped = {k: (0.0 if v < 1e-3 else (1.0 if v > 1 - 1e-3 else v)) for k, v in regs.items()}
    apply_regulations(net, snapped)
    solve(net, "failure + regulations snapped near 0/1")

    # only non-zero regulations (skip the curtailments)
    net = build_net(scenario)
    deactivate_branches(net, inactive)
    keep_nonzero = {k: v for k, v in regs.items() if v > 0.5}
    apply_regulations(net, keep_nonzero)
    solve(net, "failure + only kept-on regulations (drop curtailments)")

    # only curtailments (skip the ones near 1.0)
    net = build_net(scenario)
    deactivate_branches(net, inactive)
    keep_curtail = {k: v for k, v in regs.items() if v < 0.5}
    apply_regulations(net, keep_curtail)
    solve(net, "failure + only curtailments (drop kept-ons)")

    # bisect curtailments by sector — need to know which child belongs to which
    # sector. Build a map by walking the network.
    net = build_net(scenario)
    sector_of_child: dict[str, str] = {}
    for child in net.childs:
        sec = ""
        try:
            grid_name = str(getattr(net.node_by_id(child.node_id).grid, "name", "")).lower()
        except Exception:
            grid_name = ""
        cls = child.model.__class__.__name__
        if "Power" in cls or "Load" in cls and "p_mw" in dir(child.model):
            if "gas" in grid_name:
                sec = "gas"
            elif "water" in grid_name or "heat" in grid_name:
                sec = "heat"
            else:
                sec = "electricity"
        elif "gas" in grid_name:
            sec = "gas"
        elif "water" in grid_name or "heat" in grid_name:
            sec = "heat"
        else:
            sec = "electricity"
        sector_of_child[str(child.id)] = sec

    def by_sector(reg_map: dict[str, float], sectors: set[str]) -> dict[str, float]:
        out = {}
        for k, v in reg_map.items():
            cid = k[len("child-"):] if k.startswith("child-") else k
            if sector_of_child.get(cid) in sectors:
                out[k] = v
        return out

    curtailments = {k: v for k, v in regs.items() if v < 0.5}

    for label, sectors in [
        ("electricity", {"electricity"}),
        ("gas", {"gas"}),
        ("heat", {"heat"}),
        ("electricity+gas", {"electricity", "gas"}),
        ("electricity+heat", {"electricity", "heat"}),
        ("gas+heat", {"gas", "heat"}),
    ]:
        net = build_net(scenario)
        sub = by_sector(curtailments, sectors)
        apply_regulations(net, sub)
        solve(net, f"curtailments only / sectors={label} (n={len(sub)})")

    # zoom in: heat curtailments by child class
    heat_curt = by_sector(curtailments, {"heat"})

    # break heat curtailments down by monee child class
    def classify_heat_child(cid: str) -> str:
        for child in net.childs:  # uses last `net` from loop; just for naming
            if str(child.id) == cid:
                return child.model.__class__.__name__
        return "Unknown"

    by_cls: dict[str, dict[str, float]] = {}
    for k, v in heat_curt.items():
        cid = k[len("child-"):]
        cls = classify_heat_child(cid)
        by_cls.setdefault(cls, {})[k] = v
    log.info("heat curtailments by class: %s", {c: len(d) for c, d in by_cls.items()})

    for cls, sub in by_cls.items():
        net = build_net(scenario)
        apply_regulations(net, sub)
        solve(net, f"heat curtailments / cls={cls} only (n={len(sub)})")

    # try: floor heat regulations at small positive value (0.01) instead of 0
    net = build_net(scenario)
    floored = {k: max(0.01, v) for k, v in heat_curt.items()}
    apply_regulations(net, floored)
    solve(net, f"heat curtailments floored at 0.01 (n={len(floored)})")

    # try: floor at 0.001
    net = build_net(scenario)
    floored = {k: max(0.001, v) for k, v in heat_curt.items()}
    apply_regulations(net, floored)
    solve(net, f"heat curtailments floored at 0.001 (n={len(floored)})")

    # Sink-only sub-bisect: one sink at a time, write the LP for the first that fails
    sinks = list(by_cls.get("Sink", {}).items())
    log.info("trying each Sink curtailment individually (n=%d)", len(sinks))
    first_failing_sink = None
    for k, v in sinks[:20]:
        net = build_net(scenario)
        apply_regulations(net, {k: v})
        from mango_energy_environments.base.monee import energyflow
        res = energyflow(net)
        ok = bool(getattr(res, "success", False))
        log.info("  single %s=%.4g  success=%s", k, v, ok)
        if not ok and first_failing_sink is None:
            first_failing_sink = (k, v)
    if first_failing_sink:
        log.info("first failing single sink: %s", first_failing_sink)
        # inspect that sink's neighborhood
        cid = first_failing_sink[0][len("child-"):]
        net = build_net(scenario)
        target = None
        for child in net.childs:
            if str(child.id) == cid:
                target = child
                break
        if target is not None:
            node_id = target.node_id
            log.info("  child=%s class=%s node=%s mass_flow=%s",
                     cid, target.model.__class__.__name__, node_id,
                     getattr(target.model, "mass_flow", "?"))
            neighbours = []
            for br in net.branches:
                bid = br.id if isinstance(br.id, tuple) else (br.id,)
                if node_id in bid:
                    neighbours.append((br.model.__class__.__name__, bid))
            log.info("  branches touching node %s: %s", node_id, neighbours[:10])
            sibs = [(c.model.__class__.__name__, c.id) for c in net.childs if c.node_id == node_id]
            log.info("  children on node %s: %s", node_id, sibs)


if __name__ == "__main__":
    sys.exit(main())
