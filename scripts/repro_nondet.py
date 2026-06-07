"""Step-by-step non-determinism probe for the restoration sim.

Builds a fixed (grid, seed, failures) scenario, then drives the world one
discrete step at a time, snapshotting "relevant state" after each step. Two
independent processes write their snapshot sequences to JSON; an external
differ finds the first step whose state diverges and classifies it.

Usage:
    python scripts/repro_nondet.py OUT.json [--steps N] [--seed S]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys

from experiment.scenarios.grids import GRIDS
from experiment.scenarios.priorities import assign_load_priorities

from scare.base.util import create_failures
from scare.scenario.restoration import create_restoration_scenario_world

# strip memory addresses so object reprs are comparable across processes
_ADDR = re.compile(r"0x[0-9a-fA-F]+")


def _r(x: float, nd: int = 9) -> float:
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def _safe_repr(o) -> str:
    return _ADDR.sub("0xADDR", repr(o))[:400]


def _num_values(values: dict) -> dict:
    """Numeric entries of a monee model.values dict, rounded + stable."""
    out = {}
    for k, v in values.items():
        try:
            out[str(k)] = _r(v)
        except Exception:
            pass
    return out


def monee_digest(net) -> dict:
    """Physical state: every child/branch/node numeric value, keyed by id."""
    childs = {}
    for c in net.childs:
        try:
            childs[str(c.id)] = _num_values(dict(c.model.values))
        except Exception:
            pass
    branches = {}
    for b in net.branches:
        try:
            branches[str(b.id)] = _num_values(dict(b.model.values))
        except Exception:
            pass
    nodes = {}
    for n in net.nodes:
        try:
            nodes[str(n.id)] = _num_values(dict(n.model.values))
        except Exception:
            pass
    return {"childs": childs, "branches": branches, "nodes": nodes}


def queue_digest(world) -> list:
    """Pending message queue, in stored (sorted) order."""
    out = []
    for delivery_time, seq, sent_time, content, meta in world._pending_messages:
        out.append(
            {
                "dt": _r(delivery_time),
                "seq": seq,
                "snd": meta.get("sender_id"),
                "rcv": meta.get("receiver_id"),
                "ctype": type(content).__name__,
                "crepr": _safe_repr(content),
            }
        )
    return out


def inbox_digest(world) -> dict:
    out = {}
    for aid, agent in world._agents.items():
        try:
            q = agent.inbox.qsize()
        except Exception:
            q = -1
        if q:
            out[aid] = q
    return out


def snapshot(world, net, result) -> dict:
    return {
        "t": _r(world.clock.time),
        "step": _r(result.step_size_s) if result else None,
        "delivered": result.messages_delivered if result else None,
        "msg_seq": world._msg_seq,
        "n_recorded": len(world.recorded_messages),
        "queue": queue_digest(world),
        "inbox": inbox_digest(world),
        "monee": monee_digest(net),
    }


def _patch_deterministic_uuid() -> None:
    """Replace uuid4 with a deterministic sequential generator everywhere
    it is referenced, to test whether uuid4 is the sole non-determinism
    source."""
    import uuid as _uuid

    counter = {"n": 0}

    def _det_uuid4():
        counter["n"] += 1
        return _uuid.UUID(int=counter["n"])

    import importlib
    import os as _os

    mode = _os.environ.get("DET_UUID_MODE", "all")
    others = {
        "holonic": "scare.community.holonic",
        "repartition": "scare.community.repartition",
        "restoration": "scare.scenario.restoration",
        "balance": "scare.service.balance.balance",
        "reconfiguration": "scare.service.reconfiguration",
    }
    if mode in ("all", "auctions", "others"):
        tokens = {mode}
    else:
        tokens = set(mode.split(","))

    # constraints.py uses `uuid.uuid4()` (module attr).
    if "all" in tokens or "auctions" in tokens:
        _uuid.uuid4 = _det_uuid4
    # Re-bind in modules that did `from uuid import uuid4`.
    for tok, modname in others.items():
        if "all" in tokens or "others" in tokens or tok in tokens:
            m = importlib.import_module(modname)
            if hasattr(m, "uuid4"):
                m.uuid4 = _det_uuid4


async def run(out_path: str, max_steps: int, seed: int, grid: str) -> None:
    from mango.simulation.world import step_simulation

    import os as _os

    if _os.environ.get("DET_UUID") == "1":
        _patch_deterministic_uuid()

    # Identical input across processes: fix the failure-sampling RNG.
    random.seed(seed)
    net = GRIDS[grid]()
    priorities = assign_load_priorities(net, seed=seed)
    failures = create_failures(net, "branch", num_failures=1, delay_s_max=1.0)

    world = create_restoration_scenario_world(
        net, priorities=priorities, simulation_duration_s=30.0
    )

    from mango_energy_environments import schedule_failure

    behavior = world.environment.behavior
    for f in failures:
        schedule_failure(behavior, world, f)

    snaps = []
    snaps.append({"failures": [_safe_repr(f) for f in failures]})

    async with world:
        snaps.append({"label": "post-init", **snapshot(world, net, None)})
        for i in range(max_steps):
            result = await step_simulation(world)
            if result is None:
                snaps.append({"label": f"step-{i}", "done": True})
                break
            snaps.append({"label": f"step-{i}", **snapshot(world, net, result)})

    with open(out_path, "w") as fh:
        json.dump(snaps, fh, indent=1)
    print(f"wrote {len(snaps)} snapshots to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--grid", default="simbench_lv_small")
    args = ap.parse_args()
    asyncio.run(run(args.out, args.steps, args.seed, args.grid))


if __name__ == "__main__":
    sys.exit(main())
