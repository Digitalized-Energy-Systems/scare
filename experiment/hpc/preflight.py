"""Refuse to submit a campaign whose physics is infeasible before it starts.

``eval_full_v2_20260724-141520`` burned 350 tasks on a ``(simbench_lv_small,
pv_peak)`` pair whose simulation LP was infeasible at ``t=0`` — no failure
applied, no agent having written a setpoint. Nothing in the pipeline noticed:
the environment falls back to the unsolved net
(``RestorationEnvironmentBehavior._accept_or_keep``), the runner only errors
when *every* solve fails, and 218 of those tasks graded ``ok``.

A campaign has O(10) distinct ``(grid, scenario)`` pairs against O(1e4) tasks,
so one step-0 solve per pair costs minutes and covers the whole class. This
checks solvability only — a pair that solves can still be a bad experiment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from experiment.hpc.config import TaskSpec
from experiment.scenarios import GRIDS

logger = logging.getLogger(__name__)

_MIN_DT_H = 1e-9

#: IIS members named in the failure summary. The campaign's signatures are a
#: handful of families wide; the cap only bounds the pathological case.
_SIGNATURE_MEMBERS = 12

_BOUND_RE = re.compile(r"^\s{4}(\S+ \[[A-Z/]+\])\s*$")
_CONSTR_RE = re.compile(r"^\s{4}(\S+)\s*$")


def _normalise(member: str) -> str:
    """Collapse component ids so one physical conflict is one signature line."""
    member = re.sub(r"^branch_\d+_\d+_\d+", "branch_*", member)
    member = re.sub(r"^node_\d+", "node_*", member)
    member = re.sub(r"^child_\d+", "child_*", member)
    return re.sub(r"^R\d+$", "R#", member)


class _IISCapture(logging.Handler):
    """Collect the IIS block monee logs on an infeasible gurobipy solve."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.members: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001 — diagnostics must never break a solve
            return
        if "Irreducible Inconsistent Subsystem" not in text:
            return
        for line in text.splitlines():
            if "in IIS (" in line or "Irreducible" in line:
                continue
            m = _BOUND_RE.match(line) or _CONSTR_RE.match(line)
            if m:
                self.members.append(m.group(1))


@dataclass
class PreflightResult:
    grid: str
    scenario: dict[str, Any]
    ok: bool
    detail: str = ""

    @property
    def label(self) -> str:
        kind = (self.scenario or {}).get("kind", "clean")
        extras = {
            k: v for k, v in (self.scenario or {}).items() if k not in ("kind",)
        }
        return f"{self.grid} / kind={kind} {extras or ''}".rstrip()


def _distinct_pairs(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """One representative task per ``(grid, scenario)``, in manifest order.

    The oracle variant is skipped: it solves its own min-load-shedding LP with
    curtailable loads, so it is not the arm this guard protects.
    """
    seen: set[tuple[str, str]] = set()
    out: list[TaskSpec] = []
    for t in tasks:
        if t.variant == "oracle":
            continue
        key = (t.grid, repr(sorted((t.scenario or {}).items())))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def check_pair(task: TaskSpec, *, solve_time_limit_s: float = 120.0) -> PreflightResult:
    """Build ``task``'s grid, apply its scenario, and solve one physics step."""
    from mango_energy_environments.base.monee import create_physics_stepper

    # Imported here: runner pulls in the whole MAS stack, and plan.py is also
    # used in contexts (manifest inspection) that must not pay for it.
    from experiment.hpc.runner import _apply_scenario

    quiet = logging.getLogger("experiment.hpc.runner")
    net = GRIDS[task.grid]()
    _apply_scenario(net, task, quiet)

    capture = _IISCapture()
    gurobi_log = logging.getLogger("monee.solver.gurobipy")
    gurobi_log.addHandler(capture)
    try:
        result = create_physics_stepper(net, solve_time_limit_s=solve_time_limit_s).step(
            _MIN_DT_H
        )
        failed = bool(getattr(result, "failed", False))
        detail = str(getattr(result, "error", "") or "")[:300]
    except Exception as exc:  # noqa: BLE001 — a raising solve is also a failure
        failed, detail = True, f"{type(exc).__name__}: {exc}"[:300]
    finally:
        gurobi_log.removeHandler(capture)

    if not failed:
        return PreflightResult(task.grid, task.scenario or {}, True)
    if capture.members:
        sig = sorted({_normalise(m) for m in capture.members})
        shown = ", ".join(sig[:_SIGNATURE_MEMBERS])
        if len(sig) > _SIGNATURE_MEMBERS:
            shown += f", … (+{len(sig) - _SIGNATURE_MEMBERS} more families)"
        detail = f"{detail}  IIS: {shown}"
    return PreflightResult(task.grid, task.scenario or {}, False, detail)


def preflight_scenarios(
    tasks: list[TaskSpec], *, solve_time_limit_s: float = 120.0
) -> list[PreflightResult]:
    """Solve step 0 for every distinct ``(grid, scenario)`` pair in *tasks*."""
    pairs = _distinct_pairs(tasks)
    logger.info(
        "Preflight: solving step 0 for %d distinct (grid, scenario) pair(s) "
        "covering %d task(s) …",
        len(pairs),
        len(tasks),
    )
    results: list[PreflightResult] = []
    for i, task in enumerate(pairs, 1):
        res = check_pair(task, solve_time_limit_s=solve_time_limit_s)
        results.append(res)
        logger.info("  [%d/%d] %s — %s", i, len(pairs), res.label, "ok" if res.ok else "INFEASIBLE")
        if not res.ok:
            logger.error("      %s", res.detail)
    return results


def assert_preflight_clean(
    tasks: list[TaskSpec], *, solve_time_limit_s: float = 120.0
) -> list[PreflightResult]:
    """:func:`preflight_scenarios`, raising on any infeasible pair."""
    results = preflight_scenarios(tasks, solve_time_limit_s=solve_time_limit_s)
    bad = [r for r in results if not r.ok]
    if bad:
        lines = "\n".join(f"  - {r.label}\n      {r.detail}" for r in bad)
        raise SystemExit(
            f"Preflight failed: {len(bad)} of {len(results)} (grid, scenario) "
            f"pair(s) are infeasible at t=0, before any failure or agent "
            f"action:\n{lines}\n"
            "Fix the scenario/grid, or pass --no-preflight to submit anyway "
            "(the affected tasks will feed the unsolved net to their observers)."
        )
    logger.info("Preflight: all %d pair(s) solve at t=0.", len(results))
    return results
