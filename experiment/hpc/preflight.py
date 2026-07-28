"""Refuse to submit a campaign whose physics is infeasible before it starts.

``eval_full_v2_20260724-141520`` burned 350 tasks on a ``(simbench_lv_small,
pv_peak)`` pair whose simulation LP was infeasible at ``t=0`` — no failure
applied, no agent having written a setpoint. Nothing in the pipeline noticed:
the environment falls back to the unsolved net
(``RestorationEnvironmentBehavior._accept_or_keep``), the runner only errors
when *every* solve fails, and 218 of those tasks graded ``ok``.

A campaign has O(10) distinct ``(grid, scenario)`` pairs against O(1e4) tasks,
so one step-0 solve per pair costs minutes and covers the whole class.

Step-0 solvability alone is necessary but NOT sufficient, and eval_full_v2_
20260727 showed both gaps. ``simbench_lv_small`` passed this check while sitting
at **exactly 100.0 % of the LP's hard ampacity bound** under ``pv_peak`` — so
any reactive dispatch tipped it infeasible — and it carried a single-branch kill
switch, ``(37,36,0)``, whose failure left a CHP heat port pinning power into a
degree-1 stub. Neither is visible at t=0 with no failure and no control.

So this module reports two more things: the worst branch **loading margin**
(a number, not a boolean) and, opt-in, a **single-branch contingency scan**.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from experiment.hpc.config import TaskSpec
from experiment.scenarios import GRIDS

logger = logging.getLogger(__name__)

_MIN_DT_H = 1e-9

#: Worst-branch loading (percent) above which a pair is flagged as born past its
#: thermal design point even though it solves. Measured born loading across
#: eval_full_v2's grids: every grid except LV-S sits at 66-113 %, LV-S at 274 %
#: (600 kW PV behind a 160 kVA transformer) and 300 % under ``pv_peak``. 150 %
#: separates "heavily loaded" from "sized wrong".
_TIGHT_LOADING_PCT = 150.0

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
    #: Worst branch loading at t=0, in percent. ``None`` when the solve failed
    #: or monee exposed no loading (an unsolved net's Var defaults are phantoms).
    max_loading_pct: float | None = None
    #: Branch ids whose single failure makes the physics INFEASIBLE (not merely
    #: load-shedding). Empty unless the contingency scan ran.
    kill_switches: list[Any] = field(default_factory=list)
    n_contingencies_scanned: int = 0

    @property
    def label(self) -> str:
        kind = (self.scenario or {}).get("kind", "clean")
        extras = {
            k: v for k, v in (self.scenario or {}).items() if k not in ("kind",)
        }
        return f"{self.grid} / kind={kind} {extras or ''}".rstrip()

    @property
    def tight(self) -> bool:
        """Solves, but with no thermal headroom — see :data:`_TIGHT_LOADING_PCT`."""
        return (
            self.ok
            and self.max_loading_pct is not None
            and self.max_loading_pct >= _TIGHT_LOADING_PCT
        )

    @property
    def margin_detail(self) -> str:
        bits = []
        if self.max_loading_pct is not None:
            bits.append(f"worst branch loading {self.max_loading_pct:.1f}%")
        if self.n_contingencies_scanned:
            bits.append(
                f"{len(self.kill_switches)}/{self.n_contingencies_scanned} "
                "single-branch contingencies infeasible"
            )
            if self.kill_switches:
                bits.append(f"kill switches: {self.kill_switches}")
        return "; ".join(bits)


def _worst_loading_pct(net: Any) -> float | None:
    """Worst thermal loading over the solved net's branches, in percent.

    Delegates to the grading path's own basis handling so preflight and the
    campaign's ``constraint_compliance`` cannot disagree about what "loaded"
    means.
    """
    from experiment.eval.metrics import _branch_loading_percent

    worst: float | None = None
    for br in net.branches:
        if not getattr(br, "active", True):
            continue
        try:
            pct = _branch_loading_percent(br, net)
        except Exception:  # noqa: BLE001 — diagnostics must never break preflight
            continue
        if pct is not None and (worst is None or pct > worst):
            worst = pct
    return worst


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


def _build_net(task: TaskSpec) -> Any:
    # Imported here: runner pulls in the whole MAS stack, and plan.py is also
    # used in contexts (manifest inspection) that must not pay for it.
    from experiment.hpc.runner import _apply_scenario

    net = GRIDS[task.grid]()
    _apply_scenario(net, task, logging.getLogger("experiment.hpc.runner"))
    return net


def _solve_step0(net: Any, solve_time_limit_s: float) -> tuple[bool, str, list[str]]:
    """Solve one physics step on *net* in place. Returns (failed, detail, IIS)."""
    from mango_energy_environments.base.monee import create_physics_stepper

    capture = _IISCapture()
    gurobi_log = logging.getLogger("monee.solver.gurobipy")
    gurobi_log.addHandler(capture)
    try:
        result = create_physics_stepper(
            net, solve_time_limit_s=solve_time_limit_s
        ).step(_MIN_DT_H)
        failed = bool(getattr(result, "failed", False))
        detail = str(getattr(result, "error", "") or "")[:300]
    except Exception as exc:  # noqa: BLE001 — a raising solve is also a failure
        failed, detail = True, f"{type(exc).__name__}: {exc}"[:300]
    finally:
        gurobi_log.removeHandler(capture)
    return failed, detail, capture.members


def _annotate_iis(detail: str, members: list[str]) -> str:
    if not members:
        return detail
    sig = sorted({_normalise(m) for m in members})
    shown = ", ".join(sig[:_SIGNATURE_MEMBERS])
    if len(sig) > _SIGNATURE_MEMBERS:
        shown += f", … (+{len(sig) - _SIGNATURE_MEMBERS} more families)"
    return f"{detail}  IIS: {shown}"


def scan_branch_contingencies(
    task: TaskSpec,
    *,
    solve_time_limit_s: float = 120.0,
    limit: int | None = None,
) -> tuple[list[Any], int]:
    """Deactivate each branch in turn and report those that make the LP infeasible.

    A contingency that merely sheds load is the experiment working; one that
    makes the physics *infeasible* is a modelling defect — the LP should shed,
    not fail. Returns ``(kill_switch_ids, n_scanned)``.

    Cost is one solve per branch, so this is opt-in and ``limit`` caps the scan
    on large grids (simbench_mvlv has ~778 nodes). The net is rebuilt per
    contingency: a failed solve can leave monee's in-place state unusable.
    """
    probe = _build_net(task)
    branch_ids = [br.id for br in probe.branches if getattr(br, "active", True)]
    if limit is not None:
        branch_ids = branch_ids[:limit]
    del probe

    kill_switches: list[Any] = []
    for branch_id in branch_ids:
        net = _build_net(task)
        target = next((b for b in net.branches if b.id == branch_id), None)
        if target is None:
            continue
        target.active = False
        failed, _, _ = _solve_step0(net, solve_time_limit_s)
        if failed:
            kill_switches.append(
                list(branch_id) if isinstance(branch_id, tuple) else branch_id
            )
    return kill_switches, len(branch_ids)


def check_pair(
    task: TaskSpec,
    *,
    solve_time_limit_s: float = 120.0,
    scan_contingencies: bool = False,
    contingency_limit: int | None = None,
) -> PreflightResult:
    """Build ``task``'s grid, apply its scenario, and solve one physics step.

    Also records the worst branch loading, and — when *scan_contingencies* —
    every single-branch failure that makes the physics infeasible.
    """
    net = _build_net(task)
    failed, detail, members = _solve_step0(net, solve_time_limit_s)

    if failed:
        return PreflightResult(
            task.grid, task.scenario or {}, False, _annotate_iis(detail, members)
        )

    res = PreflightResult(
        task.grid,
        task.scenario or {},
        True,
        max_loading_pct=_worst_loading_pct(net),
    )
    del net
    if scan_contingencies:
        res.kill_switches, res.n_contingencies_scanned = scan_branch_contingencies(
            task, solve_time_limit_s=solve_time_limit_s, limit=contingency_limit
        )
    return res


def preflight_scenarios(
    tasks: list[TaskSpec],
    *,
    solve_time_limit_s: float = 120.0,
    scan_contingencies: bool = False,
    contingency_limit: int | None = None,
) -> list[PreflightResult]:
    """Solve step 0 for every distinct ``(grid, scenario)`` pair in *tasks*.

    With *scan_contingencies*, additionally sweep every single-branch failure
    per pair — one solve per branch, so O(1e2-1e3) solves for a full campaign.
    """
    pairs = _distinct_pairs(tasks)
    logger.info(
        "Preflight: solving step 0 for %d distinct (grid, scenario) pair(s) "
        "covering %d task(s)%s …",
        len(pairs),
        len(tasks),
        " + single-branch contingency scan" if scan_contingencies else "",
    )
    results: list[PreflightResult] = []
    for i, task in enumerate(pairs, 1):
        res = check_pair(
            task,
            solve_time_limit_s=solve_time_limit_s,
            scan_contingencies=scan_contingencies,
            contingency_limit=contingency_limit,
        )
        results.append(res)
        verdict = "ok" if res.ok else "INFEASIBLE"
        if res.tight or res.kill_switches:
            verdict = "ok (TIGHT)"
        logger.info("  [%d/%d] %s — %s", i, len(pairs), res.label, verdict)
        if not res.ok:
            logger.error("      %s", res.detail)
        elif res.margin_detail:
            log = logger.warning if (res.tight or res.kill_switches) else logger.info
            log("      %s", res.margin_detail)
    return results


def assert_preflight_clean(
    tasks: list[TaskSpec],
    *,
    solve_time_limit_s: float = 120.0,
    scan_contingencies: bool = False,
    contingency_limit: int | None = None,
) -> list[PreflightResult]:
    """:func:`preflight_scenarios`, raising on any infeasible pair.

    Tight-margin pairs and single-branch kill switches are reported but do NOT
    raise: a radial feeder can legitimately have a fragile branch, and only the
    campaign author can say whether it is a defect or the point of the
    experiment. They are the signal that was missing, not a submission gate.
    """
    results = preflight_scenarios(
        tasks,
        solve_time_limit_s=solve_time_limit_s,
        scan_contingencies=scan_contingencies,
        contingency_limit=contingency_limit,
    )
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
    risky = [r for r in results if r.tight or r.kill_switches]
    if risky:
        logger.warning(
            "Preflight: all %d pair(s) solve at t=0, but %d have no margin:\n%s",
            len(results),
            len(risky),
            "\n".join(f"  - {r.label}\n      {r.margin_detail}" for r in risky),
        )
    else:
        logger.info("Preflight: all %d pair(s) solve at t=0.", len(results))
    return results
