"""Centralised oracle baseline.

Runs monee's minimal-load-shedding LP on the post-failure network and returns
the optimal regulation factors plus the same outcome metrics the scare variant
produces, so the aggregator compares row-by-row. A single optimisation (no
agents, no time evolution) defining the achievable upper bound. Optimality gap
= ``(oracle.pw_served - scare.pw_served) / oracle.pw_served``.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from monee import run_energy_flow_optimization
from monee.model.child import (
    ExtHydrGrid,
    ExtPowerGrid,
    HeatGenerator,
    HeatLoad,
    PowerGenerator,
    PowerLoad,
    Sink,
    Source,
)
from monee.model.core import Var
from monee.model.formulation import (
    GAS_NONCONVEX_MIQCQP_FORMULATION,
    make_heat_convex_milp_formulation,
)
from monee.model.node import Bus
from monee.problem import (
    WEIGHT_DEMAND,
    create_min_load_shedding_problem,
)
from monee.simulation import Stepper
from monee.solver.gurobipy import GurobipySolver

from experiment.eval.claims import heat_priority_from_rows
from experiment.eval.metrics import (
    _is_deenergised,
    constraint_violations_final,
    cp_generation_breakdown,
    restoration_breakdown,
    served_breakdown,
    served_by_load,
)
from experiment.eval.results import (
    write_constraints_final_csv,
    write_served_by_load_csv,
    write_served_csv,
)
from experiment.scenarios import (
    GRIDS,
    apply_cold_day,
    apply_line_stress,
    apply_microgrid_islanding,
    apply_pv_peak,
    apply_slack_budget,
)

logger = logging.getLogger(__name__)

# Partition count for the oracle/shared ``apply_oracle_heat_linearisation``
# helper; the factory leaves DHS nonlinear, and the live-net path (runner.py)
# shares this exact helper so the two cannot drift.
_ORACLE_MCCORMICK_PARTITIONS = 16


def _apply_failures(monee_net: Any, failures: list[Any]) -> None:
    """Apply every failure to the network the same way the live simulation's
    ``apply_failures`` does, so the oracle solves the same post-failure topology.
    """
    for failure in failures:
        for branch_id in getattr(failure, "branch_ids", []) or []:
            try:
                branch = monee_net.branch_by_id(branch_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "oracle: could not find branch %s to disable: %s",
                    branch_id,
                    exc,
                )
                continue
            branch.active = False
            if hasattr(branch.model, "on_off"):
                branch.model.on_off = 0
        for node_id in getattr(failure, "node_ids", []) or []:
            try:
                monee_net.node_by_id(node_id).active = False
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "oracle: could not find node %s to disable: %s",
                    node_id,
                    exc,
                )
        custom = getattr(failure, "custom", None)
        if custom is not None:
            try:
                custom(monee_net)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "oracle: custom failure %s raised: %s",
                    getattr(failure, "custom_id", None),
                    exc,
                )


# Near-strict priority ladder mirroring SCARE's objective (tiers 2-4 steep
# 1e8/1e4/1), NOT PWSF's 8:4:2:1 which would trade tier-1 away and inflate the
# gap. Strict at the margin for any decreasing weights, size-independent. Span
# 1e6/ratio-100 keeps every tier above monee's auto_priority_floor (legacy
# 1e12/1e8/1e4/1 was priority-BLIND there). Also needs MIPGap<=1e-8 (default
# 1e-3 hid ~64k tier inversions). Validated in tests/eval/test_oracle_priority.py.
_ORACLE_TIER_WEIGHT: dict[int, float] = {1: 1e6, 2: 1e4, 3: 1e2, 4: 1.0}

# MIPGapAbs = half the objective cost of shedding a 1e-3 MW tier-4 load
# (= 0.5·WEIGHT_DEMAND·tier4·1e-3 = 0.5 at WEIGHT_DEMAND=1e3), so no tier
# decision fits inside the absolute gap; MIPGap covers the relative side.
# TimeLimit matches monee's default.
_ORACLE_MIN_LOAD_MW = 1e-3


def _oracle_threads() -> int:
    """Match Gurobi Threads to the slurm cgroup allocation.

    Default Threads=0 spawns one worker per DETECTED core (~32) while the task
    gets only cpus-per-task — scheduler thrash and per-thread node pools vs the
    16G cap. Off-slurm = 1 (the setting every preset was validated at).
    """
    try:
        return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    except ValueError:
        return 1


_ORACLE_GUROBI_PARAMS: dict[str, float] = {
    "MIPGap": 1e-9,
    "MIPGapAbs": 0.5 * WEIGHT_DEMAND * _ORACLE_TIER_WEIGHT[4] * _ORACLE_MIN_LOAD_MW,
    "TimeLimit": 300,
    "Threads": _oracle_threads(),
}

# reconfig/mvlv MIQCQP stalls at TimeLimit (eval_full_v2: reconfig 27/105
# sol_count=0 gap ~1, mvlv gap ~0.98) so "oracle" fell below optimum and SCARE
# "beat" it. Fix is longer budget only: on seed 200000002 (Threads=1)
# MIPFocus=1/NoRelHeurTime collapsed the BOUND (gap 0.81-0.90) while
# tie-restriction + baseline warm start reach 6e-4 at stock search settings.
_ORACLE_HARD_PRESET: dict[str, float] = {
    "TimeLimit": 900,
}
_ORACLE_HARD_GRIDS = frozenset({"simbench_lv_reconfig", "simbench_mvlv"})


def oracle_solver_for_task(
    grid: str, scenario: dict[str, Any] | None = None
) -> "_OracleGurobiSolver":
    """Solver with per-grid termination params; extended budget for grids (or
    microgrid/islanding scenarios) whose MIQCQP cannot converge inside the
    default 300 s."""
    hard = grid in _ORACLE_HARD_GRIDS or (scenario or {}).get("kind") == "microgrid"
    return _OracleGurobiSolver(params=dict(_ORACLE_HARD_PRESET) if hard else None)


class _OracleGurobiSolver(GurobipySolver):
    """GurobipySolver with the oracle's tight termination params that records
    per-solve termination metadata. monee's ``SolverResult`` carries no
    Status/MIPGap and the gurobipy model handle is local to ``solve()``, so
    ``_classify`` (called once per optimize) is the interception point.
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params={**_ORACLE_GUROBI_PARAMS, **(params or {})})
        self.solve_stats: dict[str, Any] = {}

    def _classify(self, gm, *, phase_label: str):
        # super() may re-solve to disambiguate INF_OR_UNBD; read stats after.
        result = super()._classify(gm, phase_label=phase_label)
        self.solve_stats = self._termination_stats(gm)
        return result

    def _termination_stats(self, gm) -> dict[str, Any]:
        def read(attr: str):
            # Gurobi raises on attrs unavailable for the model class / status
            # (e.g. MIPGap on a continuous model or without an incumbent).
            try:
                return getattr(gm, attr)
            except Exception:  # noqa: BLE001
                return None

        status = read("Status")
        sol_count = read("SolCount") or 0
        obj_val = read("ObjVal") if sol_count else None
        obj_bound = read("ObjBound")
        mip_gap = read("MIPGap") if sol_count else None
        gap_ok = True
        if mip_gap is not None and math.isfinite(mip_gap):
            abs_gap = float("inf")
            if (
                obj_val is not None
                and obj_bound is not None
                and math.isfinite(obj_val)
                and math.isfinite(obj_bound)
            ):
                abs_gap = abs(obj_val - obj_bound)
            gap_ok = mip_gap <= self._params.get(
                "MIPGap", 0.0
            ) or abs_gap <= self._params.get("MIPGapAbs", 0.0)
        return {
            "status": status,
            "sol_count": sol_count,
            "objective": obj_val,
            "obj_bound": obj_bound,
            "mip_gap": mip_gap,
            "runtime_s": read("Runtime"),
            "solve_optimal": bool(status == self._GRB.OPTIMAL and gap_ok),
        }


def _make_vm_bounds_hook(bounds_vm: tuple[float, float]):
    """Box the voltage Var the solver actually optimises.

    Under MISOCP that is ``vm_pu_squared`` (native box 0..2.25, here set to
    lo^2, hi^2), while ``vm_pu`` is a reporting Intermediate there so monee's
    ``bounds_vm`` no-ops. Also box ``vm_pu`` when it IS a Var (non-MISOCP).
    Mirrors ``_make_pressure_bounds_hook``.
    """
    lo, hi = bounds_vm

    def _apply_vm_bounds(network: Any) -> None:
        for component in network.all_components():
            model = component.model
            if type(model) is not Bus or not component.independent:
                continue
            v = getattr(model, "vm_pu", None)
            vsq = getattr(model, "vm_pu_squared", None)
            if type(v) is Var:
                v.min, v.max = lo, hi
            if type(vsq) is Var:
                vsq.min, vsq.max = lo * lo, hi * hi

    return _apply_vm_bounds


def _weight_for_load_factory(
    monee_net: Any,
    priorities: dict[str, int] | None,
    *,
    base_demand_weight: float,
    n_tiers: int = 4,
) -> Any | None:
    """Build a ``weight_for_load`` closure for monee's
    ``create_min_load_shedding_problem``.

    Returns ``model -> float | None``: ``base_demand_weight * tier_weight`` from
    the oracle ladder (``_ORACLE_TIER_WEIGHT``), or ``None`` to defer to monee's
    default (generators / slack / unmapped models). Returns ``None`` overall when
    no priorities are supplied, so monee uses its flat-weight behaviour.
    Out-of-range tiers clamp onto [1, 4].

    Tier is stamped as the attribute ``_scare_oracle_tier`` and resolved via
    ``getattr``, NOT an ``id(model)`` map: monee deep-copies the net at solve time
    (``solver/core.py`` ``input_network.copy()``), so an id-keyed map misses every
    copied model and renders the oracle priority-BLIND (verified: reversing the
    ladder didn't change dispatch). Attributes survive the copy.
    """
    if not priorities:
        return None
    stamped = 0
    for child in monee_net.childs:
        aid = f"child-{child.id}"
        tier = priorities.get(aid)
        if tier is None or int(tier) <= 0:
            continue
        child.model._scare_oracle_tier = int(tier)
        stamped += 1

    if not stamped:
        return None

    def _w(model: Any) -> float | None:
        tier = getattr(model, "_scare_oracle_tier", None)
        if tier is None:
            return None  # let monee use its default
        t = max(1, min(4, int(tier)))
        return float(base_demand_weight) * _ORACLE_TIER_WEIGHT[t]

    return _w


def _collect_slack_budgets(monee_net: Any) -> dict[str, float | None]:
    """Per-sector slack budget stamped by ``apply_slack_budget``, in monee's
    native units (MW for electricity, kg/s for gas/heat). The same value is
    stamped on every slack child of a sector, so read the first non-None.
    ``None`` when no budget was registered (e.g. heat, left unbounded).
    """
    out: dict[str, float | None] = {
        "electricity": None,
        "gas": None,
        "heat": None,
    }
    for child in monee_net.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid) and out["electricity"] is None:
            cap = getattr(m, "_scare_slack_budget_mw", None)
            if cap is not None:
                out["electricity"] = float(cap)
        elif isinstance(m, ExtHydrGrid):
            cap_kgs = getattr(m, "_scare_slack_budget_kgs", None)
            if cap_kgs is None:
                continue
            # Route by parent-node grid name: gas vs water (heat).
            try:
                grid_name = str(
                    getattr(monee_net.node_by_id(child.node_id).grid, "name", "")
                ).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name and out["gas"] is None:
                out["gas"] = float(cap_kgs)
            elif "water" in grid_name and out["heat"] is None:
                out["heat"] = float(cap_kgs)
    return out


def _heat_throughput_kgs(monee_net: Any) -> float:
    """Total heat-side boundary mass flow the heat slack may have to carry:
    the sum of |mass_flow_kgs| over Sinks/Sources on water junctions. Sink
    flows are non-regulatable, so this is a hard lower bound on a feasible
    heat-slack envelope."""
    total = 0.0
    for child in monee_net.childs:
        m = child.model
        if not isinstance(m, (Sink, Source)):
            continue
        try:
            grid_name = str(
                getattr(monee_net.node_by_id(child.node_id).grid, "name", "")
            ).lower()
            if "water" in grid_name:
                total += abs(float(getattr(m, "mass_flow_kgs", 0.0) or 0.0))
        except Exception:
            continue
    return total


def _slack_budget_summary(monee_net: Any) -> dict[str, Any]:
    """Realised slack draw vs operator budget on the post-LP network. Returns
    ``{aid: {budget, draw, violated}}`` for every slack child carrying an
    explicit budget; basis for the oracle's slack-budget-compliance claim.
    """
    out: dict[str, Any] = {}
    for child in monee_net.childs:
        m = child.model
        # After the solve, model attributes hold solved Var objects; ``.values``
        # resolves each to its numeric value.
        vals = m.values if hasattr(m, "values") else {}
        if isinstance(m, ExtPowerGrid):
            cap = getattr(m, "_scare_slack_budget_mw", None)
            if cap is None:
                continue
            try:
                draw = abs(float(vals.get("p_mw", 0.0) or 0.0))
            except (TypeError, ValueError):
                draw = 0.0
            out[f"child-{child.id}"] = {
                "sector": "electricity",
                "budget_mw": float(cap),
                "draw_mw": draw,
                "violated": draw > float(cap) * 1.001,
            }
        elif isinstance(m, ExtHydrGrid):
            cap = getattr(m, "_scare_slack_budget_kgs", None)
            if cap is None:
                continue
            try:
                draw = abs(float(vals.get("mass_flow_kgs", 0.0) or 0.0))
            except (TypeError, ValueError):
                draw = 0.0
            out[f"child-{child.id}"] = {
                "sector": "gas_or_heat",
                "budget_kgs": float(cap),
                "draw_kgs": draw,
                "violated": draw > float(cap) * 1.001,
            }
    return out


def _adapter_observe(monee_net: Any) -> Any:
    """Behavior-like object whose ``observe(aid)`` reads model state directly,
    so ``served_breakdown`` works without a mango world."""
    child_by_aid = {f"child-{c.id}": c for c in monee_net.childs}

    class _OracleBehavior:
        def observe(self, aid: str) -> dict | None:
            child = child_by_aid.get(aid)
            if child is None:
                return None
            return dict(child.model.values)

        def has_action(self, aid: str, name: str) -> bool:  # pragma: no cover
            return False

        def act(self, *_args, **_kwargs) -> None:  # pragma: no cover
            pass

    return _OracleBehavior()


def apply_oracle_heat_linearisation(net: Any) -> None:
    """Apply the McCormick-DHS heat MILP with the oracle's exact settings.

    Shared with the runner's live-net linearisation (islanding scenarios and
    ``heat_mccormick`` arms) so the live physics can never drift from the
    oracle physics it is A/B-compared against.
    """
    net.apply_formulation(
        make_heat_convex_milp_formulation(
            num_partitions=_ORACLE_MCCORMICK_PARTITIONS,
            include_heat_exchangers=False,
        )
    )


def _oracle_failure_payload(
    reason: str, solver_stats: dict[str, Any]
) -> dict[str, Any]:
    """NaN-metric payload for a solve with no usable incumbent (the
    excluded-task convention downstream)."""
    nan = float("nan")
    return {
        "served": {
            "by_tier_sector": {},
            "by_sector": {},
            "by_tier": {},
            "priority_weighted_fraction_by_sector": {},
            "priority_weighted_served": nan,
            "priority_weighted_demand": nan,
            "priority_weighted_fraction": nan,
            "n_loads": nan,
            "n_loads_served_zero": nan,
        },
        "constraint_violation_integral": {
            "electricity": nan,
            "gas": nan,
            "heat": nan,
        },
        "constraint_violations_final": {},
        "regulations": {},
        "lp_success": False,
        "failure_reason": reason,
        "solver_stats": solver_stats,
        "slack_budget_summary": {},
        "solved_net": None,
        "behavior": None,
    }


def _build_min_shed_problem(
    monee_net: Any,
    priorities: dict[str, int] | None,
) -> Any:
    """Apply the oracle formulations to *monee_net* (in place) and build the
    min-load-shedding problem both the one-shot and the temporal oracle solve.
    """
    # Linearise DHS for the oracle only: the factory leaves it nonlinear, but
    # with the binary ``on_off`` on reconfig backup branches the heat balance
    # lifts to degree 4, which Pyomo's LP/QCP writer can't serialise.
    # McCormick-DHS keeps on_off in linear terms so backup lines stay intact.
    # Safe in-place — the net is oracle-dedicated.
    apply_oracle_heat_linearisation(monee_net)
    monee_net.apply_formulation(
        GAS_NONCONVEX_MIQCQP_FORMULATION
    )
    # Enforce the operator slack budget on the LP itself via
    # ``create_min_load_shedding_problem``'s native ``ext_grid_*_bounds`` (no
    # Var.min/max mutation). The factory keeps the LP envelope at 10x budget for
    # energy-flow feasibility while stashing the policy target as
    # ``_scare_slack_budget_*``; passing the per-sector budget here puts SCARE
    # and oracle on the same problem (max served subject to the same budget).
    budgets = _collect_slack_budgets(monee_net)
    # monee defaults: el +-3 MW, gas +-10 kg/s, heat +-10 kg/s. When
    # ``apply_slack_budget`` was not invoked, keep the defaults (not unbounded).
    ext_grid_el_bounds = (
        (-budgets["electricity"], +budgets["electricity"])
        if budgets["electricity"] is not None
        else (-3, 3)
    )
    ext_grid_gas_bounds = (
        (-budgets["gas"], +budgets["gas"]) if budgets["gas"] is not None else (-10, 10)
    )
    # Heat slack is left unbounded by ``apply_slack_budget``; size it from
    # throughput because Sink mass flows are non-regulatable (shed only zeroes
    # HeatLoad q, never Sink flow) so the slack must carry their sum — a fixed
    # ±10 goes infeasible above LV (MVLV ~179 kg/s vs LV ~5).
    heat_envelope_kgs = max(10.0, 2.0 * _heat_throughput_kgs(monee_net))
    ext_grid_heat_bounds = (
        (-budgets["heat"], +budgets["heat"])
        if budgets["heat"] is not None
        else (-heat_envelope_kgs, +heat_envelope_kgs)
    )
    logger.info(
        "oracle: ext-grid budget bounds — el=%s, gas=%s, heat=%s",
        ext_grid_el_bounds,
        ext_grid_gas_bounds,
        ext_grid_heat_bounds,
    )

    # With ``priorities``, attach a per-load weight closure so the LP objective
    # discriminates between tiers; without it the oracle minimises total unserved
    # MW with every load equal (a priority-blind bound, unfair vs the MAS).
    weight_for_load = _weight_for_load_factory(
        monee_net,
        priorities,
        base_demand_weight=WEIGHT_DEMAND,
    )
    # ``bounds_*`` mirror SCARE's ``SECTOR_CONSTRAINTS`` so the LP solves on the
    # same voltage / pressure / temperature envelope. Heat converts SCARE's
    # t_k = (313.15, 403.15) via monee's water-grid t_ref = 356 K -> t_pu.
    bounds_vm_pu = (0.95, 1.05)
    bounds_pressure_pu = (0.85, 1.25)
    # Let each gas slack outlet pressure be an optimizable var within the band,
    # matching SCARE's GasPressureRegulator lever — a fixed setpoint would make
    # the oracle a LOOSER bound (SCARE could raise setpoint to hold nodes an
    # oracle must shed). Gas-only (heat-side is governed by t_k).
    for child in monee_net.childs:
        m = child.model
        if not isinstance(m, ExtHydrGrid):
            continue
        try:
            grid_name = str(
                getattr(monee_net.node_by_id(child.node_id).grid, "name", "")
            ).lower()
        except Exception:
            grid_name = ""
        if "gas" in grid_name:
            m.free_pressure_bounds = bounds_pressure_pu
    prob = create_min_load_shedding_problem(
        weight_for_load=weight_for_load,
        bounds_ext_el=ext_grid_el_bounds,
        bounds_ext_gas=ext_grid_gas_bounds,
        bounds_ext_heat=ext_grid_heat_bounds,
        bounds_vm=bounds_vm_pu,
        bounds_pressure=bounds_pressure_pu,
        bounds_t=(0.8796, 1.1325),
        check_lp=True,
        max_line_loading=1.0,
    )
    # Wired like monee's own pressure hook (``_controllable_appliables`` run
    # against the solve copy) so ``vm_pu_squared`` is actually boxed.
    prob._controllable_appliables.append(_make_vm_bounds_hook(bounds_vm_pu))
    if weight_for_load is not None:
        logger.info(
            "oracle: priority-aware mode — %d loads have per-tier weights",
            len(
                {
                    c.id
                    for c in monee_net.childs
                    if priorities.get(f"child-{c.id}", 0) > 0
                }
            ),
        )
    return prob


def _restrict_ties_to_failed_sectors(monee_net: Any, failures: list[Any]) -> None:
    """Keep backup-tie ``on_off`` binaries free only in sectors hit by a
    branch failure; pin ties in unaffected sectors open (the pre-failure
    radial operating point).

    All 15 ties free → fractional ``on_off`` fictitiously bridges the net (root
    gap ~1) and bilinear heat/Weymouth terms defeat rounding (27/105 reconfig
    solves timed out, no incumbent). A tie in a non-failed sector is not a
    reconfiguration DOF, so pinning it doesn't weaken the oracle. No-op when no
    failed sector is identifiable.
    """
    from monee.model.branch import GasPipe, GenericPowerBranch, WaterPipe

    def _sector(model: Any) -> str | None:
        if isinstance(model, GenericPowerBranch):
            return "power"
        if isinstance(model, GasPipe):
            return "gas"
        if isinstance(model, WaterPipe):
            return "water"
        return None

    branch_by_id = {b.id: b for b in monee_net.branches}
    failed_sectors: set[str] = set()
    for failure in failures:
        sectors = set()
        for bid in getattr(failure, "branch_ids", []) or []:
            branch = branch_by_id.get(tuple(bid) if isinstance(bid, list) else bid)
            if branch is not None:
                sector = _sector(branch.model)
                if sector is not None:
                    sectors.add(sector)
        if not sectors:
            # Unlocalisable failure (generator / CP-branch / node failure):
            # its deficit could make any sector's tie useful, so restricting
            # would make the oracle weaker than SCARE's reconfigurator.
            return
        failed_sectors |= sectors
    if not failed_sectors:
        return
    for branch in monee_net.branches:
        m = branch.model
        if not getattr(m, "backup", False):
            continue
        if _sector(m) in failed_sectors:
            continue
        m.backup = False
        if hasattr(m, "on_off"):
            m.on_off = 0


def _stamp_warm_start(
    monee_net: Any, regulations: dict[str, float] | None
) -> None:
    """Pre-promote decision attributes to Vars whose initial value seeds
    Gurobi's MIP start (``inject_gurobi_vars_attr`` turns ``Var.value`` into
    ``Var.Start``; monee's promotion skips attrs that are already Vars).

    Promotion otherwise resets every regulation and backup ``on_off`` to 1
    (serve-everything) — infeasible under tight slack budgets, so Gurobi gets no
    usable start (27 reconfig tasks in eval_full_v2 timed out sol_count=0). Seed
    child regulation from the pre-failure incumbent and backup ties open.
    Bounds/integrality mirror the promotion, so the problem is unchanged.
    """
    def _is_gas_grid(g: Any) -> bool:
        grids = g if isinstance(g, list) else [g]
        return any(
            gg is not None and hasattr(gg, "higher_heating_value_kwh_per_kg")
            for gg in grids
        )

    if regulations:
        # Allow-list mirrors monee's promotion classes: gurobipy injects EVERY
        # Var attr regardless of promotion, so stamping a non-promoted child
        # (GridForming on microgrid nets, shunts, storages) adds a free
        # regulation Var that turns child-linear node balances bilinear.
        allowed = (PowerLoad, HeatLoad, PowerGenerator, HeatGenerator, Source, Sink)
        for child in monee_net.childs:
            model = child.model
            if not isinstance(model, allowed):
                continue
            if not getattr(child, "active", True) or getattr(child, "ignored", False):
                continue
            # Water-grid Sinks are heating-loop mass flow — never promoted by
            # min_load_shedding, so a stamped Var there would dangle. Water
            # Sources WOULD be promoted (controllable_generators has no grid
            # check); skipping them just costs their warm value, which is fine.
            if isinstance(model, (Sink, Source)) and not _is_gas_grid(
                getattr(child, "grid", None)
            ):
                continue
            if not hasattr(model, "regulation"):
                continue
            if type(getattr(model, "regulation")) is Var:
                continue
            start = regulations.get(f"child-{child.id}")
            if start is None:
                continue
            start = min(1.0, max(0.0, float(start)))
            model.regulation = Var(start, 1, 0, name="regulation")
    for branch in monee_net.branches:
        m = branch.model
        if not getattr(m, "backup", False) or not hasattr(m, "on_off"):
            continue
        if type(m.on_off) is Var:
            continue
        m.on_off = Var(0, 1, 0, True, name="on_off")


def run_oracle(
    monee_net: Any,
    failures: list[Any],
    *,
    solver: Any = None,
    priorities: dict[str, int] | None = None,
    warm_start_regulations: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Solve minimal load shedding on the post-failure network.

    Returns a dict with the optimal regulation per child + the served
    breakdown, in the same shape the scare result composer uses.
    """
    # ``create_min_load_shedding_problem`` is exposed only via the submodule
    # ``monee.problem`` (filtered out of the top-level ``monee`` namespace).
    _apply_failures(monee_net, failures)
    if failures:
        _restrict_ties_to_failed_sectors(monee_net, failures)
    if warm_start_regulations is not None:
        _stamp_warm_start(monee_net, warm_start_regulations)
    prob = _build_min_shed_problem(monee_net, priorities)
    logger.info(
        "oracle: solving min-load-shedding LP on net (%d childs, %d branches)",
        len(monee_net.childs),
        len(monee_net.branches),
    )
    if solver is None:
        solver = _OracleGurobiSolver()
    result = run_energy_flow_optimization(
        monee_net,
        prob,
        solver=solver,
        exclude_unconnected_nodes=True,
    )
    lp_success = bool(getattr(result, "success", True))
    solver_stats = dict(getattr(solver, "solve_stats", None) or {})
    if not solver_stats:
        # Caller-supplied non-oracle solver: record what SolverResult exposes.
        solver_stats = {
            "solver_status": getattr(result, "solver_status", None),
            "termination_condition": getattr(result, "termination_condition", None),
            "objective": getattr(result, "objective", None),
        }
    logger.info(
        "oracle: solve done (success=%s, solver_stats=%s).", lp_success, solver_stats
    )

    if not lp_success:
        # No usable incumbent: do NOT fabricate zero-served results — a zeroed
        # network reads downstream as "oracle served nothing" and PWSF=0.0.
        # NaN metrics are the excluded-task convention.
        report = getattr(result, "infeasibility_report", None)
        reason = (
            "oracle LP returned no usable solution "
            f"(status={solver_stats.get('status')})"
        )
        if report is not None:
            reason = f"{reason}; {report!r}"
        logger.error("oracle: %s", reason)
        return _oracle_failure_payload(reason, solver_stats)

    solved_net = getattr(result, "network", monee_net)
    behavior = _adapter_observe(solved_net)
    served = served_breakdown(solved_net, behavior, priorities=priorities)
    return _oracle_success_payload(solved_net, behavior, served, solver_stats)


def _oracle_success_payload(
    solved_net: Any,
    behavior: Any,
    served: dict[str, Any],
    solver_stats: dict[str, Any],
) -> dict[str, Any]:
    """Shared success-result shape for the one-shot and temporal oracles —
    one place so the two row schemas cannot drift."""
    regulations: dict[str, float] = {}
    for child in solved_net.childs:
        vals = child.model.values if hasattr(child.model, "values") else {}
        regulations[f"child-{child.id}"] = float(vals.get("regulation", 1.0))

    return {
        "served": served,
        "constraint_violation_integral": {
            "electricity": 0.0,
            "gas": 0.0,
            "heat": 0.0,
        },
        "constraint_violations_final": constraint_violations_final(solved_net),
        "regulations": regulations,
        "lp_success": True,
        "solver_stats": solver_stats,
        "slack_budget_summary": _slack_budget_summary(solved_net),
        "solved_net": solved_net,
        "behavior": behavior,
    }


def run_temporal_oracle(
    monee_net: Any,
    failures: list[Any],
    *,
    n_steps: int,
    dt_h: float,
    solver: Any = None,
    priorities: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Receding min-load-shedding oracle over the physical horizon the
    Stepper-based environment integrates.

    Solves the same problem as :func:`run_oracle`, but as ``n_steps`` monee
    Stepper steps of ``dt_h`` hours with inter-step state (gas linepack, LTC
    thermal mass) carried between solves, so the oracle can exploit the
    temporal flexibility the extensions add. Receding = non-anticipative:
    each step optimises the current state only, matching the information the
    MAS has (a full-horizon plan would be a clairvoyant, unfairly tight
    bound).

    Result shape matches :func:`run_oracle` — structural breakdowns are read
    from the final step — plus a ``temporal`` sub-dict with per-step series
    (priority-weighted served fraction, total linepack).
    """
    _apply_failures(monee_net, failures)
    prob = _build_min_shed_problem(monee_net, priorities)
    if solver is None:
        # Tighter per-step cap than the one-shot oracle's 300 s: the horizon
        # multiplies it by n_steps, and the temporal experiments run on the
        # small LV grid where each step solves in seconds.
        solver = _OracleGurobiSolver(params={"TimeLimit": 60})
    logger.info(
        "temporal oracle: %d steps x %.3f h on net (%d childs, %d branches)",
        n_steps,
        dt_h,
        len(monee_net.childs),
        len(monee_net.branches),
    )
    stepper = Stepper(
        monee_net,
        solver=solver,
        optimization_problem=prob,
        on_step_error="skip",
        max_history=2,
        exclude_unconnected_nodes=True,
    )
    series: list[dict[str, Any]] = []
    pwsf_values: list[float] = []
    last: tuple[Any, Any, dict[str, Any]] | None = None
    last_stats: dict[str, Any] = {}
    final_ok = False
    for i in range(int(n_steps)):
        step_result = stepper.step(float(dt_h))
        result = getattr(step_result, "result", None)
        ok = not getattr(step_result, "failed", False) and bool(
            getattr(result, "success", True)
        )
        final_ok = ok
        if not ok:
            series.append(
                {
                    "step": i,
                    "t_h": (i + 1) * float(dt_h),
                    "success": False,
                    "error": str(getattr(step_result, "error", ""))[:500],
                }
            )
            continue
        solved = result.network
        behavior = _adapter_observe(solved)
        step_served = served_breakdown(solved, behavior, priorities=priorities)
        linepack_total = 0.0
        for branch in solved.branches:
            lp = dict(getattr(branch.model, "values", {}) or {}).get("linepack_kg")
            try:
                linepack_total += float(lp)
            except (TypeError, ValueError):
                continue
        pwsf = float(step_served["priority_weighted_fraction"])
        pwsf_values.append(pwsf)
        series.append(
            {
                "step": i,
                "t_h": (i + 1) * float(dt_h),
                "success": True,
                "priority_weighted_fraction": pwsf,
                "linepack_total_kg": linepack_total,
            }
        )
        last = (solved, behavior, step_served)
        last_stats = dict(getattr(solver, "solve_stats", None) or {})

    n_failed = sum(1 for s in series if not s["success"])
    temporal = {
        "n_steps": int(n_steps),
        "dt_h": float(dt_h),
        "horizon_h": int(n_steps) * float(dt_h),
        "n_failed_steps": n_failed,
        "priority_weighted_fraction_mean": (
            sum(pwsf_values) / len(pwsf_values) if pwsf_values else float("nan")
        ),
        "priority_weighted_fraction_min": (
            min(pwsf_values) if pwsf_values else float("nan")
        ),
        "series": series,
    }

    if last is None or not final_ok:
        # No end-of-horizon state: grading the last mid-horizon success as the
        # final result would report an easier (e.g. pre-linepack-depletion)
        # state as the oracle bound. NaN metrics = excluded-task convention.
        reason = (
            f"temporal oracle: all {n_steps} steps failed"
            if last is None
            else (
                f"temporal oracle: final step failed ({n_failed}/{n_steps} "
                "steps failed) — end-of-horizon state unknown"
            )
        )
        logger.error(reason)
        out = _oracle_failure_payload(reason, last_stats)
        out["temporal"] = temporal
        return out

    solved_net, behavior, served = last
    return {
        **_oracle_success_payload(solved_net, behavior, served, last_stats),
        "temporal": temporal,
    }


class BaselineCache:
    """Per-(grid, scenario, priorities) baseline served + incumbent regulations,
    persisted ACROSS a worker's tasks (a hit avoids re-solving the pre-failure
    baseline). Preserves the deepcopy-on-write/read contract so a returned served
    map can be mutated without poisoning the cache. ``clear()`` exists but is not
    called by default — semantics unchanged from the module-dict version."""

    def __init__(self) -> None:
        self._served: dict[str, dict[str, Any]] = {}
        self._regs: dict[str, dict[str, float]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        cached = self._served.get(key)
        return None if cached is None else copy.deepcopy(cached)

    def put(
        self, key: str, served: dict[str, Any], regs: dict[str, float] | None
    ) -> None:
        self._served[key] = copy.deepcopy(served)
        if regs:
            self._regs[key] = dict(regs)

    def get_regs(self, key: str) -> dict[str, float] | None:
        return self._regs.get(key)

    def clear(self) -> None:
        self._served.clear()
        self._regs.clear()


_BASELINE = BaselineCache()


def _baseline_cache_key(
    grid_name: str,
    scenario: dict[str, Any] | None,
    priorities: dict[str, int] | None,
) -> str:
    """Stable, hashable key for the baseline-LP cache. Same grid + scenario +
    priorities => identical LP. JSON with sorted keys is order-invariant."""
    payload = {
        "grid": grid_name,
        "scenario": scenario or {},
        "priorities": priorities or {},
    }
    return json.dumps(payload, sort_keys=True, default=str)


def compute_baseline_served(
    grid_name: str,
    *,
    scenario: dict[str, Any] | None = None,
    priorities: dict[str, int] | None = None,
    solver: Any = None,
) -> dict[str, Any]:
    """Solve the no-failure minimum-load-shedding LP on a freshly built grid
    and return its ``served_breakdown`` — the pre-failure baseline for the
    restoration ratios in :func:`metrics.restoration_breakdown`. Scenarios that
    mutate the network (e.g. ``cold_day``) are applied here too, so the baseline
    matches the post-failure demand profile.

    The grid is built fresh because the LP mutates network variable state.
    Result cached in-process by (grid, scenario, priorities); cache hits return
    a deep copy so callers can mutate freely.
    """
    cache_key = _baseline_cache_key(grid_name, scenario, priorities)
    cached = _BASELINE.get(cache_key)
    if cached is not None:
        return cached

    if grid_name not in GRIDS:
        raise SystemExit(f"Unknown grid {grid_name!r}")
    # Factory applies MISOCP (electricity) but leaves DHS nonlinear;
    # ``run_oracle`` adds the McCormick-DHS heat linearisation.
    fresh = GRIDS[grid_name]()
    if scenario:
        kind = scenario.get("kind", "clean")
        if kind == "cold_day":
            kwargs = {
                k: scenario[k]
                for k in ("supply_t_k", "heat_load_scale")
                if k in scenario
            }
            apply_cold_day(fresh, **kwargs)
        elif kind == "pv_peak":
            kwargs = {
                k: scenario[k] for k in ("gen_scale", "load_scale") if k in scenario
            }
            apply_pv_peak(fresh, **kwargs)
        elif kind == "line_stress":
            kwargs = {
                k: scenario[k]
                for k in ("load_scale", "ampacity_scale", "affect_branch_fraction")
                if k in scenario
            }
            apply_line_stress(fresh, **kwargs)
        elif kind == "microgrid":
            # Baseline LP must use the same islanding config as the sim-time LP,
            # else it overstates the post-failure restoration loss.
            carriers = scenario.get("carriers", ("electricity", "water", "gas"))
            promote_all = bool(scenario.get("promote_all_generators", True))
            former_aids = tuple(scenario.get("grid_former_aids", ()))
            apply_microgrid_islanding(
                fresh,
                carriers=carriers,
                promote_all_generators=promote_all,
                grid_former_aids=former_aids,
            )
        slack_budget_pct = scenario.get("slack_budget_pct")
        if slack_budget_pct is not None:
            hard_cap = scenario.get("slack_hard_cap_carriers")
            apply_slack_budget(
                fresh,
                float(slack_budget_pct),
                hard_cap_carriers=tuple(hard_cap) if hard_cap else None,
            )
    # Baseline has no failure to reroute around, so fix ties open and clear
    # backup — else controllable_backup_lines promotes on_off into a 15-integer
    # nonconvex MIQCQP with no feasible point found (baseline_available=False for
    # every reconfig run). Radial-open is the correct reference; the post-failure
    # oracle and SCARE keep them.
    for branch in fresh.branches:
        if getattr(branch.model, "backup", False):
            branch.model.backup = False
            if hasattr(branch.model, "on_off"):
                branch.model.on_off = 0
    out = run_oracle(fresh, [], solver=solver, priorities=priorities)
    if not out.get("lp_success", True):
        # Raise (uncached) so the runner's baseline-failure path handles it;
        # a cached NaN baseline would poison every task sharing the key.
        raise RuntimeError(
            f"baseline LP failed for grid {grid_name!r}: "
            f"{out.get('failure_reason')}"
        )
    served = out["served"]
    # Side-caches served + the incumbent's per-child regulation for the
    # post-failure oracle's MIP warm start (see baseline_regulations).
    _BASELINE.put(cache_key, served, out.get("regulations"))
    return served


def baseline_regulations(
    grid_name: str,
    *,
    scenario: dict[str, Any] | None = None,
    priorities: dict[str, int] | None = None,
) -> dict[str, float] | None:
    """Per-child regulation of the cached pre-failure baseline incumbent, or
    ``None`` when :func:`compute_baseline_served` has not run for this key."""
    return _BASELINE.get_regs(_baseline_cache_key(grid_name, scenario, priorities))


def _compose_outcomes(served: dict[str, Any]) -> dict[str, Any]:
    """The served->outcomes core both compose_oracle_result branches share."""
    return {
        "priority_weighted_demand": served["priority_weighted_demand"],
        "priority_weighted_served": served["priority_weighted_served"],
        "priority_weighted_fraction": served["priority_weighted_fraction"],
        "priority_weighted_fraction_by_sector": served.get(
            "priority_weighted_fraction_by_sector", {}
        ),
        "served_by_sector": served["by_sector"],
        "served_by_tier": served["by_tier"],
        "served_by_tier_sector": served["by_tier_sector"],
        "n_loads": served["n_loads"],
        "n_loads_served_zero": served["n_loads_served_zero"],
    }


def _oracle_claims(
    slack: Any, constraint: Any, *, heat_priority: Any = None
) -> dict[str, Any]:
    """Oracle claims envelope. No priority-invariant claim (the oracle has no
    MAS dispatch to invariant-check); heat_priority is the non-gating oracle-
    relative diagnostic, present only on the success path."""
    claims = {"slack_budget_compliance": slack, "constraint_compliance": constraint}
    if heat_priority is not None:
        claims["heat_priority"] = heat_priority
    return claims


def compose_oracle_result(
    *,
    monee_net: Any,
    failures: list[Any],
    task_meta: dict[str, Any],
    wallclock_s: float,
    solver: Any = None,
    priorities: dict[str, int] | None = None,
    baseline_served: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    simulation_duration_s: float | None = None,
    warm_start_regulations: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a result.json payload identical in shape to the scare
    composer so the aggregator can read both off the same schema.

    When ``out_dir`` is given, also writes the oracle's ``served.csv`` and
    ``served_by_load.csv`` (the per-load detail the MAS variants emit) so the
    report can compare SCARE against the oracle per (sector, tier, load) and the
    oracle-relative ``heat_priority`` diagnostic can be calibrated.

    Scenarios carrying temporal extensions (``linepack``/``ltc``) plus the
    physics-stepping keys (``physics_time_scale``, ``physics_interval_s``)
    route to :func:`run_temporal_oracle` on the same step grid the MAS
    environment integrates, so both sides face the same temporal physics.
    """
    scenario = (task_meta or {}).get("scenario") or {}
    temporal_requested = bool(scenario.get("linepack") or scenario.get("ltc"))
    physics_interval_s = scenario.get("physics_interval_s")
    interval_ok = physics_interval_s is not None and float(physics_interval_s) > 0
    if temporal_requested and interval_ok and simulation_duration_s:
        time_scale = float(scenario.get("physics_time_scale", 1.0))
        dt_h = float(physics_interval_s) * time_scale / 3600.0
        n_steps = max(1, round(float(simulation_duration_s) / float(physics_interval_s)))
        out = run_temporal_oracle(
            monee_net,
            failures,
            n_steps=n_steps,
            dt_h=dt_h,
            solver=solver,
            priorities=priorities,
        )
    else:
        if temporal_requested:
            # The env still integrates temporal physics for these scenarios —
            # a one-shot bound is not like-for-like, so make the fallback loud.
            logger.warning(
                "temporal extensions requested (linepack/ltc) but "
                "physics_interval_s=%r / simulation_duration_s=%r do not "
                "define a step grid — falling back to the ONE-SHOT oracle.",
                physics_interval_s,
                simulation_duration_s,
            )
        out = run_oracle(
            monee_net,
            failures,
            solver=solver,
            priorities=priorities,
            warm_start_regulations=warm_start_regulations,
        )
    served = out["served"]
    integral = out["constraint_violation_integral"]
    solver_stats = out.get("solver_stats", {})

    if not out.get("lp_success", True):
        # Failed solve: NaN served/PWSF (excluded-task convention), completed
        # False, no per-load CSVs (there is no solved net — the old zeroed
        # artefacts fabricated "oracle served nothing" rows).
        failure_detail = {
            "reason": out.get("failure_reason"),
            "solver_stats": solver_stats,
            "enforced_at_lp": True,
        }
        failed_claim = {"passed": False, "detail": failure_detail}
        return {
            "task": task_meta,
            "wallclock_s": wallclock_s,
            "completed": False,
            "sim_time_final": 0.0,
            "outcomes": {
                **_compose_outcomes(served),
                "constraint_violation_integral": integral,
                "constraint_violations_final": {},
                "time_to_stabilise_s": 0.0,
                "regulates_total": 0,
                "regulates_by_reason": {},
                "restoration": {"baseline_available": False},
                "oracle_lp_success": False,
                "oracle_failure_reason": out.get("failure_reason"),
                "oracle_solver_stats": solver_stats,
                "oracle_solve_optimal": False,
                "slack_budget_summary": {},
                **(
                    {"oracle_temporal": out["temporal"]}
                    if "temporal" in out
                    else {}
                ),
            },
            "claims": _oracle_claims(failed_claim, failed_claim),
            "diary": {"invariant_holds": True},
            "events": {},
            "messages": {},
            "oracle_regulations": {},
        }

    restoration = restoration_breakdown(served, baseline_served)

    # Per-load detail off the SAME solved network the served breakdown used (the
    # input ``monee_net`` reads every load at regulation 1.0; see ``run_oracle``).
    solved_net = out["solved_net"]
    behavior = out["behavior"]
    load_rows = served_by_load(solved_net, behavior, priorities=priorities)
    if out_dir is not None:
        write_served_csv(
            Path(out_dir) / "served.csv", solved_net, behavior, priorities=priorities
        )
        write_served_by_load_csv(
            Path(out_dir) / "served_by_load.csv",
            solved_net,
            behavior,
            priorities=priorities,
        )
        # Per-node constraint readings off the oracle's solved net, so the
        # voltage/pressure comparison vs SCARE is non-vacuous (oracle tasks
        # previously emitted no constraints_final.csv).
        write_constraints_final_csv(
            Path(out_dir) / "constraints_final.csv", solved_net
        )

    # Oracle-relative heat-priority diagnostic (non-gating). Computed in-memory
    # off ``load_rows`` so the oracle carries the same claim key the MAS variants
    # do; previously absent, leaving all oracle tasks NaN for this check.
    heat_priority_claim = heat_priority_from_rows(load_rows)

    # Slack budget compliance: the LP envelope is tightened to +-budget in
    # run_oracle, so a successful solve satisfies it by construction. Surfaced
    # via SCARE's ``claims`` shape for like-for-like comparison.
    slack_summary = out.get("slack_budget_summary", {})
    any_violation = any(bool(d.get("violated")) for d in slack_summary.values())
    slack_claim = {
        "passed": (not any_violation) and bool(out.get("lp_success", True)),
        "detail": {
            "per_slack": slack_summary,
            "n_violations": sum(1 for d in slack_summary.values() if d.get("violated")),
            "enforced_at_lp": True,
        },
    }

    # Grid-feasibility claim mirroring SCARE's ``constraint_compliance`` so the
    # aggregator gates both sides on the same flags. ``passed`` is True unless
    # the solve failed or a residual numerical excursion slipped through.
    constraints_final = out.get("constraint_violations_final", {})
    constraint_claim = {
        "passed": bool(constraints_final.get("passed", True))
        and bool(out.get("lp_success", True)),
        "detail": {
            "n_checked": constraints_final.get("n_checked", 0),
            "n_violations": constraints_final.get("n_violations", 0),
            "n_nongating_violations": constraints_final.get(
                "n_nongating_violations", 0
            ),
            "by_sector": constraints_final.get("by_sector", {}),
            # Per-variable breakdown so the oracle populates the same
            # by_variable__{voltage,pressure,...} summary columns SCARE does —
            # without it the per-variable comparison (e.g. voltage n_checked) is
            # vacuously 0 for every oracle task.
            "by_variable": constraints_final.get("by_variable", {}),
            "violations": constraints_final.get("violations", []),
            "nongating_violations": constraints_final.get("nongating_violations", []),
            "enforced_at_lp": True,
        },
    }

    extension_outcomes: dict[str, Any] = {}
    if "temporal" in out:
        extension_outcomes["oracle_temporal"] = out["temporal"]
    if getattr(monee_net, "islanding_config", None) is not None:
        # Count both static pruning (.ignored) and MILP e=0 decisions — the
        # latter never sets .ignored (see metrics._is_deenergised).
        extension_outcomes["oracle_islanding"] = {
            "enabled": True,
            "nodes_deenergised": sum(
                1
                for n in solved_net.nodes
                if getattr(n, "ignored", False) or _is_deenergised({}, n)
            ),
        }

    return {
        "task": task_meta,
        "wallclock_s": wallclock_s,
        "completed": True,
        "sim_time_final": 0.0,  # one-shot — no sim trajectory
        "outcomes": {
            **_compose_outcomes(served),
            **extension_outcomes,
            "n_net_nodes": len(getattr(solved_net, "nodes", []) or []),
            "constraint_violation_integral": integral,
            "constraint_violations_final": constraints_final,
            "time_to_stabilise_s": 0.0,
            "regulates_total": 0,
            "regulates_by_reason": {},
            "restoration": restoration,
            # Same measure as the MAS variants, off the oracle's solved net —
            # the ceiling for "how much do the coupling points contribute?".
            "cp_generation": cp_generation_breakdown(solved_net),
            "oracle_lp_success": out.get("lp_success", True),
            "oracle_solver_stats": solver_stats,
            "oracle_solve_optimal": solver_stats.get("solve_optimal"),
            "slack_budget_summary": slack_summary,
        },
        "claims": _oracle_claims(
            slack_claim, constraint_claim, heat_priority=heat_priority_claim
        ),
        "diary": {"invariant_holds": True},  # vacuous
        "events": {},
        "messages": {},
        "oracle_regulations": out["regulations"],
    }
