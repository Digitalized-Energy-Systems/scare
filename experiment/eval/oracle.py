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
from pathlib import Path
from typing import Any

from monee import run_energy_flow_optimization
from monee.model.child import ExtHydrGrid, ExtPowerGrid
from monee.model.core import Var
from monee.model.formulation import make_heat_convex_milp_formulation, GAS_NONCONVEX_MIQCQP_FORMULATION
from monee.model.node import Bus
from monee.problem import (
    WEIGHT_DEMAND,
    create_min_load_shedding_problem,
)
from monee.solver.gurobipy import GurobipySolver

from experiment.eval.claims import heat_priority_from_rows
from experiment.eval.metrics import (
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

# Partition count for the oracle's McCormick-DHS heat linearisation —
# matches the value the grid factory uses when DHS is enabled there.
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


# Bounded near-strict priority ladder for the oracle LP objective. The oracle
# minimises ``sum(weight_load * (1 - regulation))``, so per MW a higher tier is
# preferred to a lower one whenever ``w(tier)`` is strictly decreasing — the
# ordering is strict AT THE MARGIN for any decreasing ladder, independent of
# demand sizes. This mirrors SCARE's own objective (tier-1 hard-locked, tiers 2-4
# steep ``1e8/1e4/1``; base/util.tier_priority_weight) so the optimality gap
# measures solver quality, not a policy mismatch — NOT the PWSF metric's moderate
# 8:4:2:1, which would make the oracle trade tier-1 away and inflate the gap.
#
# Two solver-resolution constraints apply. (1) Aux floor: monee's
# ``auto_priority_floor`` lifts the shed weights off the max objective
# coefficient, so a tier whose weight sits far below the max is swamped below
# the aux terms. That is exactly why the legacy ``1e12/1e8/1e4/1`` ladder
# behaved priority-BLIND — tiers 3/4 were 1e8-1e12x below tier 1, the LP tied
# them, and the oracle shed serveable tier-1 load under forced shedding. A span
# of 1e6 (adjacent ratio 100) keeps every tier above the aux floor. (2) MIP
# TERMINATION: the aux-floor analysis says nothing about the branch-and-bound
# gap. With forced-shed objective terms at ~1e9·MW, a relative MIPGap of 1e-3
# (monee's default) exceeds the entire tier-3/4 objective contribution, so
# their dispatch inside the gap is arbitrary (~64k tier inversions measured;
# reproducibly eliminated at MIPGap=1e-8). ``_ORACLE_GUROBI_PARAMS`` below
# tightens the gap so no tier decision fits inside the termination tolerance.
# Validated by a forced-shedding oracle solve
# (see tests/eval/test_oracle_priority.py).
_ORACLE_TIER_WEIGHT: dict[int, float] = {1: 1e6, 2: 1e4, 3: 1e2, 4: 1.0}

# MIPGapAbs = half the objective cost of fully shedding a conservative
# minimum-size (1e-3 MW) load at the cheapest tier
# (0.5 · WEIGHT_DEMAND · tier-4 weight · 1e-3 MW = 0.5), so no tier decision
# can hide inside the absolute termination gap; MIPGap covers the relative
# criterion. TimeLimit matches monee's default.
_ORACLE_MIN_LOAD_MW = 1e-3
_ORACLE_GUROBI_PARAMS: dict[str, float] = {
    "MIPGap": 1e-9,
    "MIPGapAbs": 0.5 * WEIGHT_DEMAND * _ORACLE_TIER_WEIGHT[4] * _ORACLE_MIN_LOAD_MW,
    "TimeLimit": 300,
}


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
    """Bound the electricity voltage Var the solver actually optimises.

    monee's ``bounds_vm`` boxes only the ``vm_pu`` attribute, which under the
    MISOCP electricity formulation is a reporting Intermediate at solve time —
    the real Var is ``vm_pu_squared`` (box (0, 2.25)) — so the voltage envelope
    was a no-op there. Mirrors monee's ``_make_pressure_bounds_hook``
    (``pressure_squared_pu``); ``vm_pu`` is still bounded when it IS a Var
    (non-MISOCP formulations).
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

    The tier is STAMPED onto each load model as ``_scare_oracle_tier`` and the
    closure resolves it via ``getattr``, NOT an ``id(model)`` map. monee's solver
    deep-copies the network before building the objective
    (``solver/core.py``: ``input_network.copy()``), so the objective sees COPIED
    models whose ``id`` differs from the originals; an id-keyed map misses every
    one, silently returns ``None`` for all loads, and renders the oracle
    priority-BLIND (verified: reversing the ladder left the dispatch unchanged).
    A model attribute survives the copy, so the weight resolves on the copy too.
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


def _slack_budget_summary(monee_net: Any) -> dict[str, Any]:
    """Realised slack draw vs operator budget on the post-LP network. Returns
    ``{aid: {budget, draw, violated}}`` for every slack child carrying an
    explicit budget; basis for the oracle's slack-budget-compliance claim.
    """
    out: dict[str, Any] = {}
    for child in monee_net.childs:
        m = child.model
        # After the Pyomo solve, model attributes are Var objects; ``.values``
        # resolves each via ``pyomo.value(...)``.
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


def run_oracle(
    monee_net: Any,
    failures: list[Any],
    *,
    solver: Any = None,
    priorities: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Solve minimal load shedding on the post-failure network.

    Returns a dict with the optimal regulation per child + the served
    breakdown, in the same shape the scare result composer uses.
    """
    # ``create_min_load_shedding_problem`` is exposed only via the submodule
    # ``monee.problem`` (filtered out of the top-level ``monee`` namespace).
    _apply_failures(monee_net, failures)

    # Linearise the district-heating temperature physics for the oracle solve
    # only. The factory leaves DHS in its full nonlinear form; with the binary
    # ``on_off`` decision var on reconfiguration-grid backup branches, the
    # nonlinear heat balance lifts to degree 4, which Pyomo's LP/QCP writer
    # cannot serialise. McCormick-DHS replaces the balance with a piecewise
    # linear envelope (on_off then enters only linear terms), letting the oracle
    # solve with backup lines intact, comparable to SCARE. Applied here, not in
    # the factory, so only the oracle LP is affected; the net is oracle-dedicated
    # so the in-place swap is safe.
    monee_net.apply_formulation(
        make_heat_convex_milp_formulation(
            num_partitions=_ORACLE_MCCORMICK_PARTITIONS,
            include_heat_exchangers=False,
        )
    )
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
    # Heat-side ExtHydrGrid is left unbounded by ``apply_slack_budget`` (heating
    # loop mass flow is constrained by HE physics, not policy); keep the default.
    ext_grid_heat_bounds = (
        (-budgets["heat"], +budgets["heat"])
        if budgets["heat"] is not None
        else (-10, 10)
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
    # Give the oracle the SAME regulator-setpoint lever the MAS now has: let each
    # gas slack outlet pressure be an optimizable decision var within the band,
    # not a single fixed pin. Without this the oracle is a LOOSER bound — SCARE's
    # GasPressureRegulator could raise its setpoint to hold nodes in band where a
    # fixed-setpoint oracle would have to shed. Gas-only (the heat-side
    # ExtHydrGrid is governed by t_k, not pressure).
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

    solved_net = getattr(result, "network", monee_net)
    behavior = _adapter_observe(solved_net)
    served = served_breakdown(solved_net, behavior, priorities=priorities)

    integral = {"electricity": 0.0, "gas": 0.0, "heat": 0.0}

    regulations: dict[str, float] = {}
    for child in solved_net.childs:
        vals = child.model.values if hasattr(child.model, "values") else {}
        regulations[f"child-{child.id}"] = float(vals.get("regulation", 1.0))

    slack_summary = _slack_budget_summary(solved_net)

    constraints_final = constraint_violations_final(solved_net)

    return {
        "served": served,
        "constraint_violation_integral": integral,
        "constraint_violations_final": constraints_final,
        "regulations": regulations,
        "lp_success": lp_success,
        "solver_stats": solver_stats,
        "slack_budget_summary": slack_summary,
        "solved_net": solved_net,
        "behavior": behavior,
    }


_BASELINE_CACHE: dict[str, dict[str, Any]] = {}


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
    cached = _BASELINE_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

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
            apply_slack_budget(fresh, float(slack_budget_pct))
    # No-failure baseline: the backup tie-lines are normally-open and there is
    # nothing to reroute around, so fix them open for THIS solve and clear the
    # ``backup`` flag so ``controllable_backup_lines`` does not promote ``on_off``
    # to a binary. Keeping them controllable turned the reconfig baseline into a
    # 15-integer nonconvex MIQCQP that gurobi could not find a feasible point for
    # within the per-solve cap (baseline_available=False for every reconfig run);
    # the radial pre-failure operating point — tie-lines open — is also the
    # physically correct reference. The post-failure oracle and SCARE keep them.
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
    _BASELINE_CACHE[cache_key] = copy.deepcopy(served)
    return served


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
) -> dict[str, Any]:
    """Build a result.json payload identical in shape to the scare
    composer so the aggregator can read both off the same schema.

    When ``out_dir`` is given, also writes the oracle's ``served.csv`` and
    ``served_by_load.csv`` (the per-load detail the MAS variants emit) so the
    report can compare SCARE against the oracle per (sector, tier, load) and the
    oracle-relative ``heat_priority`` diagnostic can be calibrated.
    """
    out = run_oracle(monee_net, failures, solver=solver, priorities=priorities)
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
            },
            "claims": {
                "slack_budget_compliance": failed_claim,
                "constraint_compliance": failed_claim,
            },
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

    return {
        "task": task_meta,
        "wallclock_s": wallclock_s,
        "completed": True,
        "sim_time_final": 0.0,  # one-shot — no sim trajectory
        "outcomes": {
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
        "claims": {
            # No priority-invariant claim: the oracle has no MAS-side dispatch
            # to invariant-check. Other variants populate it via evaluate_task.
            "slack_budget_compliance": slack_claim,
            "constraint_compliance": constraint_claim,
            # Oracle-relative heat-priority diagnostic (non-gating): the
            # achievable feasible-subset tier ordering the LP attains, the
            # reference SCARE's controllable heat gap is measured against.
            "heat_priority": heat_priority_claim,
        },
        "diary": {"invariant_holds": True},  # vacuous
        "events": {},
        "messages": {},
        "oracle_regulations": out["regulations"],
    }
