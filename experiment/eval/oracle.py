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
from typing import Any

from monee import run_energy_flow_optimization
from monee.model.child import ExtHydrGrid, ExtPowerGrid
from monee.model.formulation import make_mccormick_dhs_formulation
from monee.problem import (
    WEIGHT_DEMAND,
    create_min_load_shedding_problem,
)
from monee.solver.pyo import PyomoSolver

from experiment.eval.metrics import (
    constraint_violations_final,
    restoration_breakdown,
    served_breakdown,
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

    Branches: ``branch.active = False`` (and ``on_off = 0``; both routes honoured
    by the solver's edge-removal). Nodes: ``node.active = False``. Custom
    (generator/compound deactivation): invoke the closure so ``net.deactivate``
    runs. Handling all three is required, else generator-failure scenarios solve
    on an unfailed network and return PWSF=1.0.
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


_ORACLE_TIER_WEIGHT: dict[int, float] = {
    # Dedicated 4-tier oracle schedule. Magnitude separation between tiers is
    # large enough that the LP cannot tie tier 1 against any lower tier under
    # numerical noise. (The L1 QP's schedule differs — it returns weight 0 for
    # tier 1 because the leader hard-locks it off-QP, unusable for the LP.)
    1: 1e12,
    2: 1e8,
    3: 1e4,
    4: 1.0,
}


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
    default (generators / slack / unmapped models). Resolves models by
    ``id(model)``. Returns ``None`` overall when no priorities are supplied, so
    monee uses its flat-weight behaviour. Out-of-range tiers clamp onto [1, 4].
    """
    if not priorities:
        return None
    model_to_tier: dict[int, int] = {}
    for child in monee_net.childs:
        aid = f"child-{child.id}"
        tier = priorities.get(aid)
        if tier is None or int(tier) <= 0:
            continue
        model_to_tier[id(child.model)] = int(tier)

    if not model_to_tier:
        return None

    def _w(model: Any) -> float | None:
        tier = model_to_tier.get(id(model))
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
                draw = abs(float(vals.get("mass_flow", 0.0) or 0.0))
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
        make_mccormick_dhs_formulation(num_partitions=_ORACLE_MCCORMICK_PARTITIONS)
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

    # GEKKO (monee's default) hits "Max Equation Length" on large grids (the LP
    # objective is one huge ``minimize(...)`` expression); Pyomo handles
    # arbitrarily large objectives.
    if solver is None:
        solver = PyomoSolver()

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
    prob = create_min_load_shedding_problem(
        weight_for_load=weight_for_load,
        ext_grid_el_bounds=ext_grid_el_bounds,
        ext_grid_gas_bounds=ext_grid_gas_bounds,
        ext_grid_heat_bounds=ext_grid_heat_bounds,
        bounds_el=(0.95, 1.05),
        bounds_gas=(0.90, 1.10),
        bounds_heat=(0.8796, 1.1325),
        check_line_loading=True,
        max_line_loading=1.0,
    )
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
    # ``exclude_unconnected_nodes=True`` is mandatory: otherwise the solver
    # assembles equations for the disconnected component, the LP goes
    # structurally infeasible (e.g. mass-balance on a heat node with no inflow
    # and non-zero load), and monee returns with ``regulation`` left at the
    # default 1.0 — which the metric reads as "everything served".
    result = run_energy_flow_optimization(
        monee_net,
        prob,
        solver=solver,
        solver_name="gurobi",
        exclude_unconnected_nodes=True,
    )
    lp_success = bool(getattr(result, "success", True))
    logger.info("oracle: solve done (success=%s).", lp_success)

    # Read the metric off ``result.network`` — the solver's internal copy,
    # where the LP's optimal ``regulation`` values live as Pyomo Vars. On the
    # input ``monee_net``, ``regulation`` starts as a plain float 1.0 and is
    # promoted to a Var only on the copy, so monee's back-write (Var-only) skips
    # it; the input net would read every load at regulation 1.0 (= fully served).
    # Slack ``p_mw`` does back-write (Var on both nets), so
    # ``_slack_budget_summary`` works on either.
    solved_net = getattr(result, "network", monee_net)
    behavior = _adapter_observe(solved_net)
    served = served_breakdown(solved_net, behavior, priorities=priorities)

    # Oracle has no time-series; report a zero integral for schema uniformity.
    integral = {"electricity": 0.0, "gas": 0.0, "heat": 0.0}

    # Per-child regulation off the solved copy. Use ``model.values`` (resolves
    # via ``pyomo.value(var)``); a direct ``float(regulation)`` raises because
    # ``regulation`` is a Var on the solved network.
    regulations: dict[str, float] = {}
    for child in solved_net.childs:
        vals = child.model.values if hasattr(child.model, "values") else {}
        regulations[f"child-{child.id}"] = float(vals.get("regulation", 1.0))

    slack_summary = _slack_budget_summary(solved_net)

    # End-of-sim hard-bound feasibility on the solved LP network. The LP enforces
    # the envelope by construction, so this should pass; scanning it anyway keeps
    # the oracle's ``constraint_compliance`` claim on SCARE's measurement path and
    # surfaces any residual numerical excursion.
    constraints_final = constraint_violations_final(solved_net)

    return {
        "served": served,
        "constraint_violation_integral": integral,
        "constraint_violations_final": constraints_final,
        "regulations": regulations,
        "lp_success": lp_success,
        "slack_budget_summary": slack_summary,
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
    # Keep backup tie-lines on the baseline LP, as the post-failure oracle and
    # SCARE do, so all three measure the same problem. The McCormick-DHS
    # linearisation in ``run_oracle`` keeps the binary ``on_off`` term linear, so
    # the backup branches need no flag-stripping.
    out = run_oracle(fresh, [], solver=solver, priorities=priorities)
    served = out["served"]
    _BASELINE_CACHE[cache_key] = copy.deepcopy(served)
    return served


def compose_oracle_result(
    *,
    monee_net: Any,
    failures: list[Any],
    task_meta: dict[str, Any],
    wallclock_s: float,
    solver: str | None = None,
    priorities: dict[str, int] | None = None,
    baseline_served: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a result.json payload identical in shape to the scare
    composer so the aggregator can read both off the same schema."""
    out = run_oracle(monee_net, failures, solver=solver, priorities=priorities)
    served = out["served"]
    integral = out["constraint_violation_integral"]
    restoration = restoration_breakdown(served, baseline_served)

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
            "by_sector": constraints_final.get("by_sector", {}),
            "violations": constraints_final.get("violations", []),
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
            "oracle_lp_success": out.get("lp_success", True),
            "slack_budget_summary": slack_summary,
        },
        "claims": {
            # No priority-invariant claim: the oracle has no MAS-side dispatch
            # to invariant-check. Other variants populate it via evaluate_task.
            "slack_budget_compliance": slack_claim,
            "constraint_compliance": constraint_claim,
        },
        "diary": {"invariant_holds": True},  # vacuous
        "events": {},
        "messages": {},
        "oracle_regulations": out["regulations"],
    }
