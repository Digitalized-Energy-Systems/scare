"""Centralised oracle baseline.

Runs monee's minimal-load-shedding LP on the post-failure network and
returns the optimal regulation factors and the same outcome metrics
the scare variant produces, so the aggregator can compare row-by-row.

No mango, no agents, no time evolution — a single optimisation that
defines the upper bound on what's achievable.  Optimality gap =
``(oracle.priority_weighted_served − scare.priority_weighted_served)
/ oracle.priority_weighted_served``.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from monee import run_energy_flow_optimization
from monee.model.child import ExtHydrGrid, ExtPowerGrid
from monee.problem import (
    WEIGHT_DEMAND,
    create_min_load_shedding_problem,
)
from monee.solver.pyo import PyomoSolver

from experiment.eval.metrics import restoration_breakdown, served_breakdown
from experiment.restoration import (
    GRIDS,
    apply_cold_day,
    apply_line_stress,
    apply_microgrid_islanding,
    apply_pv_peak,
    apply_slack_budget,
)

logger = logging.getLogger(__name__)


def _apply_failures(monee_net: Any, failures: list[Any]) -> None:
    """Apply every failure to the network *exactly* the way the live
    simulation's ``apply_failures`` does, so the oracle solves the same
    post-failure topology scare sees.

    Branches: set ``branch.active = False`` (and ``on_off = 0`` belt-and-
    braces — both routes are honoured by the solver's edge-removal pass).
    Nodes: set ``node.active = False``.
    Custom (generator/compound deactivation): invoke the closure on the
    network so ``net.deactivate(component)`` runs.

    The earlier version handled only ``branch_ids``, which silently
    swallowed every generator-failure scenario — the LP then solved on
    an unfailed network and returned PWSF=1.0.  See
    mango_energy_environments.environments.restoration.multi_energy_monee.apply_failures
    for the canonical reference.
    """
    for failure in failures:
        for branch_id in getattr(failure, "branch_ids", []) or []:
            try:
                branch = monee_net.branch_by_id(branch_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "oracle: could not find branch %s to disable: %s",
                    branch_id, exc,
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
                    node_id, exc,
                )
        custom = getattr(failure, "custom", None)
        if custom is not None:
            try:
                custom(monee_net)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "oracle: custom failure %s raised: %s",
                    getattr(failure, "custom_id", None), exc,
                )


def _weight_for_load_factory(
    monee_net: Any,
    priorities: dict[str, int] | None,
    *,
    base_demand_weight: float,
    n_tiers: int = 10,
) -> Any | None:
    """Build a ``weight_for_load`` closure for monee's
    ``create_min_load_shedding_problem``.

    Returns a callable ``model -> float | None`` where ``float`` is the
    per-load weight ``base_demand_weight × 2^(P − tier + 1)`` and
    ``None`` defers to monee's default ``demand_weight``.  The
    exponential schedule matches L1's QP weighting and the tier-
    stratified holon ADMM so the three layers agree on priority
    ordering.

    Returns ``None`` if no priorities are supplied — caller passes
    that through to ``weight_for_load=None`` so monee uses its
    legacy flat-weight behaviour (no priority discrimination).

    The closure resolves models by ``id(model)``, populated once
    from the network's child list.  Generators / slack / unmapped
    models return ``None`` so monee uses the default for them.
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
        # Schedule mirrors balance.py:_qp_priority_weight (restoration):
        # tier-1 -> 2^P, tier-P -> 2.  Anchored at the demand-weight base
        # so the auto-floor logic continues to see something close to the
        # legacy magnitude as the minimum.
        return float(base_demand_weight) * (2.0 ** max(0, n_tiers - tier + 1))

    return _w


def _collect_slack_budgets(monee_net: Any) -> dict[str, float | None]:
    """Scan slack children for budgets stamped by ``apply_slack_budget``.

    Returns the per-sector budget in monee's native units (MW for
    electricity, kg/s for gas/heat).  ``apply_slack_budget`` stamps
    the same value on every slack child of a given sector — we just
    read the first non-None we find per sector.  Returns ``None`` for
    a sector when no budget was registered (e.g. heat is intentionally
    left unbounded — apply_slack_budget never stamps it).
    """
    out: dict[str, float | None] = {
        "electricity": None, "gas": None, "heat": None,
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
    """Compute realised slack draw vs operator budget on the post-LP
    network.  Returns ``{aid: {budget, draw, violated}}`` for every
    slack child carrying an explicit budget.  Used to derive the
    slack-budget-compliance claim for the oracle (and to surface
    budget headroom in the result.json for any variant).
    """
    out: dict[str, Any] = {}
    for child in monee_net.childs:
        m = child.model
        # After Pyomo solve the model attributes are Var objects; the
        # ``.values`` dict resolves each via ``pyomo.value(...)``.
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
    """Return a behavior-like object whose ``observe(aid)`` reads the
    model state directly (so ``served_breakdown`` works unchanged
    without a mango world).
    """
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
    # NB: ``create_min_load_shedding_problem`` is exposed only via
    # the submodule ``monee.problem`` — top-level ``monee`` re-imports
    # it but it gets filtered out of the public namespace.
    _apply_failures(monee_net, failures)
    # 2026-05-24: enforce the operator slack budget on the LP itself
    # via ``create_min_load_shedding_problem``'s native
    # ``ext_grid_*_bounds`` parameters (no Var.min/max mutation).
    # The grid factory leaves the LP envelope at 10× budget for
    # energy-flow feasibility while stashing the policy target as
    # ``_scare_slack_budget_*`` on each slack child.  SCARE's agents
    # enforce that target via regulation; the oracle previously saw
    # the full 10× envelope and "won" by violating the policy.
    # Passing the per-sector budget through here puts SCARE and
    # oracle on the same problem: max served subject to the same
    # operator constraint.
    budgets = _collect_slack_budgets(monee_net)
    # monee defaults: el ±3 MW, gas ±10 kg/s, heat ±10 kg/s.  When
    # ``apply_slack_budget`` was not invoked (e.g. test paths without
    # a slack scenario knob), we keep the monee defaults rather than
    # falling back to "unbounded" — that matches the legacy contract.
    ext_grid_el_bounds = (
        (-budgets["electricity"], +budgets["electricity"])
        if budgets["electricity"] is not None else (-3, 3)
    )
    ext_grid_gas_bounds = (
        (-budgets["gas"], +budgets["gas"])
        if budgets["gas"] is not None else (-10, 10)
    )
    # Heat-side ExtHydrGrid is intentionally left unbounded by
    # ``apply_slack_budget`` (heating-loop mass flow is constrained
    # by HE physics, not by policy).  Keep the monee default so the
    # LP retains the slack envelope it always had.
    ext_grid_heat_bounds = (
        (-budgets["heat"], +budgets["heat"])
        if budgets["heat"] is not None else (-10, 10)
    )
    logger.info(
        "oracle: ext-grid budget bounds — el=%s, gas=%s, heat=%s",
        ext_grid_el_bounds, ext_grid_gas_bounds, ext_grid_heat_bounds,
    )

    # GEKKO (monee's default) hits "Max Equation Length" on grids with
    # several hundred children because the LP objective is built as a
    # single huge ``minimize(...)`` expression.  Pyomo handles
    # arbitrarily large objectives — the rest of scare already uses it
    # for the energy-flow simulation.  ``solver_name`` matches what
    # ``mango_energy_environments`` uses at simulation time so we don't
    # require any additional binaries beyond what scare already needs.
    if solver is None:
        solver = PyomoSolver()

    # When ``priorities`` is supplied, attach a per-load weight
    # closure so the LP objective discriminates between tiers.
    # Without it the oracle minimises *total* unserved MW with
    # every load equal — defining a "priority-blind upper bound"
    # that compares unfairly against the priority-aware MAS.
    weight_for_load = _weight_for_load_factory(
        monee_net, priorities, base_demand_weight=WEIGHT_DEMAND,
    )
    # ``bounds_el / bounds_gas / bounds_heat`` mirror SCARE's
    # ``SECTOR_CONSTRAINTS`` so the LP is solved on the *same*
    # voltage / pressure / temperature operating envelope SCARE
    # enforces via the clamp deadband.  Heat is converted from
    # SCARE's t_k = (283.15, 403.15) using monee's water-grid
    # reference ``t_ref = 356 K`` → t_pu ≈ (0.7954, 1.1325).
    prob = create_min_load_shedding_problem(
        weight_for_load=weight_for_load,
        ext_grid_el_bounds=ext_grid_el_bounds,
        ext_grid_gas_bounds=ext_grid_gas_bounds,
        ext_grid_heat_bounds=ext_grid_heat_bounds,
        bounds_el=(0.95, 1.05),
        bounds_gas=(0.90, 1.10),
        bounds_heat=(0.7954, 1.1325),
    )
    if weight_for_load is not None:
        logger.info(
            "oracle: priority-aware mode — %d loads have per-tier weights",
            len({c.id for c in monee_net.childs
                 if priorities.get(f'child-{c.id}', 0) > 0}),
        )
    logger.info("oracle: solving min-load-shedding LP on net (%d childs, %d branches)",
                len(monee_net.childs), len(monee_net.branches))
    # ``exclude_unconnected_nodes=True`` is mandatory: without it monee's
    # solver assembles equations for the disconnected component too, and the
    # LP becomes structurally infeasible (mass-balance on a heat node with
    # no inflow + non-zero load, etc.).  When that happens monee still
    # returns — it just leaves ``regulation`` at the constructor default of
    # 1.0, which the metric then reads as "everything served".  See the
    # IIS analysis: residuals concentrate on ``node_276_eq_*``,
    # ``node_285_eq_*`` etc. — exactly the disconnected island.
    result = run_energy_flow_optimization(
        monee_net, prob, solver=solver, solver_name="gurobi",
        exclude_unconnected_nodes=True,
    )
    lp_success = bool(getattr(result, "success", True))
    logger.info("oracle: solve done (success=%s).", lp_success)

    # Read the metric off ``result.network`` — the solver's internal
    # COPY of the input network, where the LP's optimal ``regulation``
    # values live as Pyomo Vars (via :func:`_apply` / ``inject_vars``).
    # The input ``monee_net`` is the wrong source: monee's
    # ``persist_solution`` only back-writes via ``_copy_var_values``,
    # which checks ``isinstance(dst_attr, (Var, Intermediate))`` before
    # copying.  ``ChildModel.regulation`` starts as the Python float
    # ``1.0`` on the input network and is *promoted* to a Var only on
    # the LP copy — so the back-write silently skips every load's
    # optimal regulation.  The metric on the input network then reads
    # all loads at ``regulation = 1.0`` and reports ``served = full
    # demand`` even when the LP correctly shed lower-priority tiers
    # (2026-05-24 audit: deficit scenarios showed oracle reporting
    # ``served = 0.374 MW`` despite physical supply of ``0.268 MW``).
    # Slack ``p_mw`` DOES back-write correctly because it was a Var
    # on both networks — that's why ``_slack_budget_summary`` works
    # on either net, but the regulation-dependent metric does not.
    solved_net = getattr(result, "network", monee_net)
    behavior = _adapter_observe(solved_net)
    served = served_breakdown(solved_net, behavior, priorities=priorities)

    # The oracle has no time-series; constraint integral is degenerate.
    # We still report it as zero so the result.json schema is uniform.
    integral = {"electricity": 0.0, "gas": 0.0, "heat": 0.0}

    # Per-child regulation factors for downstream analysis — read off
    # the solved copy for the same reason as above.  Use ``model.values``
    # (which calls ``pyomo.value(var)`` per attribute) rather than
    # ``getattr(model, 'regulation')`` directly — on ``result.network``
    # ``regulation`` is a monee ``Var`` object after the LP-promoted
    # ``_apply`` step, and ``float(Var)`` raises ``TypeError``.
    regulations: dict[str, float] = {}
    for child in solved_net.childs:
        vals = child.model.values if hasattr(child.model, "values") else {}
        regulations[f"child-{child.id}"] = float(vals.get("regulation", 1.0))

    slack_summary = _slack_budget_summary(solved_net)

    return {
        "served": served,
        "constraint_violation_integral": integral,
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
    """Stable, hashable key for the baseline-LP cache.

    Same grid + scenario + priorities → identical LP → identical result;
    every task in a campaign rebuilds and re-solves redundantly without
    this cache.  JSON with sorted keys keeps the key stable across
    re-orderings of equivalent dicts.
    """
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
    """Solve the no-failure minimum-load-shedding LP on a freshly built
    grid and return the resulting ``served_breakdown``.

    This is the "pre-failure baseline" used to compute restoration
    ratios in :func:`experiment.eval.metrics.restoration_breakdown` —
    the optimum operating point the network can reach with no
    contingencies, which sets the upper bound for any restoration
    metric.  Scenarios that mutate the network (e.g.\ ``cold_day``
    scaling heat loads) are applied here too, so the baseline reflects
    the same demand profile the post-failure runs see.

    Building the grid fresh is necessary because the LP mutates the
    network's variable state; we cannot reuse the same instance for
    both the baseline and the post-failure run without confusing the
    Pyomo solver.

    Result cached in-process by (grid, scenario, priorities): every
    task in a campaign with identical inputs reuses the prior LP
    solve, saving the rebuild + solve wallclock.  Cache hits return
    a deep copy so callers can mutate freely.
    """
    cache_key = _baseline_cache_key(grid_name, scenario, priorities)
    cached = _BASELINE_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    if grid_name not in GRIDS:
        raise SystemExit(f"Unknown grid {grid_name!r}")
    # Factory already applies MISOCP + McCormick.
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
                k: scenario[k]
                for k in ("gen_scale", "load_scale")
                if k in scenario
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
            # Baseline LP must use the same islanding configuration as
            # the simulation-time LP, otherwise the no-failure baseline
            # would be computed without islanding and overstate the
            # post-failure restoration loss.  Mirrors the runner's
            # ``_apply_scenario`` microgrid branch.
            carriers = scenario.get(
                "carriers", ("electricity", "water", "gas")
            )
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
    # Strip ``backup=True`` from every branch on the local copy.  The
    # baseline LP solves the no-failure case, so any backup tie-line
    # added by :func:`add_backup_lines` would stay open anyway.  Leaving
    # the flag set causes ``create_min_load_shedding_problem`` →
    # ``controllable_backup_lines`` to turn ``on_off`` into a binary
    # decision variable, which makes the gas / heat node-balance
    # constraints (``mass_flow × on_off``) bilinear — Pyomo's
    # shell-gurobi LP writer then refuses with "node_X_eq_K contains
    # nonlinear terms that cannot be written to LP format".  ``fresh``
    # is a throwaway network so we don't need to restore the flag.
    for branch in fresh.branches:
        if getattr(branch.model, "backup", False):
            branch.model.backup = False
    out = run_oracle(fresh, [], solver=solver, priorities=priorities)
    served = out["served"]
    # Stash in cache for sibling tasks with identical inputs.
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

    # Slack budget compliance: with the LP envelope now tightened to
    # ±budget in run_oracle, a successful solve necessarily satisfies
    # the budget by construction.  Surface it via the same ``claims``
    # shape SCARE uses so the aggregator can compare apples to apples.
    slack_summary = out.get("slack_budget_summary", {})
    any_violation = any(
        bool(d.get("violated")) for d in slack_summary.values()
    )
    slack_claim = {
        "passed": (not any_violation) and bool(out.get("lp_success", True)),
        "detail": {
            "per_slack": slack_summary,
            "n_violations": sum(1 for d in slack_summary.values() if d.get("violated")),
            "enforced_at_lp": True,
        },
    }

    return {
        "task": task_meta,
        "wallclock_s": wallclock_s,
        "completed": True,
        "sim_time_final": 0.0,                # one-shot — no sim trajectory
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
            "time_to_stabilise_s": 0.0,
            "regulates_total": 0,
            "regulates_by_reason": {},
            "restoration": restoration,
            "oracle_lp_success": out.get("lp_success", True),
            "slack_budget_summary": slack_summary,
        },
        "claims": {
            # Vacuous for the priority claim (oracle minimises shed
            # subject to LP constraints; there is no MAS-side priority
            # dispatch to invariant-check).  Other variants populate
            # this dict from claims.evaluate_task.
            "slack_budget_compliance": slack_claim,
        },
        "diary": {"invariant_holds": True},   # vacuous
        "events": {},
        "messages": {},
        "oracle_regulations": out["regulations"],
    }
