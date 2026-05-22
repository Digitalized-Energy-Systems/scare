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

import logging
from typing import Any

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
    from monee import run_energy_flow_optimization
    # NB: ``create_min_load_shedding_problem`` is exposed only via
    # the submodule ``monee.problem`` — top-level ``monee`` re-imports
    # it but it gets filtered out of the public namespace.
    from monee.problem import (
        WEIGHT_DEMAND,
        create_min_load_shedding_problem,
    )

    from experiment.eval.metrics import served_breakdown

    _apply_failures(monee_net, failures)

    # GEKKO (monee's default) hits "Max Equation Length" on grids with
    # several hundred children because the LP objective is built as a
    # single huge ``minimize(...)`` expression.  Pyomo handles
    # arbitrarily large objectives — the rest of scare already uses it
    # for the energy-flow simulation.  ``solver_name`` matches what
    # ``mango_energy_environments`` uses at simulation time so we don't
    # require any additional binaries beyond what scare already needs.
    if solver is None:
        from monee.solver.pyo import PyomoSolver

        solver = PyomoSolver()

    # When ``priorities`` is supplied, attach a per-load weight
    # closure so the LP objective discriminates between tiers.
    # Without it the oracle minimises *total* unserved MW with
    # every load equal — defining a "priority-blind upper bound"
    # that compares unfairly against the priority-aware MAS.
    weight_for_load = _weight_for_load_factory(
        monee_net, priorities, base_demand_weight=WEIGHT_DEMAND,
    )
    prob = create_min_load_shedding_problem(
        weight_for_load=weight_for_load,
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

    behavior = _adapter_observe(monee_net)
    served = served_breakdown(monee_net, behavior, priorities=priorities)

    # The oracle has no time-series; constraint integral is degenerate.
    # We still report it as zero so the result.json schema is uniform.
    integral = {"electricity": 0.0, "gas": 0.0, "heat": 0.0}

    # Per-child regulation factors for downstream analysis.
    regulations: dict[str, float] = {}
    for child in monee_net.childs:
        regulations[f"child-{child.id}"] = float(
            getattr(child.model, "regulation", 1.0)
        )

    return {
        "served": served,
        "constraint_violation_integral": integral,
        "regulations": regulations,
        "lp_success": lp_success,
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
    import json
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
        import copy
        return copy.deepcopy(cached)

    from experiment.restoration import GRIDS

    if grid_name not in GRIDS:
        raise SystemExit(f"Unknown grid {grid_name!r}")
    # Factory already applies MISOCP + McCormick.
    fresh = GRIDS[grid_name]()
    if scenario:
        kind = scenario.get("kind", "clean")
        if kind == "cold_day":
            from experiment.restoration import apply_cold_day

            kwargs = {
                k: scenario[k]
                for k in ("supply_t_k", "heat_load_scale")
                if k in scenario
            }
            apply_cold_day(fresh, **kwargs)
        elif kind == "pv_peak":
            from experiment.restoration import apply_pv_peak

            kwargs = {
                k: scenario[k]
                for k in ("gen_scale", "load_scale")
                if k in scenario
            }
            apply_pv_peak(fresh, **kwargs)
        elif kind == "line_stress":
            from experiment.restoration import apply_line_stress

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
            from experiment.restoration import apply_microgrid_islanding

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
            from experiment.restoration import apply_slack_budget

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
    import copy
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
    from experiment.eval.metrics import restoration_breakdown

    out = run_oracle(monee_net, failures, solver=solver, priorities=priorities)
    served = out["served"]
    integral = out["constraint_violation_integral"]
    restoration = restoration_breakdown(served, baseline_served)

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
        },
        "diary": {"invariant_holds": True},   # vacuous
        "events": {},
        "messages": {},
        "oracle_regulations": out["regulations"],
    }
