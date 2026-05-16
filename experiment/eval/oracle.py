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
    from monee.problem import create_min_load_shedding_problem

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

    prob = create_min_load_shedding_problem()
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
    """
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
        slack_budget_pct = scenario.get("slack_budget_pct")
        if slack_budget_pct is not None:
            from experiment.restoration import apply_slack_budget

            apply_slack_budget(fresh, float(slack_budget_pct))
    out = run_oracle(fresh, [], solver=solver, priorities=priorities)
    return out["served"]


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
