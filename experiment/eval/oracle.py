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
    """Mark every failed branch as out of service before solving.

    monee's branch model exposes ``on_off`` for switchable branches and
    ``in_service`` for unconditional disconnection.  Try both — falling
    back to setting the attribute that exists.
    """
    for failure in failures:
        for branch_id in getattr(failure, "branch_ids", []):
            try:
                branch = monee_net.branch_by_id(branch_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "oracle: could not find branch %s to disable: %s",
                    branch_id, exc,
                )
                continue
            model = branch.model
            if hasattr(model, "on_off"):
                model.on_off = 0
            elif hasattr(model, "in_service"):
                model.in_service = False
            else:
                # Last resort: the branch can't be marked off — log and
                # continue, the LP will still try to find a solution
                # that respects the rest of the network.
                logger.warning(
                    "oracle: branch %s has no on_off/in_service handle",
                    branch_id,
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
    run_energy_flow_optimization(monee_net, prob, solver=solver, solver_name="gurobi")
    logger.info("oracle: solve done.")

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
    }


def compose_oracle_result(
    *,
    monee_net: Any,
    failures: list[Any],
    task_meta: dict[str, Any],
    wallclock_s: float,
    solver: str | None = None,
    priorities: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a result.json payload identical in shape to the scare
    composer so the aggregator can read both off the same schema."""
    out = run_oracle(monee_net, failures, solver=solver, priorities=priorities)
    served = out["served"]
    integral = out["constraint_violation_integral"]

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
        },
        "diary": {"invariant_holds": True},   # vacuous
        "events": {},
        "messages": {},
        "oracle_regulations": out["regulations"],
    }
