"""Per-task output artefacts: metrics extraction, timeseries / diagnostics
writers, JSON sanitisation, the exact-gas re-solve, and the stale-artefact scrub.
Canonical CSV row emission stays in experiment.eval.results (called by the runner).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mango.simulation.world import WorldRecording

from experiment.hpc.config import RuntimePlan
from scare.base.runtime import diagnostics


def _extract_metrics(world: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, rec in getattr(world, "data_collections", {}).items():
        if not isinstance(rec, WorldRecording):
            continue
        ts = list(rec.timeseries)
        t = list(rec.time)
        if not ts:
            continue
        arr = np.asarray(ts, dtype=float)
        metrics[f"{name}__last"] = float(arr[-1])
        metrics[f"{name}__min"] = float(arr.min())
        metrics[f"{name}__max"] = float(arr.max())
        metrics[f"{name}__mean"] = float(arr.mean())
        metrics[f"{name}__n_samples"] = int(arr.size)
        if len(t) >= 2:
            metrics[f"{name}__integral"] = float(
                np.trapz(arr, np.asarray(t, dtype=float))
            )

    metrics["messages_total"] = len(getattr(world, "recorded_messages", []) or [])
    clock = getattr(world, "clock", None)
    metrics["sim_time_final"] = float(getattr(clock, "time", 0.0))
    return metrics


def _write_timeseries(world: Any, path: Path) -> None:
    series_map: dict[str, pd.Series] = {}
    for name, rec in getattr(world, "data_collections", {}).items():
        if not isinstance(rec, WorldRecording):
            continue
        if not rec.timeseries:
            continue
        series_map[name] = pd.Series(
            rec.timeseries, index=pd.Index(rec.time, name="time_s")
        )
    if not series_map:
        return
    df = pd.concat(series_map, axis=1).sort_index()
    df.to_csv(path)


def _dump_diagnostics(path: Path) -> None:
    path.write_text(diagnostics.dump_recent() + "\n", encoding="utf-8")


def _json_sanitize(obj: Any) -> Any:
    """NaN/inf floats -> None so result.json is standard JSON (json.dumps
    otherwise emits bare ``NaN`` tokens strict parsers reject)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _exact_gas_solved_net(net: Any, logger: logging.Logger) -> Any:
    """Re-solve the final network state with exact (nonconvex MIQCQP) gas Weymouth.

    The relaxed Weymouth the live sim uses leaves gas pressure underdetermined
    at zero-flow (load-shed) junctions: the epigraph ``m² ≤ m_sq`` lets the
    solver inflate ``m_sq`` and park ``pressure_squared_pu`` at its box maximum,
    reporting a spurious ``pressure_pu = √3``. The exact formulation (the same
    ``GAS_NONCONVEX_MIQCQP`` the oracle uses) pins gas pressure so the constraint
    check reads physical values. Returns the solved result network, or the
    original ``net`` if the exact solve is unavailable/fails.
    """
    try:
        from monee import run_energy_flow
        from monee.model.formulation import GAS_NONCONVEX_MIQCQP_FORMULATION

        net.apply_formulation(GAS_NONCONVEX_MIQCQP_FORMULATION)
        res = run_energy_flow(net, solver="gurobi", exclude_unconnected_nodes=True)
        result_net = getattr(res, "network", None)
        if getattr(res, "success", False) and result_net is not None:
            return result_net
        logger.warning(
            "Exact-gas constraint re-solve did not succeed; "
            "keeping relaxed-Weymouth gas pressures."
        )
    except Exception:  # noqa: BLE001 — never let the constraint re-solve abort output
        logger.warning(
            "Exact-gas constraint re-solve failed; keeping relaxed-Weymouth "
            "gas pressures.",
            exc_info=True,
        )
    return net


def _failing_fatal_claims(
    claims: dict[str, Any] | None, plan: RuntimePlan
) -> list[str]:
    """Failed fatal claims — these escalate ``ok`` to ``claims_failed``. The
    fatal set is overridable per-campaign via ``plan.fatal_claims``.
    """
    if not claims:
        return []
    fatal_claims = tuple(
        getattr(plan, "fatal_claims", ("priority_invariant", "monotonic_progress"))
    )
    return [
        name
        for name in fatal_claims
        if name in claims and not claims[name].get("passed", True)
    ]


def _scrub_stale_artifacts(out_dir: Path) -> None:
    for name in _TASK_ARTIFACTS:
        try:
            (out_dir / name).unlink(missing_ok=True)
        except OSError:
            pass


_TASK_ARTIFACTS = (
    "status.json",
    "exception.json",
    "result.json",
    "failures.json",
    "diagnostics.txt",
    "infeasibility_snapshot.json",
    "slack_meta.json",
    "served.csv",
    "served_by_load.csv",
    "constraints_final.csv",
    "diary.csv",
    "events.csv",
    "messages.csv",
    "timeseries.csv",
    "trajectories.csv",
)
