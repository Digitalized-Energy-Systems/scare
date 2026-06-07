"""Run a single restoration evaluation task end-to-end.

Invoked once per Slurm array task (``--task-id``, else ``$SLURM_ARRAY_TASK_ID``),
but runs identically without Slurm. Writes per-task artefacts under
``<campaign_dir>/tasks/<task_id>/``: config/failures/result/status JSON, run.log,
diagnostics.txt, exception.json (on error), and optional timeseries.csv.

Exit codes: 0 = ok, 2 = timeout, 1 = other error. status.json is always written.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import platform
import random
import signal
import sys
import time
import time as _time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mango.simulation.world import WorldRecording

from experiment.eval.claims import evaluate_task
from experiment.eval.oracle import compose_oracle_result, compute_baseline_served
from experiment.eval.results import (
    compose_result,
    write_constraints_final_csv,
    write_diary_csv,
    write_events_csv,
    write_messages_csv,
    write_result_json,
    write_served_by_load_csv,
    write_served_csv,
    write_slack_meta,
    write_trajectories_csv,
)
from experiment.hpc.config import (
    CAMPAIGN_LAYOUT,
    RuntimePlan,
    TaskSpec,
    task_dir,
)
from experiment.hpc.plan import read_manifest
from experiment.scenarios import (
    GRIDS,
    apply_cold_day,
    apply_line_stress,
    apply_microgrid_islanding,
    apply_pv_peak,
    apply_slack_budget,
    apply_temporal_extensions,
    assign_load_priorities,
)
from scare.base.config import RestorationConfiguration
from scare.base.runtime import diagnostics
from scare.base.runtime import diagnostics as _diag
from scare.base.runtime.infeasibility_capture import (
    arm_infeasibility_capture,
    disarm_infeasibility_capture,
)
from scare.base.runtime.solver_guard import install_solver_time_limit
from scare.base.util import create_failures
from scare.scenario.restoration import (
    _flush_pending_negotiations,
    create_restoration_scenario_world,
    start_restoration_simulation,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 2

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class _SolverFailureCounter(logging.Filter):
    """Count solver-status escalations to report per-task solver health.

    An infeasible solve fires as a pair (monee ERROR + pyomo.core WARNING),
    deduped within ``_DEDUPE_WINDOW_S``. Also catches Gurobi/Pyomo env
    strings so env issues are distinguishable from algorithm bugs.
    """

    _SOLVER_ERROR_MARKERS: tuple[str, ...] = (
        "GurobiError",
        "HostID mismatch",
        "License",  # Gurobi LicenseError
    )
    _DEDUPE_WINDOW_S: float = 1.0  # min spacing between distinct solves

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.infeasible_count = 0
        self.warning_count = 0
        self._last_infeasible_t: float = float("-inf")

    def _is_infeasible_msg(self, msg: str) -> bool:
        # monee.solver.pyo ERROR path.
        if "infeasible (status=" in msg or "Pyomo solve infeasible" in msg:
            return True
        # pyomo.core load_solutions WARNING path (both substrings, one record).
        if (
            "Loading a SolverResults object" in msg
            and "termination condition: infeasible" in msg
        ):
            return True
        return False

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.levelno < logging.WARNING:
            return True
        msg = record.getMessage()
        if self._is_infeasible_msg(msg):
            if record.created - self._last_infeasible_t >= self._DEDUPE_WINDOW_S:
                self.infeasible_count += 1
                self.count += 1
            self._last_infeasible_t = record.created
        elif "returned non-ok status" in msg:
            self.warning_count += 1
            self.count += 1
        elif any(marker in msg for marker in self._SOLVER_ERROR_MARKERS):
            # Gurobi env/license/host-id errors: still solver failures.
            self.warning_count += 1
            self.count += 1
        return True


def _setup_logging(log_path: Path) -> tuple[logging.FileHandler, _SolverFailureCounter]:
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Drop pre-existing handlers to avoid double-logging (e.g. earlier basicConfig).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    # Keep WARN+ on stderr so Slurm captures show-stoppers per array task.
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(stderr)

    # Suppress third-party DEBUG/INFO chatter (mango alone emits ~60k lines
    # per 30s sim). At package root so new submodules stay quiet; WARN+ surfaces.
    for noisy in (
        "pyomo",
        "gurobipy",
        "mango",
        "mango_energy_environments",
        "simbench",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    counter = _SolverFailureCounter()
    # Listen on both infeasibility emitters; the counter dedupes the pair.
    for logger_name in ("monee.solver.pyo", "pyomo.core"):
        logging.getLogger(logger_name).addFilter(counter)
    return handler, counter


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def ensure_deterministic_hashing(seed: str = "0") -> None:
    """Pin ``PYTHONHASHSEED`` for reproducibility, re-exec'ing once to apply it.

    Hash randomisation (fixed at interpreter start) varies set/frozenset order
    over agent-id strings per worker, flipping results for the same task. It can
    only be disabled before start, so set the env var and re-exec. A user-set
    value (other than ``"random"``) is respected; no-op once pinned.
    """
    current = os.environ.get("PYTHONHASHSEED")
    if current is not None and current != "random":
        return
    os.environ["PYTHONHASHSEED"] = seed
    # sys.orig_argv (3.10+) preserves the exact invocation for -m imports.
    os.execv(sys.executable, sys.orig_argv)


def _resolve_failures(monee_net: Any, plan: RuntimePlan, task: TaskSpec) -> list[Any]:
    """Draw the failure scenario. ``scenario`` may override sampling via
    ``failure_type`` (branch/generator/mixed) and ``generator_share`` (mixed).
    """
    scenario = task.scenario or {}
    failure_type = scenario.get("failure_type", "branch")
    kwargs = {}
    if "generator_share" in scenario:
        kwargs["generator_share"] = float(scenario["generator_share"])
    return create_failures(
        monee_net,
        failure_type,
        num_failures=task.n_failures,
        delay_s_max=plan.failure_delay_s_max,
        **kwargs,
    )


def _serialize_failures(failures: list[Any]) -> list[dict[str, Any]]:
    """JSON-serialise the failure list for ``failures.json``.

    Branch failures carry ``branch_ids``; generator ones carry ``custom_id``
    (the callable isn't JSON-able, but its id names the deactivated component).
    """
    out = []
    for f in failures:
        rec: dict[str, Any] = {
            "delay_s": float(f.delay_s),
            "branch_ids": [list(bid) for bid in f.branch_ids],
        }
        node_ids = list(getattr(f, "node_ids", []) or [])
        if node_ids:
            rec["node_ids"] = node_ids
        custom_id = getattr(f, "custom_id", None)
        if custom_id is not None:
            rec["custom_id"] = str(custom_id)
        out.append(rec)
    return out


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
    path.write_text(diagnostics.dump_recent() + "\n")


async def _run_simulation(
    plan: RuntimePlan,
    task: TaskSpec,
    logger: logging.Logger,
    *,
    out_dir: Path | None = None,
):
    """Build and run one scare-variant simulation.

    Returns (world, failures, net). When ``out_dir`` is given, the one-shot
    infeasibility capture writes ``infeasibility_snapshot.json`` there.
    """
    if task.grid not in GRIDS:
        raise SystemExit(f"Unknown grid {task.grid!r}; available: {sorted(GRIDS)}")

    # Cap each MISOCP under the task budget so asyncio.wait_for can preempt at
    # the next await (it can't interrupt a sync solve). Floor 30s, cap 60s.
    per_solve_cap = max(30.0, min(plan.task_timeout_s / 4.0, 60.0))
    install_solver_time_limit(per_solve_cap)

    factory = GRIDS[task.grid]
    logger.info("Building network for grid=%s", task.grid)
    # Factory applies MISOCP and leaves DHS nonlinear (McCormick-DHS is off
    # live — it can hit envelope infeasibilities; the oracle re-enables it).
    net = factory()
    _apply_scenario(net, task, logger)

    failures = _resolve_failures(net, plan, task)
    logger.info(
        "Resolved %d failure(s) for seed=%d: %s",
        len(failures),
        task.seed,
        [f.branch_ids for f in failures],
    )

    cfg = _config_from_task(task)
    logger.info(
        "Variant=%s ablation=%s sweep=%s",
        task.variant,
        task.ablation or {},
        task.sweep or {},
    )

    priorities = _resolve_priorities(net, task, logger)
    # Stash on the net so post-sim metric writers can pick it up.
    net._scare_priorities = priorities

    world = create_restoration_scenario_world(
        net,
        priorities=priorities,
        simulation_duration_s=plan.simulation_duration_s,
        config=cfg,
    )
    # One-shot capture: first failed solve drops a snapshot; disarmed
    # after the task so the next one on this worker gets a fresh window.
    if out_dir is not None:
        snapshot_path = out_dir / "infeasibility_snapshot.json"
        arm_infeasibility_capture(
            world.environment.behavior,
            snapshot_path,
            clock=world.clock,
        )
    logger.info(
        "Running simulation for %.1f s (timeout=%.0f s)",
        plan.simulation_duration_s,
        plan.task_timeout_s,
    )
    try:
        await asyncio.wait_for(
            start_restoration_simulation(world, failures, plan.simulation_duration_s),
            timeout=plan.task_timeout_s,
        )
    except asyncio.TimeoutError:
        # Timeout cancels the sim before its own flush, losing in-flight
        # gossips. Drain here so started == Σ terminals still holds.
        try:
            _flush_pending_negotiations(world)
        except Exception as flush_exc:  # noqa: BLE001
            logger.warning("flush_pending after timeout failed: %s", flush_exc)
        raise
    logger.info("Simulation finished.")
    return world, failures, net


def _run_oracle(
    plan: RuntimePlan,
    task: TaskSpec,
    logger: logging.Logger,
    *,
    baseline_served: dict[str, Any] | None = None,
):
    """Solve monee's minimal-load-shedding LP on the same failures the scare
    variant would see. Returns (net, failures, result_payload) — no agents.
    """
    if task.grid not in GRIDS:
        raise SystemExit(f"Unknown grid {task.grid!r}; available: {sorted(GRIDS)}")
    factory = GRIDS[task.grid]
    logger.info("Building network for grid=%s (oracle)", task.grid)
    # compose_oracle_result adds the McCormick-DHS linearisation for the LP.
    net = factory()
    _apply_scenario(net, task, logger)
    failures = _resolve_failures(net, plan, task)
    logger.info(
        "Oracle: %d failure(s) for seed=%d: %s",
        len(failures),
        task.seed,
        [f.branch_ids for f in failures],
    )
    priorities = _resolve_priorities(net, task, logger)
    started = _time.monotonic()
    payload = compose_oracle_result(
        monee_net=net,
        failures=failures,
        task_meta=task.to_dict(),
        wallclock_s=0.0,
        priorities=priorities,
        baseline_served=baseline_served,
    )
    payload["wallclock_s"] = round(_time.monotonic() - started, 3)
    return net, failures, payload


def _resolve_priorities(
    net: Any, task: TaskSpec, logger: logging.Logger
) -> dict[str, int] | None:
    """Build the per-load priority dict from ``scenario["priority_assignment"]``.

    Default ``"skewed"`` exercises the priority machinery (without it every load
    falls back to tier 1). Recognised: uniform/skewed/by_capacity/all_one.
    """
    scenario = task.scenario or {}
    distribution = scenario.get("priority_assignment", "skewed")
    priorities = assign_load_priorities(net, seed=task.seed, distribution=distribution)
    counts: dict[int, int] = {}
    for tier in priorities.values():
        counts[tier] = counts.get(tier, 0) + 1
    logger.info(
        "Priority assignment %r: %d loads, tier histogram=%s",
        distribution,
        len(priorities),
        sorted(counts.items()),
    )
    return priorities


def _apply_scenario(net: Any, task: TaskSpec, logger: logging.Logger) -> None:
    """Apply scenario-kind mutations from ``task.scenario["kind"]``.

    Kinds: ``clean`` (no-op), ``cold_day`` (colder slack + higher heat load),
    ``pv_peak`` (over-voltage: more gen, less load), ``line_stress`` (thermal:
    more load, lower ampacity), ``microgrid``. Each reads its own tunables from
    the scenario dict. Unknown kinds pass through (warned).

    Orthogonal to ``kind``: ``slack_budget_pct`` is applied after the kind
    mutation (per-scenario operator policy, not a baked-in grid attribute).
    """
    scenario = task.scenario or {}
    kind = scenario.get("kind", "clean")
    _KNOWN_KINDS = {"clean", "cold_day", "pv_peak", "line_stress", "microgrid"}
    if kind not in _KNOWN_KINDS:
        # Warn on unknown kinds so a typo doesn't silently produce a clean run.
        logger.warning(
            "Unknown scenario kind %r (known: %s) — falling through to "
            "no-mutation behaviour.  Check the campaign config for typos.",
            kind,
            sorted(_KNOWN_KINDS),
        )
    if kind == "cold_day":
        kwargs = {
            k: scenario[k] for k in ("supply_t_k", "heat_load_scale") if k in scenario
        }
        apply_cold_day(net, **kwargs)
        logger.info("Applied cold_day scenario: %s", kwargs or "<defaults>")
    elif kind == "pv_peak":
        kwargs = {k: scenario[k] for k in ("gen_scale", "load_scale") if k in scenario}
        apply_pv_peak(net, **kwargs)
        logger.info("Applied pv_peak scenario: %s", kwargs or "<defaults>")
    elif kind == "line_stress":
        kwargs = {
            k: scenario[k]
            for k in ("load_scale", "ampacity_scale", "affect_branch_fraction")
            if k in scenario
        }
        apply_line_stress(net, **kwargs)
        logger.info("Applied line_stress scenario: %s", kwargs or "<defaults>")
    elif kind == "microgrid":
        # Enable islanding AND promote generator-class children to GridForming*
        # so sub-islands have reference units (else it's a no-op on simbench).
        carriers = scenario.get("carriers", ("electricity", "water", "gas"))
        promote_all = bool(scenario.get("promote_all_generators", True))
        former_aids = tuple(scenario.get("grid_former_aids", ()))
        counts = apply_microgrid_islanding(
            net,
            carriers=carriers,
            promote_all_generators=promote_all,
            grid_former_aids=former_aids,
        )
        logger.info(
            "Applied microgrid scenario: carriers=%s promote_all=%s promoted=%s",
            list(carriers),
            promote_all,
            counts,
        )

    # Slack-budget policy: widens LP Var bounds to ±10·budget and stamps the
    # budget as the slack agents' rating the MAS drives toward.
    slack_budget_pct = scenario.get("slack_budget_pct")
    if slack_budget_pct is not None:
        apply_slack_budget(net, float(slack_budget_pct))
        logger.info(
            "Applied slack_budget_pct=%s (per-scenario operator policy)",
            slack_budget_pct,
        )

    # Temporal-storage extensions (GasLinepack / LumpedThermalCapacitance).
    # Single-step energyflow only adds vars — a compat smoke test, not a benchmark.
    linepack = bool(scenario.get("linepack", False))
    ltc = bool(scenario.get("ltc", False))
    if linepack or ltc:
        ltc_t_init = scenario.get("ltc_default_t_init")
        ext_counts = apply_temporal_extensions(
            net,
            linepack=linepack,
            ltc=ltc,
            ltc_default_t_init=ltc_t_init,
        )
        logger.info(
            "Applied temporal extensions: linepack=%s ltc=%s counts=%s",
            linepack,
            ltc,
            ext_counts,
        )


def _config_from_task(task: TaskSpec):
    """Compose a ``RestorationConfiguration`` from the task's variant,
    ablation, and sweep dicts. Variant maps to a base preset; ablation /
    sweep are field overrides applied on top.
    """
    if task.variant == "single_level":
        base = RestorationConfiguration(
            enable_holonic=False,
            enable_cp_admm=False,
        )
    elif task.variant == "component_level":
        # One community per connected component per sector — global L1, no
        # hierarchy. CPs join bridged communities and reconcile via
        # MultiCommunityCPRole (EMA + deadband + cooldown). multihop_constraint
        # MUST stay off: the partition collapses each sector into one group, so
        # forwarding fans out O(N^2) and OOM-kills the worker; direct neighbours
        # already give the global picture.
        base = RestorationConfiguration(
            enable_holonic=False,
            enable_cp_admm=False,
            cps_join_communities=True,
            community_partition_method="connected_component",
            enable_multihop_constraint=False,
        )
    else:
        base = RestorationConfiguration()
    overrides: dict = {}
    overrides.update(task.ablation or {})
    overrides.update(task.sweep or {})
    if not overrides:
        return base
    valid = {
        f.name for f in __import__("dataclasses", fromlist=["fields"]).fields(base)
    }
    clean = {k: v for k, v in overrides.items() if k in valid}
    skipped = set(overrides) - valid
    if skipped:
        logging.getLogger("experiment.hpc.runner").warning(
            "Ignoring unknown config overrides: %s",
            sorted(skipped),
        )
    return replace(base, **clean)


def _compute_baseline(task: TaskSpec, logger: logging.Logger) -> dict[str, Any] | None:
    """Solve the no-failure LP so restoration can be expressed as a ratio of
    pre-failure served. Returns ``None`` (logged) on failure so the task proceeds.
    """
    if task.grid not in GRIDS:
        return None
    try:
        # Throwaway net to enumerate loads, released before the heavy phase
        # so it doesn't double peak RAM (~1-3 GB on CP-heavy grids).
        base_net = GRIDS[task.grid]()
        _apply_scenario(base_net, task, logger)
        base_priorities = _resolve_priorities(base_net, task, logger)
        baseline_served = compute_baseline_served(
            task.grid,
            scenario=task.scenario or {},
            priorities=base_priorities,
        )
        logger.info(
            "Pre-failure baseline: served=%.4f / demand=%.4f (PWSF=%.4f)",
            baseline_served.get("priority_weighted_served", 0.0),
            baseline_served.get("priority_weighted_demand", 0.0),
            baseline_served.get("priority_weighted_fraction", 0.0),
        )
        del base_net, base_priorities
        gc.collect()
        return baseline_served
    except Exception as exc:  # noqa: BLE001
        logger.warning("Baseline LP failed (continuing without it): %s", exc)
        return None


def _write_oracle_outputs(
    out_dir: Path,
    plan: RuntimePlan,
    task: TaskSpec,
    logger: logging.Logger,
    baseline_served: dict[str, Any] | None,
) -> None:
    """Run the centralized oracle LP and write its result/slack/failures."""
    net, failures, oracle_metrics = _run_oracle(
        plan, task, logger, baseline_served=baseline_served
    )
    (out_dir / "failures.json").write_text(
        json.dumps(_serialize_failures(failures), indent=2)
    )
    (out_dir / "result.json").write_text(
        json.dumps(oracle_metrics, indent=2, sort_keys=True, default=str)
    )
    write_slack_meta(out_dir / "slack_meta.json", net)


def _write_simulation_outputs(
    out_dir: Path,
    plan: RuntimePlan,
    task: TaskSpec,
    logger: logging.Logger,
    started: float,
    baseline_served: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Run the MAS simulation, write every per-task artefact, and return the
    claims-validation payload (or ``None`` if validation itself failed).
    """
    world, failures, net = asyncio.run(
        _run_simulation(plan, task, logger, out_dir=out_dir)
    )
    (out_dir / "failures.json").write_text(
        json.dumps(_serialize_failures(failures), indent=2)
    )
    behavior = world.environment.behavior
    # Force a fresh energy flow so observers report post-action state, not the
    # cooldown-cached solve (else the served breakdown is stale).
    try:
        behavior.flush_energy_flow()
    except AttributeError:
        logger.debug("Behavior has no flush_energy_flow() — skipping")
    priorities = getattr(net, "_scare_priorities", None)
    payload = compose_result(
        world=world,
        monee_net=net,
        behavior=behavior,
        task_meta=task.to_dict(),
        wallclock_s=time.monotonic() - started,
        completed=True,
        extra_metrics={"legacy_metrics": _extract_metrics(world)},
        priorities=priorities,
        baseline_served=baseline_served,
    )
    write_result_json(out_dir / "result.json", payload)
    write_served_csv(out_dir / "served.csv", net, behavior, priorities=priorities)
    write_served_by_load_csv(
        out_dir / "served_by_load.csv",
        net,
        behavior,
        priorities=priorities,
    )
    write_constraints_final_csv(out_dir / "constraints_final.csv", net)
    write_diary_csv(out_dir / "diary.csv")
    write_events_csv(out_dir / "events.csv")
    write_slack_meta(out_dir / "slack_meta.json", net)
    cfg = _config_from_task(task)
    if cfg.record_messages:
        write_messages_csv(out_dir / "messages.csv", world)
    if plan.write_timeseries:
        try:
            _write_timeseries(world, out_dir / "timeseries.csv")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write timeseries.csv: %s", exc)
    if getattr(plan, "write_trajectories", False):
        try:
            write_trajectories_csv(out_dir / "trajectories.csv")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write trajectories.csv: %s", exc)
    # Claims validation: fold pass/fail into result.json for the aggregator.
    try:
        claims = evaluate_task(out_dir)
        payload["claims"] = claims
        write_result_json(out_dir / "result.json", payload)
        return claims
    except Exception as exc:  # noqa: BLE001
        logger.warning("Claims validation failed: %s", exc)
        return None


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


def run_task(campaign_dir: Path, task_id: int, *, reraise: bool = False) -> int:
    plan = RuntimePlan.from_config_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
    tasks = read_manifest(campaign_dir)
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"task_id {task_id} out of range [0, {len(tasks)})")
    task = tasks[task_id]

    out_dir = task_dir(campaign_dir, task.task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(task.to_dict(), indent=2, sort_keys=True)
    )

    handler, solver_counter = _setup_logging(out_dir / "run.log")
    logger = logging.getLogger("experiment.hpc.runner")
    logger.info(
        "Task %d  grid=%s  seed=%d  n_failures=%d",
        task.task_id,
        task.grid,
        task.seed,
        task.n_failures,
    )
    logger.info(
        "Slurm: job=%s array_task=%s host=%s",
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_ARRAY_TASK_ID"),
        os.environ.get("HOSTNAME") or platform.node(),
    )

    _seed_everything(task.seed)
    # Reset per-run diagnostics: workers reuse the process, so logs/buffers
    # carry over without arm().
    _diag.arm()
    # Per-aid trajectory logging (off by default; the log can grow large).
    _diag.set_trajectory_logging(getattr(plan, "write_trajectories", False))
    # Drop stale exception.json from a prior failed run on this worker.
    stale_exc = out_dir / "exception.json"
    if stale_exc.exists():
        try:
            stale_exc.unlink()
        except OSError:
            pass
    started = time.monotonic()
    status: dict[str, Any] = {
        "task_id": task.task_id,
        "grid": task.grid,
        "seed": task.seed,
        "n_failures": task.n_failures,
        "status": "error",
        "duration_s": 0.0,
        "solver_failures": 0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    exit_code = EXIT_ERROR
    claims: dict[str, Any] | None = None

    # Convert SIGTERM to an exception so ``finally`` still writes status.json
    # (SIGKILL is uncatchable, but we honour the SIGTERM grace window).
    _prev_term = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt("SIGTERM received — emergency shutdown")

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        # Not main thread or otherwise restricted — best-effort only.
        _prev_term = None

    baseline_served = _compute_baseline(task, logger)

    try:
        if task.variant == "oracle":
            _write_oracle_outputs(out_dir, plan, task, logger, baseline_served)
        else:
            claims = _write_simulation_outputs(
                out_dir, plan, task, logger, started, baseline_served
            )
        # Escalate ``ok`` to ``claims_failed`` when a fatal claim failed.
        failing = _failing_fatal_claims(claims, plan)
        if failing:
            logger.warning("Fatal claims failed: %s", failing)
            status["status"] = "claims_failed"
            status["failing_claims"] = failing
        else:
            status["status"] = "ok"
        exit_code = EXIT_OK

    except asyncio.TimeoutError:
        logger.error("Task timed out after %.0f s", plan.task_timeout_s)
        status["status"] = "timeout"
        (out_dir / "exception.json").write_text(
            json.dumps(
                {
                    "type": "TimeoutError",
                    "message": f"Exceeded plan.task_timeout_s={plan.task_timeout_s}",
                },
                indent=2,
            )
        )
        exit_code = EXIT_TIMEOUT

    except KeyboardInterrupt as exc:
        # SIGTERM/Ctrl-C — record ``killed`` to distinguish from an error.
        logger.error("Task killed: %s", exc)
        status["status"] = "killed"
        (out_dir / "exception.json").write_text(
            json.dumps(
                {
                    "type": "KeyboardInterrupt",
                    "message": str(exc),
                },
                indent=2,
            )
        )
        exit_code = EXIT_TIMEOUT

    except Exception as exc:  # noqa: BLE001
        logger.exception("Task failed: %s", exc)
        (out_dir / "exception.json").write_text(
            json.dumps(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        if reraise:
            raise
        exit_code = EXIT_ERROR

    finally:
        # Release the capture window so the next task on this worker is fresh.
        try:
            disarm_infeasibility_capture()
        except Exception:  # noqa: BLE001
            pass
        status["duration_s"] = round(time.monotonic() - started, 3)
        status["solver_failures"] = solver_counter.count
        status["solver_infeasibilities"] = solver_counter.infeasible_count
        status["solver_warnings"] = solver_counter.warning_count
        (out_dir / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True)
        )
        try:
            _dump_diagnostics(out_dir / "diagnostics.txt")
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "Status=%s duration=%.1fs solver_failures=%d exit=%d",
            status["status"],
            status["duration_s"],
            status["solver_failures"],
            exit_code,
        )
        logging.getLogger().removeHandler(handler)
        handler.close()
        if _prev_term is not None:
            try:
                signal.signal(signal.SIGTERM, _prev_term)
            except (ValueError, OSError):
                pass

    return exit_code


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--campaign-dir", required=True, type=Path)
    p.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Index into manifest.jsonl. If omitted, falls back to $SLURM_ARRAY_TASK_ID.",
    )
    p.add_argument(
        "--reraise",
        action="store_true",
        help="Re-raise exceptions for debugging instead of writing exception.json",
    )
    return p.parse_args()


def _resolve_task_id(args: argparse.Namespace) -> int:
    if args.task_id is not None:
        return args.task_id
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise SystemExit(
            "Pass --task-id N or run under Slurm with SLURM_ARRAY_TASK_ID set."
        )
    return int(env)


def main() -> None:
    ensure_deterministic_hashing()
    args = _parse_args()
    sys.exit(
        run_task(
            args.campaign_dir.resolve(), _resolve_task_id(args), reraise=args.reraise
        )
    )


if __name__ == "__main__":
    main()
