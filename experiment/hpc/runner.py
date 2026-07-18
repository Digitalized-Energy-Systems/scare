"""Run a single restoration evaluation task end-to-end.

Invoked once per Slurm array task (``--task-id``, else ``$SLURM_ARRAY_TASK_ID``),
but runs identically without Slurm. Writes per-task artefacts under
``<campaign_dir>/tasks/<task_id>/``: config/failures/result/status JSON, run.log,
diagnostics.txt, exception.json (on error), and optional timeseries.csv.

Exit codes: 0 = ok, 2 = timeout or killed (SIGTERM/Ctrl-C), 1 = other error.
status.json is always written.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import math
import os
import platform
import random
import signal
import subprocess
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
from experiment.eval.oracle import (
    apply_oracle_heat_linearisation,
    baseline_regulations,
    compose_oracle_result,
    compute_baseline_served,
    oracle_solver_for_task,
)
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
from scare.base.model import set_sector_timescale
from scare.base.runtime import diagnostics
from scare.base.runtime import diagnostics as _diag
from scare.base.runtime.infeasibility_capture import (
    arm_infeasibility_capture,
    disarm_infeasibility_capture,
)
from scare.base.runtime.solver_guard import install_solver_time_limit
from scare.base.runtime.trace import SimTimeLogFilter
from scare.scenario.failure_sampling import create_failures
from scare.scenario.restoration import (
    _flush_pending_negotiations,
    create_restoration_scenario_world,
    start_restoration_simulation,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 2

LOG_FORMAT = "%(asctime)s t=%(sim_t)8.3f %(levelname)s [%(name)s] %(message)s"


class _SolverFailureCounter(logging.Filter):
    """Count solver-status escalations for per-task health.

    An infeasible solve fires as a monee-ERROR + pyomo-WARNING pair, deduped
    within ``_DEDUPE_WINDOW_S``; Gurobi/Pyomo env strings are also caught to
    separate env issues from algorithm bugs.
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
        # monee.solver.gurobipy (islanding backend) + monee.simulation.stepper
        # skip-mode absorption — without these, stepper-path failures leave
        # solver_failures at 0.
        if "Gurobi solve failed without a usable solution" in msg:
            return True
        if "Stepper step" in msg and "failed" in msg:
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


# Solver-failure emitters the per-task counter attaches to; detached in run_task's
# finally since run_local reuses worker processes across tasks.
_SOLVER_FAILURE_LOGGERS: tuple[str, ...] = (
    "monee.solver.pyo",
    "pyomo.core",
    "monee.solver.gurobipy",
    "monee.simulation.stepper",
)


def _setup_logging(log_path: Path) -> tuple[logging.FileHandler, _SolverFailureCounter]:
    # Stamp every record with the current sim time (record.sim_t) so the
    # LOG_FORMAT's t=... field is always populated, including third-party logs.
    sim_time_filter = SimTimeLogFilter()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(sim_time_filter)

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
    stderr.addFilter(sim_time_filter)
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
    # Listen on every infeasibility emitter; the counter dedupes pairs (an
    # absorbed stepper failure fires as gurobipy/pyo ERROR + stepper WARNING
    # within the dedupe window).
    for logger_name in _SOLVER_FAILURE_LOGGERS:
        logging.getLogger(logger_name).addFilter(counter)
    return handler, counter


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))


def ensure_deterministic_hashing(seed: str = "0") -> None:
    """Pin ``PYTHONHASHSEED`` for reproducibility, re-exec'ing once to apply it.

    Hash randomisation (fixed at interpreter start) varies set/frozenset order
    over agent-id strings, flipping same-task results; it can only be disabled
    pre-start, so set the env var and re-exec. A user value other than
    ``"random"`` is respected.
    """
    current = os.environ.get("PYTHONHASHSEED")
    if current is not None and current != "random":
        return
    os.environ["PYTHONHASHSEED"] = seed
    # sys.orig_argv (3.10+) preserves the exact invocation for -m imports.
    if sys.platform == "win32":
        # os.exec* on Windows joins argv without quoting, splitting paths
        # containing spaces; run a properly-quoted child instead.
        proc = subprocess.run([sys.executable, *sys.orig_argv[1:]])
        sys.exit(proc.returncode)
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
    if scenario.get("generator_carriers"):
        kwargs["generator_carriers"] = tuple(scenario["generator_carriers"])
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
    path.write_text(diagnostics.dump_recent() + "\n", encoding="utf-8")


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
    # the next await (it can't interrupt a sync solve). Floor 30s, cap 300s.
    per_solve_cap = max(30.0, min(plan.task_timeout_s / 4.0, 300.0))
    install_solver_time_limit(per_solve_cap)

    factory = GRIDS[task.grid]
    logger.info("Building network for grid=%s", task.grid)
    # Factory applies MISOCP and leaves DHS nonlinear (McCormick-DHS is off
    # live — it can hit envelope infeasibilities; the oracle re-enables it).
    net = factory()
    _apply_scenario(net, task, logger)

    # Scale per-solve cap to ~1s/node: on a ~40-node grid the flat 300s ceiling
    # let 18 infeasible solves sum to the full task_timeout with no signal. Large
    # grids stay 300s; small ones drop to ~40s. Floor 30s (async-preempt granularity).
    n_nodes = sum(1 for _ in net.nodes)
    scaled_cap = max(30.0, min(per_solve_cap, float(n_nodes)))
    if scaled_cap < per_solve_cap:
        per_solve_cap = scaled_cap
        install_solver_time_limit(per_solve_cap)
        logger.info(
            "Scaled per-solve cap to %.0fs (%d nodes, grid=%s)",
            per_solve_cap,
            n_nodes,
            task.grid,
        )

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

    scenario = task.scenario or {}
    # Per-experiment horizon override: a longer sim accrues more receding-physics
    # steps at the same dt/agent budget. Not physics_time_scale, which would
    # desync sim-clock message delays from the physics dt — a duration knob only.
    sim_duration_s = float(
        scenario.get("simulation_duration_s") or plan.simulation_duration_s
    )
    world = create_restoration_scenario_world(
        net,
        priorities=priorities,
        simulation_duration_s=sim_duration_s,
        config=cfg,
        physics_time_scale=scenario.get("physics_time_scale"),
        physics_interval_s=scenario.get("physics_interval_s"),
        physics_solve_time_limit_s=per_solve_cap,
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
        sim_duration_s,
        plan.task_timeout_s,
    )
    try:
        await asyncio.wait_for(
            start_restoration_simulation(world, failures, sim_duration_s),
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
    out_dir: Path | None = None,
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
    scenario = task.scenario or {}
    # Temporal scenarios pick their own per-step solver inside the oracle.
    temporal = bool(scenario.get("linepack") or scenario.get("ltc"))
    solver = None if temporal else oracle_solver_for_task(task.grid, scenario)
    warm_regs = baseline_regulations(
        task.grid, scenario=scenario, priorities=priorities
    )
    if warm_regs:
        logger.info("Oracle: warm-starting from baseline incumbent (%d regs)",
                    len(warm_regs))
    started = _time.monotonic()
    payload = compose_oracle_result(
        monee_net=net,
        failures=failures,
        task_meta=task.to_dict(),
        wallclock_s=0.0,
        solver=solver,
        priorities=priorities,
        baseline_served=baseline_served,
        out_dir=out_dir,
        simulation_duration_s=float(
            scenario.get("simulation_duration_s") or plan.simulation_duration_s
        ),
        warm_start_regulations=warm_regs,
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
    # Process-local per-task reset+apply of the MAS agent time-scales. Always
    # called (None resets to defaults) so a slow-time-scale temporal task cannot
    # leak into the next task on a reused worker process. Must run BEFORE the
    # world/agents are built; the oracle path ignores it (no MAS).
    set_sector_timescale(scenario.get("sector_timescale"))
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

    # Heat McCormick linearisation: mandatory for islanding (the gurobipy backend
    # can't ingest nonlinear DHS balance), else opt-in via ``heat_mccormick`` so a
    # paired A/B arm runs identical heat physics (else clean-vs-microgrid confounds
    # islanding with the relaxation).
    if kind == "microgrid" or scenario.get("heat_mccormick"):
        apply_oracle_heat_linearisation(net)
        logger.info("Applied McCormick heat linearisation (oracle settings)")

    # Slack-budget policy: widens LP Var bounds to ±10·budget and stamps the
    # budget as the slack agents' rating the MAS drives toward. Carriers named
    # in slack_hard_cap_carriers instead get a HARD import cap at the budget
    # (deficit must draw storage or shed) — a storage lever, feasible only where
    # local generation + a stored buffer can cover the capped share.
    slack_budget_pct = scenario.get("slack_budget_pct")
    if slack_budget_pct is not None:
        hard_cap = scenario.get("slack_hard_cap_carriers")
        apply_slack_budget(
            net,
            float(slack_budget_pct),
            hard_cap_carriers=tuple(hard_cap) if hard_cap else None,
        )
        logger.info(
            "Applied slack_budget_pct=%s hard_cap=%s (per-scenario operator policy)",
            slack_budget_pct,
            list(hard_cap) if hard_cap else [],
        )

    # Temporal-storage extensions (GasLinepack / LumpedThermalCapacitance).
    # Single-step energyflow only adds vars — a compat smoke test, not a benchmark.
    linepack = bool(scenario.get("linepack", False))
    ltc = bool(scenario.get("ltc", False))
    if linepack or ltc:
        ltc_t_init = scenario.get("ltc_default_t_init")
        # Default the first-step steady-state anchor ON: without it every LTC
        # junction cold-starts at the t_pu per-unit default instead of the real
        # supply temperature, faking a temperature collapse. Scenarios can still
        # force it off with ltc_first_step_steady_state=false.
        ltc_steady_start = bool(scenario.get("ltc_first_step_steady_state", True))
        ext_counts = apply_temporal_extensions(
            net,
            linepack=linepack,
            ltc=ltc,
            ltc_default_t_init=ltc_t_init,
            ltc_first_step_steady_state=ltc_steady_start,
        )
        logger.info(
            "Applied temporal extensions: linepack=%s ltc=%s steady_start=%s counts=%s",
            linepack,
            ltc,
            ltc_steady_start,
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
        # One community per connected component per sector (global L1, no hierarchy);
        # CPs join bridged communities, reconciling via MultiCommunityCPRole (EMA +
        # deadband + cooldown). multihop_constraint MUST stay off: the partition
        # collapses each sector into one group, so forwarding fans out O(N^2) and
        # OOM-kills the worker; direct neighbours already give the global picture.
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


_BASELINE_SOLVE_CAP_S = 300.0


def _compute_baseline(task: TaskSpec, logger: logging.Logger) -> dict[str, Any] | None:
    """Solve the no-failure LP so restoration can be expressed as a ratio of
    pre-failure served. Returns ``None`` (logged) on failure so the task proceeds.
    """
    if task.grid not in GRIDS:
        return None
    # The baseline solve runs BEFORE _run_simulation installs the per-solve cap.
    # On reconfig grids the backup tie-line on_off binaries make it a MILP that,
    # uncapped, can spin gurobi for the full default limit merely proving
    # optimality of the (trivially feasible) radial incumbent — leaving the task
    # with no baseline (baseline_available=False). Cap it here so it returns that
    # incumbent. Idempotent with the smaller sim-phase cap installed later.
    install_solver_time_limit(_BASELINE_SOLVE_CAP_S)
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


def _write_oracle_outputs(
    out_dir: Path,
    plan: RuntimePlan,
    task: TaskSpec,
    logger: logging.Logger,
    baseline_served: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run the centralized oracle LP, write its result/slack/failures, and
    return the result payload so the caller can grade the solve outcome."""
    net, failures, oracle_metrics = _run_oracle(
        plan, task, logger, baseline_served=baseline_served, out_dir=out_dir
    )
    oracle_metrics = _json_sanitize(oracle_metrics)
    (out_dir / "failures.json").write_text(
        json.dumps(_serialize_failures(failures), indent=2), encoding="utf-8"
    )
    (out_dir / "result.json").write_text(
        json.dumps(
            oracle_metrics, indent=2, sort_keys=True, default=str, allow_nan=False
        ),
        encoding="utf-8",
    )
    write_slack_meta(out_dir / "slack_meta.json", net)
    return oracle_metrics


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


def _write_simulation_outputs(
    out_dir: Path,
    plan: RuntimePlan,
    task: TaskSpec,
    logger: logging.Logger,
    started: float,
    baseline_served: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run the MAS simulation, write every per-task artefact, and return the
    claims-validation payload.
    """
    world, failures, net = asyncio.run(
        _run_simulation(plan, task, logger, out_dir=out_dir)
    )
    (out_dir / "failures.json").write_text(
        json.dumps(_serialize_failures(failures), indent=2), encoding="utf-8"
    )
    behavior = world.environment.behavior
    # Force a fresh energy flow so observers report post-action state, not the
    # cooldown-cached solve (else the served breakdown is stale).
    try:
        behavior.flush_energy_flow()
    except AttributeError:
        logger.debug("Behavior has no flush_energy_flow() — skipping")
    # Stepper change log: topology mutations + solver-decided islanding events
    # ('islanded'/'rejoined'), the primary islanding-extension evidence.
    try:
        changes_df = behavior.network_changes_df()
        if changes_df is not None and len(changes_df):
            changes_df.to_csv(out_dir / "network_changes.csv", index=False)
    except Exception as exc:  # noqa: BLE001 — diagnostics only, never fatal
        logger.warning("network_changes.csv not written: %s", exc)
    # The stepper absorbs raising solvers (on_step_error="skip"), so a task
    # whose every solve failed would otherwise grade a frozen pre-failure
    # state as ok — mirror the oracle's refusing-to-grade guard.
    solves_ok = getattr(behavior, "_physics_solves_ok", None)
    solves_failed = int(getattr(behavior, "_physics_solves_failed", 0) or 0)
    if solves_ok is not None and int(solves_ok) == 0:
        raise RuntimeError(
            f"no physics solve succeeded during the simulation "
            f"({solves_failed} failed) — refusing to grade a frozen-state "
            "result as ok"
        )
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
    if solves_ok is not None:
        payload.setdefault("outcomes", {})["physics_solves"] = {
            "ok": int(solves_ok),
            "failed": solves_failed,
        }
    payload = _json_sanitize(payload)
    write_result_json(out_dir / "result.json", payload)
    write_served_csv(out_dir / "served.csv", net, behavior, priorities=priorities)
    write_served_by_load_csv(
        out_dir / "served_by_load.csv",
        net,
        behavior,
        priorities=priorities,
    )
    # Judge gas-pressure compliance on EXACT-Weymouth physics, not the relaxed
    # formulation the sim solves with (which reports spurious √3 over-pressure
    # at shed/zero-flow gas junctions). Served/control state above is unchanged.
    constraint_net = _exact_gas_solved_net(net, logger)
    write_constraints_final_csv(out_dir / "constraints_final.csv", constraint_net)
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
    # An exception here propagates so the task is graded "error" — a run
    # without claims columns must not land as an ungated "ok".
    claims = evaluate_task(out_dir)
    payload["claims"] = _json_sanitize(claims)
    write_result_json(out_dir / "result.json", payload)
    return claims


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


# Artifacts run_task / the oracle / the metrics writers scrub at task start so a
# re-run that fails early can't leave stale outputs for the aggregator to join.
# NOTE: incomplete — network_changes.csv (written above) is produced but not listed here.
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


def _scrub_stale_artifacts(out_dir: Path) -> None:
    for name in _TASK_ARTIFACTS:
        try:
            (out_dir / name).unlink(missing_ok=True)
        except OSError:
            pass


def run_task(campaign_dir: Path, task_id: int, *, reraise: bool = False) -> int:
    plan = RuntimePlan.from_config_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
    tasks = read_manifest(campaign_dir)
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"task_id {task_id} out of range [0, {len(tasks)})")
    task = tasks[task_id]

    out_dir = task_dir(campaign_dir, task.task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _scrub_stale_artifacts(out_dir)
    (out_dir / "config.json").write_text(
        json.dumps(task.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
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

    try:
        # Inside the try so SIGTERM during the (up to 300s) Gurobi solve still
        # lands in the handlers below: status.json written, log handler closed.
        baseline_served = _compute_baseline(task, logger)
        if task.variant == "oracle":
            payload = _write_oracle_outputs(
                out_dir, plan, task, logger, baseline_served
            )
            lp_success = bool(
                payload.get("outcomes", {}).get("oracle_lp_success", True)
            )
            completed = bool(payload.get("completed", True))
            if not (lp_success and completed):
                raise RuntimeError(
                    "oracle LP solve failed "
                    f"(lp_success={lp_success}, completed={completed}) — "
                    "refusing to grade a fictitious zero-PWSF result as ok"
                )
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
            ),
            encoding="utf-8",
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
            ),
            encoding="utf-8",
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
            ),
            encoding="utf-8",
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
        # Sampling can undershoot the requested count (small grids, clamps);
        # record what was actually injected so analysis can stratify on it.
        try:
            failures_path = out_dir / "failures.json"
            if failures_path.exists():
                n_actual = len(json.loads(failures_path.read_text(encoding="utf-8")))
                status["n_failures_actual"] = n_actual
                if n_actual < task.n_failures:
                    logger.warning(
                        "Injected %d failure(s) but %d were requested "
                        "(sampling undershoot)",
                        n_actual,
                        task.n_failures,
                    )
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        (out_dir / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
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
        for _name in _SOLVER_FAILURE_LOGGERS:
            logging.getLogger(_name).removeFilter(solver_counter)
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
