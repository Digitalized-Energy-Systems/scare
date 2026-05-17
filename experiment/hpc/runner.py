"""Run a single restoration evaluation task end-to-end.

Designed to be invoked once per Slurm array task — but runs identically
without Slurm. Reads the campaign config + manifest, runs the task, and
writes all per-task artefacts under ``<campaign_dir>/tasks/<task_id>/``:

    config.json        copy of the TaskSpec (the seed, grid, n_failures)
    failures.json      resolved branch_ids + delays the simulation saw
    result.json        scalar metrics extracted from the world
    status.json        ok|error|timeout, duration, solver-warning count
    run.log            full DEBUG log of the task
    diagnostics.txt    last actions captured by scare.base.diagnostics
    exception.json     traceback (only on error)
    timeseries.csv     per-step balance/constraint series (optional)

If ``--task-id`` is not given, ``$SLURM_ARRAY_TASK_ID`` is used.

Exit codes: 0 = ok, 2 = timeout, 1 = any other error. Always writes
status.json before exiting so the aggregator never has to guess.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import random
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiment.hpc.config import (
    CAMPAIGN_LAYOUT,
    RuntimePlan,
    TaskSpec,
    task_dir,
)
from experiment.hpc.plan import read_manifest

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 2

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class _SolverFailureCounter(logging.Filter):
    """Counts solver-status escalations on monee.solver.pyo so we can
    report solver health per task without grepping logs.

    Tracks both real infeasibilities (ERROR with ``infeasible``) and
    non-ok warnings (Gurobi stopping at gap/time limit etc).  ``count``
    is the union for backwards-compatible reporting; the split is
    available via ``infeasible_count`` and ``warning_count``.

    Also catches Gurobi / Pyomo exception strings that escape the
    monee.solver.pyo logger (license errors, host-id mismatches, env
    initialization failures) so the aggregator can distinguish solver
    environment issues from algorithm bugs.
    """

    _SOLVER_ERROR_MARKERS: tuple[str, ...] = (
        "GurobiError",
        "HostID mismatch",
        "License",  # Gurobi LicenseError
        "Pyomo solve infeasible",
    )

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.infeasible_count = 0
        self.warning_count = 0

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.levelno < logging.WARNING:
            return True
        msg = record.getMessage()
        if "infeasible (status=" in msg:
            self.infeasible_count += 1
            self.count += 1
        elif "returned non-ok status" in msg:
            self.warning_count += 1
            self.count += 1
        elif any(marker in msg for marker in self._SOLVER_ERROR_MARKERS):
            # Gurobi env / license / host-id errors do not go through the
            # monee.solver.pyo "infeasible" pathway, but they are solver
            # failures from the campaign's POV.
            self.warning_count += 1
            self.count += 1
        return True


def _setup_logging(log_path: Path) -> tuple[logging.FileHandler, _SolverFailureCounter]:
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Drop pre-existing handlers so we don't double-log to stderr from
    # any earlier basicConfig() call (e.g. the manifest CLI).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    # Keep WARN+ visible on stderr too so Slurm captures show-stopper events
    # in the (otherwise empty) per-array stderr file.
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(stderr)

    # Suppress the verbose solver loggers — pyomo, gurobipy and
    # mango.simulation emit one DEBUG line per constraint / per branch
    # which dominates the per-task log (hundreds of MB) once the
    # solver fires from multiple roles per simulation step.  Keep
    # WARN+ so genuine solver failures still surface.
    for noisy in (
        "pyomo.core",
        "pyomo.opt",
        "pyomo.contrib",
        "gurobipy",
        "mango.simulation.world",
        "mango.express",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    counter = _SolverFailureCounter()
    logging.getLogger("monee.solver.pyo").addFilter(counter)
    return handler, counter


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def _resolve_failures(monee_net: Any, plan: RuntimePlan, task: TaskSpec) -> list[Any]:
    """Draw the failure scenario for this task.

    The ``scenario`` dict on the TaskSpec can override the default
    "branch"-only sampling:

    - ``failure_type``   — ``"branch"`` / ``"generator"`` / ``"mixed"``
                           (default: ``"branch"``)
    - ``generator_share`` — for ``mixed``, fraction of the draw that
                            is generators (default 0.5)
    """
    from scare.base.util import create_failures

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

    Captures ``branch_ids`` for branch-style failures and ``custom_id``
    for generator-style ones (the underlying ``Failure.custom``
    callable isn't JSON-able, but the id we stamp on it identifies
    the deactivated component).
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
    from mango.simulation.world import WorldRecording

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
            metrics[f"{name}__integral"] = float(np.trapz(arr, np.asarray(t, dtype=float)))

    metrics["messages_total"] = len(getattr(world, "recorded_messages", []) or [])
    clock = getattr(world, "clock", None)
    metrics["sim_time_final"] = float(getattr(clock, "time", 0.0))
    return metrics


def _write_timeseries(world: Any, path: Path) -> None:
    from mango.simulation.world import WorldRecording

    import pandas as pd

    series_map: dict[str, pd.Series] = {}
    for name, rec in getattr(world, "data_collections", {}).items():
        if not isinstance(rec, WorldRecording):
            continue
        if not rec.timeseries:
            continue
        series_map[name] = pd.Series(rec.timeseries, index=pd.Index(rec.time, name="time_s"))
    if not series_map:
        return
    df = pd.concat(series_map, axis=1).sort_index()
    df.to_csv(path)


def _dump_diagnostics(path: Path) -> None:
    from scare.base import diagnostics

    path.write_text(diagnostics.dump_recent() + "\n")


async def _run_simulation(plan: RuntimePlan, task: TaskSpec, logger: logging.Logger):
    """Build and run one scare-variant simulation.  Returns (world,
    failures, net) so the caller can extract metrics + run the
    end-of-sim recordings (e.g. served breakdown via the behavior).
    """
    from experiment.restoration import GRIDS
    from scare.base.config import RestorationConfiguration
    from scare.scenario.restoration import (
        create_restoration_scenario_world,
        start_restoration_simulation,
    )

    if task.grid not in GRIDS:
        raise SystemExit(f"Unknown grid {task.grid!r}; available: {sorted(GRIDS)}")

    factory = GRIDS[task.grid]
    logger.info("Building network for grid=%s", task.grid)
    # Factory already applies MISOCP + McCormick formulations; re-applying
    # MISOCP here would overwrite the McCormick partition metadata.
    net = factory()
    _apply_scenario(net, task, logger)

    failures = _resolve_failures(net, plan, task)
    logger.info(
        "Resolved %d failure(s) for seed=%d: %s",
        len(failures), task.seed, [f.branch_ids for f in failures],
    )

    cfg = _config_from_task(task)
    logger.info("Variant=%s ablation=%s sweep=%s",
                task.variant, task.ablation or {}, task.sweep or {})

    priorities = _resolve_priorities(net, task, logger)
    # Stash on the net so the post-sim metric writers can pick it up
    # without an extra parameter through ``_run_simulation``.
    net._scare_priorities = priorities

    world = create_restoration_scenario_world(
        net,
        priorities=priorities,
        simulation_duration_s=plan.simulation_duration_s,
        config=cfg,
    )
    logger.info("Running simulation for %.1f s (timeout=%.0f s)",
                plan.simulation_duration_s, plan.task_timeout_s)
    try:
        await asyncio.wait_for(
            start_restoration_simulation(world, failures, plan.simulation_duration_s),
            timeout=plan.task_timeout_s,
        )
    except asyncio.TimeoutError:
        # Wallclock timeout fires while the simulation coroutine is
        # cancelled; ``start_restoration_simulation`` never reaches
        # ``_flush_pending_negotiations``, so any in-flight gossips
        # would be silently lost from the diary.  Drain them here so
        # the started == Σ terminals invariant holds even on timeouts.
        try:
            from scare.scenario.restoration import _flush_pending_negotiations
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
    """Build the network, draw the same failures the scare variant
    would, and solve monee's minimal-load-shedding LP.  Returns
    (net, failures, result_payload) — no world, no agents.
    """
    import time as _time

    from experiment.restoration import GRIDS

    from experiment.eval.oracle import compose_oracle_result

    if task.grid not in GRIDS:
        raise SystemExit(f"Unknown grid {task.grid!r}; available: {sorted(GRIDS)}")
    factory = GRIDS[task.grid]
    logger.info("Building network for grid=%s (oracle)", task.grid)
    # Factory already applies MISOCP + McCormick.
    net = factory()
    _apply_scenario(net, task, logger)
    failures = _resolve_failures(net, plan, task)
    logger.info(
        "Oracle: %d failure(s) for seed=%d: %s",
        len(failures), task.seed, [f.branch_ids for f in failures],
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
    """Build the per-load priority dict from the scenario's
    ``priority_assignment`` knob.

    Default is ``"skewed"`` so the priority machinery (QP weights,
    tier-aware ADMM, priority-weighted served fraction) is exercised
    by default.  monee observations don't carry a per-load priority
    field, so without this default the obs_priority fallback returned
    tier 1 for every load and the priority-aware behaviour was
    degenerate (audit P1-7).

    Recognised values for ``priority_assignment``:
    ``"uniform"``, ``"skewed"``, ``"by_capacity"``, ``"all_one"``.
    Set ``"all_one"`` explicitly to recover the legacy degenerate
    behaviour.
    """
    scenario = task.scenario or {}
    distribution = scenario.get("priority_assignment", "skewed")
    from experiment.restoration import assign_load_priorities

    priorities = assign_load_priorities(
        net, seed=task.seed, distribution=distribution
    )
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
    """Apply scenario-kind mutations to the freshly-built network.

    Scenario kinds are stored on ``task.scenario["kind"]``:

    - ``"clean"`` (default) — no mutation.
    - ``"cold_day"`` — lower the heat slack supply temperature and
      scale heat loads up (see ``experiment.restoration.apply_cold_day``).
      Tunables: ``supply_t_k`` (default 343.15 K) and
      ``heat_load_scale`` (default 1.5×).
    - ``"pv_peak"`` — sunny-midday over-voltage stress: scale up every
      ``PowerGenerator.p_mw`` and scale down every ``PowerLoad.p_mw``
      so the feeder runs reverse-power and ``vm_pu`` drifts toward the
      upper bound.  Targets the VDE-AR-N 4105 operating point.
      Tunables: ``gen_scale`` (default 3×) and ``load_scale`` (default
      0.3×).
    - ``"line_stress"`` — line-thermal stress: scale loads up and
      reduce PowerLine ``max_i_ka`` so loading_percent rises into
      overload after a single branch failure.  Exercises the branch-
      mode constraint monitor, the priority-aware home group
      assignment, and the 6c path-ranking reconfigurator.
      Tunables: ``load_scale`` (default 1.8×), ``ampacity_scale``
      (default 0.5×), ``affect_branch_fraction`` (default 1.0).

    Other kinds are silently passed through so future scenario types
    can be wired without changing existing behaviour.

    The scenario may also carry ``slack_budget_pct`` independently of
    ``kind``: when set, ``experiment.restoration.apply_slack_budget``
    is called *after* any kind-specific mutation, so the budget
    reflects the (possibly scaled) post-mutation demand.  The grid
    factory itself never bakes the slack budget in — the LP needs an
    unconstrained slack to converge, and the budget is an operator
    policy that varies by scenario, not a physical grid attribute.
    """
    scenario = task.scenario or {}
    kind = scenario.get("kind", "clean")
    _KNOWN_KINDS = {"clean", "cold_day", "pv_peak", "line_stress", "microgrid"}
    if kind not in _KNOWN_KINDS:
        # Silently passing unknown kinds through used to be the rule —
        # but a typo (``cold day`` vs ``cold_day``) then produced a
        # clean run with no warning.  Surface it so the campaign author
        # can correct the config.
        logger.warning(
            "Unknown scenario kind %r (known: %s) — falling through to "
            "no-mutation behaviour.  Check the campaign config for typos.",
            kind, sorted(_KNOWN_KINDS),
        )
    if kind == "cold_day":
        from experiment.restoration import apply_cold_day

        kwargs = {
            k: scenario[k]
            for k in ("supply_t_k", "heat_load_scale")
            if k in scenario
        }
        apply_cold_day(net, **kwargs)
        logger.info("Applied cold_day scenario: %s", kwargs or "<defaults>")
    elif kind == "pv_peak":
        from experiment.restoration import apply_pv_peak

        kwargs = {
            k: scenario[k]
            for k in ("gen_scale", "load_scale")
            if k in scenario
        }
        apply_pv_peak(net, **kwargs)
        logger.info("Applied pv_peak scenario: %s", kwargs or "<defaults>")
    elif kind == "line_stress":
        from experiment.restoration import apply_line_stress

        kwargs = {
            k: scenario[k]
            for k in ("load_scale", "ampacity_scale", "affect_branch_fraction")
            if k in scenario
        }
        apply_line_stress(net, **kwargs)
        logger.info("Applied line_stress scenario: %s", kwargs or "<defaults>")
    elif kind == "microgrid":
        # Microgrid / islanding scenario.  Opt in to monee's islanding
        # extension AND promote eligible generator-class children to
        # ``GridForming*`` so the LP has reference units to anchor sub-
        # islands on when the main slack is unreachable.  Without
        # promotion, ``enable_islanding`` is a no-op on stock simbench
        # nets (no native GridFormingMixin children); promotion is
        # the practical way to make the extension exercise something.
        from experiment.restoration import apply_microgrid_islanding

        carriers = scenario.get(
            "carriers", ("electricity", "water", "gas")
        )
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
            list(carriers), promote_all, counts,
        )

    # Operator slack-budget policy — orthogonal to ``kind`` so any
    # scenario kind can carry one.  Applied AFTER any kind-specific
    # mutation so the budget reflects the (possibly scaled) demand
    # in the post-mutation network.  ``apply_slack_budget`` widens
    # the LP Var bounds to ``±10·budget`` (keeps the LP feasible)
    # and stamps the budget as the slack agents' rating that the MAS
    # then drives toward via ``slack_target_fraction``.  The grid
    # factory itself never bakes a slack budget in — see the
    # ``GRIDS`` dict comment for the design rationale.
    slack_budget_pct = scenario.get("slack_budget_pct")
    if slack_budget_pct is not None:
        from experiment.restoration import apply_slack_budget

        apply_slack_budget(net, float(slack_budget_pct))
        logger.info(
            "Applied slack_budget_pct=%s (per-scenario operator policy)",
            slack_budget_pct,
        )


def _config_from_task(task: TaskSpec):
    """Compose a ``RestorationConfiguration`` from the task's variant,
    ablation, and sweep dictionaries.  Variant maps to a base preset;
    ablation / sweep are field overrides applied on top.
    """
    from dataclasses import replace
    from scare.base.config import RestorationConfiguration

    if task.variant == "single_level":
        base = RestorationConfiguration(
            enable_holonic=False,
            enable_cp_admm=False,
        )
    else:
        base = RestorationConfiguration()
    overrides: dict = {}
    overrides.update(task.ablation or {})
    overrides.update(task.sweep or {})
    if not overrides:
        return base
    valid = {f.name for f in __import__(
        "dataclasses", fromlist=["fields"]
    ).fields(base)}
    clean = {k: v for k, v in overrides.items() if k in valid}
    skipped = set(overrides) - valid
    if skipped:
        logging.getLogger("experiment.hpc.runner").warning(
            "Ignoring unknown config overrides: %s", sorted(skipped),
        )
    return replace(base, **clean)


def run_task(campaign_dir: Path, task_id: int, *, reraise: bool = False) -> int:
    plan = RuntimePlan.from_config_json(campaign_dir / CAMPAIGN_LAYOUT["config"])
    tasks = read_manifest(campaign_dir)
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"task_id {task_id} out of range [0, {len(tasks)})")
    task = tasks[task_id]

    out_dir = task_dir(campaign_dir, task.task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(task.to_dict(), indent=2, sort_keys=True))

    handler, solver_counter = _setup_logging(out_dir / "run.log")
    logger = logging.getLogger("experiment.hpc.runner")
    logger.info(
        "Task %d  grid=%s  seed=%d  n_failures=%d",
        task.task_id, task.grid, task.seed, task.n_failures,
    )
    logger.info("Slurm: job=%s array_task=%s host=%s",
                os.environ.get("SLURM_JOB_ID"),
                os.environ.get("SLURM_ARRAY_TASK_ID"),
                os.environ.get("HOSTNAME") or platform.node())

    _seed_everything(task.seed)
    # Toggle per-aid trajectory logging for the whole task run; off by
    # default because the log can grow large.  Enabled by setting
    # ``write_trajectories: true`` in the campaign config; consumed by
    # the C.5 cluster-synchronisation analysis.
    from scare.base import diagnostics as _diag

    _diag.set_trajectory_logging(getattr(plan, "write_trajectories", False))
    # Drop any stale exception.json from a prior failed run so the
    # aggregator's exception counts reflect only the *current* status.
    # Without this, a re-run that succeeds still shows a count in
    # ``Exception breakdown`` because the old crash file persists.
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

    # SIGTERM handler — converts the signal into a Python exception so the
    # ``finally`` block below runs and writes ``status.json``.  Without
    # this, ``timeout`` (SIGTERM) + grace + SIGKILL leaves no trace of the
    # task, and the aggregator silently drops it.  SIGKILL is uncatchable
    # by design but we honour the SIGTERM grace window.
    _prev_term = signal.getsignal(signal.SIGTERM)
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt("SIGTERM received — emergency shutdown")
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        # Not main thread or otherwise restricted — best-effort only.
        _prev_term = None

    # Pre-failure baseline.  Solve the no-failure LP on a fresh build
    # of this grid + scenario; result.json's ``outcomes.restoration``
    # then expresses post-restoration served as a ratio of this
    # baseline, surfacing absolute load lost despite the restoration
    # rather than the priority-weighted fraction alone.
    baseline_served = None
    try:
        from experiment.eval.oracle import compute_baseline_served
        from experiment.restoration import GRIDS

        if task.grid in GRIDS:
            # We need priorities for the baseline metric to be
            # comparable; resolve them from a fresh build.  Factory
            # already applies MISOCP + McCormick.
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Baseline LP failed (continuing without it): %s", exc)

    try:
        if task.variant == "oracle":
            world = None
            net, failures, oracle_metrics = _run_oracle(
                plan, task, logger, baseline_served=baseline_served
            )
            (out_dir / "failures.json").write_text(
                json.dumps(_serialize_failures(failures), indent=2)
            )
            payload = oracle_metrics
            (out_dir / "result.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str)
            )
        else:
            world, failures, net = asyncio.run(_run_simulation(plan, task, logger))
            (out_dir / "failures.json").write_text(
                json.dumps(_serialize_failures(failures), indent=2)
            )
            from experiment.eval.results import (
                compose_result,
                write_diary_csv,
                write_events_csv,
                write_messages_csv,
                write_result_json,
                write_served_by_load_csv,
                write_served_csv,
                write_trajectories_csv,
            )
            behavior = world.environment.behavior
            # End-of-sim measurement boundary: force a fresh energy flow
            # so observers report post-agent-action state instead of the
            # cooldown-cached previous solve.  Without this the served
            # breakdown reflects an older regulation value and the
            # priority-invariant claim hallucinates intra-component
            # inversions.
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
                out_dir / "served_by_load.csv", net, behavior, priorities=priorities,
            )
            write_diary_csv(out_dir / "diary.csv")
            write_events_csv(out_dir / "events.csv")
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
            # Claims validation: consume the artefacts we just wrote and
            # fold pass/fail into result.json so the aggregator can roll
            # them up.
            try:
                from experiment.eval.claims import evaluate_task

                claims = evaluate_task(out_dir)
                payload["claims"] = claims
                write_result_json(out_dir / "result.json", payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Claims validation failed: %s", exc)
        # ``ok`` means the task didn't crash.  If any *fatal* claim
        # failed (priority_invariant + monotonic_progress by default),
        # escalate to ``claims_failed`` so the aggregator stops treating
        # silent priority inversions as successful runs.  Callers can
        # override the fatal set per-campaign via plan.fatal_claims.
        fatal_claims = tuple(
            getattr(plan, "fatal_claims",
                    ("priority_invariant", "monotonic_progress"))
        )
        if claims:
            failing = [
                name for name in fatal_claims
                if name in claims and not claims[name].get("passed", True)
            ]
        else:
            failing = []
        if failing:
            logger.warning("Fatal claims failed: %s", failing)
            status["status"] = "claims_failed"
            status["failing_claims"] = failing
            exit_code = EXIT_OK
        else:
            status["status"] = "ok"
            exit_code = EXIT_OK

    except asyncio.TimeoutError:
        logger.error("Task timed out after %.0f s", plan.task_timeout_s)
        status["status"] = "timeout"
        (out_dir / "exception.json").write_text(json.dumps({
            "type": "TimeoutError",
            "message": f"Exceeded plan.task_timeout_s={plan.task_timeout_s}",
        }, indent=2))
        exit_code = EXIT_TIMEOUT

    except KeyboardInterrupt as exc:
        # SIGTERM (or Ctrl-C) — record an explicit ``killed`` status so the
        # aggregator can distinguish wallclock kill from ordinary error.
        logger.error("Task killed: %s", exc)
        status["status"] = "killed"
        (out_dir / "exception.json").write_text(json.dumps({
            "type": "KeyboardInterrupt",
            "message": str(exc),
        }, indent=2))
        exit_code = EXIT_TIMEOUT

    except Exception as exc:  # noqa: BLE001
        logger.exception("Task failed: %s", exc)
        (out_dir / "exception.json").write_text(json.dumps({
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }, indent=2))
        if reraise:
            raise
        exit_code = EXIT_ERROR

    finally:
        status["duration_s"] = round(time.monotonic() - started, 3)
        status["solver_failures"] = solver_counter.count
        status["solver_infeasibilities"] = solver_counter.infeasible_count
        status["solver_warnings"] = solver_counter.warning_count
        (out_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True))
        try:
            _dump_diagnostics(out_dir / "diagnostics.txt")
        except Exception:  # noqa: BLE001
            pass
        logger.info("Status=%s duration=%.1fs solver_failures=%d exit=%d",
                    status["status"], status["duration_s"],
                    status["solver_failures"], exit_code)
        logging.getLogger().removeHandler(handler)
        handler.close()
        if _prev_term is not None:
            try:
                signal.signal(signal.SIGTERM, _prev_term)
            except (ValueError, OSError):
                pass

    return exit_code


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--campaign-dir", required=True, type=Path)
    p.add_argument("--task-id", type=int, default=None,
                   help="Index into manifest.jsonl. If omitted, falls back to $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--reraise", action="store_true",
                   help="Re-raise exceptions for debugging instead of writing exception.json")
    return p.parse_args()


def _resolve_task_id(args: argparse.Namespace) -> int:
    if args.task_id is not None:
        return args.task_id
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise SystemExit("Pass --task-id N or run under Slurm with SLURM_ARRAY_TASK_ID set.")
    return int(env)


def main() -> None:
    args = _parse_args()
    sys.exit(run_task(args.campaign_dir.resolve(), _resolve_task_id(args), reraise=args.reraise))


if __name__ == "__main__":
    main()
