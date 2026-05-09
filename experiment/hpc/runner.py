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
import random
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
    """Counts ``Pyomo solve failed`` warnings on monee.solver.pyo so we
    can report solver health per task without grepping logs."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if (
            record.levelno >= logging.WARNING
            and "Pyomo solve failed" in record.getMessage()
        ):
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
    from monee.model.formulation import MISOCP_NETWORK_FORMULATION
    from scare.base.config import RestorationConfiguration
    from scare.scenario.restoration import (
        create_restoration_scenario_world,
        start_restoration_simulation,
    )

    if task.grid not in GRIDS:
        raise SystemExit(f"Unknown grid {task.grid!r}; available: {sorted(GRIDS)}")

    factory = GRIDS[task.grid]
    logger.info("Building network for grid=%s", task.grid)
    net = factory()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
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


def _run_oracle(plan: RuntimePlan, task: TaskSpec, logger: logging.Logger):
    """Build the network, draw the same failures the scare variant
    would, and solve monee's minimal-load-shedding LP.  Returns
    (net, failures, result_payload) — no world, no agents.
    """
    import time as _time

    from experiment.restoration import GRIDS
    from monee.model.formulation import MISOCP_NETWORK_FORMULATION

    from experiment.eval.oracle import compose_oracle_result

    if task.grid not in GRIDS:
        raise SystemExit(f"Unknown grid {task.grid!r}; available: {sorted(GRIDS)}")
    factory = GRIDS[task.grid]
    logger.info("Building network for grid=%s (oracle)", task.grid)
    net = factory()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
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
    )
    payload["wallclock_s"] = round(_time.monotonic() - started, 3)
    return net, failures, payload


def _resolve_priorities(
    net: Any, task: TaskSpec, logger: logging.Logger
) -> dict[str, int] | None:
    """Build the per-load priority dict from the scenario's
    ``priority_assignment`` knob.  Returns ``None`` when the scenario
    does not request priority diversity, preserving the legacy
    ``all-tier-1`` default that earlier campaigns relied on.

    Recognised values for ``priority_assignment``:
    ``"uniform"``, ``"skewed"``, ``"by_capacity"``, ``"all_one"``.
    """
    scenario = task.scenario or {}
    distribution = scenario.get("priority_assignment")
    if not distribution:
        return None
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

    Other kinds are silently passed through so future scenario types
    can be wired without changing existing behaviour.
    """
    scenario = task.scenario or {}
    kind = scenario.get("kind", "clean")
    if kind == "cold_day":
        from experiment.restoration import apply_cold_day

        kwargs = {
            k: scenario[k]
            for k in ("supply_t_k", "heat_load_scale")
            if k in scenario
        }
        apply_cold_day(net, **kwargs)
        logger.info("Applied cold_day scenario: %s", kwargs or "<defaults>")


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
                os.environ.get("HOSTNAME") or os.uname().nodename)

    _seed_everything(task.seed)
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

    try:
        if task.variant == "oracle":
            world = None
            net, failures, oracle_metrics = _run_oracle(plan, task, logger)
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
                write_served_csv,
            )
            behavior = world.environment.behavior
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
            )
            write_result_json(out_dir / "result.json", payload)
            write_served_csv(out_dir / "served.csv", net, behavior, priorities=priorities)
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
