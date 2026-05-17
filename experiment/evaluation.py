from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mango_energy_environments import (
    Failure,
    calc_general_resilience_performance,
    fetch_cigre_net,
    fetch_example_net,
    solve_load_shedding_optimization,
)

from scare.base.util import create_failures
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
EPSILON = 1e-6
SIMULATION_DURATION_S = 30.0


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


async def evaluate_net(
    monee_net: Any,
    failures: list[Failure],
    scenario_name: str,
    priorities: dict[str, int] | None = None,
) -> dict[str, Any]:
    try:
        baseline_net = solve_load_shedding_optimization(monee_net)
        baseline_perf = calc_general_resilience_performance(baseline_net)
    except Exception as exc:
        # Don't swallow into 0.0 — the downstream performance ratio
        # ``(baseline_perf + ε) / (mas_perf + ε)`` would mask the
        # failure as "MAS infinitely worse than baseline".  Surface the
        # error so the caller can decide whether to skip the scenario.
        logger.error("Baseline optimisation failed: %s", exc)
        raise

    world = create_restoration_scenario_world(
        monee_net,
        priorities=priorities,
        simulation_duration_s=SIMULATION_DURATION_S,
    )
    await start_restoration_simulation(world, failures, SIMULATION_DURATION_S)

    el_rec = world.data_collections.get("electrical_balance")
    mas_perf = (
        float(np.mean(el_rec.timeseries)) if el_rec and el_rec.timeseries else 0.0
    )
    performance = (baseline_perf + EPSILON) / (mas_perf + EPSILON)
    n_messages = len(world.recorded_messages)

    result = {
        "scenario": scenario_name,
        "baseline_perf": float(baseline_perf),
        "mas_perf": float(mas_perf),
        "performance": float(performance),
        "n_messages": n_messages,
    }
    logger.info(
        "Scenario %-30s  perf=%.3f  msgs=%d", scenario_name, performance, n_messages
    )

    _ensure_results_dir()
    (RESULTS_DIR / f"{scenario_name}.json").write_text(json.dumps(result, indent=2))

    return result


async def evaluate_microgrid() -> list[dict]:
    net = fetch_example_net()
    results = []
    branch_failures = create_failures(net, "branch", num_failures=1, delay_s_max=3.0)
    results.append(await evaluate_net(net, branch_failures, "microgrid_branch_failure"))
    results.append(await evaluate_net(net, [], "microgrid_no_failure"))
    return results


async def evaluate_microgrid_lines() -> list[dict]:
    net = fetch_example_net()
    results = []
    power_branches = [b for b in net.branches if not b.model.is_cp()]
    for branch in power_branches:
        failure = Failure(delay_s=2.0, branch_ids=[branch.id])
        name = f"microgrid_line_{branch.id[0]}_{branch.id[1]}"
        results.append(await evaluate_net(net, [failure], name))
    return results


async def evaluate_cigre() -> list[dict]:
    net = fetch_cigre_net()
    failures = create_failures(net, "branch", num_failures=1, delay_s_max=3.0)
    return [await evaluate_net(net, failures, "cigre_branch_failure")]


async def evaluate_net_with_priorities() -> list[dict]:
    net = fetch_example_net()
    failures = create_failures(net, "branch", num_failures=1, delay_s_max=2.0)
    results = []
    configs = [
        {"name": "equal_priority", "priorities": {}},
        {"name": "gen_high_priority", "priorities": {}},
        {"name": "load_high_priority", "priorities": {}},
        {"name": "mixed_priority", "priorities": {}},
    ]
    for cfg in configs:
        results.append(
            await evaluate_net(net, failures, cfg["name"], priorities=cfg["priorities"])
        )
    return results


async def evaluate_net_with_delays() -> list[dict]:
    net = fetch_example_net()
    failures = create_failures(net, "branch", num_failures=1, delay_s_max=2.0)
    results = []
    for delay_ms in [5.0, 20.0, 50.0, 100.0]:
        world = create_restoration_scenario_world(
            net, base_delay_ms=delay_ms, simulation_duration_s=SIMULATION_DURATION_S
        )
        await start_restoration_simulation(world, failures, SIMULATION_DURATION_S)
        el_rec = world.data_collections.get("electrical_balance")
        mas_perf = (
            float(np.mean(el_rec.timeseries)) if el_rec and el_rec.timeseries else 0.0
        )
        results.append({"scenario": f"delay_{delay_ms:.0f}ms", "mas_perf": mas_perf})
    return results


async def main() -> None:
    logger.info("=== SCARE Batch Evaluation ===")
    all_results = []
    all_results += await evaluate_microgrid()
    all_results += await evaluate_microgrid_lines()
    all_results += await evaluate_cigre()
    all_results += await evaluate_net_with_priorities()
    all_results += await evaluate_net_with_delays()

    summary_df = pd.DataFrame(all_results)
    logger.info("\n%s", summary_df.to_string())
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    logger.info("Summary written to %s", RESULTS_DIR / "summary.csv")


if __name__ == "__main__":
    asyncio.run(main())
