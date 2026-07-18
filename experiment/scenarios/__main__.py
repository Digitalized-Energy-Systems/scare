"""Demo runner: build a named grid, inject one failure, simulate, visualise.

python -m experiment.scenarios --grid simbench_lv
"""

import argparse
import asyncio
import logging
import random

from experiment.scenarios.grids import GRIDS
from scare.base.runtime.diagnostics import (
    arm,
    install_solver_failure_dump,
    negotiation_summary,
)
from scare.base.viz.visu import visualize_results
from scare.scenario.failure_sampling import create_failures
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)

logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s"
)
logging.getLogger("scare").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

install_solver_failure_dump()
arm()  # install_solver_failure_dump no longer arms as a side effect

SIMULATION_DURATION_S = 5.0
FAILURE_DELAY_S = 2.0


async def run(grid: str, seed: int | None = None, write_html: bool = True) -> None:
    if seed is not None:
        random.seed(seed)

    factory = GRIDS[grid]
    net = factory()

    logger.info("Building restoration world for grid=%s …", grid)
    world = create_restoration_scenario_world(
        net, simulation_duration_s=SIMULATION_DURATION_S
    )

    failures = create_failures(
        net, "branch", num_failures=1, delay_s_max=FAILURE_DELAY_S
    )
    logger.info(
        "Scheduled %d failure(s): %s", len(failures), [f.branch_ids for f in failures]
    )

    logger.info("Running simulation for %.0f s …", SIMULATION_DURATION_S)
    await start_restoration_simulation(world, failures, SIMULATION_DURATION_S)
    logger.info("Simulation complete.")

    logger.info("Negotiation diary: %s", negotiation_summary())

    if write_html:
        out = f"results_{grid}.html"
        visualize_results(world, write_to=out, show=False)
        logger.info("Results written to %s", out)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restoration scenario runner.")
    p.add_argument(
        "--grid",
        default="simbench_lv",
        choices=sorted(GRIDS.keys()),
        help="Network factory to use (default: simbench_lv).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for failure selection.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip writing the visualization HTML.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(args.grid, seed=args.seed, write_html=not args.no_html))
