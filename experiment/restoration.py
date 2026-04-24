import asyncio
import logging

from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import create_urban_district_net

from scare.base.util import create_failures
from scare.base.visu import visualize_results
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")
logging.getLogger("scare").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SIMULATION_DURATION_S = 30.0
FAILURE_DELAY_S = 2.0

async def run() -> None:
    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)

    logger.info("Building restoration world …")
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

    visualize_results(world, write_to="results.html", show=False)
    logger.info("Results written to results.html")


if __name__ == "__main__":
    asyncio.run(run())
