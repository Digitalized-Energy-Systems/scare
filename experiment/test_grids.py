import time

from monee import run_energy_flow

from experiment.scenarios import GRIDS

if __name__ == "__main__":
    for key, grid in GRIDS.items():
        start = time.time()
        g = grid()

        print(time.time() - start)

        start = time.time()
        result = run_energy_flow(g, solver="gurobi")
        print(result.dataframes["GenericPowerBranch"])
        print(time.time() - start)

        print(key)
        print(result.success)
