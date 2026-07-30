# SCARE – Community-based Adaptive Resilience for Energy Systems

A distributed multi-agent system for resilient multi-energy restoration.

Built on top of:

| Dependency | Role |
|---|---|
| [mango-agents](https://github.com/OFFIS-DAI/mango) | Agent framework, roles, simulation world |
| [mango-energy-environments](https://github.com/Digitalized-Energy-Systems/mango-energy-environments) | Multi-energy network physics & failure injection |
| [distributed-resource-optimization](https://github.com/Digitalized-Energy-Systems/distributed-resource-optimization) | ADMM cross-sector optimisation |
| [monee](https://github.com/Digitalized-Energy-Systems/monee) | Multi-energy network model |
| [plotly](https://plotly.com/python/) | Visualisation |
| [networkx](https://networkx.org/) | Graph algorithms |

## Architecture

```
Network component          Agent roles
─────────────────          ─────────────────────────────────────────────
Child (load / gen)    →    EnergyBalanceNegotiator + GenerationController
Node (bus)            →    ProblemDetector + GridReconfigurator
  ↳ CHP / P2G / G2P   →    + EnergyConverterRole (ADMM)
Branch (switchable)   →    GridTieSwitchOperator
Branch (heat exch.)   →    EnergyBalanceNegotiator + GenerationController
Branch (P2G / G2P)    →    EnergyConverterRole (ADMM)
```

Three named topologies are maintained per agent via `NamedTopologies`:

- **`groups`** – fully-connected clusters per connected component × sector (used by energy-balance negotiation)
- **`grid`** – physical network graph (used by grid reconfiguration path-finding)
- **`cps`** – cross-sector coupling points (used by CP optimisation)

## Quick start

```python
import asyncio
from mango_energy_environments import Failure, fetch_example_net
from scare.scenario.restoration import (
    create_restoration_scenario_world,
    start_restoration_simulation,
)


async def main():
    net = fetch_example_net()
    world = create_restoration_scenario_world(net)
    failures = [Failure(delay_s=2.0, branch_ids=[(3, 4)])]
    await start_restoration_simulation(world, failures, simulation_duration_s=30.0)


asyncio.run(main())
```

Or run the ready-made experiment:

```bash
python -m experiment.scenarios
```

## Installation

The four source dependencies above are checkouts, not releases, so they are not
listed in `pyproject.toml` (PEP 508 `file://` URLs must be absolute, which would
pin the project to one machine). Clone them as siblings of this repo and install
them first, in this order:

```bash
git clone -b feature-transactional-apis https://github.com/OFFIS-DAI/mango.git ../mango
git clone https://github.com/Digitalized-Energy-Systems/monee.git ../monee
git clone https://github.com/Digitalized-Energy-Systems/distributed-resource-optimization.git ../distributed-resource-optimization
git clone https://github.com/Digitalized-Energy-Systems/mango-energy-environments.git ../mango-energy-environments

pip install -e ../mango -e ../monee -e ../distributed-resource-optimization -e ../mango-energy-environments
pip install -e ".[dev]"
```

> **The mango branch matters.** SCARE requires `feature-transactional-apis`. A
> package named `mango-agents` also exists on PyPI; it is different code and
> will silently shadow this one if installed from the index.

Gurobi is used for the oracle and reconfiguration solves and needs a licence.

## Running

```bash
pytest tests/                          # test suite
PYTHONHASHSEED=0 pytest tests/         # as CI runs it -- see "Reproducibility"

python -m experiment.scenarios         # demo CLI: run one scenario
python -m experiment.hpc.run_local     # run a campaign locally
python -m experiment.eval.run          # build figures/report from a campaign
```

Wrappers in `scripts/` are the usual entry points -- `run_local.sh` and
`plot.sh` locally, `submit_campaign.sh` and `submit_plot.sh` on the cluster.

### Reproducibility

Set `PYTHONHASHSEED=0` for any run whose numbers you intend to compare, and
thread an explicit seed into failure sampling. Iteration order over sets and
dicts feeds coordination-identity decisions, so an unpinned hash seed makes two
runs of the same scenario incomparable.

## Package structure

```
src/scare/
├── base/                       # foundation + cross-cutting infrastructure
│   ├── model.py                #   enums, dataclasses, message types
│   ├── channel.py              #   typed pub/sub Decision primitives
│   ├── util.py                 #   unit conversions, observation & registry helpers
│   ├── config.py               #   RestorationConfiguration
│   ├── runtime/                #   sim plumbing: diagnostics, solver_guard,
│   │                           #     infeasibility_capture, comms (perturbation)
│   ├── topology/               #   topology_mirror + graph partitioning (community)
│   ├── optimization/           #   ADMM role glue + flex-actor factories
│   └── viz/                    #   plotly visualisation
├── community/                  # L2/L2.5 holonic community formation & coalitions
├── detection/role.py           # ProblemDetector
├── service/                    # agent control roles, grouped by concern
│   ├── balance/                #   gossip energy-balance (negotiator, gossip_math, trust)
│   ├── coupling/               #   L3 cross-sector coupling points (cp*, dynamic_connector)
│   ├── control/                #   L1 reactive control & enforcement (constraints,
│   │                           #     stability, voltage_droop, slack_budget, curtailment, …)
│   └── reconfiguration.py      #   GridReconfigurator + GridTieSwitchOperator
└── scenario/
    ├── restoration.py          # create_restoration_scenario_world()
    └── failure_sampling.py     # scenario failure injection

experiment/
├── scenarios/          # grid builders (GRIDS), apply_* stress modifiers, priorities; demo CLI
├── eval/               # canonical evaluation pipeline (claims, metrics, plots, report)
└── hpc/                # SLURM campaign driver (plan, submit, runner, aggregate)
```

## License

MIT – see [LICENSE](LICENSE).
