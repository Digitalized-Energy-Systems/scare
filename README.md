# SCARE – Community-based Adaptive Resilience for Energy Systems

A distributed multi-agent system for resilient multi-energy restoration.

Built on top of:

| Dependency | Role |
|---|---|
| [mango-agents](https://github.com/OFFIS-DAI/mango) | Agent framework, roles, simulation world |
| [mango-energy-environments](../mango-energy-environments) | Multi-energy network physics & failure injection |
| [distributed-resource-optimization](../mango-optimization) | ADMM cross-sector optimisation |
| [monee](../monee) | Multi-energy network model |
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

```bash
pip install -e .
```

All dependencies are resolved from local clones under `~/git/`.

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
