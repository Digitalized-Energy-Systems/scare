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
python -m experiment.restoration
```

## Installation

```bash
pip install -e .
```

All dependencies are resolved from local clones under `~/git/`.

## Package structure

```
src/scare/
├── base/
│   ├── model.py        # enums, dataclasses, message types
│   ├── util.py         # unit conversions, observation helpers, ADMM factories
│   └── visu.py         # plotly visualisation
├── community/role.py   # Community leader + CommunityParticipant
├── detection/role.py   # ProblemDetector
├── service/
│   ├── balance.py      # EnergyBalanceNegotiator (consensus chain algorithm)
│   ├── stability.py    # GenerationController
│   ├── reconfiguration.py  # GridReconfigurator + GridTieSwitchOperator
│   ├── cp.py           # EnergyConverterRole (ADMM)
│   └── chs.py          # CommunityHostingService
└── scenario/
    └── restoration.py  # create_restoration_scenario_world()

experiment/
├── restoration.py      # 30 s demo run
├── evaluation.py       # batch eval across scenarios
└── plotting.py         # comparison plots from saved results
```

## License

MIT – see [LICENSE](LICENSE).
