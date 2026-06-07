"""Multi-energy grid builders and the named-grid registry (``GRIDS``)."""

import logging
import random
from collections.abc import Callable
from statistics import median

import monee.express as mx
import simbench
from monee.io.from_pandapower import from_pandapower_net
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import generate_supply_return_mes_based_on_power_net

logger = logging.getLogger(__name__)


def create_large_lv_simbench(
    density,
    *,
    simbench_code: str = "1-LV-rural3--1-no_sw",
    backup_lines_per_sector: int = 0,
    backup_seed: int | None = None,
    cp_size_multiplier: float = 1.0,
    replace_primary_generation: bool = False,
):
    """Build a simbench LV multi-energy network.

    The grid uses an unconstrained slack so the energy-flow LP always
    converges; slack-budget shaping is a per-scenario policy applied via
    :func:`apply_slack_budget`.

    Parameters
    ----------
    density:
        Coupling-point density forwarded to monee's MES generator
        (controls how many CHP / P2G / P2H plants are placed).
    simbench_code:
        simbench network code; default ``1-LV-rural3--1-no_sw`` is the
        ~340-load benchmark, smaller variants (rural1, semiurb4) are for
        the scaling experiment.
    backup_lines_per_sector:
        If ``> 0``, add that many normally-open backup branches per
        sector via ``add_backup_lines`` — the reconfiguration fixture;
        without them the radial grid has no alternative paths.
    backup_seed:
        RNG seed for reproducible backup placement.
    cp_size_multiplier:
        Scales every coupling-point's rated output uniformly (1.0 =
        monee's per-bus default). Larger CPs amplify cross-sector
        substitution potential.
    replace_primary_generation:
        When True, each CP's rated output is absorbed from the matching
        primary generation pool, keeping total per-carrier rated
        production invariant. Reframes CP-as-redundancy (additive
        default) into CP-as-cross-carrier-dependence: losing a CP then
        disables both the unit and the primary gen it displaced.
    """

    def create():
        net = simbench.get_simbench_net(simbench_code)
        mn = from_pandapower_net(net)
        mes = generate_supply_return_mes_based_on_power_net(
            mn,
            coupling_density=density,
            centralized=False,
            couplings=("chp", "p2g", "p2h"),
            coupling_kwargs={
                "seed": 1,
                "use_hg_variants": True,
                "cp_size_multiplier": cp_size_multiplier,
                "replace_primary_generation": replace_primary_generation,
            },
            heat_kwargs={"node_based_heat_loads": True},
        )
        mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
        # McCormick is unneeded for energy flow and can make failures
        # infeasible via envelope bounds.
        # mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=16))

        if backup_lines_per_sector > 0:
            add_backup_lines(
                mes,
                n_per_sector=backup_lines_per_sector,
                seed=backup_seed,
            )
        return mes

    return create


GRIDS: dict[str, Callable[[], "object"]] = {
    # Slack budget is a per-scenario knob (``scenario.slack_budget_pct``),
    # not baked into any grid, so one base grid serves multiple operator
    # policies and the energy-flow LP always has a free slack.
    # Coupling-density variants — vary only the number of CP plants.
    "simbench_lv_low": create_large_lv_simbench(0.1),
    "simbench_lv": create_large_lv_simbench(0.2),
    "simbench_lv_high": create_large_lv_simbench(0.3),
    # Scaling pillar — three LV variants spanning ~a decade in node count:
    # small  (~15 buses, 1-LV-rural1)
    # medium (~44 buses, 1-LV-semiurb4)
    # large  (~129 buses, 1-LV-rural3 — the default ``simbench_lv``)
    # Same coupling density; slack budget keeps operator policy orthogonal.
    "simbench_lv_small": create_large_lv_simbench(
        0.2, simbench_code="1-LV-rural1--1-no_sw"
    ),
    "simbench_lv_medium": create_large_lv_simbench(
        0.2, simbench_code="1-LV-semiurb4--1-no_sw"
    ),
    # Reconfiguration pillar — default LV grid plus five seeded backup
    # branches per sector so the GridReconfigurator has alternative paths
    # when a primary line trips.
    "simbench_lv_reconfig": create_large_lv_simbench(
        0.2,
        backup_lines_per_sector=5,
        backup_seed=0,
    ),
    # CP-flexibility pillar — exercise ``cp_size_multiplier`` and
    # ``replace_primary_generation``:
    # ``cp_heavy``           additive CPs at 2× rating; primary fleet can
    #   still recover a lost CP, but the inflated capacity makes the
    # cross-sector flex shift visible when the CP-ADMM layer engages.
    # ``cp_dependent``       CPs at 1× rating that replace primary gen;
    # per-carrier rated production is invariant, but losing a CP also
    # loses the displaced primary — the regime where CP-ADMM
    # cross-sector substitution is load-bearing (gossip + islanding
    #   alone cannot recover the lost CP).
    # ``cp_heavy_dependent`` both knobs maxed: maximum cross-sector
    #   dependence; losing one CP produces a deep deficit solvable only
    # by re-routing through surviving CHP / G2P / P2H plants.
    "simbench_lv_cp_heavy": create_large_lv_simbench(
        0.3,
        cp_size_multiplier=2.0,
        replace_primary_generation=False,
    ),
    "simbench_lv_cp_dependent": create_large_lv_simbench(
        0.3,
        cp_size_multiplier=1.0,
        replace_primary_generation=True,
    ),
    "simbench_lv_cp_heavy_dependent": create_large_lv_simbench(
        0.3,
        cp_size_multiplier=2.0,
        replace_primary_generation=True,
    ),
}


def add_backup_lines(
    mes: "object",
    *,
    n_per_sector: int = 3,
    seed: int | None = None,
) -> dict[str, list[tuple]]:
    """Augment ``mes`` with normally-open backup branches in every sector.

    Backups connect pairs of leaf nodes (degree 1 in the sector
    subgraph) from opposite halves of the sorted leaf list, so each
    shortcuts a structurally long radial path — the alternative the
    ``GridReconfigurator`` path search looks for.

    Branch parameters use the median of existing branches in the sector
    for physical plausibility. ``on_off = 0`` keeps the backup inert
    until the reconfigurator closes the switch via
    ``behavior.act("switch")``. The PowerLine variant also sets
    ``backup=True`` so monee's ``controllable_backup_lines`` recognises it.

    Parameters
    ----------
    mes:
        A monee multi-energy network.
    n_per_sector:
        Backup branches per sector. 3 suits a small LV grid; scale
        roughly with leaf count for larger grids.
    seed:
        Optional RNG seed for reproducible placement.

    Returns
    -------
    dict
        ``{sector_name: [branch_id, ...]}`` of newly added branch ids.
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    sector_specs: dict[str, dict] = {
        "electricity": {
            "grid_match": "power",
            "branch_attr": ("length_m", "r_ohm_per_m", "x_ohm_per_m"),
            "creator": mx.create_line,
        },
        "gas": {
            "grid_match": "gas",
            "branch_attr": ("length_m", "diameter_m"),
            "creator": mx.create_gas_pipe,
        },
        "heat": {
            "grid_match": "water",
            "branch_attr": ("length_m", "diameter_m"),
            "creator": mx.create_water_pipe,
        },
    }

    added: dict[str, list[tuple]] = {}

    for sector, spec in sector_specs.items():
        sector_node_ids: list = []
        for node in mes.nodes:
            grid_name = str(getattr(node.grid, "name", "") or "").lower()
            if spec["grid_match"] in grid_name:
                sector_node_ids.append(node.id)
        sector_node_set = set(sector_node_ids)

        # Adjacency in this sector only.
        adj: dict = {nid: set() for nid in sector_node_ids}
        sector_branches: list = []
        for branch in mes.branches:
            a, b = branch.id[0], branch.id[1]
            if a in sector_node_set and b in sector_node_set:
                adj[a].add(b)
                adj[b].add(a)
                sector_branches.append(branch)

        if len(sector_node_ids) < 2 or not sector_branches:
            added[sector] = []
            continue

        # Degree-1 nodes are the tie-back candidates; fall back to all
        # nodes if the topology has no leaves (e.g. a closed loop).
        leaves = sorted(n for n in sector_node_ids if len(adj[n]) <= 1)
        if len(leaves) < 2:
            leaves = sorted(sector_node_ids)

        # Median branch params for this sector.
        def _median_attr(attr: str) -> float:
            vals = []
            for br in sector_branches:
                v = getattr(br.model, attr, None)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        continue
            return float(median(vals)) if vals else 1.0

        params = {a: _median_attr(a) for a in spec["branch_attr"]}

        # Pair leaf i with leaf (i + n//2): bridges head-to-tail of the
        # sorted leaf list, tending to link separate feeders. Skip pairs
        # already directly adjacent.
        n_leaves = len(leaves)
        offset = max(1, n_leaves // 2)
        candidate_pairs = []
        for i in range(n_leaves):
            j = (i + offset) % n_leaves
            if j <= i:
                continue
            a, b = leaves[i], leaves[j]
            if b in adj[a]:
                continue
            candidate_pairs.append((a, b))

        rng.shuffle(candidate_pairs)
        new_ids: list[tuple] = []
        for a, b in candidate_pairs:
            if len(new_ids) >= n_per_sector:
                break
            try:
                bid = _create_backup_branch(spec["creator"], mes, a, b, params, sector)
                new_ids.append(bid)
                # Record the edge so later iterations don't re-pick it.
                adj[a].add(b)
                adj[b].add(a)
            except Exception as exc:
                # Skip pairs the constructor rejects (e.g. incompatible
                # grid attributes).
                logger.debug(
                    "Backup branch creation skipped for (%s, %s) in %s: %s",
                    a,
                    b,
                    sector,
                    exc,
                )

        added[sector] = new_ids

    return added


def _create_backup_branch(creator, mes, from_id, to_id, params, sector: str):
    """Dispatch to the per-sector ``monee.express.create_*`` helper,
    marking the new branch ``backup=True`` and ``on_off=0``."""
    if sector == "electricity":
        bid = creator(
            mes,
            from_node_id=from_id,
            to_node_id=to_id,
            length_m=params.get("length_m", 50.0),
            r_ohm_per_m=params.get("r_ohm_per_m", 0.0001),
            x_ohm_per_m=params.get("x_ohm_per_m", 0.00007),
            on_off=0,
            name=f"backup_el_{from_id}_{to_id}",
        )
    elif sector == "gas":
        bid = creator(
            mes,
            from_node_id=from_id,
            to_node_id=to_id,
            length_m=params.get("length_m", 50.0),
            diameter_m=params.get("diameter_m", 0.05),
            on_off=0,
            name=f"backup_gas_{from_id}_{to_id}",
        )
    elif sector == "heat":
        bid = creator(
            mes,
            from_node_id=from_id,
            to_node_id=to_id,
            length_m=params.get("length_m", 50.0),
            diameter_m=params.get("diameter_m", 0.05),
            on_off=0,
            name=f"backup_heat_{from_id}_{to_id}",
        )
    else:
        raise ValueError(f"Unknown sector: {sector!r}")

    # Tag as backup so monee's LP and post-run analysis can identify it.
    # PowerLine has a native ``backup`` field; gas/water pipes don't, so
    # attach the attribute either way.
    branch = mes.branch_by_id(bid)
    try:
        branch.model.backup = True
    except Exception:
        pass
    return bid
