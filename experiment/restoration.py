import argparse
import asyncio
import logging
from collections.abc import Callable

import simbench
from monee.io.from_pandapower import from_pandapower_net
from monee.model.formulation import (
    MISOCP_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
)
from monee.network import (
    generate_supply_return_mes_based_on_power_net,
)

from scare.base.diagnostics import install_solver_failure_dump
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

install_solver_failure_dump()

SIMULATION_DURATION_S = 5.0
FAILURE_DELAY_S = 2.0

def create_large_lv_simbench(
    density,
    *,
    slack_budget_pct: float | None = None,
    simbench_code: str = "1-LV-rural3--1-no_sw",
    backup_lines_per_sector: int = 0,
    backup_seed: int | None = None,
):
    """Build a simbench LV multi-energy network.

    Parameters
    ----------
    density:
        Coupling-point density passed straight to monee's MES generator
        (controls how many CHP / P2G / P2H plants are placed).
    slack_budget_pct:
        When set, caps the external power and gas grid injections at
        this fraction of the network's total nominal load.  Defaults
        to ``None`` (unbounded slack — original behaviour).  Setting
        to e.g. 0.5 forces the MAS to actually choose what to serve
        post-failure rather than letting the slack absorb everything.
    simbench_code:
        simbench network code; the default ``1-LV-rural3--1-no_sw`` is
        the established ~340-load benchmark, but smaller variants
        (rural1, semiurb4) are useful for the scaling experiment.
    backup_lines_per_sector:
        If ``> 0``, augment the network with that many normally-open
        backup branches per sector via ``add_backup_lines``.  This is
        the test fixture for the reconfiguration pillar — without
        them the grid is purely radial and the GridReconfigurator has
        no alternative paths to discover.
    backup_seed:
        RNG seed for backup placement (reproducible test fixtures).
    """

    def create():
        net = simbench.get_simbench_net(simbench_code)
        mn = from_pandapower_net(net)
        mes = generate_supply_return_mes_based_on_power_net(
            mn,
            coupling_density=density,
            centralized=False,
            couplings=("chp", "p2g", "p2h"),
            coupling_kwargs={"seed": 1, "use_hg_variants": True},
            heat_kwargs={"node_based_heat_loads": True},
        )
        mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
        mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=16))
        if slack_budget_pct is not None:
            _bound_external_slack(mes, slack_budget_pct)
        if backup_lines_per_sector > 0:
            add_backup_lines(
                mes,
                n_per_sector=backup_lines_per_sector,
                seed=backup_seed,
            )
        return mes
    return create


def _bound_external_slack(mes, fraction: float) -> None:
    """Tighten ``ExtPowerGrid`` and ``ExtHydrGrid`` import bounds so the
    slack can't absorb every imbalance the MAS introduces.

    Cap is computed per-sector from the sum of nominal load magnitudes
    in that sector.  ``fraction`` is the share of total demand that
    the external grid is allowed to supply (e.g. 0.5 means at most
    50 % of nominal demand can come from upstream).  Generators on
    the network make up the remainder.
    """
    from monee.model.child import (
        ExtHydrGrid,
        ExtPowerGrid,
        PowerLoad,
        Sink,
    )

    # Aggregate nominal demand per sector.  Heat is excluded — heat-side
    # ExtHydrGrid imports mass-flow at fixed supply temperature, and
    # its consumption is constrained by the WaterPipe / HeatExchanger
    # physics, not by load magnitude.  Cap only the power and gas
    # external grids since those have no other resource limit.
    #
    # Sinks live on both gas *and* water junctions in monee's MES
    # convention, so route them by parent-node grid name — counting
    # heat-side Sinks toward the gas budget would inflate the cap by
    # roughly 4× and make the constraint inert in practice.
    total_p_mw = 0.0
    total_gas_mass_kgs = 0.0
    for child in mes.childs:
        m = child.model
        if isinstance(m, PowerLoad):
            total_p_mw += abs(getattr(m, "p_mw", 0.0))
        elif isinstance(m, Sink):
            try:
                grid_name = str(
                    getattr(mes.node_by_id(child.node_id).grid, "name", "")
                ).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name:
                total_gas_mass_kgs += abs(getattr(m, "mass_flow", 0.0))

    cap_p_mw = max(1e-3, fraction * total_p_mw)
    cap_gas_mass_kgs = max(1e-4, fraction * total_gas_mass_kgs)

    for child in mes.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid) and hasattr(m, "p_mw") and hasattr(m.p_mw, "min"):
            # The Var's ``min`` is set to ``-max_import_mw`` (consumption
            # convention — so import bound is the magnitude on min).
            m.p_mw.min = -cap_p_mw
            m.p_mw.max = cap_p_mw
        elif isinstance(m, ExtHydrGrid) and hasattr(m, "mass_flow") and hasattr(m.mass_flow, "min"):
            # Distinguish gas vs heat (water) by parent-node grid type.
            try:
                grid_name = str(getattr(mes.node_by_id(child.node_id).grid, "name", "")).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name:
                m.mass_flow.min = -cap_gas_mass_kgs
                m.mass_flow.max = cap_gas_mass_kgs
            # heat-side ExtHydrGrid intentionally left unbounded


GRIDS: dict[str, Callable[[], "object"]] = {
    # Coupling-density variants (unbounded slack — easy regime).
    "simbench_lv_low": create_large_lv_simbench(0.1),
    "simbench_lv": create_large_lv_simbench(0.5),
    "simbench_lv_high": create_large_lv_simbench(0.9),
    # Slack-budget variants.  These force the MAS to actually allocate
    # scarce capacity instead of letting the external grid absorb every
    # imbalance.  Mid coupling density; the budget is the discriminator.
    # Headroom (slack + local gen vs nominal load) for this grid:
    #   _30  → 87 %  (shedding required throughout)
    #   _45  → 97 %  (knife-edge — light shedding)
    #   _60  → 107 % (loose — discriminates against a stronger baseline)
    # Earlier 50/80 variants were dropped because the slack alone exceeded
    # demand and no shedding was ever required.
    "simbench_lv_constrained_60": create_large_lv_simbench(0.5, slack_budget_pct=0.6),
    "simbench_lv_constrained_45": create_large_lv_simbench(0.5, slack_budget_pct=0.45),
    "simbench_lv_constrained_30": create_large_lv_simbench(0.5, slack_budget_pct=0.3),
    # Scaling pillar.  Three simbench LV variants giving roughly a
    # decade of range in node count: small (~15 buses), medium (~44),
    # large (~129; the default ``simbench_lv``).  All built with the
    # same coupling density and the mid 45 % slack budget so the
    # only varying axis is grid size.
    "simbench_lv_small_constrained_45": create_large_lv_simbench(
        0.5, slack_budget_pct=0.45, simbench_code="1-LV-rural1--1-no_sw"
    ),
    "simbench_lv_medium_constrained_45": create_large_lv_simbench(
        0.5, slack_budget_pct=0.45, simbench_code="1-LV-semiurb4--1-no_sw"
    ),
    # ``simbench_lv_constrained_45`` above is the "large" point of
    # this scaling sweep.

    # Reconfiguration pillar.  Same constrained large LV grid plus
    # five backup branches per sector so the GridReconfigurator has
    # alternative paths to discover when a primary line trips.
    "simbench_lv_reconfig": create_large_lv_simbench(
        0.5, slack_budget_pct=0.45,
        backup_lines_per_sector=5, backup_seed=0,
    ),
}


def add_backup_lines(
    mes: "object",
    *,
    n_per_sector: int = 3,
    seed: int | None = None,
) -> dict[str, list[tuple]]:
    """Augment ``mes`` with normally-open backup branches in every sector.

    Backup branches are added between pairs of *leaf* nodes (degree 1
    in the sector subgraph) chosen from opposite halves of the sorted
    leaf list, so each backup shortcuts a structurally long path
    through the radial topology — exactly the kind of alternative the
    ``GridReconfigurator``'s path search is designed to find.

    Branch parameters are taken from the median of existing branches
    in the same sector so the backup is physically plausible (not a
    superconductor, not a hair-thin pipe).  ``on_off`` is set to 0 so
    the backup is *electrically/hydraulically inert* until the
    reconfigurator closes the switch via ``behavior.act("switch")``.
    The PowerLine variant also sets ``backup=True`` so monee's
    LP-side helpers (``controllable_backup_lines``) can recognise it.

    Parameters
    ----------
    mes:
        A monee multi-energy network.
    n_per_sector:
        How many backup branches to add per sector.  3 covers a small
        LV grid; for larger grids, scale roughly linearly with the
        number of leaves.
    seed:
        Optional RNG seed for reproducible backup placement.

    Returns
    -------
    dict
        ``{sector_name: [branch_id, ...]}`` of the newly added branch
        ids, useful for downstream analysis (e.g. measuring how many
        of them the reconfigurator actually closes).
    """
    import random as _random
    from statistics import median

    import monee.express as mx

    rng = _random.Random(seed) if seed is not None else _random.Random()

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
        # Per-sector node list and edge set.
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

        # Leaves of the sector subgraph — degree-1 nodes are the
        # natural candidates for tie-back branches.  Fall back to all
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

        # Pair the i-th leaf with the (i + n//2)-th — connects the
        # head of the leaf list to the tail, which on a sorted-by-id
        # leaf set tends to bridge separate feeder branches.  Skip
        # pairs that are already directly adjacent.
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
                # Record the branch so subsequent iterations don't
                # re-pick the same edge.
                adj[a].add(b)
                adj[b].add(a)
            except Exception as exc:
                # If the underlying constructor rejects the pair (e.g.
                # incompatible grid attributes), skip and keep going.
                import logging

                logging.getLogger(__name__).debug(
                    "Backup branch creation skipped for (%s, %s) in %s: %s",
                    a, b, sector, exc,
                )

        added[sector] = new_ids

    return added


def _create_backup_branch(creator, mes, from_id, to_id, params, sector: str):
    """Dispatch to the right ``monee.express.create_*`` helper with
    sensible per-sector parameter defaults.  Marks the resulting
    branch as ``backup=True`` and ``on_off=0``."""
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

    # Tag the new branch as backup so monee's LP and any post-run
    # analysis can identify it.  PowerLine carries a native ``backup``
    # field; gas/water pipes don't, so we attach the attribute either
    # way and let downstream code check.
    branch = mes.branch_by_id(bid)
    try:
        branch.model.backup = True
    except Exception:
        pass
    return bid


def assign_load_priorities(
    monee_net: "object",
    *,
    seed: int = 0,
    distribution: str = "skewed",
) -> dict[str, int]:
    """Assign per-load priority tiers (1 = most critical, 10 = least).

    Returns a ``priorities`` dict keyed by ``child-{id}`` suitable for
    ``create_restoration_scenario_world(priorities=...)``.  Generators
    (cap < 0) and CPs are skipped — they default to tier 0 in
    ``obs_priority``.

    ``distribution`` knobs:

    - ``"uniform"``  — uniform over [1, P].  Maximally diverse;
      stress-tests the priority-aware waterfall.
    - ``"skewed"``    — realistic: ~10 % critical (tier 1-2),
      ~70 % residential (tier 4-6), ~20 % curtailable (tier 8-10).
      The default; matches the empirical mix on the simbench LV
      grids and is the regime most likely to make Level-2 ADMM's
      priority-weighted allocation discriminable.
    - ``"by_capacity"`` — large loads get higher priority (tier 1)
      and small loads lower (tier 10).  Models the "feed the big
      hospitals first" heuristic.
    - ``"all_one"``  — everyone tier 1 (legacy behaviour, preserved
      so ablation comparisons against the no-diversity case remain
      possible).

    ``seed`` makes the random assignments deterministic.
    """
    import random as _random

    from scare.base.util import obs_capacity

    rng = _random.Random(seed * 7919 + 31)
    P = 10
    out: dict[str, int] = {}

    for child in monee_net.childs:
        # Read capacity directly from the model (no behavior yet); the
        # sign convention matches obs_capacity downstream.
        obs = dict(child.model.values)
        cap = obs_capacity(obs)
        if cap <= 0:
            # Generators and unknown — skip; they default to tier 0
            continue
        aid = f"child-{child.id}"
        if distribution == "uniform":
            out[aid] = rng.randint(1, P)
        elif distribution == "skewed":
            r = rng.random()
            if r < 0.10:
                out[aid] = rng.randint(1, 2)
            elif r < 0.80:
                out[aid] = rng.randint(4, 6)
            else:
                out[aid] = rng.randint(8, 10)
        elif distribution == "by_capacity":
            # Computed below in a second pass since it needs the
            # global capacity distribution.
            out[aid] = -1  # sentinel
        elif distribution == "all_one":
            out[aid] = 1
        else:
            raise ValueError(f"unknown priority distribution: {distribution}")

    if distribution == "by_capacity":
        # Bin by capacity quantiles — top decile to tier 1, bottom decile to tier 10.
        items = []
        for child in monee_net.childs:
            obs = dict(child.model.values)
            cap = obs_capacity(obs)
            if cap > 0:
                items.append((cap, f"child-{child.id}"))
        items.sort(reverse=True)  # largest first
        n = len(items)
        for rank, (_cap, aid) in enumerate(items):
            tier = 1 + int(rank * P / max(n, 1))
            out[aid] = max(1, min(P, tier))

    return out


def apply_cold_day(
    mes: "object",
    *,
    supply_t_k: float = 343.15,
    heat_load_scale: float = 1.5,
) -> None:
    """Mutate ``mes`` in place to simulate a cold-day stress scenario.

    Two knobs:

    - ``supply_t_k``: the heat-side ``ExtHydrGrid`` slack supply
      temperature (default ~70 °C, vs the unstressed ~83 °C).  Lower
      supply temperatures shrink the headroom downstream junctions
      have against the lower heat bound (60 °C / 333.15 K), so any
      transport delay or pipe loss can push them into violation.
    - ``heat_load_scale``: multiplier on every ``HeatLoad.q_mw_heat``
      (default 1.5×).  Makes the heat sector harder to balance against
      the available generation and thermal-corridor capacity.

    Idempotent in the sense of "build a fresh net then call once" — do
    not call this twice on the same net or the load scale stacks.
    """
    from monee.model.child import ExtHydrGrid, HeatLoad

    for child in mes.childs:
        m = child.model
        if isinstance(m, HeatLoad):
            m.q_mw_heat = float(m.q_mw_heat) * heat_load_scale
        elif isinstance(m, ExtHydrGrid):
            try:
                grid_name = str(getattr(mes.node_by_id(child.node_id).grid, "name", "")).lower()
            except Exception:
                grid_name = ""
            if "water" in grid_name or "heat" in grid_name:
                m.t_k = supply_t_k


async def run(grid: str, seed: int | None = None, write_html: bool = True) -> None:
    if seed is not None:
        import random
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

    from scare.base.diagnostics import negotiation_summary

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
