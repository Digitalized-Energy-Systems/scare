import argparse
import asyncio
import logging
import random
from collections.abc import Callable
from statistics import median

import monee.express as mx
import simbench
from monee import enable_islanding
from monee.io.from_pandapower import from_pandapower_net
from monee.model.child import (
    ExtHydrGrid,
    ExtPowerGrid,
    HeatLoad,
    PowerGenerator,
    PowerLoad,
    Sink,
    Source,
)
from monee.model.extension import (
    GasLinepack,
    GridFormingGenerator,
    GridFormingSource,
    LumpedThermalCapacitance,
)
from monee.model.formulation import (
    MISOCP_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
)
from monee.network import (
    generate_supply_return_mes_based_on_power_net,
)

from scare.base.diagnostics import install_solver_failure_dump, negotiation_summary
from scare.base.util import create_failures, obs_capacity
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


_SLACK_LP_HEADROOM_FACTOR: float = 10.0


def apply_microgrid_islanding(
    mes: "object",
    *,
    carriers: "frozenset[str] | set[str] | tuple[str, ...] | list[str]" = ("electricity", "water", "gas"),
    promote_all_generators: bool = False,
    grid_former_aids: "tuple[str, ...] | list[str] | set[str]" = (),
) -> dict[str, int]:
    """Enable monee's islanding extension on *mes* and optionally convert
    selected generator-class children into ``GridForming*`` types so the
    extension has grid-formers to anchor sub-islands on.

    Args:
        carriers:
            Carrier names to enable islanding on (``"electricity"``,
            ``"water"``, ``"gas"``). Forwards True/None per carrier;
            does not customise islanding modes.
        promote_all_generators:
            When True, every eligible generator-class child
            (PowerGenerator, gas/water Source) becomes the corresponding
            ``GridFormingGenerator`` / ``GridFormingSource`` so any
            sub-island containing one is solvable on its own. When False,
            only children whose aid is in ``grid_former_aids`` are promoted.
        grid_former_aids:
            Per-child opt-in aids of the form ``"child-{id}"``; ignored
            when ``promote_all_generators`` is True.

    Returns:
        ``{carrier: n_promoted}`` — children converted per sector.

    Side effects:
        Calls :func:`monee.enable_islanding`; the resulting
        ``NetworkIslandingConfig`` is attached and respected by every
        subsequent solve (Pyomo / Gekko).

    Notes:
        Promotion uses approximate ratings from the child's setpoint
        magnitude — sufficient for eligibility (the LP needs a leading
        reference per island, not a precise capacity).
    """
    carrier_set = frozenset(carriers)
    if not carrier_set:
        return {"electricity": 0, "water": 0, "gas": 0}

    promote_aids = set(grid_former_aids) if grid_former_aids else set()
    counters = {"electricity": 0, "water": 0, "gas": 0}

    for child in mes.childs:
        aid = f"child-{child.id}"
        if not (promote_all_generators or aid in promote_aids):
            continue
        try:
            node = mes.node_by_id(child.node_id)
        except Exception:
            continue
        grid_name = str(getattr(node.grid, "name", "") or "").lower()
        m = child.model

        if isinstance(m, PowerGenerator) and "power" in grid_name and "electricity" in carrier_set:
            p_max = max(1e-6, abs(float(getattr(m, "p_mw", 0.0) or 0.0)))
            q_max = max(1e-6, abs(float(getattr(m, "q_mvar", 0.0) or 0.0)) + 0.1 * p_max)
            child.model = GridFormingGenerator(
                p_mw_max=p_max, q_mvar_max=q_max, vm_pu=1.0,
            )
            counters["electricity"] += 1
        elif isinstance(m, Source):
            # Sources live on both gas and water nodes; route by parent
            # grid so the right carrier is counted and anchored.
            mass_max = max(1e-6, abs(float(getattr(m, "mass_flow", 0.0) or 0.0)))
            if "gas" in grid_name and "gas" in carrier_set:
                child.model = GridFormingSource(
                    pressure_pu=1.0, mass_flow_max=mass_max,
                )
                counters["gas"] += 1
            elif "water" in grid_name and "water" in carrier_set:
                child.model = GridFormingSource(
                    pressure_pu=1.0, t_k=356.0, mass_flow_max=mass_max,
                )
                counters["water"] += 1

    # None per-carrier leaves that carrier's islanding disabled.
    enable_islanding(
        mes,
        electricity=True if "electricity" in carrier_set else None,
        gas=True if "gas" in carrier_set else None,
        water=True if "water" in carrier_set else None,
    )
    return counters


def apply_temporal_extensions(
    mes: "object",
    *,
    linepack: bool = False,
    ltc: bool = False,
    ltc_default_t_init: float | None = None,
) -> dict[str, int]:
    """Attach monee's temporal-storage extensions to *mes*.

    ``GasLinepack`` (per-pipe ``linepack_kg`` / ``net_pack_kgs`` vars) and
    ``LumpedThermalCapacitance`` (per-water-junction ρ·V thermal mass)
    only activate their dynamics in monee's timeseries solver; the
    single-step ``energyflow`` used here pins ``net_pack_kgs = 0`` and
    emits no LTC inertia term. Attaching them is thus an agent-side
    compatibility check (agents must tolerate the augmented obs schema),
    not a flexibility benchmark.

    Returns ``{"linepack_pipes": n, "ltc_junctions": n}`` so the caller
    can confirm the extension found something to attach to.
    """
    counters = {"linepack_pipes": 0, "ltc_junctions": 0}
    if linepack:
        ext = GasLinepack()
        mes.add_extension(ext)
        # ``prepare`` materialises per-branch state; call it early for
        # the counter (the solver also calls it).
        try:
            ext.prepare(mes)
        except Exception:  # noqa: BLE001 — count is best-effort
            pass
        counters["linepack_pipes"] = len(getattr(ext, "_active_branches", ()))
    if ltc:
        kwargs: dict = {}
        if ltc_default_t_init is not None:
            kwargs["default_t_init"] = float(ltc_default_t_init)
        ext = LumpedThermalCapacitance(**kwargs)
        mes.add_extension(ext)
        try:
            ext.prepare(mes)
        except Exception:  # noqa: BLE001
            pass
        counters["ltc_junctions"] = len(getattr(ext, "_ltc_rho_v", {}))
    return counters


def apply_slack_budget(mes, fraction: float) -> None:
    """Register the operator's slack budget on the slack agents.

    The slack budget is a soft target the MAS enforces, not a hard LP
    bound (which could make the energy-flow LP infeasible after a failure
    shifts the imbalance). Steps:

    1. Compute the operator budget per sector from nominal load (plus
       injection) magnitudes.
    2. Set the slack Var bounds to a wide physical envelope so the LP can
       always balance: ``± _SLACK_LP_HEADROOM_FACTOR · max(budget, sector
       throughput)``. The throughput term (load + generation/source
       magnitude) keeps the LP feasible on grids where CP / generation
       flow dwarfs native load; sizing off the budget alone under-sizes
       it there. This is purely a feasibility guard; the soft budget in
       step 3 is the operator policy.
    3. Stash the budget as ``_scare_slack_budget_mw`` /
       ``_scare_slack_budget_kgs`` on the slack model; the scenario-build
       hook registers it as the slack agent's rating, then multiplies by
       ``slack_target_fraction`` for the MAS-level target.

    ``fraction`` is the share of total demand the operator wants the
    external grid to supply (e.g. 0.5 = slack target is 50 % of nominal
    demand); network generators make up the remainder and the MAS drives
    toward that distribution.
    """
    # Aggregate nominal demand per sector. Heat is excluded: heat-side
    # ExtHydrGrid imports mass-flow at fixed supply temperature, bounded
    # by WaterPipe / HeatExchanger physics rather than load magnitude.
    # Cap only power and gas external grids (no other resource limit).
    #
    # Sinks live on both gas and water junctions, so route by parent-node
    # grid name — counting heat-side Sinks toward the gas budget would
    # inflate the cap ~4× and make the constraint inert.
    #
    # Also track per-sector injection magnitude (PowerGenerator,
    # gas-side Source): all slack demand is equal regardless of producer,
    # so the budget mirrors physical slack throughput (load + injection).
    # On grids with gas-fed converters a load-only budget under-shoots.
    # The feasibility envelope below uses the same throughput basis.
    total_p_mw = 0.0
    total_gas_mass_kgs = 0.0
    total_p_gen_mw = 0.0
    total_gas_source_kgs = 0.0
    total_gas_conv_kgs = 0.0
    for child in mes.childs:
        m = child.model
        if isinstance(m, PowerLoad):
            total_p_mw += abs(getattr(m, "p_mw", 0.0))
        elif isinstance(m, PowerGenerator):
            total_p_gen_mw += abs(getattr(m, "p_mw", 0.0))
        elif isinstance(m, (Sink, Source)):
            try:
                grid_name = str(
                    getattr(mes.node_by_id(child.node_id).grid, "name", "")
                ).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name:
                if isinstance(m, Sink):
                    total_gas_mass_kgs += abs(getattr(m, "mass_flow", 0.0))
                else:
                    total_gas_source_kgs += abs(getattr(m, "mass_flow", 0.0))
    # Gas-drawing cross-sector converters (CHP) are modeled at node level
    # (CHPHGControlNode), not as Sink/Source children, so the loop above
    # misses them. Their gas input is normal slack demand.
    for node in mes.nodes:
        nm = node.model
        if type(nm).__name__ == "CHPHGControlNode":
            total_gas_conv_kgs += abs(getattr(nm, "gas_kgps", 0.0))

    cap_p_mw = max(1e-3, fraction * (total_p_mw + total_p_gen_mw))
    cap_gas_mass_kgs = max(
        1e-4,
        fraction * (total_gas_mass_kgs + total_gas_source_kgs + total_gas_conv_kgs),
    )

    # Feasibility envelope: never below the sector's physical throughput
    # (load + injection it may absorb), so the LP always balances even
    # when the operator budget is tiny relative to CP / generation flow.
    lp_p_mw = _SLACK_LP_HEADROOM_FACTOR * max(cap_p_mw, total_p_mw + total_p_gen_mw)
    lp_gas_mass_kgs = _SLACK_LP_HEADROOM_FACTOR * max(
        cap_gas_mass_kgs,
        total_gas_mass_kgs + total_gas_source_kgs + total_gas_conv_kgs,
    )

    for child in mes.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid) and hasattr(m, "p_mw") and hasattr(m.p_mw, "min"):
            # Wide LP envelope keeps the solve feasible under any
            # failure-induced imbalance; the MAS owns the soft target.
            m.p_mw.min = -lp_p_mw
            m.p_mw.max = lp_p_mw
            # Soft target the MAS drives toward.
            m._scare_slack_budget_mw = cap_p_mw
        elif isinstance(m, ExtHydrGrid) and hasattr(m, "mass_flow") and hasattr(m.mass_flow, "min"):
            try:
                grid_name = str(getattr(mes.node_by_id(child.node_id).grid, "name", "")).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name:
                m.mass_flow.min = -lp_gas_mass_kgs
                m.mass_flow.max = lp_gas_mass_kgs
                m._scare_slack_budget_kgs = cap_gas_mass_kgs
            # heat-side ExtHydrGrid left unbounded


GRIDS: dict[str, Callable[[], "object"]] = {
    # Slack budget is a per-scenario knob (``scenario.slack_budget_pct``),
    # not baked into any grid, so one base grid serves multiple operator
    # policies and the energy-flow LP always has a free slack.

    # Coupling-density variants — vary only the number of CP plants.
    "simbench_lv_low": create_large_lv_simbench(0.1),
    "simbench_lv": create_large_lv_simbench(0.2),
    "simbench_lv_high": create_large_lv_simbench(0.3),

    # Scaling pillar — three LV variants spanning ~a decade in node count:
    #   small  (~15 buses, 1-LV-rural1)
    #   medium (~44 buses, 1-LV-semiurb4)
    #   large  (~129 buses, 1-LV-rural3 — the default ``simbench_lv``)
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
        0.2, backup_lines_per_sector=5, backup_seed=0,
    ),

    # CP-flexibility pillar — exercise ``cp_size_multiplier`` and
    # ``replace_primary_generation``:
    # ``cp_heavy``           additive CPs at 2× rating; primary fleet can
    #   still recover a lost CP, but the inflated capacity makes the
    #   cross-sector flex shift visible when the CP-ADMM layer engages.
    # ``cp_dependent``       CPs at 1× rating that replace primary gen;
    #   per-carrier rated production is invariant, but losing a CP also
    #   loses the displaced primary — the regime where CP-ADMM
    #   cross-sector substitution is load-bearing (gossip + islanding
    #   alone cannot recover the lost CP).
    # ``cp_heavy_dependent`` both knobs maxed: maximum cross-sector
    #   dependence; losing one CP produces a deep deficit solvable only
    #   by re-routing through surviving CHP / G2P / P2H plants.
    "simbench_lv_cp_heavy": create_large_lv_simbench(
        0.3, cp_size_multiplier=2.0, replace_primary_generation=False,
    ),
    "simbench_lv_cp_dependent": create_large_lv_simbench(
        0.3, cp_size_multiplier=1.0, replace_primary_generation=True,
    ),
    "simbench_lv_cp_heavy_dependent": create_large_lv_simbench(
        0.3, cp_size_multiplier=2.0, replace_primary_generation=True,
    )
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
                    a, b, sector, exc,
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


def assign_load_priorities(
    monee_net: "object",
    *,
    seed: int = 0,
    distribution: str = "skewed",
) -> dict[str, int]:
    """Assign per-load priority tiers under the 4-tier model.

    Tier 1 = critical (hard-locked at the L1 leader pre-step);
    tier 2 = high, tier 3 = medium, tier 4 = sheddable (QP-weighted).

    Returns a ``priorities`` dict keyed by ``child-{id}`` for
    ``create_restoration_scenario_world(priorities=...)``. Generators
    (cap < 0) and CPs are skipped (default to tier 0 in ``obs_priority``).

    ``distribution`` knobs:

    - ``"uniform"``  — uniform over [1, 4]; maximally diverse.
    - ``"skewed"``    — realistic 10/30/40/20 % across tiers 1-4
      (default). The 10 % tier-1 share keeps the per-community supply
      pool able to cover hard-locked demand while leaving enough
      QP-weighted demand to discriminate L2 allocation.
    - ``"by_capacity"`` — large loads to tier 1, small to tier 4
      ("feed the big hospitals first").
    - ``"all_one"``  — everyone tier 1; hard-locks every load
      (typically infeasible, triggers pro-rata branch). Ablation knob.

    ``seed`` makes assignments deterministic.
    """
    rng = random.Random(seed * 7919 + 31)
    P = 4
    out: dict[str, int] = {}

    for child in monee_net.childs:
        # Capacity straight from the model (no behavior yet).
        obs = dict(child.model.values)
        cap = obs_capacity(obs)
        if cap <= 0:
            continue  # generators / unknown default to tier 0
        aid = f"child-{child.id}"
        if distribution == "uniform":
            out[aid] = rng.randint(1, P)
        elif distribution == "skewed":
            r = rng.random()
            if r < 0.10:
                out[aid] = 1   # critical (hard-locked)
            elif r < 0.40:
                out[aid] = 2   # high
            elif r < 0.80:
                out[aid] = 3   # medium
            else:
                out[aid] = 4   # sheddable
        elif distribution == "by_capacity":
            # Resolved in the second pass (needs the global distribution).
            out[aid] = -1  # sentinel
        elif distribution == "all_one":
            out[aid] = 1
        else:
            raise ValueError(f"unknown priority distribution: {distribution}")

    if distribution == "by_capacity":
        # Bin by capacity quartile: top to tier 1, bottom to tier 4.
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

    Knobs:

    - ``supply_t_k``: heat-side ``ExtHydrGrid`` slack supply temperature
      (default ~70 °C vs unstressed ~83 °C). Lower supply shrinks the
      downstream headroom against the 60 °C / 333.15 K lower bound, so
      transport delay or pipe loss can push junctions into violation.
    - ``heat_load_scale``: multiplier on every ``HeatLoad.q_mw_heat``
      (default 1.5×); makes the heat sector harder to balance.

    Call once on a fresh net — calling twice stacks the load scale.
    """
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


def apply_pv_peak(
    mes: "object",
    *,
    gen_scale: float = 1.5,
    load_scale: float = 0.4,
    slack_vm_pu: float = 1.04,
) -> None:
    """Mutate ``mes`` in place to simulate a sunny-midday over-voltage
    stress scenario (the VDE-AR-N 4105 regime).

    Three concerted factors:

    1. High HV/MV transformer tap — slack ``ExtPowerGrid.vm_pu`` set to
       ``slack_vm_pu`` (default 1.04 pu), so the feeder sits ~1 % below
       the upper bound before any PV.
    2. PV near nameplate — every ``PowerGenerator.p_mw`` magnitude ×
       ``gen_scale`` (default 1.5×). Moderate on purpose: a 3-4× scale
       trivially imbalances and the LP curtails it away, whereas 1.5×
       stays within the slack budget so voltage depends on local Q.
    3. Daytime load trough — ``PowerLoad.p_mw`` × ``load_scale``
       (default 0.4×). Reactive load left nominal (the standard targets
       active-power imbalance).

    Call once on a fresh net — calling twice stacks the scales.
    """
    # Totals used to size the slack-budget relaxation below.
    total_gen_p = 0.0
    total_load_p = 0.0

    for child in mes.childs:
        m = child.model
        if isinstance(m, PowerGenerator):
            # p_mw < 0 (load convention); scaling preserves sign.
            m.p_mw = float(m.p_mw) * gen_scale
            total_gen_p += abs(m.p_mw)
        elif isinstance(m, PowerLoad):
            m.p_mw = float(m.p_mw) * load_scale
            total_load_p += abs(m.p_mw)

    # Widen the slack so the LP stays feasible under reverse flow. This
    # scenario is about voltage behaviour, not slack scarcity, so stress
    # should come from the tap setpoint and feeder impedance rather than
    # an artificial slack cap.
    headroom = max(total_gen_p - total_load_p, total_load_p) + 0.5
    for child in mes.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid):
            # ``vm_pu`` pins the slack-bus voltage magnitude; raising it
            # shifts the whole feeder profile upward.
            try:
                m.vm_pu = float(slack_vm_pu)
            except Exception:
                pass
            # Widen the slack p_mw Var bounds to absorb reverse flow.
            try:
                if hasattr(m.p_mw, "min"):
                    m.p_mw.min = -headroom
                if hasattr(m.p_mw, "max"):
                    m.p_mw.max = +headroom
            except Exception:
                pass


def apply_line_stress(
    mes: "object",
    *,
    load_scale: float = 1.8,
    ampacity_scale: float = 0.5,
    affect_branch_fraction: float = 1.0,
) -> None:
    """Mutate ``mes`` in place to simulate a line-loading stress scenario.

    Exercises the line-loading pipeline: PowerLine agents observe
    overload, the home group leader receives a relief-MW
    ``StartBalanceNegotiation``, and the reconfigurator's path-ranking
    sees loading variation across candidate paths.

    Knobs:

    - ``load_scale`` (default 1.8×): multiplier on every
      ``PowerLoad.p_mw``; combined with reduced ampacity, guarantees an
      overload after any non-trivial branch failure.
    - ``ampacity_scale`` (default 0.5×): multiplier on every PowerLine
      ``max_i_ka``; shifts the binding constraint from voltage to
      thermal.
    - ``affect_branch_fraction`` (default 1.0): fraction of PowerLines
      to reduce, picked deterministically by sorted branch id. Sweep to
      study concentrated vs distributed reductions.

    Call once on a fresh net — calling twice stacks the scales.
    """
    for child in mes.childs:
        m = child.model
        if isinstance(m, PowerLoad):
            m.p_mw = float(m.p_mw) * load_scale

    if ampacity_scale != 1.0 and affect_branch_fraction > 0.0:
        # Reduce ampacity on the highest-``max_i_ka`` lines first, so the
        # constraint lands on the lines the reconfigurator would
        # otherwise prefer.
        powerlines = [
            b for b in mes.branches
            if hasattr(b.model, "max_i_ka")
            and type(b.model).__name__.lower().startswith("power")
        ]
        powerlines.sort(
            key=lambda b: (-float(getattr(b.model, "max_i_ka", 0.0)), b.id),
        )
        n_affect = max(1, int(len(powerlines) * affect_branch_fraction))
        for branch in powerlines[:n_affect]:
            try:
                branch.model.max_i_ka = float(branch.model.max_i_ka) * ampacity_scale
            except Exception:
                pass


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
