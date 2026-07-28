from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from distributed_resource_optimization import (
    create_sharing_target_distance_admm_coordinator,
)
from distributed_resource_optimization.carrier.mango import CoordinatorRole
from mango import RoleAgent, behavior_in, connect_topologies, mark_as_connector
from mango.agent.core import State
from mango.express.topology import Topology, create_topology, modify_topology
from mango.simulation.world import (
    SimulationWorld,
    discrete_step_until,
    record_agent_having,
    record_world,
)
from mango_energy_environments import (
    BranchFailureEvent,
    CustomFailureEvent,
    Failure,
    RestorationEnvironmentBehavior,
    create_restoration_world,
    schedule_failure,
    topology_based_on_grid,
    topology_based_on_grid_groups,
    topology_based_on_sector_grid,
)
from mango_energy_environments.environments.restoration.multi_energy_monee import (
    create_branch_aid,
)
from monee.model.child import (
    ExtHydrGrid,
    ExtPowerGrid,
    HeatLoad,
    PowerGenerator,
    PowerLoad,
    Sink,
)
from monee.model.extension import GridFormingGenerator, GridFormingSource

from scare.base.addressing import child_aid, is_child_aid, node_aid
from scare.base.config import RestorationConfiguration
from scare.base.model import (
    ConstraintViolation,
    ReconfigurationCompletedEvent,
    Sector,
    SystemStrategy,
    is_energised_reading,
)
from scare.base.optimization.admm import (
    ScareDistributedOptimizationRole as DistributedOptimizationRole,
)
from scare.base.optimization.admm_factories import (
    create_chp_admm_flex_actor,
    create_g2p_admm_flex_actor,
    create_p2g_admm_flex_actor,
    create_p2h_admm_flex_actor,
)
from scare.base.runtime import diagnostics as _diag
from scare.base.runtime.comms import install_perturbation
from scare.base.runtime.trace import set_sim_time, sim_stall_watchdog
from scare.base.topology.community import (
    communities_from_topology,
    connected_component_partition,
    label_propagation_partition,
    modularity_of_partition,
    modularity_partition,
)
from scare.base.topology.topology_mirror import mirror_from_monee
from scare.base.util import (
    first_role,
    kgps_to_mw,
    lookup_slack,
    lookup_slack_pressure,
    obs_capacity,
    obs_constraint_values,
    obs_setpoint,
    register_grid_former_rating,
    register_heat_last_sink_floor,
    register_priority,
    register_sector,
    register_slack,
    role_index,
    sector_from_grid,
    set_directional_constraint_cap,
)
from scare.base.util.ids import deterministic_uuid
from scare.community.coalition_store import CoalitionConstraintStore
from scare.community.dynamic_holon import DynamicHolonRole
from scare.community.holonic import HolonicCommunityRole
from scare.community.repartition import (
    DynamicRepartitionRole,
    RepartitionHandlerRole,
)
from scare.community.role import PreAssignedCommunityRole
from scare.community.summary import HolonSummaryRole
from scare.detection.role import ProblemDetector
from scare.service.balance.balance import (
    EnergyBalanceNegotiator,
    create_energy_balance_role,
)
from scare.service.control.constraints import GridConstraintMonitor
from scare.service.control.cp_heat_guard import CPHeatOutletGuard
from scare.service.control.gas_pressure import GasPressureRegulator
from scare.service.control.local_generation import LocalGenerationFallbackRole
from scare.service.control.slack_budget import SlackBudgetMonitor
from scare.service.control.stability import GenerationController
from scare.service.control.voltage_droop import (
    COS_PHI_LARGE,
    COS_PHI_SMALL,
    COS_PHI_THRESHOLD_MVA,
    ReactivePowerDroopRole,
    vde_cos_phi_min,
)
from scare.service.coupling.cp import EnergyConverterRole, MultiCommunityCPRole
from scare.service.coupling.cp_priority_admm_role import CPPriorityAdmmRole
from scare.service.coupling.dynamic_connector import DynamicConnectorRole
from scare.service.reconfiguration import GridReconfigurator, GridTieSwitchOperator

logger = logging.getLogger(__name__)

_SECTORS = [Sector.ELECTRICITY, Sector.GAS, Sector.HEAT]
# Sector -> monee grid-object name substring matched by
# ``topology_based_on_sector_grid`` (Sector.value won't match the repr).
_SECTOR_GRID_MATCH: dict[Sector, str] = {
    Sector.ELECTRICITY: "power",
    Sector.GAS: "gas",
    Sector.HEAT: "water",
}


def _branch_sector_str(branch: Any, monee_net: Any) -> str:
    """Sector tag for a branch (``ProblemDetector`` edge cost).

    ``"electricity"``/``"gas"``/``"heat"`` for same-sector lines, ``"cp"``
    for cross-sector plants, ``""`` when undeterminable (non-traversable).
    """
    if branch.model.is_cp():
        return "cp"
    try:
        node = monee_net.node_by_id(branch.id[0])
    except Exception:
        return ""
    sec = sector_from_grid(node.grid)
    return sec.value if sec else ""


def _heat_component_by_node(monee_net: Any) -> dict[Any, int]:
    """Build-time connected components of the water/heat network (component
    index per node id). Scopes the heat priority waterfall to peers that share
    hydraulics; a mid-run pipe split degrades to the unscoped mesh reach, never
    worse."""
    adj: dict[Any, list[Any]] = {}
    for branch in monee_net.branches:
        if _branch_sector_str(branch, monee_net) != Sector.HEAT.value:
            continue
        a, b = branch.id[0], branch.id[1]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    component: dict[Any, int] = {}
    idx = 0
    for start in adj:
        if start in component:
            continue
        stack = [start]
        component[start] = idx
        while stack:
            node = stack.pop()
            for neighbour in adj.get(node, []):
                if neighbour not in component:
                    component[neighbour] = idx
                    stack.append(neighbour)
        idx += 1
    return component


def _node_aid(node_id: Any) -> str:
    return node_aid(node_id)


def _child_aid(child_id: Any) -> str:
    return child_aid(child_id)


def _model_type_name(branch) -> str:
    return type(branch.model).__name__.lower()


def _is_cp_branch(branch) -> bool:
    return branch.model.is_cp()


def _maybe_register_slack(behavior: Any, aid: str, child: Any) -> None:
    """Register an ExtPowerGrid/ExtHydrGrid child as a slack agent so
    gossip reports its rated capacity and treats it as generator-class.

    Rating from the Var bounds; both bounds unbounded -> left unregistered.
    """
    m = child.model
    # Prefer the operator soft budget (enforcement target) over the LP Var
    # bounds (widened for feasibility); fall back to Var bounds as rating.
    budget_attr: float | None = None
    var = None
    if isinstance(m, ExtPowerGrid):
        var = getattr(m, "p_mw", None)
        budget_attr = getattr(m, "_scare_slack_budget_mw", None)
    elif isinstance(m, ExtHydrGrid):
        var = getattr(m, "mass_flow_kgs", None)
        budget_attr = getattr(m, "_scare_slack_budget_kgs", None)
    else:
        return
    if var is None:
        return
    p_min = getattr(var, "min", None)
    p_max = getattr(var, "max", None)
    if budget_attr is not None and float(budget_attr) > 0.0:
        rating = float(budget_attr)
    else:
        # Rating magnitude from whichever bound is set.
        mags: list[float] = []
        if p_min is not None:
            mags.append(abs(float(p_min)))
        if p_max is not None:
            mags.append(abs(float(p_max)))
        if not mags:
            return
        rating = max(mags)
    register_slack(behavior, aid, rating_mw=rating, p_min=p_min, p_max=p_max)


def _maybe_register_grid_former(behavior: Any, aid: str, child: Any) -> None:
    """Record a promoted island reference's rated capacity (Var bound magnitude).

    A ``GridForming*`` unit's balancing Var flips sign as it produces vs absorbs
    the island residual; recording its bound magnitude lets the holon credit it as
    fixed supply, not misread the free Var as load. Only islanding promotes formers,
    so this is inert elsewhere.
    """
    m = child.model
    if isinstance(m, GridFormingGenerator):
        var = getattr(m, "p_mw", None)
    elif isinstance(m, GridFormingSource):
        var = getattr(m, "mass_flow_kgs", None)
    else:
        return
    if var is None:
        return
    mags = [
        abs(float(b))
        for b in (getattr(var, "min", None), getattr(var, "max", None))
        if b is not None
    ]
    if mags:
        register_grid_former_rating(behavior, aid, max(mags))


def _slack_budget_for_child(child: Any) -> tuple[str, float] | None:
    """``(obs_key, budget)`` for a budgeted slack-class child, else
    ``None``. Drives whether a ``SlackBudgetMonitor`` is installed.
    """
    m = child.model
    if isinstance(m, ExtPowerGrid):
        budget = getattr(m, "_scare_slack_budget_mw", None)
        if budget is None:
            return None
        return ("p_mw", float(budget))
    if isinstance(m, ExtHydrGrid):
        budget = getattr(m, "_scare_slack_budget_kgs", None)
        if budget is None:
            return None
        return ("mass_flow_kgs", float(budget))
    return None


def _is_power_generator(child: Any) -> bool:
    """True when ``child`` is a monee ``PowerGenerator`` (every PV plant).
    Excludes PowerLoad and ExtPowerGrid (slack Q is already an LP Var).
    """
    return isinstance(child.model, PowerGenerator)


def _is_heat_side_mass_flow_sink(child: Any, monee_net: Any) -> bool:
    """True when ``child`` is a ``Sink`` on a water/heat grid.

    Such Sinks close monee's supply-return loop (topology artifact, not
    shedable demand): regulating them breaks mass balance / LP feasibility.
    Gas-sector Sinks are real consumption and stay load agents.
    """
    if not isinstance(child.model, Sink):
        return False
    try:
        grid_name = str(
            getattr(monee_net.node_by_id(child.node_id).grid, "name", "")
        ).lower()
    except Exception:  # noqa: BLE001
        return False
    return "water" in grid_name or "heat" in grid_name


def _is_cp_subordinate_child(child: Any) -> bool:
    """True when ``child`` is a CP's subordinate ``SubHG`` output.

    Its heat follows the CHP control node (not independently controllable);
    skipping it avoids dead regulate writes and double-counted heat. The
    physical injection still happens via the Var.
    """
    return type(child.model).__name__ == "SubHG"


def _inverter_s_nom_mva(child: Any) -> float | None:
    """Inverter rated apparent power in MVA.

    Prefers explicit ``s_nom_mva``; else reconstructs from rated active
    power via VDE-AR-N 4105 ``S_n = |p_n| / cos φ_min`` (two-pass to pick
    the right cos φ without self-reference).
    """
    nominal = getattr(child.model, "s_nom_mva", None)
    if nominal is not None:
        try:
            value = float(nominal)
        except (TypeError, ValueError):
            value = None
        if value is not None and value > 0.0:
            return value

    p_n = abs(float(getattr(child.model, "p_mw", 0.0) or 0.0))
    if p_n <= 0.0:
        return None
    s_nom = p_n / COS_PHI_SMALL
    if s_nom > COS_PHI_THRESHOLD_MVA:
        s_nom = p_n / COS_PHI_LARGE
    return s_nom


def _sectors_for_cp_type(cp_type: str) -> list[Sector]:
    """Return the energy sectors a CP of this type bridges."""
    if "chp" in cp_type:
        return [Sector.ELECTRICITY, Sector.HEAT, Sector.GAS]
    if "p2g" in cp_type or "powertogas" in cp_type:
        return [Sector.ELECTRICITY, Sector.GAS]
    if "g2p" in cp_type or "gastopower" in cp_type:
        return [Sector.ELECTRICITY, Sector.GAS]
    if "p2h" in cp_type or "powertoheat" in cp_type:
        return [Sector.ELECTRICITY, Sector.HEAT]
    return []


def _cp_coupling_ratios(cp_type: str) -> dict[tuple[str, str], float]:
    """Static directional efficiencies per CP type.

    ``coupling[(in_v, out_v)] = η``: 1 MW into sector in yields η MW out.
    Priors for L2.5 coalition sizing; L3 ADMM refines them.
    """
    ct = cp_type.lower()
    el = Sector.ELECTRICITY.value
    he = Sector.HEAT.value
    ga = Sector.GAS.value
    if "chp" in ct:
        # Gas in, electricity + heat out.
        return {(ga, el): 0.35, (ga, he): 0.45}
    if "p2g" in ct or "powertogas" in ct:
        return {(el, ga): 0.6}
    if "g2p" in ct or "gastopower" in ct:
        return {(ga, el): 0.4}
    if "p2h" in ct or "powertoheat" in ct:
        return {(el, he): 0.95}
    return {}


def _cp_rated_capacity_mw(obs: dict, cp_type: str) -> dict[str, float]:
    """Approximate per-sector rated output capacity of the CP in MW.

    Conservative upper bound on coalition transfer; L3 ADMM refines it.
    """
    out: dict[str, float] = {}
    ct = cp_type.lower()
    sectors = _sectors_for_cp_type(ct)
    for sec in sectors:
        if sec == Sector.ELECTRICITY:
            v = obs.get("p_mw_capacity") or obs.get("p_mw")
        elif sec == Sector.HEAT:
            v = obs.get("q_mw_set") or obs.get("q_mw_heat") or obs.get("q_mw")
        elif sec == Sector.GAS:
            v = obs.get("mass_flow_capacity_kgs") or obs.get("mass_flow_kgs")
        else:
            v = None
        if v is None:
            continue
        try:
            cap = abs(float(v))
        except (TypeError, ValueError):
            continue
        if not (cap > 0 and cap < 1e9):
            continue
        if sec == Sector.GAS:
            cap = kgps_to_mw(cap)  # kg/s to MW for comparability
        out[sec.value] = cap
    return out


def create_restoration_scenario_world(
    monee_net: Any,
    priorities: dict[str, int] | None = None,
    *,
    base_delay_ms: float = 20.0,
    strategy: SystemStrategy = SystemStrategy.GROUP_TO_CP,
    simulation_duration_s: float = 30.0,
    config: RestorationConfiguration | None = None,
    physics_time_scale: float | None = None,
    physics_interval_s: float | None = None,
    physics_solve_time_limit_s: float | None = None,
) -> SimulationWorld:
    priorities = priorities or {}
    config = config or RestorationConfiguration()

    # No priorities -> obs_priority defaults every load to tier 1,
    # collapsing the priority-aware layers. Skip the warning for 1-load tests.
    if not priorities:
        n_loads = sum(
            1
            for child in monee_net.childs
            if obs_capacity(dict(child.model.values)) > 0
        )
        if n_loads > 1:
            logger.warning(
                "No priorities dict supplied for %d loads — obs_priority "
                "will default every load to tier 1, collapsing the priority-"
                "aware machinery (QP waterfall, holon ADMM S-pull, CP "
                "consensus weights, PWSF metric) to a uniform baseline. "
                "Use experiment.scenarios.assign_load_priorities() or pass "
                "an explicit priorities= dict.",
                n_loads,
            )

    # Keep with_communication off: it would install a 20 s/hop Poisson
    # delay overriding ``static_delay_s`` and making ``base_delay_ms`` inert.
    world = create_restoration_world(
        monee_net,
        with_communication=False,
        static_delay_s=base_delay_ms / 1000.0,
        physics_time_scale=physics_time_scale,
        physics_interval_s=physics_interval_s,
        physics_solve_time_limit_s=physics_solve_time_limit_s,
    )

    behavior: RestorationEnvironmentBehavior = world.environment.behavior
    # Stash config so components read flags lazily without threading it through.
    behavior._scare_config = config
    set_directional_constraint_cap(config.enable_directional_constraint_cap)
    _cd_override = getattr(config, "energy_flow_cooldown_s_override", None)
    if _cd_override is not None and hasattr(behavior, "_energy_flow_cooldown_s"):
        behavior._energy_flow_cooldown_s = float(_cd_override)
    _max_acts = getattr(config, "energy_flow_max_acts", None)
    if _max_acts is not None and hasattr(behavior, "_energy_flow_max_acts"):
        behavior._energy_flow_max_acts = int(_max_acts)

    # Install perturbations before agents register so every send is covered.
    install_perturbation(
        world,
        base_delay_s=base_delay_ms / 1000.0,
        packet_loss_pct=config.comms_packet_loss_pct,
        latency_jitter_ms=config.comms_latency_jitter_ms,
    )

    _populate_world(world, monee_net, behavior, priorities, config)
    _build_topologies(world, monee_net, behavior, config, priorities)
    _add_system_behaviors(world, monee_net, behavior, strategy, config)
    _register_recordings(world, monee_net, behavior, priorities=priorities)

    return world


async def start_restoration_simulation(
    world: SimulationWorld,
    failures: list[Failure],
    simulation_duration_s: float = 30.0,
) -> None:
    behavior: RestorationEnvironmentBehavior = world.environment.behavior

    for failure in failures:
        schedule_failure(behavior, world, failure)

    # Mirror the sim clock into the log filter so every log line carries the
    # current simulation time (t=...). Wrap the instance's set_time so each
    # discrete-step advance updates the global before any handler logs run.
    set_sim_time(world.clock.time)
    _orig_set_time = world.clock.set_time

    def _tracking_set_time(*args, _orig=_orig_set_time, **kwargs):
        if args:
            set_sim_time(args[0])
        return _orig(*args, **kwargs)

    world.clock.set_time = _tracking_set_time

    async with world:
        # Wall-clock watchdog: if the discrete-event clock freezes (the timeout
        # failure mode), log the unsettled request/reply tasks keeping mango's
        # termination detection from returning. Plain asyncio task → invisible
        # to settle detection; ticks on real time so it fires during the hang.
        _watchdog = asyncio.ensure_future(sim_stall_watchdog(world, interval_s=10.0))
        try:
            await discrete_step_until(world, max_advance_time_s=simulation_duration_s)
            await _settle_end_of_sim(world)
        finally:
            _watchdog.cancel()
        # Flush in-flight gossip while role.context is still valid.
        _flush_pending_negotiations(world)


async def _settle_end_of_sim(world: SimulationWorld) -> None:
    """Alternate an immediate flush with a short discrete-step so controllers
    observe the post-flush converged voltages/pressures (which the throttled
    solve hid) and react before the end-of-sim snapshot; break once a flush
    leaves the env clean. Fixes the end-of-sim observation desync. Gated OFF by
    default (perturbs every task's final state; needs aggregate validation).
    """
    behavior: RestorationEnvironmentBehavior = world.environment.behavior
    config = getattr(behavior, "_scare_config", None)
    if config is None or not getattr(config, "enable_end_of_sim_settle", False):
        return
    rounds = int(getattr(config, "end_of_sim_settle_max_rounds", 3))
    chunk_s = float(getattr(config, "end_of_sim_settle_chunk_s", 2.0))
    flush = getattr(behavior, "flush_energy_flow", None)
    if flush is None or rounds <= 0 or chunk_s <= 0:
        return
    for _ in range(rounds):
        flush()  # reveal the true converged state to observers
        # max_advance_time_s is a relative duration in mango, not an absolute
        # horizon — passing clock.time+chunk_s free-runs the sim for ~its whole
        # elapsed length again per round.
        await discrete_step_until(world, max_advance_time_s=chunk_s)
        # A clean env after the chunk means controllers issued no new setpoint
        # changes — already converged, no point iterating further.
        if not getattr(behavior, "_dirty", False):
            break


def _flush_pending_negotiations(world: SimulationWorld) -> None:
    """Drain in-flight gossip into the diary so a short duration doesn't
    drop abandoned negotiations from the per-event account.
    """
    for agent in world.agents.values():
        for role in getattr(agent, "roles", []):
            if isinstance(role, EnergyBalanceNegotiator):
                role.flush_pending()


def _build_branch_sector_tables(
    monee_net: Any,
) -> tuple[dict[tuple, str], dict[Any, dict[Any, str]]]:
    """Per-branch and per-node sector lookup tables for ``ProblemDetector``
    edge costing. Returns ``(branch_sector_by_id, neighbour_sector_by_node)``.
    """
    branch_sector_by_id: dict[tuple, str] = {}
    neighbour_sector_by_node: dict[Any, dict[Any, str]] = {}
    for branch in monee_net.branches:
        sec = _branch_sector_str(branch, monee_net)
        if not sec:
            continue
        branch_sector_by_id[branch.id] = sec
        a, b = branch.id[0], branch.id[1]
        neighbour_sector_by_node.setdefault(a, {})[b] = sec
        neighbour_sector_by_node.setdefault(b, {})[a] = sec
    return branch_sector_by_id, neighbour_sector_by_node


def _register_heat_last_sink_floors(
    behavior: RestorationEnvironmentBehavior, monee_net: Any
) -> None:
    """R5 last-sink guard registry: for each heat junction with fixed local
    injection (HeatGenerator children) and exactly ONE HeatLoad, register the
    serve-fraction floor that absorbs the injection
    (``min(1, injection/|load|)``). CP outlet injection (SubHG) is excluded —
    variable, owned by the CP heat-outlet guard."""
    from monee.model.child import HeatGenerator, HeatLoad

    by_node: dict[Any, dict[str, list]] = {}
    for child in monee_net.childs:
        m = child.model
        if isinstance(m, HeatGenerator):
            by_node.setdefault(child.node_id, {}).setdefault("gen", []).append(m)
        elif isinstance(m, HeatLoad):
            by_node.setdefault(child.node_id, {}).setdefault("load", []).append(child)
    for node_id, kinds in by_node.items():
        gens = kinds.get("gen", [])
        loads = kinds.get("load", [])
        if not gens or len(loads) != 1:
            continue
        # HeatGenerator stores q_mw_heat = -q_mw (injection); HeatLoad +q_mw.
        injection = sum(abs(float(g.q_mw_heat)) for g in gens)
        load_cap = abs(float(loads[0].model.q_mw_heat))
        if injection <= 0.0 or load_cap <= 0.0:
            continue
        floor = min(1.0, injection / load_cap)
        register_heat_last_sink_floor(behavior, _child_aid(loads[0].id), floor)


def _populate_children(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
    config: RestorationConfiguration,
) -> None:
    heat_component_by_node = _heat_component_by_node(monee_net)
    if config.enable_heat_last_sink_guard:
        _register_heat_last_sink_floors(behavior, monee_net)
    for child in monee_net.childs:
        aid = _child_aid(child.id)
        # Read the model directly: observe() needs energyflow, which only
        # runs once the sim starts.
        parent_node = monee_net.node_by_id(child.node_id)
        obs = {**dict(parent_node.model.values), **dict(child.model.values)}
        sector = sector_from_grid(parent_node.grid)
        register_sector(behavior, aid, sector)
        # Register slack-class children so gossip reports rated capacity
        # and treats them as generator-class regardless of flow direction.
        _maybe_register_slack(behavior, aid, child)
        # Register a promoted island reference so the holon/gossip classify it as
        # a generator (never a sign-flipping load) and credit its delivered
        # injection as supply. Gated on the guard flag so default runs are
        # byte-identical (the registry is read only under the flag).
        if config.enable_grid_former_curtail_guard:
            _maybe_register_grid_former(behavior, aid, child)
        explicit_priority = priorities.get(aid)
        # Force slacks to tier 0: a slack is never shed. Otherwise obs_priority
        # may read the LP p_mw sign and misclassify it as load. (Formers are
        # already excluded from the curtailable/load set under the guard, so
        # their priority is not read there and needs no forcing here.)
        if explicit_priority is None:
            if lookup_slack(behavior, aid) is not None:
                explicit_priority = 0

        # Heat-side mass-flow Sinks are a topology artifact; skip so they
        # stay out of topology/partition/holon and are never curtailed.
        if _is_heat_side_mass_flow_sink(child, monee_net):
            continue

        # A CHP's SubHG follows its control node; skip to avoid separate
        # regulation and double-counting as a heat generator.
        if _is_cp_subordinate_child(child):
            continue

        # Register the resolved priority so group aggregators get the right
        # per-aid tier (not obs_priority's tier-1-for-all default).
        if explicit_priority is not None:
            register_priority(behavior, aid, int(explicit_priority))

        roles = []
        if sector is not None:
            roles.append(
                _make_balance_role(
                    behavior, sector, obs, config, priority=explicit_priority
                )
            )
            roles.append(
                GenerationController(
                    behavior, sector, ramp_to_full=config.enable_gen_ramp_to_full
                )
            )
            # Voltage / pressure / temperature monitoring.
            roles.append(
                GridConstraintMonitor(
                    behavior,
                    sector,
                    node_id=child.node_id,
                    enable_curtailment_auction=config.enable_curtailment_auction,
                    enable_generation_priority_curtailment=config.enable_generation_priority_curtailment,
                    enable_line_congestion_price=config.enable_line_congestion_price,
                    enable_curtail_auction_gating=config.enable_curtail_auction_gating,
                    enable_curtail_auction_targeting=config.enable_curtail_auction_targeting,
                    enable_line_relief_reassert=config.enable_line_relief_reassert,
                    enable_branch_downstream_relief=config.enable_branch_downstream_relief,
                    enable_multihop_constraint=config.enable_multihop_constraint,
                    enable_heat_frontier=config.enable_heat_frontier,
                    enable_heat_priority_waterfall=config.enable_heat_priority_waterfall,
                    heat_component_id=(
                        heat_component_by_node.get(child.node_id)
                        if sector == Sector.HEAT
                        else None
                    ),
                    enable_qv_auction_coordination=config.enable_qv_auction_coordination,
                    enable_qv_feeder_gate=config.enable_qv_feeder_gate,
                )
            )
            # Slack-budget enforcement: only budgeted slack-class children
            # get a monitor (helper returns None for everything else).
            if config.enable_slack_budget_monitor:
                budget_info = _slack_budget_for_child(child)
                if budget_info is not None:
                    obs_key, budget_value = budget_info
                    roles.append(
                        SlackBudgetMonitor(
                            behavior,
                            sector,
                            obs_key=obs_key,
                            budget=budget_value,
                            tol=config.slack_budget_violation_tol,
                            enable_feedback=config.enable_slack_budget_feedback,
                        )
                    )

            # Layer-0 gas regulator: the gas slack autonomously holds
            # downstream pressure in band via its pressure_pu setpoint, tried
            # before shedding. Gas ExtHydrGrid only (the heat-side slack uses
            # t_k, not pressure).
            if (
                config.enable_gas_pressure_regulator
                and sector is Sector.GAS
                and isinstance(child.model, ExtHydrGrid)
            ):
                roles.append(
                    GasPressureRegulator(
                        behavior,
                        sector,
                        gain=config.gas_pressure_regulator_gain,
                    )
                )

        # Updates the child's CommunityAssignment on a leader re-partition.
        if sector is not None:
            roles.append(RepartitionHandlerRole())

        # Local Q-V droop per inverter PowerGenerator (VDE-AR-N 4105
        # §5.7.2), orthogonal to the layers above.
        if (
            config.enable_qv_droop
            and sector == Sector.ELECTRICITY
            and _is_power_generator(child)
        ):
            s_nom = _inverter_s_nom_mva(child)
            if s_nom is not None and s_nom > 0.0:
                roles.append(
                    ReactivePowerDroopRole(
                        behavior,
                        s_nom_mva=s_nom,
                        cos_phi_min=vde_cos_phi_min(s_nom),
                        voltage_ref_pu=config.qv_droop_voltage_ref_pu,
                        settling_tau_s=config.qv_droop_settling_tau_s,
                        attack_tau_s=config.qv_droop_attack_tau_s,
                    )
                )

        _register_agent(
            world, behavior, aid, roles, monee_id=child.id, monee_type="child"
        )


def _populate_nodes(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
    config: RestorationConfiguration,
    neighbour_sector_by_node: dict[Any, dict[Any, str]],
) -> None:
    for node in monee_net.nodes:
        aid = _node_aid(node.id)
        register_sector(behavior, aid, sector_from_grid(node.grid))
        # Child addresses on this node so the detector delivers
        # FailureNotice locally (children already registered above).
        child_addrs: list[Any] = []
        for cid in getattr(node, "child_ids", []):
            child_aid = _child_aid(cid)
            child_agent = world.agents.get(child_aid)
            if child_agent is not None:
                child_addrs.append(child_agent.addr)
        roles: list[Any] = [
            ProblemDetector(
                behavior,
                node.id,
                neighbour_branch_sectors=neighbour_sector_by_node.get(node.id, {}),
                child_addrs=child_addrs,
                enable_distributed_failure_notice=config.enable_distributed_failure_notice,
                ttl_hops=config.ttl_hops,
                cp_bridge_cost=config.cp_bridge_cost,
            ),
            GridReconfigurator(
                behavior,
                node.id,
                enable_ranking=config.enable_reconfig_feasibility_ranking,
                window_s=config.reconfig_path_window_s,
            ),
        ]

        obs = dict(node.model.values)
        cp_type = _detect_cp_type_for_node(node, monee_net)
        if cp_type is not None:
            if config.enable_cp_priority_admm:
                # Node-side only for node-hosted CPs (CHP); branch-hosted
                # CPs are actuated on the branch agent in _populate_branches.
                if "chp" in cp_type.lower():
                    _attach_cp_priority_admm_role(
                        roles, behavior, aid, cp_type, obs, config
                    )
            elif config.enable_cp_admm:
                _attach_cp_roles(roles, behavior, cp_type, obs, priorities.get(aid, 0))
            elif config.cps_join_communities:
                _attach_multi_community_cp_role(roles, behavior, cp_type, config)
            if config.enable_cp_heat_outlet_guard:
                outlet_aid = _heat_outlet_aid_for_node(node, monee_net)
                if outlet_aid is not None:
                    roles.append(CPHeatOutletGuard(behavior, outlet_aid=outlet_aid))

        _register_agent(
            world, behavior, aid, roles, monee_id=node.id, monee_type="node"
        )


def _populate_branches(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
    config: RestorationConfiguration,
) -> None:
    for branch in monee_net.branches:
        aid = create_branch_aid(branch.id)
        obs = dict(branch.model.values)
        branch_type = _model_type_name(branch)

        roles = []

        if "heatexchanger" in branch_type:
            roles.append(_make_balance_role(behavior, Sector.HEAT, obs, config))
            roles.append(
                GenerationController(
                    behavior, Sector.HEAT, ramp_to_full=config.enable_gen_ramp_to_full
                )
            )

        elif "powertogas" in branch_type:
            if config.enable_cp_priority_admm:
                _attach_cp_priority_admm_role(roles, behavior, aid, "p2g", obs, config)
            elif config.enable_cp_admm:
                _attach_cp_roles(roles, behavior, "p2g", obs, priorities.get(aid, 0))
            elif config.cps_join_communities:
                _attach_multi_community_cp_role(roles, behavior, "p2g", config)
        elif "gastopower" in branch_type:
            if config.enable_cp_priority_admm:
                _attach_cp_priority_admm_role(roles, behavior, aid, "g2p", obs, config)
            elif config.enable_cp_admm:
                _attach_cp_roles(roles, behavior, "g2p", obs, priorities.get(aid, 0))
            elif config.cps_join_communities:
                _attach_multi_community_cp_role(roles, behavior, "g2p", config)
        elif "powertoheat" in branch_type:
            if config.enable_cp_priority_admm:
                _attach_cp_priority_admm_role(roles, behavior, aid, "p2h", obs, config)
            elif config.enable_cp_admm:
                _attach_cp_roles(roles, behavior, "p2h", obs, priorities.get(aid, 0))
            elif config.cps_join_communities:
                _attach_multi_community_cp_role(roles, behavior, "p2h", config)

        elif hasattr(branch.model, "on_off"):
            roles.append(GridTieSwitchOperator(behavior, branch.id))

        # Heat-outlet guard on heat-producing branch CPs: P2H/G2H inject
        # q_mw_heat at the to-node. Independent of the CP-coordination elif
        # chain above — the born regulation=1.0 injects at rated power even
        # with every L3 layer off, so the guard must not ride on any of them.
        if config.enable_cp_heat_outlet_guard and (
            "powertoheat" in branch_type or "gastoheat" in branch_type
        ):
            roles.append(
                CPHeatOutletGuard(behavior, outlet_aid=_node_aid(branch.id[1]))
            )

        # Line-loading monitor on electricity power lines;
        # home_leader_addr is filled in after the groups topology is built.
        branch_sector = _branch_sector_str(branch, monee_net)
        if (
            config.enable_line_loading_constraint
            and branch_sector == "electricity"
            and not _is_cp_branch(branch)
        ):
            roles.append(
                GridConstraintMonitor(
                    behavior,
                    Sector.ELECTRICITY,
                    branch_id=branch.id,
                    home_leader_addr=None,
                    enable_curtailment_auction=config.enable_curtailment_auction,
                    enable_generation_priority_curtailment=config.enable_generation_priority_curtailment,
                    enable_line_congestion_price=config.enable_line_congestion_price,
                    enable_curtail_auction_gating=config.enable_curtail_auction_gating,
                    enable_curtail_auction_targeting=config.enable_curtail_auction_targeting,
                    enable_line_relief_reassert=config.enable_line_relief_reassert,
                    enable_branch_downstream_relief=config.enable_branch_downstream_relief,
                    enable_line_relief_waterfall=config.enable_line_relief_waterfall,
                    enable_multihop_constraint=config.enable_multihop_constraint,
                )
            )

        if roles:
            _register_agent(
                world, behavior, aid, roles, monee_id=branch.id, monee_type="branch"
            )


def _populate_world(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
    config: RestorationConfiguration,
) -> None:
    branch_sector_by_id, neighbour_sector_by_node = _build_branch_sector_tables(
        monee_net
    )
    behavior._scare_branch_sector = branch_sector_by_id
    _populate_children(world, monee_net, behavior, priorities, config)
    _populate_nodes(
        world, monee_net, behavior, priorities, config, neighbour_sector_by_node
    )
    _populate_branches(world, monee_net, behavior, priorities, config)


def _heat_outlet_aid_for_node(node: Any, monee_net: Any) -> str | None:
    """aid whose observation carries the heat-outlet junction ``t_k`` for a
    node-hosted heat-producing CP, or None when *node* itself is not one
    (``_detect_cp_type_for_node`` also matches nodes merely incident to a CP
    branch — those are guarded on the branch agent instead).

    CHP-HG control nodes inject via a ``SubHG`` child attached at a DHS
    junction — resolve that child's node. HX control nodes (non-HG CHP/P2H/
    G2H variants) mix at their own junction and carry their own ``t_k``.
    """
    model = getattr(node, "model", None)
    sub_hg = getattr(model, "_sub_hg", None)
    if sub_hg is not None:
        for child in monee_net.childs:
            if child.model is sub_hg:
                return _node_aid(child.node_id)
        return None
    own = _model_type_name(node)
    if any(s in own for s in ("chpcontrol", "powertoheatcontrol", "gastoheatcontrol")):
        return _node_aid(node.id)
    return None


def _detect_cp_type_for_node(node: Any, monee_net: Any) -> str | None:
    """CP plant type if *node* or any incident branch is a coupling model;
    substrings match both legacy and current model names.
    """
    # Check the node's own model first: a CHP is actuated through its
    # ``chphgcontrolnode``, which branch-only detection would miss.
    own = _model_type_name(node)
    if any(s in own for s in ("chp", "powertogas", "gastopower", "powertoheat")):
        return own
    for branch in monee_net.branches_connected_to(node.id):
        t = _model_type_name(branch)
        if any(s in t for s in ("chp", "powertogas", "gastopower", "powertoheat")):
            return t
    return None


def _make_balance_role(
    behavior: Any,
    sector: Sector,
    obs: dict,
    config: RestorationConfiguration,
    *,
    priority: int | None = None,
):
    """``create_energy_balance_role`` with gossip flags from *config*."""
    return create_energy_balance_role(
        behavior,
        sector,
        obs,
        priority=priority,
        constraint_aware=config.enable_constraint_aware_gossip,
        proactive_util_ttl_s=config.proactive_util_ttl_s,
        enable_monotonic_floor=config.enable_monotonic_floor,
        enable_clpu_ramp=config.enable_clpu_ramp,
        termination_tolerance=config.gossip_termination_tolerance,
        max_hops=config.gossip_max_hops,
        enable_qp_gossip=config.enable_qp_gossip,
        enable_l2_generator_ramp=config.enable_l2_generator_ramp,
        enable_change_only_dispatch=config.enable_change_only_dispatch,
        enable_l2_priority_floor=config.enable_l2_priority_floor,
        enable_actuated_ledger_writeback=config.enable_actuated_ledger_writeback,
        enable_heat_l2_dispatch=config.enable_heat_l2_dispatch,
        enable_gen_capacity_supply=config.enable_gen_capacity_supply,
        enable_l2_allocation_reassert=config.enable_l2_allocation_reassert,
        l2_allocation_reassert_s=config.l2_allocation_reassert_s,
        component_scope=(
            config.enable_holonic and config.holon_admm_scope == "component"
        ),
    )


def _register_agent(
    world: SimulationWorld,
    behavior: Any,
    aid: str,
    roles: list,
    *,
    monee_id: Any,
    monee_type: str,
) -> Any:
    """Register a ``RoleAgent``, attach roles, and bind it to its monee
    object via ``behavior.install``.
    """
    agent = world.register(RoleAgent(), suggested_aid=aid)
    for role in roles:
        agent.add_role(role)
    behavior.install(agent, id=monee_id, type=monee_type)
    return agent


def _attach_cp_roles(
    roles: list,
    behavior: Any,
    cp_type: str,
    obs: dict,
    priority: int,
) -> None:
    """Append the three CP roles (converter + distributed-opt +
    coordinator). No-op for unknown *cp_type*.
    """
    flex_actor, sectors = _build_cp_flex_actor(cp_type, obs, priority)
    if flex_actor is None:
        return
    roles.append(EnergyConverterRole(behavior, flex_actor, sectors))
    roles.append(DistributedOptimizationRole(flex_actor))
    roles.append(CoordinatorRole(create_sharing_target_distance_admm_coordinator()))


def _cp_signed_capacity_by_sector(cp_type: str, obs: dict) -> dict[str, float]:
    """Load-convention signed per-sector capacities (MW) for a CP.

    Input read from the canonical obs key; outputs are input scaled by η.
    Sign: positive = consumes from sector, negative = produces into it.
    """
    ct = cp_type.lower()
    el = Sector.ELECTRICITY.value
    he = Sector.HEAT.value
    ga = Sector.GAS.value

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(obs.get(key, default))
        except (TypeError, ValueError):
            return default

    def _first_nonzero(*keys: str, default: float = 0.0) -> float:
        """First present, non-zero obs value across *keys*, else
        *default* (accepts model-specific and generic input keys)."""
        for k in keys:
            v = _f(k, 0.0)
            if v != 0.0:
                return v
        return default

    out: dict[str, float] = {}
    if "chp" in ct:
        # CHP at its control node (gas in -> el + heat out). Gas rate from
        # ``gas_mass_flow_kgs``; never the hydraulic ``mass_flow_kgs``. Heat
        # sits on the CP.
        cap_in = kgps_to_mw(
            abs(_first_nonzero("gas_mass_flow_kgs", "mass_flow_setpoint_kgs"))
        )
        if cap_in <= 0:
            return {}
        eta_el = _first_nonzero("efficiency_power", "eta_el", default=0.35)
        eta_he = _first_nonzero("efficiency_heat", "eta_heat", default=0.45)
        out[ga] = cap_in
        out[el] = -cap_in * eta_el
        out[he] = -cap_in * eta_he
    elif "p2g" in ct or "powertogas" in ct:
        # Size from the OUTPUT side. ``el_mw`` on a monee ``PowerToGas`` is a
        # solver Var seeded at 1.1 (``monee.model.multi:652``), not a rating:
        # reading it made all 26 units on simbench_lv_gas_dependent report an
        # identical 1.1 MW while their real rates vary 5x, overstating fleet
        # gas output 214x (20.02 MW vs a measured 0.093) and fleet electricity
        # draw to 64x the grid's ENTIRE demand. The L3 cascade then zeroed every
        # P2G to protect the electricity row, so gas CPs never dispatched.
        # ``gas_mass_flow_kgs`` carries the constructor's
        # ``mass_flow_setpoint_kgs`` — the actual sizing — as it does for CHP/G2P.
        cap_out = kgps_to_mw(
            abs(_first_nonzero("gas_mass_flow_kgs", "mass_flow_setpoint_kgs"))
        )
        if cap_out <= 0:
            return {}
        eta = _first_nonzero("eta_gas", "efficiency", default=0.6)
        out[ga] = -cap_out
        out[el] = cap_out / eta if eta > 0 else cap_out
    elif "g2p" in ct or "gastopower" in ct:
        cap_in = kgps_to_mw(
            abs(_first_nonzero("gas_mass_flow_kgs", "mass_flow_setpoint_kgs"))
        )
        if cap_in <= 0:
            return {}
        eta = _first_nonzero("efficiency_power", "eta_el", "efficiency", default=0.45)
        out[ga] = cap_in
        out[el] = -cap_in * eta
    elif "p2h" in ct or "powertoheat" in ct:
        # PowerToHeatHG exposes ``load_p_mw`` + generic ``efficiency``.
        cap_in = abs(_first_nonzero("el_mw", "load_p_mw"))
        if cap_in <= 0:
            return {}
        eta = _first_nonzero("eta_heat", "efficiency", default=0.95)
        out[el] = cap_in
        out[he] = -cap_in * eta
    return out


def _attach_cp_priority_admm_role(
    roles: list,
    behavior: Any,
    aid: str,
    cp_type: str,
    obs: dict,
    config: RestorationConfiguration,
) -> None:
    """Install :class:`CPPriorityAdmmRole` (sole L3 path under
    ``enable_cp_priority_admm``). No-op when signed capacity can't be
    derived; reachability filter and address books injected by the wire pass.
    """
    sectors = _sectors_for_cp_type(cp_type)
    if not sectors:
        return
    capacity_by_sector = _cp_signed_capacity_by_sector(cp_type, obs)
    if not capacity_by_sector:
        return
    roles.append(
        CPPriorityAdmmRole(
            behavior,
            cp_id=aid,
            capacity_by_sector=capacity_by_sector,
            bridged_sectors=sectors,
            algorithm=config.cp_admm_algorithm,
            r_regularization=config.cp_admm_r_regularization,
            heat_supply_from_deficit=config.enable_heat_cp_supply,
            demand_union=config.enable_cp_demand_union,
            gossip_warm_start=config.enable_cp_gossip_warm_start,
            watchdog_s=config.holon_watchdog_s,
        )
    )


def _attach_multi_community_cp_role(
    roles: list,
    behavior: Any,
    cp_type: str,
    config: RestorationConfiguration,
) -> None:
    """``component_level`` counterpart to :func:`_attach_cp_roles`:
    a single :class:`MultiCommunityCPRole` over the bridged sectors,
    without the flex-actor/ADMM roles (unused in the baseline).
    """
    sectors = _sectors_for_cp_type(cp_type)
    if not sectors:
        return
    roles.append(
        MultiCommunityCPRole(
            behavior,
            sectors,
            ema_alpha=config.cp_oscillation_ema_alpha,
            deadband_mw=config.cp_oscillation_deadband_mw,
            min_interval_s=config.cp_oscillation_min_interval_s,
        )
    )


def _build_cp_flex_actor(
    cp_type: str, obs: dict, priority: int
) -> tuple[Any, list[Sector]]:
    if "chp" in cp_type:
        actor = create_chp_admm_flex_actor(obs, priority)
        return actor, [Sector.ELECTRICITY, Sector.HEAT, Sector.GAS]
    elif "p2g" in cp_type or "powertogas" in cp_type:
        actor = create_p2g_admm_flex_actor(obs, priority)
        return actor, [Sector.ELECTRICITY, Sector.GAS]
    elif "g2p" in cp_type or "gastopower" in cp_type:
        actor = create_g2p_admm_flex_actor(obs, priority)
        return actor, [Sector.ELECTRICITY, Sector.GAS]
    elif "p2h" in cp_type or "powertoheat" in cp_type:
        actor = create_p2h_admm_flex_actor(obs, priority)
        return actor, [Sector.ELECTRICITY, Sector.HEAT]
    return None, []


_LABEL_PROPAGATION_RADIUS: dict[Sector, int] = {
    Sector.ELECTRICITY: 2,
    Sector.GAS: 2,
    Sector.HEAT: 2,
}

# Weighting horizon for line-home demand ranking; must exceed the max priority tier.
_LINE_HOME_PRIORITY_TIERS = 10


def _node_priority_weighted_demand(
    node_id: Any, monee_net: Any, priorities: dict[str, int]
) -> float:
    """Sum of priority-weighted load capacity at a node (generators and
    zero-cap children skipped). Lines home to the lower-demand endpoint.
    """
    try:
        node = monee_net.node_by_id(node_id)
    except Exception:
        return 0.0
    total = 0.0
    P = _LINE_HOME_PRIORITY_TIERS
    for cid in getattr(node, "child_ids", []) or []:
        try:
            child = monee_net.child_by_id(cid)
        except Exception:
            continue
        obs = dict(child.model.values)
        cap = obs_capacity(obs)
        if cap <= 0:
            continue
        aid = child_aid(cid)
        tier = priorities.get(aid, 1)
        weight = 2.0 ** max(0, P - tier)
        total += weight * cap
    return total


def _line_home_endpoint(branch: Any, monee_net: Any, priorities: dict[str, int]) -> Any:
    """PowerLine home endpoint: lower priority-weighted demand wins,
    ties break to the smaller node id.

    A childless junction is avoided: the monitor attaches (and gets its home
    leader) only via a home-node child in some community, so homing to a
    child-less endpoint would leave home_leader_addr=None and endpoint relief
    a silent no-op. Prefer the child-bearing endpoint when exactly one has children.
    """

    def _has_children(nid: Any) -> bool:
        try:
            node = monee_net.node_by_id(nid)
        except Exception:
            return False
        return bool(getattr(node, "child_ids", []) or [])

    a, b = branch.id[0], branch.id[1]
    has_a, has_b = _has_children(a), _has_children(b)
    if has_a != has_b:
        return a if has_a else b
    pwd_a = _node_priority_weighted_demand(a, monee_net, priorities)
    pwd_b = _node_priority_weighted_demand(b, monee_net, priorities)
    if pwd_a < pwd_b:
        return a
    if pwd_b < pwd_a:
        return b
    return (
        a
        if (a < b if isinstance(a, int) and isinstance(b, int) else str(a) < str(b))
        else b
    )


def _branch_downstream_load_addrs(monee_net: Any, world: Any) -> dict[str, list[Any]]:
    """Per electricity PowerLine branch, addresses of the loads downstream
    of it (the subtree cut off the slack when removed) — the bidder set for
    the branch-downstream auction. Empty when removal yields no clean cut.
    """
    from collections import defaultdict, deque

    # node_id -> [load addr], plus the slack node set.
    node_loads: dict[Any, list[Any]] = defaultdict(list)
    slack_nodes: set[Any] = set()
    for child in monee_net.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid):
            slack_nodes.add(child.node_id)
            continue
        try:
            cap = obs_capacity(dict(m.values))
        except Exception:
            continue
        if cap <= 0:  # generators/non-loads can't be shed
            continue
        ag = world.agents.get(child_aid(child.id))
        if ag is not None:
            node_loads[child.node_id].append(ag.addr)

    # Undirected electricity adjacency keyed by branch aid so one edge
    # can be excluded during the cut test.
    adj: dict[Any, set[tuple[Any, str]]] = defaultdict(set)
    endpoints: dict[str, tuple[Any, Any]] = {}
    for branch in monee_net.branches:
        if _is_cp_branch(branch):
            continue
        if _branch_sector_str(branch, monee_net) != "electricity":
            continue
        # Open backup ties are non-conductive; including them closes cycles
        # so no branch yields a clean cut (empty bidder sets everywhere).
        if not int(getattr(branch.model, "on_off", 1) or 0):
            continue
        a, b = branch.id[0], branch.id[1]
        b_aid = create_branch_aid(branch.id)
        adj[a].add((b, b_aid))
        adj[b].add((a, b_aid))
        endpoints[b_aid] = (a, b)

    def _reach(start: set[Any], skip_aid: str) -> set[Any]:
        seen = set(start)
        dq = deque(start)
        while dq:
            n = dq.popleft()
            for nb, e_aid in adj.get(n, ()):
                if e_aid == skip_aid or nb in seen:
                    continue
                seen.add(nb)
                dq.append(nb)
        return seen

    result: dict[str, list[Any]] = {}
    for b_aid, (a, b) in endpoints.items():
        reach = _reach(slack_nodes, b_aid)  # nodes still fed without this branch
        a_up, b_up = a in reach, b in reach
        if a_up == b_up:
            # No clean subtree (both fed, or both cut off).
            result[b_aid] = []
            continue
        down_root = b if a_up else a
        comp = _reach({down_root}, b_aid)
        addrs: list[Any] = []
        for nd in comp:
            addrs.extend(node_loads.get(nd, ()))
        result[b_aid] = addrs
    return result


def _build_topologies(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    config: RestorationConfiguration,
    priorities: dict[str, int] | None = None,
) -> None:
    priorities = priorities or {}
    # --- Per-sector physical topologies (Level-1) ---
    # Each ``sector_grid_<sector>`` mirrors one network's adjacency; label
    # propagation later carves it into bounded sub-communities.
    sector_grid_topos: dict[Sector, Topology] = {}
    for sector in _SECTORS:
        sector_str = _SECTOR_GRID_MATCH[sector]
        include_branches = ["HeatExchanger"] if sector == Sector.HEAT else []
        with create_topology(tid=f"sector_grid_{sector.value}") as t:
            topology_based_on_sector_grid(
                monee_net,
                t,
                world,
                sector=sector_str,
                include_nodes=False,
                include_childs=True,
                include_branches=include_branches,
            )
        sector_grid_topos[sector] = t

    # --- PowerLine home-endpoint resolution ---
    # Each PowerLine joins one endpoint group (lower-demand). Built before
    # the groups loop so it attaches to the right community.
    powerline_home_node: dict[str, Any] = {}
    powerline_branch_agent: dict[str, Any] = {}
    if config.enable_line_loading_constraint:
        for branch in monee_net.branches:
            b_aid = create_branch_aid(branch.id)
            agent_obj = world.agents.get(b_aid)
            if agent_obj is None:
                continue
            if _is_cp_branch(branch):
                continue
            sector_str = _branch_sector_str(branch, monee_net)
            if sector_str != "electricity":
                continue
            home_node_id = _line_home_endpoint(branch, monee_net, priorities)
            powerline_home_node[b_aid] = home_node_id
            powerline_branch_agent[b_aid] = agent_obj

    # --- Groups topology: one cluster per sub-community ---
    # Each partition becomes one topology node holding all member agents
    # (mutual NORMAL neighbours after injection, so gossip works unchanged).
    group_leaders_by_sector: dict[Sector, list] = {}
    # Static member set per leader for ``DynamicRepartitionRole`` to compare
    # against post-failure reachability.
    leader_to_members: dict[Any, list[Any]] = {}
    branch_to_leader: dict[str, Any] = {}
    # Per-sector, per-coalition child aids for per-coalition balance series;
    # keyed by sector value + index for stable CSV columns.
    coalition_members_by_sector: dict[str, dict[int, list[str]]] = {
        s.value: {} for s in _SECTORS
    }
    with create_topology(tid="groups") as groups_topo:
        for sector in _SECTORS:
            radius = (
                config.community_label_propagation_radius
                or _LABEL_PROPAGATION_RADIUS.get(sector, 2)
            )
            communities = communities_from_topology(
                sector_grid_topos[sector],
                max_radius=radius,
                method=config.community_partition_method,
                modularity_iterations=config.community_modularity_iterations,
                modularity_resolution=config.community_modularity_resolution,
            )
            for members in communities:
                if not members:
                    continue
                # Add PowerLine branch agents whose home endpoint is in
                # this community (single-home: one group each).
                if sector == Sector.ELECTRICITY and powerline_home_node:
                    member_aids = {m.aid for m in members}
                    for b_aid, home_node_id in list(powerline_home_node.items()):
                        try:
                            home_node = monee_net.node_by_id(home_node_id)
                        except Exception:
                            continue
                        home_child_aids = {
                            child_aid(cid)
                            for cid in getattr(home_node, "child_ids", []) or []
                        }
                        if home_child_aids & member_aids:
                            branch_agent = powerline_branch_agent.get(b_aid)
                            if branch_agent is not None and branch_agent not in members:
                                members.append(branch_agent)
                                # Mark owner; home_leader_addr set below
                                # once the leader is known.
                                branch_to_leader[b_aid] = None
                            # Single-home: drop so it isn't attached again.
                            powerline_home_node.pop(b_aid, None)
                node_id = groups_topo.add_node(*members)
                leader = members[0]
                leader_to_members[leader] = list(members)
                groups_topo.set_characteristic(node_id, leader, "leader")
                # Build-time id: no agent context here, so derive from the
                # coalition's own identity rather than a counter.
                community_id = deterministic_uuid(sector.value, node_id, leader.aid)
                # Child aids in this coalition (branch agents skipped — no
                # ``regulation`` key); index = position within the sector.
                child_member_aids = [m.aid for m in members if is_child_aid(m.aid)]
                coalition_idx = len(coalition_members_by_sector[sector.value])
                coalition_members_by_sector[sector.value][coalition_idx] = (
                    child_member_aids
                )
                for member in members:
                    member.add_role(PreAssignedCommunityRole(community_id))
                    # Fill home_leader for branches just attached here.
                    if (
                        member.aid in branch_to_leader
                        and branch_to_leader[member.aid] is None
                    ):
                        branch_to_leader[member.aid] = leader
                group_leaders_by_sector.setdefault(sector, []).append(leader)
            sizes = sorted(len(c) for c in communities) if communities else []
            try:
                # Recompute the label dict for the modularity diagnostic
                # (communities_from_topology discards it).
                if config.community_partition_method == "modularity":
                    lbl = modularity_partition(
                        sector_grid_topos[sector].graph,
                        max_iterations=config.community_modularity_iterations,
                        resolution=config.community_modularity_resolution,
                    )
                elif config.community_partition_method == "connected_component":
                    lbl = connected_component_partition(sector_grid_topos[sector].graph)
                else:
                    lbl = label_propagation_partition(
                        sector_grid_topos[sector].graph, max_radius=radius
                    )
                q_score = modularity_of_partition(
                    sector_grid_topos[sector].graph,
                    lbl,
                    resolution=config.community_modularity_resolution,
                )
            except Exception:
                q_score = float("nan")
            logger.info(
                "Sector %s [%s] partition: n_comm=%d sizes(min/median/max)=%s/%s/%s "
                "modularity_Q=%.4f",
                sector.value,
                config.community_partition_method,
                len(communities),
                sizes[0] if sizes else 0,
                sizes[len(sizes) // 2] if sizes else 0,
                sizes[-1] if sizes else 0,
                q_score,
            )

    # Patch each branch monitor's home_leader_addr now leaders are known.
    if branch_to_leader:
        for b_aid, leader in branch_to_leader.items():
            if leader is None:
                continue
            branch_agent = powerline_branch_agent.get(b_aid)
            if branch_agent is None:
                continue
            for role in getattr(branch_agent, "roles", []):
                if (
                    isinstance(role, GridConstraintMonitor)
                    and role.branch_id is not None
                ):
                    role.home_leader_addr = leader.addr

    # Branch-downstream relief: give each branch monitor the loads through
    # it so an overload auction relieves THAT line (else legacy endpoint relief).
    if config.enable_branch_downstream_relief and powerline_branch_agent:
        downstream = _branch_downstream_load_addrs(monee_net, world)
        attached = n_loads = 0
        for b_aid, addrs in downstream.items():
            if not addrs:
                continue
            branch_agent = powerline_branch_agent.get(b_aid)
            if branch_agent is None:
                continue
            for role in getattr(branch_agent, "roles", []):
                if (
                    isinstance(role, GridConstraintMonitor)
                    and role.branch_id is not None
                ):
                    role.set_downstream_loads(addrs)
                    attached += 1
                    n_loads += len(addrs)
        logger.info(
            "Branch-downstream relief: attached downstream load sets to %d "
            "branch monitors (%d load-targets total)",
            attached,
            n_loads,
        )

    # Patch every SlackBudgetMonitor's home_leader_addr so it routes the
    # over-budget magnitude into L1's QP curtailment (needed when there's no L2).
    for leader, members in leader_to_members.items():
        for member in members:
            for role in getattr(member, "roles", []):
                if isinstance(role, SlackBudgetMonitor):
                    role.home_leader_addr = leader.addr

    # Grid topology: node agents only, for GridReconfigurator path search.
    with create_topology(tid="grid") as grid_topo:
        topology_based_on_grid(
            monee_net,
            grid_topo,
            world,
            include_childs=False,
            include_cps=False,
        )

    # CPs topology: CP agents cluster; the cluster leader triggers ADMM.
    with create_topology(tid="cps") as cps_topo:
        topology_based_on_grid_groups(
            monee_net,
            cps_topo,
            world,
            separate_sectors=None,
            include_cps=True,
            include_nodes=False,
            include_childs=False,
        )

    # Per-leader maps for the dynamic re-partition role (node ids, member
    # addresses, branch adjacency). Recomputed here, not threaded through.
    aid_to_node_id: dict[str, Any] = {
        child_aid(c.id): c.node_id for c in monee_net.childs
    }
    aid_to_addr: dict[str, Any] = {
        aid: agent.addr for aid, agent in world.agents.items()
    }
    sector_branches_by_sector: dict[Sector, dict[tuple, tuple[Any, Any]]] = {
        s: {} for s in _SECTORS
    }
    sector_value_to_enum = {s.value: s for s in _SECTORS}
    for branch in monee_net.branches:
        sec = _branch_sector_str(branch, monee_net)
        if sec in sector_value_to_enum:
            a, b = branch.id[0], branch.id[1]
            sector_branches_by_sector[sector_value_to_enum[sec]][branch.id] = (a, b)

    # Central grid mirror shared by the L2/L3 dynamic roles; only
    # ``set_on_branch_failure`` (below) mutates it.
    mirror = mirror_from_monee(
        monee_net,
        branch_sector_resolver=lambda b: _branch_sector_str(b, monee_net),
    )
    behavior._scare_topology_mirror = mirror

    # Leader-aid -> node-id lookup: per-sector for L2 dynamic-holon, with
    # a cross-sector union below for L3 dynamic-connector.
    sector_leader_node_ids: dict[Sector, dict[str, Any]] = {}
    for sec, leaders in group_leaders_by_sector.items():
        sector_leader_node_ids[sec] = {
            ldr.aid: aid_to_node_id.get(ldr.aid)
            for ldr in leaders
            if aid_to_node_id.get(ldr.aid) is not None
        }
    # Cross-sector union for the L3 lookup.
    all_leader_node_ids: dict[str, Any] = {}
    for table in sector_leader_node_ids.values():
        all_leader_node_ids.update(table)

    # CP metadata ``{cp_aid: meta}`` for the L3 coord (component membership,
    # setpoint inputs, dispatch addr). Empty when CP ADMM is off.
    cp_meta_by_aid: dict[str, dict[str, Any]] = {}
    if config.enable_cp_admm or config.enable_cp_priority_admm:
        # Node-hosted CPs (e.g. CHP at a coupling node).
        for node in monee_net.nodes:
            cp_type = _detect_cp_type_for_node(node, monee_net)
            if cp_type is None:
                continue
            aid = node_aid(node.id)
            agent = world.agents.get(aid)
            if agent is None:
                continue
            obs = dict(node.model.values)
            cp_meta_by_aid[aid] = {
                "sectors": _sectors_for_cp_type(cp_type),
                "coupling_ratios": _cp_coupling_ratios(cp_type),
                "rated_capacity_mw": _cp_rated_capacity_mw(obs, cp_type),
                "addr": agent.addr,
                "node_id": node.id,
            }
        # Branch-hosted CPs (P2G / G2P / P2H lines).
        for branch in monee_net.branches:
            if not _is_cp_branch(branch):
                continue
            branch_type = _model_type_name(branch)
            aid = create_branch_aid(branch.id)
            agent = world.agents.get(aid)
            if agent is None:
                continue
            obs = dict(branch.model.values)
            cp_meta_by_aid[aid] = {
                "sectors": _sectors_for_cp_type(branch_type),
                "coupling_ratios": _cp_coupling_ratios(branch_type),
                "rated_capacity_mw": _cp_rated_capacity_mw(obs, branch_type),
                "addr": agent.addr,
                # Either endpoint gives the same component; use from_node.
                "node_id": branch.id[0],
            }

    # Per-sector leader address book for the cross-sector coalition initiator.
    peer_leader_addrs: dict[Sector, dict[str, Any]] = {
        sec: {ldr.aid: ldr.addr for ldr in leaders}
        for sec, leaders in group_leaders_by_sector.items()
    }

    # Attach L2 / fallback roles to each leader and mark as connectors for
    # the cps<->groups link. Fallback always installs; holonic is gated.
    for sector, leaders in group_leaders_by_sector.items():
        for leader in leaders:
            mark_as_connector(leader, connector_type=sector.value)

            # L2 dynamic-holon filter, built before HolonicCommunityRole so
            # it can be passed as ``live_member_filter``.
            dyn_holon_role = None
            leader_node = aid_to_node_id.get(leader.aid)
            if (
                config.enable_holonic
                and config.enable_dynamic_holon_topology
                and leader_node is not None
            ):
                dyn_holon_role = DynamicHolonRole(
                    behavior,
                    sector,
                    leader_node,
                    sector_leader_node_ids.get(sector, {}),
                    mirror,
                )

            # CoalitionConstraintStore per (leader, sector), shared between
            # L2.5 (writer) and L2 (reader). Only when coalitions enabled.
            coalition_store = None
            if config.enable_holon_summary and config.enable_holon_coalition:
                coalition_store = CoalitionConstraintStore()

            if config.enable_holonic:
                leader.add_role(
                    HolonicCommunityRole(
                        sector,
                        max_holon_size=config.holon_max_size,
                        admm_max_iters=config.holon_admm_max_iters,
                        admm_abs_tol=config.holon_admm_abs_tol,
                        enable_tier_stratified_admm=config.enable_tier_stratified_holon_admm,
                        priority_tiers=config.priority_tiers,
                        admm_scope=config.holon_admm_scope,
                        enable_priority_allocation=config.enable_priority_holon_allocation,
                        enable_change_only_dispatch=config.enable_change_only_dispatch,
                        heat_rebalance_period_s=(
                            config.heat_l2_rebalance_s
                            if config.enable_heat_l2_dispatch
                            else None
                        ),
                        live_member_filter=dyn_holon_role,
                        coalition_constraint_store=coalition_store,
                        my_node_id=leader_node,
                        leader_node_ids=sector_leader_node_ids.get(sector, {}),
                        topology_mirror=mirror,
                        watchdog_s=config.holon_watchdog_s,
                    )
                )
            if dyn_holon_role is not None:
                leader.add_role(dyn_holon_role)
            leader.add_role(LocalGenerationFallbackRole(behavior, sector))

            # Failure-driven re-partition: each leader gets its sector's
            # branch slice and member node map. Triggered by BranchFailureEvent.
            members = leader_to_members.get(leader, [leader])
            member_node_ids = {
                m.aid: aid_to_node_id[m.aid] for m in members if m.aid in aid_to_node_id
            }
            member_addrs = {m.aid: aid_to_addr.get(m.aid, m.addr) for m in members}
            my_node_id = aid_to_node_id.get(leader.aid)
            if my_node_id is not None:
                leader.add_role(
                    DynamicRepartitionRole(
                        behavior,
                        sector,
                        my_node_id,
                        member_node_ids,
                        member_addrs,
                        sector_branches_by_sector.get(sector, {}),
                    )
                )

            # L2.5 cross-holon priority-inversion detector + coalition
            # formation, publishing per-tier served/demand on the mesh.
            if config.enable_holon_summary:
                # Node map for deliverability-aware coalition ADMM, via the
                # shared mirror (so failures propagate across L2/L2.5).
                summary_member_nodes = {
                    m.aid: aid_to_node_id[m.aid]
                    for m in leader_to_members.get(leader, [leader])
                    if m.aid in aid_to_node_id
                }
                leader.add_role(
                    HolonSummaryRole(
                        behavior,
                        sector,
                        period_s=config.holon_summary_period_s,
                        inversion_tol=config.holon_summary_inversion_tol,
                        enable_coalition=config.enable_holon_coalition,
                        coalition_accept_window_s=(
                            config.holon_coalition_accept_window_s
                        ),
                        coalition_constraint_ttl_s=(
                            config.holon_coalition_constraint_ttl_s
                        ),
                        priority_tiers=config.priority_tiers,
                        admm_max_iters=config.holon_admm_max_iters,
                        admm_abs_tol=config.holon_admm_abs_tol,
                        cp_budget_nominal=config.enable_cp_nominal_budget,
                        coalition_delivered_supply=(
                            config.enable_coalition_delivered_supply
                        ),
                        cp_commitment_actuatable=(
                            config.enable_cp_admm and not config.enable_cp_priority_admm
                        ),
                        my_node_id=aid_to_node_id.get(leader.aid),
                        member_node_ids=summary_member_nodes,
                        mirror=mirror,
                        constraint_store=coalition_store,
                        enable_cross_sector_coalitions=(
                            config.enable_cross_sector_coalitions
                        ),
                        cp_meta=cp_meta_by_aid,
                        peer_leader_addrs=peer_leader_addrs,
                        enable_heat_cp_supply=config.enable_heat_cp_supply,
                        heat_refresh_s=config.heat_cp_supply_refresh_s,
                        watchdog_s=config.holon_watchdog_s,
                    )
                )

    # Holons topology (L2): chunk same-sector leaders into cliques; collect
    # coalition unions per holon. Legacy holon/sector scope only — component
    # scope elects a per-component coordinator via the holon_summary mesh below
    # and never reads the clique topology, so building it would be dead work.
    holon_members_by_sector: dict[str, dict[int, list[str]]] = {
        s.value: {} for s in _SECTORS
    }
    if config.enable_holonic and config.holon_admm_scope != "component":
        holon_chunk_size = config.holon_max_size
        with create_topology(tid="holons") as holon_topo:
            for sector, leaders in group_leaders_by_sector.items():
                if len(leaders) < 2:
                    continue
                ordered = sorted(leaders, key=lambda a: a.aid)
                for start in range(0, len(ordered), holon_chunk_size):
                    chunk = ordered[start : start + holon_chunk_size]
                    if len(chunk) < 2:
                        continue
                    chunk_nids = [holon_topo.add_node(member) for member in chunk]
                    for i, nid_a in enumerate(chunk_nids):
                        for nid_b in chunk_nids[i + 1 :]:
                            holon_topo.add_edge(nid_a, nid_b)
                    # Holon child aids = union of chunk coalitions (ordered,
                    # deduped).
                    holon_child_aids: list[str] = []
                    seen_aids: set[str] = set()
                    for leader in chunk:
                        for member in leader_to_members.get(leader, [leader]):
                            aid = getattr(member, "aid", None)
                            if aid and is_child_aid(aid) and aid not in seen_aids:
                                seen_aids.add(aid)
                                holon_child_aids.append(aid)
                    holon_idx = len(holon_members_by_sector[sector.value])
                    holon_members_by_sector[sector.value][holon_idx] = holon_child_aids

    # L2.5 holon-summary mesh: per-sector full clique of leaders broadcasting
    # ``HolonSummary``. Built whenever holonic is on — NOT gated on
    # enable_holon_summary: this mesh is also the election substrate for the
    # per-component L2 ADMM coordinator and the LeaderEmerged re-registration
    # broadcast; only the summary/coalition ROLES stay gated.
    if config.enable_holonic:
        # CP agents bridging each sector join too under the L3 priority-ADMM
        # cutover (CPPriorityAdmmRole reads supply/demand from HolonSummary).
        cp_agents_by_sector: dict[Sector, list[Any]] = {sec: [] for sec in _SECTORS}
        if config.enable_cp_priority_admm:
            for aid_, meta in cp_meta_by_aid.items():
                agent = world.agents.get(aid_)
                if agent is None:
                    continue
                # Under demand_union a CP joins ALL sector meshes (not just its
                # bridged ones) so e.g. a P2G gossip initiator physically receives
                # heat HolonSummary and can fold heat demand into its round. CPs
                # are passive members, so extra membership only adds delivery.
                join_secs = (
                    list(cp_agents_by_sector)
                    if config.enable_cp_demand_union
                    else meta.get("sectors", [])
                )
                for sec in join_secs:
                    if sec in cp_agents_by_sector:
                        cp_agents_by_sector[sec].append(agent)
        for sector, leaders in group_leaders_by_sector.items():
            cps_on_mesh = cp_agents_by_sector.get(sector, [])
            participants = list(leaders) + list(cps_on_mesh)
            if len(participants) < 2:
                continue
            with create_topology(tid=f"holon_summary_{sector.value}") as t:
                nids = [t.add_node(a) for a in participants]
                for i, nid_a in enumerate(nids):
                    for nid_b in nids[i + 1 :]:
                        t.add_edge(nid_a, nid_b)

    # CP-side topology wiring (connector marks + cps<->groups cross-link),
    # needed under CP ADMM and the ``component_level`` baseline.
    if (
        config.enable_cp_admm
        or config.cps_join_communities
        or config.enable_cp_priority_admm
    ):
        # Mark CP branch agents as connectors for the sectors they bridge.
        for branch in monee_net.branches:
            if not _is_cp_branch(branch):
                continue
            b_aid = create_branch_aid(branch.id)
            if b_aid not in world.agents:
                continue
            for sector in _sectors_for_cp_type(_model_type_name(branch)):
                mark_as_connector(world.agents[b_aid], connector_type=sector.value)

        # Mark CP node agents as connectors for the sectors they bridge.
        for node in monee_net.nodes:
            cp_type = _detect_cp_type_for_node(node, monee_net)
            if cp_type is None:
                continue
            n_aid = _node_aid(node.id)
            if n_aid not in world.agents:
                continue
            for sector in _sectors_for_cp_type(cp_type):
                mark_as_connector(world.agents[n_aid], connector_type=sector.value)

        # Link the CP topology to the groups topology for each sector.
        for sector in _SECTORS:
            connect_topologies(cps_topo, groups_topo, sector.value)

    # One EnergyConverterRole index reused by both CP passes below. Sound because
    # pass-1 only adds DynamicConnectorRole (a different type), so the
    # first-EnergyConverterRole per agent is stable between the passes.
    cp_role_index = (
        role_index(world.agents.values(), EnergyConverterRole)
        if config.enable_cp_admm
        else {}
    )

    # L3 dynamic CP-connector filter, built after the CP topology so it can
    # walk the populated EnergyConverterRole agents. Purely additive.
    if config.enable_cp_admm and config.enable_dynamic_cp_topology:
        for aid, cp_role in cp_role_index.items():
            agent = world.agents[aid]
            cp_node_id = aid_to_node_id.get(aid)
            if cp_node_id is None:
                # Branch-hosted CP: use the first endpoint (reachability is
                # symmetric, so either gives the same classes).
                for branch in monee_net.branches:
                    if create_branch_aid(branch.id) == agent.aid:
                        cp_node_id = branch.id[0]
                        break
            if cp_node_id is None:
                continue
            dyn_conn = DynamicConnectorRole(
                behavior,
                cp_node_id,
                all_leader_node_ids,
                mirror,
            )
            agent.add_role(dyn_conn)
            cp_role.set_connector_filter(dyn_conn)

    # Wire multi-sector L3 state on every CP role under ``enable_cp_admm``
    # (coord election + dispatch are core); before this it's legacy per-CP.
    if config.enable_cp_admm:
        for aid, cp_role in cp_role_index.items():
            meta = cp_meta_by_aid.get(aid)
            if meta is None:
                continue
            cp_role.wire_multi_sector_l3(
                topology_mirror=mirror,
                my_node_id=meta["node_id"],
                cp_meta_by_aid=cp_meta_by_aid,
                leader_addrs_by_sector=peer_leader_addrs,
                leader_node_ids=all_leader_node_ids,
            )

    # Wire each :class:`CPPriorityAdmmRole` with the mirror and address books
    # to gossip ``CPSummary`` and filter peer CPs by reachability.
    if config.enable_cp_priority_admm:
        peer_cp_addrs = {aid_: meta["addr"] for aid_, meta in cp_meta_by_aid.items()}
        peer_cp_node_ids = {
            aid_: meta["node_id"] for aid_, meta in cp_meta_by_aid.items()
        }
        for agent in world.agents.values():
            cp_role = first_role(agent, CPPriorityAdmmRole)
            if cp_role is None:
                continue
            meta = cp_meta_by_aid.get(agent.aid)
            if meta is None:
                continue
            # Inject home_node_id at wire time so the role tests in isolation.
            cp_role.home_node_id = meta["node_id"]
            # Exclude self from the peer book.
            peers_excl_self = {
                aid_: addr for aid_, addr in peer_cp_addrs.items() if aid_ != agent.aid
            }
            cp_role.wire(
                topology_mirror=mirror,
                peer_cp_addrs=peers_excl_self,
                peer_cp_node_ids=peer_cp_node_ids,
            )

    # On each BranchFailureEvent mark the edge broken in both the grid
    # topology and the mirror (idempotent).
    def _on_branch_failed(bid: tuple) -> None:
        mirror.mark_broken(tuple(bid))
        _mark_grid_edge_broken(grid_topo, bid)
        # Force one immediate recompute so the first post-failure action sees
        # true state instead of stale pre-failure voltages the cooldown would
        # otherwise serve (one solve per failure; see enable_recompute_on_failure).
        if getattr(config, "enable_recompute_on_failure", False):
            _flush = getattr(behavior, "flush_energy_flow", None)
            if callable(_flush):
                try:
                    _flush()
                except Exception:  # noqa: BLE001 — never let a flush abort failure handling
                    pass

    behavior.set_on_branch_failure(_on_branch_failed)

    # Stash coalition / holon maps for ``_register_recordings``.
    behavior._scare_coalitions = coalition_members_by_sector
    behavior._scare_holons = holon_members_by_sector


def _mark_grid_edge_broken(grid_topo: Topology, branch_id: tuple) -> None:
    from_aid = _node_aid(branch_id[0])
    to_aid = _node_aid(branch_id[1])
    from_nid = to_nid = None

    for nid in grid_topo.graph.nodes:
        node_data = grid_topo.graph.nodes[nid].get("node")
        if node_data is None:
            continue
        for a in node_data.agents:
            if a.aid == from_aid:
                from_nid = nid
            elif a.aid == to_aid:
                to_nid = nid
        if from_nid is not None and to_nid is not None:
            break

    if from_nid is not None and to_nid is not None:
        # CP branches span sector grids, so the same-sector topology may
        # lack the edge even when both nodes appear; missing = nothing to mark.
        edges = grid_topo.graph.edges
        if (from_nid, to_nid) in edges or (to_nid, from_nid) in edges:
            with modify_topology(grid_topo) as t:
                t.set_edge_state(from_nid, to_nid, State.BROKEN)


def _add_system_behaviors(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    strategy: SystemStrategy,
    config: RestorationConfiguration,
) -> None:
    def _trigger_balance(role: EnergyBalanceNegotiator, event: Any) -> None:
        # On heat, skip failure events (a severed corridor isn't a setpoint
        # mismatch); ConstraintViolation / Reconfig are heat's triggers.
        if role.sector == Sector.HEAT and isinstance(
            event, (CustomFailureEvent, BranchFailureEvent)
        ):
            return
        role.context.schedule_instant_task(role.trigger_balance_negotiation())

    def _trigger_cp(role: EnergyConverterRole, event: Any) -> None:
        role.context.schedule_instant_task(role.trigger_cp_negotiation())

    def _trigger_repartition(role: DynamicRepartitionRole, event: Any) -> None:
        branch_id = getattr(event, "branch_id", None)
        if branch_id is None:
            return
        role.on_branch_failure(tuple(branch_id))

    # Every failure feeds repartition (needs only the branch_id),
    # regardless of the FailureNotice propagation flag.
    behavior_in(
        world,
        _trigger_repartition,
        on_global_event=BranchFailureEvent,
        role_types=DynamicRepartitionRole,
    )

    # L2/L3 dynamic-topology triggers: BranchFailureEvent schedules each
    # role's debounced reassess. Off features install no role, so no listeners.
    if config.enable_holonic and config.enable_dynamic_holon_topology:

        def _trigger_dyn_holon(role: DynamicHolonRole, event: Any) -> None:
            branch_id = getattr(event, "branch_id", None)
            if branch_id is None:
                return
            role.on_branch_failure(tuple(branch_id))

        behavior_in(
            world,
            _trigger_dyn_holon,
            on_global_event=BranchFailureEvent,
            role_types=DynamicHolonRole,
        )

    if config.enable_cp_admm and config.enable_dynamic_cp_topology:

        def _trigger_dyn_connector(role: DynamicConnectorRole, event: Any) -> None:
            branch_id = getattr(event, "branch_id", None)
            if branch_id is None:
                return
            role.on_branch_failure(tuple(branch_id))

        behavior_in(
            world,
            _trigger_dyn_connector,
            on_global_event=BranchFailureEvent,
            role_types=DynamicConnectorRole,
        )

    # ``component_level`` baseline: reset each MultiCommunityCPRole's EMA
    # on a branch failure so stale (possibly islanded) signal doesn't bleed in.
    if config.cps_join_communities:

        def _trigger_multi_community_cp(role: MultiCommunityCPRole, event: Any) -> None:
            branch_id = getattr(event, "branch_id", None)
            if branch_id is None:
                return
            role.on_branch_failure(tuple(branch_id))

        behavior_in(
            world,
            _trigger_multi_community_cp,
            on_global_event=BranchFailureEvent,
            role_types=MultiCommunityCPRole,
        )

    # Invalidate coalition constraints on a BranchFailureEvent so the L2
    # ADMM round redecides without a stale pre-failure coalition fraction.
    if config.enable_holon_summary and config.enable_holon_coalition:

        def _trigger_coalition_invalidation(role: HolonSummaryRole, event: Any) -> None:
            branch_id = getattr(event, "branch_id", None)
            if branch_id is None:
                return
            role.on_branch_failure(tuple(branch_id))

        behavior_in(
            world,
            _trigger_coalition_invalidation,
            on_global_event=BranchFailureEvent,
            role_types=HolonSummaryRole,
        )

    if strategy in (SystemStrategy.GROUP_TO_CP, SystemStrategy.SIMULTANEOUSLY):
        # CustomFailureEvent always uses the centralised path (no physical
        # branch to propagate through).
        behavior_in(
            world,
            _trigger_balance,
            on_global_event=CustomFailureEvent,
            role_types=EnergyBalanceNegotiator,
        )
        # BranchFailureEvent uses the centralised callback only when
        # distributed FailureNotice propagation is disabled.
        if not config.enable_distributed_failure_notice:
            behavior_in(
                world,
                _trigger_balance,
                on_global_event=BranchFailureEvent,
                role_types=EnergyBalanceNegotiator,
            )

    if (
        strategy in (SystemStrategy.CP_TO_GROUP, SystemStrategy.SIMULTANEOUSLY)
        and config.enable_cp_admm
    ):
        behavior_in(
            world,
            _trigger_cp,
            on_global_event=BranchFailureEvent,
            role_types=EnergyConverterRole,
        )
        behavior_in(
            world,
            _trigger_cp,
            on_global_event=CustomFailureEvent,
            role_types=EnergyConverterRole,
        )

    # Constraint violations trigger rebalancing to restore feasibility.
    behavior_in(
        world,
        _trigger_balance,
        on_global_event=ConstraintViolation,
        role_types=EnergyBalanceNegotiator,
    )

    # After reconfiguration closes tie switches, rebalance to exploit
    # newly reachable resources.
    behavior_in(
        world,
        _trigger_balance,
        on_global_event=ReconfigurationCompletedEvent,
        role_types=EnergyBalanceNegotiator,
    )


def _register_recordings(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    *,
    priorities: dict[str, int] | None = None,
) -> None:
    priorities = priorities or {}
    el_child_aids = _child_aids_for_sector(monee_net, Sector.ELECTRICITY)
    gas_child_aids = _child_aids_for_sector(monee_net, Sector.GAS)
    heat_child_aids = _child_aids_for_sector(monee_net, Sector.HEAT)

    def _sum_regulation(child_aids: list[str]) -> float:
        total = 0.0
        for aid in child_aids:
            obs = behavior.observe(aid)
            total += float(obs.get("regulation", 0.0)) if obs else 0.0
        return total

    record_world(world, "electrical_balance", lambda: _sum_regulation(el_child_aids))
    record_world(world, "gas_balance", lambda: _sum_regulation(gas_child_aids))
    record_world(world, "heat_balance", lambda: _sum_regulation(heat_child_aids))

    # Slack trajectory: one ``slack__<sector>__<aid>`` column per
    # ExtPowerGrid/ExtHydrGrid child, the LP operating point per tick.
    def _slack_obs(aid: str, key: str) -> float:
        obs = behavior.observe(aid)
        if not obs or key not in obs:
            return 0.0
        try:
            return float(obs[key])
        except (TypeError, ValueError):
            return 0.0

    for child in monee_net.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid):
            obs_key = "p_mw"
        elif isinstance(m, ExtHydrGrid):
            obs_key = "mass_flow_kgs"
        else:
            continue
        try:
            node = monee_net.node_by_id(child.node_id)
            sector = sector_from_grid(node.grid)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "slack recording: sector lookup failed for %s: %s", child.id, exc
            )
            continue
        if sector is None:
            continue
        aid = _child_aid(child.id)
        col = f"slack__{sector.value}__{aid}"
        # Default args capture per-child values (avoid late-binding).
        record_world(world, col, lambda a=aid, k=obs_key: _slack_obs(a, k))

    # Gas regulator control: the slack pressure setpoint per tick (the
    # GasPressureRegulator's lever). Before the regulator first actuates,
    # ``lookup_slack_pressure`` is None — fall back to the pinned node pressure.
    def _slack_pressure_setpoint(aid: str) -> float:
        sp = lookup_slack_pressure(behavior, aid)
        if sp is not None:
            return float(sp)
        return _slack_obs(aid, "pressure_pu")

    for child in monee_net.childs:
        if not isinstance(child.model, ExtHydrGrid):
            continue
        try:
            sector = sector_from_grid(monee_net.node_by_id(child.node_id).grid)
        except Exception:  # noqa: BLE001
            continue
        if sector is not Sector.GAS:
            continue
        aid = _child_aid(child.id)
        record_world(
            world,
            f"slack_pressure__gas__{aid}",
            lambda a=aid: _slack_pressure_setpoint(a),
        )

    # Per-coalition (L1) and per-holon (L2) regulation sums, so plots show
    # each group converging independently (the aggregate hides that).
    coalitions = getattr(behavior, "_scare_coalitions", {}) or {}
    holons = getattr(behavior, "_scare_holons", {}) or {}

    def _make_sum(aids: list[str]):
        # Capture ``aids`` to avoid late-binding.
        return lambda: _sum_regulation(aids)

    for sec_value, by_idx in coalitions.items():
        for idx, member_aids in by_idx.items():
            if not member_aids:
                continue
            record_world(
                world,
                f"coalition_balance__{sec_value}__{idx:03d}",
                _make_sum(member_aids),
            )
    for sec_value, by_idx in holons.items():
        for idx, member_aids in by_idx.items():
            if not member_aids:
                continue
            record_world(
                world,
                f"holon_balance__{sec_value}__{idx:03d}",
                _make_sum(member_aids),
            )

    # Per-(sector, tier) regulation sum (loads only, to exclude generator/
    # slack noise) so a higher-tier drop with a lower tier still served flags.
    # Gas consumers are Sink models (heat-side return Sinks are a topology
    # artifact, excluded), so gas gets its tier series too.
    load_aids = {
        _child_aid(c.id)
        for c in monee_net.childs
        if isinstance(c.model, (PowerLoad, HeatLoad))
        or (
            isinstance(c.model, Sink) and not _is_heat_side_mass_flow_sink(c, monee_net)
        )
    }
    sector_aids = {
        Sector.ELECTRICITY.value: el_child_aids,
        Sector.GAS.value: gas_child_aids,
        Sector.HEAT.value: heat_child_aids,
    }
    by_sector_tier: dict[tuple[str, int], list[str]] = {}
    for sec_value, aids in sector_aids.items():
        for aid in aids:
            if aid not in load_aids:
                continue
            tier = int(priorities.get(aid, 0))
            if tier < 1:
                continue
            by_sector_tier.setdefault((sec_value, tier), []).append(aid)
    for (sec_value, tier), member_aids in by_sector_tier.items():
        record_world(
            world,
            f"tier_balance__{sec_value}__{tier}",
            _make_sum(member_aids),
        )

    record_agent_having(
        world,
        "regulation",
        EnergyBalanceNegotiator,
        lambda agent: float((behavior.observe(agent.aid) or {}).get("regulation", 0.0)),
    )

    # De-energised readings are dropped to match the compliance gate (isolated
    # gas ~0 or ~sqrt(3), heat t_k~0). See ``is_energised_reading``.
    def _constraint_values(child_aids: list[str], key: str) -> list[float]:
        vals: list[float] = []
        for aid in child_aids:
            obs = behavior.observe(aid)
            if not (obs and key in obs):
                continue
            v = float(obs[key])
            # Drop de-energised and non-finite readings; both are solver
            # artefacts that would poison the min/max/avg aggregates.
            if not is_energised_reading(key, v):
                continue
            vals.append(v)
        return vals

    def _avg_constraint(child_aids: list[str], key: str) -> float:
        vals = _constraint_values(child_aids, key)
        return sum(vals) / len(vals) if vals else 0.0

    def _min_constraint(child_aids: list[str], key: str) -> float:
        vals = _constraint_values(child_aids, key)
        return min(vals) if vals else 0.0

    def _max_constraint(child_aids: list[str], key: str) -> float:
        vals = _constraint_values(child_aids, key)
        return max(vals) if vals else 0.0

    for prefix, aids, key in (
        ("vm_pu", el_child_aids, "vm_pu"),
        ("pressure_pu", gas_child_aids, "pressure_pu"),
        ("t_k", heat_child_aids, "t_k"),
    ):
        record_world(
            world, f"avg_{prefix}", lambda a=aids, k=key: _avg_constraint(a, k)
        )
        record_world(
            world, f"min_{prefix}", lambda a=aids, k=key: _min_constraint(a, k)
        )
        record_world(
            world, f"max_{prefix}", lambda a=aids, k=key: _max_constraint(a, k)
        )

    # --- Line-loading aggregates ---
    # Per-tick max / p95 / avg of electricity branch loading_percent — one
    # thermal-stress panel without a column per branch.
    el_branch_aids: list[str] = []
    for branch in monee_net.branches:
        try:
            sector_str = _branch_sector_str(branch, monee_net)
        except Exception:  # noqa: BLE001
            continue
        if sector_str != "electricity":
            continue
        if _is_cp_branch(branch):
            continue
        el_branch_aids.append(create_branch_aid(branch.id))

    def _branch_loadings() -> list[float]:
        out: list[float] = []
        for aid in el_branch_aids:
            obs = behavior.observe(aid)
            if not obs:
                continue
            vals = obs_constraint_values(obs, Sector.ELECTRICITY)
            lp = vals.get("loading_percent")
            if lp is None:
                continue
            out.append(abs(float(lp)))
        return out

    def _max_loading() -> float:
        v = _branch_loadings()
        return max(v) if v else 0.0

    def _avg_loading() -> float:
        v = _branch_loadings()
        return (sum(v) / len(v)) if v else 0.0

    def _p95_loading() -> float:
        v = sorted(_branch_loadings())
        if not v:
            return 0.0
        # Linear-interpolated 95th percentile.
        idx = 0.95 * (len(v) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(v) - 1)
        frac = idx - lo
        return v[lo] * (1.0 - frac) + v[hi] * frac

    if el_branch_aids:
        record_world(world, "max_line_loading_percent", _max_loading)
        record_world(world, "avg_line_loading_percent", _avg_loading)
        record_world(world, "p95_line_loading_percent", _p95_loading)

    # --- Per-tier demand fulfillment over time ---
    # ``tier_demand_mw`` / ``tier_served_mw`` per tier. Classify via
    # ``child.model.values`` not observe (net-results not built until t=0 solve).
    load_aids_by_tier: dict[int, list[str]] = {}
    _LOAD_CLASSES: tuple[type, ...] = (HeatLoad, PowerLoad)
    for child in monee_net.childs:
        if not isinstance(child.model, _LOAD_CLASSES):
            continue
        model_vals = dict(getattr(child.model, "values", {}) or {})
        cap = obs_capacity(model_vals)
        if not (cap > 0):
            continue
        aid = _child_aid(child.id)
        tier = int(priorities.get(aid, 0)) if priorities else 0
        load_aids_by_tier.setdefault(tier, []).append(aid)

    def _tier_demand(aids: list[str]) -> float:
        total = 0.0
        for aid in aids:
            obs = behavior.observe(aid)
            if obs:
                total += float(obs_capacity(obs))
        return total

    def _tier_served(aids: list[str]) -> float:
        total = 0.0
        for aid in aids:
            obs = behavior.observe(aid)
            if not obs:
                continue
            cap = float(obs_capacity(obs))
            if cap <= 0:
                continue
            sp = float(obs_setpoint(obs))
            total += max(0.0, min(cap, sp))
        return total

    for tier, aids in sorted(load_aids_by_tier.items()):
        if not aids:
            continue
        record_world(
            world,
            f"tier_demand_mw__{tier:02d}",
            lambda aids=aids: _tier_demand(aids),
        )
        record_world(
            world,
            f"tier_served_mw__{tier:02d}",
            lambda aids=aids: _tier_served(aids),
        )

    # Infeasible recompute reuses the old ``_net_results`` (frozen metrics);
    # ``id()`` drifts only on an accepted solve, so record the last-change time.
    # Hold the last-seen results object and compare by identity, not id(): a
    # GC'd results object's address can be reused, aliasing a genuinely new
    # solve as "unchanged" and freezing the freshness timestamp. Retaining the
    # object (the same one behavior._net_results already holds) blocks reuse.
    _freshness_state: dict[str, Any] = {
        "obj": behavior._net_results,
        "t": float(world.clock.time),
    }

    def _last_feasible_solve_t() -> float:
        nr = getattr(behavior, "_net_results", None)
        if nr is not None and nr is not _freshness_state["obj"]:
            _freshness_state["obj"] = nr
            _freshness_state["t"] = float(world.clock.time)
        return _freshness_state["t"]

    record_world(world, "last_feasible_solve_t", _last_feasible_solve_t)

    # --- Physics-extension trajectories ---
    # Gated on the scenario actually attaching the extension so standard
    # campaigns keep their timeseries schema unchanged.
    ext_names = {type(e).__name__ for e in getattr(monee_net, "extensions", ())}

    def _result_net() -> Any:
        nr = getattr(behavior, "_net_results", None)
        return getattr(nr, "network", None)

    if "GasLinepack" in ext_names:

        def _linepack_total_kg() -> float:
            net = _result_net()
            if net is None:
                return 0.0
            total = 0.0
            for branch in net.branches:
                lp = dict(getattr(branch.model, "values", {}) or {}).get("linepack_kg")
                try:
                    total += float(lp)
                except (TypeError, ValueError):
                    continue
            return total

        record_world(world, "linepack_total_kg", _linepack_total_kg)

    if "LumpedThermalCapacitance" in ext_names:
        _water_node_ids = [
            n.id
            for n in monee_net.nodes
            if "water" in str(getattr(n.grid, "name", "") or "").lower()
        ]

        def _junction_temps() -> list[float]:
            net = _result_net()
            if net is None:
                return []
            vals: list[float] = []
            for nid in _water_node_ids:
                try:
                    node = net.node_by_id(nid)
                except Exception:  # noqa: BLE001 — node may be pruned
                    continue
                v = dict(getattr(node.model, "values", {}) or {}).get("t_k")
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v > 0.0:  # de-energised junctions read ~0
                    vals.append(v)
            return vals

        def _junction_t_mean_k() -> float:
            v = _junction_temps()
            return sum(v) / len(v) if v else 0.0

        def _junction_t_min_k() -> float:
            v = _junction_temps()
            return min(v) if v else 0.0

        record_world(world, "ltc_junction_t_mean_k", _junction_t_mean_k)
        record_world(world, "ltc_junction_t_min_k", _junction_t_min_k)

    if getattr(monee_net, "islanding_config", None) is not None:

        def _node_deenergised(n: Any) -> bool:
            # Two channels: static pre-solve pruning (.ignored) AND the
            # islanding MILP's per-node energisation binaries solved to 0 —
            # the latter is the extension's actual shedding decision and never
            # sets .ignored.
            if getattr(n, "ignored", False):
                return True
            vals = dict(getattr(n.model, "values", {}) or {})
            for key in ("e_el", "e_gas", "e_water"):
                v = vals.get(key)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v < 0.5:
                    return True
            return False

        def _nodes_deenergised() -> float:
            net = _result_net()
            if net is None:
                return 0.0
            return float(sum(1 for n in net.nodes if _node_deenergised(n)))

        def _islanded_events() -> float:
            st = getattr(behavior, "stepper", None)
            if st is None:
                return 0.0
            return float(sum(1 for c in st.changes if c.kind == "islanded"))

        record_world(world, "nodes_deenergised", _nodes_deenergised)
        record_world(world, "islanded_events_cum", _islanded_events)

    # --- Emergent metrics ---
    # Per-sector event counts read directly from the diagnostics ledger
    # (emits are role-local, so a global-event hook would never fire).
    def _local_gen_request_count(sec_value: str) -> int:
        return sum(
            1
            for r in _diag.event_log()
            if r.kind == "local_gen_request" and r.sector == sec_value
        )

    def _negotiations_finished_count(sec_value: str) -> int:
        return sum(
            1
            for r in _diag.negotiation_log()
            if r.event == "finished" and r.sector == sec_value
        )

    def _last_event_time(sec_value: str) -> float:
        last = 0.0
        for r in _diag.event_log():
            if r.sector == sec_value and r.t > last:
                last = r.t
        for r in _diag.negotiation_log():
            if r.sector == sec_value and r.t > last:
                last = r.t
        return last

    for sector in _SECTORS:
        s = sector.value
        record_world(
            world,
            f"local_gen_requests_{s}",
            lambda s=s: _local_gen_request_count(s),
        )
        record_world(
            world,
            f"negotiations_finished_{s}",
            lambda s=s: _negotiations_finished_count(s),
        )
        record_world(
            world,
            f"time_since_last_event_{s}",
            lambda s=s: max(0.0, world.clock.time - _last_event_time(s)),
        )


def _child_aids_for_sector(monee_net: Any, sector: Sector) -> list[str]:
    aids = []
    for child in monee_net.childs:
        try:
            node = monee_net.node_by_id(child.node_id)
            if sector_from_grid(node.grid) == sector:
                aids.append(_child_aid(child.id))
        except Exception as exc:
            logger.debug("Could not determine sector for child %s: %s", child.id, exc)
    return aids
