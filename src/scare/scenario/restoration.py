from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from distributed_resource_optimization import (
    create_admm_sharing_data,
    create_admm_start,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
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
    edge_centrality,
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

from scare.base import diagnostics as _diag
from scare.base.admm import (
    ScareDistributedOptimizationRole as DistributedOptimizationRole,
)
from scare.base.channel import SectorImbalanceBeacon
from scare.base.community import (
    communities_from_topology,
    connected_component_partition,
    label_propagation_partition,
    modularity_of_partition,
    modularity_partition,
)
from scare.base.comms import install_perturbation
from scare.base.config import RestorationConfiguration
from scare.base.model import (
    ConstraintViolation,
    ReconfigurationCompletedEvent,
    Sector,
    SystemStrategy,
)
from scare.base.topology_mirror import GridTopologyMirror, mirror_from_monee
from scare.base.util import (
    create_chp_admm_flex_actor,
    create_g2p_admm_flex_actor,
    create_p2g_admm_flex_actor,
    create_p2h_admm_flex_actor,
    get_by_branch_id,
    kgps_to_mw,
    lookup_slack,
    obs_capacity,
    obs_constraint_values,
    obs_setpoint,
    register_priority,
    register_sector,
    register_slack,
    sector_from_grid,
)
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
from scare.service.balance import EnergyBalanceNegotiator, create_energy_balance_role
from scare.service.constraints import GridConstraintMonitor
from scare.service.cp import EnergyConverterRole, MultiCommunityCPRole
from scare.service.cp_priority_admm_role import CPPriorityAdmmRole
from scare.service.dynamic_connector import DynamicConnectorRole
from scare.service.local_generation import LocalGenerationFallbackRole
from scare.service.reconfiguration import GridReconfigurator, GridTieSwitchOperator
from scare.service.slack_budget import SlackBudgetMonitor
from scare.service.stability import GenerationController
from scare.service.voltage_droop import (
    COS_PHI_LARGE,
    COS_PHI_SMALL,
    COS_PHI_THRESHOLD_MVA,
    ReactivePowerDroopRole,
    vde_cos_phi_min,
)

logger = logging.getLogger(__name__)

_SECTORS = [Sector.ELECTRICITY, Sector.GAS, Sector.HEAT]
# Substrings of the grid repr used by ``topology_based_on_sector_grid`` to
# match monee nodes belonging to one sector.  Monee tags grids as
# PowerGrid / GasGrid / WaterGrid, so Sector.value alone ("electricity" /
# "heat") would never match — we have to map to the grid-object name.
_SECTOR_GRID_MATCH: dict[Sector, str] = {
    Sector.ELECTRICITY: "power",
    Sector.GAS: "gas",
    Sector.HEAT: "water",
}

def _branch_sector_str(branch: Any, monee_net: Any) -> str:
    """Sector tag for a physical branch — used by the distributed
    failure-notice propagation in ``ProblemDetector`` to decide which
    edges to traverse and at what cost.

    Returns ``"electricity"`` / ``"gas"`` / ``"heat"`` for same-sector
    pipes/lines, ``"cp"`` for cross-sector coupling plants (CHP, P2G,
    G2P, P2H), or ``""`` for branches whose sector can't be determined
    (defensive fallback — those edges become non-traversable in the
    propagation, which is the conservative choice).
    """
    if branch.model.is_cp():
        return "cp"
    try:
        node = monee_net.node_by_id(branch.id[0])
    except Exception:
        return ""
    sec = sector_from_grid(node.grid)
    return sec.value if sec else ""


def _node_aid(node_id: Any) -> str:
    return f"node-{node_id}"


def _child_aid(child_id: Any) -> str:
    return f"child-{child_id}"


def _model_type_name(branch) -> str:
    return type(branch.model).__name__.lower()


def _is_cp_branch(branch) -> bool:
    return branch.model.is_cp()


def _maybe_register_slack(behavior: Any, aid: str, child: Any) -> None:
    """Register an ``ExtPowerGrid`` / ``ExtHydrGrid`` child as a slack
    agent so the gossip obs helpers report its rated capacity (not the
    LP's current operating point) and classify it as generator-class.

    The rating comes from the Var's bounds: ``p_mw.min`` / ``p_mw.max``
    on ExtPowerGrid, ``mass_flow.min`` / ``mass_flow.max`` on
    ExtHydrGrid.  When a side is unbounded (``None``), the registered
    rating uses the other side's magnitude; when both are unbounded,
    we leave the slack unregistered (the gossip then sees the LP's
    current value, same as today — degenerate but safe).
    """
    m = child.model
    # F3: prefer the operator-level "soft budget" stamped on the model
    # by ``_bound_external_slack`` over the LP Var bounds — the LP
    # bounds are now widened so the energy-flow solve stays feasible,
    # while the soft budget carries the operator's actual target for
    # the MAS to enforce.  Fall back to the Var bounds (treating them
    # as the rating) when no explicit budget was registered.
    budget_attr: float | None = None
    var = None
    if isinstance(m, ExtPowerGrid):
        var = getattr(m, "p_mw", None)
        budget_attr = getattr(m, "_scare_slack_budget_mw", None)
    elif isinstance(m, ExtHydrGrid):
        var = getattr(m, "mass_flow", None)
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
        # Derive a positive rating magnitude from whichever bound is set.
        mags: list[float] = []
        if p_min is not None:
            mags.append(abs(float(p_min)))
        if p_max is not None:
            mags.append(abs(float(p_max)))
        if not mags:
            return
        rating = max(mags)
    register_slack(behavior, aid, rating_mw=rating, p_min=p_min, p_max=p_max)


def _slack_budget_for_child(child: Any) -> tuple[str, float] | None:
    """Return ``(obs_key, budget)`` for a slack-class child carrying an
    operator-policy budget; ``None`` for non-slack children and for
    slack children that ``apply_slack_budget`` left unbudgeted (heat-
    side ExtHydrGrid).

    Used at scenario build time by ``_populate_world`` to decide whether
    to install a ``SlackBudgetMonitor`` role, and by the per-task
    ``slack_meta.json`` writer to surface the budget + LP-envelope to
    downstream plot tooling.
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
        return ("mass_flow", float(budget))
    return None


def _is_power_generator(child: Any) -> bool:
    """True when ``child`` is a monee ``PowerGenerator`` — the simbench
    LV networks we target represent every PV plant as one.  Other
    electrical child types (PowerLoad, ExtPowerGrid) are excluded:
    loads aren't inverter-coupled in this model, and ExtPowerGrid is a
    slack injector whose Q is already a Var the LP solves for.
    """
    return isinstance(child.model, PowerGenerator)


def _is_heat_side_mass_flow_sink(child: Any, monee_net: Any) -> bool:
    """True when ``child`` is a ``Sink`` sitting on a water/heat grid.

    monee's supply-return MES convention represents each heat consumer
    as a (HeatLoad, Sink) pair on adjacent junctions of the supply and
    return pipes: the HeatLoad withdraws thermal energy, the Sink
    withdraws the matching return-line mass flow to close the loop.
    The Sink is a topology artifact, not an independently shedable
    demand — its ``regulation < 1`` breaks junction mass balance and
    presolves the energy-flow LP into infeasibility (see
    ``apply_regulate``'s heat-Sink guard).

    Excluding these from the agent layer (no EnergyBalanceNegotiator,
    no priority registration) keeps the holonic supply-priority ADMM
    from allocating quota to phantom demands.  Gas-sector Sinks model
    real consumption and remain regular load agents.
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
    """True when ``child`` is a coupling-point's *subordinate output*
    rather than an independent device.

    monee's ``CHPHG`` injects its heat through a ``SubHG`` child whose
    ``q_mw_heat`` is a Var pinned by the control-node equation
    (``sub_hg.q_mw_heat == -efficiency_heat · gas_kgps · regulation ·
    3.6·hhv``) — it *follows* the ``CHPHGControlNode`` and is never
    independently controllable.  SCARE controls (and counts) the CHP at
    its control node (the cascade CP, which carries the heat leg), so
    rolling ``SubHG`` as a standalone heat generator would (a) let the
    heat layer issue regulate writes that can't actually move the
    device and (b) double-count the same CHP heat in
    ``supply_by_sector[heat]`` — once via this child and again via the
    control-node CP — inflating the cascade's ``base_supply[heat]``.
    Skip it at agent-build time; the physical heat injection still
    happens through the Var in the energy-flow solve.
    """
    return type(child.model).__name__ == "SubHG"


def _inverter_s_nom_mva(child: Any) -> float | None:
    """Return the inverter's rated apparent power in MVA.

    Preference order:
    1. An explicit ``s_nom_mva`` attribute on the monee model
       (forward-compatible if the importer ever sets it directly).
    2. Reconstruct from the rated active power via the VDE-AR-N 4105
       displacement-factor envelope: ``S_n = |p_n| / cos φ_min`` with
       the size-dependent cos φ_min (0.95 for S_n ≤ 13.8 kVA, else 0.9).
       Two passes — start with the small-inverter cos φ, then re-check
       size — to avoid a self-referential ``s_nom`` definition.
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
    # First pass with the small-inverter cos φ.
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

    Keyed by ``(in_sector_v, out_sector_v)``: ``coupling[(I, O)] = η``
    means feeding 1 MW into sector I produces η MW in sector O.  Used
    by L2.5 cross-sector coalitions to size the cross-sector transfer
    that a CP commitment implies — the values are static priors
    (real-world efficiency curves shift with operating point, but the
    coalition is a coarse commitment that L3 ADMM refines).
    """
    ct = cp_type.lower()
    el = Sector.ELECTRICITY.value
    he = Sector.HEAT.value
    ga = Sector.GAS.value
    if "chp" in ct:
        # CHP burns gas, produces electricity AND heat.
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

    Reads the relevant obs key per sector that the CP can drive.
    Used as an upper bound on the cross-sector transfer the coalition
    can commit to.  Values are intentionally conservative — the L3
    ADMM refines within these bounds.
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
            v = obs.get("mass_flow_capacity") or obs.get("mass_flow")
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
            # Convert kg/s to MW so the coalition can compare across sectors
            cap = kgps_to_mw(cap)
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
) -> SimulationWorld:
    priorities = priorities or {}
    config = config or RestorationConfiguration()

    # Warn when no priority assignment is supplied — obs_priority will
    # then default every load to tier 1, collapsing the 1024× priority
    # spread of the QP/ADMM layers to a uniform baseline.  The HPC
    # runner sets ``priority_assignment: "skewed"`` by default; callers
    # that drive create_restoration_scenario_world directly are easy
    # to miss.  Count loads from the network so we don't fire the
    # warning on degenerate single-load test scenarios.
    if not priorities:
        n_loads = sum(
            1 for child in monee_net.childs
            if obs_capacity(dict(child.model.values)) > 0
        )
        if n_loads > 1:
            logger.warning(
                "No priorities dict supplied for %d loads — obs_priority "
                "will default every load to tier 1, collapsing the priority-"
                "aware machinery (QP waterfall, holon ADMM S-pull, CP "
                "consensus weights, PWSF metric) to a uniform baseline. "
                "Use experiment.restoration.assign_load_priorities() or pass "
                "an explicit priorities= dict.",
                n_loads,
            )

    # with_communication=True installs a Poisson delay provider with mean
    # 20 s/hop, which silently overrides ``static_delay_s`` and makes the
    # ms-scale ``base_delay_ms`` inert.  Keep the static delay active so the
    # configured delay is what actually runs.
    world = create_restoration_world(
        monee_net,
        with_communication=False,
        static_delay_s=base_delay_ms / 1000.0,
    )

    behavior: RestorationEnvironmentBehavior = world.environment.behavior
    # Stash on behavior for components that read flags lazily (e.g.
    # heat_recovery in GridConstraintMonitor) without needing the full
    # config wired through every constructor.
    behavior._scare_config = config

    # Install comms perturbations *before* agents register themselves
    # so the new simulation picks up every send_message that follows.
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

    async with world:
        await discrete_step_until(world, max_advance_time_s=simulation_duration_s)
        # Walk every EnergyBalanceNegotiator and record any still-active
        # gossip as "abandoned" in the diagnostics ledger.  Done before
        # the world context exits so role.context is still valid.
        _flush_pending_negotiations(world)


def _flush_pending_negotiations(world: SimulationWorld) -> None:
    """Drain in-flight gossip state into the negotiation diary so a
    timed-out wallclock or short ``simulation_duration_s`` doesn't leave
    silently-abandoned negotiations missing from the per-event account.
    """
    for agent in world._agents.values():
        for role in getattr(agent, "roles", []):
            if isinstance(role, EnergyBalanceNegotiator):
                role.flush_pending()


def _build_branch_sector_tables(
    monee_net: Any,
) -> tuple[dict[tuple, str], dict[Any, dict[Any, str]]]:
    """Build the per-branch + per-node sector lookup tables that the
    ``ProblemDetector`` needs to decide forward-cost on each grid edge
    without ever consulting the global graph.  Returned tuple is
    ``(branch_sector_by_id, neighbour_sector_by_node)``.
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


def _populate_children(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
    config: RestorationConfiguration,
) -> None:
    for child in monee_net.childs:
        aid = _child_aid(child.id)
        # Read from network model directly — behavior.observe() requires energyflow
        # to have run (initialize()), which only happens when the simulation starts.
        parent_node = monee_net.node_by_id(child.node_id)
        obs = {**dict(parent_node.model.values), **dict(child.model.values)}
        sector = sector_from_grid(parent_node.grid)
        register_sector(behavior, aid, sector)
        # F1: register slack-class children so the gossip observation
        # layer can report their *rated* capacity instead of the LP's
        # current operating point, and treat them as generator-class
        # in the priority waterfall regardless of import / export
        # direction.  See ``register_slack`` for the data model.
        _maybe_register_slack(behavior, aid, child)
        explicit_priority = priorities.get(aid)
        # Slack agents are generator-class (tier 0) regardless of the
        # caller-supplied priorities map — the LP's sign can flip but
        # the role of a slack is always to absorb/supply at the
        # network boundary, never to be shed.  Without this override
        # the construction-time ``obs_priority`` call in
        # ``create_energy_balance_role`` reads the LP's current p_mw
        # sign and may classify the slack as tier 1 (load).
        if explicit_priority is None:
            if lookup_slack(behavior, aid) is not None:
                explicit_priority = 0

        # Heat-side mass-flow Sinks are a monee topology artifact (the
        # return-line partner of a HeatLoad), not an independently
        # curtailable demand — see ``_is_heat_side_mass_flow_sink``.
        # Skipping their agent registration entirely keeps them out of
        # the sector topology / community partition / holon membership,
        # so the dispatcher never tries to curtail them.
        if _is_heat_side_mass_flow_sink(child, monee_net):
            continue

        # A CHP's SubHG heat output follows its control node (the CP);
        # it is not an independent agent — skip so it is neither
        # separately regulated nor double-counted as a heat generator
        # (see ``_is_cp_subordinate_child``).
        if _is_cp_subordinate_child(child):
            continue

        # Register the resolved priority on ``behavior`` so anyone
        # that aggregates across the group (e.g.
        # ``EnergyBalanceNegotiator._handle_ask_flex`` building
        # ``demand_by_sector_priority`` for the tier-stratified holon
        # ADMM) gets the correct per-aid tier instead of falling
        # back to ``obs_priority``'s tier-1-for-all-loads default.
        # Skip slacks since they're already classified via the slack
        # registry.
        if explicit_priority is not None:
            register_priority(behavior, aid, int(explicit_priority))

        roles = []
        if sector is not None:
            roles.append(_make_balance_role(behavior, sector, obs, config, priority=explicit_priority))
            roles.append(GenerationController(behavior, sector))
            # Grid constraint monitoring (voltage / pressure / temperature)
            roles.append(
                GridConstraintMonitor(
                    behavior,
                    sector,
                    node_id=child.node_id,
                    enable_curtailment_auction=config.enable_curtailment_auction,
                    enable_curtail_auction_gating=config.enable_curtail_auction_gating,
                    enable_curtail_auction_targeting=config.enable_curtail_auction_targeting,
                    enable_line_relief_reassert=config.enable_line_relief_reassert,
                    enable_branch_downstream_relief=config.enable_branch_downstream_relief,
                    enable_multihop_constraint=config.enable_multihop_constraint,
                    enable_heat_frontier=config.enable_heat_frontier,
                    enable_heat_priority_waterfall=config.enable_heat_priority_waterfall,
                )
            )
            # Slack-budget enforcement.  Only slack-class children carry
            # the ``_scare_slack_budget_*`` attribute; for non-slack
            # children ``_slack_budget_for_child`` returns None and the
            # monitor is skipped.  Heat-side ExtHydrGrid is intentionally
            # unbudgeted by ``apply_slack_budget`` and falls through
            # naturally (budget = None).
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

        # Every community-member child also receives a passive
        # handler that updates its CommunityAssignment when a
        # leader-driven re-partition lands.  Tiny role — single
        # message subscription, no periodic tasks.
        if sector is not None:
            roles.append(RepartitionHandlerRole())

        # Local Q-V droop at every inverter-coupled PowerGenerator.
        # Follows VDE-AR-N 4105 §5.7.2; lives entirely at the device and
        # composes orthogonally with the gossip/holonic/CP layers above
        # (separate decision variable, separate timescale).  Apparent-
        # power capability circle is derived from the simbench-rated
        # |p_n| and the VDE cos φ_min envelope.
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
                    )
                )

        _register_agent(world, behavior, aid, roles, monee_id=child.id, monee_type="child")


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
        # Resolve the addresses of children sitting on this node so the
        # detector can deliver FailureNotice locally.  Children are
        # registered in the previous loop, so their agents already exist.
        child_addrs: list[Any] = []
        for cid in getattr(node, "child_ids", []):
            child_aid = _child_aid(cid)
            child_agent = world._agents.get(child_aid)
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
                # Node-side install only when the CP is hosted on the
                # node itself (CHP).  For branch-hosted CPs
                # (P2G/G2P/P2H), :func:`_detect_cp_type_for_node`
                # returns the incident-branch type but the actuator
                # lives on the branch agent, which gets its own install
                # below in :func:`_populate_branches`.  Installing the
                # node-side role for those would attach to an agent
                # with no ``regulate`` action — ``apply_regulate`` then
                # silently no-ops every commit.
                if "chp" in cp_type.lower():
                    _attach_cp_priority_admm_role(roles, behavior, aid, cp_type, obs, config)
            elif config.enable_cp_admm:
                _attach_cp_roles(roles, behavior, cp_type, obs, priorities.get(aid, 0))
            elif config.cps_join_communities:
                _attach_multi_community_cp_role(roles, behavior, cp_type, config)

        _register_agent(world, behavior, aid, roles, monee_id=node.id, monee_type="node")


def _populate_branches(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
    config: RestorationConfiguration,
    centrality: dict,
) -> None:
    for branch in monee_net.branches:
        aid = create_branch_aid(branch.id)
        obs = dict(branch.model.values)
        branch_type = _model_type_name(branch)

        roles = []

        if "heatexchanger" in branch_type:
            roles.append(_make_balance_role(behavior, Sector.HEAT, obs, config))
            roles.append(GenerationController(behavior, Sector.HEAT))

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
            cent = get_by_branch_id(centrality, branch.id)
            roles.append(GridTieSwitchOperator(behavior, branch.id, centrality=cent))

        # Line-loading monitor on electricity power lines (whether
        # switchable or not).  home_leader_addr is filled in by
        # ``_assign_line_monitor_home_leaders`` after the groups
        # topology is built.
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
                    enable_curtail_auction_gating=config.enable_curtail_auction_gating,
                    enable_curtail_auction_targeting=config.enable_curtail_auction_targeting,
                    enable_line_relief_reassert=config.enable_line_relief_reassert,
                    enable_branch_downstream_relief=config.enable_branch_downstream_relief,
                    enable_line_relief_waterfall=config.enable_line_relief_waterfall,
                    enable_multihop_constraint=config.enable_multihop_constraint,
                )
            )

        if roles:
            _register_agent(world, behavior, aid, roles, monee_id=branch.id, monee_type="branch")


def _populate_world(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
    config: RestorationConfiguration,
) -> None:
    centrality = edge_centrality(monee_net)
    branch_sector_by_id, neighbour_sector_by_node = _build_branch_sector_tables(monee_net)
    behavior._scare_branch_sector = branch_sector_by_id
    _populate_children(world, monee_net, behavior, priorities, config)
    _populate_nodes(
        world, monee_net, behavior, priorities, config, neighbour_sector_by_node
    )
    _populate_branches(world, monee_net, behavior, priorities, config, centrality)


def _detect_cp_type_for_node(node: Any, monee_net: Any) -> str | None:
    """Return the CP plant type if any branch incident to *node* is one
    of the cross-sector coupling models.  Substrings cover both the
    historical ``chpmodel`` / ``powertogasmodel`` / ``gastopower``
    naming and monee's current ``PowerToGas`` / ``PowerToHeatHG`` /
    ``GasToPower`` / ``Chp`` class names.
    """
    # The node's *own* model type — a CHP is actuated through its
    # ``chphgcontrolnode`` (a node, not a branch), so branch-only
    # detection missed every CHP and left it without a CP role (the
    # gas-consuming lever the cascade's ``Σ r·c[gas] ≤ B_gas`` needs
    # to throttle).  Check the node itself first.
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
    """``create_energy_balance_role`` with the per-scenario gossip flags
    plumbed from *config* in one place.
    """
    return create_energy_balance_role(
        behavior,
        sector,
        obs,
        priority=priority,
        constraint_aware=config.enable_constraint_aware_gossip,
        enable_monotonic_floor=config.enable_monotonic_floor,
        enable_clpu_ramp=config.enable_clpu_ramp,
        termination_tolerance=config.gossip_termination_tolerance,
        max_hops=config.gossip_max_hops,
        enable_qp_gossip=config.enable_qp_gossip,
        enable_l2_priority_floor=config.enable_l2_priority_floor,
        enable_heat_mw_balance=config.enable_heat_mw_balance,
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
    """Register a ``RoleAgent`` with *aid*, attach each role, and bind it
    to its monee object via ``behavior.install``.
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
    """Mutate *roles* in place: append the three CP roles
    (``EnergyConverterRole`` + ``DistributedOptimizationRole`` +
    ``CoordinatorRole``).  No-op when ``_build_cp_flex_actor`` returns
    ``None`` (unknown *cp_type*).  Returns ``None``.
    """
    flex_actor, sectors = _build_cp_flex_actor(cp_type, obs, priority)
    if flex_actor is None:
        return
    roles.append(EnergyConverterRole(behavior, flex_actor, sectors))
    roles.append(DistributedOptimizationRole(flex_actor))
    roles.append(
        CoordinatorRole(create_sharing_target_distance_admm_coordinator())
    )


def _cp_signed_capacity_by_sector(
    cp_type: str, obs: dict
) -> dict[str, float]:
    """Derive load-convention signed effective capacities per sector
    for a CP from its branch / node obs, expressed in MW.

    Mirrors the extraction logic in :mod:`scare.base.admm_factories`
    (which the legacy ``EnergyConverterRole`` consumes through its
    flex actor): each CP has one input sector and one or two output
    sectors; the input capacity is read from the canonical obs key
    (``el_mw`` for electric input, ``gas_kgps`` for gas input,
    converted to MW); the output capacity is the input capacity
    scaled by η (``eta_el`` / ``eta_heat`` / ``eta_gas`` falling back
    to the generic ``efficiency`` field, then to the published
    defaults in :func:`_cp_coupling_ratios`).

    The signed convention is load-convention: positive = the CP
    consumes from that sector, negative = the CP produces into it.
    The kernel's single-knob substitution ``x_i = r_i · c_i`` then
    honours each CP's input/output physics automatically.
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
        """First present, non-zero obs value across *keys*; fallback to
        *default*.  Lets each CP class accept the actual monee model's
        rated-input key (e.g. CHPHG carries ``mass_flow_setpoint``,
        PowerToHeatHG carries ``load_p_mw``) without breaking older
        models that used the canonical generic keys (``gas_kgps``,
        ``el_mw``)."""
        for k in keys:
            v = _f(k, 0.0)
            if v != 0.0:
                return v
        return default

    out: dict[str, float] = {}
    if "chp" in ct:
        # The CHP is actuated through its ``chphgcontrolnode`` agent —
        # monee's single controllability point for a CHPHG (gas in →
        # electricity + heat out all scale with the control node's
        # ``regulation``).  Its obs carries the *actual* gas input rate
        # in ``gas_kgps`` (kg/s) plus ``efficiency_power`` /
        # ``efficiency_heat``.  ``mass_flow_setpoint`` is the equivalent
        # key on the *compound* obs (the control node instead carries a
        # per-unit ``mass_flow`` = 1, which must NOT be used as a rate),
        # so: ``gas_kgps`` first, ``mass_flow_setpoint`` as the
        # compound-obs fallback; never ``mass_flow``.
        #
        # Heat *is* on the CP because the control node is the device's
        # controllability point.  The ``SubHG`` child that physically
        # injects that heat at the heat node is slaved to this control
        # node (``sub_hg.q_mw_heat`` is a Var fixed by the control-node
        # equation) and is therefore *not* an independent agent — it is
        # skipped at agent-build time (see ``_is_cp_subordinate_child``)
        # so the same CHP heat isn't also counted as a standalone heat
        # generator in ``supply_by_sector`` (which would double-count it
        # in the cascade's ``base_supply[heat]``).
        cap_in = kgps_to_mw(abs(_first_nonzero("gas_kgps", "mass_flow_setpoint")))
        if cap_in <= 0:
            return {}
        eta_el = _first_nonzero("efficiency_power", "eta_el", default=0.35)
        eta_he = _first_nonzero("efficiency_heat", "eta_heat", default=0.45)
        out[ga] = cap_in
        out[el] = -cap_in * eta_el
        out[he] = -cap_in * eta_he
    elif "p2g" in ct or "powertogas" in ct:
        cap_in = abs(_first_nonzero("el_mw", "load_p_mw"))
        if cap_in <= 0:
            return {}
        eta = _first_nonzero("eta_gas", "efficiency", default=0.6)
        out[el] = cap_in
        out[ga] = -cap_in * eta
    elif "g2p" in ct or "gastopower" in ct:
        cap_in = kgps_to_mw(abs(_first_nonzero("gas_kgps", "mass_flow_setpoint")))
        if cap_in <= 0:
            return {}
        eta = _first_nonzero("efficiency_power", "eta_el", "efficiency",
                             default=0.45)
        out[ga] = cap_in
        out[el] = -cap_in * eta
    elif "p2h" in ct or "powertoheat" in ct:
        # PowerToHeatHG exposes ``load_p_mw`` (rated el input) and the
        # generic ``efficiency`` — *not* ``el_mw``/``eta_heat``.  Same
        # mis-key story as CHP: the extractor returned ``{}`` and P2H
        # was never wired into the cascade.
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
    """Install :class:`CPPriorityAdmmRole` in place of the legacy
    ``EnergyConverterRole`` / ``DistributedOptimizationRole`` /
    ``CoordinatorRole`` triple.  No-op when the CP's signed capacity
    cannot be derived (unknown *cp_type* or empty capacity dict).

    Replaces the legacy bundle entirely — the replicated kernel is
    the only L3 path under ``enable_cp_priority_admm=True``.  The
    role's cross-sector reachability filter, peer-CP address book,
    and node-id table are injected post-construction by the wire pass
    in ``create_restoration_scenario_world``.
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
        )
    )


def _attach_multi_community_cp_role(
    roles: list,
    behavior: Any,
    cp_type: str,
    config: RestorationConfiguration,
) -> None:
    """``component_level`` counterpart to :func:`_attach_cp_roles`.

    Installs a single :class:`MultiCommunityCPRole` that mirrors the
    sectors the legacy CP-ADMM stack would have covered, without
    spinning up the flex-actor / ADMM coordinator / distributed-
    optimisation roles (none of which run in the baseline).  Sector
    list is derived from *cp_type* the same way
    :func:`_build_cp_flex_actor` does, so the role's per-sector EMA
    tracks exactly the sectors the CP physically bridges.
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

# Number of priority tiers (kept in sync with balance._PRIORITY_TIERS).
# Local copy to keep this helper free of service-layer imports.
_LINE_HOME_PRIORITY_TIERS = 10


def _node_priority_weighted_demand(
    node_id: Any, monee_net: Any, priorities: dict[str, int]
) -> float:
    """Sum of priority-weighted load capacity at a node.

    Used to pick the home group of a PowerLine branch (3a in the plan):
    the line is assigned to the endpoint with the *lower* weighted
    demand, so a future overload-driven shed falls on the less-critical
    side.  Generators (cap < 0) and zero-cap children are skipped.
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
        aid = f"child-{cid}"
        tier = priorities.get(aid, 1)
        weight = 2.0 ** max(0, P - tier)
        total += weight * cap
    return total


def _line_home_endpoint(
    branch: Any, monee_net: Any, priorities: dict[str, int]
) -> Any:
    """Pick a PowerLine branch's home group endpoint (lower priority-
    weighted demand wins).  Ties break to the smaller node id so the
    assignment is deterministic across runs.
    """
    a, b = branch.id[0], branch.id[1]
    pwd_a = _node_priority_weighted_demand(a, monee_net, priorities)
    pwd_b = _node_priority_weighted_demand(b, monee_net, priorities)
    if pwd_a < pwd_b:
        return a
    if pwd_b < pwd_a:
        return b
    return a if (a < b if isinstance(a, int) and isinstance(b, int) else str(a) < str(b)) else b

def _branch_downstream_load_addrs(monee_net: Any, world: Any) -> dict[str, list[Any]]:
    """For every electricity PowerLine branch, the addresses of the loads
    electrically *downstream* of it — the loads on the side that becomes
    disconnected from the slack (``ExtPowerGrid``) when the branch is
    removed, i.e. the subtree whose power physically flows through the
    branch.  Shedding exactly those loads reduces the branch's loading
    ~1:1; the curtailment auction (branch-downstream mode) uses them as its
    bidder set instead of the whole component.

    Branches whose removal does NOT cleanly split a single side off the
    slack (meshed / part of a cycle, or already-disconnected) get an empty
    list and fall back to the legacy endpoint relief.
    """
    from collections import defaultdict, deque

    # node_id -> [load addr]; and the slack node set.
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
        if cap <= 0:  # generators / non-loads can't be shed for relief
            continue
        ag = world._agents.get(f"child-{child.id}")
        if ag is not None:
            node_loads[child.node_id].append(ag.addr)

    # Undirected electricity adjacency keyed by branch aid (so a single
    # branch's edge can be excluded during the cut test).
    adj: dict[Any, set[tuple[Any, str]]] = defaultdict(set)
    endpoints: dict[str, tuple[Any, Any]] = {}
    for branch in monee_net.branches:
        if _is_cp_branch(branch):
            continue
        if _branch_sector_str(branch, monee_net) != "electricity":
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
            # both sides still fed (cycle) or both cut off — no clean subtree.
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
    # --- Per-sector physical topologies ---
    # Each ``sector_grid_<sector>`` topology mirrors the physical adjacency
    # of one energy network.  Label propagation runs on these graphs to
    # carve each sector into bounded sub-communities (Level-1 of the
    # hierarchy described in docs/chapter_method.tex).
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

    # --- PowerLine home-endpoint resolution (3a in the plan) ---
    # Each PowerLine branch joins exactly one of its two endpoint
    # groups.  The chosen endpoint is the one with lower priority-
    # weighted demand so an overload-driven shed falls on the less-
    # critical side.  Built before the groups loop so the augmentation
    # below can attach the branch agent to the correct community.
    powerline_home_node: dict[str, Any] = {}
    powerline_branch_agent: dict[str, Any] = {}
    if config.enable_line_loading_constraint:
        for branch in monee_net.branches:
            b_aid = create_branch_aid(branch.id)
            agent_obj = world._agents.get(b_aid)
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
    # ``communities_from_topology`` runs radius-bounded label propagation
    # on the per-sector physical graph and returns a deterministic
    # partition.  Each partition becomes one topology node holding all
    # member agents — same-node agents are mutual NORMAL neighbours after
    # injection, so the existing gossip protocol works unchanged.
    group_leaders_by_sector: dict[Sector, list] = {}
    # Per-leader member list captured at community-formation time so the
    # ``DynamicRepartitionRole`` has the static member set to compare
    # against post-failure reachability.  ``topology_neighbors`` works
    # at runtime but doesn't give the role construction-time access.
    leader_to_members: dict[Any, list[Any]] = {}
    branch_to_leader: dict[str, Any] = {}
    # Per-sector, per-coalition list of child aids — stashed on
    # ``behavior`` so ``_register_recordings`` can emit per-coalition
    # balance series (validity plots: "is each coalition trending to
    # equilibrium?").  Keyed by sector value + sequential index so the
    # CSV column names stay stable across runs.
    coalition_members_by_sector: dict[str, dict[int, list[str]]] = {
        s.value: {} for s in _SECTORS
    }
    with create_topology(tid="groups") as groups_topo:
        for sector in _SECTORS:
            radius = config.community_label_propagation_radius or _LABEL_PROPAGATION_RADIUS.get(sector, 2)
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
                # Augment electricity communities with PowerLine branch
                # agents whose home endpoint sits in this community
                # (3a single-home: branch joins exactly one group).
                if sector == Sector.ELECTRICITY and powerline_home_node:
                    member_aids = {m.aid for m in members}
                    for b_aid, home_node_id in list(powerline_home_node.items()):
                        try:
                            home_node = monee_net.node_by_id(home_node_id)
                        except Exception:
                            continue
                        home_child_aids = {
                            f"child-{cid}"
                            for cid in getattr(home_node, "child_ids", []) or []
                        }
                        if home_child_aids & member_aids:
                            branch_agent = powerline_branch_agent.get(b_aid)
                            if branch_agent is not None and branch_agent not in members:
                                members.append(branch_agent)
                                # Mark which leader owns this branch so the
                                # monitor's home_leader_addr can be set
                                # once the leader is known (below).
                                branch_to_leader[b_aid] = None
                            # Each branch is single-home — drop it so it
                            # isn't attached to the other endpoint's group
                            # in a future iteration.
                            powerline_home_node.pop(b_aid, None)
                node_id = groups_topo.add_node(*members)
                leader = members[0]
                leader_to_members[leader] = list(members)
                groups_topo.set_characteristic(node_id, leader, "leader")
                community_id = uuid4()
                # Capture the child aids in this coalition (skip branch
                # agents — their ``observe()`` doesn't carry a
                # ``regulation`` key).  The sector key is the loop
                # variable, the integer index is the position of this
                # coalition within its sector list so columns sort
                # deterministically.
                child_member_aids = [
                    m.aid for m in members if m.aid.startswith("child-")
                ]
                coalition_idx = len(coalition_members_by_sector[sector.value])
                coalition_members_by_sector[sector.value][coalition_idx] = child_member_aids
                for member in members:
                    member.add_role(PreAssignedCommunityRole(community_id))
                    # Fill the home_leader pointer for any branches
                    # we just attached to this community.
                    if member.aid in branch_to_leader and branch_to_leader[member.aid] is None:
                        branch_to_leader[member.aid] = leader
                group_leaders_by_sector.setdefault(sector, []).append(leader)
            sizes = sorted(len(c) for c in communities) if communities else []
            try:
                # Recompute the per-node label dict for the diagnostic.
                # The communities_from_topology call lost it, so we
                # invoke the same partitioner once more on the sector
                # graph.  Cheap.
                if config.community_partition_method == "modularity":
                    lbl = modularity_partition(
                        sector_grid_topos[sector].graph,
                        max_iterations=config.community_modularity_iterations,
                        resolution=config.community_modularity_resolution,
                    )
                elif config.community_partition_method == "connected_component":
                    lbl = connected_component_partition(
                        sector_grid_topos[sector].graph
                    )
                else:
                    lbl = label_propagation_partition(
                        sector_grid_topos[sector].graph, max_radius=radius
                    )
                q_score = modularity_of_partition(
                    sector_grid_topos[sector].graph, lbl,
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

    # Patch every branch monitor's home_leader_addr now that the
    # community leaders are known.  Done as a post-pass so the
    # priority-weighted endpoint assignment above can use the leader
    # that was actually chosen for the community, not a guess.
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

    # Branch-downstream relief: give each electricity branch monitor the
    # loads that flow through it (the slack-disconnected subtree when the
    # branch is removed), so a line-overload auction sheds loads that
    # actually relieve THAT line rather than the most-willing load in the
    # component.  Computed once; branches with no clean subtree keep the
    # legacy endpoint relief.
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
                    role._downstream_load_addrs = list(addrs)
                    attached += 1
                    n_loads += len(addrs)
        logger.info(
            "Branch-downstream relief: attached downstream load sets to %d "
            "branch monitors (%d load-targets total)", attached, n_loads,
        )

    # Patch every SlackBudgetMonitor's home_leader_addr the same way.
    # Without this, the monitor's only escalation channel is a local
    # ``BalanceProblem`` event whose ``imbalance`` field gets dropped
    # by ``_on_balance_problem`` — ``trigger_balance_negotiation``
    # recomputes the target from ``_reported_setpoint``, which for a
    # slack returns the *operator target* (``slack_target_fraction ·
    # rating``), not the actual over-budget draw.  Gossip then runs
    # on a near-zero target and sheds nothing.  Routing the over-
    # budget magnitude via ``StartBalanceNegotiation(override_target)``
    # bypasses that path and feeds the real imbalance straight into
    # L1's QP curtailment.  Required for single_level /
    # component_level (no L2 ADMM to do the priority shedding); SCARE
    # also benefits when L2 is slow to react.
    for leader, members in leader_to_members.items():
        for member in members:
            for role in getattr(member, "roles", []):
                if isinstance(role, SlackBudgetMonitor):
                    role.home_leader_addr = leader.addr

    # Grid topology: node agents only, used by GridReconfigurator for path search.
    with create_topology(tid="grid") as grid_topo:
        topology_based_on_grid(
            monee_net,
            grid_topo,
            world,
            include_childs=False,
            include_cps=False,
        )

    # CPs topology: CP agents form clusters; the cluster leader triggers ADMM.
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

    # Build per-leader maps for the dynamic re-partition role.  Each
    # leader needs: its own parent node_id, the per-member parent
    # node_id, the per-member current agent address, and the
    # per-sector branch adjacency.  All cheap to recompute here
    # rather than thread through from ``_populate_world``.
    aid_to_node_id: dict[str, Any] = {
        f"child-{c.id}": c.node_id for c in monee_net.childs
    }
    aid_to_addr: dict[str, Any] = {
        aid: agent.addr for aid, agent in world._agents.items()
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

    # Build the central physical-grid mirror (Concept C).  Constructed
    # here, before the per-leader wiring loop below, so the L2 dynamic-
    # holon role and the post-pass L3 dynamic-connector role can both
    # take a reference.  The mirror is a passive data structure; the
    # ``set_on_branch_failure`` callback later in this function is the
    # only producer of state changes.
    mirror = mirror_from_monee(
        monee_net,
        branch_sector_resolver=lambda b: _branch_sector_str(b, monee_net),
    )
    behavior._scare_topology_mirror = mirror

    # Sector-wide leader-aid → node-id lookup.  The L2 dynamic-holon
    # role needs this for *every* potential same-sector holon peer, not
    # just the leader's own group members; the L3 dynamic-connector
    # role needs it cross-sector for every group leader regardless of
    # sector.  Built once and partitioned by sector below.
    sector_leader_node_ids: dict[Sector, dict[str, Any]] = {}
    for sec, leaders in group_leaders_by_sector.items():
        sector_leader_node_ids[sec] = {
            ldr.aid: aid_to_node_id.get(ldr.aid) for ldr in leaders
            if aid_to_node_id.get(ldr.aid) is not None
        }
    # Cross-sector union for the L3 lookup.
    all_leader_node_ids: dict[str, Any] = {}
    for table in sector_leader_node_ids.values():
        all_leader_node_ids.update(table)

    # Cross-sector coalition CP metadata.  Walked once over the
    # constructed world so HolonSummaryRole gets a ready-made
    # ``{cp_aid: meta}`` map without re-traversing monee_net at role
    # construction time.  Empty when CP ADMM is disabled (no CPs at
    # all) or when cross-sector coalitions are disabled (we still
    # compute it cheaply but the role ignores it).
    # CP metadata table.  Built whenever CP ADMM is enabled (formerly
    # gated on holon_summary too; Option B's multi-sector L3 needs
    # this table independent of L2.5).  Each entry carries enough state
    # for the L3 coord to (a) decide which CPs are in its multi-sector
    # component (``node_id``), (b) build the gradient-step setpoint
    # decision (``rated_capacity_mw``, ``coupling_ratios``), (c)
    # actually dispatch the allocation (``addr``).
    cp_meta_by_aid: dict[str, dict[str, Any]] = {}
    if config.enable_cp_admm or config.enable_cp_priority_admm:
        # Node-hosted CPs (e.g. CHP at a coupling node).
        for node in monee_net.nodes:
            cp_type = _detect_cp_type_for_node(node, monee_net)
            if cp_type is None:
                continue
            aid = f"node-{node.id}"
            agent = world._agents.get(aid)
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
            agent = world._agents.get(aid)
            if agent is None:
                continue
            obs = dict(branch.model.values)
            cp_meta_by_aid[aid] = {
                "sectors": _sectors_for_cp_type(branch_type),
                "coupling_ratios": _cp_coupling_ratios(branch_type),
                "rated_capacity_mw": _cp_rated_capacity_mw(obs, branch_type),
                "addr": agent.addr,
                # Branch CPs span two endpoints — for reachability
                # checks we pick from_node (branch.id[0]).  Either end
                # produces the same active-component answer via the
                # mirror because the branch itself is an edge in that
                # component.
                "node_id": branch.id[0],
            }

    # Set of all CP host node ids — used by HolonicCommunityRole to
    # decide whether to defer its L2 round to L3 (Option B).  For
    # branch CPs we register both endpoints so a leader reachable
    # from either side sees "CP in my multi-sector component".
    cp_node_ids: set[Any] = set()
    if config.enable_cp_admm or config.enable_cp_priority_admm:
        for node in monee_net.nodes:
            if _detect_cp_type_for_node(node, monee_net) is not None:
                cp_node_ids.add(node.id)
        for branch in monee_net.branches:
            if _is_cp_branch(branch):
                cp_node_ids.add(branch.id[0])
                cp_node_ids.add(branch.id[1])

    # Per-sector leader address book used by the cross-sector
    # coalition initiator to reach peer-sector leaders directly.
    peer_leader_addrs: dict[Sector, dict[str, Any]] = {
        sec: {ldr.aid: ldr.addr for ldr in leaders}
        for sec, leaders in group_leaders_by_sector.items()
    }

    # Attach Level-2 / fallback roles to each group leader and mark them
    # as cross-topology connectors for the cps↔groups link.
    # ``LocalGenerationFallbackRole`` always installs (it's the safety
    # net); ``HolonicCommunityRole`` is gated on ``enable_holonic`` so
    # the single-level ablation can run.
    for sector, leaders in group_leaders_by_sector.items():
        for leader in leaders:
            mark_as_connector(leader, connector_type=sector.value)

            # Concept C — L2 dynamic-holon filter.  Constructed before
            # ``HolonicCommunityRole`` so we can pass it via
            # ``live_member_filter`` and the host role consults it at
            # peer-iteration time.  The role only fires when a holon
            # has formed and a branch failure later islands a member,
            # so the construction cost is essentially free when there
            # are no failures.
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

            # One CoalitionConstraintStore per (leader, sector) shared
            # between L2.5 (writer, on every coalition allocation) and
            # L2 (reader, before each supply-priority dispatch).  Only
            # built when M2 coalitions are actually enabled.
            coalition_store = None
            if (
                config.enable_holon_summary
                and config.enable_holon_coalition
            ):
                coalition_store = CoalitionConstraintStore()

            if config.enable_holonic:
                leader.add_role(
                    HolonicCommunityRole(
                        sector,
                        max_holon_size=config.holon_max_size,
                        admm_max_iters=config.holon_admm_max_iters,
                        admm_abs_tol=config.holon_admm_abs_tol,
                        enable_hebbian_formation=config.enable_hebbian_formation,
                        hebbian_warmup_s=config.hebbian_warmup_s,
                        hebbian_threshold=config.hebbian_threshold,
                        enable_tier_stratified_admm=config.enable_tier_stratified_holon_admm,
                        priority_tiers=config.priority_tiers,
                        admm_mode=config.holon_admm_mode,
                        admm_scope=config.holon_admm_scope,
                        enable_priority_allocation=config.enable_priority_holon_allocation,
                        live_member_filter=dyn_holon_role,
                        coalition_constraint_store=coalition_store,
                        my_node_id=leader_node,
                        leader_node_ids=sector_leader_node_ids.get(sector, {}),
                        topology_mirror=mirror,
                        cp_node_ids=cp_node_ids,
                    )
                )
            if dyn_holon_role is not None:
                leader.add_role(dyn_holon_role)
            leader.add_role(LocalGenerationFallbackRole(behavior, sector))

            # Failure-driven dynamic re-partition.  Each leader gets
            # the slice of the per-sector branch table covering its
            # own sector and the parent-node mapping for every member
            # of its (initial) community.  Triggered globally by
            # BranchFailureEvent (wired below in
            # ``_add_system_behaviors``).
            members = leader_to_members.get(leader, [leader])
            member_node_ids = {m.aid: aid_to_node_id[m.aid] for m in members
                               if m.aid in aid_to_node_id}
            member_addrs = {m.aid: aid_to_addr.get(m.aid, m.addr)
                            for m in members}
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

            # Periodic SectorImbalanceUpdate publisher so L3 (CP ADMM)
            # can trigger from a local-imbalance predicate without
            # waiting for an L1 NegotiationFinishedEvent that may never
            # arrive (see scare.base.channel docstring for the
            # specific pathology this fixes).
            if config.enable_cp_admm:
                leader.add_role(SectorImbalanceBeacon(behavior, sector))

            # L2.5 milestone 1: cross-holon priority-inversion
            # detector.  Publishes per-tier served/demand on the
            # sector-wide ``holon_summary_<sector>`` mesh and
            # records ``priority_inversion_detected`` diagnostic
            # events when received summaries show an inversion
            # across holons.  No optimisation runs here yet —
            # milestone 2 will add coalition formation.
            if config.enable_holon_summary:
                # Spatial wiring for deliverability-aware coalition
                # ADMM: the leader's monee node and the per-owned-
                # member node-id map let the role build per-actor
                # reachability via the shared ``mirror`` instance.
                # All three are also passed to DynamicHolonRole below
                # — we deliberately share the mirror so failures
                # propagate consistently across L2/L2.5 roles.
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
                    )
                )

    # Holons topology: partition same-sector group leaders into chunks
    # of ``HolonicCommunityRole.max_holon_size`` and add edges only
    # within each chunk (Level-2 of the hierarchy).  A single full-clique
    # would let only the lex-smallest leader initiate, leaving every
    # other leader orphaned; chunked cliques give one initiator per
    # chunk, so all leaders join exactly one holon.  Skipped entirely
    # when the holonic layer is disabled — the topology stays empty.
    #
    # While we're walking the chunks we also collect the union of
    # coalition memberships per holon so the recording layer can emit a
    # ``holon_balance__<sec>__<idx>`` series (validity plot input).
    holon_members_by_sector: dict[str, dict[int, list[str]]] = {
        s.value: {} for s in _SECTORS
    }
    if config.enable_holonic:
        holon_chunk_size = config.holon_max_size
        with create_topology(tid="holons") as holon_topo:
            for sector, leaders in group_leaders_by_sector.items():
                if len(leaders) < 2:
                    continue
                ordered = sorted(leaders, key=lambda a: a.aid)
                for start in range(0, len(ordered), holon_chunk_size):
                    chunk = ordered[start:start + holon_chunk_size]
                    if len(chunk) < 2:
                        continue
                    chunk_nids = [holon_topo.add_node(member) for member in chunk]
                    for i, nid_a in enumerate(chunk_nids):
                        for nid_b in chunk_nids[i + 1:]:
                            holon_topo.add_edge(nid_a, nid_b)
                    # Union of the chunk leaders' coalition memberships
                    # → child aids that belong to this holon.  Dedup via
                    # set; preserve order in output for stable plots.
                    holon_child_aids: list[str] = []
                    seen_aids: set[str] = set()
                    for leader in chunk:
                        for member in leader_to_members.get(leader, [leader]):
                            aid = getattr(member, "aid", None)
                            if aid and aid.startswith("child-") and aid not in seen_aids:
                                seen_aids.add(aid)
                                holon_child_aids.append(aid)
                    holon_idx = len(holon_members_by_sector[sector.value])
                    holon_members_by_sector[sector.value][holon_idx] = holon_child_aids

    # L2.5 holon-summary mesh (milestone 1).  Per-sector full clique of
    # all group leaders, used as the broadcast channel for post-
    # rebalance ``HolonSummary`` publications.  No optimisation rides
    # on this topology — it's a diagnostic/observability layer that
    # surfaces cross-holon priority inversions to every leader so the
    # coalition-formation logic (milestone 2) can decide whether to
    # convene an ad-hoc cross-chunk ADMM.  Cheap by construction: O(N²)
    # edges per sector, but every message carries only ``2 × n_tiers``
    # floats and the period is in seconds, not milliseconds.
    if config.enable_holonic and config.enable_holon_summary:
        # Collect the CP agents that bridge each sector — they need
        # to be on the per-sector summary mesh to receive HolonSummary
        # under the L3 priority-ADMM cutover (CPPriorityAdmmRole reads
        # leader supply/demand slices from these summaries).  Empty
        # when the cutover flag is off; in that case the topology is
        # leader-only as before.
        cp_agents_by_sector: dict[Sector, list[Any]] = {
            sec: [] for sec in _SECTORS
        }
        if config.enable_cp_priority_admm:
            for aid_, meta in cp_meta_by_aid.items():
                agent = world._agents.get(aid_)
                if agent is None:
                    continue
                for sec in meta.get("sectors", []):
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
                    for nid_b in nids[i + 1:]:
                        t.add_edge(nid_a, nid_b)

    # CP-side topology wiring is needed both when CP ADMM is enabled
    # (the marks + cross-link feed the ``topology_connectors(... tid="cps")``
    # lookup that :class:`EnergyConverterRole` uses to fan flex queries
    # out to leaders) and when the ``component_level`` baseline is
    # active (the same marks + cross-link route per-community
    # NegotiationFinishedEvent deliveries from each leader to the
    # CP-side :class:`MultiCommunityCPRole`).  Skipped only when the
    # CPs are dead weight (no CP role installed on either path).
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
            if b_aid not in world._agents:
                continue
            for sector in _sectors_for_cp_type(_model_type_name(branch)):
                mark_as_connector(world._agents[b_aid], connector_type=sector.value)

        # Mark CP node agents as connectors for the sectors they bridge.
        for node in monee_net.nodes:
            cp_type = _detect_cp_type_for_node(node, monee_net)
            if cp_type is None:
                continue
            n_aid = _node_aid(node.id)
            if n_aid not in world._agents:
                continue
            for sector in _sectors_for_cp_type(cp_type):
                mark_as_connector(world._agents[n_aid], connector_type=sector.value)

        # Link the CP topology to the groups topology for each sector.
        for sector in _SECTORS:
            connect_topologies(cps_topo, groups_topo, sector.value)

    # Concept C — Layer 3 dynamic CP-connector filter.  Constructed
    # *after* the CP topology + connector marks are set up so we can
    # walk the CP agents that were already populated with
    # ``EnergyConverterRole`` in ``_populate_world``.  Post-construction
    # injection (``cp_role._live_connector_filter = dyn_role``) keeps
    # ``_populate_world`` free of mirror dependencies and means the L3
    # role is purely additive — ablating it leaves the CP role's
    # peer-iteration path unchanged.
    if config.enable_cp_admm and config.enable_dynamic_cp_topology:
        for agent in world._agents.values():
            cp_role = None
            for role in getattr(agent, "roles", []):
                if isinstance(role, EnergyConverterRole):
                    cp_role = role
                    break
            if cp_role is None:
                continue
            cp_node_id = aid_to_node_id.get(agent.aid)
            if cp_node_id is None:
                # CP installed on a branch (P2G / G2P / P2H).  Pick the
                # branch's first endpoint as the reference node — it's
                # the same convention the mirror uses for branch
                # endpoints, and reachability is symmetric so either
                # endpoint produces the same connectivity classes.
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
            cp_role._live_connector_filter = dyn_conn

    # Option B — wire the multi-sector L3 state on every CP role.
    # Runs whenever ``enable_cp_admm`` is True, independent of the
    # dynamic-CP-topology flag, because L3 coord election + dispatch
    # are core (not optional) once Option B is the default behaviour
    # of EnergyConverterRole.  Pre-call the role stays in legacy per-CP
    # mode — see :meth:`EnergyConverterRole._multi_sector_l3_enabled`.
    if config.enable_cp_admm:
        for agent in world._agents.values():
            cp_role = None
            for role in getattr(agent, "roles", []):
                if isinstance(role, EnergyConverterRole):
                    cp_role = role
                    break
            if cp_role is None:
                continue
            meta = cp_meta_by_aid.get(agent.aid)
            if meta is None:
                continue
            cp_role.wire_multi_sector_l3(
                topology_mirror=mirror,
                my_node_id=meta["node_id"],
                cp_meta_by_aid=cp_meta_by_aid,
                leader_addrs_by_sector=peer_leader_addrs,
                leader_node_ids=all_leader_node_ids,
            )

    # L3 priority-ADMM cutover — wire each :class:`CPPriorityAdmmRole`
    # with the cross-sector topology mirror and the address books it
    # needs to gossip ``CPSummary`` and to filter peer CPs by
    # reachability.  Mirror-driven filtering replaces the legacy
    # ``DynamicConnectorRole`` for the new path; failures invalidate
    # peer reachability through the same shared mirror update that
    # drives L2 dynamic membership.
    if config.enable_cp_priority_admm:
        peer_cp_addrs = {
            aid_: meta["addr"] for aid_, meta in cp_meta_by_aid.items()
        }
        peer_cp_node_ids = {
            aid_: meta["node_id"] for aid_, meta in cp_meta_by_aid.items()
        }
        for agent in world._agents.values():
            cp_role = None
            for role in getattr(agent, "roles", []):
                if isinstance(role, CPPriorityAdmmRole):
                    cp_role = role
                    break
            if cp_role is None:
                continue
            meta = cp_meta_by_aid.get(agent.aid)
            if meta is None:
                continue
            # Inject home_node_id (set on the role at wire time, not
            # at construction, so the same role object can be tested
            # in isolation).
            cp_role.home_node_id = meta["node_id"]
            # Exclude self from the peer book.
            peers_excl_self = {
                aid_: addr for aid_, addr in peer_cp_addrs.items()
                if aid_ != agent.aid
            }
            cp_role.wire(
                topology_mirror=mirror,
                peer_cp_addrs=peers_excl_self,
                peer_cp_node_ids=peer_cp_node_ids,
            )

    # Keep the grid topology current AND update the mirror: every
    # ``BranchFailureEvent`` marks the edge broken so GridReconfigurator
    # only routes through live edges, and the mirror's reachability
    # queries reflect the same physical truth for the dynamic-topology
    # layer.  Order matters less than people think — both updates are
    # idempotent and the role-side reactions run with a debounce.
    def _on_branch_failed(bid: tuple) -> None:
        mirror.mark_broken(tuple(bid))
        _mark_grid_edge_broken(grid_topo, bid)

    behavior.set_on_branch_failure(_on_branch_failed)

    # Stash the coalition / holon membership maps on ``behavior`` so the
    # recording layer (``_register_recordings``) can subscribe one
    # per-group balance series for each.  Done at the end of topology
    # construction so all sectors have been processed.
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
        # Cross-sector CP branches (PowerToGas / GasToPower /
        # PowerToHeatHG) connect nodes from different sector grids, so
        # the same-sector ``grid`` topology may not contain an edge
        # between the two endpoints even though both nodes individually
        # appear in it.  Guard against the resulting ``KeyError``: a
        # missing CP edge isn't an error, it just means there's no
        # state to mark broken in the per-sector path-search graph.
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
        # Heat sector negotiation is constraint-driven only for setpoint-
        # imbalance triggers: a severed thermal corridor does not show
        # up as a setpoint mismatch (see docs/chapter_method.tex §3.1,
        # heat caveat), so ``CustomFailureEvent`` and ``BranchFailureEvent``
        # are skipped on heat.  ``ConstraintViolation`` IS the canonical
        # heat trigger (BalanceProblem path) and ``ReconfigurationCompletedEvent``
        # needs to re-explore newly reachable corridors — both must fire
        # on heat as well as electricity / gas.
        if role.sector == Sector.HEAT and isinstance(
            event, (CustomFailureEvent, BranchFailureEvent)
        ):
            return
        role.context.schedule_instant_task(
            role.trigger_balance_negotiation()
        )

    def _trigger_cp(role: EnergyConverterRole, event: Any) -> None:
        role.context.schedule_instant_task(role.trigger_cp_negotiation())

    def _trigger_repartition(
        role: DynamicRepartitionRole, event: Any
    ) -> None:
        branch_id = getattr(event, "branch_id", None)
        if branch_id is None:
            return
        # event.branch_id is a tuple; normalize for set hashing.
        role.on_branch_failure(tuple(branch_id))

    # Every failure feeds the leader's per-sector reachability view,
    # regardless of whether the distributed FailureNotice propagation
    # is enabled (the path is independent — repartition needs the
    # branch_id, not the propagated FailureNotice message).
    behavior_in(
        world,
        _trigger_repartition,
        on_global_event=BranchFailureEvent,
        role_types=DynamicRepartitionRole,
    )

    # Concept C — Layer 2 / Layer 3 dynamic-topology triggers.  Same
    # shape as the L1 hook above: the global ``BranchFailureEvent``
    # carries the failed branch id; the role's ``on_branch_failure``
    # callback schedules a debounced reassess against the shared
    # mirror.  Both subscribe unconditionally even when the feature
    # is disabled — the role just isn't installed in that case, so
    # ``behavior_in``'s role-type filter naturally yields no
    # listeners.
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

    # ``component_level`` baseline: reset the per-sector EMA on every
    # MultiCommunityCPRole when a branch fails, so stale signal from a
    # community that may have just been islanded off the CP doesn't
    # bleed into the post-failure decision.  No-op when the role isn't
    # installed (other variants).
    if config.cps_join_communities:
        def _trigger_multi_community_cp(
            role: MultiCommunityCPRole, event: Any
        ) -> None:
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

    # L2.5 milestone 2: coalition constraints are invalidated by any
    # ``BranchFailureEvent`` so the post-failure L2 ADMM round (which
    # is itself triggered by the failure via existing paths) is free
    # to redecide allocations without being overridden by a stale
    # coalition fraction recorded against the pre-failure topology.
    if config.enable_holon_summary and config.enable_holon_coalition:
        def _trigger_coalition_invalidation(
            role: HolonSummaryRole, event: Any
        ) -> None:
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
        # ``CustomFailureEvent`` always keeps the centralised path —
        # those failures don't necessarily correspond to a physical
        # branch and can't be propagated through the grid topology.
        behavior_in(
            world,
            _trigger_balance,
            on_global_event=CustomFailureEvent,
            role_types=EnergyBalanceNegotiator,
        )
        # ``BranchFailureEvent`` goes through the centralised callback
        # only when the distributed ``FailureNotice`` propagation has
        # been disabled (ablation: ``enable_distributed_failure_notice
        # = False``).  Otherwise the per-leader trigger flows through
        # ``ProblemDetector → FailureNotice → EnergyBalanceNegotiator``.
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

    # Constraint violations also trigger rebalancing so the group can
    # shed load or adjust generation to restore feasibility.
    behavior_in(
        world,
        _trigger_balance,
        on_global_event=ConstraintViolation,
        role_types=EnergyBalanceNegotiator,
    )

    # After grid reconfiguration closes tie switches, new resources may
    # become reachable — trigger balance negotiation to exploit them.
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

    # External-grid slack trajectory.  One column per ExtPowerGrid /
    # ExtHydrGrid child, named ``slack__<sector>__<aid>``, carrying the
    # LP-chosen operating point at each tick — ``p_mw`` on ExtPowerGrid
    # (electricity) and ``mass_flow`` on ExtHydrGrid (gas / heat).  Used
    # by the validity overview's slack-trajectory plot to confirm the
    # MAS is driving the slack toward the operator's budgeted infeed
    # rather than letting it absorb everything.
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
            obs_key = "mass_flow"
        else:
            continue
        try:
            node = monee_net.node_by_id(child.node_id)
            sector = sector_from_grid(node.grid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("slack recording: sector lookup failed for %s: %s",
                         child.id, exc)
            continue
        if sector is None:
            continue
        aid = _child_aid(child.id)
        col = f"slack__{sector.value}__{aid}"
        # Closure-captures ``aid`` / ``obs_key`` per child via default
        # args so the lambda doesn't see the loop's last values.
        record_world(world, col, lambda a=aid, k=obs_key: _slack_obs(a, k))

    # Per-coalition (Level-1 group) and per-holon (Level-2 chunk)
    # regulation sums.  Emitted as ``coalition_balance__<sec>__<idx>`` /
    # ``holon_balance__<sec>__<idx>`` columns in ``timeseries.csv`` so
    # the validity plots can show whether each coalition / holon is
    # trending to equilibrium independently — failure-mode for the
    # multi-level design is "one coalition flat-lines while its
    # neighbour still oscillates", which the per-sector aggregate hides.
    # Default-fed empty dicts make these no-ops when the topology
    # builder didn't run (e.g. legacy harness paths).
    coalitions = getattr(behavior, "_scare_coalitions", {}) or {}
    holons = getattr(behavior, "_scare_holons", {}) or {}

    def _make_sum(aids: list[str]):
        # Closure factory — capturing ``aids`` as an explicit argument
        # avoids Python's late-binding gotcha that would otherwise make
        # every recorded lambda see the last loop's aid list.
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

    # Per-(sector, tier) regulation sum — ``tier_balance__<sector>__<tier>``.
    # The per-sector aggregate balance above cannot tell a *correct
    # priority-ordered* low-tier shed (intended: the L3→L2→L1 cascade
    # re-sheds the least-critical tier to free supply) from a genuine
    # regret switch.  Recording the regulation sum per (sector, tier) lets
    # ``_check_monotonic_progress`` flag only the latter: a drop in a
    # higher-priority tier while a lower-priority tier in the SAME sector
    # is still served.  Same-sector so cross-sector independence (e.g. an
    # electricity shed while heat is fully served) is never mis-flagged.
    # Restrict to actual consumer loads (PowerLoad / HeatLoad) — the same
    # population ``served_by_load`` scores.  Summing regulation over *all*
    # sector children would fold in generators / sources / converters /
    # slack, which for the gas sector (no native consumer loads here) is
    # pure noise and would mis-flag a meaningless "gas tier" drop.
    load_aids = {
        _child_aid(c.id) for c in monee_net.childs
        if isinstance(c.model, (PowerLoad, HeatLoad))
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

    # Record constraint-related metrics when the observations expose them.
    # Average is the population mean across the sector's nodes; min/max
    # surface the actual extremes, which are what trigger a violation if
    # they fall outside the operating envelope (the average hides those).
    def _constraint_values(child_aids: list[str], key: str) -> list[float]:
        vals: list[float] = []
        for aid in child_aids:
            obs = behavior.observe(aid)
            if obs and key in obs:
                vals.append(float(obs[key]))
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
        ("vm_pu",       el_child_aids,   "vm_pu"),
        ("pressure_pu", gas_child_aids,  "pressure_pu"),
        ("t_k",         heat_child_aids, "t_k"),
    ):
        record_world(world, f"avg_{prefix}", lambda a=aids, k=key: _avg_constraint(a, k))
        record_world(world, f"min_{prefix}", lambda a=aids, k=key: _min_constraint(a, k))
        record_world(world, f"max_{prefix}", lambda a=aids, k=key: _max_constraint(a, k))

    # --- Line-loading aggregates ---
    # Per-tick rollup of electricity branch loading_percent across every
    # power line in the network.  Three scalars (max / p95 / avg) give
    # the system-state overview a single thermal-stress panel without
    # adding one column per branch.  Uses obs_constraint_values so the
    # fraction-vs-percent quirk in monee's GenericPowerBranch is handled
    # uniformly (see util.obs_constraint_values).
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
        # Linear-interpolated 95th percentile — small list, fine.
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
    # For each priority tier present in the priorities map, emit two
    # columns: ``tier_demand_mw__<tier>`` (static demand sum) and
    # ``tier_served_mw__<tier>`` (sum of clamped served per tick).  The
    # downstream system_state_overview plot derives the fraction from
    # these.  Records both totals so the plot can show absolute MW and
    # served fraction without recomputing in post-processing.

    # Classification runs against ``child.model.values`` rather than
    # ``behavior.observe(aid)`` because the behaviour's first net-results
    # snapshot is only built by the initial solve, which happens after
    # ``_register_recordings`` returns — calling ``observe`` here would
    # AttributeError on a None ``_net_results``.
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

    # --- Net-results freshness tracking ---
    # The mango-energy-environments behavior keeps the previous
    # ``_net_results`` whenever a recompute returns infeasible (via
    # ``_accept_or_keep``), which silently freezes every observation-
    # based metric (avg_vm_pu, *_balance, …) at the last-feasible state.
    # We can't tell from the outside whether the underlying solve
    # succeeded, but we *can* detect when ``_net_results`` actually
    # changed: a new ``SolverResult`` object is constructed on every
    # accepted solve, so ``id(behavior._net_results)`` drifts whenever
    # a recompute went through.  Recording the timestamp of the most
    # recent change as ``last_feasible_solve_t`` lets the downstream
    # plots mask the stale-data segment instead of drawing a misleading
    # flat envelope.  Initial value is the scenario t=0 solve.
    _freshness_state = {
        "id": id(behavior._net_results) if behavior._net_results is not None else None,
        "t": float(world.clock.time),
    }

    def _last_feasible_solve_t() -> float:
        nr = getattr(behavior, "_net_results", None)
        if nr is not None:
            cur = id(nr)
            if cur != _freshness_state["id"]:
                _freshness_state["id"] = cur
                _freshness_state["t"] = float(world.clock.time)
        return _freshness_state["t"]

    record_world(world, "last_feasible_solve_t", _last_feasible_solve_t)

    # --- Emergent metrics ---
    # Track event counts per sector by polling the diagnostics ledger
    # that ``record_event`` and ``record_negotiation`` already feed.
    # The earlier ``behavior_in(on_global_event=...)`` hooks never fired
    # — emits in balance.py / local_generation.py are role-local, and that
    # mango hook only triggers on ``environment.emit_global_event``
    # (audit P0-4).  Reading the ledger keeps the recordings live and
    # avoids needing to plumb global emission paths.
    def _local_gen_request_count(sec_value: str) -> int:
        return sum(
            1 for r in _diag.event_log()
            if r.kind == "local_gen_request" and r.sector == sec_value
        )

    def _negotiations_finished_count(sec_value: str) -> int:
        return sum(
            1 for r in _diag.negotiation_log()
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
