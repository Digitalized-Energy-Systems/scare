from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from mango import RoleAgent, connect_topologies, mark_as_connector
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

from scare.base.community import communities_from_topology
from scare.base.config import RestorationConfiguration
from scare.base.model import (
    ConstraintViolation,
    IslandingRequest,
    NegotiationFinishedEvent,
    ReconfigurationCompletedEvent,
    Sector,
    SystemStrategy,
)
from scare.base.util import (
    create_chp_admm_flex_actor,
    create_g2p_admm_flex_actor,
    create_p2g_admm_flex_actor,
    get_by_branch_id,
    register_sector,
    sector_from_grid,
)
from scare.community.holonic import HolonicCommunityRole
from scare.community.role import PreAssignedCommunityRole
from scare.detection.role import ProblemDetector
from scare.service.balance import EnergyBalanceNegotiator, create_energy_balance_role
from scare.service.constraints import GridConstraintMonitor
from scare.service.cp import EnergyConverterRole
from scare.service.islanding import IslandingFallbackRole
from scare.service.reconfiguration import GridReconfigurator, GridTieSwitchOperator
from scare.service.stability import GenerationController

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

_CP_BRANCH_TYPES = ("powertogasmodel", "gastopower", "chpmodel", "heatexchangermodel")


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
    return branch.model.is_cp() or any(
        t in _model_type_name(branch) for t in _CP_BRANCH_TYPES
    )


def _sectors_for_cp_type(cp_type: str) -> list[Sector]:
    """Return the energy sectors a CP of this type bridges."""
    if "chp" in cp_type:
        return [Sector.ELECTRICITY, Sector.HEAT, Sector.GAS]
    if "p2g" in cp_type or "powertogas" in cp_type:
        return [Sector.ELECTRICITY, Sector.GAS]
    if "g2p" in cp_type or "gastopower" in cp_type:
        return [Sector.ELECTRICITY, Sector.GAS]
    return []


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
    # Agent dropout is a TODO — see ``schedule_agent_dropout`` in
    # ``scare/base/comms.py`` for the contract.  When implemented,
    # call site goes here.
    from scare.base.comms import install_perturbation

    install_perturbation(
        world,
        base_delay_s=base_delay_ms / 1000.0,
        packet_loss_pct=config.comms_packet_loss_pct,
        latency_jitter_ms=config.comms_latency_jitter_ms,
    )

    _populate_world(world, monee_net, behavior, priorities, config)
    _build_topologies(world, monee_net, behavior, config)
    _add_system_behaviors(world, monee_net, behavior, strategy, config)
    _register_recordings(world, monee_net, behavior)

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
    from scare.service.balance import EnergyBalanceNegotiator

    for agent in world._agents.values():
        for role in getattr(agent, "roles", []):
            if isinstance(role, EnergyBalanceNegotiator):
                role.flush_pending()


def _populate_world(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
    config: RestorationConfiguration,
) -> None:
    from distributed_resource_optimization import (
        create_sharing_target_distance_admm_coordinator,
    )
    from distributed_resource_optimization.carrier.mango import CoordinatorRole

    from scare.base.admm import (
        ScareDistributedOptimizationRole as DistributedOptimizationRole,
    )


    centrality = edge_centrality(monee_net)

    # --- Build distributed-propagation lookup tables ---------------------
    # ``ProblemDetector`` needs three pieces of locally-acquirable state:
    #   - the failing branch's sector (read once at endpoint detection)
    #   - the sector of each grid edge leaving its node (decides forward
    #     cost in ``_propagate``)
    #   - the addresses of children co-located on its node (where the
    #     notice is delivered for negotiator triggering)
    # Computed once at scenario time so each detector receives only what
    # it locally needs — no detector ever queries the global graph.
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
    behavior._scare_branch_sector = branch_sector_by_id

    for child in monee_net.childs:
        aid = _child_aid(child.id)
        # Read from network model directly — behavior.observe() requires energyflow
        # to have run (initialize()), which only happens when the simulation starts.
        parent_node = monee_net.node_by_id(child.node_id)
        obs = {**dict(parent_node.model.values), **dict(child.model.values)}
        sector = sector_from_grid(parent_node.grid)
        register_sector(behavior, aid, sector)
        explicit_priority = priorities.get(aid)

        roles = []
        if sector is not None:
            roles.append(
                create_energy_balance_role(
                    behavior,
                    sector,
                    obs,
                    priority=explicit_priority,
                    constraint_aware=config.enable_constraint_aware_gossip,
                    enable_monotonic_floor=config.enable_monotonic_floor,
                    enable_clpu_ramp=config.enable_clpu_ramp,
                    termination_tolerance=config.gossip_termination_tolerance,
                    max_hops=config.gossip_max_hops,
                    enable_qp_gossip=config.enable_qp_gossip,
                )
            )
            roles.append(GenerationController(behavior, sector))
            # Grid constraint monitoring (voltage / pressure / temperature)
            roles.append(
                GridConstraintMonitor(
                    behavior,
                    sector,
                    node_id=child.node_id,
                    enable_curtailment_auction=config.enable_curtailment_auction,
                    enable_multihop_constraint=config.enable_multihop_constraint,
                    enable_heat_recovery=config.enable_heat_recovery,
                )
            )

        agent = world.register(RoleAgent(), suggested_aid=aid)
        for role in roles:
            agent.add_role(role)
        behavior.install(agent, id=child.id, type="child")

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
            GridReconfigurator(behavior, node.id),
        ]

        obs = dict(node.model.values)
        cp_type = _detect_cp_type_for_node(node, monee_net)
        if cp_type is not None and config.enable_cp_admm:
            flex_actor, sectors = _build_cp_flex_actor(
                cp_type, obs, priorities.get(aid, 0)
            )
            if flex_actor is not None:
                roles.append(EnergyConverterRole(behavior, flex_actor, sectors))
                roles.append(DistributedOptimizationRole(flex_actor))
                roles.append(
                    CoordinatorRole(create_sharing_target_distance_admm_coordinator())
                )

        agent = world.register(RoleAgent(), suggested_aid=aid)
        for role in roles:
            agent.add_role(role)
        behavior.install(agent, id=node.id, type="node")

    for branch in monee_net.branches:
        aid = create_branch_aid(branch.id)
        obs = dict(branch.model.values)
        branch_type = _model_type_name(branch)

        roles = []

        if "heatexchanger" in branch_type:
            roles.append(
                create_energy_balance_role(
                    behavior,
                    Sector.HEAT,
                    obs,
                    constraint_aware=config.enable_constraint_aware_gossip,
                    enable_monotonic_floor=config.enable_monotonic_floor,
                    enable_clpu_ramp=config.enable_clpu_ramp,
                    termination_tolerance=config.gossip_termination_tolerance,
                    max_hops=config.gossip_max_hops,
                    enable_qp_gossip=config.enable_qp_gossip,
                )
            )
            roles.append(GenerationController(behavior, Sector.HEAT))

        elif "powertogasmodel" in branch_type and config.enable_cp_admm:
            flex_actor, sectors = _build_cp_flex_actor(
                "p2g", obs, priorities.get(aid, 0)
            )
            if flex_actor:
                roles.append(EnergyConverterRole(behavior, flex_actor, sectors))
                roles.append(DistributedOptimizationRole(flex_actor))
                roles.append(
                    CoordinatorRole(create_sharing_target_distance_admm_coordinator())
                )

        elif "gastopower" in branch_type and config.enable_cp_admm:
            flex_actor, sectors = _build_cp_flex_actor(
                "g2p", obs, priorities.get(aid, 0)
            )
            if flex_actor:
                roles.append(EnergyConverterRole(behavior, flex_actor, sectors))
                roles.append(DistributedOptimizationRole(flex_actor))
                roles.append(
                    CoordinatorRole(create_sharing_target_distance_admm_coordinator())
                )

        elif hasattr(branch.model, "on_off"):
            cent = get_by_branch_id(centrality, branch.id)
            roles.append(GridTieSwitchOperator(behavior, branch.id, centrality=cent))

        if roles:
            agent = world.register(RoleAgent(), suggested_aid=aid)
            for role in roles:
                agent.add_role(role)
            behavior.install(agent, id=branch.id, type="branch")


def _detect_cp_type_for_node(node: Any, monee_net: Any) -> str | None:
    """Return the CP plant type if any branch incident to *node* is one
    of the cross-sector coupling models.  Substrings cover both the
    historical ``chpmodel`` / ``powertogasmodel`` / ``gastopower``
    naming and monee's current ``PowerToGas`` / ``PowerToHeatHG`` /
    ``GasToPower`` / ``Chp`` class names.
    """
    for branch in monee_net.branches_connected_to(node.id):
        t = _model_type_name(branch)
        if any(s in t for s in ("chp", "powertogas", "gastopower", "powertoheat")):
            return t
    return None


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
    return None, []


_LABEL_PROPAGATION_RADIUS: dict[Sector, int] = {
    Sector.ELECTRICITY: 2,
    Sector.GAS: 2,
    Sector.HEAT: 2,
}

def _build_topologies(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    config: RestorationConfiguration,
) -> None:
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

    # --- Groups topology: one cluster per sub-community ---
    # ``communities_from_topology`` runs radius-bounded label propagation
    # on the per-sector physical graph and returns a deterministic
    # partition.  Each partition becomes one topology node holding all
    # member agents — same-node agents are mutual NORMAL neighbours after
    # injection, so the existing gossip protocol works unchanged.
    group_leaders_by_sector: dict[Sector, list] = {}
    with create_topology(tid="groups") as groups_topo:
        for sector in _SECTORS:
            radius = _LABEL_PROPAGATION_RADIUS.get(sector, 2)
            communities = communities_from_topology(
                sector_grid_topos[sector], max_radius=radius
            )
            for members in communities:
                if not members:
                    continue
                node_id = groups_topo.add_node(*members)
                leader = members[0]
                groups_topo.set_characteristic(node_id, leader, "leader")
                community_id = uuid4()
                for member in members:
                    member.add_role(PreAssignedCommunityRole(community_id))
                group_leaders_by_sector.setdefault(sector, []).append(leader)
            logger.info(
                "Sector %s: label propagation produced %d communities",
                sector.value,
                len(communities),
            )

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

    # Attach Level-2 / fallback roles to each group leader and mark them
    # as cross-topology connectors for the cps↔groups link.  ``IslandingFallbackRole``
    # always installs (it's the safety net); ``HolonicCommunityRole`` is
    # gated on ``enable_holonic`` so the single-level ablation can run.
    for sector, leaders in group_leaders_by_sector.items():
        for leader in leaders:
            mark_as_connector(leader, connector_type=sector.value)
            if config.enable_holonic:
                leader.add_role(
                    HolonicCommunityRole(sector, max_holon_size=config.holon_max_size)
                )
            leader.add_role(IslandingFallbackRole(behavior, sector))

    # Holons topology: partition same-sector group leaders into chunks
    # of ``HolonicCommunityRole.max_holon_size`` and add edges only
    # within each chunk (Level-2 of the hierarchy).  A single full-clique
    # would let only the lex-smallest leader initiate, leaving every
    # other leader orphaned; chunked cliques give one initiator per
    # chunk, so all leaders join exactly one holon.  Skipped entirely
    # when the holonic layer is disabled — the topology stays empty.
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

    # CP-side topology wiring is only needed when CP ADMM is enabled —
    # the connector marks + topology link drive the
    # ``topology_connectors(... tid="cps")`` lookup that ``EnergyConverterRole``
    # uses; with no CP role installed the marks would be dead weight.
    if config.enable_cp_admm:
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

    # Keep the grid topology current: mark edges BROKEN when a branch fails so
    # GridReconfigurator only routes through live edges.
    behavior.set_on_branch_failure(
        lambda bid: _mark_grid_edge_broken(grid_topo, bid)
    )


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
    from mango import behavior_in

    def _trigger_balance(role: EnergyBalanceNegotiator, event: Any) -> None:
        # Heat sector negotiation is constraint-driven only.  Setpoint
        # imbalance does not capture the temperature-deficit problem
        # that arises from a severed thermal corridor (see
        # docs/chapter_method.tex §3.1, heat caveat).  Heat groups
        # negotiate via BalanceProblem ← ConstraintViolation instead.
        if role.sector == Sector.HEAT:
            return
        role.context.schedule_instant_task(
            role.trigger_balance_negotiation()
        )

    def _trigger_cp(role: EnergyConverterRole, event: Any) -> None:
        role.context.schedule_instant_task(role.trigger_cp_negotiation())

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
) -> None:
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

    record_agent_having(
        world,
        "regulation",
        EnergyBalanceNegotiator,
        lambda agent: float((behavior.observe(agent.aid) or {}).get("regulation", 0.0)),
    )

    # Record constraint-related metrics when the observations expose them.
    def _avg_constraint(child_aids: list[str], key: str) -> float:
        vals = []
        for aid in child_aids:
            obs = behavior.observe(aid)
            if obs and key in obs:
                vals.append(float(obs[key]))
        return sum(vals) / len(vals) if vals else 0.0

    record_world(
        world,
        "avg_vm_pu",
        lambda: _avg_constraint(el_child_aids, "vm_pu"),
    )
    record_world(
        world,
        "avg_pressure_pu",
        lambda: _avg_constraint(gas_child_aids, "pressure_pu"),
    )
    record_world(
        world,
        "avg_t_k",
        lambda: _avg_constraint(heat_child_aids, "t_k"),
    )

    # --- Emergent metrics ---
    # Track event counts per sector so post-hoc analysis can judge
    # convergence quality (how many rounds until quiescence) and
    # fallback activation rate (unresolved deficits triggering
    # islanding).  The counters are plain dicts mutated by event
    # callbacks registered via ``behavior_in``.
    _metric_state = {
        "islanding": {s.value: 0 for s in _SECTORS},
        "negotiations": {s.value: 0 for s in _SECTORS},
        "last_event_time": {s.value: 0.0 for s in _SECTORS},
    }

    from mango import behavior_in

    def _on_islanding(_role, event):
        _metric_state["islanding"][event.sector.value] += 1
        _metric_state["last_event_time"][event.sector.value] = (
            world.clock.time
        )

    def _on_negotiation_finished(_role, event):
        _metric_state["negotiations"][event.sector.value] += 1
        _metric_state["last_event_time"][event.sector.value] = (
            world.clock.time
        )

    behavior_in(
        world,
        _on_islanding,
        on_global_event=IslandingRequest,
        role_types=EnergyBalanceNegotiator,
    )
    behavior_in(
        world,
        _on_negotiation_finished,
        on_global_event=NegotiationFinishedEvent,
        role_types=EnergyBalanceNegotiator,
    )

    for sector in _SECTORS:
        s = sector.value
        record_world(
            world,
            f"islanding_requests_{s}",
            lambda s=s: _metric_state["islanding"][s],
        )
        record_world(
            world,
            f"negotiations_finished_{s}",
            lambda s=s: _metric_state["negotiations"][s],
        )
        record_world(
            world,
            f"time_since_last_event_{s}",
            lambda s=s: max(
                0.0,
                world.clock.time - _metric_state["last_event_time"][s],
            ),
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
