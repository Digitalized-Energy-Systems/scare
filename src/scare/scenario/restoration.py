from __future__ import annotations

import logging
from typing import Any

from mango import agent_composed_of, connect_topologies, mark_as_connector
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
)
from mango_energy_environments.environments.restoration.multi_energy_monee import (
    create_branch_aid,
)

from scare.base.model import (
    Sector,
    SystemStrategy,
)
from scare.base.util import (
    create_chp_admm_flex_actor,
    create_g2p_admm_flex_actor,
    create_p2g_admm_flex_actor,
    get_by_branch_id,
    obs_sector,
)
from scare.detection.role import ProblemDetector
from scare.service.balance import EnergyBalanceNegotiator, create_energy_balance_role
from scare.service.cp import EnergyConverterRole
from scare.service.reconfiguration import GridReconfigurator, GridTieSwitchOperator
from scare.service.stability import GenerationController

logger = logging.getLogger(__name__)

_SECTORS = [Sector.ELECTRICITY, Sector.GAS, Sector.HEAT]
_SECTOR_STRINGS = [s.value for s in _SECTORS]

_CP_BRANCH_TYPES = ("powertogasmodel", "gastopower", "chpmodel", "heatexchangermodel")


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
) -> SimulationWorld:
    priorities = priorities or {}

    world = create_restoration_world(
        monee_net,
        with_communication=True,
        static_delay_s=base_delay_ms / 1000.0,
    )

    behavior: RestorationEnvironmentBehavior = world.environment.behavior

    _populate_world(world, monee_net, behavior, priorities)
    _build_topologies(world, monee_net, behavior)
    _add_system_behaviors(world, monee_net, behavior, strategy)
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


def _populate_world(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    priorities: dict[str, int],
) -> None:
    from distributed_resource_optimization import (
        create_sharing_target_distance_admm_coordinator,
    )
    from distributed_resource_optimization.carrier.mango import (
        CoordinatorRole,
        DistributedOptimizationRole,
    )

    centrality = edge_centrality(monee_net)

    for child in monee_net.childs:
        aid = _child_aid(child.id)
        # Read from network model directly — behavior.observe() requires energyflow
        # to have run (initialize()), which only happens when the simulation starts.
        parent_node = monee_net.node_by_id(child.node_id)
        obs = {**dict(parent_node.model.values), **dict(child.model.values)}
        sector = obs_sector(obs)
        explicit_priority = priorities.get(aid)

        roles = []
        if sector is not None:
            roles.append(
                create_energy_balance_role(
                    behavior, sector, obs, priority=explicit_priority
                )
            )
            roles.append(GenerationController(behavior, sector))

        agent = agent_composed_of(*roles, suggested_aid=aid)
        world.register(agent, suggested_aid=aid)
        behavior.install(agent, id=child.id, type="child")

    for node in monee_net.nodes:
        aid = _node_aid(node.id)
        roles: list[Any] = [
            ProblemDetector(behavior, node.id),
            GridReconfigurator(behavior, node.id),
        ]

        obs = dict(node.model.values)
        cp_type = _detect_cp_type_for_node(node, monee_net)
        if cp_type is not None:
            flex_actor, sectors = _build_cp_flex_actor(
                cp_type, obs, priorities.get(aid, 0)
            )
            if flex_actor is not None:
                roles.append(EnergyConverterRole(behavior, flex_actor, sectors))
                roles.append(DistributedOptimizationRole(flex_actor))
                roles.append(
                    CoordinatorRole(create_sharing_target_distance_admm_coordinator())
                )

        agent = agent_composed_of(*roles, suggested_aid=aid)
        world.register(agent, suggested_aid=aid)
        behavior.install(agent, id=node.id, type="node")

    for branch in monee_net.branches:
        aid = create_branch_aid(branch.id)
        obs = dict(branch.model.values)
        branch_type = _model_type_name(branch)

        roles = []

        if "heatexchanger" in branch_type:
            roles.append(create_energy_balance_role(behavior, Sector.HEAT, obs))
            roles.append(GenerationController(behavior, Sector.HEAT))

        elif "powertogasmodel" in branch_type:
            flex_actor, sectors = _build_cp_flex_actor(
                "p2g", obs, priorities.get(aid, 0)
            )
            if flex_actor:
                roles.append(EnergyConverterRole(behavior, flex_actor, sectors))
                roles.append(DistributedOptimizationRole(flex_actor))
                roles.append(
                    CoordinatorRole(create_sharing_target_distance_admm_coordinator())
                )

        elif "gastopower" in branch_type:
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
            agent = agent_composed_of(*roles, suggested_aid=aid)
            world.register(agent, suggested_aid=aid)
            behavior.install(agent, id=branch.id, type="branch")


def _detect_cp_type_for_node(node: Any, monee_net: Any) -> str | None:
    for branch in monee_net.branches_connected_to(node.id):
        t = _model_type_name(branch)
        for cp_type in ("chpmodel", "powertogasmodel", "gastopower"):
            if cp_type in t:
                return cp_type
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


def _build_topologies(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
) -> None:
    # Groups topology: sector-separated fully-connected clusters of child agents.
    # HeatExchanger branches are included so heat-exchanger agents join the heat
    # group and participate in heat-sector gossip negotiations.
    with create_topology(tid="groups") as groups_topo:
        topology_based_on_grid_groups(
            monee_net,
            groups_topo,
            world,
            separate_sectors=_SECTOR_STRINGS,
            include_cps=False,
            include_nodes=False,
            include_childs=True,
            include_branches=["HeatExchanger"],
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

    # Mark group leaders as connectors for their energy sector.
    # topology_based_on_grid_groups assigns "leader" to the first agent in each
    # cluster.  We detect the sector from the EnergyBalanceNegotiator role on
    # the agent — this avoids calling behavior.observe() before the simulation
    # has started (energyflow has not run yet at this point).
    from mango.express.topology import topology_characteristic

    for aid, agent in world._agents.items():
        if topology_characteristic(agent, tid="groups") != "leader":
            continue
        for role in getattr(agent, "roles", []):
            if isinstance(role, EnergyBalanceNegotiator):
                mark_as_connector(agent, connector_type=role.sector.value)
                break

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
    # After this call, topology_connectors(role, tid="cps") on a CP leader
    # returns the group-leader addresses for that sector, and
    # topology_connectors(role, tid="groups") on a group leader returns the
    # CP addresses that bridge its sector.
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
        with modify_topology(grid_topo) as t:
            t.set_edge_state(from_nid, to_nid, State.BROKEN)


def _add_system_behaviors(
    world: SimulationWorld,
    monee_net: Any,
    behavior: RestorationEnvironmentBehavior,
    strategy: SystemStrategy,
) -> None:
    from mango import behavior_in

    def _trigger_balance(role: EnergyBalanceNegotiator, event: Any) -> None:
        role.context.schedule_instant_task(
            role.trigger_balance_negotiation()
        )

    def _trigger_cp(role: EnergyConverterRole, event: Any) -> None:
        role.context.schedule_instant_task(role.trigger_cp_negotiation())

    if strategy in (SystemStrategy.GROUP_TO_CP, SystemStrategy.SIMULTANEOUSLY):
        behavior_in(
            world,
            _trigger_balance,
            on_global_event=BranchFailureEvent,
            role_types=EnergyBalanceNegotiator,
        )
        behavior_in(
            world,
            _trigger_balance,
            on_global_event=CustomFailureEvent,
            role_types=EnergyBalanceNegotiator,
        )

    if strategy in (SystemStrategy.CP_TO_GROUP, SystemStrategy.SIMULTANEOUSLY):
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


def _child_aids_for_sector(monee_net: Any, sector: Sector) -> list[str]:
    aids = []
    for child in monee_net.childs:
        try:
            node = monee_net.node_by_id(child.node_id)
            grid = str(node.grid) if not isinstance(node.grid, str) else node.grid
            if sector.value in grid.lower():
                aids.append(_child_aid(child.id))
        except Exception as exc:
            logger.debug("Could not determine sector for child %s: %s", child.id, exc)
    return aids
