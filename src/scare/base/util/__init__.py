from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from monee.model.child import ExtHydrGrid, ExtPowerGrid, Sink
from monee.model.extension import GridFormingGenerator, GridFormingSource

from scare.base.model import SECTOR_CONSTRAINTS, Sector
from scare.base.runtime.diagnostics import record_event, record_regulate
from scare.base.util.addressing import create_branch_aid
from scare.base.util.blackboard import (
    _COOLDOWN_BYPASS_TIER_THRESHOLD,
    _CP_HEAT_CEILING_TTL_S,
    _FEEDER_VOLTAGE_TTL_S,
    _LINE_CURTAIL_LOCK_TTL_S,
    _LINE_RELIEF_HANDOFF_HEADROOM_PCT,
    _LINE_RELIEF_RESTORE_STEP,
    _QV_LOCK_RELEASE_MARGIN_PU,
    _QV_LOCK_RELEASE_V_CEILING_PU,
    _QV_LOCK_RESTORE_STEP,
    _QV_RELIEF_TTL_S,
    _REGULATE_DEDUP_TOL,
    CURTAIL_AUCTION_REASON,
    GEN_RESTORE_REASONS,
    HEAT_RECOVERY_REASON,
    L1_REACTIVE_SHED_REASONS,
    L2_ALLOCATION_REASONS,
    LINE_CONGESTION_REASON,
    _cp_heat_ceiling_store,
    _feeder_voltage_store,
    _gen_curtail_lock_store,
    _grid_former_rating_store,
    _heat_curtail_lock_store,
    _is_heat_side_mass_flow_sink,
    _is_slack_class_child,
    _l2_floor_store,
    _last_regulate_store,
    _last_regulate_t_store,
    _line_congestion_store,
    _line_curtail_lock_store,
    _line_relief_headroom_store,
    _qv_relief_store,
    _slack_eff_budget_store,
    _slack_pressure_store,
    _stale_obs_state,
    apply_regulate,
    cp_heat_ceiling,
    feeder_max_voltage,
    has_gen_curtail_lock,
    has_heat_curtail_lock,
    has_line_curtail_lock,
    is_grid_former_child,
    l2_effective_floor,
    last_actuated_factor,
    line_congestion_ceiling,
    line_relief_headroom,
    lookup_grid_former_rating,
    lookup_slack_eff_budget,
    lookup_slack_pressure,
    note_actuated_factor,
    publish_cp_heat_ceiling,
    publish_line_congestion_price,
    publish_line_relief_headroom,
    publish_node_voltage,
    publish_qv_relief,
    qv_relief_avail,
    qv_relief_voltage,
    refresh_line_curtail_lock,
    register_grid_former_rating,
    set_l2_priority_floor,
    set_slack_eff_budget,
    set_slack_pressure,
)
from scare.base.util.constraints import (
    _CAP_STATE,
    _CLAMP_DEFAULT_DEADBAND,
    _CLAMP_TIER_DEADBAND,
    _DirectionalCapState,
    clamp_to_constraints,
    constraint_allowed_fraction,
    constraint_utilization,
    set_directional_constraint_cap,
)
from scare.base.util.obs import (
    _CAPACITY_KEYS,
    _CONSTRAINT_OBS_KEYS,
    _UNBOUND_MAX_I_KA,
    _get_behavior_store,
    _obs_branch_loading_percent,
    _priority_store,
    _sector_store,
    _slack_store,
    _SlackMeta,
    lookup_priority,
    lookup_sector,
    lookup_slack,
    obs_capacity,
    obs_constraint_values,
    obs_min_max,
    obs_priority,
    obs_sector,
    obs_setpoint,
    register_priority,
    register_sector,
    register_slack,
)
from scare.base.util.priority import (
    DEFAULT_PRIORITY_TIERS,
    aggregate_priority_weight,
    clamp_tier_monotonic,
    compute_priority_weighted_shares,
    tier_priority_weight,
    tier_priority_weight_strict,
)
from scare.base.util.units import (
    HHV,
    efficiency_vector,
    kgps_to_mw,
    mw_to_kgps,
    sector_color,
    sector_from_grid,
)


def safe_observe(
    behavior: Any,
    aid: str,
    *,
    exc: type[BaseException] | tuple[type[BaseException], ...] = (
        AttributeError,
        KeyError,
    ),
    empty_to_none: bool = False,
) -> dict | None:
    """One guarded ``behavior.observe(aid)``. Swallows only *exc* (returning
    None); anything outside *exc* propagates. With ``empty_to_none`` a falsy obs
    also returns None. Each caller passes its own exception breadth so the
    per-site swallow semantics are preserved."""
    try:
        obs = behavior.observe(aid)
    except exc:
        return None
    if empty_to_none and not obs:
        return None
    return obs


def async_dispatch(role: Any, *, on_receive: Any = None) -> Any:
    """Return the ``coro_fn -> sync callback`` wrapper a role's ``setup`` needs.

    ``context.subscribe_message`` takes a sync callback, so every role hand-rolled
    the same closure. Scheduling the coroutine (rather than awaiting it) is what
    keeps handler work on mango's simulation clock instead of a side track.

    ``on_receive`` runs on the raw meta before dispatch, for roles that record
    the sender.
    """

    def _wrap(coro_fn: Any) -> Any:
        def _sync(msg: Any, meta: Any) -> None:
            if on_receive is not None:
                on_receive(meta)
            role.context.schedule_instant_task(coro_fn(msg, meta))

        return _sync

    return _wrap


def first_role(agent: Any, role_type: type) -> Any:
    """First role of *role_type* on *agent* (side-effect-free read), or None."""
    for role in getattr(agent, "roles", []):
        if isinstance(role, role_type):
            return role
    return None


def role_index(agents: Any, role_type: type) -> dict[str, Any]:
    """``aid -> first role of role_type`` over *agents* that have one, in
    iteration order."""
    idx: dict[str, Any] = {}
    for agent in agents:
        role = first_role(agent, role_type)
        if role is not None:
            idx[agent.aid] = role
    return idx
