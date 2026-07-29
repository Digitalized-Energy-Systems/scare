"""Per-behavior control-loop blackboard (slack budgets, curtail/relief/congestion
locks, ceilings, feeder voltage, stale-obs) and the ``apply_regulate`` funnel.

Imports obs / constraints one-way (they are leaves), so no import cycle; the
package ``__init__`` re-exports every name below.
"""

from __future__ import annotations

from typing import Any

from monee.model.child import ExtHydrGrid, ExtPowerGrid, Sink
from monee.model.extension import (
    GridFormingGenerator,
    GridFormingSource,
    NetworkIslandingConfig,
)

from scare.base.addressing import is_child_aid
from scare.base.config import cfg_value
from scare.base.model import Sector
from scare.base.runtime.diagnostics import record_event, record_regulate
from scare.base.util.constraints import constraint_allowed_fraction
from scare.base.util.obs import _get_behavior_store


def _slack_eff_budget_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_slack_eff_budget")


def set_slack_eff_budget(behavior: Any, aid: str, value: float) -> None:
    """Record a slack's loss-compensated effective budget (maintained by
    ``SlackBudgetMonitor``) so control targets ``B - losses`` and the actual
    draw lands at operator budget ``B``."""
    _slack_eff_budget_store(behavior)[aid] = float(value)


def lookup_slack_eff_budget(behavior: Any, aid: str) -> float | None:
    return _slack_eff_budget_store(behavior).get(aid)


def _slack_pressure_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_slack_pressure")


def set_slack_pressure(behavior: Any, aid: str, value: float) -> None:
    """Command a gas slack's pressure setpoint via the env ``set_pressure`` action
    (writes ``ExtHydrGrid.pressure_pu``, marks the net dirty; the next solve re-pins
    the slack node). Records the value only when the action fires, so ``lookup``
    never reports an un-applied setpoint; no-op on a child without the action."""
    if behavior.has_action(aid, "set_pressure"):
        behavior.act(aid, "set_pressure", float(value))
        _slack_pressure_store(behavior)[aid] = float(value)


def lookup_slack_pressure(behavior: Any, aid: str) -> float | None:
    """Last commanded slack pressure setpoint [p.u.], or ``None`` if the
    regulator has not actuated this slack yet (its boundary is still the
    ``ExtHydrGrid`` construction default)."""
    return _slack_pressure_store(behavior).get(aid)


def _grid_former_rating_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_grid_former_ratings")


def register_grid_former_rating(behavior: Any, aid: str, rating: float) -> None:
    """Record a promoted island reference (``GridForming*``) and |rated capacity|
    from its Var bound. A former's free p_mw/mass_flow flips ``obs_capacity``'s
    load/generator sign, so registry membership (built from the child model, no
    live ``_net``) is the former-identity signal: the holon credits its DELIVERED
    injection as supply, never phantom demand. Native unit, positive."""
    r = abs(float(rating))
    if r > 0.0:
        _grid_former_rating_store(behavior)[str(aid)] = r


def lookup_grid_former_rating(behavior: Any, aid: str) -> float | None:
    return _grid_former_rating_store(behavior).get(str(aid))


_REGULATE_DEDUP_TOL: float = 1e-3


L2_ALLOCATION_REASONS: frozenset[str] = frozenset(
    {"holon_supply_priority", "holon_tier_alloc", "l2_reassert"}
)


L1_REACTIVE_SHED_REASONS: frozenset[str] = frozenset(
    {"balance", "stability", "tier1_starvation"}
)


CURTAIL_AUCTION_REASON: str = "curtail"


LINE_CONGESTION_REASON: str = "line_congestion"


HEAT_RECOVERY_REASON: str = "heat_recovery"


GEN_RESTORE_REASONS: frozenset[str] = frozenset(
    {"self_local_gen", "local_gen_fallback", "gen_ramp_to_full", "l2_gen_ramp"}
)


def _last_regulate_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_last_regulate")


def note_actuated_factor(behavior: Any, aid: str, factor: float) -> None:
    """Sync the dedup cache with a regulate written outside
    :func:`apply_regulate` (e.g. the gossip path's direct ``act``). Without
    it a later L2 re-dispatch dedups against a stale value and silently
    drops, leaving a gossip-shed load unrestored."""
    _last_regulate_store(behavior)[str(aid)] = float(factor)


def last_actuated_factor(behavior: Any, aid: str) -> float | None:
    """Last regulate factor actuated for *aid* (via :func:`apply_regulate` or
    synced with :func:`note_actuated_factor`); ``None`` before any write."""
    value = _last_regulate_store(behavior).get(str(aid))
    return None if value is None else float(value)


def _l2_floor_store(behavior: Any) -> dict[str, float]:
    """Per-aid served fraction the component-scope holon ADMM last assigned."""
    return _get_behavior_store(behavior, "_scare_l2_floor")


def set_l2_priority_floor(behavior: Any, aid: str, factor: float) -> None:
    """Set the per-aid L2 priority floor directly, with NO actuation. Lets an
    unchanged L2 allocation re-assert the floor (so a fresh/drifted L1 gossip
    still honours the holon's priority decision) without re-dispatching and
    abandoning an in-flight gossip."""
    _l2_floor_store(behavior)[aid] = float(factor)


def _heat_last_sink_floor_store(behavior: Any) -> dict[str, float]:
    """Per-aid serve-fraction floor for a HeatLoad that is the SOLE sink at a
    junction with fixed local heat injection: ``aid -> floor``. Populated at
    build time (``register_heat_last_sink_floor``); enforced in
    :func:`apply_regulate` when ``enable_heat_last_sink_guard`` is on."""
    return _get_behavior_store(behavior, "_scare_heat_last_sink_floor")


def register_heat_last_sink_floor(behavior: Any, aid: str, floor: float) -> None:
    """Register the minimum serve fraction that absorbs the junction's fixed
    local injection (``min(1, injection/|load cap|)``)."""
    _heat_last_sink_floor_store(behavior)[str(aid)] = max(0.0, min(1.0, float(floor)))


def heat_last_sink_floor(behavior: Any, aid: str) -> float | None:
    return _heat_last_sink_floor_store(behavior).get(str(aid))


def _heat_curtail_lock_store(behavior: Any) -> dict[str, float]:
    """Per-aid heat curtailment-auction lock (regulation level held for a
    live temperature violation). An entry means the auction owns the load
    and L2 must defer. Set by ``curtail``, lifted by ``heat_recovery``."""
    return _get_behavior_store(behavior, "_scare_heat_curtail_lock")


def _line_curtail_lock_store(behavior: Any) -> dict[str, tuple]:
    """Per-aid electricity line-relief lock ``aid -> (factor, t_set)``. Set by the
    curtail auction; while fresh (re-armed within ``_LINE_CURTAIL_LOCK_TTL_S`` each
    poll the line is over) L2 restores DEFER, else the holon re-serves a just-shed
    load. Freshness-lifted on clear; electricity analogue of the heat curtail lock."""
    return _get_behavior_store(behavior, "_scare_line_curtail_lock")


_LINE_CURTAIL_LOCK_TTL_S: float = 3.0


def has_line_curtail_lock(behavior: Any, aid: str, now: float) -> bool:
    """True iff *aid* holds a FRESH line-relief lock as of sim-time ``now``."""
    entry = _line_curtail_lock_store(behavior).get(str(aid))
    if entry is None:
        return False
    _factor, t_set = entry
    return (now - float(t_set)) < _LINE_CURTAIL_LOCK_TTL_S


def refresh_line_curtail_lock(behavior: Any, aid: str, now: float) -> None:
    """Re-stamp an EXISTING line-relief lock to ``now`` (keeping its factor)
    so it stays fresh without shedding further. Called by the branch monitor
    every poll the line is over, so the lock survives gaps between curtail
    writes. No-op when no lock entry exists."""
    store = _line_curtail_lock_store(behavior)
    entry = store.get(str(aid))
    if entry is not None:
        factor, _t_set = entry
        store[str(aid)] = (factor, float(now))


def _line_relief_headroom_store(behavior: Any) -> dict[str, tuple]:
    """Per-aid branch loading headroom for the line-relief hand-off:
    ``aid -> (headroom_pct, t_set)``.

    ``headroom_pct = hi - loading_percent`` for the branch whose downstream
    subtree contains this load, published every poll by the line-relief branch
    monitor. The hand-off in :func:`apply_regulate` reads it to decide whether
    the line has room to accept a bounded restore step. Freshness-stamped."""
    return _get_behavior_store(behavior, "_scare_line_relief_headroom")


def publish_line_relief_headroom(
    behavior: Any, aid: str, headroom_pct: float, now: float
) -> None:
    """Record the current branch loading headroom (%-points below the limit)
    available to *aid* for the line-relief restore hand-off."""
    _line_relief_headroom_store(behavior)[str(aid)] = (float(headroom_pct), float(now))


def line_relief_headroom(behavior: Any, aid: str, now: float) -> float | None:
    """Fresh branch loading headroom (%-points below limit) for *aid*, or None
    when none is published / the reading is stale."""
    entry = _line_relief_headroom_store(behavior).get(str(aid))
    if entry is None:
        return None
    headroom, t_set = entry
    if (now - float(t_set)) >= _LINE_CURTAIL_LOCK_TTL_S:
        return None
    return float(headroom)


# A CP's credit must outlive the L2 flex round that reads it but not a failure
# that stops the converter; one rebalance period.
_CP_SUPPLY_TTL_S: float = 5.0


def _cp_supply_store(behavior: Any) -> dict[str, tuple]:
    """Per-CP delivered production for the L2 supply pool:
    ``aid -> ({sector: mw_produced}, t_set)``.

    L2 builds ``supply_by_sector`` by walking its NODE CHILDREN
    (``balance._compute_flex_report``), but every converter is a monee *branch*
    and so has no member aid. On a grid whose carrier is produced only by
    converters (``simbench_lv_gas_dependent`` sets ``gas_gen_share=0``) the pool
    therefore reads zero and the holon sheds 100 % of that carrier's load —
    correctly, given what it was told. This store is how a CP hands its
    committed output back to the leaders that need it. Freshness-stamped."""
    return _get_behavior_store(behavior, "_scare_cp_supply")


def publish_cp_supply(
    behavior: Any,
    aid: str,
    supply_by_leader_sector: dict[str, dict[str, float]],
    now: float,
) -> None:
    """Record a CP's DELIVERED production, split per consuming holon leader.

    Delivered, not rated: mirrors the ``gen_supply = abs(sp)`` convention for
    generator children, so a throttled converter cannot inflate the pool.

    Addressed per leader because the pool is summed across leaders
    (``holon_component.aggregate_holon_flex``) while mango hands every leader of
    a sector the SAME grid-wide connector list. An unaddressed credit would
    therefore be counted once per leader — ~39x for gas on
    ``simbench_lv_gas_dependent``. The CP splits its output across the leaders it
    actually serves, so the sum over leaders is exactly what it produced."""
    _cp_supply_store(behavior)[str(aid)] = (
        {
            str(leader): {
                str(s): float(v) for s, v in by_sector.items() if float(v) > 0.0
            }
            for leader, by_sector in supply_by_leader_sector.items()
        },
        float(now),
    )


def lookup_cp_supply(
    behavior: Any, aid: str, leader_aid: str, now: float
) -> dict[str, float] | None:
    """Fresh per-sector production *aid* attributed to *leader_aid*.

    None when absent or stale. A credit addressed to a different leader is not
    visible here — that is what keeps the cross-leader sum conservative."""
    entry = _cp_supply_store(behavior).get(str(aid))
    if entry is None:
        return None
    by_leader, t_set = entry
    if (now - float(t_set)) >= _CP_SUPPLY_TTL_S:
        return None
    share = by_leader.get(str(leader_aid))
    return dict(share) if share else None


def _line_congestion_store(behavior: Any) -> dict[tuple[str, str], tuple]:
    """Additive congestion price per ``(branch_key, aid) -> (price, t)``,
    ``price = 1 - ceiling``. Keyed by branch so a generator under several congested
    branches SUMS prices instead of one overwriting another. Freshness-stamped;
    stale entries drop from the sum."""
    return _get_behavior_store(behavior, "_scare_line_congestion_price")


def publish_line_congestion_price(
    behavior: Any, branch_key: str, aid: str, price: float, now: float
) -> None:
    """Record *branch_key*'s congestion price (``1 - ceiling``) for generator
    *aid*. ``price <= 0`` clears this branch's entry (the line has headroom)."""
    store = _line_congestion_store(behavior)
    key = (str(branch_key), str(aid))
    if price <= 0.0:
        store.pop(key, None)
    else:
        store[key] = (float(price), float(now))


def line_congestion_ceiling(behavior: Any, aid: str, now: float, ttl: float) -> float:
    """Generation ceiling (max regulation factor) for *aid* = ``1 - Σ fresh
    branch prices``, clamped to ``[0, 1]``. Returns 1.0 (no cap) when no fresh
    price is published."""
    store = _line_congestion_store(behavior)
    total = 0.0
    for (_branch, a), (price, t_set) in list(store.items()):
        if a == str(aid) and (now - float(t_set)) < ttl:
            total += float(price)
    return max(0.0, min(1.0, 1.0 - total))


_CP_HEAT_CEILING_TTL_S: float = 5.0


def _cp_heat_ceiling_store(behavior: Any) -> dict[str, tuple]:
    """Per-CP regulation ceiling ``aid -> (ceiling, t_set)`` from
    ``CPHeatOutletGuard``. Enforced in :func:`apply_regulate` on every
    ``sector="cp"`` write because the L3 kernels re-commit the deficit factor each
    round (delivered heat is measured at load setpoints, which CP injection can't
    raise), so a one-shot wind-down without a held cap is overwritten. Freshness-stamped."""
    return _get_behavior_store(behavior, "_scare_cp_heat_ceiling")


def publish_cp_heat_ceiling(
    behavior: Any, aid: str, ceiling: float, now: float
) -> None:
    """Record the guard's regulation ceiling for CP *aid*. ``ceiling >= 1.0``
    clears the entry (no cap)."""
    store = _cp_heat_ceiling_store(behavior)
    if ceiling >= 1.0:
        store.pop(str(aid), None)
    else:
        store[str(aid)] = (max(0.0, float(ceiling)), float(now))


def cp_heat_ceiling(behavior: Any, aid: str, now: float) -> float | None:
    """Fresh regulation ceiling for CP *aid*, or None when none published or
    the entry is stale (guard released / dead)."""
    entry = _cp_heat_ceiling_store(behavior).get(str(aid))
    if entry is None:
        return None
    ceiling, t_set = entry
    if (now - float(t_set)) >= _CP_HEAT_CEILING_TTL_S:
        return None
    return float(ceiling)


_LINE_RELIEF_HANDOFF_HEADROOM_PCT: float = 8.0


_LINE_RELIEF_RESTORE_STEP: float = 0.1


def _gen_curtail_lock_store(behavior: Any) -> dict[str, float]:
    """Per-aid generator over-voltage curtail-lock: ``aid -> t_set``.

    Set by the auction (``curtail``) when it sheds a generator below full for
    a live node violation (PV over-voltage). While fresh, the local-gen
    RESTORE paths DEFER instead of ramping straight back to full, else the
    auction/restore pair limit-cycles and over-voltage never clears.
    Freshness-lifted; gated on ``enable_curtail_ramp_interlock``."""
    return _get_behavior_store(behavior, "_scare_gen_curtail_lock")


def has_gen_curtail_lock(behavior: Any, aid: str, now: float) -> bool:
    """True iff *aid* holds a FRESH generator over-voltage curtail-lock."""
    t_set = _gen_curtail_lock_store(behavior).get(str(aid))
    if t_set is None:
        return False
    return (now - float(t_set)) < _LINE_CURTAIL_LOCK_TTL_S


def _qv_relief_store(behavior: Any) -> dict[str, tuple]:
    """Per-aid reactive voltage state from the Q(U) droop:
    ``aid -> (t_set, relief_pu, v_pu)``.

    ``relief_pu = (q_max − |q_cmd|) · |dV/dQ|`` is the extra p.u. voltage
    reduction the inverter's unused reactive capability could still provide
    (not yet in ``vm_pu``); ``v_pu`` is the latest local voltage. Read by the
    auction to shed only residual over-voltage and by the gen curtail-lock to
    release active only once reactive holds voltage in-band. Freshness-stamped."""
    return _get_behavior_store(behavior, "_scare_qv_relief")


_QV_RELIEF_TTL_S: float = 2.0


def publish_qv_relief(
    behavior: Any, aid: str, relief_pu: float, now: float, v_pu: float = 0.0
) -> None:
    """Record the reactive voltage-relief and current voltage at *aid*."""
    _qv_relief_store(behavior)[str(aid)] = (
        float(now),
        max(0.0, float(relief_pu)),
        float(v_pu),
    )


def qv_relief_avail(behavior: Any, aid: str, now: float) -> float:
    """Fresh reactive voltage-relief (p.u.) still available at *aid*, or 0.0
    when none is published / the reading is stale."""
    entry = _qv_relief_store(behavior).get(str(aid))
    if entry is None:
        return 0.0
    t_set, relief = entry[0], entry[1]
    if (now - float(t_set)) >= _QV_RELIEF_TTL_S:
        return 0.0
    return max(0.0, float(relief))


def qv_relief_voltage(behavior: Any, aid: str, now: float) -> float | None:
    """Fresh local voltage (p.u.) the droop at *aid* last observed, or None when
    none is published / the reading is stale."""
    entry = _qv_relief_store(behavior).get(str(aid))
    if entry is None or len(entry) < 3:
        return None
    t_set, v_pu = entry[0], entry[2]
    if (now - float(t_set)) >= _QV_RELIEF_TTL_S:
        return None
    return float(v_pu)


_QV_LOCK_RELEASE_MARGIN_PU: float = 1e-3


_QV_LOCK_RELEASE_V_CEILING_PU: float = 1.03


_QV_LOCK_RESTORE_STEP: float = 0.1


def _feeder_voltage_store(behavior: Any) -> dict[str, tuple]:
    """Per-aid electricity node voltage: ``aid -> (t_set, vm_pu)``."""
    return _get_behavior_store(behavior, "_scare_feeder_voltage")


_FEEDER_VOLTAGE_TTL_S: float = 2.0


def publish_node_voltage(behavior: Any, aid: str, vm_pu: float, now: float) -> None:
    """Record this node's latest voltage on the shared feeder ledger."""
    _feeder_voltage_store(behavior)[str(aid)] = (float(now), float(vm_pu))


def feeder_max_voltage(
    behavior: Any, now: float, *, exclude_aid: str | None = None
) -> float | None:
    """Max fresh node voltage (p.u.) published on the feeder, excluding
    ``exclude_aid``; ``None`` when nothing fresh is published."""
    store = _feeder_voltage_store(behavior)
    ex = None if exclude_aid is None else str(exclude_aid)
    mx = None
    for aid, entry in store.items():
        if ex is not None and aid == ex:
            continue
        t_set, v = entry
        if (now - float(t_set)) >= _FEEDER_VOLTAGE_TTL_S:
            continue
        if mx is None or float(v) > mx:
            mx = float(v)
    return mx


def has_heat_curtail_lock(behavior: Any, aid: str) -> bool:
    """True iff *aid* is held by a temperature-driven curtailment lock (vs an
    L2 priority shed, which has no lock). Lets the frontier controller restore
    only loads it shed for temperature, never claw back a priority decision."""
    return str(aid) in _heat_curtail_lock_store(behavior)


def l2_effective_floor(
    behavior: Any,
    aid: str,
    obs: dict,
    sector: Sector,
    tier: int | None,
) -> float | None:
    """The served fraction an L1 reactive shed must not push below:
    ``min(L2 allocation, constraint-allowed fraction)``; ``None`` if unallocated.

    Capping by the constraint fraction makes the floor yield to physical
    shedding, so the floor only blocks balance-driven shedding below the
    priority decision.
    """
    alloc = _l2_floor_store(behavior).get(aid)
    if alloc is None:
        return None
    return min(alloc, constraint_allowed_fraction(obs, sector, tier=tier))


def _last_regulate_t_store(behavior: Any) -> dict[str, float]:
    """Per-aid timestamp of the last applied regulate (sim-time cooldown gate)."""
    return _get_behavior_store(behavior, "_scare_last_regulate_t")


def _stale_obs_state(behavior: Any) -> dict[str, Any]:
    """Per-behavior tracker of regulate-on-stale-obs. ``_net_results`` is replaced
    only on a successful solve, so its ``id()`` is a freshness oracle: a SECOND
    regulate on the same aid against an unchanged id ⇒ acting on stale state.
    Tracked per-aid so a batch of agents dispatched between two solves (each a
    first write on a fresh snapshot) is not mislabelled stale.
    """
    return _get_behavior_store(
        behavior,
        "_scare_stale_obs_state",
        factory=lambda: {
            "last_id": None,
            "applied_aids": set(),
            "stale_landed": 0,
            "warned_for_id": None,
        },
    )


_COOLDOWN_BYPASS_TIER_THRESHOLD: int = 2


def _is_slack_class_child(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a monee ``ExtPowerGrid`` / ``ExtHydrGrid`` slack child.

    Writing ``regulation < 1`` clamps the slack's free Var and the next solve
    goes infeasible once the network needs headroom, so curtail/stability/gossip
    writes must skip slacks. Class-based not registry-based: the unbounded
    heat-side ExtHydrGrid never registers yet is structurally a slack.
    """
    if not is_child_aid(aid):
        return False
    try:
        cid = int(aid[len("child-") :])
    except ValueError:
        return False
    net = getattr(behavior, "_net", None)
    if net is None:
        return False
    try:
        child = net.child_by_id(cid)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(child.model, (ExtPowerGrid, ExtHydrGrid))


def is_grid_former_child(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a promoted island reference (``GridForming*``).

    Like a slack, curtailing it (regulation < 1) collapses its island's balance;
    the ``enable_grid_former_curtail_guard`` write path forces such writes to
    full so the reference stays up.
    """
    if not is_child_aid(aid):
        return False
    try:
        cid = int(aid[len("child-") :])
    except ValueError:
        return False
    net = getattr(behavior, "_net", None)
    if net is None:
        return False
    try:
        child = net.child_by_id(cid)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(child.model, (GridFormingGenerator, GridFormingSource))


def islanding_config_of(monee_net: Any) -> Any:
    """The net's ``NetworkIslandingConfig``, or None.

    ``enable_islanding`` attaches the config twice — as ``net.islanding_config``
    and as an extension — but a net handed back by a solver is a
    ``Network.copy()``, and older monee revisions dropped the attribute in
    ``__deepcopy__`` while keeping the extension. Resolving from the extension
    list (what monee's own ``prepare_solve_network`` does) keeps grading and
    reporting in agreement with the solve whichever net a caller holds; reading
    the attribute alone made the oracle grade an islanded net as un-islanded and
    zero exactly the load the extension had restored.
    """
    cfg = getattr(monee_net, "islanding_config", None)
    if cfg is not None:
        return cfg
    return next(
        (
            e
            for e in getattr(monee_net, "extensions", ()) or ()
            if isinstance(e, NetworkIslandingConfig)
        ),
        None,
    )


def _is_heat_side_mass_flow_sink(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a monee ``Sink`` child on a water/heat junction.

    A heat consumer is a (HeatLoad, Sink) pair; curtailing the Sink's mass flow
    without cutting upstream supply makes the junction mass balance infeasible,
    so thermal curtailment must go through the HeatLoad instead. Gas-sector
    Sinks model real consumption and stay curtailable.
    """
    if not is_child_aid(aid):
        return False
    try:
        cid = int(aid[len("child-") :])
    except ValueError:
        return False
    net = getattr(behavior, "_net", None)
    if net is None:
        return False
    try:
        child = net.child_by_id(cid)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(child.model, Sink):
        return False
    try:
        grid_name = str(getattr(net.node_by_id(child.node_id).grid, "name", "")).lower()
    except Exception:  # noqa: BLE001
        return False
    return "water" in grid_name or "heat" in grid_name


def apply_regulate(
    behavior: Any,
    aid: str,
    factor: float,
    *,
    sector: str,
    reason: str,
    timestamp: float,
    tolerance: float = _REGULATE_DEDUP_TOL,
    priority_tier: int | None = None,
) -> bool:
    """Apply a regulate action, suppressing requests that set the same factor
    (within ``tolerance``) the agent already holds.

    Also enforces a sim-time cooldown when ``cooldown_s > 0`` ("max one solve
    every Δt"); ``priority_tier`` lets critical loads bypass it. Returns True
    if applied, False if suppressed (no act call, no diagnostics).
    """
    factor = max(0.0, min(1.0, factor))

    _cfg = getattr(behavior, "_scare_config", None)

    # --- Grid-former curtailment guard --------------------------------
    # A promoted GridForming* unit is its island's reference; curtailing it
    # (factor < 1) collapses the island balance and the islanding solve goes
    # infeasible. Force any curtailment write on a former back to full so it
    # stays up. Inert unless enabled and the aid is a grid-former.
    if (
        factor < 1.0 - tolerance
        and getattr(_cfg, "enable_grid_former_curtail_guard", False)
        and is_grid_former_child(behavior, aid)
    ):
        record_event(
            t=float(timestamp),
            kind="grid_former_curtail_blocked",
            aid=str(aid),
            sector=str(sector),
            detail=f"reason={reason} requested_factor={factor:.4f} forced=1.0",
        )
        factor = 1.0

    # --- CP heat-outlet ceiling (converter writes only) ----------------
    # Single funnel for every L3 commit path: clamp a CP write to the fresh
    # ceiling the heat-outlet guard holds for an (almost) over-temperature
    # outlet junction, so deficit-driven kernel re-commits cannot undo the
    # guard's wind-down. Inert while no ceiling is published.
    if str(sector) == "cp":
        _cp_ceil = cp_heat_ceiling(behavior, str(aid), float(timestamp))
        if _cp_ceil is not None and factor > _cp_ceil:
            if factor > _cp_ceil + tolerance:
                record_event(
                    t=float(timestamp),
                    kind="cp_regulate_capped_to_heat_ceiling",
                    aid=str(aid),
                    sector=str(sector),
                    detail=(
                        f"reason={reason} requested_factor={factor:.4f} "
                        f"ceiling={_cp_ceil:.4f}"
                    ),
                )
            factor = _cp_ceil

    # --- Heat curtailment-auction lock (heat sector only) -------------
    # While the auction holds a heat load down for a live temperature
    # violation, L2 allocation writes DEFER rather than claw it back up
    # (breaks the cold-day re-dispatch/re-cool cycle). Set by "curtail",
    # lifted as "heat_recovery" ramps back to ~1.0.
    try:
        _sector_e = Sector(sector) if not isinstance(sector, Sector) else sector
    except ValueError:
        _sector_e = None
    # --- Last-sink floor (heat only) ----------------------------------
    # A junction's sole HeatLoad must keep absorbing the co-located fixed
    # injection; below the floor the junction has no cooling draw and runs
    # away thermally (permanent: MW-layer sheds set no lock, so no restore
    # path fires). Clamp any writer's shed up to the floor.
    if _sector_e is Sector.HEAT and cfg_value(_cfg, "enable_heat_last_sink_guard"):
        _floor = heat_last_sink_floor(behavior, str(aid))
        if _floor is not None and factor < _floor - tolerance:
            record_event(
                t=float(timestamp),
                kind="heat_last_sink_floor_clamped",
                aid=str(aid),
                sector=str(sector),
                detail=f"reason={reason} requested_factor={factor:.4f} "
                f"floor={_floor:.4f}",
            )
            factor = _floor

    if _sector_e is Sector.HEAT and getattr(_cfg, "enable_heat_curtail_lock", True):
        _lock = _heat_curtail_lock_store(behavior)
        if reason == CURTAIL_AUCTION_REASON:
            # Lock only when the auction holds the load BELOW full service;
            # a near-1.0 curtail carries no claim, and locking at ~1.0 would
            # wrongly block the holon from shedding the load for MW reasons.
            if factor < 1.0 - tolerance:
                _lock[str(aid)] = factor
            else:
                _lock.pop(str(aid), None)
        elif reason == HEAT_RECOVERY_REASON:
            if factor >= 1.0 - tolerance:
                _lock.pop(str(aid), None)
            else:
                _lock[str(aid)] = factor
        elif reason in L2_ALLOCATION_REASONS and str(aid) in _lock:
            _current = _last_regulate_store(behavior).get(str(aid), _lock[str(aid)])
            if factor > float(_current) + tolerance:
                # A restore: recovery of a temperature shed belongs to the
                # frontier (restores when the region is warm) — L2 must not
                # claw it back early.
                record_event(
                    t=float(timestamp),
                    kind="regulate_deferred_to_curtail_lock",
                    aid=str(aid),
                    sector=str(sector),
                    detail=f"reason={reason} lock={_lock[str(aid)]:.4f} "
                    f"requested_factor={factor:.4f}",
                )
                return False
            # A further shed passes — deepening only helps t_k feasibility.
            # Track the deeper hold; the frontier still restores from the
            # lock once the region warms.
            _lock[str(aid)] = min(_lock[str(aid)], factor)

    # --- Electricity line-relief lock ---------------------------------
    # While the line-relief auction holds a load down for an overloaded line,
    # L2 must DEFER else it re-serves the just-shed load and the line never
    # clears. Freshness-lifted. Gated on the downstream-relief flag.
    if _sector_e is Sector.ELECTRICITY and cfg_value(
        _cfg, "enable_branch_downstream_relief"
    ):
        _lline = _line_curtail_lock_store(behavior)
        if reason == CURTAIL_AUCTION_REASON:
            if factor < 1.0 - tolerance:
                _lline[str(aid)] = (factor, float(timestamp))
            else:
                _lline.pop(str(aid), None)
        elif reason in L2_ALLOCATION_REASONS and has_line_curtail_lock(
            behavior, aid, float(timestamp)
        ):
            # A further shed passes; only a RESTORE is interlocked. Mechanism B:
            # hand back one bounded, headroom-gated step per tick — a one-shot
            # release limit-cycles loading 40–170%.
            _current = _last_regulate_store(behavior).get(str(aid), 0.0)
            if factor > float(_current) + tolerance:  # a restore, not a shed
                _headroom = line_relief_headroom(behavior, aid, float(timestamp))
                if (
                    _headroom is not None
                    and _headroom >= _LINE_RELIEF_HANDOFF_HEADROOM_PCT
                ):
                    factor = min(factor, float(_current) + _LINE_RELIEF_RESTORE_STEP)
                    if factor >= 1.0 - tolerance:
                        _lline.pop(str(aid), None)  # fully restored — drop the lock
                    else:
                        _prev = _lline.get(str(aid))
                        _lline[str(aid)] = (
                            _prev[0] if _prev else factor,
                            float(timestamp),
                        )  # keep fresh; ramp continues next tick
                    record_event(
                        t=float(timestamp),
                        kind="line_curtail_lock_released_to_headroom",
                        aid=str(aid),
                        sector=str(sector),
                        detail=f"reason={reason} stepped_factor={factor:.4f} "
                        f"headroom={_headroom:.2f}",
                    )
                    # fall through: the bounded restore step applies.
                else:
                    record_event(
                        t=float(timestamp),
                        kind="regulate_deferred_to_line_lock",
                        aid=str(aid),
                        sector=str(sector),
                        detail=f"reason={reason} requested_factor={factor:.4f}",
                    )
                    return False
            # else: further shed — fall through to apply it.

    # --- Electricity generator over-voltage curtail-lock --------------
    # Curtail-vs-ramp interlock. When the auction sheds a generator for a live
    # node violation (PV over-voltage), the local-gen RESTORE paths must DEFER
    # rather than ramp straight back to 1.0, else the auction/restore pair
    # limit-cycles and over-voltage never clears. Freshness-lifted.
    if _sector_e is Sector.ELECTRICITY and cfg_value(
        _cfg, "enable_curtail_ramp_interlock"
    ):
        _lgen = _gen_curtail_lock_store(behavior)
        if reason == CURTAIL_AUCTION_REASON:
            if factor < 1.0 - tolerance:
                _lgen[str(aid)] = float(timestamp)
            else:
                _lgen.pop(str(aid), None)
        elif reason in GEN_RESTORE_REASONS and has_gen_curtail_lock(
            behavior, aid, float(timestamp)
        ):
            # Mechanism B: hand active back only when the Q(U) droop holds v
            # in-band (v ≤ ceiling) AND has spare reactive headroom, one bounded
            # step per tick. A one-shot release re-breached over-voltage in
            # validation v1. Saturated / still-elevated droop ⇒ keep deferring.
            _qv_v = qv_relief_voltage(behavior, aid, float(timestamp))
            if (
                getattr(_cfg, "enable_qv_auction_coordination", False)
                and _qv_v is not None
                and _qv_v <= _QV_LOCK_RELEASE_V_CEILING_PU
                and qv_relief_avail(behavior, aid, float(timestamp))
                >= _QV_LOCK_RELEASE_MARGIN_PU
            ):
                current = _last_regulate_store(behavior).get(str(aid), 0.0)
                factor = min(factor, float(current) + _QV_LOCK_RESTORE_STEP)
                if factor >= 1.0 - tolerance:
                    _lgen.pop(str(aid), None)  # fully restored — drop the lock
                else:
                    _lgen[str(aid)] = float(timestamp)  # keep fresh; ramp continues
                record_event(
                    t=float(timestamp),
                    kind="gen_curtail_lock_released_to_qv",
                    aid=str(aid),
                    sector=str(sector),
                    detail=f"reason={reason} stepped_factor={factor:.4f} v={_qv_v:.4f}",
                )
                # fall through: the bounded restore step applies this tick.
            else:
                record_event(
                    t=float(timestamp),
                    kind="regulate_deferred_to_gen_curtail_lock",
                    aid=str(aid),
                    sector=str(sector),
                    detail=f"reason={reason} requested_factor={factor:.4f}",
                )
                return False

    # --- L2 priority-floor reconciliation -----------------------------
    # The holon ADMM is authoritative on which tier is served; L1 must not
    # undo it. Record the floor on L2 writes; clamp L1 reactive sheds UP to it.
    # tier-1-immune ``constraint_allowed_fraction`` re-asserts the tier-1
    # hard-lock. Generators (tier <= 0) excluded.
    if cfg_value(_cfg, "enable_l2_priority_floor"):
        if reason in L2_ALLOCATION_REASONS:
            # Cap the holon allocation by the constraint-allowed fraction: the
            # MW-based ADMM is blind to per-node physics and would otherwise
            # restore an out-of-bounds node to ~1.0 and pin it there.
            try:
                _sector = Sector(sector) if not isinstance(sector, Sector) else sector
            except ValueError:
                _sector = None
            # HEAT exempt — the frontier controller owns its temperature;
            # capping here would re-shed feasible heat loads on transient
            # t_k dips. El/gas keep the cap.
            if (
                _sector is not None
                and _sector is not Sector.HEAT
                and priority_tier is not None
            ):
                _obs = behavior.observe(aid) or {}
                factor = min(
                    factor,
                    constraint_allowed_fraction(_obs, _sector, tier=int(priority_tier)),
                )
            # Load-side construct only (tier >= 1). A generator dispatch (e.g.
            # the R3 ramp, which passes priority_tier=None) must NOT leave a
            # floor: a generator-keyed floor is clamped UP by the L1 consumer,
            # pinning generation high and blocking back-down in reduction rounds.
            if priority_tier is not None and int(priority_tier) >= 1:
                _l2_floor_store(behavior)[aid] = factor
        elif (
            reason in L1_REACTIVE_SHED_REASONS
            and priority_tier is not None
            and int(priority_tier) >= 1
        ):
            try:
                _sector = Sector(sector) if not isinstance(sector, Sector) else sector
            except ValueError:
                _sector = None
            if _sector is not None:
                _obs = behavior.observe(aid) or {}
                _floor = l2_effective_floor(
                    behavior, aid, _obs, _sector, int(priority_tier)
                )
                if _floor is not None and factor < _floor:
                    factor = _floor

    if factor < 1.0 - tolerance and _is_heat_side_mass_flow_sink(behavior, aid):
        record_event(
            t=float(timestamp),
            kind="regulate_blocked_heat_sink",
            aid=str(aid),
            sector=str(sector),
            detail=f"reason={reason} requested_factor={factor:.4f}",
        )
        return False
    if factor < 1.0 - tolerance and _is_slack_class_child(behavior, aid):
        record_event(
            t=float(timestamp),
            kind="regulate_blocked_slack",
            aid=str(aid),
            sector=str(sector),
            detail=f"reason={reason} requested_factor={factor:.4f}",
        )
        return False
    last = _last_regulate_store(behavior).get(aid)
    if last is not None and abs(factor - last) < tolerance:
        return False
    cfg = getattr(behavior, "_scare_config", None)
    cooldown_s = getattr(cfg, "cooldown_s", 0.0) if cfg is not None else 0.0
    if cooldown_s > 0:
        last_t_store = _last_regulate_t_store(behavior)
        last_t = last_t_store.get(aid)
        if last_t is not None and (timestamp - last_t) < cooldown_s:
            critical = (
                priority_tier is not None
                and 0 < int(priority_tier) <= _COOLDOWN_BYPASS_TIER_THRESHOLD
            )
            if not critical:
                record_event(
                    t=float(timestamp),
                    kind="regulate_suppressed_by_cooldown",
                    aid=str(aid),
                    sector=str(sector),
                    detail=(
                        f"reason={reason} factor={factor:.4f} "
                        f"since_last={timestamp - last_t:.3f}s "
                        f"tier={priority_tier}"
                    ),
                )
                return False
    if not behavior.has_action(aid, "regulate"):
        return False

    # Stale-observation detector: a second regulate on THIS aid against an
    # unchanged net_results means it computes against a stale snapshot. Keyed
    # per-aid so batched multi-agent dispatch on one solve isn't flagged.
    state = _stale_obs_state(behavior)
    current_id = id(getattr(behavior, "_net_results", None))
    if state["last_id"] != current_id:
        state["last_id"] = current_id
        state["applied_aids"] = set()
        state["warned_for_id"] = None
    aid_key = str(aid)
    if aid_key in state["applied_aids"]:
        state["stale_landed"] += 1
        if state["warned_for_id"] != current_id:
            record_event(
                t=float(timestamp),
                kind="regulate_on_stale_obs",
                aid=str(aid),
                sector=str(sector),
                detail=(f"reason={reason} stale_landed_total={state['stale_landed']}"),
            )
            state["warned_for_id"] = current_id
    state["applied_aids"].add(aid_key)

    behavior.act(aid, "regulate", factor)
    _last_regulate_store(behavior)[aid] = factor
    _last_regulate_t_store(behavior)[aid] = timestamp

    record_regulate(
        t=timestamp,
        aid=aid,
        sector=sector,
        factor=factor,
        reason=reason,
    )
    return True
