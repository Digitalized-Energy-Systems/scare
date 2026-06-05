from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from monee.model.child import ExtHydrGrid, ExtPowerGrid, Sink

from scare.base.diagnostics import record_event, record_regulate
from scare.base.model import SECTOR_CONSTRAINTS, Sector

# Higher heating value of natural gas. MW/(kg/s) conversion factor is
# 3.6*HHV (1 kWh/s = 3.6 MW); do NOT read HHV itself as MW/(kg/s).
HHV: float = 15.3  # kWh/kg

_CAPACITY_KEYS = (
    "p_mw",
    "q_mw_heat",       # heat load capacity [MW]
    "q_mw_set",        # heat exchanger setpoint [MW]
    "q_mw",            # heat branch actual power [MW]
    "mass_flow",
    "p_kw",
    "q_mvar",
    "p_mw_capacity",
    "mass_flow_capacity",
)


def mw_to_kgps(value: float) -> float:
    return value / (3.6 * HHV)


def kgps_to_mw(value: float) -> float:
    return value * 3.6 * HHV


def obs_capacity(
    obs: dict,
    *,
    behavior: Any = None,
    aid: str | None = None,
) -> float:
    """Return the rated capacity for this agent's child.

    For load/generator/Sink/Source children the rated value lives
    directly in ``obs``. For slack children (``ExtPowerGrid`` /
    ``ExtHydrGrid``) the obs key carries the LP's *current* operating
    point, not the rating, so return the registered rating instead when
    the slack registry resolves (see ``register_slack``).
    """
    if behavior is not None and aid is not None:
        slack = lookup_slack(behavior, aid)
        if slack is not None:
            return slack.cap
    for key in _CAPACITY_KEYS:
        if key in obs:
            return float(obs[key])
    return 0.0


def obs_setpoint(
    obs: dict,
    *,
    behavior: Any = None,
    aid: str | None = None,
) -> float:
    """Return the current dispatched power (load convention).

    Non-slack children: ``setpoint = capacity * regulation``. Slack
    children have no regulation knob; the dispatched value is the
    LP-chosen value in ``obs``.
    """
    if behavior is not None and aid is not None:
        slack = lookup_slack(behavior, aid)
        if slack is not None:
            # Slack: the LP-chosen operating point is in the obs key.
            for key in _CAPACITY_KEYS:
                if key in obs:
                    return float(obs[key])
            return 0.0
    return obs_capacity(obs) * float(obs.get("regulation", 1.0))


def obs_min_max(
    obs: dict,
    *,
    behavior: Any = None,
    aid: str | None = None,
) -> tuple[float, float]:
    """Return (delta_min, delta_max) relative to current setpoint.

    For slack children the δ-range is the full Var bound range minus the
    current value (headroom for both import and export). Other children
    stay in ``[-sp, cap-sp]`` / ``[cap-sp, -sp]``.
    """
    if behavior is not None and aid is not None:
        slack = lookup_slack(behavior, aid)
        if slack is not None:
            sp = obs_setpoint(obs, behavior=behavior, aid=aid)
            return (slack.dmin_abs - sp, slack.dmax_abs - sp)
    cap = obs_capacity(obs)
    sp = obs_setpoint(obs)
    if cap < 0:
        return (cap - sp, -sp)
    else:
        return (-sp, cap - sp)


def sector_from_grid(grid: Any) -> Sector | None:
    """Resolve a Sector from a monee grid object via its .name attribute.

    Returns None for multi-grid nodes (e.g. CHPControlNode): they
    straddle sectors and the sector must be chosen explicitly by context.
    """
    if grid is None or isinstance(grid, (list, tuple)):
        return None
    name = str(getattr(grid, "name", "")).lower()
    if "power" in name:
        return Sector.ELECTRICITY
    if "gas" in name:
        return Sector.GAS
    if "water" in name or "heat" in name:
        return Sector.HEAT
    return None


def _get_behavior_store(behavior: Any, attr: str, factory=dict) -> Any:
    """Lazy ``getattr(behavior, attr) or factory()`` accessor for the
    per-behavior registries. Storing on the behavior ties registry
    lifetime to the simulation world.
    """
    store = getattr(behavior, attr, None)
    if store is None:
        store = factory()
        setattr(behavior, attr, store)
    return store


def _sector_store(behavior: Any) -> dict[str, Sector]:
    return _get_behavior_store(behavior, "_scare_sectors")


def register_sector(behavior: Any, aid: str, sector: Sector | None) -> None:
    if sector is not None:
        _sector_store(behavior)[aid] = sector


def lookup_sector(behavior: Any, aid: str) -> Sector | None:
    return _sector_store(behavior).get(aid)


# ---------------------------------------------------------------------------
# Slack-agent metadata
# ---------------------------------------------------------------------------
#
# Slack children (ExtPowerGrid / ExtHydrGrid) carry their rated capacity
# in the ``p_mw`` / ``mass_flow`` Var bounds, which are not in the runtime
# obs dict (only the current value is). Without this registry the gossip
# negotiator would read a slack's "capacity" as the LP's current value
# (and treat an importing slack as a load). The registry holds the rated
# capacity + δ-range so ``obs_capacity`` / ``obs_min_max`` / ``obs_priority``
# return physically meaningful values for slacks.

@dataclass(frozen=True)
class _SlackMeta:
    """Cached slack rating + δ-range for one ExtPowerGrid / ExtHydrGrid child.

    ``cap`` follows monee's load convention; a slack is always a source
    from the local network's view, so ``cap < 0`` (generator-priority).
    ``dmin_abs`` / ``dmax_abs`` are the absolute Var bounds; relative
    deltas are derived in ``obs_min_max``.

    Units are the slack's native sector unit — MW for an ExtPowerGrid
    (``p_mw``), kg/s for an ExtHydrGrid gas slack (``mass_flow``). Values
    are produced and consumed within one sector and are NOT MW-normalised;
    a consumer pooling a gas slack with MW quantities must ``kgps_to_mw`` first.
    """
    cap: float          # rated output, < 0 (generator convention, native unit)
    dmin_abs: float     # min absolute Var value (p_mw / mass_flow)
    dmax_abs: float     # max absolute Var value (p_mw / mass_flow)


def _slack_store(behavior: Any) -> dict[str, "_SlackMeta"]:
    return _get_behavior_store(behavior, "_scare_slacks")


def register_slack(
    behavior: Any,
    aid: str,
    *,
    rating_mw: float,
    p_min: float | None = None,
    p_max: float | None = None,
) -> None:
    """Register a slack-class agent's rating.

    ``rating_mw`` is the positive magnitude of rated transformer /
    pipeline capacity. ``p_min`` / ``p_max`` are the ``p_mw`` Var bounds
    (load convention: negative = export, positive = import); if both None
    the slack is bidirectional at ``[-rating_mw, +rating_mw]``.

    Despite the name, the value is stored in the slack's native sector
    unit — for an ExtHydrGrid gas slack callers pass kg/s (the
    ``mass_flow`` budget), not MW. Gas consumers treat ``cap`` as kg/s;
    code crossing gas into shared-MW space must ``kgps_to_mw`` it.
    """
    if rating_mw <= 0.0:
        # A non-positive rating would leave the slack unregistered, so
        # obs_capacity/obs_priority fall back to the LP's current value
        # and reclassify the slack as a load. Surface the bad input.
        logging.getLogger(__name__).warning(
            "register_slack(%s, rating_mw=%s): non-positive rating; "
            "slack will fall back to LP-value capacity, which is "
            "rarely what callers want.",
            aid, rating_mw,
        )
        return
    if p_min is None:
        p_min = -float(rating_mw)
    if p_max is None:
        p_max = +float(rating_mw)
    _slack_store(behavior)[aid] = _SlackMeta(
        cap=-float(rating_mw),  # generator-class sign
        dmin_abs=float(p_min),
        dmax_abs=float(p_max),
    )


def lookup_slack(behavior: Any, aid: str) -> "_SlackMeta | None":
    return _slack_store(behavior).get(aid)


def _slack_eff_budget_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_slack_eff_budget")


def set_slack_eff_budget(behavior: Any, aid: str, value: float) -> None:
    """Record a slack's effective budget — the loss-compensated cap the
    supply pool advertises, maintained by ``SlackBudgetMonitor``'s integral
    feedback. Used in place of nominal ``|cap|`` so control targets
    ``B - losses`` and the slack's actual draw lands at operator budget ``B``."""
    _slack_eff_budget_store(behavior)[aid] = float(value)


def lookup_slack_eff_budget(behavior: Any, aid: str) -> float | None:
    return _slack_eff_budget_store(behavior).get(aid)


def _priority_store(behavior: Any) -> dict[str, int]:
    return _get_behavior_store(behavior, "_scare_priorities")


def register_priority(behavior: Any, aid: str, tier: int) -> None:
    """Record an agent's priority tier on the behavior so callers that
    don't own the role (e.g. aggregating across all group members) can
    look it up.

    Without this registry, ``obs_priority`` falls back to uniform
    priorities and every per-tier feature degenerates to a single-tier
    baseline. Stored values are ints >= 0; tier 0 is reserved for
    generator-class agents and slacks.
    """
    _priority_store(behavior)[aid] = int(tier)


def lookup_priority(behavior: Any, aid: str) -> int | None:
    return _priority_store(behavior).get(aid)


# ---------------------------------------------------------------------------
# Regulate-action de-duplication
# ---------------------------------------------------------------------------

# Tolerance below which re-applying the same regulation factor is a no-op.
# 1e-3 (0.1% of capacity) is below the precision of any monitored
# constraint, so sub-promille re-applies don't churn monee state.
_REGULATE_DEDUP_TOL: float = 1e-3


# Regulate reasons carrying the L2 holon's authoritative per-tier
# allocation — these SET the load's L2 floor. L1 reactive sheds are
# clamped UP to that floor so a supply-poor local group can't undo a
# served-tier decision the component ADMM just made.
L2_ALLOCATION_REASONS: frozenset[str] = frozenset(
    {"holon_supply_priority", "holon_tier_alloc"}
)
L1_REACTIVE_SHED_REASONS: frozenset[str] = frozenset({"balance", "stability"})

# Reason written by the community curtailment auction. Acts as the heat-only
# L2 defer signal: while a heat load holds an auction curtailment for a live
# violation, L2 allocation writes defer to it (see ``apply_regulate``).
CURTAIL_AUCTION_REASON: str = "curtail"
# Reason written by the heat un-shed recovery loop; lifts the heat curtail
# lock as it ramps a recovered load back toward full service.
HEAT_RECOVERY_REASON: str = "heat_recovery"


def _last_regulate_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_last_regulate")


def note_actuated_factor(behavior: Any, aid: str, factor: float) -> None:
    """Sync the per-aid dedup cache with a regulate actuation written
    outside :func:`apply_regulate`.

    The gossip path writes ``behavior.act("regulate", …)`` directly,
    bypassing the dedup; the cache then keeps the last ``apply_regulate``
    value, not the gossip's write. A later L2 re-dispatch would then dedup
    against the stale cache and silently drop, leaving a gossip-shed load
    unrestored. Call this after every direct write to keep the cache truthful.
    """
    _last_regulate_store(behavior)[str(aid)] = float(factor)


def _l2_floor_store(behavior: Any) -> dict[str, float]:
    """Per-aid L2 priority allocation: the served fraction the
    component-scope holon ADMM most recently assigned to this load."""
    return _get_behavior_store(behavior, "_scare_l2_floor")


def _heat_curtail_lock_store(behavior: Any) -> dict[str, float]:
    """Per-aid heat curtailment-auction lock: the regulation level the
    auction holds a heat load at for a live temperature violation. An entry
    means the auction owns this load and L2 writes must defer. Set by
    ``reason="curtail"``, lifted by ``heat_recovery`` ramp-up (see
    :func:`apply_regulate`)."""
    return _get_behavior_store(behavior, "_scare_heat_curtail_lock")


def _line_curtail_lock_store(behavior: Any) -> dict[str, tuple]:
    """Per-aid electricity line-relief lock: ``aid -> (factor, t_set)``.

    Set by the branch-downstream line-relief auction (``reason="curtail"``).
    While an entry is fresh (re-asserted within ``_LINE_CURTAIL_LOCK_TTL_S``,
    which the auction does every poll the line is over), L2 holon writes to
    that load DEFER, else the holon re-serves a just-shed load and the line
    never clears. Freshness-lifted (no explicit release): once the line drops
    <=100% the auction stops re-arming and the entry goes stale. Electricity
    analogue of the heat curtail lock."""
    return _get_behavior_store(behavior, "_scare_line_curtail_lock")


# How long a line-relief lock stays authoritative after its last refresh.
# Must exceed the electricity monitor poll (~0.5 s) so it survives between
# re-arms, but be short enough to release within a second or two of the
# line clearing.
_LINE_CURTAIL_LOCK_TTL_S: float = 3.0


def has_line_curtail_lock(behavior: Any, aid: str, now: float) -> bool:
    """True iff *aid* holds a FRESH line-relief lock as of sim-time ``now``."""
    entry = _line_curtail_lock_store(behavior).get(str(aid))
    if entry is None:
        return False
    _factor, t_set = entry
    return (now - float(t_set)) < _LINE_CURTAIL_LOCK_TTL_S


def refresh_line_curtail_lock(behavior: Any, aid: str, now: float) -> None:
    """Re-stamp an EXISTING line-relief lock entry to ``now`` (keeping its
    held factor) so it stays fresh, without shedding the load further.

    The branch monitor calls this every poll while the line is over (or in
    the release hysteresis band) so the lock survives gaps between the
    auction's curtail writes — otherwise the lock ages out mid-relief and
    L2 claws the loads back up. No-op for loads with no lock entry."""
    store = _line_curtail_lock_store(behavior)
    entry = store.get(str(aid))
    if entry is not None:
        factor, _t_set = entry
        store[str(aid)] = (factor, float(now))


def has_heat_curtail_lock(behavior: Any, aid: str) -> bool:
    """True iff *aid* is held by a temperature-driven curtailment lock —
    the auction or heat frontier controller shed it (``reason="curtail"``),
    as opposed to an L2 priority shed (no lock). Lets the frontier
    controller restore only loads it shed for temperature, never claw back
    a priority decision."""
    return str(aid) in _heat_curtail_lock_store(behavior)


def l2_effective_floor(
    behavior: Any,
    aid: str,
    obs: dict,
    sector: Sector,
    tier: int | None,
) -> float | None:
    """The served fraction an L1 reactive shed must not push below:
    ``min(L2 allocation, constraint-allowed fraction)``.

    Returns ``None`` when the holon has not yet allocated to this load.
    Capping by the constraint-allowed fraction means the floor yields
    continuously to the physical shedding the local constraint requires,
    so curtailment/clamp own the violation window while the floor only
    blocks balance-driven shedding below the priority decision.
    """
    alloc = _l2_floor_store(behavior).get(aid)
    if alloc is None:
        return None
    return min(alloc, constraint_allowed_fraction(obs, sector, tier=tier))


def _last_regulate_t_store(behavior: Any) -> dict[str, float]:
    """Per-aid timestamp of the last applied regulate (sim-time cooldown gate)."""
    return _get_behavior_store(behavior, "_scare_last_regulate_t")


def _stale_obs_state(behavior: Any) -> dict[str, Any]:
    """Per-behavior tracker of regulate-on-stale-observation events.

    ``behavior._net_results`` is replaced only on a successful solve (an
    infeasible ``energyflow`` solve re-uses the previous result), so its
    ``id()`` is a cheap freshness oracle. This tracks the last-seen id;
    if unchanged and an apply has already landed on it, the new regulate
    is acting on stale state and is counted / surfaced once via a
    ``regulate_on_stale_obs`` event.
    """
    return _get_behavior_store(
        behavior,
        "_scare_stale_obs_state",
        factory=lambda: {
            "last_id": None,
            "applies_on_current_id": 0,
            "stale_landed": 0,
            "warned_for_id": None,
        },
    )


# Tiers at or below this threshold bypass the global cooldown: a critical
# dispatch must not be dropped just because it arrives within cooldown_s of
# an unrelated update. Lower tiers still pay the cooldown.
_COOLDOWN_BYPASS_TIER_THRESHOLD: int = 2


def _is_slack_class_child(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a monee ``ExtPowerGrid`` / ``ExtHydrGrid`` child —
    the network's slack-class boundary.

    Slacks have a free p_mw / mass_flow Var; writing ``regulation < 1``
    clamps the slack to a fraction of its envelope and the next solve goes
    infeasible the moment the network needs more headroom. Curtailment /
    stability / gossip writes must skip slacks.

    Class-based rather than registry-based: the heat-side ExtHydrGrid is
    intentionally unbounded (no operator slack discipline) and never lands
    in the ``register_slack`` registry, yet is still structurally a slack.
    """
    if not aid.startswith("child-"):
        return False
    try:
        cid = int(aid[len("child-"):])
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


def _is_heat_side_mass_flow_sink(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a monee ``Sink`` child on a water/heat junction.

    Heat consumers are modelled as a (HeatLoad, Sink) pair sharing a
    junction: HeatLoad withdraws thermal energy (``q_mw_heat``), Sink
    withdraws the matching return-line mass flow. Forcing
    ``Sink.regulation < 1`` zeroes the mass-flow withdrawal without zeroing
    upstream supply, making the junction mass balance infeasible. Thermal
    curtailment must instead go through the HeatLoad (``q_mw_heat *
    regulation``), which leaves mass flow untouched. Gas-sector Sinks model
    real gas consumption and stay curtailable.
    """
    if not aid.startswith("child-"):
        return False
    try:
        cid = int(aid[len("child-"):])
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
        grid_name = str(
            getattr(net.node_by_id(child.node_id).grid, "name", "")
        ).lower()
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
    """Apply a regulate action, suppressing requests that would set the
    same factor (within ``tolerance``) the agent already holds.

    Also enforces a sim-time cooldown when ``cooldown_s > 0``: same-aid
    writes within ``cooldown_s`` of the previous applied write are
    suppressed regardless of factor delta ("max one solve every Δt").
    ``priority_tier`` (when set) lets critical loads bypass the cooldown
    gate — see ``_COOLDOWN_BYPASS_TIER_THRESHOLD``.

    Returns ``True`` if applied, ``False`` if suppressed (no behavior.act
    call, no diagnostics record).
    """
    factor = max(0.0, min(1.0, factor))

    _cfg = getattr(behavior, "_scare_config", None)

    # --- Heat curtailment-auction lock (heat sector only) -------------
    # While the auction holds a heat load down for a live temperature
    # violation it is the authoritative shedding lever: L2 allocation writes
    # DEFER rather than claw the load back up. Breaks the cold-day limit
    # cycle where MW-based holon re-dispatch restores a just-curtailed cold
    # node and re-cools it. Set by auction ("curtail") writes, lifted as
    # ``heat_recovery`` ramps the load back to ~1.0. Heat-scoped: other
    # sectors and unlocked heat loads fall through to the L2 path below.
    try:
        _sector_e = Sector(sector) if not isinstance(sector, Sector) else sector
    except ValueError:
        _sector_e = None
    if _sector_e is Sector.HEAT and getattr(
        _cfg, "enable_heat_curtail_lock", True
    ):
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
            # Auction owns this load — L2 must not correct it.
            record_event(
                t=float(timestamp),
                kind="regulate_deferred_to_curtail_lock",
                aid=str(aid),
                sector=str(sector),
                detail=f"reason={reason} lock={_lock[str(aid)]:.4f} "
                       f"requested_factor={factor:.4f}",
            )
            return False

    # --- Electricity line-relief lock ---------------------------------
    # While the line-relief auction holds an electricity load down to relieve
    # an overloaded line, L2 must DEFER, else it re-serves the just-shed load
    # every cycle and the line never clears. Freshness-lifted: the auction
    # re-asserts every poll the line is over, so a stale entry stops
    # deferring. Gated on the downstream-relief flag and electricity sector.
    if _sector_e is Sector.ELECTRICITY and getattr(
        _cfg, "enable_branch_downstream_relief", False
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
            record_event(
                t=float(timestamp),
                kind="regulate_deferred_to_line_lock",
                aid=str(aid),
                sector=str(sector),
                detail=f"reason={reason} requested_factor={factor:.4f}",
            )
            return False

    # --- L2 priority-floor reconciliation -----------------------------
    # The component-scope holon ADMM is authoritative on which tier gets
    # served; L1 must not undo it. Record the floor on L2 writes; clamp L1
    # reactive sheds (here ``stability``; gossip ``balance`` writes bypass
    # this and are floored in ``_apply_setpoint``). Applies to all tiers:
    # ``constraint_allowed_fraction`` is tier-1-immune (1.0), so tier-1's
    # floor is its L2 allocation, re-asserting the tier-1 hard-lock against
    # stability erosion while the curtailment auction can still shed tier-1
    # when a constraint demands it. Generators (tier <= 0) excluded.
    if getattr(_cfg, "enable_l2_priority_floor", False):
        if reason in L2_ALLOCATION_REASONS:
            # Cap the holon allocation (applied factor and stored floor) by
            # the load's constraint-allowed fraction. The MW-based L2 ADMM is
            # blind to per-node physics; without the cap a holon write
            # restores an out-of-bounds node to ~1.0 and the floor pins it
            # there. ``constraint_allowed_fraction`` is tier-1-immune (1.0),
            # matching ``l2_effective_floor``'s read-time cap, so the stored
            # floor is never above feasibility regardless of caller.
            try:
                _sector = (
                    Sector(sector) if not isinstance(sector, Sector) else sector
                )
            except ValueError:
                _sector = None
            # HEAT is exempt — the frontier controller owns its temperature
            # (and locks managed loads, so this write already defers); capping
            # here would re-shed feasible heat loads on transient t_k dips.
            # El/gas keep the cap.
            if (
                _sector is not None
                and _sector is not Sector.HEAT
                and priority_tier is not None
            ):
                _obs = behavior.observe(aid) or {}
                factor = min(
                    factor,
                    constraint_allowed_fraction(
                        _obs, _sector, tier=int(priority_tier)
                    ),
                )
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

    # Stale-observation detector: if the LP has not re-solved since the
    # previous apply, this regulate computes against a stale net_results
    # snapshot (otherwise hidden by the LP infeasibility cascade).
    state = _stale_obs_state(behavior)
    current_id = id(getattr(behavior, "_net_results", None))
    if state["last_id"] == current_id and state["applies_on_current_id"] > 0:
        state["stale_landed"] += 1
        if state["warned_for_id"] != current_id:
            record_event(
                t=float(timestamp),
                kind="regulate_on_stale_obs",
                aid=str(aid),
                sector=str(sector),
                detail=(
                    f"reason={reason} stale_landed_total={state['stale_landed']}"
                ),
            )
            state["warned_for_id"] = current_id
    elif state["last_id"] != current_id:
        state["last_id"] = current_id
        state["applies_on_current_id"] = 0
        state["warned_for_id"] = None
    state["applies_on_current_id"] += 1

    behavior.act(aid, "regulate", factor)
    _last_regulate_store(behavior)[aid] = factor
    if cooldown_s > 0:
        _last_regulate_t_store(behavior)[aid] = timestamp

    record_regulate(
        t=timestamp,
        aid=aid,
        sector=sector,
        factor=factor,
        reason=reason,
    )
    return True


def obs_sector(
    obs: dict,
    *,
    behavior: Any = None,
    aid: str | None = None,
) -> Sector | None:
    """Resolve the energy sector an observation belongs to.

    Prefers the (behavior, aid) sector registry. The obs-key heuristic
    is a last-resort fallback only: monee junction obs dicts are
    shape-identical between gas and water, so key inference is unreliable.
    """
    if behavior is not None and aid is not None:
        found = lookup_sector(behavior, aid)
        if found is not None:
            return found
    if "p_mw" in obs or "p_kw" in obs or "p_mw_capacity" in obs:
        return Sector.ELECTRICITY
    if "q_mw_heat" in obs or "q_mw_set" in obs or "q_mw" in obs:
        return Sector.HEAT
    if "q_mvar" in obs and "p_mw" not in obs:
        return Sector.HEAT
    return None


def create_branch_aid(branch_id: tuple) -> str:
    a, b = branch_id[0], branch_id[1]
    hi, lo = (a, b) if a > b else (b, a)
    return f"branch-{hi}-{lo}"


def get_by_branch_id(centrality: dict, branch_id: tuple) -> float:
    if branch_id in centrality:
        return centrality[branch_id]
    rev = (branch_id[1], branch_id[0]) + branch_id[2:]
    return centrality.get(rev, 0.0)


# Re-export for backwards compatibility with callers importing it here.
from scare.base.failure_sampling import create_failures  # noqa: E402,F401


def efficiency_vector(eta_el: float, eta_heat: float, eta_gas: float) -> np.ndarray:
    return np.array([eta_el, eta_heat, eta_gas], dtype=float)


# Re-export for backwards compatibility with callers importing them here.
from scare.base.admm_factories import (  # noqa: E402,F401
    create_chp_admm_flex_actor,
    create_g2p_admm_flex_actor,
    create_p2g_admm_flex_actor,
    create_p2h_admm_flex_actor,
)


def sector_color(sector: Sector) -> str:
    return {Sector.GAS: "green", Sector.HEAT: "red", Sector.ELECTRICITY: "orange"}[
        sector
    ]


# ---------------------------------------------------------------------------
# Grid-constraint observation helpers
# ---------------------------------------------------------------------------

# Constraint-relevant obs keys. Must match monee model.values keys, which
# are per-unit / SI (Kelvin) — NOT engineering units (bar, °C).
_CONSTRAINT_OBS_KEYS: dict[Sector, dict[str, str]] = {
    Sector.ELECTRICITY: {
        "vm_pu": "vm_pu",              # Bus
        "loading_percent": "loading_percent",  # PowerLine
    },
    Sector.GAS: {
        "pressure_pu": "pressure_pu",  # Junction
    },
    Sector.HEAT: {
        "t_k": "t_k",                  # Junction [K]
    },
}


def obs_constraint_values(obs: dict, sector: Sector) -> dict[str, float]:
    """Extract grid-constraint measurements from an observation dict.

    ``loading_percent`` has two monee variants: a fraction ([0,1], from
    ``GenericPowerBranch``) and an actual percent (×100, from
    ``IntermediateEq``). ``SECTOR_CONSTRAINTS`` uses percent, so the
    fraction form is scaled by 100×; the discriminator is magnitude (a
    value ≤ 5 can only be the fraction form).

    ``loading_percent`` is a Python property, not in ``model.values``, so
    fall back to the max of the ``loading_from/to_percent`` Vars when the
    bare key is missing.
    """
    keys = _CONSTRAINT_OBS_KEYS.get(sector, {})
    result: dict[str, float] = {}
    for var, obs_key in keys.items():
        raw: float | None = None
        if obs_key in obs:
            raw = float(obs[obs_key])
        elif var == "loading_percent":
            lf = obs.get("loading_from_percent")
            lt = obs.get("loading_to_percent")
            if lf is not None or lt is not None:
                raw = max(
                    abs(float(lf)) if lf is not None else 0.0,
                    abs(float(lt)) if lt is not None else 0.0,
                )
        if raw is None:
            continue
        if var == "loading_percent" and abs(raw) <= 5.0:
            raw = raw * 100.0
        result[var] = raw
    return result


def constraint_utilization(
    value: float, bound_low: float, bound_high: float
) -> float:
    """Return 0..1 indicating how close *value* is to violating a bound.

    0.0 = at the centre of the feasible range.
    1.0 = at or beyond a bound.
    """
    span = bound_high - bound_low
    if span <= 0:
        return 1.0
    mid = (bound_low + bound_high) / 2.0
    return min(1.0, abs(value - mid) / (span / 2.0))


def obs_priority(
    obs: dict,
    *,
    behavior: Any = None,
    aid: str | None = None,
    record_default_fallback_t: float | None = None,
) -> int:
    """Read an explicit priority value from an observation dict.

    monee obs carry no ``priority`` key, so this is meaningful only when
    callers pre-populate priorities (via the priority registry or an
    explicit obs key). The fallback returns tier 0 for generators and tier
    4 (sheddable) for loads; callers needing tier diversity must register
    priorities or set ``priority_assignment`` in the scenario.

    Slack agents are always tier 0 regardless of the LP's current sign —
    a slack supplies/absorbs at the network boundary and is never shed.

    Pass ``record_default_fallback_t`` to surface a one-shot
    ``priority_default_fallback`` event the first time a (behavior, aid)
    takes the fallback branch, so missed registrations show up in events.
    """
    if behavior is not None and aid is not None:
        if lookup_slack(behavior, aid) is not None:
            return 0
        registered = lookup_priority(behavior, aid)
        if registered is not None:
            return registered
    if "priority" in obs:
        return int(obs["priority"])
    cap = obs_capacity(obs)
    # Unannotated loads default to tier 4 (sheddable). Tier 1 is hard-locked
    # at x=1, so defaulting there would over-assign critical priority to
    # unregistered loads; tier 4 means missing annotation -> first to shed.
    fallback = 0 if cap < 0 else 4
    if (
        record_default_fallback_t is not None
        and behavior is not None
        and aid is not None
        and cap > 0  # loads only; generators legitimately default to tier 0
    ):
        seen = getattr(behavior, "_scare_prio_fallback_seen", None)
        if seen is None:
            seen = set()
            behavior._scare_prio_fallback_seen = seen
        if aid not in seen:
            seen.add(aid)
            record_event(
                t=float(record_default_fallback_t),
                kind="priority_default_fallback",
                aid=str(aid),
                detail=f"fallback_tier={fallback}",
            )
    return fallback


def compute_priority_weighted_shares(
    demand_by_priority_per_group: list[dict[int, float]],
    served_by_priority_per_group: list[dict[int, float]],
    total_available: float,
) -> list[float]:
    """Compute each group's share of *total_available* via waterfall allocation.

    From the highest-priority tier down, allocate proportionally to
    unserved demand within each tier until the budget is exhausted, so
    critical loads across all groups are served before any low-priority
    load. Returns one share per group, summing to at most *total_available*.
    """
    n = len(demand_by_priority_per_group)
    shares = [0.0] * n
    if total_available <= 0 or n == 0:
        return shares

    all_tiers = sorted(
        {t for d in demand_by_priority_per_group for t in d}
    )
    remaining = total_available

    for tier in all_tiers:
        if remaining <= 1e-9:
            break
        tier_unserved = []
        for i in range(n):
            demand = demand_by_priority_per_group[i].get(tier, 0.0)
            served = served_by_priority_per_group[i].get(tier, 0.0)
            tier_unserved.append(max(0.0, demand - served))

        total_tier = sum(tier_unserved)
        if total_tier <= 1e-9:
            continue

        allocatable = min(remaining, total_tier)
        for i in range(n):
            share = allocatable * (tier_unserved[i] / total_tier)
            shares[i] += share
        remaining -= allocatable

    return shares


def aggregate_priority_weight(
    demand_by_priority: dict[int, float],
    served_by_priority: dict[int, float],
) -> float:
    """Compute a scalar urgency weight from priority-tier demand breakdown.

    Higher-priority tiers contribute more weight per unit unserved demand.
    Used by the L3 CP S-coefficient to pull allocation toward sectors with
    high-priority unmet demand. Uses the strict-monotone schedule
    (:func:`tier_priority_weight_strict`), not the L1 QP schedule which
    returns 0 for tier 1 and would mask tier-1 unmet demand here.
    """
    weight = 0.0
    for tier, demand in demand_by_priority.items():
        served = served_by_priority.get(tier, 0.0)
        unserved = max(0.0, demand - served)
        weight += unserved * tier_priority_weight_strict(int(tier))
    return weight


# 4-tier priority model with hard tier-1 enforcement.
#
# Tier 1 = critical: leader pre-applies ``regulation = 1`` before the
# gossip QP runs, then carries a defensive QP weight of 1.0 (its δ-box
# collapses post-pre-step, so the QP pins δ=0 regardless of weight).
# Tiers 2-4 = QP-weighted with steep exponents so the proportional
# equilibrium is effectively strict. Generators (tier <= 0) keep unit weight.
DEFAULT_PRIORITY_TIERS: int = 4

# Restoration (target > 0): higher-priority tiers get higher weight.
# Tier 1 weight is 0 — hard-locked at the pre-step, so it must not
# participate in the QP and contributes nothing to the dual normaliser.
_TIER_WEIGHT_RESTORATION: dict[int, float] = {
    1: 0.0,
    2: 1e8,
    3: 1e4,
    4: 1.0,
}

# Curtailment (target < 0): lowest-priority tier sheds first.  Tier 1 is
# always pre-locked at full and never sheds via the QP.
_TIER_WEIGHT_CURTAILMENT: dict[int, float] = {
    1: 0.0,
    2: 1.0,
    3: 1e4,
    4: 1e8,
}


def tier_priority_weight(
    tier: int,
    *,
    regime: int = 1,
    priority_tiers: int = DEFAULT_PRIORITY_TIERS,
) -> float:
    """Single source of truth for the per-tier QP weight (L1 gossip).

    4-tier schedule with hard tier-1 enforcement off-QP:

    * ``regime > 0`` (restoration): tier 2 → 1e8, 3 → 1e4, 4 → 1.
    * ``regime < 0`` (curtailment): tier 4 → 1e8 (sheds first), 3 → 1e4, 2 → 1.
    * ``regime == 0``: 1.0.
    Tier 1 returns the defensive weight 1.0 in all regimes (hard-locked
    at the pre-step, never negotiated via the QP).

    ``priority_tiers`` is kept for API compatibility; the schedule is
    fixed at 4 tiers and inputs are clamped to ``[1, 4]``.
    """
    p = max(0, int(tier))
    if regime == 0 or p <= 0:
        return 1.0
    p = min(p, 4)
    if regime > 0:
        return _TIER_WEIGHT_RESTORATION.get(p, 1.0)
    return _TIER_WEIGHT_CURTAILMENT.get(p, 1.0)


def tier_priority_weight_strict(
    tier: int,
    *,
    priority_tiers: int = DEFAULT_PRIORITY_TIERS,
) -> float:
    """Strictly-monotone tier weight for waterfall-style sorts.

    L2's supply-priority waterfall sorts cells by weight, and tier 1 must
    sort first — but the QP schedule gives tier 1 a low weight. This
    returns a schedule strictly decreasing in tier (tier 1 → P, tier P →
    1), keeping sort-by-weight intact without the wild magnitudes that
    would destabilise the ADMM sharing-distance objective.
    """
    P = max(1, int(priority_tiers))
    p = max(1, min(P, int(tier)))
    return float(P - p + 1)


def remap_legacy_priority(tier: int) -> int:
    """Map a legacy 10-tier value onto the 4-tier schedule.

    Buckets: ``{1,2,3}→1``, ``{4,5}→2``, ``{6,7}→3``, ``{8,9,10}→4``.
    Tier 0 (generator class) passes through unchanged.
    """
    t = int(tier)
    if t <= 0:
        return 0
    if t <= 3:
        return 1
    if t <= 5:
        return 2
    if t <= 7:
        return 3
    return 4


# Tier-aware deadbands for ``clamp_to_constraints``: higher deadband =
# measurement must drift closer to a hard bound before the clamp throttles.
# Tier 1 is fully immune (handled in ``clamp_to_constraints``); lower
# tiers throttle more aggressively as priority drops.
_CLAMP_TIER_DEADBAND: dict[int, float] = {
    2: 0.95,
    3: 0.90,
    4: 0.85,
}
_CLAMP_DEFAULT_DEADBAND: float = 0.85  # untagged / out-of-range tiers


def clamp_to_constraints(
    setpoint: float,
    obs: dict,
    sector: Sector,
    *,
    tier: int | None = None,
) -> float:
    """Clamp a proposed setpoint so it stays within local constraint bounds.

    When a local grid measurement approaches a hard bound, reduce the
    proposed setpoint to avoid actuating a violation. Activates only past
    a tier-dependent deadband; above it the allowed fraction ramps linearly
    to zero:

        allowed = (1 - util) / (1 - DEADBAND)   for util ∈ [DEADBAND, 1]
        allowed = 1.0                            for util < DEADBAND

    The deadband prevents normal LV voltage drift from cutting every load
    and overriding the priority-aware gossip waterfall. ``tier`` is the
    load's priority tier (1 = most critical). Tier 1 is immune to clamping
    (its pre-step lock at ``regulation = 1`` must not be overruled by a soft
    proximity signal; a true ConstraintViolation re-checks it). Tiers 2/3/4
    use deadbands 0.95/0.90/0.85; ``None`` uses the uniform 0.85 default.
    """
    cap = obs_capacity(obs)
    if cap == 0.0:
        return setpoint

    tightest_fraction = constraint_allowed_fraction(obs, sector, tier=tier)
    if tightest_fraction < 1.0:
        max_abs = tightest_fraction * abs(cap)
        setpoint = max(-max_abs, min(max_abs, setpoint))

    return setpoint


def constraint_allowed_fraction(
    obs: dict,
    sector: Sector,
    *,
    tier: int | None = None,
) -> float:
    """Tightest constraint-allowed served fraction ``∈ [0, 1]`` from local
    grid measurements, using the same tier-dependent deadband as
    :func:`clamp_to_constraints` (tier 1 immune → 1.0; tiers 2/3/4 use the
    ``_CLAMP_TIER_DEADBAND`` schedule).

    The fraction of rated capacity the load may be served at given local
    physics, before the priority decision. Shared by ``clamp_to_constraints``
    and the L2 priority-floor (``l2_effective_floor``) so the floor relaxes
    by exactly the amount the clamp sheds — they never fight over a load.
    """
    # Tier 1 is immune to the soft proximity clamp; a true
    # ConstraintViolation re-checks its feasibility instead.
    if tier is not None and int(tier) == 1:
        return 1.0
    if tier is not None and int(tier) >= 2:
        deadband = _CLAMP_TIER_DEADBAND.get(int(tier), _CLAMP_DEFAULT_DEADBAND)
    else:
        deadband = _CLAMP_DEFAULT_DEADBAND
    width = max(1e-9, 1.0 - deadband)

    tightest_fraction = 1.0
    for var, (lo, hi) in SECTOR_CONSTRAINTS.get(sector, {}).items():
        if var not in obs:
            continue
        val = float(obs[var])
        if not math.isfinite(val):
            continue
        util = constraint_utilization(val, lo, hi)
        if util <= deadband:
            allowed = 1.0
        else:
            allowed = max(0.0, (1.0 - util) / width)
        tightest_fraction = min(tightest_fraction, allowed)
    return tightest_fraction
