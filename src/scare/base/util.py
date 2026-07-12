from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from monee.model.child import ExtHydrGrid, ExtPowerGrid, Sink

from scare.base.model import SECTOR_CONSTRAINTS, Sector
from scare.base.runtime.diagnostics import record_event, record_regulate

# Natural gas HHV. MW/(kg/s) factor is 3.6*HHV, not HHV itself.
# Must match the fluid of the simulated grids: all benchmark nets are built
# with gas_type="lgas" (monee model/grid.py), not hgas (15.3).
HHV: float = 11.79011  # kWh/kg (lgas)

_CAPACITY_KEYS = (
    "p_mw",
    "q_mw_heat",  # heat load capacity [MW]
    "q_mw_set",  # heat exchanger setpoint [MW]
    "q_mw",  # heat branch actual power [MW]
    "mass_flow_kgs",
    "p_kw",
    "q_mvar",
    "p_mw_capacity",
    "mass_flow_capacity_kgs",
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

    Slack children carry the LP's current operating point in obs, not the
    rating, so return the registered rating when it resolves.
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

    Non-slack: ``capacity * regulation``. Slack has no regulation knob;
    the dispatched value is the LP-chosen value in obs.
    """
    if behavior is not None and aid is not None:
        slack = lookup_slack(behavior, aid)
        if slack is not None:
            # Slack: LP-chosen operating point is in the obs key.
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

    Slack δ-range is the full Var bound range minus the current value;
    other children stay in ``[-sp, cap-sp]`` / ``[cap-sp, -sp]``.
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

    Returns None for multi-grid nodes (e.g. CHPControlNode) which straddle
    sectors and must be resolved by context.
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
    """Lazy per-behavior registry accessor; storing on the behavior ties
    registry lifetime to the simulation world."""
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
# Slack children carry rated capacity in Var bounds not present in the obs
# dict. This registry holds the rating + δ-range so obs_* helpers don't
# mistake a slack's current LP value for its capacity.


@dataclass(frozen=True)
class _SlackMeta:
    """Cached slack rating + δ-range for one ExtPowerGrid / ExtHydrGrid child.

    ``cap < 0`` (a slack is always a source). Values are in the slack's
    native sector unit (MW for power, kg/s for gas), NOT MW-normalised;
    pooling a gas slack with MW quantities needs ``kgps_to_mw`` first.
    """

    cap: float  # rated output, < 0 (generator convention, native unit)
    dmin_abs: float  # min absolute Var value (p_mw / mass_flow)
    dmax_abs: float  # max absolute Var value (p_mw / mass_flow)


def _slack_store(behavior: Any) -> dict[str, _SlackMeta]:
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

    ``rating_mw`` is the positive rated capacity. ``p_min`` / ``p_max`` are
    the Var bounds (load convention); both None ⇒ bidirectional
    ``[-rating_mw, +rating_mw]``. Despite the name, stored in the slack's
    native unit — gas slacks pass kg/s, not MW.
    """
    if rating_mw <= 0.0:
        # Non-positive rating leaves the slack unregistered, falling back
        # to the LP value and reclassifying it as a load. Surface it.
        logging.getLogger(__name__).warning(
            "register_slack(%s, rating_mw=%s): non-positive rating; "
            "slack will fall back to LP-value capacity, which is "
            "rarely what callers want.",
            aid,
            rating_mw,
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


def lookup_slack(behavior: Any, aid: str) -> _SlackMeta | None:
    return _slack_store(behavior).get(aid)


def _slack_eff_budget_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_slack_eff_budget")


def set_slack_eff_budget(behavior: Any, aid: str, value: float) -> None:
    """Record a slack's loss-compensated effective budget (maintained by
    ``SlackBudgetMonitor``) so control targets ``B - losses`` and the actual
    draw lands at operator budget ``B``."""
    _slack_eff_budget_store(behavior)[aid] = float(value)


def lookup_slack_eff_budget(behavior: Any, aid: str) -> float | None:
    return _slack_eff_budget_store(behavior).get(aid)


def _slack_cp_reserve_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_slack_cp_reserve")


def set_slack_cp_reserve(behavior: Any, aid: str, mw: float) -> None:
    """Record the MW of a slack's budget already consumed beyond it — the
    measured over-draw the SlackBudgetMonitor sees. The holon supply pool debits
    the slack's credited budget by this so the electricity holon balances native
    load against the budget NET of the cross-sector (CP) draw + losses riding
    the slack, and sheds native load until the physical draw lands at B (the CP
    draw is otherwise invisible to the holon — see project_slack_compliance_rootcause)."""
    _slack_cp_reserve_store(behavior)[str(aid)] = max(0.0, float(mw))


def lookup_slack_cp_reserve(behavior: Any, aid: str) -> float | None:
    return _slack_cp_reserve_store(behavior).get(str(aid))


def _slack_pressure_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_slack_pressure")


def set_slack_pressure(behavior: Any, aid: str, value: float) -> None:
    """Command a gas slack's pressure setpoint — the regulator lever.

    Writes the ``ExtHydrGrid.pressure_pu`` boundary through the environment's
    ``set_pressure`` action (marks the net dirty; the next energy-flow solve
    re-pins the slack node, see ``ExtHydrGrid.overwrite``) and records the
    commanded value so the regulator role and the slack recorder can read the
    current setpoint. No-op on a child without the action (non-gas-slack) — the
    store is written only when the action actually fires, so ``lookup`` never
    reports a setpoint that was never applied."""
    if behavior.has_action(aid, "set_pressure"):
        behavior.act(aid, "set_pressure", float(value))
        _slack_pressure_store(behavior)[aid] = float(value)


def lookup_slack_pressure(behavior: Any, aid: str) -> float | None:
    """Last commanded slack pressure setpoint [p.u.], or ``None`` if the
    regulator has not actuated this slack yet (its boundary is still the
    ``ExtHydrGrid`` construction default)."""
    return _slack_pressure_store(behavior).get(aid)


def _priority_store(behavior: Any) -> dict[str, int]:
    return _get_behavior_store(behavior, "_scare_priorities")


def register_priority(behavior: Any, aid: str, tier: int) -> None:
    """Record an agent's priority tier so callers that don't own the role
    can look it up; without it ``obs_priority`` falls back to uniform
    priorities. Tier 0 is reserved for generators and slacks."""
    _priority_store(behavior)[aid] = int(tier)


def lookup_priority(behavior: Any, aid: str) -> int | None:
    return _priority_store(behavior).get(aid)


# ---------------------------------------------------------------------------
# Regulate-action de-duplication
# ---------------------------------------------------------------------------

# Re-applying the same factor within this tolerance is a no-op (below any
# monitored-constraint precision, so it doesn't churn monee state).
_REGULATE_DEDUP_TOL: float = 1e-3


# Reasons carrying the L2 holon's per-tier allocation — these SET the L2
# floor; L1 reactive sheds are clamped UP to it so a supply-poor group
# can't undo a served-tier decision.
L2_ALLOCATION_REASONS: frozenset[str] = frozenset(
    {"holon_supply_priority", "holon_tier_alloc"}
)
# ``tier1_starvation`` is the tier-1 hard pre-step zeroing non-tier-1 loads —
# an L1 reactive shed like the others, so the L2 floor clamps it too.
L1_REACTIVE_SHED_REASONS: frozenset[str] = frozenset(
    {"balance", "stability", "tier1_starvation"}
)

# Community curtailment auction reason; heat-only L2 defer signal (see
# ``apply_regulate``).
CURTAIL_AUCTION_REASON: str = "curtail"
# Soft congestion-price curtailment for line loading. Deliberately NOT
# ``CURTAIL_AUCTION_REASON``: it must NOT arm the gen over-voltage curtail-lock
# (which clamps PV to ~0 and starves downstream load — the A/B-refuted
# pathology). The congestion ceiling is a reversible cap enforced in the gossip
# ``_apply_setpoint``; when the line clears, the price decays and the cap lifts.
LINE_CONGESTION_REASON: str = "line_congestion"
# Heat un-shed recovery reason; lifts the heat curtail lock as it ramps a
# recovered load back toward full service.
HEAT_RECOVERY_REASON: str = "heat_recovery"

# Generator RESTORE reasons (inline self-dispatch, fallback role, gen-ramp
# controller, and the R3 L2 service-fraction gen ramp). All ramp a generator
# UP; while the auction holds it down for a local violation they must DEFER —
# see the gen curtail-lock in ``apply_regulate``.
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


def clamp_tier_monotonic(fraction_by_tier: dict[int, float]) -> dict[int, float]:
    """Clamp per-tier service fractions non-increasing in tier number (tier 1
    = highest priority). Priority-safe: only ever lowers a lower-priority tier
    to its higher tier's level. Mutates and returns ``fraction_by_tier``.
    Shared by the component-allocation and coalition dispatch paths so a
    coalition merge can't reintroduce a tier inversion."""
    cap = 1.0
    for tier in sorted(t for t in fraction_by_tier if t >= 1):
        fraction_by_tier[tier] = min(fraction_by_tier[tier], cap)
        cap = fraction_by_tier[tier]
    return fraction_by_tier


def _heat_curtail_lock_store(behavior: Any) -> dict[str, float]:
    """Per-aid heat curtailment-auction lock (regulation level held for a
    live temperature violation). An entry means the auction owns the load
    and L2 must defer. Set by ``curtail``, lifted by ``heat_recovery``."""
    return _get_behavior_store(behavior, "_scare_heat_curtail_lock")


def _line_curtail_lock_store(behavior: Any) -> dict[str, tuple]:
    """Per-aid electricity line-relief lock: ``aid -> (factor, t_set)``.

    Set by the line-relief auction (``curtail``). While fresh (re-asserted
    within ``_LINE_CURTAIL_LOCK_TTL_S`` every poll the line is over), L2
    writes DEFER else the holon re-serves a just-shed load. Freshness-lifted:
    once the line clears the auction stops re-arming and it goes stale.
    Electricity analogue of the heat curtail lock."""
    return _get_behavior_store(behavior, "_scare_line_curtail_lock")


# Line-relief lock TTL after last refresh. Exceeds the monitor poll (~0.5 s)
# so it survives between re-arms, but short enough to release soon after clear.
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


def _line_congestion_store(behavior: Any) -> dict[tuple[str, str], tuple]:
    """Additive congestion price per ``(branch_key, aid)``: ``-> (price, t)``.

    Each overloaded branch publishes a per-downstream-generator price
    ``price = 1 - ceiling`` (0 = no curtail, 1 = full). Keyed by branch so a
    generator downstream of several congested branches accumulates the SUM
    (:func:`line_congestion_ceiling`), rather than a later branch overwriting an
    earlier one. Freshness-stamped; a stale entry drops out of the sum."""
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


def line_congestion_ceiling(
    behavior: Any, aid: str, now: float, ttl: float
) -> float:
    """Generation ceiling (max regulation factor) for *aid* = ``1 - Σ fresh
    branch prices``, clamped to ``[0, 1]``. Returns 1.0 (no cap) when no fresh
    price is published."""
    store = _line_congestion_store(behavior)
    total = 0.0
    for (_branch, a), (price, t_set) in list(store.items()):
        if a == str(aid) and (now - float(t_set)) < ttl:
            total += float(price)
    return max(0.0, min(1.0, 1.0 - total))


# TTL (sim-s) for a CP heat-outlet ceiling entry. The guard republishes every
# poll (~1 s); a stale entry means the guard died or its junction obs vanished
# — release the cap rather than pin the CP down forever.
_CP_HEAT_CEILING_TTL_S: float = 5.0


def _cp_heat_ceiling_store(behavior: Any) -> dict[str, tuple]:
    """Per-CP regulation ceiling from the heat-outlet guard:
    ``aid -> (ceiling, t_set)``.

    Written by ``CPHeatOutletGuard`` (the sensor/controller half); enforced in
    :func:`apply_regulate` for every ``sector="cp"`` write. Enforcement must
    live there — the L3 kernels re-commit the deficit-filling factor every
    round (delivered heat is measured at load setpoints, which CP injection
    can never raise), so a one-shot wind-down without a held cap is
    immediately overwritten. Freshness-stamped."""
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


# Min fresh loading headroom (%-points below the limit) at which the line lock
# hands active back (analogue of ``_QV_LOCK_RELEASE_MARGIN_PU``). The restore
# step raises loading, so this cushion keeps the closed loop settling just
# under the limit rather than re-breaching it.
_LINE_RELIEF_HANDOFF_HEADROOM_PCT: float = 8.0

# Max regulation the line lock hands back per restore cycle (Mechanism B). The
# hand-back is a headroom-gated ramp re-evaluated each cycle, so a speed knob
# (any value in (0, 1] is stable), not load-bearing.
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


# --- Coordinated Q(U)-droop / curtailment-auction reactive-relief ledger ---
# Shared state through which the Q(U) droop tells the auction (and gen
# curtail-lock) how much more over-voltage relief its reactive lever can still
# deliver. Gated on ``enable_qv_auction_coordination`` (see config.py).


def _qv_relief_store(behavior: Any) -> dict[str, tuple]:
    """Per-aid reactive voltage state from the Q(U) droop:
    ``aid -> (t_set, relief_pu, v_pu)``.

    ``relief_pu = (q_max − |q_cmd|) · |dV/dQ|`` is the extra p.u. voltage
    reduction the inverter's unused reactive capability could still provide
    (not yet in ``vm_pu``); ``v_pu`` is the latest local voltage. Read by the
    auction to shed only residual over-voltage and by the gen curtail-lock to
    release active only once reactive holds voltage in-band. Freshness-stamped."""
    return _get_behavior_store(behavior, "_scare_qv_relief")


# Published reactive-relief reading TTL. Exceeds the droop poll (~0.5 s) so it
# survives between ticks, but short so a stalled inverter's relief expires.
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


# Min fresh reactive headroom (p.u.) at which the gen curtail-lock releases
# early (Mechanism B): the droop must re-absorb at least this much of the rise
# that ramping active back causes.
_QV_LOCK_RELEASE_MARGIN_PU: float = 1e-3

# Upper voltage (p.u.) below which the droop is holding the node genuinely
# in-band, so handing active back is safe. Above it restoring risks re-breach.
# Anchored at the VDE deadband top.
_QV_LOCK_RELEASE_V_CEILING_PU: float = 1.03

# Max regulation the gen lock hands back per restore cycle (Mechanism B). The
# hand-back is a voltage-gated ramp re-evaluated each cycle, so a speed knob
# (any value in (0, 1] is stable), not load-bearing.
_QV_LOCK_RESTORE_STEP: float = 0.1


# --- Phase-2 feeder-voltage ledger -----------------------------------------
# Shared blackboard of per-node voltage so an inverter's auction can see whether
# the FEEDER (not just its own node) is over-voltage. Reuses the per-world
# behaviour store (like the curtail-locks). Assumes one LV feeder per
# electricity grid (true for the simbench_lv_* family).


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
    """Per-behavior tracker of regulate-on-stale-observation events.

    ``behavior._net_results`` is replaced only on a successful solve, so its
    ``id()`` is a cheap freshness oracle: if unchanged and an apply already
    landed on it, the regulate is acting on stale state.
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


# Tiers at or below this bypass the cooldown: a critical dispatch must not be
# dropped for arriving within cooldown_s of an unrelated update.
_COOLDOWN_BYPASS_TIER_THRESHOLD: int = 2


def _is_slack_class_child(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a monee ``ExtPowerGrid`` / ``ExtHydrGrid`` slack child.

    Writing ``regulation < 1`` clamps the slack's free Var and the next solve
    goes infeasible once the network needs headroom, so curtail/stability/gossip
    writes must skip slacks. Class-based not registry-based: the unbounded
    heat-side ExtHydrGrid never registers yet is structurally a slack.
    """
    if not aid.startswith("child-"):
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


def _is_heat_side_mass_flow_sink(behavior: Any, aid: str) -> bool:
    """True iff *aid* is a monee ``Sink`` child on a water/heat junction.

    A heat consumer is a (HeatLoad, Sink) pair; curtailing the Sink's mass flow
    without cutting upstream supply makes the junction mass balance infeasible,
    so thermal curtailment must go through the HeatLoad instead. Gas-sector
    Sinks model real consumption and stay curtailable.
    """
    if not aid.startswith("child-"):
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
            _current = _last_regulate_store(behavior).get(
                str(aid), _lock[str(aid)]
            )
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
            # A further shed always passes (it only helps the line); only a
            # RESTORE is interlocked. Mechanism B: hand active back one bounded
            # step per cycle when the branch has fresh loading headroom below the
            # limit, re-gated each tick — else a one-shot release slams the load
            # to full and the line re-overloads (relaxation limit cycle leaving
            # loading oscillating 40–170%).
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
    if _sector_e is Sector.ELECTRICITY and getattr(
        _cfg, "enable_curtail_ramp_interlock", False
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
            # Mechanism B (coordinated hand-off): hand active back only when the
            # Q(U) droop is BOTH holding the node in-band (v ≤ ceiling) AND has
            # spare reactive headroom to re-absorb the rise — and only one bounded
            # STEP per cycle (closed-loop, re-gated each tick). A one-shot release
            # re-breached over-voltage in validation v1.
            # Saturated / still-elevated droop ⇒ keep deferring.
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
    if getattr(_cfg, "enable_l2_priority_floor", False):
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

    # Stale-observation detector: an unchanged net_results since the last
    # apply means this regulate computes against a stale snapshot.
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
                detail=(f"reason={reason} stale_landed_total={state['stale_landed']}"),
            )
            state["warned_for_id"] = current_id
    elif state["last_id"] != current_id:
        state["last_id"] = current_id
        state["applies_on_current_id"] = 0
        state["warned_for_id"] = None
    state["applies_on_current_id"] += 1

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


def obs_sector(
    obs: dict,
    *,
    behavior: Any = None,
    aid: str | None = None,
) -> Sector | None:
    """Resolve the energy sector an observation belongs to.

    Prefers the (behavior, aid) registry; the obs-key heuristic is a
    last-resort fallback (gas/water junction obs are shape-identical).
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
from scare.scenario.failure_sampling import create_failures  # noqa: E402,F401


def efficiency_vector(eta_el: float, eta_heat: float, eta_gas: float) -> np.ndarray:
    return np.array([eta_el, eta_heat, eta_gas], dtype=float)


# Re-export for backwards compatibility with callers importing them here.
from scare.base.optimization.admm_factories import (  # noqa: E402,F401
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

# Constraint-relevant obs keys. Must match monee model.values keys (per-unit /
# SI Kelvin, NOT bar/°C).
_CONSTRAINT_OBS_KEYS: dict[Sector, dict[str, str]] = {
    Sector.ELECTRICITY: {
        "vm_pu": "vm_pu",  # Bus
        "loading_percent": "loading_percent",  # PowerLine
    },
    Sector.GAS: {
        "pressure_pu": "pressure_pu",  # Junction
    },
    Sector.HEAT: {
        "t_k": "t_k",  # Junction [K]
    },
}


# Mirrors monee problem/utils.py and the eval grader: max_i_ka at/above this
# sentinel means the branch carries no current rating to grade against.
_UNBOUND_MAX_I_KA: float = 999.0


def _obs_branch_loading_percent(obs: dict) -> float | None:
    """Branch loading percent from a branch observation, in the exact basis
    the oracle's ``line_loading_limit`` enforces and the eval grader re-judges
    in (``metrics._branch_loading_percent``): apparent power against
    ``max_s_mva`` when the branch is MVA-rated, else the from-side current
    fraction ``loading_from_pu``. The to-side current is never divided by
    ``max_i_ka``: that rating is expressed in the from-side voltage basis
    (monee ``io/matpower.py``), so a transformer's ``loading_to_pu`` is
    inflated by the voltage ratio — 50x on a 20/0.4 kV trafo, the phantom
    ~4000% overloads that mass-shed tiers 2-4 in eval_full_20260702-133421.
    """
    try:
        mva = float(obs["max_s_mva"])
    except (KeyError, TypeError, ValueError):
        mva = math.nan
    if math.isfinite(mva) and mva > 0.0:
        s_sides: list[float] = []
        for side in ("from", "to"):
            try:
                p = float(obs[f"p_{side}_mw"])
                q = float(obs[f"q_{side}_mvar"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(p) and math.isfinite(q):
                s_sides.append(math.hypot(p, q))
        if s_sides:
            return 100.0 * max(s_sides) / mva
    try:
        lf = float(obs["loading_from_pu"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(lf):
        return None
    try:
        max_i = float(obs["max_i_ka"])
    except (KeyError, TypeError, ValueError):
        max_i = math.nan
    if math.isfinite(max_i) and not (0.0 < max_i < _UNBOUND_MAX_I_KA):
        return None
    return 100.0 * abs(lf)


def obs_constraint_values(obs: dict, sector: Sector) -> dict[str, float]:
    """Extract grid-constraint measurements from an observation dict.

    ``loading_percent``: a direct key is already percent; otherwise it is
    derived from the branch flow/loading vars via
    ``_obs_branch_loading_percent`` (MVA basis when rated, else the from-side
    current fraction x100).
    """
    keys = _CONSTRAINT_OBS_KEYS.get(sector, {})
    result: dict[str, float] = {}
    for var, obs_key in keys.items():
        raw: float | None = None
        if obs_key in obs:
            raw = float(obs[obs_key])
        elif var == "loading_percent":
            raw = _obs_branch_loading_percent(obs)
        if raw is None:
            continue
        result[var] = raw
    return result


def constraint_utilization(
    value: float, bound_low: float, bound_high: float, *, unclamped: bool = False
) -> float:
    """Return how close *value* is to violating a bound.

    0.0 = at the centre of the feasible range.
    1.0 = at or beyond a bound.

    Clamped to ``[0, 1]`` by default. Pass ``unclamped=True`` to let values past a
    bound exceed 1.0 (the fractional overshoot), e.g. for integrating the
    out-of-bounds area; every other caller relies on the ``1.0`` ceiling.
    """
    span = bound_high - bound_low
    if span <= 0:
        return 1.0
    mid = (bound_low + bound_high) / 2.0
    u = abs(value - mid) / (span / 2.0)
    return u if unclamped else min(1.0, u)


def obs_priority(
    obs: dict,
    *,
    behavior: Any = None,
    aid: str | None = None,
    record_default_fallback_t: float | None = None,
) -> int:
    """Read an explicit priority tier for a (behavior, aid) or obs dict.

    Meaningful only when priorities are pre-populated; the fallback is tier 0
    for generators, tier 4 (sheddable) for loads. Slacks are always tier 0
    (never shed). ``record_default_fallback_t`` surfaces a one-shot
    ``priority_default_fallback`` event the first time the fallback is taken.
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
    # Unannotated loads default to tier 4 (first to shed); defaulting to the
    # hard-locked tier 1 would over-assign critical priority.
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

    From the highest tier down, allocate proportionally to unserved demand
    until the budget is exhausted. Returns one share per group, summing to at
    most *total_available*.
    """
    n = len(demand_by_priority_per_group)
    shares = [0.0] * n
    if total_available <= 0 or n == 0:
        return shares

    all_tiers = sorted({t for d in demand_by_priority_per_group for t in d})
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
    """Scalar urgency weight from a priority-tier demand breakdown.

    Higher tiers weigh more per unit unserved demand. Used by the L3 CP
    S-coefficient. Uses the strict-monotone schedule, not the L1 QP schedule
    which returns 0 for tier 1 and would mask tier-1 unmet demand.
    """
    weight = 0.0
    for tier, demand in demand_by_priority.items():
        served = served_by_priority.get(tier, 0.0)
        unserved = max(0.0, demand - served)
        weight += unserved * tier_priority_weight_strict(int(tier))
    return weight


# 4-tier priority model with hard tier-1 enforcement. Tier 1 is pre-locked at
# ``regulation = 1`` off-QP; tiers 2-4 are QP-weighted with steep exponents so
# the equilibrium is effectively strict. Generators (tier <= 0) keep unit weight.
DEFAULT_PRIORITY_TIERS: int = 4

# Restoration (target > 0): higher tiers get higher weight. Tier 1 weight is 0
# (hard-locked at the pre-step, must not enter the QP or the dual normaliser).
_TIER_WEIGHT_RESTORATION: dict[int, float] = {
    1: 0.0,
    2: 1e8,
    3: 1e4,
    4: 1.0,
}

# Curtailment (target < 0): lowest tier sheds first. Tier 1 pre-locked at full.
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

    ``regime > 0`` restoration: tier 2→1e8, 3→1e4, 4→1. ``regime < 0``
    curtailment: 4→1e8 (sheds first), 3→1e4, 2→1. ``regime == 0``: 1.0.
    Tier 1 returns 0.0 (hard-locked off-QP; must not enter the QP or the dual
    normaliser). ``priority_tiers`` kept for API compatibility; schedule is
    fixed at 4 tiers, inputs clamped to ``[1, 4]``.
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
    """Strictly-monotone tier weight (tier 1 → P, tier P → 1) for
    waterfall-style sorts; tier 1 must sort first, which the QP schedule's low
    tier-1 weight breaks. Avoids the QP's wild magnitudes that would
    destabilise the ADMM sharing-distance objective.
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


# Tier-aware deadbands for the clamp: higher deadband = measurement must drift
# closer to a hard bound before throttling. Tier 1 is fully immune.
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
    """Clamp a proposed setpoint within local constraint bounds.

    Past a tier-dependent deadband the allowed fraction ramps linearly to zero
    (``(1-util)/(1-DEADBAND)``); the deadband stops normal LV drift from cutting
    every load and overriding the gossip waterfall. Tier 1 is immune (its
    pre-step lock must not be overruled; a true ConstraintViolation re-checks
    it). Tiers 2/3/4 → 0.95/0.90/0.85; ``None`` → 0.85.
    """
    cap = obs_capacity(obs)
    if cap == 0.0:
        return setpoint

    tightest_fraction = constraint_allowed_fraction(obs, sector, tier=tier)
    if tightest_fraction < 1.0:
        max_abs = tightest_fraction * abs(cap)
        setpoint = max(-max_abs, min(max_abs, setpoint))

    return setpoint


# Direction-aware constraint capping (see constraint_allowed_fraction). Set per
# build from ``RestorationConfiguration.enable_directional_constraint_cap``.
# Module-global (not threaded through every caller) because it is a physics
# invariant, not a per-actor policy; each task rebuilds and re-sets it.
_DIRECTIONAL_CONSTRAINT_CAP: bool = True


def set_directional_constraint_cap(enabled: bool) -> None:
    """Toggle the direction-aware serving cap in :func:`constraint_allowed_fraction`."""
    global _DIRECTIONAL_CONSTRAINT_CAP
    _DIRECTIONAL_CONSTRAINT_CAP = bool(enabled)


def constraint_allowed_fraction(
    obs: dict,
    sector: Sector,
    *,
    tier: int | None = None,
) -> float:
    """Tightest constraint-allowed served fraction ``∈ [0, 1]`` from local
    measurements (same tier deadband as :func:`clamp_to_constraints`).

    The capacity fraction the actor may be served at given local physics, before
    the priority decision. Shared with the L2 priority-floor so the floor relaxes
    by exactly the amount the clamp sheds.

    DIRECTION-AWARE. Serving more (larger ``|setpoint|``) moves node state
    variables (vm_pu, pressure_pu, t_k) DOWN for a load (consumption pulls them
    down) and UP for a generator (injection pushes them up). Only the bound that
    serving pushes the value TOWARD may cap: capping the other side would
    shed/curtail the very actor that RELIEVES the violation. The canonical case:
    over-voltage on a PV-surplus feeder is relieved by SERVING load (it draws the
    surplus down), so a load must not be capped by an over-voltage reading — the
    symmetric ``constraint_utilization`` used to do exactly that and strand the
    load shed. Over-voltage still caps GENERATORS (they cause it). Tier-1 immune.
    """
    # Tier 1 immune to the soft clamp; a true ConstraintViolation re-checks it.
    if tier is not None and int(tier) == 1:
        return 1.0
    if tier is not None and int(tier) >= 2:
        deadband = _CLAMP_TIER_DEADBAND.get(int(tier), _CLAMP_DEFAULT_DEADBAND)
    else:
        deadband = _CLAMP_DEFAULT_DEADBAND
    width = max(1e-9, 1.0 - deadband)

    # Generators (cap < 0) inject → serving raises state vars; loads → lowers.
    serving_raises = obs_capacity(obs) < 0

    tightest_fraction = 1.0
    for var, (lo, hi) in SECTOR_CONSTRAINTS.get(sector, {}).items():
        if var not in obs:
            continue
        val = float(obs[var])
        if not math.isfinite(val):
            continue
        if _DIRECTIONAL_CONSTRAINT_CAP:
            half = (hi - lo) / 2.0
            if half <= 0.0:
                continue
            mid = (lo + hi) / 2.0
            # One-sided utilization in the WORSENING direction only. Serving
            # raises val (generator) ⇒ the HIGH bound worsens; serving lowers
            # val (load) ⇒ the LOW bound worsens. The opposite side is relieved
            # by serving → no cap.
            util = (val - mid) / half if serving_raises else (mid - val) / half
            util = max(0.0, min(1.0, util))
        else:
            # Legacy symmetric behaviour: caps on proximity to EITHER bound.
            util = constraint_utilization(val, lo, hi)
        if util <= deadband:
            allowed = 1.0
        else:
            allowed = max(0.0, (1.0 - util) / width)
        tightest_fraction = min(tightest_fraction, allowed)
    return tightest_fraction
