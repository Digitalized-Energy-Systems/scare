from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from monee.model.child import ExtHydrGrid, ExtPowerGrid, Sink

from scare.base.diagnostics import record_event, record_regulate
from scare.base.model import SECTOR_CONSTRAINTS, Sector

# Higher heating value of natural gas in kWh/kg.  The MW-per-(kg/s)
# factor is 3.6·HHV ≈ 55 (1 kWh/s = 3.6 MW), applied in the converters
# below — do NOT read 15.3 itself as MW/(kg/s).
HHV: float = 15.3  # kWh/kg for natural gas

_CAPACITY_KEYS = (
    "p_mw",
    "q_mw_heat",       # heat childs: heat-load capacity in MW
    "q_mw_set",        # heat branches (heat exchangers): heat setpoint in MW
    "q_mw",            # heat branches: actual heat power in MW
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

    For ``PowerLoad`` / ``PowerGenerator`` / ``HeatLoad`` / ``Sink`` /
    ``Source`` the rated value lives directly in ``obs`` (``p_mw``,
    ``q_mw_heat``, ``mass_flow``, …) — those keys carry the rated
    quantity unchanged through the simulation.

    For ``ExtPowerGrid`` / ``ExtHydrGrid`` the corresponding key
    carries the *current* operating point (the LP picks it every
    step), not the rating.  When the slack-registry hint resolves we
    return the registered rating instead — see ``register_slack``.
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

    For non-slack children ``setpoint = capacity * regulation``.  For
    slack children there is no regulation; the dispatched value is the
    LP-chosen ``p_mw`` / ``mass_flow`` itself, which lives directly in
    ``obs``.
    """
    if behavior is not None and aid is not None:
        slack = lookup_slack(behavior, aid)
        if slack is not None:
            # Slack agents have no regulation knob — the LP picks the
            # actual operating point and stores it in the obs key
            # corresponding to the slack's Var.
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

    For slack children the δ-range is the *full Var bound range minus
    the current value*, capturing the slack's headroom in both
    directions (import and export).  For all other children δ stays in
    ``[-sp, cap-sp]`` / ``[cap-sp, -sp]`` as before.
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

    Returns None for multi-grid nodes (e.g. CHPControlNode) because they
    straddle sectors and the sector has to be chosen explicitly by
    context.
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
    """Lazy ``getattr(behavior, attr) or factory()`` accessor used by the
    per-behavior registries below.  Storing on the behavior ties the
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
# Slack-agent metadata (F1)
# ---------------------------------------------------------------------------
#
# ExtPowerGrid / ExtHydrGrid children carry their rated import/export
# capacity in the Var bounds on ``p_mw`` / ``mass_flow``.  Those bounds
# are not part of the runtime observation dict (only the *current value*
# is), so without an out-of-band registry the gossip negotiator would
# read a slack agent's "capacity" as whatever the LP picked this step
# (and treat it as a load when the slack is importing).  The registry
# below carries the *rated* capacity + bounded ``δ`` range so that
# ``obs_capacity`` / ``obs_min_max`` / ``obs_priority`` can return the
# physically meaningful values for slack children.

@dataclass(frozen=True)
class _SlackMeta:
    """Cached slack rating + δ-range information for one ExtPowerGrid /
    ExtHydrGrid child.  ``cap`` follows monee's load convention:
    negative for sources (generator-class), positive for sinks; the
    slack is always a source from the local network's perspective, so
    ``cap < 0`` (generator-priority).  ``dmin_abs`` / ``dmax_abs`` are
    the absolute bounds on the slack Var; deltas relative to the
    current setpoint are derived in ``obs_min_max``.

    Units are the slack's *native* sector units — MW for an
    ExtPowerGrid (``p_mw`` Var), kg/s for an ExtHydrGrid gas slack
    (``mass_flow`` Var).  These values are produced and consumed within
    a single sector (the gas gossip reads a gas slack's ``cap`` as
    kg/s), so the field is not MW-normalised; any consumer that pools a
    gas slack with MW quantities must ``kgps_to_mw`` first.
    """
    cap: float          # generator-convention rated output, < 0 (native unit)
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

    ``rating_mw`` is the absolute magnitude (positive) of the rated
    transformer / pipeline capacity.  ``p_min`` / ``p_max`` are the
    actual ``p_mw`` Var bounds (load convention: negative = export,
    positive = import).  If both are None, the slack is assumed
    bidirectional at ``rating_mw``: ``[-rating_mw, +rating_mw]``.

    NB: despite the ``rating_mw`` name, the value is stored in the
    slack's *native* sector unit — for an ExtHydrGrid gas slack callers
    pass kg/s (the ``mass_flow`` budget), not MW.  This is consistent
    because every gas-sector consumer treats the stored ``cap`` as
    kg/s; only code that crosses gas into a shared-MW space (e.g. the
    L3 CP-priority kernel) must ``kgps_to_mw`` it.
    """
    if rating_mw <= 0.0:
        # Silent no-op here would leave the slack child unregistered,
        # which downstream ``obs_capacity`` / ``obs_priority`` falls
        # back on the LP's current operating value — i.e. the slack
        # gets reclassified as a load.  Surface the bad input instead.
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
        cap=-float(rating_mw),  # generator-class sign convention
        dmin_abs=float(p_min),
        dmax_abs=float(p_max),
    )


def lookup_slack(behavior: Any, aid: str) -> "_SlackMeta | None":
    return _slack_store(behavior).get(aid)


def _slack_eff_budget_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_slack_eff_budget")


def set_slack_eff_budget(behavior: Any, aid: str, value: float) -> None:
    """Record a slack's *effective* budget — the loss-compensated cap the
    supply pool should advertise, maintained by ``SlackBudgetMonitor``'s
    integral feedback.  ``EnergyBalanceNegotiator._handle_ask_flex`` reads
    it (via :func:`lookup_slack_eff_budget`) in place of the nominal
    ``|cap|`` so the L1/L2/L3 control targets ``B - losses`` and the
    slack's *actual* draw lands at the operator budget ``B``."""
    _slack_eff_budget_store(behavior)[aid] = float(value)


def lookup_slack_eff_budget(behavior: Any, aid: str) -> float | None:
    return _slack_eff_budget_store(behavior).get(aid)


def _priority_store(behavior: Any) -> dict[str, int]:
    return _get_behavior_store(behavior, "_scare_priorities")


def register_priority(behavior: Any, aid: str, tier: int) -> None:
    """Record an agent's priority tier on the behavior so callers
    that don't own the role (e.g. ``EnergyBalanceNegotiator._handle_ask_flex``
    aggregating across all group members) can look it up.

    Without this registry, ``obs_priority(obs)`` falls back to tier 0
    for generators and tier 1 for loads — uniform priorities — and
    every per-tier feature (QP gossip weighting, tier-stratified
    holon ADMM, ``compute_priority_weighted_shares``) degenerates to
    a single-tier baseline.

    Stored values are integers ≥ 0; tier 0 is reserved for
    generator-class agents and slacks.
    """
    _priority_store(behavior)[aid] = int(tier)


def lookup_priority(behavior: Any, aid: str) -> int | None:
    return _priority_store(behavior).get(aid)


# ---------------------------------------------------------------------------
# Regulate-action de-duplication
# ---------------------------------------------------------------------------

# Default tolerance below which a re-application of the same regulation
# factor counts as a no-op.  Heat recovery + cold-load-pickup ramp +
# constraint-aware clamping all produce sub-promille steps that drive
# behavior.act → monee state-dirty churn without changing the operating
# point in any physically observable way.  1e-3 (0.1 % of capacity) is
# below the precision of any constraint we actually monitor.
_REGULATE_DEDUP_TOL: float = 1e-3


# Reasons whose regulate write carries the L2 holon's authoritative
# per-tier allocation — these *set* the load's L2 floor.  Reasons that
# shed reactively at L1 (gossip ``balance`` actuation, ``stability``
# re-apply) are clamped *up* to that floor so a supply-poor local group
# can't undo a served-tier decision the component ADMM just made.
L2_ALLOCATION_REASONS: frozenset[str] = frozenset(
    {"holon_supply_priority", "holon_tier_alloc"}
)
L1_REACTIVE_SHED_REASONS: frozenset[str] = frozenset({"balance", "stability"})

# Reason written by the community curtailment auction (and its L0 self-bid)
# — already surfaced as "curtailment auction" in the eval plots.  Used as
# the heat-only L2 defer signal (see ``apply_regulate`` / the heat curtail
# lock): while a heat load holds an auction curtailment for a live
# violation, L2 allocation writes defer to it.
CURTAIL_AUCTION_REASON: str = "curtail"
# Reason written by the heat un-shed recovery loop; lifts the heat curtail
# lock as it ramps a recovered load back toward full service.
HEAT_RECOVERY_REASON: str = "heat_recovery"


def _last_regulate_store(behavior: Any) -> dict[str, float]:
    return _get_behavior_store(behavior, "_scare_last_regulate")


def note_actuated_factor(behavior: Any, aid: str, factor: float) -> None:
    """Sync the per-aid dedup cache with a regulate actuation that was
    written *outside* :func:`apply_regulate`.

    The gossip path (``EnergyBalanceNegotiator._apply_setpoint``) writes
    ``behavior.act("regulate", …)`` directly so its micro-steps bypass
    the no-op dedup.  But that also means the dedup cache
    (``_scare_last_regulate``) keeps the value from the *last*
    ``apply_regulate`` call, not the gossip's actual write.  When a later
    L2 re-dispatch (``apply_regulate(reason="holon_supply_priority")``)
    asks for the load's allocated factor, the dedup compares against the
    stale cache and silently drops the write — so a load the gossip shed
    to 0 is never restored even though L2 re-dispatched it (eval
    task-105 child-260: shed at t=5.12, six re-dispatches at t≥5.44 all
    deduped, stays 0 to end-of-sim).  Calling this after every direct
    actuator write keeps the cache truthful so corrective writes land.
    """
    _last_regulate_store(behavior)[str(aid)] = float(factor)


def _l2_floor_store(behavior: Any) -> dict[str, float]:
    """Per-aid L2 priority allocation: the served fraction the
    component-scope holon ADMM most recently assigned to this load."""
    return _get_behavior_store(behavior, "_scare_l2_floor")


def _heat_curtail_lock_store(behavior: Any) -> dict[str, float]:
    """Per-aid heat curtailment-auction lock: the regulation level the
    community auction is currently holding a heat load at, in response to a
    live local temperature violation.  Presence of an entry means the
    auction owns this load and L2 allocation writes must defer.  Set by
    ``reason="curtail"`` writes, lifted by ``heat_recovery`` ramp-up — see
    :func:`apply_regulate`."""
    return _get_behavior_store(behavior, "_scare_heat_curtail_lock")


def has_heat_curtail_lock(behavior: Any, aid: str) -> bool:
    """True iff *aid* is currently held by a temperature-driven curtailment
    lock — i.e. the curtailment auction or the heat frontier controller shed
    it (``reason="curtail"``), as opposed to an L2 *priority* shed (which
    sets no lock).  Used by the frontier controller to only restore loads it
    shed for temperature, never to claw back a priority decision."""
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

    Returns ``None`` when the holon has not allocated to this load yet
    (no floor to enforce).  Capping the L2 allocation by the
    constraint-allowed fraction means the floor yields, per-load and
    continuously, to exactly the physical shedding the local constraint
    requires — so curtailment/clamp own the violation window while the
    floor only blocks *balance-driven* shedding below the priority
    decision.
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

    The host environment's ``_accept_or_keep`` re-uses the previous
    ``SolverResult`` whenever an ``energyflow`` solve returns
    infeasible (see ``base.infeasibility_capture``).  Concretely:
    ``behavior._net_results`` is replaced *only* on a successful
    solve, so its ``id()`` is a cheap freshness oracle.

    This helper tracks the last-seen ``id(_net_results)``.  Each
    ``apply_regulate`` call compares the current id against the
    cached one — if unchanged AND at least one apply has already
    landed on this id, the new regulate is acting on stale state
    and is counted in ``stale_landed`` / surfaced via a one-shot
    ``regulate_on_stale_obs`` event for the affected sector.
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


# Tiers at or below this threshold bypass the global cooldown.  A
# tier-1 dispatch decision that would otherwise be silently dropped by
# the global cooldown gate (typical case: gossip lands at t and an L2
# supply-priority decision tries to land at t+ε < cooldown_s) is too
# important to defer — the cooldown's wallclock-cost rationale does
# not apply to the rare critical-load update.  Lower tiers still pay
# the cooldown so the SCADA-cycle scheduling assumption holds for the
# bulk of dispatches.
_COOLDOWN_BYPASS_TIER_THRESHOLD: int = 2


def _is_slack_class_child(behavior: Any, aid: str) -> bool:
    """True iff *aid* refers to a monee ``ExtPowerGrid`` or ``ExtHydrGrid``
    child — the network's slack-class boundary.

    Slacks have a *free* p_mw / mass_flow Var the LP picks within a
    wide physical envelope; writing ``regulation < 1`` clamps the
    effective slack contribution to a fraction of that envelope, and
    the next solve diagnoses infeasible the moment the network needs
    more headroom than the clamped fraction.  Curtailment / stability
    / gossip writes must therefore skip slacks.

    Class-based rather than registry-based because heat-side
    ExtHydrGrid is intentionally unbounded by ``apply_slack_budget``
    (the heat LP has no operator-side slack discipline) and thus
    never lands in the ``register_slack`` registry — yet it is still
    structurally a slack.
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
    """True iff *aid* refers to a monee ``Sink`` child on a water/heat
    grid junction.

    On monee's heat-sector convention every heat consumer is modelled
    as a (HeatLoad, Sink) pair sharing one junction: HeatLoad withdraws
    thermal energy (``q_mw_heat``), Sink withdraws the matching return-
    line mass flow.  Forcing ``Sink.regulation < 1`` zeroes the mass-
    flow withdrawal without zeroing the upstream supply (the pipe is
    still pushing water in via the SubHE/HeatExchanger) — the junction
    mass-flow balance becomes ``positive = 0`` and presolve declares
    the LP infeasible.  Gas-sector Sinks model real gas consumption
    and remain curtailable.

    Curtailment for thermal control should go through the HeatLoad
    (``q_mw_heat * regulation``), which leaves mass flow untouched.
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

    Also enforces a sim-time cooldown when
    ``behavior._scare_config.cooldown_s > 0``: regulate writes for the
    same aid that arrive within ``cooldown_s`` of the previous applied
    write are suppressed regardless of factor delta.  This is the
    "max one solve every Δt" knob discussed for the wallclock cost
    reduction; it lets the SCADA-cycle-style scheduling assumption be
    expressed as a single config flag.

    ``priority_tier`` (when set) lets critical loads bypass the global
    cooldown gate — see ``_COOLDOWN_BYPASS_TIER_THRESHOLD``.  Without
    this bypass, a tier-1 L2/holon decision that arrives shortly
    after an unrelated gossip update is silently dropped; with it, the
    suppression is visible via a ``regulate_suppressed_by_cooldown``
    event whenever a non-critical update is gated.

    Returns ``True`` if the action was applied, ``False`` if suppressed
    (no behavior.act call, no diagnostics record).
    """
    factor = max(0.0, min(1.0, factor))

    _cfg = getattr(behavior, "_scare_config", None)

    # --- Heat curtailment-auction lock (heat sector only) -------------
    # While the community curtailment auction holds a heat load down for a
    # live temperature violation, it is the authoritative shedding lever:
    # L2 allocation writes DEFER (skip) rather than claw the load back up.
    # This breaks the cold-day limit cycle where the MW-based holon
    # re-dispatch restores a just-curtailed cold node, re-cools it below the
    # t_k floor, and the two layers oscillate.  The lock is set by auction
    # ("curtail") writes and lifted as ``heat_recovery`` ramps the recovered
    # load back to ~1.0; with recovery disabled it persists (shed-and-stay
    # for a permanent failure).  Strictly heat-scoped: other sectors and
    # unlocked heat loads fall through to the normal L2 path below.
    try:
        _sector_e = Sector(sector) if not isinstance(sector, Sector) else sector
    except ValueError:
        _sector_e = None
    if _sector_e is Sector.HEAT and getattr(
        _cfg, "enable_heat_curtail_lock", True
    ):
        _lock = _heat_curtail_lock_store(behavior)
        if reason == CURTAIL_AUCTION_REASON:
            # Only lock when the auction is actually holding the load BELOW
            # full service.  A no-op / near-1.0 curtail (tiny winning share)
            # carries no claim, and locking at ~1.0 would wrongly block the
            # holon from legitimately shedding the load for MW reasons.
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

    # --- L2 priority-floor reconciliation -----------------------------
    # The component-scope holon ADMM is authoritative on which tier gets
    # served; L1 must not undo it.  Record the floor on L2 writes; clamp
    # L1 reactive sheds (here: ``stability`` — gossip ``balance`` writes
    # bypass ``apply_regulate`` and are floored in
    # ``EnergyBalanceNegotiator._apply_setpoint``).  Applies to *all*
    # load tiers including tier 1: ``constraint_allowed_fraction`` is
    # tier-1-immune (returns 1.0), so tier-1's floor is simply its L2
    # allocation — this re-asserts the tier-1 hard-lock against
    # ``stability`` erosion (eval task-18 child-90 settling at 0.984)
    # while the curtailment auction (``reason="curtail"``, not clamped
    # here) can still shed tier-1 when a constraint physically demands
    # it.  Generators (priority_tier <= 0) are excluded.  Gated on the
    # config flag so the behaviour is A/B-able.
    if getattr(_cfg, "enable_l2_priority_floor", False):
        if reason in L2_ALLOCATION_REASONS:
            # Cap the holon allocation — both the applied factor and the
            # stored floor — by the load's local constraint-allowed
            # fraction.  The L2 ADMM decides priority on MW grounds and is
            # blind to per-node physics; without this cap a holon write
            # restores an out-of-bounds node (cold-day heat junction below
            # the t_k floor) to ~1.0 and the floor then clamps L1 reactive
            # temperature sheds back up, pinning the node at an infeasible
            # temperature.  ``constraint_allowed_fraction`` is tier-1-immune
            # (returns 1.0), so tier-1's allocation is unaffected — matching
            # ``l2_effective_floor``'s read-time cap, so the stored floor is
            # never above feasibility regardless of caller.
            try:
                _sector = (
                    Sector(sector) if not isinstance(sector, Sector) else sector
                )
            except ValueError:
                _sector = None
            # HEAT is exempt — the frontier controller owns its temperature
            # (and locks managed loads so this write already defers); capping
            # here would re-shed feasible heat loads on transient t_k dips
            # (A2 over-shed).  El/gas keep the cap.
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

    # Stale-observation detector (H13).  If the LP has not been
    # re-solved since the previous apply landed on this behavior,
    # this regulate is computing against the previous net_results
    # snapshot — the LP infeasibility cascade hides this otherwise.
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

    Preferred path: look up the (behavior, aid) pair in the sector
    registry populated at world-construction time.  The obs-key
    heuristic is retained only as a last-resort fallback — monee
    junction obs dicts are shape-identical between gas and water, so
    any inference from keys alone is unreliable.
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


# Re-export ``create_failures`` from :mod:`scare.base.failure_sampling`
# (the actual implementation) for backwards compatibility with existing
# callers that import it from this module.
from scare.base.failure_sampling import create_failures  # noqa: E402,F401


def efficiency_vector(eta_el: float, eta_heat: float, eta_gas: float) -> np.ndarray:
    return np.array([eta_el, eta_heat, eta_gas], dtype=float)


# Re-export the CP flex-actor factories from
# :mod:`scare.base.admm_factories` (the actual implementations) for
# backwards compatibility with existing callers.
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

# Keys in observation dicts that carry constraint-relevant quantities.
# These must match the keys returned by monee model.values, which are
# in per-unit / SI (Kelvin) — *not* in engineering units (bar, °C).
_CONSTRAINT_OBS_KEYS: dict[Sector, dict[str, str]] = {
    Sector.ELECTRICITY: {
        "vm_pu": "vm_pu",              # from Bus model
        "loading_percent": "loading_percent",  # from PowerLine model
    },
    Sector.GAS: {
        "pressure_pu": "pressure_pu",  # from Junction model
    },
    Sector.HEAT: {
        "t_k": "t_k",                  # from Junction model (Kelvin)
    },
}


def obs_constraint_values(obs: dict, sector: Sector) -> dict[str, float]:
    """Extract grid-constraint measurements from an observation dict.

    For ``loading_percent`` the underlying monee model exposes two
    variants: ``GenericPowerBranch`` reports it as a *fraction*
    (``i_from_ka / max_i_ka`` ∈ [0, 1]) while the
    ``IntermediateEq`` form in ``monee.model.core`` reports it as an
    actual percent (× 100).  ``SECTOR_CONSTRAINTS`` uses the percent
    convention, so we auto-scale the fraction form by 100×.  The
    discriminator is the magnitude: a value ≤ 5 cannot meaningfully
    represent a real loading-percent (even a 500 % overload would be
    catastrophic), so any value at that scale must be the fraction
    form and is multiplied up.

    The branch model exposes ``loading_from_percent`` /
    ``loading_to_percent`` as raw Vars but ``loading_percent`` is only
    a Python property — so it is *not* in ``model.values``.  Fall back
    to the max of the per-side Vars when the bare key is missing.
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

    monee observations do not carry a ``priority`` key, so this
    accessor is only meaningful when callers pre-populate priorities
    via :func:`experiment.restoration.assign_load_priorities` and
    pass them explicitly to the metric / role layer (see
    ``EnergyBalanceNegotiator._build_priorities``).  The fallback
    below returns tier 0 for generators (negative capacity) and tier 1
    for loads — a uniform-priority degenerate baseline.  Callers that
    require tier diversity should set ``priority_assignment`` in the
    scenario or feed an explicit priority dict.

    Slack agents are always classified as tier 0 (generator-class)
    regardless of the LP's current sign — the sign flips depending on
    import / export direction, but the role of the slack is always to
    supply / absorb at the network boundary, never to be shed.

    Pass ``record_default_fallback_t`` to surface a one-shot
    ``priority_default_fallback`` event the first time a given
    (behavior, aid) takes the tier-0/1 fallback branch.  Used by
    higher-level callers (e.g. ``_handle_ask_flex``) so missed priority
    registrations show up in events.csv rather than silently degrading
    the tier-stratified allocation.
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
    # Default unannotated loads to tier 4 (sheddable).  Under the 4-tier
    # model tier 1 is hard-locked at ``x = 1``, so falling back to tier
    # 1 would catastrophically over-assign critical priority to loads
    # whose actual priority was never registered — turning every smoke
    # grid without explicit ``priority_assignment`` into a perpetually
    # infeasible-trivial allocation.  Tier 4 is the conservative
    # default: missing annotations → lowest priority → first to shed.
    fallback = 0 if cap < 0 else 4
    if (
        record_default_fallback_t is not None
        and behavior is not None
        and aid is not None
        and cap > 0  # only loads — generators legitimately default to tier 0
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

    Starting from the highest-priority tier (lowest number), allocate
    proportionally to unserved demand within each tier until the budget
    is exhausted.  This guarantees that critical loads across all groups
    are served before any low-priority load receives resources.

    Returns a list of shares (one per group), summing to at most
    *total_available*.
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

    Higher-priority tiers contribute more weight per unit of unserved
    demand.  Used by the L3 CP S-coefficient to pull allocation toward
    sectors with high-priority unmet demand.

    Uses the strict-monotone tier schedule
    (:func:`tier_priority_weight_strict`) rather than the L1 QP
    schedule (:func:`tier_priority_weight`), because the L1 schedule
    returns 0 for tier 1 (hard-locked off-QP) — which would mask
    tier-1 unmet demand from this urgency aggregate.  The strict
    schedule keeps tier 1 ranking first while staying well-conditioned.
    """
    weight = 0.0
    for tier, demand in demand_by_priority.items():
        served = served_by_priority.get(tier, 0.0)
        unserved = max(0.0, demand - served)
        weight += unserved * tier_priority_weight_strict(int(tier))
    return weight


# Deadband threshold for ``clamp_to_constraints``: utilization must
# exceed this fraction of the feasible range before clamping kicks in.
# A 5%-bounded voltage envelope (±5% around 1.0 pu) means utilization
# values up to ~0.5 are everyday-normal operating drift, not a sign of
# stress.  The original linear-from-zero formula shed loads to 50 % of
# rated demand at vm_pu=1.025 pu (perfectly normal) — that's the source
# of the priority-inversion observed on simbench_lv (tier-1 critical
# load served at 65 % while tier-6 loads at 100 %, because tier-1
# happened to sit in a slightly higher-voltage neighbourhood).  The
# deadband restores the intent of "near-violation, throttle".
# 4-tier priority model with hard tier-1 enforcement.
#
# Tier 1 = critical: leader pre-applies ``regulation = 1`` before the
# gossip QP runs (see ``EnergyBalanceNegotiator._pre_apply_tier1_hard``).
# Tiers 2–4 = QP-weighted, with very steep exponents so the proportional
# QP equilibrium is effectively strict for any realistic deficit.
#
# Tier 1 carries a defensive QP weight of 1.0 — after the pre-step its
# δ-box collapses (``[-cap, 0]`` in restoration, ``[0, cap]`` in
# curtailment), so the QP's clamp pins δ=0 regardless of weight.
#
# Generators (priority ≤ 0) keep the legacy unit weight.
DEFAULT_PRIORITY_TIERS: int = 4

# Restoration (target > 0): higher-priority tiers get higher weight.
# Tier 1's weight is 0 — it's hard-locked at the leader pre-step, so it
# must not participate in the QP (a_i = 0 ⇒ δ = clamp(0·λ, …) = 0 if
# 0 ∈ box, which holds whenever the pre-step has already moved tier-1
# loads to their cap).  Its ledger entry also contributes nothing to
# the dual normaliser.
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

    Implements the 4-tier schedule with hard tier-1 enforcement off-QP:

    * ``regime > 0`` (restoration / lost-load): tier 2 → 1e8, tier 3 →
      1e4, tier 4 → 1.  Tier 1 is hard-locked at ``x = 1`` by the
      leader's pre-step and returns the defensive weight 1.0 here.
    * ``regime < 0`` (curtailment / lost-gen): tier 4 → 1e8 (sheds
      first), tier 3 → 1e4, tier 2 → 1.  Tier 1 returns 1.0 — it never
      sheds via the QP.
    * ``regime == 0``: 1.0 (no negotiation direction).

    The ``priority_tiers`` argument is preserved for API compatibility
    but the schedule is fixed to 4 tiers; tiers outside ``[1, 4]`` are
    clamped before lookup so legacy 10-tier callers degrade gracefully
    via the remap helper.
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

    The QP schedule (``tier_priority_weight``) returns a low weight for
    tier 1 because tier-1 is hard-locked at the L1 pre-step.  But L2's
    supply-priority waterfall sorts cells by weight to decide allocation
    order, and tier 1 must come first regardless.  This helper returns a
    schedule strictly decreasing in tier number (tier 1 → P, tier P →
    1), keeping the waterfall's sort-by-weight semantics intact while
    avoiding the wild magnitudes that would destabilise the ADMM
    sharing-distance objective.
    """
    P = max(1, int(priority_tiers))
    p = max(1, min(P, int(tier)))
    return float(P - p + 1)


def remap_legacy_priority(tier: int) -> int:
    """Map a legacy 10-tier value onto the new 4-tier schedule.

    Bucketing: ``{1, 2, 3} → 1``, ``{4, 5} → 2``, ``{6, 7} → 3``,
    ``{8, 9, 10} → 4``.  Out-of-range values are clamped.  Tier 0
    (generator class) passes through unchanged.
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


# Tier-aware deadbands for ``clamp_to_constraints``: the higher the
# tier's deadband, the closer to a hard bound a measurement must drift
# before the clamp throttles the load.  Tier 1 is fully immune (clamp
# no-ops, see ``clamp_to_constraints``) because the pre-step has hard-
# locked it at ``x = 1``; the clamp must not break that invariant.  The
# remaining tiers shade more aggressively as priority drops.
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

    Conservative-feasibility helper (improvements.txt §5): when a local
    grid measurement is approaching a hard bound, reduce the proposed
    setpoint to avoid actuating a violation.

    Activates only past a tier-dependent deadband — the default 0.85
    is the everyday LV operating drift threshold; tier 1/2 critical
    loads get a tighter 0.99 deadband so the clamp does not overrule
    the priority waterfall by truncating critical demand once the
    shared upstream variable (e.g. ``loading_percent``) drifts past
    0.85.  Above the deadband, the allowed fraction ramps linearly
    to zero:

        allowed = (1 - util) / (1 - DEADBAND)   for util ∈ [DEADBAND, 1]
        allowed = 1.0                            for util < DEADBAND

    Without the deadband, normal LV voltage variation (vm_pu=1.02-1.03)
    cuts every load to 50-70 % of cap — completely overriding the
    priority-aware gossip waterfall.  Confirmed root cause of the
    priority-invariant failure on task-0 simbench_lv.

    ``tier`` is the load's priority tier (1 = most critical, higher =
    less critical).  Tier 1 is immune to clamping — its pre-step lock
    at ``regulation = 1`` must not be overridden by a soft proximity
    signal; if a true ConstraintViolation fires, the next negotiation
    re-evaluates feasibility instead.  Tiers 2 / 3 / 4 get progressively
    tighter deadbands (0.95 / 0.90 / 0.85).  When ``None``, the legacy
    uniform 0.85 deadband is used — preserves byte-compatible behaviour
    for callers that do not (yet) propagate tier information.
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
    """Tightest constraint-allowed served fraction ``∈ [0, 1]`` from the
    local grid measurements, using the same tier-dependent deadband as
    :func:`clamp_to_constraints` (tier 1 immune → 1.0; tiers 2/3/4 use
    the ``_CLAMP_TIER_DEADBAND`` schedule).

    Returns the fraction of rated capacity the load may be served at
    *given local physics*, before the priority decision is considered.
    Shared by ``clamp_to_constraints`` (which applies it to a setpoint)
    and the L2 priority-floor (``l2_effective_floor``), so the floor
    relaxes by *exactly* the amount the clamp sheds — they can never
    fight over the same load (the eval task-72 cold-day regression,
    where a coarse violation flag let them disagree).
    """
    # Tier-1 is immune to the soft proximity clamp — its pre-step lock at
    # regulation=1 must not be overruled by a near-bound signal; a true
    # ConstraintViolation re-checks tier-1 feasibility instead.
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
