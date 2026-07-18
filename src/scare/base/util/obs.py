"""Observation accessors and the identity registries (sector / slack / priority)
they depend on.

Self-contained LEAF: imports only ``model`` + ``diagnostics``, so the rest of
``util`` (and the blackboard control registries) import ``obs_*`` / ``lookup_*``
from here one-way, with no import cycle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from scare.base.model import Sector
from scare.base.runtime.diagnostics import record_event

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


def _priority_store(behavior: Any) -> dict[str, int]:
    return _get_behavior_store(behavior, "_scare_priorities")


def register_priority(behavior: Any, aid: str, tier: int) -> None:
    """Record an agent's priority tier so callers that don't own the role
    can look it up; without it ``obs_priority`` falls back to uniform
    priorities. Tier 0 is reserved for generators and slacks."""
    _priority_store(behavior)[aid] = int(tier)


def lookup_priority(behavior: Any, aid: str) -> int | None:
    return _priority_store(behavior).get(aid)


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
