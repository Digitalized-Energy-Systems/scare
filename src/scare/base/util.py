from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from mango_energy_environments import Failure

from scare.base.model import Sector

HHV: float = 15.3  # MW / (kg/s) for natural gas

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


def _sector_store(behavior: Any) -> dict[str, Sector]:
    store = getattr(behavior, "_scare_sectors", None)
    if store is None:
        store = {}
        behavior._scare_sectors = store
    return store


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
    """
    cap: float          # generator-convention rated output, < 0
    dmin_abs: float     # min absolute p_mw the Var can take
    dmax_abs: float     # max absolute p_mw the Var can take


def _slack_store(behavior: Any) -> dict[str, "_SlackMeta"]:
    store = getattr(behavior, "_scare_slacks", None)
    if store is None:
        store = {}
        behavior._scare_slacks = store
    return store


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
    """
    if rating_mw <= 0.0:
        # Silent no-op here would leave the slack child unregistered,
        # which downstream ``obs_capacity`` / ``obs_priority`` falls
        # back on the LP's current operating value — i.e. the slack
        # gets reclassified as a load.  Surface the bad input instead.
        import logging
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


def _last_regulate_store(behavior: Any) -> dict[str, float]:
    store = getattr(behavior, "_scare_last_regulate", None)
    if store is None:
        store = {}
        behavior._scare_last_regulate = store
    return store


def _last_regulate_t_store(behavior: Any) -> dict[str, float]:
    """Per-aid timestamp of the last applied regulate, used by the
    sim-time cooldown gate.  Lives on the behavior so it tears down
    with the simulation world.
    """
    store = getattr(behavior, "_scare_last_regulate_t", None)
    if store is None:
        store = {}
        behavior._scare_last_regulate_t = store
    return store


def apply_regulate(
    behavior: Any,
    aid: str,
    factor: float,
    *,
    sector: str,
    reason: str,
    timestamp: float,
    tolerance: float = _REGULATE_DEDUP_TOL,
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

    Returns ``True`` if the action was applied, ``False`` if suppressed
    (no behavior.act call, no diagnostics record).
    """
    factor = max(0.0, min(1.0, factor))
    last = _last_regulate_store(behavior).get(aid)
    if last is not None and abs(factor - last) < tolerance:
        return False
    cfg = getattr(behavior, "_scare_config", None)
    cooldown_s = getattr(cfg, "cooldown_s", 0.0) if cfg is not None else 0.0
    if cooldown_s > 0:
        last_t_store = _last_regulate_t_store(behavior)
        last_t = last_t_store.get(aid)
        if last_t is not None and (timestamp - last_t) < cooldown_s:
            return False
    if not behavior.has_action(aid, "regulate"):
        return False

    behavior.act(aid, "regulate", factor)
    _last_regulate_store(behavior)[aid] = factor
    if cooldown_s > 0:
        _last_regulate_t_store(behavior)[aid] = timestamp

    from scare.base.diagnostics import record_regulate

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


def create_failures(
    monee_net: Any,
    failure_type: str = "branch",
    *,
    num_failures: int = 1,
    delay_s_max: float = 5.0,
    generator_share: float = 0.5,
) -> list[Failure]:
    """Sample ``num_failures`` failure events on the network.

    ``failure_type`` selects the population of contingencies:

    - ``"branch"``  — non-CP branches only (lines, pipes).  Original
      behaviour; matches the simbench-LV stress tests historically used.
    - ``"generator"`` — generator-class components only.  Covers
      ``PowerGenerator`` / ``HeatGenerator`` / ``Source`` (childs),
      ``CHP`` / ``PowerToHeat`` / ``CHPHG`` (compounds), and
      ``GasToPower`` / ``PowerToGas`` / ``PowerToHeatHG`` (branches).
      Each deactivation goes through ``net.deactivate(component)`` via
      the ``Failure.custom`` hook.
    - ``"mixed"`` — random mix; ``generator_share`` controls what
      fraction of the draw is generators (rest are non-CP branches).
      Default 50 / 50.
    - ``"concentrated"`` — pick a random node with $\ge 2$ incident
      branches, then sample up to ``num_failures`` of *its* incident
      branches.  Concentrates the impact on one locale, producing the
      stark per-group capacity asymmetry that the holonic ADMM layer
      exists to resolve: the affected sub-community sees a heavy local
      deficit while adjacent groups still have slack capacity.
    """
    if failure_type == "branch":
        return _sample_branch_failures(monee_net, num_failures, delay_s_max)
    if failure_type == "generator":
        return _sample_generator_failures(monee_net, num_failures, delay_s_max)
    if failure_type == "mixed":
        n_gen = int(round(num_failures * max(0.0, min(1.0, generator_share))))
        n_branch = num_failures - n_gen
        out = _sample_branch_failures(monee_net, n_branch, delay_s_max)
        out += _sample_generator_failures(monee_net, n_gen, delay_s_max)
        return out
    if failure_type == "concentrated":
        return _sample_concentrated_failures(monee_net, num_failures, delay_s_max)
    return []


def _sample_concentrated_failures(
    monee_net: Any, num_failures: int, delay_s_max: float
) -> list[Failure]:
    """Pick a load-rich node and disconnect up to ``num_failures``
    branches in its 1-hop neighbourhood.  Concentration target chosen
    to maximise lost served capacity: among nodes whose 1-hop subtree
    contains the most load demand, pick one and cut its incident
    non-CP branches plus the cuts that strand the largest descendants.

    The picker prefers branches whose removal disconnects load-bearing
    nodes from the rest of the network so the failure creates a
    measurable per-group deficit.  This is the regime where the
    holonic ADMM has work to do — without it, the affected
    sub-community sees a stark deficit while neighbouring groups
    retain slack.
    """
    candidates = [b for b in monee_net.branches if not b.model.is_cp()]
    if not candidates:
        return []
    # Build node → incident-branches map and load-capacity-per-node
    by_node: dict = {}
    for b in candidates:
        for nid in (b.id[0], b.id[1]):
            by_node.setdefault(nid, []).append(b)
    load_at_node: dict = {}
    for child in monee_net.childs:
        nid = child.node_id
        obs = dict(child.model.values)
        cap = obs_capacity(obs)
        if cap > 0:
            load_at_node[nid] = load_at_node.get(nid, 0.0) + cap
    # 1-hop "neighbourhood load" = load at this node + load at neighbours
    nbr_load: dict = {}
    for n, branches in by_node.items():
        score = load_at_node.get(n, 0.0)
        for b in branches:
            other = b.id[1] if b.id[0] == n else b.id[0]
            score += load_at_node.get(other, 0.0)
        if score > 0 and len(branches) >= 1:
            nbr_load[n] = score
    if not nbr_load:
        # Fall back to original behaviour: any branched node.
        eligible = [n for n, bs in by_node.items() if len(bs) >= 2]
        if not eligible:
            eligible = list(by_node.keys())
        target_node = random.choice(eligible)
    else:
        # Pick from top-25 % by neighbourhood load to keep some
        # seed-variance while staying in load-rich territory.
        ranked = sorted(nbr_load.items(), key=lambda kv: -kv[1])
        top_k = max(1, len(ranked) // 4)
        target_node = random.choice([n for n, _ in ranked[:top_k]])
    # BFS outward from the target node, accumulating branches in a
    # cluster until we have ``num_failures``.  Each frontier expansion
    # adds the branches incident to the new nodes, keeping the failure
    # set spatially concentrated rather than spread across the grid.
    selected: list = []
    selected_ids: set = set()
    visited: set = {target_node}
    frontier: list = [target_node]
    while frontier and len(selected) < num_failures:
        # Sort frontier nodes by load impact on their incident branches
        next_frontier: list = []
        for node in frontier:
            incident = sorted(
                by_node.get(node, []),
                key=lambda b: -load_at_node.get(
                    b.id[1] if b.id[0] == node else b.id[0], 0.0
                ),
            )
            for b in incident:
                if id(b) in selected_ids:
                    continue
                if len(selected) >= num_failures:
                    break
                selected.append(b)
                selected_ids.add(id(b))
                other = b.id[1] if b.id[0] == node else b.id[0]
                if other not in visited:
                    visited.add(other)
                    next_frontier.append(other)
        frontier = next_frontier
    return [
        Failure(delay_s=random.uniform(0.0, delay_s_max), branch_ids=[b.id])
        for b in selected
    ]


def _sample_branch_failures(
    monee_net: Any, num_failures: int, delay_s_max: float
) -> list[Failure]:
    candidates = [b for b in monee_net.branches if not b.model.is_cp()]
    selected = random.sample(candidates, min(num_failures, len(candidates)))
    return [
        Failure(delay_s=random.uniform(0.0, delay_s_max), branch_ids=[b.id])
        for b in selected
    ]


def _sample_generator_failures(
    monee_net: Any, num_failures: int, delay_s_max: float
) -> list[Failure]:
    """Pick generator-class components and wrap each in a ``Failure``
    whose ``custom`` callable deactivates the component on the live
    network.  The component is captured by closure so the callable
    works even after the failure has been dispatched.
    """
    candidates = list(_iter_generator_candidates(monee_net))
    if not candidates:
        return []
    selected = random.sample(candidates, min(num_failures, len(candidates)))
    out: list[Failure] = []
    for kind, component in selected:
        # Branch-class CP plants (P2G / G2P / P2H-HG) can use the
        # built-in ``branch_ids`` deactivation pathway directly — that's
        # cheaper and surfaces the same BranchFailureEvent the rest of
        # scare's logic listens for.
        if kind == "branch":
            out.append(Failure(
                delay_s=random.uniform(0.0, delay_s_max),
                branch_ids=[component.id],
            ))
            continue
        # Childs / compounds need ``custom`` because the Failure dataclass
        # only natively handles branches and nodes.
        comp = component  # bind for closure

        def _deactivate(net, _c=comp) -> None:
            net.deactivate(_c)

        out.append(Failure(
            delay_s=random.uniform(0.0, delay_s_max),
            custom=_deactivate,
            custom_id=f"{kind}:{getattr(component, 'id', repr(component))}",
        ))
    return out


def _iter_generator_candidates(monee_net: Any):
    """Yield ``(kind, component)`` pairs for every deactivatable
    generator-class component on the network.

    ``kind`` ∈ {``"child"``, ``"compound"``, ``"branch"``} drives the
    failure dispatch rule in ``_sample_generator_failures``.
    """
    from monee.model.child import HeatGenerator, PowerGenerator, Source
    from monee.model.multi import (
        CHP,
        CHPHG,
        GasToPower,
        PowerToGas,
        PowerToHeat,
        PowerToHeatHG,
    )

    child_classes = (PowerGenerator, HeatGenerator, Source)
    compound_classes = (CHP, CHPHG, PowerToHeat)
    branch_classes = (GasToPower, PowerToGas, PowerToHeatHG)

    for child in monee_net.childs:
        if isinstance(child.model, child_classes):
            yield ("child", child)
    for compound in getattr(monee_net, "compounds", []):
        if isinstance(compound.model, compound_classes):
            yield ("compound", compound)
    for branch in monee_net.branches:
        if isinstance(branch.model, branch_classes):
            yield ("branch", branch)


def efficiency_vector(eta_el: float, eta_heat: float, eta_gas: float) -> np.ndarray:
    return np.array([eta_el, eta_heat, eta_gas], dtype=float)


def create_chp_admm_flex_actor(chp_obs: dict, priority: int):
    """CHP: produces electricity + heat from gas."""
    from distributed_resource_optimization import create_admm_flex_actor_one_to_many

    cap = kgps_to_mw(float(chp_obs.get("gas_kgps", obs_capacity(chp_obs))))
    eta = efficiency_vector(
        chp_obs.get("eta_el", 0.35), chp_obs.get("eta_heat", 0.45), -1.0
    )
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def create_p2g_admm_flex_actor(p2g_obs: dict, priority: int):
    """P2G: consumes electricity, produces gas."""
    from distributed_resource_optimization import create_admm_flex_actor_one_to_many

    cap = float(p2g_obs.get("el_mw", obs_capacity(p2g_obs)))
    eta = efficiency_vector(-1.0, 0.0, p2g_obs.get("eta_gas", 0.6))
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def create_g2p_admm_flex_actor(g2p_obs: dict, priority: int):
    """G2P: consumes gas, produces electricity."""
    from distributed_resource_optimization import create_admm_flex_actor_one_to_many

    cap = kgps_to_mw(float(g2p_obs.get("gas_kgps", obs_capacity(g2p_obs))))
    eta = efficiency_vector(g2p_obs.get("eta_el", 0.45), 0.0, -1.0)
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


def create_p2h_admm_flex_actor(p2h_obs: dict, priority: int):
    """P2H: consumes electricity, produces heat (high-grade or low-grade)."""
    from distributed_resource_optimization import create_admm_flex_actor_one_to_many

    cap = float(p2h_obs.get("el_mw", obs_capacity(p2h_obs)))
    eta = efficiency_vector(-1.0, p2h_obs.get("eta_heat", 0.9), 0.0)
    return create_admm_flex_actor_one_to_many(cap, eta, np.full(3, float(priority)))


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
    """
    if behavior is not None and aid is not None:
        if lookup_slack(behavior, aid) is not None:
            return 0
    if "priority" in obs:
        return int(obs["priority"])
    cap = obs_capacity(obs)
    return 0 if cap < 0 else 1


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

    Higher-priority tiers contribute exponentially more weight per unit of
    unserved demand.  This is used as the ADMM S parameter to pull
    allocation toward groups with critical unserved loads.
    """
    weight = 0.0
    for tier, demand in demand_by_priority.items():
        served = served_by_priority.get(tier, 0.0)
        unserved = max(0.0, demand - served)
        # Exponential weighting: tier 1 → 2^9=512, tier 10 → 2^0=1
        tier_weight = 2.0 ** max(0, 10 - tier)
        weight += unserved * tier_weight
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
_CLAMP_UTILIZATION_DEADBAND: float = 0.85


def clamp_to_constraints(
    setpoint: float,
    obs: dict,
    sector: Sector,
) -> float:
    """Clamp a proposed setpoint so it stays within local constraint bounds.

    Conservative-feasibility helper (improvements.txt §5): when a local
    grid measurement is approaching a hard bound, reduce the proposed
    setpoint to avoid actuating a violation.

    Activates only past the ``_CLAMP_UTILIZATION_DEADBAND`` — utilization
    levels below that represent normal operating drift inside the
    designed ±5 % LV envelope, not stress.  Above the deadband, the
    allowed fraction ramps linearly to zero:

        allowed = (1 - util) / (1 - DEADBAND)   for util ∈ [DEADBAND, 1]
        allowed = 1.0                            for util < DEADBAND

    Without the deadband, normal LV voltage variation (vm_pu=1.02-1.03)
    cuts every load to 50-70 % of cap — completely overriding the
    priority-aware gossip waterfall.  Confirmed root cause of the
    priority-invariant failure on task-0 simbench_lv.
    """
    from scare.base.model import SECTOR_CONSTRAINTS

    bounds = SECTOR_CONSTRAINTS.get(sector, {})
    cap = obs_capacity(obs)
    if cap == 0.0:
        return setpoint

    deadband = _CLAMP_UTILIZATION_DEADBAND
    width = max(1e-9, 1.0 - deadband)

    # Determine the tightest constraint across all local variables.
    tightest_fraction = 1.0
    for var, (lo, hi) in bounds.items():
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

    if tightest_fraction < 1.0:
        max_abs = tightest_fraction * abs(cap)
        setpoint = max(-max_abs, min(max_abs, setpoint))

    return setpoint
