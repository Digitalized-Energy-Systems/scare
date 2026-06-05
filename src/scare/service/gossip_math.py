"""Pure numerical core of the gossip balance protocol.

Extracted from :class:`scare.service.balance.EnergyBalanceNegotiator`: the
primal-dual QP update, the diminishing step schedule, and the per-agent ledger
arithmetic (merge, total-δ, responsiveness sum). Side-effect-free functions over
scalars and the ledger dict, so they are unit-testable without a mango
role/context — matching the file's existing pure-helper convention
(``_deterministic_next`` etc.).

Ledger entry schema, keyed by agent address-string::

    (delta, counter_when_set, priority, saturated_flag)
"""

from __future__ import annotations

from scare.base.util import tier_priority_weight


def qp_priority_weight(priority: int, target_sign: int, *, priority_tiers: int) -> float:
    """Priority cost weight ``a_i`` for the QP responsiveness. Single source of
    truth is ``tier_priority_weight``: tiers 2-4 get 1e8 / 1e4 / 1.0, tier 1 a
    defensive 1.0 (hard-locked upstream), generators 1.0."""
    return tier_priority_weight(
        priority, regime=int(target_sign), priority_tiers=priority_tiers,
    )


def qp_primal(a_i: float, lam: float, dmin: float, dmax: float) -> float:
    """Closed-form primal update ``δ_i = clamp(a_i · λ, dmin, dmax)``.

    sign(λ) tracks sign(T): restoration (T>0) raises λ, pushing δ positive
    (loads up, generators shed); curtailment (T<0) drives λ negative. Box
    clamping enforces feasibility.
    """
    return max(dmin, min(dmax, a_i * lam))


def compute_lambda_seed(
    target: float, n_neighbours: int, *, priority: int, priority_tiers: int,
) -> float:
    """Seed λ so the originator's first δ aims at its fair share
    ``target / n_seed`` (``λ₀ = target / (n_seed · a_self)``). Clamped to
    ``|target|`` so pathological tier combinations can't inject an unbounded
    first step."""
    target_sign = 1 if target > 0 else (-1 if target < 0 else 0)
    a_self = max(qp_priority_weight(priority, target_sign, priority_tiers=priority_tiers), 1.0)
    n_seed = max(2, n_neighbours + 1)
    lambda_seed = target / (n_seed * a_self)
    return max(-abs(target), min(abs(target), lambda_seed))


def entry_responsiveness(prio: int, target_sign: int, *, priority_tiers: int) -> float:
    """``a_i`` from a ledger entry's stored priority, used to estimate
    ``Σ a_j`` for dual-step normalisation. Mirrors :func:`qp_priority_weight`
    so the dual update agrees with each agent's own primal step."""
    return tier_priority_weight(
        int(prio), regime=int(target_sign), priority_tiers=priority_tiers,
    )


def step_size(convergence_rate: float, counter: int, *, step_decay_k0: float) -> float:
    """Robbins-Monro diminishing step (P3): ``γ_s / (1 + k / k0)``. Satisfies
    ``Σ γ_k = ∞`` and ``Σ γ_k² < ∞`` so the dynamics converge a.s. under
    bounded-variance noise. At ``counter = 0`` equals the constant step."""
    return convergence_rate / (1.0 + max(0, counter) / step_decay_k0)


def ledger_total_delta(memory: dict[str, tuple]) -> float:
    """``Σ δ_i`` across all participants in the ledger."""
    return sum(v[0] for v in memory.values())


def ledger_merge(
    memory: dict[str, tuple], incoming: dict[str, tuple], *, byzantine_cap: float,
) -> None:
    """Merge ``incoming`` ledger entries into ``memory`` in place, keeping the
    newest-counter entry per agent (avoids the double-counting an aggregate
    digest suffers in cyclic graphs). Each δ is clipped to ``±byzantine_cap`` so
    one misbehaving agent can't corrupt the group total. Tolerates 3-tuple
    legacy entries (``saturated`` defaults False)."""
    for k, v in incoming.items():
        local = memory.get(k)
        if local is None or local[1] < v[1]:
            if len(v) >= 4:
                delta, ctr, prio, sat = v[0], v[1], v[2], bool(v[3])
            else:
                delta, ctr, prio = v[0], v[1], v[2]
                sat = False
            if delta > byzantine_cap or delta < -byzantine_cap:
                delta = max(-byzantine_cap, min(byzantine_cap, delta))
            memory[k] = (delta, ctr, prio, sat)


def ledger_sum_responsiveness(
    memory: dict[str, tuple], target_sign: int, *, priority_tiers: int,
) -> float:
    """``Σ a_j`` over *unsaturated* ledger entries (the dual-step denominator).

    Saturated agents sit at a box bound and add no δ for further λ, so counting
    them inflates the denominator and slows the agents that can still move.
    Falls back to all entries (then 1.0) so the denominator is never zero.
    """
    sum_a = sum(
        entry_responsiveness(int(v[2]), target_sign, priority_tiers=priority_tiers)
        for v in memory.values()
        if not v[3]
    )
    if sum_a <= 0.0:
        sum_a = sum(
            entry_responsiveness(int(v[2]), target_sign, priority_tiers=priority_tiers)
            for v in memory.values()
        ) or 1.0
    return sum_a
