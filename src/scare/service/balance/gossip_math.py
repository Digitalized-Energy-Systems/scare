"""Pure numerical core of the gossip balance protocol.

Side-effect-free primal-dual QP update, diminishing step schedule, and ledger
arithmetic (merge, total-δ, responsiveness sum) — unit-testable without a mango
context.

Ledger entry schema, keyed by agent address-string::

    (delta, counter_when_set, priority, saturated_flag)
"""

from __future__ import annotations

from scare.base.util import tier_priority_weight


def qp_priority_weight(
    priority: int, target_sign: int, *, priority_tiers: int
) -> float:
    """Priority cost weight ``a_i`` for QP responsiveness; delegates to
    ``tier_priority_weight`` (the single source of truth)."""
    return tier_priority_weight(
        priority,
        regime=int(target_sign),
        priority_tiers=priority_tiers,
    )


def qp_primal(a_i: float, lam: float, dmin: float, dmax: float) -> float:
    """Closed-form primal update ``δ_i = clamp(a_i · λ, dmin, dmax)``.

    sign(λ) tracks sign(T): T>0 (restoration) raises λ, pushing δ positive;
    T<0 (curtailment) drives it negative. Box clamping enforces feasibility.
    """
    return max(dmin, min(dmax, a_i * lam))


def compute_lambda_seed(
    target: float,
    n_neighbours: int,
    *,
    priority: int,
    priority_tiers: int,
) -> float:
    """Seed λ so the originator's first δ aims at its fair share
    ``λ₀ = target / (n_seed · a_self)``, clamped to ``|target|`` to bound the
    first step."""
    target_sign = 1 if target > 0 else (-1 if target < 0 else 0)
    a_self = max(
        qp_priority_weight(priority, target_sign, priority_tiers=priority_tiers), 1.0
    )
    n_seed = max(2, n_neighbours + 1)
    lambda_seed = target / (n_seed * a_self)
    return max(-abs(target), min(abs(target), lambda_seed))


def entry_responsiveness(prio: int, target_sign: int, *, priority_tiers: int) -> float:
    """``a_i`` from a ledger entry's stored priority, for ``Σ a_j`` dual-step
    normalisation. Mirrors :func:`qp_priority_weight` so dual and primal agree."""
    return tier_priority_weight(
        int(prio),
        regime=int(target_sign),
        priority_tiers=priority_tiers,
    )


def step_size(convergence_rate: float, counter: int, *, step_decay_k0: float) -> float:
    """Robbins-Monro diminishing step ``γ_s / (1 + k / k0)``: satisfies
    ``Σ γ_k = ∞``, ``Σ γ_k² < ∞`` for a.s. convergence. Constant at ``counter=0``."""
    return convergence_rate / (1.0 + max(0, counter) / step_decay_k0)


def ledger_total_delta(memory: dict[str, tuple]) -> float:
    """``Σ δ_i`` across all participants in the ledger."""
    return sum(v[0] for v in memory.values())


def ledger_merge(
    memory: dict[str, tuple],
    incoming: dict[str, tuple],
    *,
    byzantine_cap: float,
) -> None:
    """Merge ``incoming`` into ``memory`` in place, keeping the newest-counter
    entry per agent (avoids double-counting in cyclic graphs). Each δ is clipped
    to ``±byzantine_cap``. Tolerates 3-tuple legacy entries (``saturated`` False)."""
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
    memory: dict[str, tuple],
    target_sign: int,
    *,
    priority_tiers: int,
) -> float:
    """``Σ a_j`` over *unsaturated* ledger entries (dual-step denominator).

    Saturated agents are at a box bound and add no δ, so counting them would
    inflate the denominator. Falls back to all entries (then 1.0) to avoid zero.
    """
    sum_a = sum(
        entry_responsiveness(int(v[2]), target_sign, priority_tiers=priority_tiers)
        for v in memory.values()
        if not v[3]
    )
    if sum_a <= 0.0:
        sum_a = (
            sum(
                entry_responsiveness(
                    int(v[2]), target_sign, priority_tiers=priority_tiers
                )
                for v in memory.values()
            )
            or 1.0
        )
    return sum_a
