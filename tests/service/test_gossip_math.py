"""Unit tests for the pure gossip math extracted from EnergyBalanceNegotiator."""

from __future__ import annotations

from scare.service.balance.gossip_math import (
    compute_lambda_seed,
    entry_responsiveness,
    ledger_merge,
    ledger_sum_responsiveness,
    ledger_total_delta,
    qp_primal,
    qp_priority_weight,
    step_size,
)

_TIERS = 4


# --------------------------------------------------------------------------- #
# qp_primal — closed-form primal update
# --------------------------------------------------------------------------- #


def test_qp_primal_clamps_to_box():
    assert qp_primal(2.0, 3.0, -1.0, 1.0) == 1.0  # a_i*lam=6 -> dmax
    assert qp_primal(1.0, -5.0, -2.0, 2.0) == -2.0  # -> dmin
    assert qp_primal(2.0, 0.1, -1.0, 1.0) == 0.2  # interior


# --------------------------------------------------------------------------- #
# qp_priority_weight / entry_responsiveness
# --------------------------------------------------------------------------- #


def test_priority_weight_matches_entry_responsiveness():
    # The receiver's Σa estimate must agree with each agent's own primal a_i.
    for prio in (1, 2, 3, 4):
        for sign in (-1, 1):
            assert qp_priority_weight(
                prio, sign, priority_tiers=_TIERS
            ) == entry_responsiveness(prio, sign, priority_tiers=_TIERS)


def test_priority_weight_curtailment_ordering():
    # Curtailment regime (sign=-1): lower-priority (higher tier number) sheds
    # more, so its weight dominates.
    w2 = qp_priority_weight(2, -1, priority_tiers=_TIERS)
    w3 = qp_priority_weight(3, -1, priority_tiers=_TIERS)
    w4 = qp_priority_weight(4, -1, priority_tiers=_TIERS)
    assert w4 > w3 > w2


# --------------------------------------------------------------------------- #
# step_size — Robbins-Monro schedule
# --------------------------------------------------------------------------- #


def test_step_size_decays():
    assert step_size(0.6, 0, step_decay_k0=20) == 0.6  # k=0 -> constant
    assert step_size(0.6, 20, step_decay_k0=20) == 0.3  # k=k0 -> half
    seq = [step_size(0.6, k, step_decay_k0=20) for k in range(0, 40, 5)]
    assert all(a > b for a, b in zip(seq, seq[1:]))  # strictly decreasing


# --------------------------------------------------------------------------- #
# compute_lambda_seed
# --------------------------------------------------------------------------- #


def test_lambda_seed_sign_and_clamp():
    pos = compute_lambda_seed(10.0, 4, priority=2, priority_tiers=_TIERS)
    assert 0.0 < pos <= 10.0
    neg = compute_lambda_seed(-10.0, 4, priority=2, priority_tiers=_TIERS)
    assert -10.0 <= neg < 0.0
    assert compute_lambda_seed(0.0, 4, priority=2, priority_tiers=_TIERS) == 0.0


def test_lambda_seed_fair_share_direction():
    # Larger group -> smaller per-agent seed (target / (n_seed * a_self)).
    small = compute_lambda_seed(10.0, 1, priority=2, priority_tiers=_TIERS)
    large = compute_lambda_seed(10.0, 9, priority=2, priority_tiers=_TIERS)
    assert small > large


# --------------------------------------------------------------------------- #
# ledger arithmetic
# --------------------------------------------------------------------------- #


def test_ledger_total_delta():
    mem = {"a": (1.5, 3, 2, False), "b": (-0.5, 2, 3, True)}
    assert ledger_total_delta(mem) == 1.0
    assert ledger_total_delta({}) == 0.0


def test_ledger_merge_keeps_highest_counter():
    mem = {"a": (1.0, 5, 2, False)}
    # Stale incoming (counter 3 < 5) is ignored.
    ledger_merge(mem, {"a": (2.0, 3, 2, False)}, byzantine_cap=100.0)
    assert mem["a"] == (1.0, 5, 2, False)
    # Newer incoming (counter 6 > 5) wins.
    ledger_merge(mem, {"a": (2.0, 6, 2, True)}, byzantine_cap=100.0)
    assert mem["a"] == (2.0, 6, 2, True)


def test_ledger_merge_byzantine_clip():
    mem: dict[str, tuple] = {}
    ledger_merge(mem, {"big": (9e9, 1, 3, False)}, byzantine_cap=5.0)
    ledger_merge(mem, {"neg": (-9e9, 1, 3, False)}, byzantine_cap=5.0)
    assert mem["big"][0] == 5.0
    assert mem["neg"][0] == -5.0


def test_ledger_merge_tolerates_legacy_3tuple():
    mem: dict[str, tuple] = {}
    ledger_merge(mem, {"c": (0.5, 2, 4)}, byzantine_cap=100.0)
    assert mem["c"] == (0.5, 2, 4, False)  # saturated defaults False


def test_ledger_sum_responsiveness_excludes_saturated():
    one = entry_responsiveness(2, 1, priority_tiers=_TIERS)
    mem = {
        "free1": (0.1, 1, 2, False),
        "free2": (0.1, 1, 2, False),
        "sat": (0.5, 1, 2, True),  # saturated -> excluded
    }
    got = ledger_sum_responsiveness(mem, 1, priority_tiers=_TIERS)
    assert abs(got - 2 * one) < 1e-9


def test_ledger_sum_responsiveness_fallbacks():
    # All saturated -> fall back to all entries (non-zero).
    mem = {"a": (0.5, 1, 2, True), "b": (0.5, 1, 2, True)}
    assert ledger_sum_responsiveness(mem, 1, priority_tiers=_TIERS) > 0.0
    # Empty ledger -> never zero (the `or 1.0` guard).
    assert ledger_sum_responsiveness({}, 1, priority_tiers=_TIERS) == 1.0
