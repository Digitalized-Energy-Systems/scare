"""Unit tests for GridFormerPolicy — the single authority for the promoted-island
grid-former reference policy extracted from EnergyBalanceNegotiator.

Characterizes the exact behavior the negotiator delegated to it: guard-gated
former identity and the delivered+probe supply credit.
"""

from __future__ import annotations

from types import SimpleNamespace

from scare.base.util import register_grid_former_rating
from scare.service.balance.grid_former import GridFormerPolicy


class _Behavior:
    """Minimal behavior stand-in supporting attribute storage (the rating
    registry stashes ``_scare_grid_former_ratings`` on the behavior)."""


def _behavior(guard: bool) -> _Behavior:
    b = _Behavior()
    b._scare_config = SimpleNamespace(enable_grid_former_curtail_guard=guard)
    return b


def test_is_former_false_when_guard_off():
    b = _behavior(False)
    register_grid_former_rating(b, "gf-1", 5.0)
    policy = GridFormerPolicy(b, probe_share=0.0)
    assert policy.is_former("gf-1") is False


def test_is_former_true_when_guard_on_and_registered():
    b = _behavior(True)
    register_grid_former_rating(b, "gf-1", 5.0)
    policy = GridFormerPolicy(b, probe_share=0.0)
    assert policy.is_former("gf-1") is True


def test_is_former_false_for_unregistered_aid():
    b = _behavior(True)
    policy = GridFormerPolicy(b, probe_share=0.0)
    assert policy.is_former("load-2") is False


def test_is_former_reads_flag_lazily():
    # The guard flag is read on every call, never snapshotted at construction.
    b = _behavior(False)
    register_grid_former_rating(b, "gf-1", 5.0)
    policy = GridFormerPolicy(b, probe_share=0.0)
    assert policy.is_former("gf-1") is False
    b._scare_config.enable_grid_former_curtail_guard = True
    assert policy.is_former("gf-1") is True


def test_supply_credit_no_rating_is_delivered():
    policy = GridFormerPolicy(_behavior(True), probe_share=0.5)
    assert policy.supply_credit("x", -3.0) == 3.0  # delivered = max(0, -sp)
    assert policy.supply_credit("x", 2.0) == 0.0  # positive sp -> no injection


def test_supply_credit_with_rating_and_probe_share():
    b = _behavior(True)
    register_grid_former_rating(b, "gf-1", 10.0)
    policy = GridFormerPolicy(b, probe_share=0.5)
    # delivered = 4; credit = 4 + 0.5*(10-4) = 7.0
    assert policy.supply_credit("gf-1", -4.0) == 7.0


def test_supply_credit_probe_zero_matches_delivered():
    b = _behavior(True)
    register_grid_former_rating(b, "gf-1", 10.0)
    policy = GridFormerPolicy(b, probe_share=0.0)  # the shipped default
    assert policy.supply_credit("gf-1", -4.0) == 4.0
