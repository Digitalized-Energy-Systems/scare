"""Truth-table tests for the two named CP-leader predicates extracted from the
scattered ``char != 'leader' ...`` gates in EnergyConverterRole.

- ``_is_cps_cluster_leader`` gates the leader-ONLY sites (must stay leader-scoped
  even under L3).
- ``_acts_as_cp_leader`` gates the full-form sites (leader OR L3-wired).

Both are pure blackboard reads, so the role is built via ``object.__new__`` with
only the two dependencies stubbed (the topology-characteristic module function
and ``self._component``).
"""

from __future__ import annotations

from types import SimpleNamespace

import scare.service.coupling.cp as cp_mod
from scare.service.coupling.cp import EnergyConverterRole


def _role(char: str, l3_enabled: bool, monkeypatch) -> EnergyConverterRole:
    role = object.__new__(EnergyConverterRole)
    role._component = SimpleNamespace(enabled=lambda: l3_enabled)
    monkeypatch.setattr(cp_mod, "topology_characteristic", lambda self, tid: char)
    return role


def test_is_cps_cluster_leader(monkeypatch):
    assert _role("leader", False, monkeypatch)._is_cps_cluster_leader() is True
    assert _role("member", True, monkeypatch)._is_cps_cluster_leader() is False


def test_acts_as_cp_leader_truth_table(monkeypatch):
    # leader -> True regardless of L3
    assert _role("leader", False, monkeypatch)._acts_as_cp_leader() is True
    # non-leader + L3 wired -> True (the coordinator is often not the leader)
    assert _role("member", True, monkeypatch)._acts_as_cp_leader() is True
    # non-leader + no L3 -> False (legacy per-CP path is leader-only)
    assert _role("member", False, monkeypatch)._acts_as_cp_leader() is False
