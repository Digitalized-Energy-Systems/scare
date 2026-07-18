"""Lock for first_role / role_index (replace the repeated O(agents) first-role
isinstance scans in the restoration wiring)."""

from __future__ import annotations

from types import SimpleNamespace

from scare.base.util import first_role, role_index


class RoleA:
    pass


class RoleB:
    pass


def _agent(aid, *roles):
    return SimpleNamespace(aid=aid, roles=list(roles))


def test_first_role_returns_first_match_or_none():
    a = _agent("x", RoleB(), RoleA(), RoleA())
    assert isinstance(first_role(a, RoleA), RoleA)
    assert isinstance(first_role(a, RoleB), RoleB)
    assert first_role(_agent("y"), RoleA) is None  # no matching role


def test_first_role_tolerates_missing_roles_attr():
    assert first_role(SimpleNamespace(aid="z"), RoleA) is None


def test_role_index_maps_aid_to_first_role_in_order():
    a1, a2, a3 = _agent("a1", RoleA()), _agent("a2", RoleB()), _agent("a3", RoleA(), RoleB())
    idx = role_index([a1, a2, a3], RoleA)
    assert list(idx) == ["a1", "a3"]  # a2 has no RoleA; insertion order preserved
    assert all(isinstance(r, RoleA) for r in idx.values())
