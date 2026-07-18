"""Branch aid construction (delegating to scare.base.addressing) and the
branch-keyed centrality lookup."""

from __future__ import annotations

from scare.base.addressing import branch_aid


def create_branch_aid(branch_id: tuple) -> str:
    return branch_aid(branch_id)


def get_by_branch_id(centrality: dict, branch_id: tuple) -> float:
    if branch_id in centrality:
        return centrality[branch_id]
    rev = (branch_id[1], branch_id[0]) + branch_id[2:]
    return centrality.get(rev, 0.0)
