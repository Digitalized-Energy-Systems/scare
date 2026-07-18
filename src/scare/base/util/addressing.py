"""Branch aid construction, delegating to scare.base.addressing."""

from __future__ import annotations

from scare.base.addressing import branch_aid


def create_branch_aid(branch_id: tuple) -> str:
    return branch_aid(branch_id)
