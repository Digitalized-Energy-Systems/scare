"""Single authority for the promoted-island grid-former reference policy.

A ``GridForming*`` unit is its island's slack/voltage reference: at regulation=1
its ``p_mw`` is a free LP Var that absorbs the island residual, so the MAS must
neither curtail it (δ-box pinned, no actuation) nor read its sign-flipped
capacity as phantom load — it must credit the DELIVERED injection as supply
instead. This object owns both halves (``is_former`` for the actuation/exclusion
guard, ``supply_credit`` for the accounting) so a policy change touches one file
and the two halves can no longer drift. Mirrors ``HeatFrontierController``: a
plain object with pure blackboard reads, unit-testable without a mango context.

The guard flag is read lazily on every ``is_former`` call (never snapshotted at
construction) so it stays byte-identical to a mid-run toggle.
"""

from __future__ import annotations

from typing import Any

from scare.base.util import lookup_grid_former_rating


class GridFormerPolicy:
    def __init__(self, behavior: Any, *, probe_share: float) -> None:
        self.behavior = behavior
        self.probe_share = probe_share

    def is_former(self, aid: str) -> bool:
        """True iff *aid* is a promoted island reference and the guard is on."""
        if not getattr(
            getattr(self.behavior, "_scare_config", None),
            "enable_grid_former_curtail_guard",
            False,
        ):
            return False
        return lookup_grid_former_rating(self.behavior, aid) is not None

    def supply_credit(self, aid: str, sp: float) -> float:
        """Supply to credit a grid-former: ``delivered + share*(rating -
        delivered)``, ``delivered = max(0, -sp)`` (its current injection). The
        probe offers a slice of unused headroom so the holon can allocate load
        the free-Var former then produces to meet, vs ratcheting on delivered."""
        delivered = max(0.0, -float(sp))
        rating = lookup_grid_former_rating(self.behavior, aid)
        if rating is None:
            return delivered
        return delivered + self.probe_share * max(0.0, float(rating) - delivered)
