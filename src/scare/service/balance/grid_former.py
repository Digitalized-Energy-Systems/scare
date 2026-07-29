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

# Credit ``delivered + share*(rating-delivered)`` so the free-Var former produces
# to meet offered load; a positive share over-credits when it shares an island with
# a budgeted slack (offered load routes through the slack and the L2 floor blocks
# re-shed). recoverable_islanding seed 100000023: share 0.0 -> PWSF 0.42, gas slack
# PASS; 0.5 -> 0.66, FAIL (110% over); rating -> 0.77, FAIL (151%). Default 0 keeps
# the gas slack compliant. Analogue of ``_HEAT_L2_PROBE_SHARE`` (heat has no slack
# budget).
# CAVEAT: those three points were measured while promotion silently shrank the gas
# slack budget 25% (fixed 2026-07-29) and while the guard was off, so the >0 shares
# were over-crediting against a budget that was itself too small — re-measure before
# quoting them as a refutation of a positive share.
GRID_FORMER_SUPPLY_PROBE_SHARE: float = 0.0


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
