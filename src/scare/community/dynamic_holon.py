"""Failure-driven dynamic holon membership (Layer 2, Concept C).

On branch failure, BFS the shared ``GridTopologyMirror`` over same-sector
live edges and drop islanded members from this leader's peer view, so the
ADMM doesn't allocate flow that can't physically be served.
``HolonicCommunityRole`` consults the filter via ``LivePeerFilter``.
Does not mutate the mango topology — dropped membership lives on role state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_characteristic

from scare.base.model import Sector
from scare.base.runtime.diagnostics import record_event
from scare.base.topology.topology_mirror import GridTopologyMirror

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class DynamicHolonRole(Role):
    """L2 dynamic-topology role — drops unreachable holon members from the
    local peer view. Implements :class:`LivePeerFilter`."""

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        my_node_id: Any,
        aid_to_node_id: dict[str, Any],
        mirror: GridTopologyMirror,
        *,
        debounce_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self._my_node_id = my_node_id
        # All potential holon peers (sector leaders + self), resolved at
        # build to avoid walking the registry at failure time.
        self._aid_to_node_id = dict(aid_to_node_id)
        self._mirror = mirror
        self._debounce_s = debounce_s
        # Dropped members stay dropped (restore re-eval not yet wired).
        self._unreachable_aids: set[str] = set()
        self._reassess_pending: bool = False

    # --- LivePeerFilter protocol ---

    def is_live(self, addr: Any) -> bool:
        """True when ``addr`` is reachable via live same-sector edges.
        Unknown aids are admitted (additive — live until proven otherwise)."""
        aid = getattr(addr, "aid", None) or str(addr)
        return aid not in self._unreachable_aids

    # --- Role lifecycle ---

    def setup(self) -> None:
        # Global ``BranchFailureEvent`` dispatches to ``on_branch_failure``.
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Schedule a debounced reassess pass. The mirror is updated by the
        scenario callback first; failures in the window collapse into one."""
        if self._reassess_pending:
            return
        self._reassess_pending = True
        try:
            self.context.schedule_timestamp_task(
                self._reassess_membership(),
                timestamp=self.context.current_timestamp + self._debounce_s,
            )
        except Exception:
            # Fallback when the scheduler isn't attached yet.
            self.context.schedule_instant_task(self._reassess_membership())

    async def _reassess_membership(self) -> None:
        self._reassess_pending = False

        # Only the group leader runs the filter.
        if topology_characteristic(self, tid="groups") != "leader":
            return

        reachable_nodes = self._mirror.reachable_from(
            self._my_node_id, sector=self.sector
        )

        # Check every known member aid (cheap) so members added to the holon
        # later aren't missed, regardless of formation timing.
        newly_unreachable: list[str] = []
        for aid, node_id in self._aid_to_node_id.items():
            if aid in self._unreachable_aids:
                continue
            if aid == self.context.aid:
                # Never drop self — the holon needs a leader.
                continue
            if node_id not in reachable_nodes:
                newly_unreachable.append(aid)

        if not newly_unreachable:
            return

        self._unreachable_aids.update(newly_unreachable)

        record_event(
            t=self.context.current_timestamp,
            kind="holon_member_unreachable",
            aid=self.context.aid,
            sector=self.sector.value,
            detail=(
                f"new={len(newly_unreachable)} total_dropped={len(self._unreachable_aids)} "
                f"sample={sorted(newly_unreachable)[:3]}"
            ),
        )
        logger.info(
            "[%s] holon dropped %d unreachable members (sector=%s, total_dropped=%d)",
            self.context.aid,
            len(newly_unreachable),
            self.sector.value,
            len(self._unreachable_aids),
        )
