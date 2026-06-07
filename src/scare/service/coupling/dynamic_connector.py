"""Failure-driven dynamic CP-connector membership (Layer 3, Concept C).

A CP plant (CHP, P2G, G2P, P2H) discovers peer group leaders via the ``cps``
topology. On branch failure, BFS the shared :class:`GridTopologyMirror` from the
CP's node over any sector edge plus CP bridges and drop islanded leaders, so the
CP doesn't negotiate flow over a vanished path. :class:`EnergyConverterRole`
consults the filter via :class:`LivePeerFilter`.

Mirrors :class:`DynamicHolonRole`; differs only in reachability flavour
(cross-sector-with-CP here vs sector-bounded L2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_characteristic

from scare.base.runtime.diagnostics import record_event
from scare.base.topology.topology_mirror import GridTopologyMirror

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class DynamicConnectorRole(Role):
    """L3 dynamic-topology role — drops unreachable group leaders from the CP's
    connector view. Implements :class:`LivePeerFilter`."""

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        my_node_id: Any,
        leader_aid_to_node_id: dict[str, Any],
        mirror: GridTopologyMirror,
        *,
        debounce_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self._my_node_id = my_node_id
        # All potential group-leader peers across bridged sectors, resolved
        # at build. ``EnergyConverterRole`` narrows at runtime via the filter.
        self._leader_aid_to_node = dict(leader_aid_to_node_id)
        self._mirror = mirror
        self._debounce_s = debounce_s
        self._unreachable_aids: set[str] = set()
        self._reassess_pending: bool = False

    # --- LivePeerFilter protocol ---

    def is_live(self, addr: Any) -> bool:
        """True when ``addr`` is reachable via the live cross-sector graph
        (any sector edges plus CP bridges). Unknown aids admitted (additive)."""
        aid = getattr(addr, "aid", None) or str(addr)
        return aid not in self._unreachable_aids

    # --- Role lifecycle ---

    def setup(self) -> None:
        # Global ``BranchFailureEvent`` dispatches to ``on_branch_failure``.
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Schedule a debounced reassess pass; the mirror is updated by the
        central callback first."""
        if self._reassess_pending:
            return
        self._reassess_pending = True
        try:
            self.context.schedule_timestamp_task(
                self._reassess_membership(),
                timestamp=self.context.current_timestamp + self._debounce_s,
            )
        except Exception:
            self.context.schedule_instant_task(self._reassess_membership())

    async def _reassess_membership(self) -> None:
        self._reassess_pending = False

        # Only the cps-topology leader runs the filter (non-leaders don't
        # initiate ADMM); leadership is fixed at runtime.
        if topology_characteristic(self, tid="cps") != "leader":
            return

        # Cross-sector reachability, including via other CP bridges.
        reachable_nodes = self._mirror.reachable_from(
            self._my_node_id, sector=None, allow_cp_bridges=True
        )

        newly_unreachable: list[str] = []
        for aid, node_id in self._leader_aid_to_node.items():
            if aid in self._unreachable_aids:
                continue
            if node_id not in reachable_nodes:
                newly_unreachable.append(aid)

        if not newly_unreachable:
            return

        self._unreachable_aids.update(newly_unreachable)

        record_event(
            t=self.context.current_timestamp,
            kind="cp_connector_unreachable",
            aid=self.context.aid,
            sector="cp",
            detail=(
                f"new={len(newly_unreachable)} total_dropped={len(self._unreachable_aids)} "
                f"sample={sorted(newly_unreachable)[:3]}"
            ),
        )
        logger.info(
            "[%s] CP dropped %d unreachable group leaders (total_dropped=%d)",
            self.context.aid,
            len(newly_unreachable),
            len(self._unreachable_aids),
        )
