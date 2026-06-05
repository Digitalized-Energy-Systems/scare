"""Failure-driven dynamic CP-connector membership (Layer 3, Concept C).

A coupling-point (CP) plant — CHP, P2G, G2P, P2H — bridges two or three
sectors. At scenario build the ``cps`` topology is connected to the
``groups`` topology per bridged sector, so the CP discovers peer group
leaders via ``topology_connectors(self, tid="cps")``.

When a branch failure islands one of those leaders from the CP's node
(across the CP's cross-sector connectivity, including its own bridge
edges), the static peer list goes stale: the CP would negotiate flow
across a path that no longer exists.

This role sits next to :class:`EnergyConverterRole` and maintains a
live-connector filter analogous to L2's :class:`DynamicHolonRole`:

- On ``BranchFailureEvent`` it BFS-explores the shared
  :class:`GridTopologyMirror` from this CP's parent node, traversing
  any sector edge plus CP bridges.
- Group-leader aids whose node is no longer reachable join the
  unreachable set.
- :class:`EnergyConverterRole` consumes the filter via
  :class:`LivePeerFilter` so its ``trigger_cp_negotiation`` peer loop
  skips islanded leaders.

Mirrors :class:`DynamicHolonRole`; the only difference is the
reachability flavour (cross-sector-with-CP here vs sector-bounded L2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_characteristic

from scare.base.diagnostics import record_event
from scare.base.topology_mirror import GridTopologyMirror

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class DynamicConnectorRole(Role):
    """L3 dynamic-topology role — drops physically unreachable group
    leaders from the CP's connector iteration view.

    Implements :class:`LivePeerFilter` so :class:`EnergyConverterRole`
    can consult it without taking a direct dependency on this class.
    """

    def __init__(
        self,
        behavior: "RestorationEnvironmentBehavior",
        my_node_id: Any,
        leader_aid_to_node_id: dict[str, Any],
        mirror: GridTopologyMirror,
        *,
        debounce_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self._my_node_id = my_node_id
        # Every potential group-leader peer across every bridged sector,
        # resolved at scenario build. Runtime narrowing happens when
        # ``EnergyConverterRole`` calls ``topology_connectors`` and the
        # filter is applied to the result.
        self._leader_aid_to_node = dict(leader_aid_to_node_id)
        self._mirror = mirror
        self._debounce_s = debounce_s
        self._unreachable_aids: set[str] = set()
        self._reassess_pending: bool = False

    # ------------------------------------------------------------------
    # LivePeerFilter protocol
    # ------------------------------------------------------------------

    def is_live(self, addr: Any) -> bool:
        """True when ``addr`` is still reachable from this CP through the
        live cross-sector graph (any sector edges plus CP bridges).
        Unknown aids are admitted (additive semantics, as L2).
        """
        aid = getattr(addr, "aid", None) or str(addr)
        return aid not in self._unreachable_aids

    # ------------------------------------------------------------------
    # Role lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        # Trigger is the global ``BranchFailureEvent`` wired by the
        # scenario builder; no subscribe_message (as L1/L2 roles).
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Schedule a debounced reassess pass. The mirror is updated by
        the central callback first, so the reachability view is
        consistent by the time the debounce fires.
        """
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

        # Only the cps-topology leader runs the filter; non-leaders
        # don't initiate ADMM. mango doesn't mutate topology
        # characteristics at runtime, so leadership won't change here.
        if topology_characteristic(self, tid="cps") != "leader":
            return

        # Cross-sector reachability across the full multi-sector graph,
        # including via other CP bridges.
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
            "[%s] CP dropped %d unreachable group leaders "
            "(total_dropped=%d)",
            self.context.aid,
            len(newly_unreachable),
            len(self._unreachable_aids),
        )
