"""Failure-driven dynamic CP-connector membership (Layer 3, Concept C).

A coupling-point (CP) plant — CHP, P2G, G2P, P2H — connects two or
three energy sectors.  At scenario build the ``cps`` topology is
linked to the ``groups`` topology via
``connect_topologies(cps_topo, groups_topo, sector.value)`` for each
sector the CP bridges, so the CP can discover its peer group leaders
by calling ``topology_connectors(self, tid="cps")``.

When a branch failure islands one of those group leaders from the CP's
node — through the CP's *cross-sector* connectivity (which includes
the CP's own bridge edges) — the static peer list is stale.  The CP
will still send ``AskForAvailableFlex`` to the unreachable leader,
wait for an answer that never arrives, and possibly time out only
after the ADMM round has already started.  Even worse, an answer
*does* arrive (the leader is alive on the message bus, just not
physically connected), and the ADMM allocates flow across a path
that no longer exists.

This role sits next to :class:`EnergyConverterRole` and maintains a
*live-connector filter* analogous to L2's
:class:`DynamicHolonRole`:

- On ``BranchFailureEvent`` it BFS-explores the shared
  :class:`GridTopologyMirror` from this CP's parent node, traversing
  *any* sector edge plus CP bridges (the CP needs to know which group
  leaders it can still physically reach across the multi-sector
  graph).
- Group-leader aids whose node is no longer reachable are added to
  the unreachable set.
- :class:`EnergyConverterRole` consumes the filter via
  :class:`LivePeerFilter` so its ``trigger_cp_negotiation`` peer
  iteration silently skips islanded leaders.

Implementation mirrors :class:`DynamicHolonRole` deliberately — both
layers compose the same protocol, the only difference is the
reachability flavour (sector-bounded for L2 vs cross-sector-with-CP
for L3).
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
        # Lookup table for every potential group-leader peer across
        # every sector this CP bridges.  Resolved at scenario build
        # time; the runtime narrowing happens implicitly when
        # ``EnergyConverterRole`` calls ``topology_connectors`` and we
        # filter the result.
        self._leader_aid_to_node = dict(leader_aid_to_node_id)
        self._mirror = mirror
        self._debounce_s = debounce_s
        self._unreachable_aids: set[str] = set()
        self._reassess_pending: bool = False

    # ------------------------------------------------------------------
    # LivePeerFilter protocol
    # ------------------------------------------------------------------

    def is_live(self, addr: Any) -> bool:
        """Return True when ``addr`` is still physically reachable from
        this CP through the live cross-sector graph (any sector edges
        plus CP bridges).

        Unknown aids admitted — same additive semantics as L2.
        """
        aid = getattr(addr, "aid", None) or str(addr)
        return aid not in self._unreachable_aids

    # ------------------------------------------------------------------
    # Role lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        # Trigger is the global ``BranchFailureEvent`` wired by the
        # scenario builder.  No subscribe_message — same pattern as
        # L1 / L2 dynamic-topology roles.
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Schedule a debounced reassess pass.  Same shape as the L1 /
        L2 roles — mirror is already updated by the central callback,
        so by the time our debounce fires the reachability view is
        consistent across the system.
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

        # Only the cps-topology leader bothers — non-leaders don't
        # initiate ADMM, so the filter is moot for them.  A non-leader
        # CP that becomes a leader later via a topology change would
        # cold-start the filter, but mango doesn't mutate topology
        # characteristics at runtime so this isn't a current concern.
        if topology_characteristic(self, tid="cps") != "leader":
            return

        # Cross-sector reachability — the CP cares about reaching group
        # leaders across the full multi-sector graph, including via
        # other CP bridges.
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
