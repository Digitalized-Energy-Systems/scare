"""Failure-driven dynamic holon membership (Layer 2, Concept C).

A holon is a chunk of same-sector group leaders that ran an inter-group
ADMM together, chunked once at scenario build
([_build_topologies][scare.scenario.restoration._build_topologies]).
When a branch failure islands one of its leaders, the ADMM keeps
treating it as a peer and redistributes flow as if it could still help,
producing allocations that physically cannot be served.

This role sits next to ``HolonicCommunityRole`` on every holon-eligible
leader and maintains a live-member filter:

- On ``BranchFailureEvent``, BFS the shared ``GridTopologyMirror`` from
  this leader's node through same-sector live edges (a holon spans one
  sector).
- Members (resolved lazily — the holon may not have formed yet) whose
  node is no longer reachable join the unreachable set.
- ``HolonicCommunityRole`` consumes the filter via ``LivePeerFilter``
  so its ``_try_rebalance`` / ADMM peer loop skips islanded members.

Like ``DynamicRepartitionRole`` (L1), this does not mutate the mango
``holons`` topology at runtime — dropped membership lives on the role's
state and is consulted by the host role at peer-iteration time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango.express.topology import topology_characteristic

from scare.base.diagnostics import record_event
from scare.base.model import Sector
from scare.base.topology_mirror import GridTopologyMirror, LivePeerFilter

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class DynamicHolonRole(Role):
    """L2 dynamic-topology role — drops physically unreachable holon
    members from the local peer-iteration view.

    Implements :class:`LivePeerFilter` so ``HolonicCommunityRole`` can
    consult it without taking a direct dependency on this class.
    """

    def __init__(
        self,
        behavior: "RestorationEnvironmentBehavior",
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
        # Every potential holon peer (sector-wide leaders + self),
        # resolved at scenario build to avoid walking the agent registry
        # at failure time. ``HolonicCommunityRole._holon_member_addrs``
        # is the runtime narrowing.
        self._aid_to_node_id = dict(aid_to_node_id)
        self._mirror = mirror
        self._debounce_s = debounce_s
        # Members declared unreachable. A dropped member stays dropped
        # (restore re-evaluation not yet wired).
        self._unreachable_aids: set[str] = set()
        self._reassess_pending: bool = False

    # ------------------------------------------------------------------
    # LivePeerFilter protocol
    # ------------------------------------------------------------------

    def is_live(self, addr: Any) -> bool:
        """True when ``addr`` is still reachable from this leader through
        live same-sector edges. Unknown aids are admitted (the layer is
        additive — a peer is live until proven otherwise), which also
        covers the gap before the first reassess pass.
        """
        aid = getattr(addr, "aid", None) or str(addr)
        return aid not in self._unreachable_aids

    # ------------------------------------------------------------------
    # Role lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        # No subscribe_message — the global ``BranchFailureEvent`` wired
        # by the scenario builder dispatches to ``on_branch_failure``
        # (as L1's repartition role).
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Schedule a debounced reassess pass. The mirror is updated by
        the scenario's ``set_on_branch_failure`` callback first, so the
        reachability view is consistent when the debounce fires.
        Failures within the window collapse into one reassess.
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
            # Fallback when the scheduler isn't attached yet.
            self.context.schedule_instant_task(self._reassess_membership())

    async def _reassess_membership(self) -> None:
        self._reassess_pending = False

        # Only the group leader runs the filter; non-leaders aren't
        # holon initiators.
        if topology_characteristic(self, tid="groups") != "leader":
            return

        # Same-sector reachability from this leader's node.
        reachable_nodes = self._mirror.reachable_from(
            self._my_node_id, sector=self.sector
        )

        # Check every known member aid, not just the lazily-populated
        # ``HolonicCommunityRole._holon_member_addrs`` — otherwise we'd
        # miss members the holon adds later. The full set is cheap and
        # gives a stable filter regardless of holon-formation timing.
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
            "[%s] holon dropped %d unreachable members "
            "(sector=%s, total_dropped=%d)",
            self.context.aid,
            len(newly_unreachable),
            self.sector.value,
            len(self._unreachable_aids),
        )
