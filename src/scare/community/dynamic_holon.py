"""Failure-driven dynamic holon membership (Layer 2, Concept C).

A holon is a chunk of same-sector group leaders that ran an inter-group
ADMM together.  The chunking is decided once at scenario build time
([_build_topologies][scare.scenario.restoration._build_topologies]),
which is a perfectly reasonable cyber-organisational decision *while*
all chunk members can physically exchange energy.  When a branch
failure islands one of the chunk's leaders from the rest, the holon's
ADMM keeps treating that leader as a peer — it waits for the leader's
``AvailableFlexAnswer``, then redistributes flow as if the lost leader
could still help.  The result is allocations that physically cannot
be served.

This role closes that gap by sitting next to ``HolonicCommunityRole``
on every holon-eligible leader and maintaining a *live-member filter*:

- On ``BranchFailureEvent``, BFS the shared ``GridTopologyMirror`` from
  this leader's parent node through same-sector live edges (the holon
  spans one sector, so cross-sector traversal is not relevant).
- Members of the holon (resolved lazily — the holon may not have formed
  yet when the first failure fires) whose parent node is no longer
  reachable get added to the unreachable set.
- ``HolonicCommunityRole`` consumes the filter via the
  ``LivePeerFilter`` Protocol so its ``_try_rebalance`` / ADMM peer
  iteration silently skips islanded members.

Like ``DynamicRepartitionRole`` (L1), this role does *not* mutate the
mango ``holons`` topology graph at runtime — the dropped membership
lives on the role's own state and is consulted by the host role at the
moment of peer iteration.  This keeps the dynamic decision local to
each leader without requiring mango runtime topology mutation.
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
        # Lookup table for *every* potential holon peer (sector-wide
        # leaders + this leader itself).  Resolved at scenario build
        # time so the role doesn't need to walk the agent registry at
        # failure time.  ``HolonicCommunityRole._holon_member_addrs``
        # is the runtime narrowing.
        self._aid_to_node_id = dict(aid_to_node_id)
        self._mirror = mirror
        self._debounce_s = debounce_s
        # Aid set of members we've declared unreachable.  Populated
        # incrementally — once a member is dropped it stays dropped
        # until a restore event re-evaluates (not yet wired; future).
        self._unreachable_aids: set[str] = set()
        self._reassess_pending: bool = False

    # ------------------------------------------------------------------
    # LivePeerFilter protocol
    # ------------------------------------------------------------------

    def is_live(self, addr: Any) -> bool:
        """Return True when ``addr`` is still physically reachable from
        this leader through live same-sector edges.

        Unknown aids are *admitted* (returning ``True``) — the dynamic
        layer is purely additive, so an unfamiliar peer is treated as
        live until proven otherwise.  This also gracefully handles the
        period between scenario build and the first reassess pass.
        """
        aid = getattr(addr, "aid", None) or str(addr)
        return aid not in self._unreachable_aids

    # ------------------------------------------------------------------
    # Role lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        # No subscribe_message — the trigger is the global
        # ``BranchFailureEvent`` registered by the scenario builder via
        # ``behavior_in(world, ..., on_global_event=BranchFailureEvent,
        # role_types=DynamicHolonRole)``.  That dispatch lands on
        # ``on_branch_failure`` below, same shape as L1's repartition
        # role.
        pass

    def on_branch_failure(self, branch_id: tuple) -> None:
        """Schedule a debounced reassess pass.

        The mirror is already updated centrally by the scenario's
        ``set_on_branch_failure`` callback (see ``_build_topologies``),
        so by the time our debounce window fires the reachability view
        is consistent.  Multiple failures within the debounce window
        collapse into a single reassess.
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
            # Defensive fallback — instant task if the scheduler is
            # not yet attached (mirrors L1's repartition role).
            self.context.schedule_instant_task(self._reassess_membership())

    async def _reassess_membership(self) -> None:
        self._reassess_pending = False

        # Only the leader of the underlying group bothers; non-leaders
        # are not holon initiators and the filter is moot for them.
        if topology_characteristic(self, tid="groups") != "leader":
            return

        # Same-sector reachability from this leader's node.
        reachable_nodes = self._mirror.reachable_from(
            self._my_node_id, sector=self.sector
        )

        # Cross-check every member aid we know about.  We can't restrict
        # to ``HolonicCommunityRole._holon_member_addrs`` here because
        # that's lazily populated and we'd miss members the holon may
        # add later.  Iterating the full known set is cheap and produces
        # a stable filter regardless of holon-formation timing.
        newly_unreachable: list[str] = []
        for aid, node_id in self._aid_to_node_id.items():
            if aid in self._unreachable_aids:
                continue
            if aid == self.context.aid:
                # Don't drop ourselves — the holon needs a leader.
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
