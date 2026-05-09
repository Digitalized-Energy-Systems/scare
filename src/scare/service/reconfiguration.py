from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mango import Role, State
from mango.express.topology import topology_neighbors

from scare.base.model import (
    GridPathMessage,
    GridPathResult,
    LineFailure,
    ReconfigurationCompletedEvent,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class GridReconfigurator(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        node_id: Any,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.node_id = node_id
        # search_id → sentinel marking an in-flight search initiated by
        # this agent.  The result handler drops it on first arrival.
        self._pending_searches: dict[str, Any] = {}
        # search_id → set of already-asked AgentAddresses on the
        # originator.  Kept symmetrical with ``_pending_searches``.
        self._asked_by_search: dict[str, set] = {}
        # search_id values this agent has already forwarded — prevents
        # exponential message blowup when the BFS fan-out re-enters the
        # same node from multiple ancestors.  Each agent contributes at
        # most one forward burst per search.
        self._forwarded_searches: set[str] = set()

    def setup(self) -> None:
        def _wrap(coro_fn):
            def _sync(msg, meta):
                self.context.schedule_instant_task(coro_fn(msg, meta))
            return _sync

        self.context.subscribe_message(
            self,
            _wrap(self._handle_path_message),
            lambda msg, meta: isinstance(msg, GridPathMessage),
        )
        self.context.subscribe_message(
            self,
            _wrap(self._handle_path_result),
            lambda msg, meta: isinstance(msg, GridPathResult),
        )
        self.context.subscribe_event(self, LineFailure, self._on_line_failure)

    def _on_line_failure(self, event: LineFailure, _src: Any) -> None:
        logger.info(
            "[%s] branch failure %s – starting path search",
            self.context.aid,
            event.branch_id,
        )
        self.context.schedule_instant_task(
            self._start_path_search(event.source_node_id, event.target_node_id)
        )

    async def _start_path_search(self, from_node_id: Any, to_node_id: Any) -> None:
        """Broadcast a path-search request to all reachable grid neighbours.

        Fire-and-forget: this coroutine returns once the broadcast is
        sent.  The result is handled asynchronously by
        ``_handle_path_result`` (which closes tie switches and records
        the reconfiguration event).  Awaiting the result here would keep
        this task in a running-but-not-sleeping state and deadlock
        mango's ``tasks_complete_or_sleeping`` barrier — the simulation
        step-loop only advances when every active task is sleeping or
        done.
        """
        grid_neighbours = self._reachable_grid_neighbours()

        if not grid_neighbours:
            logger.warning(
                "[%s] no grid neighbours – cannot reconfigure", self.context.aid
            )
            return

        search_id = str(uuid4())
        # Mark the search as in-flight so ``_handle_path_result`` can
        # accept its result.  Stored as a sentinel rather than a future.
        self._pending_searches[search_id] = True
        self._asked_by_search[search_id] = {self.context.addr}

        msg = GridPathMessage(
            source_addr=self.context.addr,
            target_addr=None,  # unused — termination is by node_id
            target_node_id=to_node_id,
            path=[self.context.addr],
            asked_agents=[self.context.addr],
            uncertain_connections=[],
            search_id=search_id,
        )

        for addr in grid_neighbours:
            self._asked_by_search[search_id].add(addr)
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_path_message(self, message: GridPathMessage, meta: dict) -> None:
        my_addr = self.context.addr
        grid_neighbours = self._reachable_grid_neighbours()

        new_path = message.path + [my_addr]
        already_asked = set(message.asked_agents) | {my_addr}

        # Termination by node_id: any agent whose local node_id matches
        # the target is a valid terminator, even if it's not currently
        # reachable as a direct grid neighbour of the initiator (the
        # failed branch normally severs that direct edge).
        if (
            message.target_node_id is not None
            and self.node_id == message.target_node_id
        ):
            result = GridPathResult(
                path=new_path,
                uncertain_connections=message.uncertain_connections,
                search_id=message.search_id,
            )
            await self.context.send_message(result, receiver_addr=message.source_addr)
            return

        # Per-search dedup: forward each search_id at most once per agent.
        # Without this, the BFS fan-out grows multiplicatively in
        # multi-cycle graphs and floods the message bus before the path
        # can return — every alternative path that re-enters this node
        # would otherwise re-broadcast.  The first arrival is kept; later
        # arrivals are silently dropped (their `path` would be longer or
        # equal anyway under BFS).
        if message.search_id and message.search_id in self._forwarded_searches:
            return
        if message.search_id:
            self._forwarded_searches.add(message.search_id)

        for addr in grid_neighbours:
            if addr in already_asked:
                continue
            new_uncertain = list(message.uncertain_connections)
            if self._is_uncertain_connection(addr):
                new_uncertain.append((my_addr, addr))

            fwd = GridPathMessage(
                source_addr=message.source_addr,
                target_addr=message.target_addr,
                target_node_id=message.target_node_id,
                path=new_path,
                asked_agents=list(already_asked | {addr}),
                uncertain_connections=new_uncertain,
                search_id=message.search_id,
            )
            await self.context.send_message(fwd, receiver_addr=addr)
        # Dead-end branches stay silent — the initiator's timeout handles
        # the case where no path exists; sending empty results caused the
        # initiator to prematurely abandon searches that other branches
        # would have completed.

    async def _handle_path_result(self, message: GridPathResult, meta: dict) -> None:
        if not message.path:
            return
        # Accept only results that match an in-flight search this agent
        # initiated (or, for legacy senders without a search_id, any
        # currently-pending search).
        if message.search_id and message.search_id not in self._pending_searches:
            return
        if not message.search_id and not self._pending_searches:
            return
        # Drop further duplicates for the same search.
        target_search = message.search_id or next(iter(self._pending_searches))
        self._pending_searches.pop(target_search, None)
        self._asked_by_search.pop(target_search, None)

        logger.info(
            "[%s] reconfiguration path found: %s  uncertain: %s",
            self.context.aid,
            [_addr_aid(a) for a in message.path],
            [(_addr_aid(a), _addr_aid(b)) for a, b in message.uncertain_connections],
        )
        if message.uncertain_connections:
            await self._close_tie_switches(message.uncertain_connections)
            from scare.base.diagnostics import record_event

            record_event(
                t=self.context.current_timestamp,
                kind="reconfiguration_completed",
                aid=self.context.aid,
                detail=f"switches={len(message.uncertain_connections)}",
            )
            # mango raises KeyError if no role subscribed; the diagnostics
            # ledger entry above is the load-bearing signal, so swallow.
            try:
                self.context.emit_event(
                    ReconfigurationCompletedEvent(
                        closed_switches=len(message.uncertain_connections)
                    )
                )
            except KeyError:
                pass

    def _reachable_grid_neighbours(self) -> list:
        """Return grid neighbours reachable via either a live edge
        (``State.NORMAL``) or a normally-open backup edge
        (``State.INACTIVE``).  Backups must be included so the path
        search can discover and close them; ``BROKEN`` edges (failed
        branches) are excluded — they're the very topology break that
        triggered this search.
        """
        normal = topology_neighbors(self, tid="grid", state=State.NORMAL)
        inactive = topology_neighbors(self, tid="grid", state=State.INACTIVE)
        seen: set = set()
        result: list = []
        for addr in (*normal, *inactive):
            if addr in seen:
                continue
            seen.add(addr)
            result.append(addr)
        return result

    def _is_uncertain_connection(self, neighbour_addr: Any) -> bool:
        """Return True if the branch to neighbour_addr has an unknown switch state."""
        branch_aid = _branch_aid_from_addrs(self.context.addr, neighbour_addr)
        obs = self.behavior.observe(branch_aid)
        if obs:
            return obs.get("on_off", 1) == 0
        return False

    async def _close_tie_switches(self, uncertain: list[tuple[Any, Any]]) -> None:
        from scare.base.diagnostics import record_switch

        for from_addr, to_addr in uncertain:
            branch_aid = _branch_aid_from_addrs(from_addr, to_addr)
            if self.behavior.has_action(branch_aid, "switch"):
                self.behavior.act(branch_aid, "switch")
                record_switch(
                    t=self.context.current_timestamp,
                    aid=branch_aid,
                    reason="reconfig_close",
                )
                logger.info("[%s] closed tie switch %s", self.context.aid, branch_aid)


def _addr_aid(addr: Any) -> str:
    return getattr(addr, "aid", str(addr))


def _branch_aid_from_addrs(addr_a: Any, addr_b: Any) -> str:
    def _extract_id(addr: Any) -> int:
        aid = addr.aid if hasattr(addr, "aid") else str(addr)
        try:
            return int(aid.split("-")[-1])
        except ValueError:
            return 0

    a, b = _extract_id(addr_a), _extract_id(addr_b)
    hi, lo = (a, b) if a > b else (b, a)
    return f"branch-{hi}-{lo}"


class GridTieSwitchOperator(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        branch_id: tuple,
        centrality: float = 0.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.branch_id = branch_id
        self.centrality = centrality

    def setup(self) -> None:
        pass

    def close_switch(self) -> None:
        if self.behavior.has_action(self.context.aid, "switch"):
            from scare.base.diagnostics import record_switch

            self.behavior.act(self.context.aid, "switch")
            record_switch(
                t=self.context.current_timestamp,
                aid=self.context.aid,
                reason="tie_switch_close",
            )
            logger.info("[%s] tie switch closed", self.context.aid)
        else:
            logger.warning("[%s] no 'switch' action available", self.context.aid)
