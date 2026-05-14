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
        *,
        enable_ranking: bool = False,
        window_s: float = 1.5,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.node_id = node_id
        # search_id → sentinel marking an in-flight search initiated by
        # this agent.  In legacy mode the result handler drops the
        # sentinel on first arrival; in ranking mode it stays until the
        # window closes.
        self._pending_searches: dict[str, Any] = {}
        # search_id → set of already-asked AgentAddresses on the
        # originator.  Kept symmetrical with ``_pending_searches``.
        self._asked_by_search: dict[str, set] = {}
        # search_id values this agent has already forwarded — prevents
        # exponential message blowup when the BFS fan-out re-enters the
        # same node from multiple ancestors.  In legacy mode this is a
        # set (forward at most once); in ranking mode it stores the
        # lowest max_loading_percent we have forwarded so far, and we
        # allow re-forward when a strictly better path arrives.
        self._forwarded_searches: set[str] = set()
        self._forwarded_loading: dict[str, float] = {}
        # 6c — path-feasibility ranking state.
        self.enable_ranking = enable_ranking
        self.window_s = window_s
        # search_id → buffered candidate results awaiting window close.
        self._search_results: dict[str, list[GridPathResult]] = {}

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
        if self.enable_ranking:
            self._search_results[search_id] = []
            # Close the ranking window after ``window_s`` and act on the
            # best buffered result.
            deadline = self.context.current_timestamp + self.window_s
            self.context.schedule_timestamp_task(
                self._finalise_ranked_search(search_id),
                timestamp=deadline,
            )

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
        incoming_max_loading = float(getattr(message, "max_loading_percent", 0.0) or 0.0)

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
                max_loading_percent=incoming_max_loading,
            )
            await self.context.send_message(result, receiver_addr=message.source_addr)
            return

        # Per-search dedup.
        # Legacy mode: forward each search_id at most once per agent.
        # Ranking mode: re-forward only when a strictly better
        # (lower max_loading) copy arrives — bounded by the loading
        # monotone, so the search still terminates.
        if message.search_id:
            if self.enable_ranking:
                prev_best = self._forwarded_loading.get(message.search_id)
                _loading_eps = 1e-6
                if (
                    prev_best is not None
                    and incoming_max_loading >= prev_best - _loading_eps
                ):
                    return
                self._forwarded_loading[message.search_id] = incoming_max_loading
            else:
                if message.search_id in self._forwarded_searches:
                    return
                self._forwarded_searches.add(message.search_id)

        for addr in grid_neighbours:
            if addr in already_asked:
                continue
            new_uncertain = list(message.uncertain_connections)
            if self._is_uncertain_connection(addr):
                new_uncertain.append((my_addr, addr))

            # Update the running max_loading from the line we are about
            # to traverse (my_addr → addr).  Failure to read the branch
            # observation is treated as zero — better to under-attribute
            # loading than to refuse to forward.
            branch_loading = self._branch_loading_percent(addr)
            new_max_loading = max(incoming_max_loading, branch_loading)

            fwd = GridPathMessage(
                source_addr=message.source_addr,
                target_addr=message.target_addr,
                target_node_id=message.target_node_id,
                path=new_path,
                asked_agents=list(already_asked | {addr}),
                uncertain_connections=new_uncertain,
                search_id=message.search_id,
                max_loading_percent=new_max_loading,
            )
            await self.context.send_message(fwd, receiver_addr=addr)
        # Dead-end branches stay silent — the initiator's timeout handles
        # the case where no path exists; sending empty results caused the
        # initiator to prematurely abandon searches that other branches
        # would have completed.

    def _branch_loading_percent(self, neighbour_addr: Any) -> float:
        """Read the loading of the branch between this node and a grid
        neighbour, normalised to the percent convention used by
        SECTOR_CONSTRAINTS (see ``obs_constraint_values``).
        """
        from scare.base.model import Sector
        from scare.base.util import obs_constraint_values

        branch_aid = _branch_aid_from_addrs(self.context.addr, neighbour_addr)
        obs = self.behavior.observe(branch_aid)
        if not obs:
            return 0.0
        values = obs_constraint_values(obs, Sector.ELECTRICITY)
        return float(values.get("loading_percent", 0.0) or 0.0)

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
        target_search = message.search_id or next(iter(self._pending_searches))

        # Ranking mode: buffer and let the window-close handler pick the
        # best.  Don't pop the sentinel here so subsequent results from
        # alternative paths are still accepted within the window.
        if self.enable_ranking and target_search in self._search_results:
            self._search_results[target_search].append(message)
            logger.debug(
                "[%s] reconfiguration candidate buffered (search=%s, "
                "max_loading=%.2f, len=%d)",
                self.context.aid,
                target_search[:8],
                float(getattr(message, "max_loading_percent", 0.0) or 0.0),
                len(message.path),
            )
            return

        # Legacy mode: first-arrival wins; close switches immediately.
        self._pending_searches.pop(target_search, None)
        self._asked_by_search.pop(target_search, None)

        await self._act_on_path_result(message)

    async def _finalise_ranked_search(self, search_id: str) -> None:
        """Close the ranking window for ``search_id`` and act on the best
        buffered candidate (lowest peak ``max_loading_percent``; among
        ties, shortest path).  Times out silently when no candidate
        arrived — falls back to the legacy timeout-handles-it behaviour.
        """
        candidates = self._search_results.pop(search_id, None)
        self._pending_searches.pop(search_id, None)
        self._asked_by_search.pop(search_id, None)
        # Forwarders' loading state can also be flushed; bounded growth
        # is the only correctness property, leaks across searches are
        # cosmetic.
        self._forwarded_loading.pop(search_id, None)

        if not candidates:
            logger.info(
                "[%s] reconfiguration ranking window closed: no candidates",
                self.context.aid,
            )
            return

        best = min(
            candidates,
            key=lambda r: (
                float(getattr(r, "max_loading_percent", 0.0) or 0.0),
                len(r.path),
            ),
        )
        logger.info(
            "[%s] reconfiguration ranking window closed: %d candidates, "
            "best max_loading=%.2f path_len=%d",
            self.context.aid,
            len(candidates),
            float(getattr(best, "max_loading_percent", 0.0) or 0.0),
            len(best.path),
        )
        await self._act_on_path_result(best)

    async def _act_on_path_result(self, message: GridPathResult) -> None:
        logger.info(
            "[%s] reconfiguration path found: %s  uncertain: %s",
            self.context.aid,
            [_addr_aid(a) for a in message.path],
            [(_addr_aid(a), _addr_aid(b)) for a, b in message.uncertain_connections],
        )
        if message.uncertain_connections:
            await self._close_tie_switches(message.uncertain_connections)
            from scare.base.diagnostics import record_event
            from scare.base.model import StartBalanceNegotiation

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

            # The scenario-level ``behavior_in(on_global_event=
            # ReconfigurationCompletedEvent, _trigger_balance, ...)`` hook
            # never fires because emits are role-local — see audit P0-4.
            # Reach the balance leaders the only way available to us from
            # a role context: send each grid-topology neighbour an
            # explicit ``StartBalanceNegotiation``.  Leaders trigger their
            # negotiation; non-leaders ignore the message.  This is how
            # newly-reachable load shifts through after a tie switch
            # closes.
            grid_neighbours = topology_neighbors(self, tid="grid")
            for addr in grid_neighbours:
                try:
                    await self.context.send_message(
                        StartBalanceNegotiation(), receiver_addr=addr,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "[%s] post-reconfig balance trigger send failed for %s: %s",
                        self.context.aid, addr, exc,
                    )

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
