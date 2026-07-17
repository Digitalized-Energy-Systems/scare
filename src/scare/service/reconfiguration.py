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
    Sector,
    StartBalanceNegotiation,
)
from scare.base.runtime.diagnostics import record_event, record_switch
from scare.base.util import _get_behavior_store, obs_constraint_values

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
        # search_id → sentinel for an in-flight search. Legacy drops on first
        # result; ranking keeps until the window closes.
        self._pending_searches: dict[str, Any] = {}
        # Per-search forwarding dedup against BFS fan-out re-entry. Legacy:
        # forward once. Ranking: track lowest max_loading, re-forward if better.
        self._forwarded_searches: set[str] = set()
        self._forwarded_loading: dict[str, float] = {}
        self.enable_ranking = enable_ranking
        self.window_s = window_s
        # search_id → candidates buffered until window close.
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
        # Both endpoints detect the failure independently; the env "switch"
        # action is a TOGGLE, so two successful searches over the same open tie
        # would close it and then re-open it. Only one endpoint initiates.
        if not is_initiating_endpoint(event.source_node_id, event.target_node_id):
            logger.debug(
                "[%s] branch failure %s – peer endpoint initiates the path search",
                self.context.aid,
                event.branch_id,
            )
            return
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

        Fire-and-forget: result handled async by ``_handle_path_result``.
        Awaiting here would deadlock mango's tasks_complete_or_sleeping barrier.
        """
        grid_neighbours = self._reachable_grid_neighbours()

        if not grid_neighbours:
            logger.warning(
                "[%s] no grid neighbours – cannot reconfigure", self.context.aid
            )
            return

        search_id = str(uuid4())
        # Mark in-flight so ``_handle_path_result`` accepts its result.
        self._pending_searches[search_id] = True
        if self.enable_ranking:
            self._search_results[search_id] = []
            # Close the ranking window after ``window_s``.
            deadline = self.context.current_timestamp + self.window_s
            self.context.schedule_timestamp_task(
                self._finalise_ranked_search(search_id),
                timestamp=deadline,
            )

        msg = GridPathMessage(
            source_addr=self.context.addr,
            target_addr=None,  # unused — termination by node_id
            target_node_id=to_node_id,
            path=[self.context.addr],
            asked_agents=[self.context.addr],
            uncertain_connections=[],
            search_id=search_id,
        )

        for addr in grid_neighbours:
            await self.context.send_message(msg, receiver_addr=addr)

    async def _handle_path_message(self, message: GridPathMessage, meta: dict) -> None:
        my_addr = self.context.addr
        grid_neighbours = self._reachable_grid_neighbours()

        new_path = message.path + [my_addr]
        already_asked = set(message.asked_agents) | {my_addr}
        incoming_max_loading = float(
            getattr(message, "max_loading_percent", 0.0) or 0.0
        )

        # Termination by node_id: any matching agent terminates, even if not a
        # direct neighbour of the initiator (the failed branch severs that edge).
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

        # Per-search dedup. Legacy: forward each search_id once. Ranking:
        # re-forward only on a strictly-lower max_loading (monotone, terminates).
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
            switch_state = self._branch_switch_state(addr)
            if switch_state is None and self._branch_has_switch(addr):
                # Unknown switch state: neither traversable-as-closed nor
                # safely closable later (the toggle could open it).
                continue
            new_uncertain = list(message.uncertain_connections)
            if switch_state == 0:
                new_uncertain.append((my_addr, addr))

            # Carry running max_loading across the traversed line; an unreadable
            # branch counts as zero (under-attribute rather than refuse).
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
        # Dead-end branches stay silent — the initiator's timeout covers the
        # no-path case; empty results would abort searches others could finish.

    def _branch_loading_percent(self, neighbour_addr: Any) -> float:
        """Branch loading to a neighbour, percent convention of SECTOR_CONSTRAINTS."""
        branch_aid = _branch_aid_from_addrs(self.context.addr, neighbour_addr)
        obs = self.behavior.observe(branch_aid)
        if not obs:
            return 0.0
        values = obs_constraint_values(obs, Sector.ELECTRICITY)
        return float(values.get("loading_percent", 0.0) or 0.0)

    async def _handle_path_result(self, message: GridPathResult, meta: dict) -> None:
        if not message.path:
            return
        # Accept only results matching an in-flight search; legacy senders
        # without a search_id match any pending search.
        if message.search_id and message.search_id not in self._pending_searches:
            return
        if not message.search_id and not self._pending_searches:
            return
        target_search = message.search_id or next(iter(self._pending_searches))

        # Ranking mode: buffer and let the window-close handler pick the best.
        # Keep the sentinel so later results within the window are accepted.
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

        # Legacy mode: first arrival wins; close switches immediately.
        self._pending_searches.pop(target_search, None)

        await self._act_on_path_result(message)

    async def _finalise_ranked_search(self, search_id: str) -> None:
        """Close the ranking window and act on the best candidate.

        Lowest peak ``max_loading_percent``, ties broken by shortest path.
        Silent when no candidate arrived.
        """
        candidates = self._search_results.pop(search_id, None)
        self._pending_searches.pop(search_id, None)
        # Flush forwarder loading state.
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

            record_event(
                t=self.context.current_timestamp,
                kind="reconfiguration_completed",
                aid=self.context.aid,
                detail=f"switches={len(message.uncertain_connections)}",
            )
            # mango raises KeyError if no role subscribed; the ledger entry
            # above is the load-bearing signal.
            try:
                self.context.emit_event(
                    ReconfigurationCompletedEvent(
                        closed_switches=len(message.uncertain_connections)
                    )
                )
            except KeyError:
                pass

            # Role-local emits miss the scenario-level hook, so reach balance
            # leaders directly: send each grid neighbour a
            # ``StartBalanceNegotiation`` to shift newly-reachable load.
            grid_neighbours = topology_neighbors(self, tid="grid")
            for addr in grid_neighbours:
                try:
                    await self.context.send_message(
                        StartBalanceNegotiation(),
                        receiver_addr=addr,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "[%s] post-reconfig balance trigger send failed for %s: %s",
                        self.context.aid,
                        addr,
                        exc,
                    )

    def _reachable_grid_neighbours(self) -> list:
        """Grid neighbours via a live (``NORMAL``) or backup (``INACTIVE``) edge.

        Backups are included so the search can discover and close them;
        ``BROKEN`` edges are excluded.
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

    def _branch_switch_state(self, neighbour_addr: Any) -> int | None:
        """Observed ``on_off`` of the branch to ``neighbour_addr``; None when
        the state is unknown (no obs / no ``on_off`` reading)."""
        branch_aid = _branch_aid_from_addrs(self.context.addr, neighbour_addr)
        return switch_state_from_obs(self.behavior.observe(branch_aid))

    def _branch_has_switch(self, neighbour_addr: Any) -> bool:
        branch_aid = _branch_aid_from_addrs(self.context.addr, neighbour_addr)
        return self.behavior.has_action(branch_aid, "switch")

    async def _close_tie_switches(self, uncertain: list[tuple[Any, Any]]) -> None:
        # Shared in-flight ledger across all reconfigurator roles: switch acts
        # don't trigger a re-solve, so the observed ``on_off`` can hold the
        # pre-close snapshot indefinitely and the act-time re-check alone
        # would double-toggle (re-open) a tie closed by a concurrent search.
        inflight: set[str] = _get_behavior_store(
            self.behavior, "_scare_ties_closed_inflight", set
        )
        for from_addr, to_addr in uncertain:
            branch_aid = _branch_aid_from_addrs(from_addr, to_addr)
            if not self.behavior.has_action(branch_aid, "switch"):
                continue
            # The env "switch" action is a TOGGLE: re-check state at act time
            # so a branch already closed (e.g. by an earlier search) isn't
            # re-opened, and never toggle a branch whose state is unknown.
            state = switch_state_from_obs(self.behavior.observe(branch_aid))
            if state == 1:
                # Fresh observation confirms the close landed.
                inflight.discard(branch_aid)
            if branch_aid in inflight:
                logger.info(
                    "[%s] skip tie switch %s (close in flight)",
                    self.context.aid,
                    branch_aid,
                )
                continue
            if not should_close_tie(state):
                logger.info(
                    "[%s] skip tie switch %s (state=%s: %s)",
                    self.context.aid,
                    branch_aid,
                    state,
                    "already closed" if state == 1 else "unknown",
                )
                continue
            self.behavior.act(branch_aid, "switch")
            inflight.add(branch_aid)
            record_switch(
                t=self.context.current_timestamp,
                aid=branch_aid,
                reason="reconfig_close",
            )
            logger.info("[%s] closed tie switch %s", self.context.aid, branch_aid)


def is_initiating_endpoint(source_node_id: Any, target_node_id: Any) -> bool:
    """True when the endpoint at ``source_node_id`` owns the path search for a
    failed branch. Deterministic and symmetric (smaller node id wins) so
    exactly one of the two detecting endpoints initiates."""
    if type(source_node_id) is type(target_node_id):
        try:
            return source_node_id < target_node_id
        except TypeError:
            pass
    return str(source_node_id) < str(target_node_id)


def switch_state_from_obs(obs: dict | None) -> int | None:
    """``on_off`` from a branch observation; None when unknown."""
    if not obs or "on_off" not in obs:
        return None
    try:
        return int(obs["on_off"])
    except (TypeError, ValueError):
        return None


def should_close_tie(switch_state: int | None) -> bool:
    """Close only a branch KNOWN to be open. The env action is a toggle, so
    acting on a closed (1) or unknown (None) branch could open it."""
    return switch_state == 0


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
            self.behavior.act(self.context.aid, "switch")
            record_switch(
                t=self.context.current_timestamp,
                aid=self.context.aid,
                reason="tie_switch_close",
            )
            logger.info("[%s] tie switch closed", self.context.aid)
        else:
            logger.warning("[%s] no 'switch' action available", self.context.aid)
