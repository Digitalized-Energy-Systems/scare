from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mango import Role
from mango.express.topology import topology_neighbors

from scare.base.model import (
    GridPathMessage,
    GridPathResult,
    LineFailure,
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
        # search_id → asyncio.Future[GridPathResult]
        self._pending_searches: dict[str, Any] = {}
        # search_id → set of already-asked AgentAddresses
        self._asked_by_search: dict[str, set] = {}

    def setup(self) -> None:
        self.context.subscribe_message(
            self,
            self._handle_path_message,
            lambda msg, meta: isinstance(msg, GridPathMessage),
        )
        self.context.subscribe_message(
            self,
            self._handle_path_result,
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
        import asyncio

        grid_neighbours = topology_neighbors(self, tid="grid")

        if not grid_neighbours:
            logger.warning(
                "[%s] no grid neighbours – cannot reconfigure", self.context.aid
            )
            return

        search_id = str(uuid4())
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_searches[search_id] = fut
        self._asked_by_search[search_id] = {self.context.addr}

        target_aid = f"node-{to_node_id}"
        target_addr = None
        for a in grid_neighbours:
            if a.aid == target_aid:
                target_addr = a
                break

        msg = GridPathMessage(
            source_addr=self.context.addr,
            target_addr=target_addr,
            path=[self.context.addr],
            asked_agents=[self.context.addr],
            uncertain_connections=[],
        )

        for addr in grid_neighbours:
            self._asked_by_search[search_id].add(addr)
            await self.context.send_message(msg, receiver_addr=addr)

        try:
            result: GridPathResult = await asyncio.wait_for(fut, timeout=10.0)
            logger.info(
                "[%s] reconfiguration path found: %s  uncertain: %s",
                self.context.aid,
                result.path,
                result.uncertain_connections,
            )
            if result.uncertain_connections:
                await self._close_tie_switches(result.uncertain_connections)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] path search %s timed out", self.context.aid, search_id[:8]
            )
        finally:
            self._pending_searches.pop(search_id, None)
            self._asked_by_search.pop(search_id, None)

    async def _handle_path_message(self, message: GridPathMessage, meta: dict) -> None:
        my_addr = self.context.addr
        grid_neighbours = topology_neighbors(self, tid="grid")

        new_path = message.path + [my_addr]
        already_asked = set(message.asked_agents) | {my_addr}

        if message.target_addr is not None and my_addr == message.target_addr:
            result = GridPathResult(
                path=new_path,
                uncertain_connections=message.uncertain_connections,
            )
            await self.context.send_message(result, receiver_addr=message.source_addr)
            return

        forwarded = False
        for addr in grid_neighbours:
            if addr in already_asked:
                continue
            new_uncertain = list(message.uncertain_connections)
            if self._is_uncertain_connection(addr):
                new_uncertain.append((my_addr, addr))

            fwd = GridPathMessage(
                source_addr=message.source_addr,
                target_addr=message.target_addr,
                path=new_path,
                asked_agents=list(already_asked | {addr}),
                uncertain_connections=new_uncertain,
            )
            await self.context.send_message(fwd, receiver_addr=addr)
            forwarded = True

        if not forwarded:
            result = GridPathResult(path=[], uncertain_connections=[])
            await self.context.send_message(result, receiver_addr=message.source_addr)

    async def _handle_path_result(self, message: GridPathResult, meta: dict) -> None:
        for search_id, fut in list(self._pending_searches.items()):
            if not fut.done() and message.path:
                fut.set_result(message)
                break

    def _is_uncertain_connection(self, neighbour_addr: Any) -> bool:
        """Return True if the branch to neighbour_addr has an unknown switch state."""
        branch_aid = _branch_aid_from_addrs(self.context.addr, neighbour_addr)
        obs = self.behavior.observe(branch_aid)
        if obs:
            return obs.get("on_off", 1) == 0
        return False

    async def _close_tie_switches(self, uncertain: list[tuple[Any, Any]]) -> None:
        for from_addr, to_addr in uncertain:
            branch_aid = _branch_aid_from_addrs(from_addr, to_addr)
            if self.behavior.has_action(branch_aid, "switch"):
                self.behavior.act(branch_aid, "switch")
                logger.info("[%s] closed tie switch %s", self.context.aid, branch_aid)


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
            logger.info("[%s] tie switch closed", self.context.aid)
        else:
            logger.warning("[%s] no 'switch' action available", self.context.aid)
