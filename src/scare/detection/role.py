from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role
from mango_energy_environments import BranchFailureEvent

from scare.base.model import LineFailure

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class ProblemDetector(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        node_id: Any,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.node_id = node_id

    def on_global_event(self, event: Any) -> None:
        if isinstance(event, BranchFailureEvent):
            self._on_branch_failure(event)

    def _on_branch_failure(self, event: BranchFailureEvent) -> None:
        from_id, to_id = event.branch_id[0], event.branch_id[1]
        if self.node_id not in (from_id, to_id):
            return
        logger.info(
            "[%s] forwarding branch failure %s", self.context.aid, event.branch_id
        )
        self.context.emit_event(
            LineFailure(
                source_node_id=self.node_id,
                target_node_id=to_id if self.node_id == from_id else from_id,
                branch_id=event.branch_id,
            )
        )
