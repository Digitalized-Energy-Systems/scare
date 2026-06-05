"""Scare-local wrapper around ``DistributedOptimizationRole``.

The upstream role uses a deny-list filter that accepts any content
except two internal wrapper types, which means any scare message
(``GridPathMessage``, ``AskEnergyMessage``, …) routed to a CP-type
agent gets dispatched into the ADMM handler and crashes with
``AttributeError: 'GridPathMessage' object has no attribute 'v'``.

This subclass replaces the filter with an allow-list of the message
types the ADMM algorithm actually exchanges, so scare-domain messages
pass through untouched.
"""

from __future__ import annotations

from typing import Any

from distributed_resource_optimization.algorithm.admm.core import (
    ADMMAnswer,
    ADMMMessage,
    ADMMStart,
)
from distributed_resource_optimization.algorithm.core import on_exchange_message
from distributed_resource_optimization.carrier.mango import (
    DistributedOptimizationRole,
    MangoCarrier,
    _CarrierReply,
    _CarrierRequest,
)

_ADMM_CONTENT_TYPES: tuple[type, ...] = (
    _CarrierRequest,
    ADMMMessage,
    ADMMAnswer,
    ADMMStart,
)


class ScareDistributedOptimizationRole(DistributedOptimizationRole):
    def setup(self) -> None:
        self._carrier = MangoCarrier(self, self._include_self)
        self.context.subscribe_message(
            self,
            self._handle_optimization,
            lambda c, m: isinstance(c, _ADMM_CONTENT_TYPES),
        )
        self.context.subscribe_message(
            self,
            self._handle_reply,
            lambda c, m: isinstance(c, _CarrierReply),
        )

    def _handle_optimization(self, content: Any, meta: dict) -> None:
        if isinstance(content, _CarrierRequest):
            actual_meta = {**meta, "_request_id": content.request_id}
            actual_content = content.content
        else:
            actual_meta = meta
            actual_content = content

        # Route through mango's scheduler, not bare ``asyncio.create_task``:
        # under discrete-stepping the world only advances tasks the scheduler
        # knows about; an orphan task would run off the simulation clock and
        # its sends would be flushed in a later step.
        self.context.schedule_instant_task(
            on_exchange_message(
                self.algorithm, self._carrier, actual_content, actual_meta
            )
        )
