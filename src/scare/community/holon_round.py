"""L2 rebalance round state: the flex-collection buffers, the round lock and
the reactive-trigger throttle.
"""

from __future__ import annotations

from typing import Any

from mango import sender_addr as mango_sender_addr

from scare.base.model import AvailableFlexAnswer


class RebalanceRound:
    """One L2 flex-collection round plus the gate that decides when the next
    one may start. Owned by the holonic role; reads the agent id through it.
    """

    def __init__(self, role: Any) -> None:
        self._role = role
        # Collected member flex answers; ``senders`` holds the sender per
        # answer to route the allocation back as override target.
        self.answers: list[AvailableFlexAnswer] = []
        self.senders: list[Any] = []
        self.expected: int = 0
        # Round tag for the collection timeout: a stale timeout from a
        # completed round must not release/fire a later round.
        self.token: int = 0
        # Stamped on AskForAvailableFlex and echoed by responders; a straggler
        # from round N must not count into round N+1.
        self.round_id: str = ""
        self.active: bool = False
        # Reactive triggers within ``rebalance_min_gap_s`` of this are dropped.
        self.last_t: float = float("-inf")
        # Watchdog no-change skip: reactive trigger sets True, a successful
        # rebalance clears it. True initially so the first tick runs.
        self.dirty: bool = True
        # Guards a single deferred retry at gap-expiry so a throttled trigger
        # runs when the fuse clears, not at the slow watchdog tick.
        self.retry_pending: bool = False

    def open(self, *, expected: int, now: float) -> int:
        """Take the lock and start a fresh collection round; returns its token.

        Clears ``dirty`` because we are committed; a trigger arriving during
        execution re-sets it so the next watchdog tick fires.
        """
        self.active = True
        self.last_t = now
        self.dirty = False
        self.answers = []
        self.senders = []
        self.expected = expected
        self.token += 1
        self.round_id = f"{self._role.context.aid}/{self.token}"
        return self.token

    def add(
        self, message: AvailableFlexAnswer, meta: dict, member_keys: set[str]
    ) -> bool:
        """Record an answer. True iff the round is now complete.

        Rejects answers outside the open round or from non-members; both would
        otherwise inflate the count.
        """
        if not self.active:
            return False
        # Strict round identity: asks stamp a fresh id per round and the sole
        # responder (balance._handle_ask_flex) always echoes it, so a round-N
        # straggler can't double-count a member into round N+1.
        if getattr(message, "round_id", "") != self.round_id:
            return False
        sender = mango_sender_addr(meta)
        if member_keys:
            sender_key = str(sender)
            if (
                sender_key != str(self._role.context.addr)
                and sender_key not in member_keys
            ):
                return False
        self.answers.append(message)
        self.senders.append(sender)
        return len(self.answers) >= self.expected

    def drain(self) -> tuple[list[AvailableFlexAnswer], list[Any]]:
        """Take the collected answers and release the lock early, so a pending
        collection timeout cannot re-enter the solve.
        """
        answers = self.answers[:]
        senders = self.senders[:]
        self.answers = []
        self.senders = []
        self.expected = 0
        self.active = False
        return answers, senders

    def restore(self, answers: list[AvailableFlexAnswer], senders: list[Any]) -> None:
        """Put a drained round back so a fallback solver re-reads it.

        Without this the fallback re-guards on ``active`` and re-reads the
        buffers, so it would be a silent no-op.
        """
        self.answers = answers
        self.senders = senders
        self.active = True
