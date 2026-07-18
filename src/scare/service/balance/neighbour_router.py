from __future__ import annotations

from scare.service.balance.trust import TrustLedger, hash_weighted_choice


class NeighbourRouter:
    """Trust-weighted neighbour directory for the gossip negotiator.

    Owns the trust ledger + last-seen map, filters live peers, and picks the
    K-weighted next hop. Topology and behavior lookups stay on the Role, which
    passes the candidate neighbours (and the current time) in.
    """

    def __init__(self, trust: TrustLedger) -> None:
        self._trust = trust
        self._last_seen: dict[str, float] = {}

    def record_sender(self, key: str, now: float) -> None:
        self._last_seen[key] = now
        # B.1: each received message recovers the K-score.
        self._trust.on_message_received(key, now)

    def touch(self, addrs: list, now: float) -> None:
        # Seed last-seen for just-contacted neighbours; the ledger's optimistic
        # initial score is the grace period, so this drives no liveness decision.
        for addr in addrs:
            key = str(addr)
            if key not in self._last_seen:
                self._last_seen[key] = now

    def live(self, neighbours: list, now: float) -> list:
        """Neighbours whose trust score K_ij exceeds the liveness threshold
        (unknown neighbours bootstrap optimistically at K = 1.0)."""
        return [a for a in neighbours if self._trust.is_live(str(a), now)]

    def scored(self, neighbours: list, now: float) -> list[float]:
        return [self._trust.score(str(a), now) for a in neighbours]

    def next_hop(self, neighbours: list, nid: str, counter: int, now: float):
        """B.1: K-weighted deterministic next-hop; proportional to trust K_ij,
        uniform SHA256-modulo when all K equal, routing around low-K peers."""
        if not neighbours:
            return None
        return hash_weighted_choice(
            neighbours, self.scored(neighbours, now), f"{nid}:{counter}"
        )
