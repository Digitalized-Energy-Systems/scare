"""Continuous coupling-weight (trust) dynamics on the gossip topology.

Each neighbour pair carries a reliability score K(t) in [0, 1] replacing the
binary BROKEN flag: it recovers on message arrivals and decays in silence; a
confirmed BROKEN edge clamps to 0. K feeds the live-neighbour predicate
(K >= tau), biases gossip next-hop selection, and weights reported utilisation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

# Decay/recovery rates tuned so ~8 s of silence drives K below the 0.5
# liveness threshold.
_DEFAULT_DECAY_RATE_PER_S: float = 0.1
_DEFAULT_RECOVER_RATE: float = 0.6
_DEFAULT_LIVENESS_THRESHOLD: float = 0.5
_DEFAULT_INITIAL: float = 1.0
_MIN_K: float = 0.0
_MAX_K: float = 1.0


@dataclass
class TrustParams:
    """Trust-dynamics knobs. All times in seconds."""

    decay_rate_per_s: float = _DEFAULT_DECAY_RATE_PER_S
    recover_rate: float = _DEFAULT_RECOVER_RATE
    liveness_threshold: float = _DEFAULT_LIVENESS_THRESHOLD
    initial: float = _DEFAULT_INITIAL


class TrustLedger:
    """Per-neighbour continuous trust scores.

    Owned by an agent role; not thread-safe. Callers pass the current
    simulation timestamp into every operation.
    """

    def __init__(self, params: TrustParams | None = None) -> None:
        self._params = params or TrustParams()
        # addr_str -> (K, last_update_t)
        self._scores: dict[str, tuple[float, float]] = {}
        # Confirmed-BROKEN addrs (sticky zero).
        self._broken: set[str] = set()

    # --- Single-link accessors ---

    def score(self, addr_key: str, now: float) -> float:
        """Current K-score, applying decay since last touch."""
        if addr_key in self._broken:
            return _MIN_K
        entry = self._scores.get(addr_key)
        if entry is None:
            # Unknown neighbours start at the prior (no elapsed time).
            return self._params.initial
        k, t_last = entry
        return self._decay(k, now - t_last)

    def is_live(self, addr_key: str, now: float) -> bool:
        return self.score(addr_key, now) >= self._params.liveness_threshold

    # --- State transitions ---

    def on_message_received(self, addr_key: str, now: float) -> float:
        """Record a reception; nudges K toward 1."""
        if addr_key in self._broken:
            return _MIN_K
        prior = self.score(addr_key, now)
        new_k = min(_MAX_K, prior + self._params.recover_rate * (_MAX_K - prior))
        self._scores[addr_key] = (new_k, now)
        return new_k

    def on_silence(self, addr_key: str, now: float) -> float:
        """Apply decay without recovery (neighbour probed but didn't reply)."""
        if addr_key in self._broken:
            return _MIN_K
        prior = self.score(addr_key, now)
        # score() already decayed; persist so later calls don't compound it.
        self._scores[addr_key] = (prior, now)
        return prior

    def mark_broken(self, addr_key: str) -> None:
        """Confirmed branch failure — score sticks at zero."""
        self._broken.add(addr_key)
        self._scores[addr_key] = (_MIN_K, 0.0)

    def clear_broken(self, addr_key: str, now: float) -> None:
        """Branch recovered — re-enable from a cautious half-initial value."""
        self._broken.discard(addr_key)
        self._scores[addr_key] = (self._params.initial * 0.5, now)

    # --- Aggregate views ---

    def filter_live(self, addr_keys: Iterable[str], now: float) -> list[str]:
        """Subset of addr_keys whose K-score is above threshold."""
        return [k for k in addr_keys if self.is_live(k, now)]

    def scored(self, addr_keys: Iterable[str], now: float) -> list[tuple[str, float]]:
        """``(addr_key, K)`` pairs in input order."""
        return [(k, self.score(k, now)) for k in addr_keys]

    def snapshot(self, now: float) -> dict[str, float]:
        """``{addr_key: K}`` for diagnostics."""
        return {k: self.score(k, now) for k in self._scores}

    # --- Internal ---

    def _decay(self, k: float, dt: float) -> float:
        if dt <= 0.0:
            return max(_MIN_K, min(_MAX_K, k))
        # Linear approx of exp(-decay_rate_per_s * dt) * k, clamped to [0, 1].
        return max(_MIN_K, k * max(0.0, 1.0 - self._params.decay_rate_per_s * dt))


def hash_weighted_choice(
    addrs: list,
    weights: list[float],
    seed: str,
) -> object | None:
    """Deterministic hash-based selection biased by weights.

    Picks the index whose cumulative weight covers a SHA256-derived fraction
    in [0, total), so identical inputs yield identical output. Returns None for
    empty inputs; hash-uniform pick if all weights are non-positive.
    """
    if not addrs:
        return None
    total = sum(max(0.0, w) for w in weights)
    if total <= 0.0:
        h = hashlib.sha256(seed.encode()).digest()
        return addrs[int.from_bytes(h[:4], "big") % len(addrs)]
    h = hashlib.sha256(seed.encode()).digest()
    # 32-bit fractional draw in [0, total).
    u = (int.from_bytes(h[:4], "big") / 2**32) * total
    cum = 0.0
    for addr, w in zip(addrs, weights):
        cum += max(0.0, w)
        if u < cum:
            return addr
    return addrs[-1]
