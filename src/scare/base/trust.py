"""Continuous coupling-weight (trust) dynamics on the gossip topology.

Implements the *adaptive coupling weight* layer of the SCARE adaptive
dynamical network (B.1 in the algorithms chapter): each neighbour pair
$(i, j)$ carries a continuous reliability score $K_{ij}(t) \\in [0, 1]$
that replaces the binary BROKEN/operational flag.

The score is updated multiplicatively from observed message arrivals and
decays in their absence:

    K(t + dt) = (1 - eta_decay * dt) * K(t)              # silent step
    K(t)      = clip(K(t) + eta_recover * (1 - K(t)), 0, 1)  # message step

A confirmed BROKEN edge clamps the score to 0 immediately.

The trust score plugs into:

* ``_live_neighbours`` in :mod:`scare.service.balance` — replaces the
  binary heartbeat predicate with a continuous threshold ``K >= tau``.
* ``_deterministic_next`` in :mod:`scare.service.balance` — biases the
  hash-based next-hop selection toward high-K neighbours.
* ``GridConstraintMonitor._handle_constraint_state`` in
  :mod:`scare.service.constraints` — weights the utilisation of a
  reported neighbour by the receiver's K-score for the link the message
  arrived on.

Theoretical context: the continuous coupling generalises Berner et al.
(2023) §11.2's exponentially weighted moving averages on link state.  In
the unsaturated regime the spectral gap of the gossip graph Laplacian
becomes a *random* variable whose mean is shifted by the average K-score;
this gives an analytical handle on the convergence-rate effect of
unreliable links via Proposition~7 of the chapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# Default per-second decay rate (silence aging out the score) and per-event
# recovery rate (a fresh message restoring it).  Values chosen so that:
#  - 1 missed second of messages reduces K by ~0.1
#  - one reception fully recovers a moderately decayed K
#  - 8 seconds of silence (the existing heartbeat-multiple of 1s poll)
#    drives K below 0.5, the default liveness threshold.
_DEFAULT_DECAY_RATE_PER_S: float = 0.1
_DEFAULT_RECOVER_RATE: float = 0.6
_DEFAULT_LIVENESS_THRESHOLD: float = 0.5
_DEFAULT_INITIAL: float = 1.0
_MIN_K: float = 0.0
_MAX_K: float = 1.0


@dataclass
class TrustParams:
    """Tunable knobs for the trust dynamics.  All times in seconds."""

    decay_rate_per_s: float = _DEFAULT_DECAY_RATE_PER_S
    recover_rate: float = _DEFAULT_RECOVER_RATE
    liveness_threshold: float = _DEFAULT_LIVENESS_THRESHOLD
    initial: float = _DEFAULT_INITIAL


class TrustLedger:
    """Per-neighbour continuous trust scores.

    Owned by an agent role; not thread-safe (mango is single-threaded
    per agent).  Time is the agent's simulation clock; callers pass the
    current timestamp into every operation.
    """

    def __init__(self, params: TrustParams | None = None) -> None:
        self._params = params or TrustParams()
        # addr_str -> (K, last_update_t)
        self._scores: dict[str, tuple[float, float]] = {}
        # addr_str -> True if confirmed BROKEN (sticky zero)
        self._broken: set[str] = set()

    # ------------------------------------------------------------------
    # Single-link accessors
    # ------------------------------------------------------------------

    def score(self, addr_key: str, now: float) -> float:
        """Return the *current* K-score, applying decay since last touch."""
        if addr_key in self._broken:
            return _MIN_K
        entry = self._scores.get(addr_key)
        if entry is None:
            # Unknown neighbours start at the prior; on first contact the
            # decay term is zero so they're treated as fully trusted.
            return self._params.initial
        k, t_last = entry
        return self._decay(k, now - t_last)

    def is_live(self, addr_key: str, now: float) -> bool:
        return self.score(addr_key, now) >= self._params.liveness_threshold

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def on_message_received(self, addr_key: str, now: float) -> float:
        """Record a successful message reception; nudges K toward 1."""
        if addr_key in self._broken:
            return _MIN_K
        prior = self.score(addr_key, now)
        new_k = min(_MAX_K, prior + self._params.recover_rate * (_MAX_K - prior))
        self._scores[addr_key] = (new_k, now)
        return new_k

    def on_silence(self, addr_key: str, now: float) -> float:
        """Apply decay without recovery (called when probing a neighbour
        that didn't reply)."""
        if addr_key in self._broken:
            return _MIN_K
        prior = self.score(addr_key, now)
        # Decay was already applied in score(); persist the result so
        # subsequent calls see the lower value without compounding it.
        self._scores[addr_key] = (prior, now)
        return prior

    def mark_broken(self, addr_key: str) -> None:
        """Confirmed branch failure — score sticks at zero."""
        self._broken.add(addr_key)
        self._scores[addr_key] = (_MIN_K, 0.0)

    def clear_broken(self, addr_key: str, now: float) -> None:
        """Branch recovered (e.g. tie switch closed) — re-enable from a
        cautious initial value rather than 1.0."""
        self._broken.discard(addr_key)
        self._scores[addr_key] = (self._params.initial * 0.5, now)

    # ------------------------------------------------------------------
    # Aggregate views
    # ------------------------------------------------------------------

    def filter_live(self, addr_keys: Iterable[str], now: float) -> list[str]:
        """Return the subset of addr_keys whose K-score is above threshold."""
        return [k for k in addr_keys if self.is_live(k, now)]

    def scored(self, addr_keys: Iterable[str], now: float) -> list[tuple[str, float]]:
        """Return ``(addr_key, K)`` pairs in the same order as the input."""
        return [(k, self.score(k, now)) for k in addr_keys]

    def snapshot(self, now: float) -> dict[str, float]:
        """Return a {addr_key: K} dict for diagnostics."""
        return {k: self.score(k, now) for k in self._scores}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decay(self, k: float, dt: float) -> float:
        if dt <= 0.0:
            return max(_MIN_K, min(_MAX_K, k))
        # Discrete approximation of exp(-decay_rate_per_s * dt) · k, but
        # cheaper and monotonically faithful.  Capped to [0, 1].
        return max(_MIN_K, k * max(0.0, 1.0 - self._params.decay_rate_per_s * dt))


def hash_weighted_choice(
    addrs: list,
    weights: list[float],
    seed: str,
) -> object | None:
    """Deterministic hash-based selection biased by weights.

    Generalises ``_deterministic_next``: instead of a uniform modulo,
    pick the index whose cumulative weight covers the SHA256-derived
    fraction in [0, total].  Identical inputs always produce the same
    output (stability), and high-weight addresses are picked
    proportionally more often (bias).

    Falls back to None for empty inputs and to the first address if all
    weights are non-positive.
    """
    if not addrs:
        return None
    import hashlib

    total = sum(max(0.0, w) for w in weights)
    if total <= 0.0:
        h = hashlib.sha256(seed.encode()).digest()
        return addrs[int.from_bytes(h[:4], "big") % len(addrs)]
    h = hashlib.sha256(seed.encode()).digest()
    # 32-bit fractional draw in [0, total)
    u = (int.from_bytes(h[:4], "big") / 2**32) * total
    cum = 0.0
    for addr, w in zip(addrs, weights):
        cum += max(0.0, w)
        if u < cum:
            return addr
    return addrs[-1]
