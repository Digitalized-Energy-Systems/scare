"""TTL-bounded per-key value store — the shared primitive behind the hand-rolled
"value + t_set, get-if-fresh" registries (line/gen curtail locks, relief headroom,
congestion price, cp heat ceiling, qv relief, feeder voltage).

Three invariants are load-bearing and must not drift:
- freshness is a STRICT ``now - t_set < ttl`` (exactly-at-ttl is STALE);
- ``get`` is NON-EVICTING — a stale-but-present entry is left in place so a later
  ``stamp`` can revive it (the line-curtail-lock re-arm depends on this);
- ``stamp`` is a NO-OP when the key is absent (it re-stamps, never inserts).

The payload is opaque (scalar / tuple / None) — each caller keeps its own packing;
the store owns only the ``(payload, t_set)`` envelope and the staleness test. A
per-call ``ttl`` override lets a caller (line-congestion) pass its own freshness
window without folding it into construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class FreshnessStore:
    def __init__(self, data: dict[Any, tuple[Any, float]], ttl: float) -> None:
        self._data = data  # {key: (payload, t_set)} — persisted on the behavior
        self.ttl = ttl

    @classmethod
    def on(cls, behavior: Any, attr: str, ttl: float) -> FreshnessStore:
        """Bind to the per-behavior store at ``attr`` (same lifetime as the sim
        world), sharing the underlying dict across all wrappers for that attr."""
        from scare.base.util import _get_behavior_store

        return cls(_get_behavior_store(behavior, attr), ttl)

    def put(self, key: Any, payload: Any, now: float) -> None:
        self._data[key] = (payload, float(now))

    def get(self, key: Any, now: float, ttl: float | None = None) -> Any:
        """Payload iff fresh (``now - t_set < ttl``), else None. NON-EVICTING: a
        stale entry survives the read."""
        entry = self._data.get(key)
        if entry is None:
            return None
        payload, t_set = entry
        if now - t_set < (self.ttl if ttl is None else ttl):
            return payload
        return None

    def pop(self, key: Any) -> None:
        self._data.pop(key, None)

    def stamp(self, key: Any, now: float) -> None:
        """Re-stamp an EXISTING key's freshness, keeping its payload. No-op when
        the key is absent (never inserts a phantom entry)."""
        entry = self._data.get(key)
        if entry is None:
            return
        self._data[key] = (entry[0], float(now))

    def items_fresh(
        self, now: float, ttl: float | None = None
    ) -> Iterator[tuple[Any, Any]]:
        cutoff = self.ttl if ttl is None else ttl
        for key, (payload, t_set) in self._data.items():
            if now - t_set < cutoff:
                yield key, payload
