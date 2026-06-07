"""Cross-sector coalition envelope for a coupling-point (CP) agent.

A L2.5 coalition can commit a CP to directional per-sector flows for a TTL
window; this holds that commitment and clamps an ADMM result vector to it.
``None`` flows => the CP runs free. Pure data + arithmetic.
"""

from __future__ import annotations

from scare.base.model import Sector


class CoalitionEnvelope:
    """Active coalition envelope for one CP."""

    def __init__(self) -> None:
        self._flows_mw: dict[Sector, float] | None = None
        self._expires_at: float = -1.0
        self._coalition_id: str = ""

    @property
    def coalition_id(self) -> str:
        return self._coalition_id

    @property
    def flows_mw(self) -> dict[Sector, float] | None:
        return self._flows_mw

    def set(
        self,
        flows: dict[Sector, float],
        ttl_s: float,
        coalition_id: str,
        *,
        now: float,
    ) -> None:
        """Install (latest-wins) a flow commitment expiring at ``now + ttl_s``."""
        self._flows_mw = flows
        self._expires_at = now + float(ttl_s)
        self._coalition_id = coalition_id

    def active(self, now: float) -> bool:
        """True iff an envelope is set and unexpired; clears an expired one."""
        if self._flows_mw is None:
            return False
        if now > self._expires_at:
            self._flows_mw = None
            return False
        return True

    def clamp(
        self,
        result: list,
        result_index: dict[Sector, int],
        *,
        now: float,
    ) -> list | None:
        """Overwrite each sector dim of ``result`` (in place) with the committed
        flow and return the pre-clamp snapshot; ``None`` if no envelope is active.

        ``result_index`` maps each sector to its index in the flat
        ``[el, heat, gas]`` ADMM result vector.
        """
        if not self.active(now):
            return None
        envelope = self._flows_mw or {}
        pre_clamp = list(result)
        for sector, idx in result_index.items():
            if idx >= len(result) or sector not in envelope:
                continue
            result[idx] = float(envelope[sector])
        return pre_clamp
