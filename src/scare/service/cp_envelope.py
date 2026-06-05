"""Cross-sector coalition envelope for a coupling-point (CP) agent.

Extracted from :class:`scare.service.cp.EnergyConverterRole`. A L2.5 coalition
can commit a CP to directional per-sector flows for a TTL window; this object
holds that commitment and clamps an ADMM result vector to it. ``None`` flows =>
the CP runs free. Pure data + arithmetic, so it is unit-testable without a
mango context (the role keeps the logging / ``record_event`` side-effects).
"""

from __future__ import annotations

from scare.base.model import Sector


class CoalitionEnvelope:
    """Active coalition envelope for one CP.

    State mirrors the three fields the role previously held inline
    (``_envelope_flows_mw`` / ``_envelope_expires_at`` /
    ``_envelope_coalition_id``).
    """

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
        """Install (latest-wins) a directional flow commitment expiring at
        ``now + ttl_s``."""
        self._flows_mw = flows
        self._expires_at = now + float(ttl_s)
        self._coalition_id = coalition_id

    def active(self, now: float) -> bool:
        """True iff an envelope is set and unexpired. Matches the original
        ``_envelope_active`` side-effect: an expired envelope is cleared."""
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
        """When an envelope is active, overwrite each sector dimension of
        ``result`` (in place) with the committed flow and return the pre-clamp
        snapshot. Returns ``None`` (result untouched) when no envelope is active.

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
