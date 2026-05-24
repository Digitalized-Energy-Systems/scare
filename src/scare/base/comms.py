"""Communication-perturbation helpers for the robustness experiments.

Replaces the world's default ``SimpleCommunicationSimulation`` with one
that injects packet loss and / or latency jitter as configured on
``RestorationConfiguration``.  Kept tiny: composes mango's existing
``SimpleCommunicationSimulation`` (lossy + static delay) and
``DelayProviderCommunicationSimulation`` (callable-based jitter).
"""

from __future__ import annotations

import random
from typing import Any

from mango.simulation.communication import SimpleCommunicationSimulation


def install_perturbation(
    world: Any,
    *,
    base_delay_s: float,
    packet_loss_pct: float,
    latency_jitter_ms: float,
) -> None:
    """Replace ``world.communication_sim`` with one that matches the
    requested perturbation.  No-op if both perturbation knobs are zero
    (preserves the default-config invariance check).
    """
    if packet_loss_pct <= 0 and latency_jitter_ms <= 0:
        return

    # Pure packet loss with static delay.
    if latency_jitter_ms <= 0:
        world.communication_sim = SimpleCommunicationSimulation(
            loss_percent=packet_loss_pct / 100.0,
            default_delay_s=base_delay_s,
        )
        return

    # Latency jitter (with optional packet loss).  Use the lossy
    # ``SimpleCommunicationSimulation`` as the carrier so packet loss is
    # always honoured, and shadow its ``default_delay_s`` per call with
    # a Gaussian draw to inject jitter.  The earlier implementation
    # used ``DelayProviderCommunicationSimulation`` which had no loss
    # support and silently dropped ``packet_loss_pct`` (audit P1-3).
    sigma_s = latency_jitter_ms / 1000.0
    loss_frac = max(0.0, packet_loss_pct / 100.0) if packet_loss_pct > 0 else 0.0

    class _JitteredLossySim(SimpleCommunicationSimulation):
        """Loss + per-message Gaussian-jittered delay."""

        def __init__(self, *, loss_percent: float, default_delay_s: float,
                     sigma_s: float) -> None:
            # ``self._jitter_mean_s`` is populated by the property setter
            # below when ``super().__init__`` runs ``self.default_delay_s = …``.
            self._jitter_sigma_s = sigma_s
            super().__init__(loss_percent=loss_percent, default_delay_s=default_delay_s)

        @property
        def default_delay_s(self) -> float:  # type: ignore[override]
            return max(
                0.0, random.gauss(self._jitter_mean_s, self._jitter_sigma_s)
            )

        @default_delay_s.setter
        def default_delay_s(self, value: float) -> None:
            # ``SimpleCommunicationSimulation.__init__`` writes the
            # static delay through ``self.default_delay_s = ...``; route
            # those writes into the mean instead so the base class's
            # constructor still works and subsequent sets behave
            # symmetrically.
            self._jitter_mean_s = float(value)

    world.communication_sim = _JitteredLossySim(
        loss_percent=loss_frac,
        default_delay_s=base_delay_s,
        sigma_s=sigma_s,
    )


