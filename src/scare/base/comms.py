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

from mango.simulation.communication import (
    CommunicationSimulationResult,
    PackageResult,
    SimpleCommunicationSimulation,
)


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
    # always honoured, and override ``calculate_communication`` to inject
    # a per-message Gaussian-jittered delay.
    #
    # Two correctness requirements drive the implementation:
    #
    # 1. **Quantize the delay to a coarse grid.**  mango is a
    #    discrete-event simulator: it advances the clock to the next
    #    distinct message-delivery timestamp and runs a full agent sweep
    #    + an energyflow MISOCP re-solve at each one.  A *continuous*
    #    Gaussian delay mints a unique delivery timestamp per message, so
    #    N messages sent at one instant explode into O(N) tiny discrete
    #    steps — each ~0.5 s of solver wall-clock.  That is the
    #    robustness_latency timeout (tasks 56-59: ~25-350x slowdown,
    #    never reaching 10 s sim-time inside the 1500 s wall).  Snapping
    #    the delay to a grid bounds the number of distinct delivery
    #    timestamps per instant to ~16 regardless of message count, while
    #    still producing variable latency + reordering (the experiment's
    #    actual intent).
    # 2. **Deterministic per package.**  mango's contract
    #    (``communication.py``) is that re-evaluating the same
    #    ``MessagePackage`` yields the same result.  Seeding a local RNG
    #    from the package identity (sender, receiver, sent_time) honours
    #    that — the previous ``random.gauss`` on every property read both
    #    broke determinism and re-rolled loss on every access.
    #
    # The earlier implementation used ``DelayProviderCommunicationSimulation``
    # which had no loss support and silently dropped ``packet_loss_pct``
    # (audit P1-3); this carrier keeps loss.
    sigma_s = latency_jitter_ms / 1000.0
    loss_frac = max(0.0, packet_loss_pct / 100.0) if packet_loss_pct > 0 else 0.0
    # Quantum: bound ±2σ to ~16 buckets (σ/4), but never finer than the
    # base delay — the simulator's natural tick.
    quantum_s = max(base_delay_s, sigma_s / 4.0)

    class _JitteredLossySim(SimpleCommunicationSimulation):
        """Loss + per-message Gaussian-jittered delay, grid-quantized and
        deterministic per package."""

        def __init__(self, *, loss_percent: float, default_delay_s: float,
                     sigma_s: float, quantum_s: float) -> None:
            self._jitter_sigma_s = sigma_s
            self._quantum_s = quantum_s
            super().__init__(loss_percent=loss_percent, default_delay_s=default_delay_s)

        def _jittered_delay(self, msg: Any) -> float:
            # Deterministic per (sender, receiver, sent_time): re-evaluating
            # the same package returns the same delay (mango contract).
            seed = hash((msg.sender_id, msg.receiver_id, round(msg.sent_time, 9)))
            rng = random.Random(seed)
            raw = rng.gauss(self.default_delay_s, self._jitter_sigma_s)
            # Snap to the grid so distinct delivery timestamps stay O(1)
            # in message count.
            return max(0.0, round(raw / self._quantum_s) * self._quantum_s)

        def calculate_communication(self, current_time, messages):
            results = []
            for msg in messages:
                key = (msg.sender_id, msg.receiver_id)
                if key in self.delay_s_directed_edge_dict:
                    delay_s = self.delay_s_directed_edge_dict[key]
                else:
                    delay_s = self._jittered_delay(msg)
                reached = random.random() >= self.loss_percent
                results.append(PackageResult(reached=reached, delay_s=delay_s))
            return CommunicationSimulationResult(package_results=results)

    world.communication_sim = _JitteredLossySim(
        loss_percent=loss_frac,
        default_delay_s=base_delay_s,
        sigma_s=sigma_s,
        quantum_s=quantum_s,
    )


