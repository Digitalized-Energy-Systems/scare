"""Communication-perturbation helpers: inject packet loss / latency jitter."""

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
    """Replace ``world.communication_sim`` to match the perturbation.

    No-op if both knobs are zero (preserves default-config invariance).
    """
    if packet_loss_pct <= 0 and latency_jitter_ms <= 0:
        return

    # Both paths (pure loss and jitter) go through the seeded subclass below:
    # mango's stock SimpleCommunicationSimulation draws loss from the global
    # RNG, making the loss pattern depend on send order / prior stream
    # consumption — reruns of the same task would drop different messages.

    # Latency jitter (optional loss). Two correctness requirements:
    # 1. Quantize the delay to a grid: mango re-solves a MISOCP at each distinct
    #    delivery timestamp, so a continuous delay would explode N co-sent
    #    messages into O(N) solver-heavy steps. The grid bounds that to ~16.
    # 2. Deterministic per package: seed the RNG from package identity so
    #    delay/loss are independent of send order and prior RNG-stream
    #    consumption (a rerun of the same task perturbs the same messages).
    #    Trade-off: messages sharing (sender, receiver, sent_time) share fate.
    sigma_s = latency_jitter_ms / 1000.0
    loss_frac = max(0.0, packet_loss_pct / 100.0) if packet_loss_pct > 0 else 0.0
    # Quantum: ±2σ over ~16 buckets, but never finer than the base delay.
    quantum_s = max(base_delay_s, sigma_s / 4.0)

    class _JitteredLossySim(SimpleCommunicationSimulation):
        """Loss + Gaussian-jittered delay, grid-quantized, deterministic per package."""

        def __init__(
            self,
            *,
            loss_percent: float,
            default_delay_s: float,
            sigma_s: float,
            quantum_s: float,
        ) -> None:
            self._jitter_sigma_s = sigma_s
            self._quantum_s = quantum_s
            super().__init__(loss_percent=loss_percent, default_delay_s=default_delay_s)

        def _jittered_delay(self, msg: Any) -> float:
            if self._jitter_sigma_s <= 0 or self._quantum_s <= 0:
                return max(0.0, self.default_delay_s)
            # Seeded by package identity so the same package yields the same delay.
            seed = hash((msg.sender_id, msg.receiver_id, round(msg.sent_time, 9)))
            rng = random.Random(seed)
            raw = rng.gauss(self.default_delay_s, self._jitter_sigma_s)
            # Snap to the grid to keep distinct delivery timestamps O(1).
            return max(0.0, round(raw / self._quantum_s) * self._quantum_s)

        def calculate_communication(self, current_time, messages):
            results = []
            for msg in messages:
                key = (msg.sender_id, msg.receiver_id)
                if key in self.delay_s_directed_edge_dict:
                    delay_s = self.delay_s_directed_edge_dict[key]
                else:
                    delay_s = self._jittered_delay(msg)
                # Loss is package-seeded like the delay (requirement 2 above);
                # the base class draws from the global RNG instead.
                loss_seed = hash(
                    (msg.sender_id, msg.receiver_id, round(msg.sent_time, 9), "loss")
                )
                reached = random.Random(loss_seed).random() >= self.loss_percent
                results.append(PackageResult(reached=reached, delay_s=delay_s))
            return CommunicationSimulationResult(package_results=results)

    world.communication_sim = _JitteredLossySim(
        loss_percent=loss_frac,
        default_delay_s=base_delay_s,
        sigma_s=sigma_s,
        quantum_s=quantum_s,
    )
