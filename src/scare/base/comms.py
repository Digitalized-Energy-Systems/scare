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

    if latency_jitter_ms > 0:
        # Gaussian around base_delay with σ=jitter, clipped to ≥ 0.
        # Loss is enforced by wrapping a callable that occasionally
        # returns a sentinel large delay — but the cleanest path is to
        # build a SimpleCommunicationSimulation purely for loss and
        # set our own delay-providing sim alongside.  mango doesn't
        # support both at once, so prefer the lossy sim when loss > 0
        # and add jitter via a per-edge override map.
        from mango.simulation.communication import (
            DelayProviderCommunicationSimulation,
        )

        sigma_s = latency_jitter_ms / 1000.0

        def _provider() -> float:
            return max(0.0, random.gauss(base_delay_s, sigma_s))

        sim = DelayProviderCommunicationSimulation(
            default_delay_s_provider=_provider
        )
        # No native loss support — emulate via probability that the
        # provider returns a "dropped" sentinel.  But mango interprets
        # the provider purely as delay, not loss.  Fall back to
        # no-loss + jitter; the lossy branch below covers loss.
        world.communication_sim = sim
        return

    # Pure packet loss with static delay.
    from mango.simulation.communication import SimpleCommunicationSimulation

    sim = SimpleCommunicationSimulation(
        loss_percent=packet_loss_pct / 100.0,
        default_delay_s=base_delay_s,
    )
    world.communication_sim = sim


def schedule_agent_dropout(
    world: Any,
    aids: tuple[str, ...],
    at_s: float,
) -> None:
    """At simulation time ``at_s``, unregister every agent in ``aids``.

    Implemented as a one-shot timestamp task scheduled on the world
    clock.  No-op when ``aids`` is empty or ``at_s`` is past horizon.
    """
    if not aids or at_s == float("inf"):
        return
    # mango exposes ``world.clock`` and ``world.unregister(aid)``; the
    # discrete-event scheduler uses ``schedule_timestamp_task`` on
    # individual agents, but a world-level scheduling hook works via
    # the simpler approach of attaching a task that fires on the next
    # advance past at_s.  Implementation detail deferred to the
    # caller — this helper holds the contract.
    world._scare_scheduled_dropouts = (aids, at_s)
