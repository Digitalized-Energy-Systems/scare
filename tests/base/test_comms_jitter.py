"""Tests for the latency-jitter communication sim.

Pins two properties of the jittered delivery delay: it is deterministic
per package, and it is quantized to a coarse grid so the number of
distinct delivery timestamps stays bounded (preventing a discrete-event
step explosion under jitter).
"""

from __future__ import annotations

from types import SimpleNamespace

from mango.simulation.communication import MessagePackage

from scare.base.runtime.comms import install_perturbation


def _make_sim(*, base_delay_s=0.02, jitter_ms=200.0, loss_pct=0.0):
    world = SimpleNamespace(communication_sim=None)
    install_perturbation(
        world,
        base_delay_s=base_delay_s,
        packet_loss_pct=loss_pct,
        latency_jitter_ms=jitter_ms,
    )
    return world.communication_sim


def _pkg(sender, receiver, t):
    return MessagePackage(
        sender_id=sender, receiver_id=receiver, sent_time=t, content=None
    )


def test_jitter_delay_is_deterministic_per_package():
    """Re-evaluating the same package must return the same delay
    (mango's determinism contract)."""
    sim = _make_sim()
    msg = _pkg("a", "b", 1.0)
    r1 = sim.calculate_communication(1.0, [msg]).package_results[0].delay_s
    r2 = sim.calculate_communication(1.0, [msg]).package_results[0].delay_s
    assert r1 == r2


def test_jitter_delays_are_grid_quantized():
    """All jittered delays land on a coarse grid, so the count of distinct
    delivery timestamps stays bounded regardless of message count."""
    sim = _make_sim(base_delay_s=0.02, jitter_ms=200.0)
    # 500 messages from distinct senders sent at the same instant.
    msgs = [_pkg(f"s{i}", "leader", 5.0) for i in range(500)]
    delays = [r.delay_s for r in sim.calculate_communication(5.0, msgs).package_results]
    distinct = set(delays)
    # quantum = max(0.02, 0.2/4) = 0.05; ±2σ ≈ ±0.4 → ~16 buckets.
    assert len(distinct) <= 25, f"too many distinct delivery times: {len(distinct)}"
    quantum = 0.05
    for d in distinct:
        # Each delay is an integer multiple of the quantum (within fp tol).
        assert abs(d / quantum - round(d / quantum)) < 1e-6


def test_jitter_quantum_scales_with_sigma():
    """A larger sigma must still bound the bucket count (quantum grows
    with sigma), not produce ever-finer distinct timestamps."""
    sim = _make_sim(base_delay_s=0.02, jitter_ms=1000.0)
    msgs = [_pkg(f"s{i}", "leader", 3.0) for i in range(500)]
    delays = {r.delay_s for r in sim.calculate_communication(3.0, msgs).package_results}
    assert len(delays) <= 25, f"too many distinct delivery times: {len(delays)}"


def test_no_jitter_is_noop():
    """Both knobs zero → no perturbation sim installed."""
    world = SimpleNamespace(communication_sim="sentinel")
    install_perturbation(
        world, base_delay_s=0.02, packet_loss_pct=0.0, latency_jitter_ms=0.0
    )
    assert world.communication_sim == "sentinel"


def test_packet_loss_preserved_with_jitter():
    """Jitter carrier must still honour packet loss."""
    sim = _make_sim(jitter_ms=200.0, loss_pct=100.0)
    msgs = [_pkg("a", "b", 1.0)]
    res = sim.calculate_communication(1.0, msgs).package_results[0]
    assert res.reached is False  # 100% loss → never reaches
