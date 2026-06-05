"""Contract test for the replicated-L2-kernel cutover.

The design replaces the elected coordinator in
:class:`HolonicCommunityRole` with a per-leader replicated kernel: every
leader runs :func:`allocate_supply_priority` on its locally-replicated
peer view and writes only its own slice via a self-addressed
``StartBalanceNegotiation``.

Marked ``xfail(strict=True)``: fails today (cutover not landed), turns
into a CI-red XPASS once it ships, prompting the marker's removal.

Properties pinned:

1. No ``ComponentAllocation`` envelope is ever sent (no coordinator).
2. Every ``StartBalanceNegotiation`` dispatch is self-addressed
   (``receiver_addr.aid == self.context.aid``).
3. Priority order is preserved (tier-2 fraction >= tier-4 under scarcity).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from scare.base.channel import HolonSummary
from scare.base.model import Sector, StartBalanceNegotiation
from scare.community.summary import HolonSummaryRole


@dataclass
class _SentMessage:
    payload: Any
    receiver_addr: Any


class _Addr:
    def __init__(self, aid: str) -> None:
        self.aid = aid


class _FakeContext:
    def __init__(self, aid: str) -> None:
        self.aid = aid
        self.addr = _Addr(aid)
        self.current_timestamp = 100.0
        self.sent: list[_SentMessage] = []

    async def send_message(self, payload: Any, *, receiver_addr: Any) -> None:
        self.sent.append(_SentMessage(payload=payload, receiver_addr=receiver_addr))

    def schedule_instant_task(self, coro: Any) -> None:
        self._pending = coro


def _summary(
    aid: str,
    *,
    supply_mw: float,
    demand: dict[int, float],
    served: dict[int, float],
    version: int = 1,
) -> HolonSummary:
    """Construct an extended-schema HolonSummary."""
    sec = Sector.ELECTRICITY.value
    return HolonSummary(
        publisher=aid,
        version=version,
        caused_by={},
        timestamp_s=0.0,
        sector=Sector.ELECTRICITY,
        per_tier_served_mw=dict(served),
        per_tier_demand_mw=dict(demand),
        supply_by_sector={sec: supply_mw} if supply_mw > 0.0 else {},
        demand_by_sector_priority={sec: dict(demand)} if demand else {},
        served_by_sector_priority={sec: dict(served)} if served else {},
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 3 cutover not yet landed: HolonSummaryRole still relies "
        "on the elected-coordinator path. This test pins the post-cutover "
        "contract (no ComponentAllocation, self-addressed dispatch only)."
    ),
)
def test_replicated_kernel_no_coordinator_self_dispatch_only() -> None:
    """Two-leader scarcity: 10 MW supply + 10 MW tier-2 demand at A,
    10 MW tier-4 demand at B. Each leader runs the supply-priority
    waterfall on its peer view and applies only its own slice — no
    coordinator, no ``ComponentAllocation`` fan-out.
    """
    ctx_b = _FakeContext("leader-B")  # lex-larger; a follower in the elected path

    # The kernel reads inputs off the gossiped summary mesh, not behavior.
    role_b = HolonSummaryRole(
        behavior=object(),  # type: ignore[arg-type]
        sector=Sector.ELECTRICITY,
        enable_replicated_l2_kernel=True,  # type: ignore[call-arg]
    )
    role_b._context = ctx_b  # type: ignore[attr-defined]

    role_b._peer_summaries = {  # type: ignore[attr-defined]
        "leader-A": _summary(
            "leader-A",
            supply_mw=10.0,
            demand={2: 10.0, 4: 0.0},
            served={2: 0.0, 4: 0.0},
        ),
        "leader-B": _summary(
            "leader-B",
            supply_mw=0.0,
            demand={2: 0.0, 4: 10.0},
            served={2: 0.0, 4: 0.0},
        ),
    }

    asyncio.run(role_b._run_replicated_kernel())  # type: ignore[attr-defined]

    payloads = [m.payload for m in ctx_b.sent]
    assert not any(
        type(p).__name__ == "ComponentAllocation" for p in payloads
    ), (
        "replicated kernel must not send ComponentAllocation; got: "
        f"{[type(p).__name__ for p in payloads]}"
    )

    sbn_sends = [
        m for m in ctx_b.sent if isinstance(m.payload, StartBalanceNegotiation)
    ]
    assert sbn_sends, (
        "replicated kernel must dispatch at least one StartBalanceNegotiation"
    )
    for m in sbn_sends:
        ra_aid = getattr(m.receiver_addr, "aid", None)
        assert ra_aid == ctx_b.aid, (
            "StartBalanceNegotiation must be self-addressed; got "
            f"receiver={ra_aid!r}, self={ctx_b.aid!r}"
        )

    last = sbn_sends[-1].payload
    frac_map = getattr(last, "service_fraction_by_sector_priority", None)
    assert frac_map is not None, (
        "expected service_fraction_by_sector_priority on the dispatch envelope"
    )
    sec = Sector.ELECTRICITY.value
    f2 = frac_map.get(sec, {}).get(2, 0.0)
    f4 = frac_map.get(sec, {}).get(4, 0.0)
    assert f2 >= f4 - 1e-6, (
        f"replicated kernel violated priority order: tier-2={f2}, tier-4={f4}"
    )
