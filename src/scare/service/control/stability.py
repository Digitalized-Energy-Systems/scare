from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role

from scare.base.model import SECTOR_TIMESCALE, NegotiationFinishedEvent, Sector
from scare.base.util import (
    _last_regulate_store,
    _last_regulate_t_store,
    apply_regulate,
    constraint_allowed_fraction,
    has_gen_curtail_lock,
    lookup_priority,
    lookup_slack,
    obs_capacity,
    obs_setpoint,
)

if TYPE_CHECKING:
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


class NodeObserver(Role):
    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        *,
        control_until_s: float = 30.0,
        poll_period_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.control_until_s = control_until_s
        self.poll_period_s = poll_period_s

    def setup(self) -> None:
        self.context.schedule_periodic_task(self._observe, delay=self.poll_period_s)

    async def _observe(self) -> None:
        if self.context.current_timestamp > self.control_until_s:
            return
        try:
            obs = self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return
        if obs:
            logger.debug("[%s] obs=%s", self.context.aid, obs)


class GenerationController(Role):
    """Apply the gossip-decided setpoint as a regulate action.

    On the agent's own ``NegotiationFinishedEvent``, converts the new setpoint
    into a clamped factor, honouring the CLPU ramp and monotonic floor so this
    path doesn't bypass the safety nets gossip applies in ``_apply_setpoint``.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        ramp_to_full: bool = False,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        # Last applied factor + timestamp for the CLPU ramp bound.
        self._last_factor: float | None = None
        self._last_t: float | None = None
        self._floor: float = 0.0
        # Ramp-to-full lever (opt-in): periodically drive this generator toward
        # rated output so idle local supply is used (see enable_gen_ramp_to_full).
        self.ramp_to_full = bool(ramp_to_full)
        # Change-point mirror of the shared regulate store (read-only): the
        # ramp's respect window must also see balance-layer gen writes, which
        # bypass this role's ``_last_factor``/``_last_t``.
        self._ext_factor_seen: float | None = None
        self._ext_change_t: float = -1e9

    def setup(self) -> None:
        self.context.subscribe_event(
            self, NegotiationFinishedEvent, self._on_negotiation_finished
        )
        # Electricity-only: the elec gen curtail-lock defers GEN_RESTORE under
        # over-voltage; the HEAT lock does not, so ramping heat gens would rely
        # solely on the local t_k cap — out of scope and less safe.
        if self.ramp_to_full and self.sector is Sector.ELECTRICITY:
            poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 1.0)
            self.context.schedule_periodic_task(self._ramp_to_full, delay=poll)

    _RAMP_TOL: float = 1e-3
    # Don't ramp a generator back up within this window of a dispatch that held
    # it DOWN, else the ramp fights a legitimate over-supply/stability shed
    # (bounded but wasteful limit cycle).
    _RAMP_RESPECT_DISPATCH_S: float = 3.0

    async def _ramp_to_full(self) -> None:
        """Ramp a dispatchable generator toward rated output (reg→1.0), capped by
        the local constraint-allowed fraction. Routed through ``apply_regulate``
        with a GEN_RESTORE reason so the over-voltage curtail-ramp interlock
        defers it while the auction holds the generator down. No-op for
        loads/slacks and for generators already at (or constrained below) full.
        """
        try:
            obs = self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return
        if not obs:
            return
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap >= 0:
            return  # only dispatchable generators (cap < 0)
        if lookup_slack(self.behavior, self.context.aid) is not None:
            return  # slack injector, not a dispatchable generator
        now = float(self.context.current_timestamp)
        if has_gen_curtail_lock(self.behavior, self.context.aid, now):
            return  # over-voltage auction owns it — don't ramp into a violation
        rated = abs(cap)
        cur_factor = abs(obs_setpoint(obs, behavior=self.behavior, aid=self.context.aid))
        cur_factor = cur_factor / rated if rated > 0 else 1.0
        # Local-physics cap: don't ramp into a local over-voltage / overload.
        allowed = constraint_allowed_fraction(obs, self.sector, tier=0)
        target = min(1.0, allowed)
        if target <= cur_factor + self._RAMP_TOL:
            return  # already at full (or constrained below it) — nothing to do
        # Respect a recent dispatch that held this generator below target (a
        # stability / MW-balance over-supply shed): don't ramp it back up inside
        # the respect window, else the two loops thrash. Only ramp generators
        # dispatch is not actively holding down.
        if (
            self._last_t is not None
            and self._last_factor is not None
            and (now - self._last_t) < self._RAMP_RESPECT_DISPATCH_S
            and self._last_factor < target - self._RAMP_TOL
        ):
            return
        if self._recent_external_hold(now, target):
            return
        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            target,
            sector=self.sector.value,
            reason="gen_ramp_to_full",
            timestamp=now,
        )
        if applied:
            # Sync the change-point mirror so the ramp's own write isn't
            # counted as an external hold next tick.
            self._ext_factor_seen = _last_regulate_store(self.behavior).get(
                str(self.context.aid), self._ext_factor_seen
            )
            logger.debug(
                "[%s] gen ramp-to-full: %.2f -> %.2f",
                self.context.aid,
                cur_factor,
                target,
            )

    def _recent_external_hold(self, now: float, target: float) -> bool:
        """True when the shared regulate store shows a recent write (any path:
        balance gossip, L2 allocation, curtailment) holding this generator
        below ``target``. Writes are detected as store change-points at poll
        granularity; the cooldown timestamp store refines them when populated.
        """
        aid = str(self.context.aid)
        store_factor = _last_regulate_store(self.behavior).get(aid)
        if store_factor is None:
            return False
        if (
            self._ext_factor_seen is None
            or abs(store_factor - self._ext_factor_seen) > self._RAMP_TOL
        ):
            self._ext_factor_seen = store_factor
            self._ext_change_t = now
        store_t = _last_regulate_t_store(self.behavior).get(aid)
        last_write_t = max(
            self._ext_change_t, store_t if store_t is not None else -1e9
        )
        return (
            (now - last_write_t) < self._RAMP_RESPECT_DISPATCH_S
            and store_factor < target - self._RAMP_TOL
        )

    def _on_negotiation_finished(
        self, event: NegotiationFinishedEvent, _src: Any
    ) -> None:
        if event.sector != self.sector:
            return
        try:
            obs = self.behavior.observe(self.context.aid)
        except (AttributeError, KeyError):
            return
        if not obs:
            return
        cap = obs_capacity(obs, behavior=self.behavior, aid=self.context.aid)
        if cap == 0.0:
            return
        # Both load (cap>0) and generator (cap<0) give ``new_setpoint/cap`` in
        # [0,1] when gossip honoured its box constraints. abs() defensively
        # clamps an opposite-sign setpoint, but log it so the bug surfaces.
        raw_factor = event.new_setpoint / cap
        if (cap > 0 and event.new_setpoint < -1e-9) or (
            cap < 0 and event.new_setpoint > 1e-9
        ):
            logger.warning(
                "[%s] stability: new_setpoint sign disagrees with cap "
                "(sp=%.4g, cap=%.4g) — clamping with abs()",
                self.context.aid,
                event.new_setpoint,
                cap,
            )
            raw_factor = abs(raw_factor)
        factor = max(0.0, min(1.0, raw_factor))

        # Same safety nets as ``_apply_setpoint`` so this path can't jump.
        cfg = getattr(self.behavior, "_scare_config", None)
        enable_floor = getattr(cfg, "enable_monotonic_floor", True)
        enable_ramp = getattr(cfg, "enable_clpu_ramp", True)

        now = self.context.current_timestamp
        # CLPU ramp: bound ramp-up to ``convergence_rate``/s; decreases
        # pass through immediately (shedding can't wait).
        if (
            enable_ramp
            and self._last_factor is not None
            and self._last_t is not None
            and factor > self._last_factor
        ):
            rate = SECTOR_TIMESCALE.get(self.sector, {}).get("convergence_rate", 0.6)
            dt = max(0.0, now - self._last_t)
            factor = min(factor, self._last_factor + rate * dt)

        # Monotonic floor for loads: never regress below a restored factor.
        # Generators (cap < 0) are exempt — they ramp both ways.
        if enable_floor and cap > 0 and factor < self._floor:
            factor = self._floor

        applied = apply_regulate(
            self.behavior,
            self.context.aid,
            factor,
            sector=self.sector.value,
            reason="stability",
            timestamp=now,
            priority_tier=lookup_priority(self.behavior, self.context.aid),
        )
        if applied:
            self._last_factor = factor
            self._last_t = now
            # Ratchet the floor only on an applied factor; a suppressed write
            # (cooldown/dedup) must not raise the floor and clamp up a later shed.
            if cap > 0:
                self._floor = max(self._floor, factor)
