"""Slack-budget enforcement on external-grid children.

The LP Var bounds are widened to ``10 × budget`` for feasibility, so the LP
over-draws past the operator budget unless pushed back. This role polls the
slack draw and, when ``|draw| > budget·(1+tol)``, records a violation and
emits a ``BalanceProblem`` to trigger a rebalance round. A re-fire suppressor
keeps a persistent over-budget state from flooding the event ledger.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role

from scare.base.model import (
    SECTOR_TIMESCALE,
    BalanceProblem,
    ConstraintViolation,
    Sector,
    StartBalanceNegotiation,
)
from scare.base.runtime.diagnostics import record_event
from scare.base.util import set_slack_cp_reserve, set_slack_eff_budget

if TYPE_CHECKING:
    from mango import AgentAddress
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# Min sim-seconds between re-firing a violation on the same slack while
# over-budget persists, sized so a gossip round completes between nudges.
_REFIRE_COOLDOWN_S: float = 2.0

# Cache-gate tolerance in ``_monitor``: draw unchanged and no active
# violation ⇒ no-op tick.
_MONITOR_DELTA_TOL: float = 1e-4

# Effective-budget integral-feedback gain (per-poll move ``-gain·(|draw|-B)``).
# 0.3 damps the one-cycle draw-tracking lag; aggressive gains overshoot.
_FEEDBACK_GAIN: float = 0.3

# Floor on effective budget as a fraction of nominal. 0 lets feedback wind
# down as far as L1/L2 shedding needs; raise only for a sector with no
# sheddable demand.
_FEEDBACK_FLOOR_FRAC: float = 0.0


class SlackBudgetMonitor(Role):
    """Poll a slack's LP-chosen draw against the operator budget; on violation
    record an event and emit a BalanceProblem.

    Args:
        obs_key: ``"p_mw"`` (ExtPowerGrid) or ``"mass_flow_kgs"`` (ExtHydrGrid).
        budget: positive per-sector allowance.
        tol: relative tolerance; trigger threshold is ``budget·(1+tol)``.
        home_leader_addr: when set, an excursion also sends
            ``StartBalanceNegotiation(override_target=imbalance)`` so L1's QP
            can shed without an L2 holon; ``None`` emits only the local
            ``BalanceProblem``.
    """

    def __init__(
        self,
        behavior: RestorationEnvironmentBehavior,
        sector: Sector,
        *,
        obs_key: str,
        budget: float,
        tol: float = 0.05,
        home_leader_addr: AgentAddress | None = None,
        enable_feedback: bool = True,
        cp_aware: bool = False,
        restore: bool = False,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.obs_key = obs_key
        self.budget = float(budget)
        self.tol = float(tol)
        self.home_leader_addr = home_leader_addr
        self.enable_feedback = bool(enable_feedback)
        # CP-aware slack supply (opt-in): publish the measured over-draw so the
        # holon supply pool debits the slack's budget by it (routes the deficit
        # through holonic balancing). See enable_cp_aware_slack_supply.
        self.cp_aware = bool(cp_aware)
        # Restore lever (opt-in): when under budget, ask the leader to restore
        # load up to B (positive override). See enable_slack_restore.
        self.restore = bool(restore)
        self._restore_emit_t: float = float("-inf")
        self._violation_active: bool = False
        self._last_emit_t: float = float("-inf")
        # Last observed draw; poll skips when unchanged and no active violation.
        self._last_obs_val: float | None = None
        # Loss-compensated effective budget, tightened toward ``B - losses``
        # so the actual draw lands at B.
        self._eff_budget: float = float(budget)

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 1.0)
        self.context.schedule_periodic_task(self._monitor, delay=poll)

    def _safe_observe(self) -> dict[str, Any] | None:
        try:
            obs = self.behavior.observe(self.context.aid)
        except Exception:  # noqa: BLE001
            return None
        if not obs:
            return None
        return obs

    def _try_emit_event(self, event: Any) -> None:
        try:
            self.context.emit_event(event)
        except KeyError:
            # No co-located subscriber (defensive).
            pass

    async def _monitor(self) -> None:
        if self.budget <= 0.0:
            return
        obs = self._safe_observe()
        if obs is None or self.obs_key not in obs:
            return
        try:
            val = float(obs[self.obs_key])
        except (TypeError, ValueError):
            return
        # Effective-budget feedback runs every poll (ahead of the cache gate so
        # it converges at a steady draw). Drives the advertised budget to
        # ``B - losses`` so the actual draw settles at ``B``.
        if self.enable_feedback:
            err = abs(val) - self.budget
            # Deadband: stop correcting inside ``[B(1-tol), B(1+tol)]`` so the
            # loop settles instead of hunting around exact ``B``.
            if abs(err) > self.tol * self.budget:
                self._eff_budget -= _FEEDBACK_GAIN * err
                lo = _FEEDBACK_FLOOR_FRAC * self.budget
                if self._eff_budget > self.budget:
                    self._eff_budget = self.budget
                elif self._eff_budget < lo:
                    self._eff_budget = lo
                set_slack_eff_budget(self.behavior, self.context.aid, self._eff_budget)
        # Cache gate: skip when draw unchanged and no active violation. An
        # active violation keeps evaluating so cooldown/cleared transitions fire.
        if (
            not self._violation_active
            and self._last_obs_val is not None
            and abs(val - self._last_obs_val) < _MONITOR_DELTA_TOL
        ):
            return
        self._last_obs_val = val
        magnitude = abs(val)
        threshold = self.budget * (1.0 + self.tol)
        now = float(self.context.current_timestamp)
        # CP-aware reserve: publish the over-draw EVERY over-budget poll (not
        # cooldown-gated) so the holon supply pool tracks the latest excess and
        # the shed converges to draw == B. Cleared below when in-budget.
        if self.cp_aware:
            set_slack_cp_reserve(
                self.behavior, self.context.aid, max(0.0, magnitude - self.budget)
            )
        if magnitude > threshold:
            if (
                not self._violation_active
                or (now - self._last_emit_t) >= _REFIRE_COOLDOWN_S
            ):
                self._violation_active = True
                self._last_emit_t = now
                logger.warning(
                    "[%s] SLACK BUDGET VIOLATION %s=%.4f budget=%.4f tol=%.2f",
                    self.context.aid,
                    self.obs_key,
                    val,
                    self.budget,
                    self.tol,
                )
                record_event(
                    t=now,
                    kind="slack_budget_violation",
                    aid=self.context.aid,
                    sector=self.sector.value,
                    detail=(
                        f"{self.obs_key}={val:.4f} budget={self.budget:.4f} "
                        f"threshold={threshold:.4f}"
                    ),
                )
                # ConstraintViolation lets the aggregator count slack-budget
                # breaches alongside voltage/pressure/temperature breaches.
                self._try_emit_event(
                    ConstraintViolation(
                        sector=self.sector,
                        variable=f"slack_{self.obs_key}",
                        value=val,
                        bound_low=-self.budget,
                        bound_high=self.budget,
                        node_id=None,
                    )
                )
                # Over-budget magnitude in gossip-target convention (negative
                # ⇒ shed net load). The unsigned ``-(|val|-budget)`` targets
                # correctly for both ExtPowerGrid (import p_mw<0) and ExtHydrGrid
                # (import positive mass_flow); a signed form would mis-target.
                imbalance = -(abs(val) - self.budget)
                # Routing: with a leader, send ``override_target`` (below) and
                # skip the local BalanceProblem — it would race and win with a
                # community-local target that masks the override. Without a
                # leader, keep the BalanceProblem as a fallback.
                if self.home_leader_addr is None:
                    self._try_emit_event(
                        BalanceProblem(
                            sector=self.sector,
                            imbalance=imbalance,
                        )
                    )
                # Direct L1 path for variants without L2: send the leader a
                # ``StartBalanceNegotiation`` whose ``override_target`` is the
                # shed amount, fed straight into the L1 QP (bypasses the slack
                # policy target).
                if self.home_leader_addr is not None:
                    try:
                        await self.context.send_message(
                            StartBalanceNegotiation(
                                override_target=imbalance,
                            ),
                            receiver_addr=self.home_leader_addr,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "[%s] slack-budget override send failed: %s",
                            self.context.aid,
                            exc,
                        )
        else:
            # Cleared: re-arm so the next over-budget excursion fires again.
            self._violation_active = False
            if self.cp_aware:
                set_slack_cp_reserve(self.behavior, self.context.aid, 0.0)
            # Restore / serve-more lever: when genuinely UNDER budget (below the
            # lower deadband edge), there is unused import headroom; ask the
            # leader to restore load up to B. Positive override_target =>
            # restoration (highest priority first). Deadband-gated (only below
            # B*(1-tol)) and cooldown-gated so it can't oscillate with the shed
            # path. Restore toward the lower deadband edge, leaving a tol margin
            # so a restore does not immediately bounce into an over-budget shed.
            if (
                self.restore
                and self.home_leader_addr is not None
                and magnitude < self.budget * (1.0 - self.tol)
                and (now - self._restore_emit_t) >= _REFIRE_COOLDOWN_S
            ):
                headroom = self.budget * (1.0 - self.tol) - magnitude
                if headroom > _MONITOR_DELTA_TOL:
                    self._restore_emit_t = now
                    record_event(
                        t=now,
                        kind="slack_budget_restore",
                        aid=self.context.aid,
                        sector=self.sector.value,
                        detail=(
                            f"{self.obs_key}={val:.4f} budget={self.budget:.4f} "
                            f"restore_headroom={headroom:.4f}"
                        ),
                    )
                    try:
                        await self.context.send_message(
                            StartBalanceNegotiation(override_target=headroom),
                            receiver_addr=self.home_leader_addr,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "[%s] slack-budget restore send failed: %s",
                            self.context.aid,
                            exc,
                        )
