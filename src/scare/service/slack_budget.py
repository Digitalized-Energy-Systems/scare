"""Slack-budget enforcement on external-grid children.

The LP Var bounds on each ``ExtPowerGrid`` / ``ExtHydrGrid`` slack are
widened to ``10 × budget`` so the energy-flow solve stays feasible under
failure-induced imbalance, so the LP will over-draw past the operator
budget unless something pushes back. This role is that mechanism: it
polls the slack draw at the sector poll period and, when
``|obs[p_mw|mass_flow]| > budget·(1 + tol)``, records a
``slack_budget_violation`` event and emits a ``BalanceProblem`` to
trigger a rebalance round via the existing constraint-monitor path
(gossip, curtailment-auction, multihop). A re-fire suppressor keeps a
persistent over-budget state from flooding the event ledger.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role

from scare.base.diagnostics import record_event
from scare.base.util import set_slack_eff_budget
from scare.base.model import (
    SECTOR_TIMESCALE,
    BalanceProblem,
    ConstraintViolation,
    Sector,
    StartBalanceNegotiation,
)

if TYPE_CHECKING:
    from mango import AgentAddress
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# Minimum sim-seconds between re-firing a violation on the same slack
# while over-budget persists. Sized so a typical gossip round completes
# before the next nudge, avoiding one BalanceProblem per poll tick.
_REFIRE_COOLDOWN_S: float = 2.0

# Cache-gate tolerance in ``_monitor``: if the draw hasn't moved by more
# than this and no violation is active, the tick is a no-op.
_MONITOR_DELTA_TOL: float = 1e-4

# Effective-budget integral-feedback gain. Each poll moves the budget by
# ``-gain · (|draw| - B)``. Draw tracks the advertised budget ~1:1 with a
# one-cycle lag, so an aggressive gain overshoots (slack settles under
# budget); 0.3 damps that lag.
_FEEDBACK_GAIN: float = 0.3

# Floor on the effective budget as a fraction of nominal. Every load class
# carries a ``regulate`` action, so feedback can drive eff_budget to 0 as
# far as L1/L2 shedding needs; a non-zero floor would cap the wind-down and
# leave the slack structurally over-budget. Raise only for a sector with no
# sheddable demand.
_FEEDBACK_FLOOR_FRAC: float = 0.0


class SlackBudgetMonitor(Role):
    """Polls a slack's LP-chosen draw against the operator-policy budget;
    on violation, records an event and emits a BalanceProblem.

    Constructor inputs:

    - ``behavior`` — shared ``RestorationEnvironmentBehavior``.
    - ``sector``   — sector this slack lives on (drives poll period).
    - ``obs_key``  — ``"p_mw"`` (ExtPowerGrid) or ``"mass_flow"``
                     (ExtHydrGrid).
    - ``budget``   — positive per-sector allowance
                     (``_scare_slack_budget_*``).
    - ``tol``      — relative tolerance; trigger threshold is
                     ``budget·(1+tol)``.
    - ``home_leader_addr`` — address of the slack's community leader.
                     When set, an over-budget excursion also sends
                     ``StartBalanceNegotiation(override_target=imbalance)``
                     so L1's QP can shed without an L2
                     ``HolonicCommunityRole``. When ``None``, only the
                     local ``BalanceProblem`` is emitted.
    """

    def __init__(
        self,
        behavior: "RestorationEnvironmentBehavior",
        sector: Sector,
        *,
        obs_key: str,
        budget: float,
        tol: float = 0.05,
        home_leader_addr: "AgentAddress | None" = None,
        enable_feedback: bool = True,
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.obs_key = obs_key
        self.budget = float(budget)
        self.tol = float(tol)
        self.home_leader_addr = home_leader_addr
        self.enable_feedback = bool(enable_feedback)
        self._violation_active: bool = False
        self._last_emit_t: float = float("-inf")
        # Last observed draw; the poll skips when unchanged and no
        # violation is active.
        self._last_obs_val: float | None = None
        # Loss-compensated effective budget (integral feedback in
        # ``_monitor``), tightened toward ``B - losses`` so the actual
        # draw lands at B.
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
            # No co-located subscriber (defensive; a slack child normally
            # carries an EnergyBalanceNegotiator).
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
        # Effective-budget feedback runs every poll (ahead of the cache
        # gate, so it converges even at a steady draw). Integral correction
        # drives the advertised budget to ``B - losses`` so the actual draw
        # settles at ``B``, closing the loss/supply-pool gap the per-setpoint
        # control can't see.
        if self.enable_feedback:
            err = abs(val) - self.budget
            # Deadband: stop correcting once the draw is inside
            # ``[B(1-tol), B(1+tol)]`` so the loop settles instead of
            # hunting (and over-shedding) around exact ``B``.
            if abs(err) > self.tol * self.budget:
                self._eff_budget -= _FEEDBACK_GAIN * err
                lo = _FEEDBACK_FLOOR_FRAC * self.budget
                if self._eff_budget > self.budget:
                    self._eff_budget = self.budget
                elif self._eff_budget < lo:
                    self._eff_budget = lo
                set_slack_eff_budget(
                    self.behavior, self.context.aid, self._eff_budget
                )
        # Cache gate: skip when the draw is unchanged and no violation is
        # active. An active violation keeps evaluating so the re-fire
        # cooldown and cleared transition still fire.
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
        if magnitude > threshold:
            if (
                not self._violation_active
                or (now - self._last_emit_t) >= _REFIRE_COOLDOWN_S
            ):
                self._violation_active = True
                self._last_emit_t = now
                logger.warning(
                    "[%s] SLACK BUDGET VIOLATION %s=%.4f budget=%.4f tol=%.2f",
                    self.context.aid, self.obs_key, val, self.budget, self.tol,
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
                # ConstraintViolation lets the aggregator/claim machinery
                # count slack-budget breaches alongside voltage/pressure/
                # temperature breaches.
                self._try_emit_event(ConstraintViolation(
                    sector=self.sector,
                    variable=f"slack_{self.obs_key}",
                    value=val,
                    bound_low=-self.budget,
                    bound_high=self.budget,
                    node_id=None,
                ))
                # Over-budget magnitude in gossip-target convention
                # (negative ⇒ shed net load, reducing an importing slack's
                # draw). An over-budget slack is always over-importing, so
                # the response is always to shed toward budget regardless of
                # the raw sign used to encode import. ``-(|val| - budget)``
                # works for both ExtPowerGrid (import is p_mw<0) and
                # ExtHydrGrid (import is positive mass_flow); a signed form
                # would mis-target the positive-import case.
                imbalance = -(abs(val) - self.budget)
                # Routing: with a known leader, send the over-budget
                # magnitude as ``override_target`` (below) and skip the
                # local BalanceProblem. The two paths race on the same
                # ``_active`` flag and the sync BalanceProblem tends to win,
                # but it computes ``target = -Σ_local_setpoints`` — a
                # community-local imbalance far below the real shed target —
                # so it would mask the override. Without a leader, keep the
                # BalanceProblem as a fallback so some round fires.
                if self.home_leader_addr is None:
                    self._try_emit_event(BalanceProblem(
                        sector=self.sector,
                        imbalance=imbalance,
                    ))
                # Direct L1 path for variants without L2: send the leader a
                # ``StartBalanceNegotiation`` whose ``override_target`` is the
                # amount to shed. ``_handle_start_balance`` bypasses
                # ``_reported_setpoint`` (slack policy target, not actual
                # draw) and feeds it straight into the L1 QP. ``imbalance``
                # is negative (gossip-target convention: shed net load).
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
                            self.context.aid, exc,
                        )
        else:
            # Cleared: re-arm so the next over-budget excursion fires again.
            self._violation_active = False
