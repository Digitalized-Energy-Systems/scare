"""Slack-budget enforcement on external-grid children.

The LP Var bounds are widened to ``10 × budget`` for feasibility, so the LP
over-draws past the operator budget unless pushed back. Enforcement runs two
loops off the measured draw ``|obs|``:

* Effective-budget integral (the dominant lever): each poll an integral
  correction drives an advertised ``_eff_budget`` toward ``B·(1−margin)`` and
  publishes it via ``set_slack_eff_budget``. The holon supply pool reads it and
  sheds native load until the physical draw lands at B. This is what the
  ``slack_budget_compliance`` claim actually turns on.
* Over-budget override: when ``|draw| > B·(1+tol)`` the role records a
  ``slack_budget_violation``, emits a ``ConstraintViolation``, and sends the
  home leader a ``StartBalanceNegotiation(override_target)``. Its main effect is
  to *trigger* an L2 rebalance round (so the integral above is read); the
  override's own shed is clamped by the L2 priority floor. A re-fire cooldown
  keeps a persistent over-budget state from flooding the event ledger. With no
  leader it falls back to a local ``BalanceProblem``.
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
from scare.base.util import set_slack_eff_budget

if TYPE_CHECKING:
    from mango import AgentAddress
    from mango_energy_environments import RestorationEnvironmentBehavior

logger = logging.getLogger(__name__)


# Floor on sim-seconds between re-firing a violation on the same slack while
# over-budget persists. The effective cooldown is ``max(this, 2·poll)`` (see
# ``setup``) so it always spans at least two polls: on gas ``poll`` is itself
# 2.0s, so a flat 2.0s cooldown re-fired on every poll and cancelled the
# correction the previous nudge had started.
_REFIRE_COOLDOWN_S: float = 2.0

# Cache-gate tolerance in ``_monitor``: draw unchanged and no active
# violation ⇒ no-op tick.
_MONITOR_DELTA_TOL: float = 1e-4

# Effective-budget integral-feedback gain (per-poll move ``-gain·(|draw|-B)``).
# 0.3 damps the one-cycle draw-tracking lag; aggressive gains overshoot.
_FEEDBACK_GAIN: float = 0.3

# Feedback target margin (fraction of B). Targets ``B·(1−margin)`` so the settle band ``[target±tol·B]``
# tops out at B, tol below the claim's ``B·(1+tol)``; old target B put the band edge on the claim line,
# failing the peak-graded claim (eval_full_v2 gas median 1.024). Sized == tol keeps the lower edge above zero.
_FEEDBACK_TARGET_MARGIN: float = 0.05


class SlackBudgetMonitor(Role):
    """Poll a slack's LP-chosen draw against the operator budget; enforce it via
    the effective-budget integral (published to the holon supply pool) and an
    over-budget override that triggers an L2 rebalance.

    Args:
        obs_key: ``"p_mw"`` (ExtPowerGrid) or ``"mass_flow_kgs"`` (ExtHydrGrid).
        budget: positive per-sector allowance.
        tol: relative tolerance; trigger threshold is ``budget·(1+tol)``.
        home_leader_addr: when set, an excursion sends
            ``StartBalanceNegotiation(override_target=imbalance)`` to the leader
            (triggering the L2 round that reads ``_eff_budget``); ``None`` falls
            back to a local ``BalanceProblem``.
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
        # Effective re-fire cooldown; recomputed in ``setup`` as
        # ``max(_REFIRE_COOLDOWN_S, 2·poll)`` once the poll period is known.
        self._cooldown_s: float = _REFIRE_COOLDOWN_S
        # Last observed draw; poll skips when unchanged and no active violation.
        self._last_obs_val: float | None = None
        # Loss-compensated effective budget, tightened toward ``B - losses``
        # so the actual draw lands at B.
        self._eff_budget: float = float(budget)

    def setup(self) -> None:
        poll = SECTOR_TIMESCALE.get(self.sector, {}).get("poll_period_s", 1.0)
        self._cooldown_s = max(_REFIRE_COOLDOWN_S, 2.0 * poll)
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
        # it converges at a steady draw). Drives the advertised budget so the
        # actual draw settles at ``B·(1−margin)`` — the settle band's top edge is
        # then B, a ``tol`` cushion below the compliance claim's ``B·(1+tol)``.
        if self.enable_feedback:
            target = self.budget * (1.0 - _FEEDBACK_TARGET_MARGIN)
            err = abs(val) - target
            # Deadband: stop correcting inside ``[target−tol·B, target+tol·B]``
            # so the loop settles instead of hunting around the target.
            if abs(err) > self.tol * self.budget:
                self._eff_budget -= _FEEDBACK_GAIN * err
                # ``err`` is unbounded above, so a large excursion can drive the
                # integral negative; clamp to ``[0, B]`` before publishing.
                self._eff_budget = min(self.budget, max(0.0, self._eff_budget))
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
        if magnitude > threshold:
            if (
                not self._violation_active
                or (now - self._last_emit_t) >= self._cooldown_s
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
