"""Slack-budget enforcement on external-grid children.

``apply_slack_budget`` stamps an operator-policy budget on every
``ExtPowerGrid`` / ``ExtHydrGrid`` model attribute (``_scare_slack_
budget_mw`` / ``_scare_slack_budget_kgs``).  The LP Var bounds are
intentionally widened to ``10 × budget`` so the energy-flow solve stays
feasible across failure-induced imbalance, which means the LP itself
will happily draw more than the budget if no other mechanism pushes
back.

This role is that mechanism.  It polls the slack child's observation at
the sector's poll period; whenever ``|obs[<p_mw|mass_flow>]| > budget
· (1 + tol)`` it:

1. Records a ``slack_budget_violation`` diagnostics event (so
   aggregators / claims can count violations campaign-wide).
2. Emits a local ``BalanceProblem`` so the co-located
   ``EnergyBalanceNegotiator`` triggers a rebalance round — the same
   path the constraint monitor already uses, so the existing gossip,
   curtailment-auction, and multihop propagation infrastructure light
   up without bespoke plumbing.

A re-fire suppressor mirrors ``GridConstraintMonitor`` so a single
persistent over-budget steady state doesn't flood the event ledger.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mango import Role

from scare.base.diagnostics import record_event
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
# while the over-budget condition persists.  Without this a continuously
# over-budget slack would emit one BalanceProblem per poll tick, which
# is both noisy and counter-productive (the rebalance round can't
# finish in one tick).  Sized so a typical gossip round (sub-second to
# a few seconds depending on group size) gets to complete before the
# next nudge.
_REFIRE_COOLDOWN_S: float = 2.0

# Tolerance for the cache-gate inside ``_monitor``: if the observed
# slack draw hasn't moved by more than this since the last poll and
# no violation is currently active, the whole tick is a no-op.
_MONITOR_DELTA_TOL: float = 1e-4


class SlackBudgetMonitor(Role):
    """Polls an external-grid slack's LP-chosen draw against the
    operator-policy budget; on violation, records an event + emits a
    BalanceProblem.

    Constructor inputs:

    - ``behavior`` — the shared ``RestorationEnvironmentBehavior``.
    - ``sector``   — sector this slack lives on (drives poll period).
    - ``obs_key``  — ``"p_mw"`` for ``ExtPowerGrid``, ``"mass_flow"``
                     for ``ExtHydrGrid``.
    - ``budget``   — positive magnitude; the operator's per-sector
                     allowance (``_scare_slack_budget_*`` value).
    - ``tol``      — relative tolerance ``[0..]``: a draw of
                     ``budget·(1+tol)`` is the trigger threshold.
                     Default mirrors ``RestorationConfiguration``.
    - ``home_leader_addr`` — address of the community leader that owns
                     the slack child's group, patched in by a post-
                     scenario-build pass once leaders are picked.  When
                     set, an over-budget excursion also sends a
                     ``StartBalanceNegotiation(override_target=-imbalance)``
                     so L1's QP curtailment can shed even without a L2
                     ``HolonicCommunityRole`` (single_level /
                     component_level fix).  When ``None``, only the
                     local ``BalanceProblem`` is emitted — preserves
                     the legacy path SCARE's L2 already consumes.
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
    ) -> None:
        super().__init__()
        self.behavior = behavior
        self.sector = sector
        self.obs_key = obs_key
        self.budget = float(budget)
        self.tol = float(tol)
        self.home_leader_addr = home_leader_addr
        self._violation_active: bool = False
        self._last_emit_t: float = float("-inf")
        # Cache of the last observed slack draw — the polling watchdog
        # skips when the value hasn't moved and no violation is active.
        self._last_obs_val: float | None = None

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
            # No co-located subscriber (rare on a slack child agent
            # which always carries an EnergyBalanceNegotiator, but kept
            # defensive in case the construction path ever skips it).
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
        # Cache gate: skip the whole pass when the slack draw hasn't
        # moved and no violation is currently active.  An active
        # violation must keep evaluating so the re-fire cooldown and
        # cleared transition still fire correctly.
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
                # ConstraintViolation lets the existing aggregator /
                # claim machinery count slack-budget breaches alongside
                # voltage / pressure / temperature breaches.
                self._try_emit_event(ConstraintViolation(
                    sector=self.sector,
                    variable=f"slack_{self.obs_key}",
                    value=val,
                    bound_low=-self.budget,
                    bound_high=self.budget,
                    node_id=None,
                ))
                # Over-budget magnitude in gossip-target convention
                # (negative ⇒ shed net load, which reduces an importing
                # slack's draw).  In the restoration setting an
                # over-budget slack is always over-*importing* (the
                # failure removed supply and the slack fills the
                # deficit), so the correct response is always to shed
                # toward the budget — independent of which raw sign the
                # slack uses to encode import.  The previous form
                # ``val - budget if val > 0 else val + budget`` is
                # ``sign(val)·(|val|-budget)``: it assumed import is
                # always the negative direction, which holds for
                # ``ExtPowerGrid`` (p_mw<0 ⇒ import) but NOT for the
                # ``ExtHydrGrid`` slacks that report import as a
                # *positive* mass_flow.  Such a slack received a positive
                # "add-load" target and was driven to the 10x LP envelope
                # (eval task-84/85: +900% over-budget gas).  ``-(|val| -
                # budget)`` is identical to the old form for negative-
                # import slacks and fixes the positive-import ones.
                imbalance = -(abs(val) - self.budget)
                # Routing:
                # * If we have a known community leader, send a
                #   ``StartBalanceNegotiation`` with the real
                #   over-budget magnitude as ``override_target`` — the
                #   leader's ``_handle_start_balance`` bypasses
                #   ``_reported_setpoint`` (which reports the slack's
                #   *target* draw, not its actual one) and feeds the
                #   correct shed magnitude straight into the L1 QP.
                # * Skip the local ``BalanceProblem`` emit in that case.
                #   The two paths race on the same ``_active`` flag,
                #   and the BalanceProblem (sync ``emit_event``) tends
                #   to win — it triggers ``trigger_balance_negotiation``
                #   which computes ``target = -Σ_local_setpoints`` (a
                #   small community-local imbalance, an order of
                #   magnitude below the actual over-budget magnitude).
                #   Suppressing the emit when the override is being
                #   sent stops the legacy path from masking the real
                #   shed target.  Without a known leader (early in
                #   scenario build, or a setup without communities),
                #   we keep the BalanceProblem emit as a fallback so
                #   *some* round fires.
                if self.home_leader_addr is None:
                    self._try_emit_event(BalanceProblem(
                        sector=self.sector,
                        imbalance=imbalance,
                    ))
                # Direct L1 path for variants without L2: send the
                # community leader a ``StartBalanceNegotiation`` whose
                # ``override_target`` is the *amount to shed* in
                # gossip-target convention.  ``_handle_start_balance``
                # bypasses ``_reported_setpoint`` (which returns the
                # slack's policy target rather than its actual draw)
                # and feeds the value straight into the L1 QP.
                #
                # Gossip-target sign (per line-overload relief and
                # _start_gossip): NEGATIVE ⇒ reduce net load by that
                # magnitude.  ``imbalance`` above is now always negative
                # (shed toward budget), so ``override_target = imbalance``
                # feeds the correct shed magnitude straight into the L1
                # QP for an over-importing slack of either sign
                # convention.
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
            # Cleared — re-arm so the next over-budget excursion fires
            # an event again (otherwise an oscillating slack would emit
            # once and stay silent).
            self._violation_active = False
