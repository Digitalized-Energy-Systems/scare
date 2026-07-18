"""Capture the LP state at the first physics-solve failure per run.

:func:`arm_infeasibility_capture` wraps the behavior's ``_solve_physics``
seam so the first failed/infeasible solve writes a JSON snapshot of the
``_net`` state, then returns the result unchanged.
:func:`disarm_infeasibility_capture` resets the window for re-arming.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scare.base.addressing import child_aid
from scare.base.runtime.trace import optimization

logger = logging.getLogger(__name__)


def _summarize_net(net: Any) -> dict[str, Any]:
    """JSON-serialisable snapshot of the monee network state.

    Captures the dimensions the LP depends on (active branches, child
    regulation factors) — enough to rebuild the failing LP standalone.
    """
    inactive_branches: list[Any] = []
    branches_total = 0
    for br in net.branches:
        branches_total += 1
        if not getattr(br, "active", True):
            inactive_branches.append(list(br.id) if isinstance(br.id, tuple) else br.id)

    regulations: dict[str, float] = {}
    for child in net.childs:
        try:
            reg = float(getattr(child.model, "regulation", 1.0))
        except (TypeError, ValueError):
            reg = float("nan")
        if reg != 1.0:
            regulations[child_aid(child.id)] = reg

    inactive_nodes: list[Any] = []
    for node in net.nodes:
        if not getattr(node, "active", True):
            inactive_nodes.append(node.id)

    return {
        "n_branches_total": branches_total,
        "n_branches_inactive": len(inactive_branches),
        "inactive_branches": inactive_branches,
        "inactive_nodes": inactive_nodes,
        "n_children_with_nondefault_regulation": len(regulations),
        "nondefault_regulations": regulations,
    }


def _summarize_result(result: Any) -> dict[str, Any]:
    if getattr(result, "failed", False):
        # monee StepResult from a skipped step: the solver raised (or reported
        # unsuccessful) and only the exception survives.
        return {
            "success": False,
            "objective": None,
            "termination_condition": "step_failed",
            "report": _stringify(getattr(result, "error", None))[:2000],
        }
    return {
        "success": bool(getattr(result, "success", False)),
        "objective": _to_float(getattr(result, "objective", None)),
        "termination_condition": _stringify(
            getattr(result, "termination_condition", None)
        ),
        "report": _stringify(getattr(result, "report", None))[:2000],
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return repr(value)


class RestoreHandle:
    """Restores a behavior's original ``_solve_physics`` seam. Idempotent."""

    def __init__(self, behavior: Any, original: Any) -> None:
        self._behavior = behavior
        self._original = original
        self._restored = False

    def restore(self) -> None:
        if self._restored:
            return
        if self._behavior is not None and self._original is not None:
            self._behavior._solve_physics = self._original
        self._restored = True


class CaptureWindow:
    """One-shot infeasibility-capture window over a behavior's ``_solve_physics``
    seam. ``arm`` patches the seam and returns a :class:`RestoreHandle`; ``disarm``
    (and ``handle.restore()``) reinstalls the ORIGINAL seam rather than leaving the
    pass-through wrapper installed, so re-arming the same instance never stacks."""

    def __init__(self) -> None:
        self._armed = False
        self._captured = False
        self._out_path: str | None = None
        self._clock_ref: Any = None
        self._handle: RestoreHandle | None = None

    def _capture(self, behavior: Any, candidate: Any, sim_time: float) -> None:
        if self._out_path is None:
            return
        try:
            net = getattr(behavior, "_net", None)
            snapshot = {
                "sim_time_s": float(sim_time),
                "net": _summarize_net(net) if net is not None else None,
                "result": _summarize_result(candidate),
            }
            Path(self._out_path).write_text(json.dumps(snapshot, indent=2))
            logger.warning(
                "infeasibility snapshot written to %s (sim_t=%.3f)",
                self._out_path,
                sim_time,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("infeasibility snapshot failed: %s", exc)

    def _current_sim_time(self, behavior: Any) -> float:
        if self._clock_ref is not None and hasattr(self._clock_ref, "time"):
            return float(self._clock_ref.time)
        if behavior is not None and hasattr(behavior, "_last_energy_flow_t"):
            return float(behavior._last_energy_flow_t)
        return 0.0

    def arm(
        self, behavior: Any, out_path: Path | str, clock: Any = None
    ) -> RestoreHandle:
        # Restore any prior patch first so wrappers never stack on re-arm.
        if self._handle is not None:
            self._handle.restore()
        self._armed = True
        self._captured = False
        self._out_path = str(out_path)
        self._clock_ref = clock
        original = behavior._solve_physics

        def _wrapped_step(dt_h, _orig=original):
            try:
                n_childs = len(behavior._net.childs)
            except Exception:  # noqa: BLE001
                n_childs = "?"
            with optimization("energyflow", solver="gurobi", n_childs=n_childs):
                step_result = _orig(dt_h)
            try:
                result = getattr(step_result, "result", None)
                failed = (
                    step_result is None
                    or getattr(step_result, "failed", False)
                    or not bool(getattr(result, "success", True))
                )
                if self._armed and not self._captured and failed:
                    self._capture(
                        behavior,
                        result if result is not None else step_result,
                        self._current_sim_time(behavior),
                    )
                    self._captured = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("infeasibility wrapper post-hook failed: %s", exc)
            return step_result

        behavior._solve_physics = _wrapped_step
        self._handle = RestoreHandle(behavior, original)
        return self._handle

    def disarm(self) -> None:
        self._armed = False
        self._captured = False
        self._out_path = None
        self._clock_ref = None
        if self._handle is not None:
            self._handle.restore()
            self._handle = None


_WINDOW = CaptureWindow()


def arm_infeasibility_capture(
    behavior: Any,
    out_path: Path | str,
    clock: Any = None,
) -> RestoreHandle:
    """Install the capture wrapper for the next run over the behavior's
    per-instance ``_solve_physics`` seam. ``out_path``'s parent must exist;
    ``clock`` (if given) is preferred for ``sim_time_s``. Returns a handle that
    restores the original seam."""
    return _WINDOW.arm(behavior, out_path, clock)


def disarm_infeasibility_capture() -> None:
    """Disable capture, reset state, and restore the original ``_solve_physics``."""
    _WINDOW.disarm()
