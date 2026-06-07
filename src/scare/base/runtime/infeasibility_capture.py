"""Capture the LP state at the first energyflow infeasibility per run.

:func:`arm_infeasibility_capture` monkey-patches the ``energyflow`` symbol
so the first infeasible solve writes a JSON snapshot of the ``_net`` state,
then returns the result unchanged. Idempotent;
:func:`disarm_infeasibility_capture` resets to pass-through for re-arming.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mango_energy_environments.environments.restoration import (
    multi_energy_monee as _host,
)

logger = logging.getLogger(__name__)


# Single process-wide capture window; armed/disarmed per task by the runner.
_CaptureCtx: dict[str, Any] = {
    "armed": False,
    "captured": False,
    "out_path": None,
    "original": None,
}


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
            regulations[f"child-{child.id}"] = reg

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


def _capture(behavior: Any, candidate: Any, sim_time: float) -> None:
    out_path = _CaptureCtx.get("out_path")
    if out_path is None:
        return
    try:
        net = getattr(behavior, "_net", None)
        snapshot = {
            "sim_time_s": float(sim_time),
            "net": _summarize_net(net) if net is not None else None,
            "result": _summarize_result(candidate),
        }
        Path(out_path).write_text(json.dumps(snapshot, indent=2))
        logger.warning(
            "infeasibility snapshot written to %s (sim_t=%.3f)",
            out_path,
            sim_time,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("infeasibility snapshot failed: %s", exc)


def arm_infeasibility_capture(
    behavior: Any,
    out_path: Path | str,
    clock: Any = None,
) -> None:
    """Install the wrapped ``energyflow`` for the next run.

    ``behavior`` supplies the ``_net`` to sample; ``out_path``'s parent
    must exist. ``clock`` (if given) is preferred for ``sim_time_s``: the
    behavior's ``_last_energy_flow_t`` lags until the wrapper returns.
    """
    _CaptureCtx["armed"] = True
    _CaptureCtx["captured"] = False
    _CaptureCtx["out_path"] = str(out_path)
    _CaptureCtx["clock_ref"] = clock
    if _CaptureCtx.get("original") is None:
        _CaptureCtx["original"] = _host.energyflow

        def _wrapped(monee_net):
            original = _CaptureCtx["original"]
            assert original is not None
            result = original(monee_net)
            try:
                if (
                    _CaptureCtx["armed"]
                    and not _CaptureCtx["captured"]
                    and not bool(getattr(result, "success", True))
                ):
                    behavior_ref = _CaptureCtx.get("behavior_ref")
                    clock_ref = _CaptureCtx.get("clock_ref")
                    if clock_ref is not None and hasattr(clock_ref, "time"):
                        sim_time = float(clock_ref.time)
                    elif behavior_ref is not None and hasattr(
                        behavior_ref, "_last_energy_flow_t"
                    ):
                        sim_time = float(behavior_ref._last_energy_flow_t)
                    else:
                        sim_time = 0.0
                    _capture(behavior_ref, result, sim_time)
                    _CaptureCtx["captured"] = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("infeasibility wrapper post-hook failed: %s", exc)
            return result

        _host.energyflow = _wrapped
    _CaptureCtx["behavior_ref"] = behavior


def disarm_infeasibility_capture() -> None:
    """Disable capture and reset state; leaves the patch in pass-through mode."""
    _CaptureCtx["armed"] = False
    _CaptureCtx["captured"] = False
    _CaptureCtx["out_path"] = None
    _CaptureCtx["behavior_ref"] = None
    _CaptureCtx["clock_ref"] = None
