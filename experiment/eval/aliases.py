"""Display-only alias loader for the plotting + report layer.

The runtime pipeline keeps canonical identifiers (e.g. ``simbench_lv_small``,
``enable_holonic=False``); only the figure / markdown layer translates to the
display names in ``experiment/configs/display_aliases.json``, which follow the
dissertation's naming (grid IDs ``S1``..``S8``, chapter prose names for the
experiments, ``SCARE`` / ``Oracle`` / ``Single-level`` / ``Component-level``
variants). Missing entries pass through unchanged.

Scenario / ablation / sweep aliasing is rule-based, not table-based: their
keys are flat ``a=b;c=d`` strings, and :func:`alias_scenario`,
:func:`alias_ablation` and :func:`alias_sweep` parse them into readable labels
(e.g. ``concentrated ×5 · skewed priorities``, ``no holonic waterfall``)
without a config entry per combination.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "display_aliases.json"
)

_SECTIONS = ("grids", "experiments", "variants", "ablation_flags")


@lru_cache(maxsize=4)
def _load(path: str | None = None) -> dict[str, dict[str, str]]:
    p = Path(path) if path else _DEFAULT_CONFIG
    if not p.exists():
        return {s: {} for s in _SECTIONS}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {s: {} for s in _SECTIONS}
    return {s: data.get(s, {}) or {} for s in _SECTIONS}


def alias_grid(name: str, *, config_path: str | None = None) -> str:
    if not isinstance(name, str):
        return str(name)
    return _load(config_path)["grids"].get(name, name)


def alias_experiment(name: str, *, config_path: str | None = None) -> str:
    if not isinstance(name, str):
        return str(name)
    return _load(config_path)["experiments"].get(name, name)


def alias_variant(name: str, *, config_path: str | None = None) -> str:
    if not isinstance(name, str):
        return str(name)
    return _load(config_path)["variants"].get(name, name)


def _parse_flat_key(key: Any) -> list[tuple[str, str]] | None:
    """``"a=b;c=d"`` → ``[("a", "b"), ("c", "d")]``; ``None`` when nothing
    parses (caller falls back to the raw string)."""
    pairs: list[tuple[str, str]] = []
    for tok in str(key).split(";"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            pairs.append((k.strip(), v.strip()))
    return pairs or None


# Heuristic prettifier for flag / parameter names missing from the table:
# strip ``enable_``, space out underscores, restore the common acronyms.
_TOKEN_CASE = {
    "cp": "CP",
    "cps": "CPs",
    "admm": "ADMM",
    "qp": "QP",
    "qv": "Q(U)",
    "l1": "Layer-1",
    "l2": "Layer-2",
    "l3": "Layer-3",
    "ttl": "TTL",
    "clpu": "CLPU",
    "vvw": "Volt-VAr-Watt",
    "pwsf": "PWSF",
}


def _pretty_flag(key: str, *, config_path: str | None = None) -> str:
    table = _load(config_path)["ablation_flags"]
    if key in table:
        return table[key]
    stem = key[len("enable_") :] if key.startswith("enable_") else key
    return " ".join(_TOKEN_CASE.get(t, t) for t in stem.split("_"))


def alias_ablation(key: Any, *, config_path: str | None = None) -> str:
    """Readable label for a flat ablation key.

    ``enable_x=False`` reads as the mechanism being removed ("no holonic
    waterfall"), ``enable_x=True`` as being armed; other parameters render
    as ``name = value``. The ``default`` arm is the unablated full system.
    """
    if key is None or key == "" or key == "default":
        return "full system"
    pairs = _parse_flat_key(key)
    if pairs is None:
        return str(key)
    parts: list[str] = []
    for k, v in pairs:
        pretty = _pretty_flag(k, config_path=config_path)
        if k.startswith("enable_") and v in ("False", "false"):
            parts.append(f"no {pretty}")
        elif k.startswith("enable_") and v in ("True", "true"):
            parts.append(f"with {pretty}")
        else:
            parts.append(f"{pretty} = {v}")
    return ", ".join(parts)


def alias_sweep(key: Any, *, config_path: str | None = None) -> str:
    """Readable label for a flat sweep key (``param=value``)."""
    if key is None or key == "" or key == "default":
        return "default"
    pairs = _parse_flat_key(key)
    if pairs is None:
        return str(key)
    return ", ".join(
        f"{_pretty_flag(k, config_path=config_path)} = {v}" for k, v in pairs
    )


def alias_scenario(scenario: Any) -> str:
    """Rule-based scenario aliasing.

    ``scenario`` accepts the flat ``"a=b;c=d"`` key produced by
    :func:`experiment.hpc.aggregate._key_of` or a dict.  Returns a readable
    label surfacing the salient knobs (failure composition, count, priority
    assignment, slack budget, cold-day scale); unrecognised keys are appended
    verbatim so distinct scenarios never collapse to the same label.
    """
    if scenario is None or scenario == "" or scenario == "default":
        return "default"
    if isinstance(scenario, dict):
        sc = {str(k): str(v) for k, v in scenario.items()}
    elif isinstance(scenario, str):
        sc = dict(_parse_flat_key(scenario) or [])
        if not sc:
            return scenario
    else:
        return str(scenario)

    handled = {
        "kind",
        "failure_type",
        "n_failures",
        "max_failures",
        "generator_share",
        "heat_load_scale",
        "priority_assignment",
        "slack_budget_pct",
    }
    parts: list[str] = []
    kind = sc.get("kind", "clean")
    ft = sc.get("failure_type")
    n_fail = sc.get("n_failures") or sc.get("max_failures")
    times = f" ×{n_fail}" if n_fail else ""

    if ft == "concentrated":
        parts.append(f"concentrated{times}")
    elif ft == "island":
        parts.append(f"islanding{times}")
    elif ft == "generator":
        parts.append(f"generator outage{times}")
    elif ft == "mixed":
        share = sc.get("generator_share")
        try:
            parts.append(f"mixed ({int(float(share) * 100)}% generators)")
        except (TypeError, ValueError):
            parts.append("mixed failures")
    elif ft == "branch":
        parts.append(f"branch failures{times}")
    elif kind == "cold_day":
        scale = sc.get("heat_load_scale")
        parts.append(f"cold-day (heat ×{scale})" if scale else "cold-day")
    elif n_fail and kind == "clean":
        parts.append(f"random failures{times}")
    else:
        parts.append(kind.replace("_", "-"))

    pa = sc.get("priority_assignment")
    if pa and pa != "all_one":
        parts.append(f"{pa} priorities")

    sb = sc.get("slack_budget_pct")
    if sb:
        try:
            parts.append(f"slack {float(sb) * 100:g}%")
        except ValueError:
            parts.append(f"slack {sb}")

    parts.extend(f"{k}={v}" for k, v in sc.items() if k not in handled)
    return " · ".join(p for p in parts if p) or "default"


__all__ = [
    "alias_grid",
    "alias_experiment",
    "alias_variant",
    "alias_ablation",
    "alias_sweep",
    "alias_scenario",
]
