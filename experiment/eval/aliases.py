"""Display-only alias loader for the plotting + report layer.

The runtime pipeline (runner / aggregator) keeps the canonical long
names like ``simbench_lv_constrained_45``; only the figure / markdown
stitching layer translates to the short labels defined in
``experiment/configs/display_aliases.json``.  Missing entries pass
through unchanged.

Scenario aliasing is rule-based rather than table-based, because the
scenario keys are flat ``a=b;c=d`` strings produced by the aggregator
and writing them out long-hand for every campaign would be tedious.
``alias_scenario`` parses the key and produces a tight label like
``conc-5;skewed`` or ``cold-day;1.5x`` so the report tables stay
readable without a config entry per scenario combination.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "display_aliases.json"
)


@lru_cache(maxsize=4)
def _load(path: str | None = None) -> dict[str, dict[str, str]]:
    p = Path(path) if path else _DEFAULT_CONFIG
    if not p.exists():
        return {"grids": {}, "experiments": {}, "variants": {}}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"grids": {}, "experiments": {}, "variants": {}}
    return {
        "grids":       data.get("grids", {}) or {},
        "experiments": data.get("experiments", {}) or {},
        "variants":    data.get("variants", {}) or {},
    }


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


def alias_scenario(scenario: Any) -> str:
    """Rule-based scenario aliasing.

    ``scenario`` accepts the flat ``"a=b;c=d"`` key produced by
    :func:`experiment.hpc.aggregate._key_of` or a dict.  Returns a
    compact label that surfaces the salient knobs (failure type,
    count, priority assignment, cold-day scale) and drops the noisy
    defaults.
    """
    if scenario is None or scenario == "" or scenario == "default":
        return "default"
    if isinstance(scenario, dict):
        sc = dict(scenario)
    elif isinstance(scenario, str):
        sc = {}
        for tok in scenario.split(";"):
            if "=" in tok:
                k, v = tok.split("=", 1)
                sc[k.strip()] = v.strip()
        if not sc:
            return scenario
    else:
        return str(scenario)

    parts: list[str] = []
    kind = sc.get("kind", "clean")
    ft = sc.get("failure_type")
    n_fail = sc.get("n_failures") or sc.get("max_failures")

    if ft == "concentrated":
        parts.append(f"conc{n_fail or ''}")
    elif ft == "generator":
        parts.append(f"gen{n_fail or ''}")
    elif ft == "mixed":
        share = sc.get("generator_share")
        if share:
            try:
                parts.append(f"mix{int(float(share)*100)}")
            except Exception:
                parts.append("mix")
        else:
            parts.append("mix")
    elif ft == "branch":
        parts.append(f"br{n_fail or ''}")
    elif kind == "cold_day":
        scale = sc.get("heat_load_scale")
        parts.append(f"cold{scale or ''}x")
    elif n_fail and kind == "clean":
        parts.append(f"clean{n_fail}")
    else:
        parts.append(kind)

    pa = sc.get("priority_assignment")
    if pa and pa != "all_one":
        parts.append(pa)

    return ";".join(p for p in parts if p) or "default"


__all__ = [
    "alias_grid",
    "alias_experiment",
    "alias_variant",
    "alias_scenario",
]
