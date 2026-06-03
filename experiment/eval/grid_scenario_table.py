"""Collect the unique grid-scenarios of a campaign and emit a LaTeX table.

A *grid-scenario* is one unique combination of

    (grid factory, slack-budget policy)

appearing anywhere in a campaign config's experiments.  Everything else in
the row — the underlying simbench network, whether the grid carries
normally-open backup branches, its node count, and its coupling-point (CP)
count — is a property of the grid factory, resolved by inspecting the factory
closure and building the grid once.

Each unique combination is given a short id (``S1``, ``S2``, …) and a pretty
name so the dissertation text can refer to scenarios by name instead of by the
raw ``simbench_lv_cp_heavy_dependent`` factory key.

Usage::

    python -m experiment.eval.grid_scenario_table \
        --config experiment/configs/eval_full.json \
        --out experiment/_runs/grid_scenarios.tex

``--config`` accepts either a campaign config JSON or a campaign run directory
containing ``config.json``.  Without ``--out`` the table is printed to stdout.
Pass ``--no-build`` to skip the (slow) grid builds — node / CP counts then
read ``—`` but the rest of the table is produced instantly.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.hpc.config import CampaignConfig
from experiment.restoration import GRIDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pretty-name map
# ---------------------------------------------------------------------------
#
# Hand-written labels for the known grid factories.  A factory not in the map
# falls back to a label derived from its key (see ``_grid_pretty``), so the
# table still renders for campaigns that introduce new grids.
_GRID_PRETTY: dict[str, str] = {
    "simbench_lv_low": "LV, low CP density",
    "simbench_lv": "LV, large",
    "simbench_lv_high": "LV, high CP density",
    "simbench_lv_small": "LV, small",
    "simbench_lv_medium": "LV, medium",
    "simbench_lv_reconfig": "LV with backup branches",
    "simbench_lv_cp_heavy": "CP-heavy (2x, additive)",
    "simbench_lv_cp_dependent": "CP-dependent (replacing)",
    "simbench_lv_cp_heavy_dependent": "CP-heavy-dependent (2x, replacing)",
}

# Friendly short names for the simbench codes we use.
_SIMBENCH_PRETTY: dict[str, str] = {
    "1-LV-rural1--1-no_sw": "1-LV-rural1",
    "1-LV-rural3--1-no_sw": "1-LV-rural3",
    "1-LV-semiurb4--1-no_sw": "1-LV-semiurb4",
}


def _grid_pretty(grid_name: str) -> str:
    if grid_name in _GRID_PRETTY:
        return _GRID_PRETTY[grid_name]
    # Fallback: strip the simbench prefix and title-case the rest.
    label = grid_name.replace("simbench_", "").replace("_", " ").strip()
    return label[:1].upper() + label[1:] if label else grid_name


def _slack_label(slack_budget_pct: float | None) -> str:
    """Plain-text label for a single slack budget (e.g. ``"45%"``), with
    ``∞`` for an unbudgeted (operator-policy-free) slack.  LaTeX escaping of
    the ``%`` / ``∞`` is applied once at render time by :func:`_tex_escape`."""
    if slack_budget_pct is None:
        return "∞"
    return f"{slack_budget_pct * 100:g}%"


def _slack_cell(slacks: list[float | None]) -> str:
    """Plain-text label for the full set of slack budgets a grid is run
    under — numeric values ascending, the unbudgeted ``∞`` last (e.g.
    ``"30%, 45%, 60%, ∞"``)."""
    nums = sorted(s for s in slacks if s is not None)
    parts = [_slack_label(s) for s in nums]
    if any(s is None for s in slacks):
        parts.append(_slack_label(None))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Grid introspection
# ---------------------------------------------------------------------------


@dataclass
class GridFacts:
    """Static properties of one grid factory."""
    simbench_code: str
    backup_lines_per_sector: int
    cp_size_multiplier: float
    replace_primary_generation: bool
    density: float
    n_nodes: int | None        # None when --no-build
    n_cps: int | None
    n_backup_branches: int | None

    @property
    def has_backup(self) -> bool:
        return self.backup_lines_per_sector > 0


def _closure_params(grid_name: str) -> dict[str, Any]:
    """Read the construction params off the grid factory's closure.

    ``GRIDS[name]`` is the ``create`` thunk returned by
    ``create_large_lv_simbench``; its free variables capture the keyword
    arguments (simbench_code, backup count, CP knobs, density).  Reading them
    avoids building the grid just to learn its configuration.
    """
    factory = GRIDS[grid_name]
    free = getattr(factory, "__code__", None)
    closure = getattr(factory, "__closure__", None)
    if free is None or closure is None:
        return {}
    cells = dict(zip(free.co_freevars, closure))
    return {k: c.cell_contents for k, c in cells.items()}


def _count_cps(net: Any) -> int:
    """Number of cross-sector coupling plants on a built network.

    Mirrors ``scare.base.failure_sampling._iter_generator_candidates``: a CP is
    either a coupling *compound* (CHP / CHPHG / PowerToHeat control node) or a
    coupling *branch* (``branch.model.is_cp()`` — GasToPower / PowerToGas /
    PowerToHeatHG).  The two populations are disjoint, so the total is their
    sum.
    """
    from scare.base.failure_sampling import CHP, CHPHG, PowerToHeat

    cp_compound_classes = (CHP, CHPHG, PowerToHeat)
    n_compound = sum(
        1 for c in getattr(net, "compounds", []) or []
        if isinstance(c.model, cp_compound_classes)
    )
    n_branch = 0
    for b in net.branches:
        try:
            if b.model.is_cp():
                n_branch += 1
        except Exception:  # noqa: BLE001 — defensive on exotic branch models
            continue
    return n_compound + n_branch


def grid_facts(grid_name: str, *, build: bool = True) -> GridFacts:
    params = _closure_params(grid_name)
    n_nodes = n_cps = n_backup = None
    if build:
        net = GRIDS[grid_name]()
        n_nodes = len(net.nodes)
        n_cps = _count_cps(net)
        n_backup = sum(
            1 for b in net.branches if getattr(b.model, "backup", False)
        )
    return GridFacts(
        simbench_code=params.get("simbench_code", "?"),
        backup_lines_per_sector=int(params.get("backup_lines_per_sector", 0) or 0),
        cp_size_multiplier=float(params.get("cp_size_multiplier", 1.0) or 1.0),
        replace_primary_generation=bool(params.get("replace_primary_generation", False)),
        density=float(params.get("density", 0.0) or 0.0),
        n_nodes=n_nodes,
        n_cps=n_cps,
        n_backup_branches=n_backup,
    )


# ---------------------------------------------------------------------------
# Campaign → unique grid-scenarios
# ---------------------------------------------------------------------------


@dataclass
class GridScenario:
    scenario_id: str            # S1, S2, …
    grid_name: str
    slack_budgets: list[float | None]   # every operator slack budget the grid runs under
    facts: GridFacts

    @property
    def name(self) -> str:
        return _grid_pretty(self.grid_name)


def _load_config(path: Path) -> CampaignConfig:
    """Accept either a config JSON or a campaign run dir holding config.json."""
    if path.is_dir():
        path = path / "config.json"
    return CampaignConfig.from_json(path)


def collect_grid_scenarios(
    cfg: CampaignConfig, *, build: bool = True
) -> list[GridScenario]:
    """Walk every experiment's (grids x scenarios) and return one row per
    unique grid, in first-appearance order.

    Rows that differ *only* in their operator slack budget are collapsed:
    everything else in the row (simbench network, backup branches, node and
    CP counts) is a property of the grid, so the slack budget is carried as
    the list of distinct values the grid is exercised under.
    """
    facts_cache: dict[str, GridFacts] = {}
    slacks_by_grid: dict[str, list[float | None]] = {}
    order: list[str] = []

    for exp in cfg.experiments:
        scenarios = exp.scenarios or [{"kind": "clean"}]
        for grid in exp.grids:
            grid_name = grid.name
            if grid_name not in GRIDS:
                logger.warning(
                    "experiment %r references unknown grid %r; skipping",
                    exp.name, grid_name,
                )
                continue
            if grid_name not in slacks_by_grid:
                slacks_by_grid[grid_name] = []
                order.append(grid_name)
                facts_cache[grid_name] = grid_facts(grid_name, build=build)
            for scenario in scenarios:
                slack = scenario.get("slack_budget_pct")
                slack = float(slack) if slack is not None else None
                if slack not in slacks_by_grid[grid_name]:
                    slacks_by_grid[grid_name].append(slack)

    return [
        GridScenario(
            scenario_id=f"S{i}",
            grid_name=grid_name,
            slack_budgets=slacks_by_grid[grid_name],
            facts=facts_cache[grid_name],
        )
        for i, grid_name in enumerate(order, start=1)
    ]


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def _tex_escape(s: str) -> str:
    return (
        s.replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("∞", r"$\infty$")
    )


def _cell_int(v: int | None) -> str:
    return "---" if v is None else str(v)


def render_latex(
    scenarios: list[GridScenario], *, label: str = "tab:grid-scenarios",
) -> str:
    """Render the unique grid configurations as a booktabs LaTeX table.

    Assumes the document preamble provides ``booktabs``, ``amsmath`` (for
    ``$\\mathcal{S}_{\\rm cap}$`` / ``$\\infty$``), the ``acronym`` package
    (``\\ac{CP}``), and a fixed-width ``x`` column type (``x{4cm}``).
    """
    lines: list[str] = []
    lines.append(r"\begin{table}[thb]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Unique grid configurations evaluated in the campaign. "
        r"Backup denotes normally-open reconfiguration branches per sector; "
        r"CPs counts cross-sector coupling plants (CHP / P2G / G2P / P2H).}"
    )
    lines.append(rf"  \label{{{label}}}")
    lines.append(r"  \begin{tabular}{lx{4cm}lcccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{ID} & \textbf{Description} & \textbf{SimBench} & "
        r"\boldmath{$\mathcal{S}_{\rm cap}$} & \textbf{Backup} & "
        r"\textbf{Nodes} & \textbf{\ac{CP}} \\"
    )
    lines.append(r"    \midrule")
    for gs in scenarios:
        f = gs.facts
        simbench = _SIMBENCH_PRETTY.get(f.simbench_code, f.simbench_code)
        backup = (
            f"{f.backup_lines_per_sector}/sector" if f.has_backup else "none"
        )
        lines.append(
            "    "
            + " & ".join([
                rf"\texttt{{{gs.scenario_id}}}",
                _tex_escape(gs.name),
                _tex_escape(simbench),
                _tex_escape(_slack_cell(gs.slack_budgets)),
                backup,
                _cell_int(f.n_nodes),
                _cell_int(f.n_cps),
            ])
            + r" \\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_plain(scenarios: list[GridScenario]) -> str:
    """Compact text summary (logged to stderr / shown when no --out)."""
    rows = [("ID", "Description", "SimBench", "Slack", "Backup", "Nodes", "CPs")]
    for gs in scenarios:
        f = gs.facts
        rows.append((
            gs.scenario_id,
            gs.name,
            _SIMBENCH_PRETTY.get(f.simbench_code, f.simbench_code),
            _slack_cell(gs.slack_budgets),
            f"{f.backup_lines_per_sector}/sector" if f.has_backup else "none",
            "-" if f.n_nodes is None else str(f.n_nodes),
            "-" if f.n_cps is None else str(f.n_cps),
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for j, r in enumerate(rows):
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
        if j == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path,
        default=Path("experiment/configs/eval_full.json"),
        help="campaign config JSON, or a run dir containing config.json",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="write the LaTeX table here (default: stdout)",
    )
    p.add_argument(
        "--label", default="tab:grid-scenarios",
        help="LaTeX \\label for the table",
    )
    p.add_argument(
        "--no-build", action="store_true",
        help="skip grid builds — node/CP counts render as '---' (fast)",
    )
    return p.parse_args()


def main() -> None:
    # ``force`` so our handler wins even if an imported module (simbench /
    # pyomo) already called basicConfig and pinned the root level higher,
    # which would otherwise swallow the INFO summary.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    args = _parse_args()
    cfg = _load_config(args.config)
    scenarios = collect_grid_scenarios(cfg, build=not args.no_build)
    if not scenarios:
        raise SystemExit("No grid-scenarios found in the campaign.")
    latex = render_latex(scenarios, label=args.label)
    logger.info(
        "Collected %d unique grid-scenario(s) from %s\n%s",
        len(scenarios), args.config, render_plain(scenarios),
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(latex, encoding="utf-8")
        logger.info("Wrote LaTeX table → %s", args.out)
    else:
        print(latex)


if __name__ == "__main__":
    main()
