"""Standalone reproducer for the monee ``run_energy_flow`` infeasibility.

Deactivating a single non-CP branch on a simbench-LV multi-energy network
makes the next ``monee.run_energy_flow(... exclude_unconnected_nodes=True)``
return ``success = False`` with ``termination_condition = infeasible``. The
MIS pinpoints per-node balance equations (``node_{n}_eq_1`` / ``_eq_3``)
relaxable by ~5e-3.

No scare imports — builds the network via ``simbench`` + ``monee``, applies
the same slack-budget shaping scare uses, and triggers the infeasibility by
deactivating one branch. Drop into the monee repo as-is to iterate on the
upstream fix.

Required packages::

    monee, simbench, pandapower, mango_energy_environments  (and a MILP
    solver pyomo can drive — gurobi / scip / appsi_highs / cbc)

Usage::

    python repro_monee_infeasibility.py
    python repro_monee_infeasibility.py --branch 380 15 --solver gurobi
    python repro_monee_infeasibility.py --simbench-code 1-LV-rural3--1-no_sw \\
        --coupling-density 0.3 --cp-size-multiplier 2.0 \\
        --slack-budget-pct 0.45 --branch 380 15

Defaults reproduce the simbench_lv_cp_heavy case (branch ``(380, 15, 0)``,
slack 0.45); use ``--branch 328 271`` for the smaller ``simbench_lv`` variant.
"""
from __future__ import annotations

import argparse
import logging
import sys

import monee
import simbench
from monee.io.from_pandapower import from_pandapower_net
from monee.model.child import ExtHydrGrid, ExtPowerGrid, PowerLoad, Sink
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import generate_supply_return_mes_based_on_power_net


# Widens the slack LP envelope over the operator's soft budget for
# headroom; mirrors experiment/restoration.py:_SLACK_LP_HEADROOM_FACTOR.
_SLACK_LP_HEADROOM_FACTOR: float = 10.0


def build_simbench_mes(
    *,
    simbench_code: str,
    coupling_density: float,
    cp_size_multiplier: float,
    replace_primary_generation: bool,
):
    """Build the same simbench multi-energy network as scare's
    ``create_large_lv_simbench``, without scare imports.
    """
    pp_net = simbench.get_simbench_net(simbench_code)
    mn = from_pandapower_net(pp_net)
    mes = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=coupling_density,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={
            "seed": 1,
            "use_hg_variants": True,
            "cp_size_multiplier": cp_size_multiplier,
            "replace_primary_generation": replace_primary_generation,
        },
        heat_kwargs={"node_based_heat_loads": True},
    )
    mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
    return mes


def apply_slack_budget(mes, fraction: float) -> None:
    """Operator slack-budget shaping — port of scare's
    ``experiment.restoration.apply_slack_budget``.

    Widens the slack ``p_mw`` / ``mass_flow`` LP envelope to
    ``±10 × fraction × Σ|nominal loads|`` for headroom after a branch
    failure shifts the imbalance. Heat-side ExtHydrGrid left unbounded.
    """
    total_p_mw = 0.0
    total_gas_mass_kgs = 0.0
    for child in mes.childs:
        m = child.model
        if isinstance(m, PowerLoad):
            total_p_mw += abs(getattr(m, "p_mw", 0.0))
        elif isinstance(m, Sink):
            try:
                grid_name = str(
                    getattr(mes.node_by_id(child.node_id).grid, "name", "")
                ).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name:
                total_gas_mass_kgs += abs(getattr(m, "mass_flow", 0.0))

    cap_p_mw = max(1e-3, fraction * total_p_mw)
    cap_gas_mass_kgs = max(1e-4, fraction * total_gas_mass_kgs)
    lp_p_mw = _SLACK_LP_HEADROOM_FACTOR * cap_p_mw
    lp_gas_mass_kgs = _SLACK_LP_HEADROOM_FACTOR * cap_gas_mass_kgs

    for child in mes.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid) and hasattr(m, "p_mw") \
                and hasattr(m.p_mw, "min"):
            m.p_mw.min = -lp_p_mw
            m.p_mw.max = lp_p_mw
            m._scare_slack_budget_mw = cap_p_mw
        elif isinstance(m, ExtHydrGrid) and hasattr(m, "mass_flow") \
                and hasattr(m.mass_flow, "min"):
            try:
                grid_name = str(
                    getattr(mes.node_by_id(child.node_id).grid, "name", "")
                ).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name:
                m.mass_flow.min = -lp_gas_mass_kgs
                m.mass_flow.max = lp_gas_mass_kgs
                m._scare_slack_budget_kgs = cap_gas_mass_kgs


def _resolve_branch_id(net, from_node: int, to_node: int):
    """Return the full ``(from, to, idx)`` branch id for the endpoints
    (either orientation); the third element disambiguates parallels.
    """
    for branch in net.branches:
        if branch.id[:2] == (from_node, to_node):
            return branch.id
        if branch.id[:2] == (to_node, from_node):
            return branch.id
    sample = [b.id for b in list(net.branches)[:5]]
    raise SystemExit(
        f"No branch between nodes {from_node} and {to_node}.  "
        f"First 5 branch ids: {sample} ..."
    )


def _solve(net, *, solver: str, phase: str) -> bool:
    print(f"\n=== {phase} ===")
    result = monee.run_energy_flow(
        net, solver=solver, exclude_unconnected_nodes=True
    )
    ok = bool(getattr(result, "success", False))
    obj = getattr(result, "objective", None)
    print(f"  success={ok}  objective={obj}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--simbench-code", default="1-LV-rural3--1-no_sw",
        help="simbench grid code (default reproduces scare's simbench_lv).",
    )
    parser.add_argument(
        "--coupling-density", type=float, default=0.3,
        help="MES coupling-point density (0.3 = scare's cp_heavy default).",
    )
    parser.add_argument(
        "--cp-size-multiplier", type=float, default=2.0,
        help="CP rated-output multiplier (2.0 = scare's cp_heavy default).",
    )
    parser.add_argument(
        "--replace-primary-generation", action="store_true", default=False,
        help="If set, CPs replace primary generation (scare's cp_dependent).",
    )
    parser.add_argument(
        "--slack-budget-pct", type=float, default=0.45,
        help="Operator slack budget (None to skip).",
    )
    parser.add_argument(
        "--branch", nargs=2, type=int, default=[380, 15],
        metavar=("FROM_NODE", "TO_NODE"),
        help="Endpoints of the branch to deactivate after the initial solve.",
    )
    parser.add_argument(
        "--solver", default="gurobi",
        help="Pyomo solver name (gurobi / scip / appsi_highs / cbc).",
    )
    args = parser.parse_args(argv)

    # Surface monee's solve-result ERROR (carries the MIS report) and
    # pyomo.core's load_solutions WARNING.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("pyomo.core").setLevel(logging.WARNING)
    logging.getLogger("pyomo.solvers").setLevel(logging.WARNING)
    logging.getLogger("monee.solver").setLevel(logging.INFO)
    logging.getLogger("pyomo.contrib.iis").setLevel(logging.INFO)

    print(
        f"Building simbench MES "
        f"(code={args.simbench_code!r}, density={args.coupling_density}, "
        f"cp_size={args.cp_size_multiplier}x, "
        f"replace_primary={args.replace_primary_generation}) ..."
    )
    net = build_simbench_mes(
        simbench_code=args.simbench_code,
        coupling_density=args.coupling_density,
        cp_size_multiplier=args.cp_size_multiplier,
        replace_primary_generation=args.replace_primary_generation,
    )
    if args.slack_budget_pct is not None:
        apply_slack_budget(net, float(args.slack_budget_pct))
    print(
        f"  nodes={len(list(net.nodes))}  branches={len(list(net.branches))}  "
        f"childs={len(list(net.childs))}"
    )

    branch_id = _resolve_branch_id(net, args.branch[0], args.branch[1])
    print(f"Target branch: {branch_id}")

    ok_before = _solve(net, solver=args.solver, phase="Initial energy flow")
    if not ok_before:
        print("\n  ! Initial solve already infeasible — grid/slack mismatch.")
        return 2

    branch = net.branch_by_id(branch_id)
    branch.active = False
    print(
        f"\nDeactivated branch {branch_id} "
        f"(model={type(branch.model).__name__}).  Re-solving ..."
    )

    ok_after = _solve(
        net, solver=args.solver, phase="Post-failure energy flow"
    )
    if ok_after:
        print(
            "\n  Reproduction did NOT trigger the infeasibility this time.\n"
            "  Try another branch (e.g. --branch 328 271) or a different "
            "grid / density."
        )
        return 0

    print(
        "\nReproduced.  Above output should include the monee.solver.pyo\n"
        "ERROR with the Minimal Intractable System (MIS).  The MIS\n"
        "names the per-node balance equations that fail; that's the\n"
        "constraint set monee needs to relax / drop on the disconnected\n"
        "side after the branch deactivation."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
