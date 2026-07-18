"""Throwaway: render every bar plot in ``plots.py`` on synthetic data so the
restyle (outline / orientation / size / top legend / CVD patterns) can be
eyeballed. Writes PNG (+ the usual HTML/PDF) under ``c:/tmp/barplots``.

Run: ``python -m experiment.eval._render_bar_samples``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import experiment.eval.plots as plots

OUT = Path("c:/tmp/barplots")
OUT.mkdir(parents=True, exist_ok=True)

# Also emit PNG (kaleido) alongside the canonical HTML/PDF for quick viewing.
_orig_save = plots._save


def _save_with_png(fig, out_path):
    p = _orig_save(fig, out_path)
    try:
        fig.write_image(str(p.with_suffix(".png")), format="png", scale=2)
    except Exception as e:  # pragma: no cover
        print("PNG fail", out_path, e)
    return p


plots._save = _save_with_png

rng = np.random.default_rng(7)

GRIDS = ["simbench_lv_45", "simbench_mv_110", "oberrhein_220"]
VARIANTS = ["scare", "oracle", "single_level", "component_level"]
SEEDS = list(range(6))
TIERS = list(range(1, 7))
SECTORS = ["electricity", "gas", "heat"]
ABLATIONS = ["default", "no_holonic", "no_curtail_auction", "no_cp_admm", "no_qu"]
REASONS = [
    "balance",
    "self_local_gen",
    "holon_supply_priority",
    "holon_tier_alloc",
    "cp_admm",
    "curtail",
    "heat_recovery",
    "stability",
    "local_gen_fallback",
]


def _variant_quality(v: str) -> float:
    return {
        "oracle": 0.92,
        "scare": 0.82,
        "component_level": 0.74,
        "single_level": 0.6,
    }.get(v, 0.7)


rows = []
for grid in GRIDS:
    for v in VARIANTS:
        for seed in SEEDS:
            base_q = _variant_quality(v) * (0.85 + 0.15 * (GRIDS.index(grid) % 3) / 2)
            pwsf = float(np.clip(base_q + rng.normal(0, 0.05), 0, 1))
            compliant = rng.random() > (0.1 if v != "single_level" else 0.4)
            row = {
                "grid": grid,
                "variant": v,
                "seed": seed,
                "ablation": rng.choice(ABLATIONS),
                "sweep": f"slack_budget={rng.choice([0.05, 0.1, 0.2, 0.4])}",
                "n_failures": int(rng.integers(1, 6)),
                "duration_s": float(rng.uniform(50, 400)),
                "outcomes__priority_weighted_fraction": pwsf,
                # compliance gate
                "claims__slack_budget_compliance__passed": bool(compliant),
                "claims__constraint_compliance__passed": bool(
                    compliant or rng.random() > 0.3
                ),
                "claims__priority_protection__passed": bool(rng.random() > 0.2),
                "claims__restoration_quality__passed": bool(rng.random() > 0.25),
                # restoration absolute
                "outcomes__restoration__total_served_baseline_mw": float(
                    rng.uniform(20, 80)
                ),
                "outcomes__restoration__total_served_post_mw": float(
                    rng.uniform(15, 70)
                ),
                "outcomes__restoration__raw_restoration_ratio": float(
                    np.clip(base_q + rng.normal(0, 0.05), 0, 1)
                ),
                "outcomes__restoration__pwsf_restoration_ratio": float(
                    np.clip(base_q + 0.05 + rng.normal(0, 0.04), 0, 1)
                ),
                "outcomes__restoration__absolute_load_dropped_mw": float(
                    rng.uniform(1, 18)
                ),
                # diary
                "diary__finished": int(rng.integers(40, 90)),
                "diary__stalled": int(rng.integers(0, 12)),
                "diary__cancelled": int(rng.integers(0, 10)),
                "diary__timed_out": int(rng.integers(0, 6)),
                "diary__abandoned": int(rng.integers(0, 5)),
                "diary__skipped_balanced": int(rng.integers(0, 20)),
                "diary__skipped_singleton": int(rng.integers(0, 15)),
                # solver health
                "solver_infeasibilities": float(rng.uniform(0, 3)),
                "solver_warnings": float(rng.uniform(0, 5)),
                # time to stabilise (oracle hard-codes 0.0 → excluded in plot)
                "outcomes__time_to_stabilise_s": (
                    0.0 if v == "oracle" else float(rng.uniform(5, 45))
                ),
                # constraint violation integral
                "outcomes__constraint_violation_integral__electricity": float(
                    rng.uniform(0, 2)
                ),
                "outcomes__constraint_violation_integral__gas": float(
                    rng.uniform(0, 1.5)
                ),
                "outcomes__constraint_violation_integral__heat": float(
                    rng.uniform(0, 1)
                ),
                # per-variable violations
                "claims__constraint_compliance__detail__by_variable__voltage__n_violations": float(
                    rng.uniform(0, 4)
                ),
                "claims__constraint_compliance__detail__by_variable__pressure__n_violations": float(
                    rng.uniform(0, 3)
                ),
                "claims__constraint_compliance__detail__by_variable__line_load__n_violations": float(
                    rng.uniform(0, 2)
                ),
                "claims__constraint_compliance__detail__by_variable__temperature__n_violations": float(
                    rng.uniform(0, 5)
                ),
                "claims__slack_budget_compliance__detail__n_steady_breaches": float(
                    rng.uniform(0, 2)
                ),
            }
            # per-tier restoration ratios + loss split (tier 1 best, decaying)
            for t in TIERS:
                decay = max(0.1, 1.0 - 0.11 * (t - 1))
                row[f"outcomes__restoration__by_tier__{t}__ratio"] = float(
                    np.clip(decay + rng.normal(0, 0.04), 0, 1.05)
                )
                row[f"outcomes__restoration__by_tier__{t}__agent_only_ratio"] = float(
                    np.clip(decay + 0.05 + rng.normal(0, 0.03), 0, 1.05)
                )
                row[f"outcomes__restoration__by_tier__{t}__disconnect_lost_mw"] = float(
                    rng.uniform(0, 0.6) * t
                )
                row[f"outcomes__restoration__by_tier__{t}__agent_shed_mw"] = float(
                    rng.uniform(0, 0.4) * t
                )
            # per-sector restoration ratio
            for s in SECTORS:
                row[f"outcomes__restoration__by_sector__{s}__ratio"] = float(
                    np.clip(0.8 + rng.normal(0, 0.08), 0, 1.05)
                )
            # regulates by reason
            for r in REASONS:
                row[f"outcomes__regulates_by_reason__{r}"] = float(rng.uniform(0, 8))
            rows.append(row)

df = pd.DataFrame(rows)

# served-by-tier needs its own long frame
served_rows = []
for s in SECTORS:
    for t in TIERS:
        served_rows.append(
            {
                "sector": s,
                "tier": t,
                "fraction": float(
                    np.clip(1.0 - 0.1 * (t - 1) + rng.normal(0, 0.05), 0, 1)
                ),
            }
        )
served = pd.DataFrame(served_rows)

# ---- per-task timeseries / events / trajectories (for the line + overview
# plots) -------------------------------------------------------------------
FAIL_T = 12.0
t = np.linspace(0, 60, 240)


def _settle(target, drop_at=FAIL_T, depth=0.4, tau=8.0):
    """A signal that sits at ``target``, dips at the failure, recovers."""
    base = np.full_like(t, target)
    post = t >= drop_at
    base[post] = target - depth * target * np.exp(-(t[post] - drop_at) / tau)
    return base + rng.normal(0, 0.005 * max(abs(target), 1.0), t.shape)


ts = {"time_s": t, "last_feasible_solve_t": t}  # lfs==t → no staleness
# sector balances (restoration / system-balance trajectory)
ts["electrical_balance"] = _settle(1.0)
ts["gas_balance"] = _settle(1.0, depth=0.3)
ts["heat_balance"] = _settle(1.0, depth=0.25)
# control variables + envelopes (constraint-envelope + overview)
ts["avg_vm_pu"] = _settle(1.0, depth=0.06, tau=10)
ts["min_vm_pu"] = ts["avg_vm_pu"] - rng.uniform(0.01, 0.04, t.shape)
ts["max_vm_pu"] = ts["avg_vm_pu"] + rng.uniform(0.01, 0.04, t.shape)
ts["avg_pressure_pu"] = _settle(1.0, depth=0.05, tau=9)
ts["min_pressure_pu"] = ts["avg_pressure_pu"] - rng.uniform(0.01, 0.03, t.shape)
ts["max_pressure_pu"] = ts["avg_pressure_pu"] + rng.uniform(0.01, 0.03, t.shape)
ts["avg_t_k"] = _settle(350.0, depth=0.02, tau=12)
ts["min_t_k"] = ts["avg_t_k"] - rng.uniform(2, 8, t.shape)
ts["max_t_k"] = ts["avg_t_k"] + rng.uniform(2, 8, t.shape)
# line loading (overview panel 3)
ts["max_line_loading_percent"] = np.clip(_settle(70, depth=-0.6, tau=10), 0, 130)
ts["p95_line_loading_percent"] = ts["max_line_loading_percent"] * 0.85
ts["avg_line_loading_percent"] = ts["max_line_loading_percent"] * 0.55
# slack children (slack_trajectory + overview panel 1)
ts["slack__electricity__ext_grid_0"] = _settle(2.0, depth=-0.5, tau=7)
ts["slack__electricity__ext_grid_1"] = _settle(1.2, depth=-0.4, tau=7)
ts["slack__gas__ext_hydr_0"] = _settle(0.8, depth=-0.3, tau=7)
ts["slack__heat__ext_heat_0"] = _settle(0.5, depth=-0.3, tau=7)
# per-tier demand/served (overview panels 4-5)
for ti in TIERS:
    dem = 10.0 / ti
    ts[f"tier_demand_mw__{ti}"] = np.full_like(t, dem)
    keep = max(0.2, 1.0 - 0.12 * (ti - 1))
    ts[f"tier_served_mw__{ti}"] = _settle(dem * keep, depth=0.5, tau=9)
# coalition / holon balances (level-1 / level-2 line views)
for sec, n in (("electricity", 3), ("gas", 2), ("heat", 2)):
    for k in range(n):
        ts[f"coalition_balance__{sec}__{k}"] = _settle(
            1.0, depth=0.3 + 0.1 * k, tau=6 + k
        )
for sec, n in (("electricity", 2), ("gas", 1), ("heat", 1)):
    for k in range(n):
        ts[f"holon_balance__{sec}__{k}"] = _settle(1.0, depth=0.25, tau=7 + k)
timeseries = pd.DataFrame(ts)

events = pd.DataFrame(
    [
        {"t": FAIL_T, "kind": "line_failure"},
        {"t": FAIL_T + 3.0, "kind": "local_gen_request"},
        {"t": FAIL_T + 6.0, "kind": "reconfiguration_completed"},
        {"t": FAIL_T + 9.5, "kind": "constraint_violation"},
    ]
)

slack_meta = {
    "ext_grid_0": {"budget": 2.6, "lp_envelope": 3.2},
    "ext_grid_1": {"budget": 1.6, "lp_envelope": 2.0},
    "ext_hydr_0": {"budget": 1.1, "lp_envelope": 1.4},
}

# wide per-child regulation-factor trajectories (forward-filled)
traj = {"time_s": t}
for i in range(24):
    if i % 3 == 0:  # some constant (filtered out by the plot)
        traj[f"child_{i}"] = np.ones_like(t)
    else:
        traj[f"child_{i}"] = np.clip(
            _settle(1.0, depth=rng.uniform(0.2, 0.8), tau=rng.uniform(5, 12)), 0, 1.5
        )
trajectories = pd.DataFrame(traj)

jobs = [
    (
        "variant_comparison",
        lambda: plots.variant_comparison_bar(df, OUT / "variant_comparison"),
    ),
    ("ablation_impact", lambda: plots.ablation_impact_bar(df, OUT / "ablation_impact")),
    ("served_by_tier", lambda: plots.served_by_tier(served, OUT / "served_by_tier")),
    ("claims_pass_rate", lambda: plots.claims_pass_rate(df, OUT / "claims_pass_rate")),
    (
        "restoration_vs_baseline",
        lambda: plots.restoration_vs_baseline_bar(df, OUT / "restoration_vs_baseline"),
    ),
    (
        "restoration_by_tier",
        lambda: plots.restoration_by_tier_bar(df, OUT / "restoration_by_tier"),
    ),
    (
        "restoration_loss_split_by_tier",
        lambda: plots.restoration_loss_split_by_tier_bar(
            df, OUT / "restoration_loss_split_by_tier"
        ),
    ),
    (
        "agent_only_ratio_by_tier",
        lambda: plots.agent_only_ratio_by_tier_bar(
            df, OUT / "agent_only_ratio_by_tier"
        ),
    ),
    (
        "restoration_ratio_by_variant",
        lambda: plots.restoration_ratio_by_variant_bar(
            df, OUT / "restoration_ratio_by_variant"
        ),
    ),
    (
        "absolute_load_lost",
        lambda: plots.absolute_load_lost_bar(df, OUT / "absolute_load_lost"),
    ),
    ("diary_outcomes", lambda: plots.diary_outcomes_bar(df, OUT / "diary_outcomes")),
    ("solver_health", lambda: plots.solver_health_bar(df, OUT / "solver_health")),
    (
        "regulates_by_reason",
        lambda: plots.regulates_by_reason_bar(df, OUT / "regulates_by_reason"),
    ),
    (
        "restoration_by_sector",
        lambda: plots.restoration_by_sector_bar(df, OUT / "restoration_by_sector"),
    ),
    (
        "constraint_violation_integral",
        lambda: plots.constraint_violation_integral_bar(
            df, OUT / "constraint_violation_integral"
        ),
    ),
    (
        "constraint_violations_by_variable",
        lambda: plots.constraint_violations_by_variable_bar(
            df, OUT / "constraint_violations_by_variable"
        ),
    ),
    # --- non-bar plot types ---
    (
        "optimality_gap_scatter",
        lambda: plots.optimality_gap_scatter(df, OUT / "optimality_gap_scatter"),
    ),
    (
        "optimality_gap_box",
        lambda: plots.optimality_gap_box(df, OUT / "optimality_gap_box"),
    ),
    (
        "robustness_curve",
        lambda: plots.robustness_curve(
            df,
            OUT / "robustness_curve",
            sweep_param="slack_budget",
            x_label="slack budget (p.u.)",
            title="Robustness vs slack budget",
        ),
    ),
    ("cascading_curve", lambda: plots.cascading_curve(df, OUT / "cascading_curve")),
    (
        "sweep_curve_dual",
        lambda: plots.sweep_curve_dual(
            df,
            OUT / "sweep_curve_dual",
            sweep_param="slack_budget",
            x_label="slack budget (p.u.)",
            title="Served + wallclock vs slack budget",
        ),
    ),
    (
        "time_to_stabilise_box",
        lambda: plots.time_to_stabilise_box(df, OUT / "time_to_stabilise_box"),
    ),
    (
        "restoration_trajectory",
        lambda: plots.restoration_trajectory(
            timeseries, events, OUT / "restoration_trajectory", failure_t=FAIL_T
        ),
    ),
    (
        "system_balance_trajectory",
        lambda: plots.system_balance_trajectory(
            timeseries, events, OUT / "system_balance_trajectory", failure_t=FAIL_T
        ),
    ),
    (
        "constraint_envelope_trajectory",
        lambda: plots.constraint_envelope_trajectory(
            timeseries,
            events,
            OUT / "constraint_envelope_trajectory",
            failure_t=FAIL_T,
            solver_failures=0,
        ),
    ),
    (
        "slack_trajectory",
        lambda: plots.slack_trajectory(
            timeseries,
            OUT / "slack_trajectory",
            failure_t=FAIL_T,
            slack_meta=slack_meta,
        ),
    ),
    (
        "gas_slack_pressure_trajectory",
        lambda: plots.gas_slack_pressure_trajectory(
            timeseries, OUT / "gas_slack_pressure_trajectory", failure_t=FAIL_T
        ),
    ),
    (
        "coalition_balance_lines",
        lambda: plots.coalition_balance_lines(
            timeseries, OUT / "coalition_balance_lines"
        ),
    ),
    (
        "holon_balance_lines",
        lambda: plots.holon_balance_lines(timeseries, OUT / "holon_balance_lines"),
    ),
    (
        "regulation_per_child_lines",
        lambda: plots.regulation_per_child_lines(
            trajectories, OUT / "regulation_per_child_lines"
        ),
    ),
    (
        "system_state_overview",
        lambda: plots.system_state_overview(
            timeseries, events, OUT / "system_state_overview", failure_t=FAIL_T
        ),
    ),
]

for name, fn in jobs:
    try:
        fn()
        print("OK  ", name)
    except Exception as e:
        import traceback

        print("FAIL", name, e)
        traceback.print_exc()

print("\nWrote PNGs to", OUT)
