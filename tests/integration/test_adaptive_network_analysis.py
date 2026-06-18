"""Smoke tests for the adaptive-network analysis bundle.

Fixtures synthesise SCARE-shaped task folders (config.json,
timeseries.csv, served.csv, events.csv) whose order parameter has a
known saddle-node-like transition along a sweep axis, so the analysis
should pick it up.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from experiment.eval.adaptive_network_analysis import (
    cluster_synchronisation,
    continue_bifurcation,
    fit_critical_exponents,
    fit_mean_field_reduction,
    load_eval_root,
)


def _write_task(
    root: Path,
    task_id: str,
    *,
    n_failures: int,
    eta_curve: np.ndarray,
    served_total: float = 1.0,
    served_demand: float = 1.0,
    duration_s: float = 30.0,
) -> Path:
    """Synthesise a SCARE-shaped task folder."""
    task_dir = root / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "config.json").write_text(
        json.dumps({"task_id": task_id, "n_failures": n_failures})
    )

    n = len(eta_curve)
    t = np.linspace(0.0, duration_s, n)
    bal = (1.0 - eta_curve) * 100.0  # imbalance = 100 * (1 - eta)
    cols = ["time_s", "electrical_balance", "gas_balance", "heat_balance"]
    rows = "\n".join(f"{t[i]},{bal[i]},{bal[i]},{bal[i]}" for i in range(n))
    (task_dir / "timeseries.csv").write_text(",".join(cols) + "\n" + rows + "\n")

    (task_dir / "served.csv").write_text(
        "sector,tier,demand,served,fraction\n"
        f"electricity,1,{served_demand},{served_total},"
        f"{served_total / served_demand:.6f}\n"
    )

    (task_dir / "events.csv").write_text("t,kind,aid,sector,detail\n")
    return task_dir


@pytest.fixture
def sweep_root(tmp_path: Path) -> Path:
    """Build a synthetic sweep with a saddle-node at n_failures = 5."""
    root = tmp_path / "eval"
    rng = np.random.default_rng(42)
    for nf in range(1, 11):
        # Below threshold: full restoration. Above: residual deficit + variance.
        for seed in range(3):
            taskid = f"{nf:02d}_{seed}"
            n = 200
            t = np.linspace(0.0, 30.0, n)
            if nf < 5:
                eta = 1.0 - np.exp(-0.5 * t)
                served = 0.95 + rng.normal(0, 0.02)
            else:
                eta = 0.6 - 0.05 * (nf - 5) + 0.1 * np.sin(0.5 * t)
                eta = np.clip(eta, 0.0, 1.0)
                served = 0.55 - 0.03 * (nf - 5) + rng.normal(0, 0.05)
            served = float(np.clip(served, 0.0, 1.0))
            _write_task(
                root,
                taskid,
                n_failures=nf,
                eta_curve=eta,
                served_total=served,
            )
    return root


def test_load_eval_root(sweep_root: Path):
    runs = load_eval_root(sweep_root)
    assert len(runs) == 30  # 10 nf x 3 seeds


def test_mean_field_fit_runs(sweep_root: Path):
    runs = load_eval_root(sweep_root)
    fit = fit_mean_field_reduction(runs)
    assert fit.n_runs > 0
    # gamma should be positive (relaxation toward eta_inf)
    assert fit.gamma > 0.0
    # eta_inf is a fitted intercept; the mixed-regime sweep cannot be
    # represented exactly by the linear ODE, so require finite, not bounded.
    assert math.isfinite(fit.eta_infinity)


def test_bifurcation_continuation_detects_jump(sweep_root: Path):
    runs = load_eval_root(sweep_root)
    cont = continue_bifurcation(runs, axis="n_failures", bin_count=5)
    assert len(cont.points) >= 3
    # Sweep has a jump at nf=5; detector should flag a saddle-node candidate.
    saddles = [d for d in cont.detected if d["type"].startswith("saddle")]
    assert len(saddles) >= 1


def test_critical_exponents_fits(sweep_root: Path):
    runs = load_eval_root(sweep_root)
    cont = continue_bifurcation(runs, axis="n_failures", bin_count=5)
    fit = fit_critical_exponents(cont, p_critical=5.0, side="below")
    assert fit.n_points >= 2
    # Below-critical fit: beta/gamma magnitudes are unconstrained but finite.
    assert math.isfinite(fit.beta)
    assert math.isfinite(fit.gamma)


def test_cluster_synchronisation_handles_missing_traj(sweep_root: Path, tmp_path: Path):
    runs = load_eval_root(sweep_root)
    # No trajectories.csv: should return an empty result with a note.
    result = cluster_synchronisation(runs[0])
    assert result.n_devices == 0
    assert result.notes


def test_cluster_synchronisation_with_trajectories(tmp_path: Path):
    """Two synchronised devices vs one out-of-phase one."""
    task_dir = tmp_path / "tasks" / "001"
    task_dir.mkdir(parents=True)
    (task_dir / "config.json").write_text(json.dumps({"task_id": "001"}))
    (task_dir / "timeseries.csv").write_text("time_s\n0.0\n")
    (task_dir / "served.csv").write_text("sector,tier,demand,served,fraction\n")
    (task_dir / "events.csv").write_text("t,kind,aid,sector,detail\n")

    n = 100
    t = np.linspace(0.0, 10.0, n)
    a = np.sin(t)
    b = np.sin(t) + 0.1 * np.random.default_rng(0).standard_normal(n)
    c = np.cos(t)
    rows = "\n".join(f"{t[i]},{a[i]},{b[i]},{c[i]}" for i in range(n))
    traj = task_dir / "trajectories.csv"
    traj.write_text("time_s,dev_a,dev_b,dev_c\n" + rows + "\n")

    from experiment.eval.adaptive_network_analysis import load_task

    run = load_task(task_dir)
    assert run is not None
    result = cluster_synchronisation(run, threshold=0.6, aid_traj_csv=traj)
    assert result.n_devices == 3
    # dev_a and dev_b should cluster together, dev_c separate.
    assert result.cluster_assignment["dev_a"] == result.cluster_assignment["dev_b"]
    assert result.cluster_assignment["dev_c"] != result.cluster_assignment["dev_a"]
