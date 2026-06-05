"""Adaptive-network post-hoc analysis bundle for SCARE eval runs.

Co-evolutionary network dynamics tools, decoupled from the simulator:

* **C.2 + C.3** — :func:`fit_mean_field_reduction` /
  :func:`continue_bifurcation`: collapse a sweep of run trajectories
  into a low-dimensional ODE for the order parameter $\\bar\\eta(t)$
  and continue its equilibrium across a parameter axis to detect
  saddle-node / Hopf bifurcations.
* **C.5** — :func:`cluster_synchronisation`: cluster the per-device
  regulation trajectories $\\{r_i(t)\\}$ by Pearson correlation and
  compare against the static topology.
* **C.7** — :func:`fit_critical_exponents`: fit power laws
  $\\bar\\eta_\\infty \\sim |p - p_c|^{\\beta}$ and
  $\\sigma^2 \\sim |p - p_c|^{-\\gamma}$ near a candidate critical point.

Reads the standard eval task layout: one folder per task with
``timeseries.csv``, ``served.csv``, ``events.csv``, ``config.json``.

CLI examples
============

  python -m experiment.eval.adaptive_network_analysis \\
      --eval-root experiment/_runs/eval/<run> \\
      --sweep-axis topo_rate_hz                                 \\
      --output experiment/_runs/analysis_adaptive

  python -m experiment.eval.adaptive_network_analysis ... --only mean-field
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


# =====================================================================
# Common loaders
# =====================================================================


@dataclass
class TaskRun:
    """One simulation's worth of data, lifted into memory."""

    task_id: str
    task_dir: Path
    config: dict
    # 2-D matrix [time x quantity]; column 0 is time_s.
    timeseries: np.ndarray
    timeseries_columns: list[str]
    served: list[dict]
    events: list[dict]

    @property
    def t(self) -> np.ndarray:
        return self.timeseries[:, 0]

    def col(self, name: str) -> np.ndarray | None:
        if name not in self.timeseries_columns:
            return None
        return self.timeseries[:, self.timeseries_columns.index(name)]

    def served_fraction(self) -> float:
        if not self.served:
            return 0.0
        total_dem = 0.0
        total_srv = 0.0
        for r in self.served:
            d = r.get("demand")
            s = r.get("served")
            if not isinstance(d, (int, float)) or not isinstance(s, (int, float)):
                continue
            if not (math.isfinite(d) and math.isfinite(s)):
                continue
            total_dem += d
            total_srv += s
        return total_srv / total_dem if total_dem > 0 else 0.0

    # Event kinds signifying a topology mutation; names match
    # ``scare.base.diagnostics.record_event``.
    _TOPOLOGY_EVENT_KINDS = frozenset({
        "line_failure",
        "branch_failure",
        "tie_switch_close",
        "reconfiguration_completed",
    })

    def topology_dirtiness(self) -> float:
        """Branch state changes per second over the run."""
        if not self.events:
            return 0.0
        rate = 0
        max_t = 0.0
        for ev in self.events:
            try:
                t_ev = float(ev.get("t", 0.0))
            except (TypeError, ValueError):
                continue
            if t_ev > max_t:
                max_t = t_ev
            if ev.get("kind", "") in self._TOPOLOGY_EVENT_KINDS:
                rate += 1
        if rate == 0 or max_t <= 0:
            return 0.0
        return rate / max_t


def load_task(task_dir: Path) -> TaskRun | None:
    """Load one task folder.  Returns ``None`` on missing fields."""
    config_path = task_dir / "config.json"
    ts_path = task_dir / "timeseries.csv"
    served_path = task_dir / "served.csv"
    events_path = task_dir / "events.csv"
    if not (config_path.exists() and ts_path.exists()):
        return None
    with config_path.open() as f:
        config = json.load(f)
    cols, ts = _read_csv_floats(ts_path)
    served = _read_csv_dicts(served_path) if served_path.exists() else []
    events = _read_csv_dicts(events_path) if events_path.exists() else []
    return TaskRun(
        task_id=task_dir.name,
        task_dir=task_dir,
        config=config,
        timeseries=ts,
        timeseries_columns=cols,
        served=served,
        events=events,
    )


def load_eval_root(eval_root: Path) -> list[TaskRun]:
    tasks_dir = eval_root / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"No tasks/ subdir in {eval_root}")
    runs: list[TaskRun] = []
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        run = load_task(d)
        if run is not None:
            runs.append(run)
    return runs


def discover_axes(runs: Iterable[TaskRun], *, max_depth: int = 2) -> dict[str, set]:
    """Return dotted-path config keys whose value differs across runs.

    Candidate sweep axes: anything with >=2 distinct values.  Returns
    ``{dotted_key: set_of_values}``; values are str-coerced for
    hashability.
    """
    seen: dict[str, set] = defaultdict(set)

    def walk(prefix: str, val, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(val, dict):
            for k, v in val.items():
                walk(f"{prefix}.{k}" if prefix else k, v, depth + 1)
        elif isinstance(val, (list, tuple)):
            seen[prefix].add(repr(val))
        else:
            seen[prefix].add(str(val))

    for r in runs:
        walk("", r.config, 0)
    # Single-valued keys are constants, not axes.
    return {k: v for k, v in seen.items() if len(v) >= 2}


def filter_runs(
    runs: Iterable[TaskRun], filter_spec: str | None
) -> list[TaskRun]:
    """Stratify a run set by ``key=value`` predicates.

    ``filter_spec`` is a comma-separated list, e.g.
    ``"variant=scare,grid=simbench_lv"``.  Empty/None is the identity.
    Dotted keys (``scenario.kind=clean``) are supported; each predicate
    matches exactly via str-coerced comparison.
    """
    runs = list(runs)
    if not filter_spec:
        return runs
    preds: list[tuple[str, str]] = []
    for clause in filter_spec.split(","):
        clause = clause.strip()
        if not clause or "=" not in clause:
            continue
        k, v = clause.split("=", 1)
        preds.append((k.strip(), v.strip()))
    if not preds:
        return runs
    out: list[TaskRun] = []
    for r in runs:
        if all(str(_config_get(r.config, k)) == v for k, v in preds):
            out.append(r)
    return out


def _read_csv_floats(path: Path) -> tuple[list[str], np.ndarray]:
    rows: list[list[float]] = []
    cols: list[str] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        cols = next(reader, [])
        for parts in reader:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(cols)))
    return cols, arr


_NUMERIC_TOKENS = {"nan", "inf", "+inf", "-inf"}


def _read_csv_dicts(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        cols = next(reader, [])
        for parts in reader:
            if len(parts) != len(cols):
                continue
            out.append({c: _coerce_scalar(v) for c, v in zip(cols, parts)})
    return out


def _coerce_scalar(s: str):
    """Return float for numeric/nan/inf tokens, else the original string."""
    sl = s.strip().lower()
    if sl in _NUMERIC_TOKENS:
        return float(sl)
    try:
        return float(s)
    except ValueError:
        return s


# =====================================================================
# C.2 + C.3 — Mean-field reduction + bifurcation continuation
# =====================================================================


@dataclass
class MeanFieldFit:
    """Output of :func:`fit_mean_field_reduction`.

    Encodes the closed-form ODE
    $\\dot{\\bar\\eta} = \\gamma\\,(T_{\\infty}(\\bar K) - \\bar\\eta)
                  - \\nu_{\\rm topo}\\,(\\bar\\eta - \\bar\\eta_{\\infty}(\\bar K))$
    parameters fit jointly across all runs in a sweep.
    """

    gamma: float
    nu_topo: float
    eta_infinity: float
    rms_residual: float
    n_runs: int


def fit_mean_field_reduction(
    runs: Iterable[TaskRun],
    *,
    served_col: str | None = None,
    sample_period_s: float | None = None,
    min_pairs: int = 4,
) -> MeanFieldFit:
    """Fit the mean-field ODE parameters across a set of runs.

    The order parameter $\\bar\\eta(t)$ is the served-load fraction over
    time, falling back to a normalised ``electrical_balance`` proxy when
    served data is unavailable.  Fit by linear regression on
    $\\dot{\\bar\\eta} \\approx -\\gamma_{\\rm eff}\\,(\\bar\\eta - \\eta_\\infty)$,
    a one-parameter compression of the slow-fast joint dynamics.
    """
    runs = list(runs)
    if not runs:
        raise ValueError("fit_mean_field_reduction: no runs")

    # Collect (eta, eta_dot) pairs across runs; decimation aims for
    # ~16 bins per run, never below 0.05 s.
    pairs_eta: list[float] = []
    pairs_dot: list[float] = []

    for r in runs:
        eta_series = _eta_series(r, served_col=served_col)
        if eta_series is None or len(eta_series) < 4:
            continue
        t = r.t
        # Flat trajectories carry no information.
        if float(np.std(eta_series)) < 1e-9:
            continue
        period = sample_period_s
        if period is None:
            span = float(t[-1] - t[0])
            period = max(0.05, span / 16.0) if span > 0 else 1.0
        decim = _decimate_to_period(t, eta_series, period)
        if decim is None:
            continue
        td, eta = decim
        if len(eta) < 3:
            continue
        # Forward differences for eta_dot.
        dt = np.diff(td)
        eta_dot = np.diff(eta) / np.where(dt > 0, dt, 1.0)
        pairs_eta.extend(eta[:-1].tolist())
        pairs_dot.extend(eta_dot.tolist())

    if len(pairs_eta) < min_pairs:
        raise ValueError(
            f"fit_mean_field_reduction: only {len(pairs_eta)} usable pairs "
            f"(need {min_pairs}); trajectories may be flat or runs too short."
        )

    eta_arr = np.asarray(pairs_eta, dtype=float)
    dot_arr = np.asarray(pairs_dot, dtype=float)

    # Fit eta_dot = a*eta + b, where a = -gamma, b = gamma*eta_inf.
    A = np.vstack([eta_arr, np.ones_like(eta_arr)]).T
    coef, residuals, _, _ = np.linalg.lstsq(A, dot_arr, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    gamma = max(0.0, -a)
    eta_inf = b / gamma if gamma > 0 else float("nan")
    pred = a * eta_arr + b
    rms = float(np.sqrt(np.mean((dot_arr - pred) ** 2)))

    # nu_topo proxy: mean topology dirtiness across runs.
    nu_topo = float(np.mean([r.topology_dirtiness() for r in runs]))

    return MeanFieldFit(
        gamma=gamma,
        nu_topo=nu_topo,
        eta_infinity=eta_inf,
        rms_residual=rms,
        n_runs=len(runs),
    )


def _eta_series(run: TaskRun, *, served_col: str | None = None) -> np.ndarray | None:
    """Estimate the served-load fraction over time.

    Tries (in order) ``served_col``, ``electrical_balance``,
    ``gas_balance``, ``heat_balance``; first column with non-zero
    variance wins.  Balance columns are normalised so an initially
    imbalanced system starts at $\\eta = 0$ and converges toward
    $\\eta = 1$ as the imbalance is absorbed.
    """
    candidates: list[str] = []
    if served_col is not None:
        candidates.append(served_col)
    candidates.extend(["electrical_balance", "gas_balance", "heat_balance"])

    for c in candidates:
        col = run.col(c)
        if col is None or len(col) < 2:
            continue
        if c == served_col:
            denom = float(np.max(np.abs(col))) or 1.0
            return col / denom
        init = abs(float(col[0])) or 1.0
        series = 1.0 - (np.abs(col) / init)
        if float(np.std(series)) > 1e-9:
            return series
    return None


def _decimate_to_period(
    t: np.ndarray, x: np.ndarray, period: float
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(t) < 2:
        return None
    t_min, t_max = float(t[0]), float(t[-1])
    if t_max <= t_min:
        return None
    n_bins = max(1, int((t_max - t_min) / period))
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_idx = np.digitize(t, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    out_t = np.zeros(n_bins)
    out_x = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i, b in enumerate(bin_idx):
        out_t[b] += t[i]
        out_x[b] += x[i]
        counts[b] += 1
    mask = counts > 0
    return out_t[mask] / counts[mask], out_x[mask] / counts[mask]


@dataclass
class BifurcationContinuationPoint:
    """One sample on the equilibrium continuation $\\bar\\eta(p)$."""

    parameter_value: float
    eta_eq: float
    eta_var: float
    n_runs: int


@dataclass
class BifurcationContinuation:
    axis: str
    points: list[BifurcationContinuationPoint] = field(default_factory=list)
    detected: list[dict] = field(default_factory=list)


def continue_bifurcation(
    runs: Iterable[TaskRun],
    *,
    axis: str,
    extract: callable | None = None,
    bin_count: int = 8,
) -> BifurcationContinuation:
    """Continue the equilibrium $\\bar\\eta_\\infty(p)$ along a swept axis $p$.

    ``axis`` is read from each run's ``config.json``.  Runs are bucketed
    into ``bin_count`` quantile bins on the axis; each bin records mean
    and variance of the run-end served fraction.  Bins where variance
    jumps abruptly relative to neighbours flag candidate bifurcations.
    """
    runs = list(runs)
    if not runs:
        raise ValueError("continue_bifurcation: no runs")

    if extract is None:
        def extract(r: TaskRun) -> float | None:
            v = _config_get(r.config, axis)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

    samples: list[tuple[float, float]] = []
    for r in runs:
        p = extract(r)
        if p is None:
            continue
        samples.append((p, r.served_fraction()))
    if not samples:
        return BifurcationContinuation(axis=axis)

    samples.sort(key=lambda x: x[0])
    parr = np.asarray([s[0] for s in samples])
    earr = np.asarray([s[1] for s in samples])

    # Quantile bins of the parameter axis.
    qs = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(parr, qs)
    points: list[BifurcationContinuationPoint] = []
    for i in range(bin_count):
        lo, hi = edges[i], edges[i + 1]
        if i == bin_count - 1:
            mask = (parr >= lo) & (parr <= hi)
        else:
            mask = (parr >= lo) & (parr < hi)
        if not mask.any():
            continue
        points.append(
            BifurcationContinuationPoint(
                parameter_value=float(np.mean(parr[mask])),
                eta_eq=float(np.mean(earr[mask])),
                eta_var=float(np.var(earr[mask])),
                n_runs=int(mask.sum()),
            )
        )

    # Relative thresholds so detection works whether the order parameter
    # spans [0, 1] or [0.95, 1.0]:
    #   * saddle-node: largest one-step delta-eta / eta range > 0.25
    #   * critical slowing down: variance > 3x the bin-wise median
    # Need >=3 bins: with 2, the only step IS the range, so the
    # relative-step detector trivially fires.
    detected: list[dict] = []
    if len(points) < 3:
        return BifurcationContinuation(axis=axis, points=points, detected=detected)

    eta_range = max(p.eta_eq for p in points) - min(p.eta_eq for p in points)
    eta_range = max(eta_range, 1e-9)
    var_median = float(np.median([p.eta_var for p in points]))

    for k in range(len(points)):
        prev = points[k - 1] if k > 0 else None
        nxt = points[k + 1] if k < len(points) - 1 else None
        # Saddle-node: significant step into or out of this bin.
        steps = []
        if prev is not None:
            steps.append(abs(points[k].eta_eq - prev.eta_eq))
        if nxt is not None:
            steps.append(abs(nxt.eta_eq - points[k].eta_eq))
        if steps and max(steps) / eta_range > 0.25:
            detected.append({
                "type": "saddle-node?",
                "near": points[k].parameter_value,
                "max_step_rel": float(max(steps) / eta_range),
            })
        # Critical slowing down: variance peak relative to global median.
        if var_median > 0 and points[k].eta_var > 3.0 * var_median:
            detected.append({
                "type": "critical-slowing-down?",
                "near": points[k].parameter_value,
                "var_peak": float(points[k].eta_var),
                "var_ratio_vs_median": float(points[k].eta_var / var_median),
            })

    return BifurcationContinuation(axis=axis, points=points, detected=detected)


def _config_get(cfg: dict, dotted: str):
    """Walk a dotted path into a nested config dict."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# =====================================================================
# C.5 — Cluster synchronisation
# =====================================================================


@dataclass
class ClusterSynchronisationResult:
    n_devices: int
    n_clusters: int
    cluster_assignment: dict[str, int]   # aid -> cluster id
    static_groups: dict[str, str]        # aid -> static group label
    dynamic_vs_static_score: float       # Rand-style partition overlap
    notes: list[str] = field(default_factory=list)


def cluster_synchronisation(
    run: TaskRun,
    *,
    threshold: float = 0.6,
    aid_traj_csv: Path | None = None,
) -> ClusterSynchronisationResult:
    """Cluster per-device regulation trajectories by Pearson correlation
    and compare against the static topology grouping.

    Requires a per-aid ``trajectories.csv`` in the task folder (the
    default ``timeseries.csv`` is per-system).  Returns an empty result
    with a note when that file is missing.
    """
    notes: list[str] = []
    traj_path = aid_traj_csv if aid_traj_csv is not None else (
        run.task_dir / "trajectories.csv"
    )

    if not traj_path.exists():
        notes.append(
            "trajectories.csv missing; instrument diagnostics.py to log "
            "per-aid r_i(t) for cluster-sync analysis."
        )
        return ClusterSynchronisationResult(
            n_devices=0, n_clusters=0, cluster_assignment={},
            static_groups={}, dynamic_vs_static_score=float("nan"),
            notes=notes,
        )

    cols, ts = _read_csv_floats(traj_path)
    if not cols or len(cols) < 3:
        notes.append("trajectories.csv has no per-aid columns")
        return ClusterSynchronisationResult(
            n_devices=0, n_clusters=0, cluster_assignment={},
            static_groups={}, dynamic_vs_static_score=float("nan"),
            notes=notes,
        )

    # First column is time; the rest are aids.
    aid_cols = cols[1:]
    X = ts[:, 1:]
    if X.shape[0] < 4:
        notes.append("not enough samples to correlate")
        return ClusterSynchronisationResult(
            n_devices=X.shape[1], n_clusters=0, cluster_assignment={},
            static_groups={}, dynamic_vs_static_score=float("nan"),
            notes=notes,
        )

    # Correlation matrix (Pearson) — handle near-constant columns.
    Xz = X - np.mean(X, axis=0, keepdims=True)
    norms = np.linalg.norm(Xz, axis=0)
    nz = norms > 1e-9
    keep_cols = np.where(nz)[0]
    if keep_cols.size < 2:
        notes.append("trajectories are constant; no co-variance")
        return ClusterSynchronisationResult(
            n_devices=X.shape[1], n_clusters=0, cluster_assignment={},
            static_groups={}, dynamic_vs_static_score=float("nan"),
            notes=notes,
        )
    Xn = Xz[:, keep_cols] / norms[keep_cols]
    C = (Xn.T @ Xn)

    # Greedy single-linkage on |C| > threshold.
    n = C.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if abs(C[i, j]) > threshold:
                union(i, j)

    # Derive cluster labels for the kept aids.
    clusters: dict[str, int] = {}
    next_id = 0
    label_for_root: dict[int, int] = {}
    for li, ci in enumerate(keep_cols):
        root = find(li)
        if root not in label_for_root:
            label_for_root[root] = next_id
            next_id += 1
        clusters[aid_cols[ci]] = label_for_root[root]

    # Static groups: read from ``static_groups.csv`` if present.
    static_groups: dict[str, str] = {}
    sg_path = traj_path.parent / "static_groups.csv"
    if sg_path.exists():
        rows = _read_csv_dicts(sg_path)
        for r in rows:
            aid = r.get("aid")
            grp = r.get("group")
            if aid and grp:
                static_groups[str(aid)] = str(grp)

    score = _partition_overlap(clusters, static_groups)

    return ClusterSynchronisationResult(
        n_devices=len(aid_cols),
        n_clusters=next_id,
        cluster_assignment=clusters,
        static_groups=static_groups,
        dynamic_vs_static_score=score,
        notes=notes,
    )


def _partition_overlap(
    a: dict[str, int], b: dict[str, str]
) -> float:
    """Pair-counting Rand index (unadjusted) between two partitions.

    Returns NaN if either partition is empty.  Avoids a scikit-learn
    dependency.
    """
    if not a or not b:
        return float("nan")
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return float("nan")
    n = len(keys)
    same_a = 0
    same_b = 0
    same_both = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            ai, bj = a[keys[i]], a[keys[j]]
            ai2, bj2 = b[keys[i]], b[keys[j]]
            ea = ai == bj
            eb = ai2 == bj2
            same_a += int(ea)
            same_b += int(eb)
            same_both += int(ea and eb)
            total += 1
    if total == 0:
        return float("nan")
    return (
        same_both + (total - same_a - same_b + same_both)
    ) / total


# =====================================================================
# C.7 — Critical-exponent fitting
# =====================================================================


@dataclass
class CriticalExponentFit:
    parameter_axis: str
    p_critical: float
    beta: float            # eta ~ |p - p_c|^beta
    gamma: float           # var(eta) ~ |p - p_c|^-gamma
    rms_eta: float
    rms_var: float
    n_points: int


def fit_critical_exponents(
    cont: BifurcationContinuation,
    *,
    p_critical: float | None = None,
    side: str = "below",
) -> CriticalExponentFit:
    """Fit power laws around the candidate critical point.

    If ``p_critical`` is omitted, take the parameter value of the
    largest variance peak from ``cont.detected`` (if any) or the
    midpoint of the swept axis as a fallback.  ``side`` selects which
    branch to fit:

      * ``"below"`` — points with ``p < p_critical``
      * ``"above"`` — points with ``p > p_critical``
      * ``"both"``  — symmetric two-sided fit (folds |p - p_c|).
    """
    points = list(cont.points)
    if not points:
        raise ValueError("fit_critical_exponents: empty continuation")

    if p_critical is None:
        crit = [d for d in cont.detected if d["type"] == "critical-slowing-down?"]
        if crit:
            p_critical = max(crit, key=lambda d: d["var_peak"])["near"]
        else:
            p_critical = float(np.mean([p.parameter_value for p in points]))

    if side == "below":
        sub = [p for p in points if p.parameter_value < p_critical]
    elif side == "above":
        sub = [p for p in points if p.parameter_value > p_critical]
    else:
        sub = points

    sub = [p for p in sub if abs(p.parameter_value - p_critical) > 1e-9]
    if len(sub) < 2:
        raise ValueError("fit_critical_exponents: not enough points")

    dp = np.array([abs(p.parameter_value - p_critical) for p in sub])
    eta = np.array([p.eta_eq for p in sub])
    var = np.array([max(p.eta_var, 1e-9) for p in sub])

    # log(|eta - inf|) = log(A) + beta*log(dp)
    eta_inf = float(np.median(eta))
    eta_dev = np.abs(eta - eta_inf) + 1e-9
    log_dp = np.log(dp)
    log_eta = np.log(eta_dev)
    log_var = np.log(var)

    # Log-log linear regression.
    A = np.vstack([log_dp, np.ones_like(log_dp)]).T
    beta_coef, *_ = np.linalg.lstsq(A, log_eta, rcond=None)
    gamma_coef, *_ = np.linalg.lstsq(A, log_var, rcond=None)
    beta = float(beta_coef[0])
    gamma = float(-gamma_coef[0])

    pred_eta = beta_coef[0] * log_dp + beta_coef[1]
    pred_var = gamma_coef[0] * log_dp + gamma_coef[1]
    rms_eta = float(np.sqrt(np.mean((log_eta - pred_eta) ** 2)))
    rms_var = float(np.sqrt(np.mean((log_var - pred_var) ** 2)))

    return CriticalExponentFit(
        parameter_axis=cont.axis,
        p_critical=p_critical,
        beta=beta,
        gamma=gamma,
        rms_eta=rms_eta,
        rms_var=rms_var,
        n_points=len(sub),
    )


# =====================================================================
# Plotting (matplotlib, optional dependency)
# =====================================================================


def _try_import_matplotlib():
    """Return ``(plt, mpl)`` or ``(None, None)`` if matplotlib is missing.

    Uses the Agg backend for headless rendering; the analysis still runs
    when matplotlib is absent.
    """
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        return plt, matplotlib
    except Exception:
        return None, None


def plot_mean_field(
    runs: Iterable[TaskRun],
    fit: MeanFieldFit,
    *,
    output: Path,
    served_col: str | None = None,
    sample_runs: int = 6,
) -> bool:
    """Two-panel plot: phase portrait $(\\bar\\eta, \\dot{\\bar\\eta})$ +
    sample $\\bar\\eta(t)$ trajectories.

    Returns True iff the plot was written.
    """
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return False
    runs = list(runs)
    pairs_eta: list[float] = []
    pairs_dot: list[float] = []
    sample_curves: list[tuple[np.ndarray, np.ndarray]] = []
    for r in runs:
        e = _eta_series(r, served_col=served_col)
        if e is None or len(e) < 4 or float(np.std(e)) < 1e-9:
            continue
        t = r.t
        span = float(t[-1] - t[0])
        period = max(0.05, span / 16.0) if span > 0 else 1.0
        decim = _decimate_to_period(t, e, period)
        if decim is None:
            continue
        td, eta = decim
        if len(eta) < 3:
            continue
        dt = np.diff(td)
        eta_dot = np.diff(eta) / np.where(dt > 0, dt, 1.0)
        pairs_eta.extend(eta[:-1].tolist())
        pairs_dot.extend(eta_dot.tolist())
        if len(sample_curves) < sample_runs:
            sample_curves.append((td, eta))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    if pairs_eta:
        eta_arr = np.asarray(pairs_eta)
        dot_arr = np.asarray(pairs_dot)
        ax1.scatter(eta_arr, dot_arr, s=8, alpha=0.4)
        # Fitted line: dot = -gamma*(eta - eta_inf).
        xs = np.linspace(np.min(eta_arr), np.max(eta_arr), 50)
        ys = -fit.gamma * (xs - fit.eta_infinity)
        ax1.plot(xs, ys, "r-", lw=2,
                 label=f"$\\dot{{\\bar\\eta}} = -{fit.gamma:.3f}\\,(\\bar\\eta - {fit.eta_infinity:.3f})$")
        ax1.axhline(0.0, color="k", lw=0.5)
        ax1.legend(loc="best", fontsize=9)
    ax1.set_xlabel(r"$\bar\eta$ (order parameter)")
    ax1.set_ylabel(r"$\dot{\bar\eta}$")
    ax1.set_title(f"C.2 phase portrait  (RMS={fit.rms_residual:.3f}, n={fit.n_runs})")
    ax1.grid(alpha=0.3)

    for td, eta in sample_curves:
        ax2.plot(td, eta, alpha=0.6, lw=1.0)
    ax2.axhline(fit.eta_infinity, color="r", ls="--", lw=1,
                label=fr"$\eta_\infty={fit.eta_infinity:.3f}$")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel(r"$\bar\eta(t)$")
    ax2.set_title(f"Sample trajectories ({len(sample_curves)} runs)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return True


def plot_bifurcation(
    cont: BifurcationContinuation,
    *,
    output: Path,
) -> bool:
    """Two-panel plot: $\\eta_{\\rm eq}(p)$ with error bars + variance vs $p$
    (log-y, the critical-slowing-down signature)."""
    plt, _ = _try_import_matplotlib()
    if plt is None or not cont.points:
        return False

    p = np.asarray([pt.parameter_value for pt in cont.points])
    eta = np.asarray([pt.eta_eq for pt in cont.points])
    var = np.asarray([pt.eta_var for pt in cont.points])
    n = np.asarray([pt.n_runs for pt in cont.points])
    sem = np.sqrt(var / np.where(n > 0, n, 1))  # standard error of the mean

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.errorbar(p, eta, yerr=sem, fmt="o-", lw=1.5, capsize=3)
    ax1.set_xlabel(cont.axis)
    ax1.set_ylabel(r"$\eta_{\rm eq}$")
    ax1.set_title(f"C.3 bifurcation continuation along {cont.axis}")
    ax1.grid(alpha=0.3)

    # Annotate detected bifurcations with vertical lines.
    color = {"saddle-node?": "tab:red",
             "critical-slowing-down?": "tab:purple"}
    for d in cont.detected:
        ax1.axvline(d["near"], color=color.get(d["type"], "gray"),
                    ls="--", alpha=0.6,
                    label=d["type"] if d["type"] not in
                          [t.get_label() for t in ax1.get_lines()] else None)
        ax2.axvline(d["near"], color=color.get(d["type"], "gray"),
                    ls="--", alpha=0.6)
    if cont.detected:
        ax1.legend(loc="best", fontsize=8)

    if (var > 0).any():
        ax2.semilogy(p, np.maximum(var, 1e-12), "o-", lw=1.5, color="tab:purple")
    else:
        ax2.plot(p, var, "o-", lw=1.5, color="tab:purple")
    ax2.set_xlabel(cont.axis)
    ax2.set_ylabel(r"$\sigma^2(\eta)$")
    ax2.set_title("Variance of $\\eta$  (critical-slowing-down signature)")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return True


def plot_critical_exponents(
    cont: BifurcationContinuation,
    fit: CriticalExponentFit,
    *,
    output: Path,
) -> bool:
    """Log-log plot of $|\\eta - \\eta_\\infty|$ and $\\sigma^2$ vs
    $|p - p_c|$ with the fitted power-laws overlaid."""
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return False
    sub = [pt for pt in cont.points
           if abs(pt.parameter_value - fit.p_critical) > 1e-9]
    if len(sub) < 2:
        return False

    dp = np.array([abs(pt.parameter_value - fit.p_critical) for pt in sub])
    eta = np.array([pt.eta_eq for pt in sub])
    var = np.array([max(pt.eta_var, 1e-12) for pt in sub])
    eta_inf = float(np.median(eta))
    eta_dev = np.abs(eta - eta_inf) + 1e-12

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog(dp, eta_dev, "o", ms=8)
    if dp.min() > 0:
        xs = np.geomspace(dp.min(), dp.max(), 50)
        ys = np.exp(np.mean(np.log(eta_dev) - fit.beta * np.log(dp))) * xs ** fit.beta
        ax1.loglog(xs, ys, "r-", lw=2,
                   label=fr"$|\eta-\eta_\infty| \sim |p-p_c|^{{{fit.beta:.3f}}}$  (RMS={fit.rms_eta:.3f})")
    ax1.set_xlabel(rf"$|{cont.axis} - p_c|$, $p_c={fit.p_critical:.3f}$")
    ax1.set_ylabel(r"$|\eta - \eta_\infty|$")
    ax1.set_title(r"C.7 critical exponent $\beta$")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    ax2.loglog(dp, var, "o", ms=8, color="tab:purple")
    if dp.min() > 0:
        xs = np.geomspace(dp.min(), dp.max(), 50)
        ys = np.exp(np.mean(np.log(var) + fit.gamma * np.log(dp))) * xs ** (-fit.gamma)
        ax2.loglog(xs, ys, "r-", lw=2,
                   label=fr"$\sigma^2 \sim |p-p_c|^{{-{fit.gamma:.3f}}}$  (RMS={fit.rms_var:.3f})")
    ax2.set_xlabel(rf"$|{cont.axis} - p_c|$")
    ax2.set_ylabel(r"$\sigma^2(\eta)$")
    ax2.set_title(r"Critical exponent $\gamma$")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return True


def plot_cluster_synchronisation(
    run: TaskRun,
    result: ClusterSynchronisationResult,
    *,
    output: Path,
) -> bool:
    """Two-panel plot: per-aid Pearson correlation heatmap (sorted by
    cluster id) + cluster-id colour bar with optional static-group
    overlay."""
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return False
    if result.n_devices == 0 or not result.cluster_assignment:
        return False

    traj_path = run.task_dir / "trajectories.csv"
    if not traj_path.exists():
        return False
    cols, ts = _read_csv_floats(traj_path)
    if not cols or len(cols) < 3 or ts.shape[0] < 4:
        return False
    aid_cols = cols[1:]
    X = ts[:, 1:]
    Xz = X - np.mean(X, axis=0, keepdims=True)
    norms = np.linalg.norm(Xz, axis=0)
    nz = norms > 1e-9
    keep = np.where(nz)[0]
    if keep.size < 2:
        return False
    Xn = Xz[:, keep] / norms[keep]
    C = Xn.T @ Xn
    keep_aids = [aid_cols[i] for i in keep]

    # Sort columns by cluster id for a block-diagonal heatmap.
    order = sorted(
        range(len(keep_aids)),
        key=lambda i: (result.cluster_assignment.get(keep_aids[i], -1), keep_aids[i]),
    )
    C_sorted = C[np.ix_(order, order)]
    sorted_aids = [keep_aids[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8.5),
        gridspec_kw={"height_ratios": [10, 1]},
    )
    im = ax1.imshow(C_sorted, vmin=-1, vmax=1, cmap="coolwarm",
                    interpolation="nearest")
    fig.colorbar(im, ax=ax1, label="Pearson r", shrink=0.8)
    ax1.set_title(
        f"C.5 trajectory correlation matrix\n"
        f"{result.n_clusters} dynamic clusters, "
        f"static-overlap = {result.dynamic_vs_static_score:.3f}"
    )
    if len(sorted_aids) <= 30:
        ax1.set_xticks(range(len(sorted_aids)))
        ax1.set_xticklabels(sorted_aids, rotation=90, fontsize=6)
        ax1.set_yticks(range(len(sorted_aids)))
        ax1.set_yticklabels(sorted_aids, fontsize=6)
    else:
        ax1.set_xticks([])
        ax1.set_yticks([])

    cluster_strip = np.array([
        [result.cluster_assignment.get(a, -1) for a in sorted_aids]
    ], dtype=float)
    ax2.imshow(cluster_strip, aspect="auto",
               cmap="tab20", interpolation="nearest")
    ax2.set_yticks([])
    if result.static_groups:
        # Draw the static groups as text labels under the cluster strip.
        ax2.set_xticks(range(len(sorted_aids)))
        ax2.set_xticklabels(
            [result.static_groups.get(a, "") for a in sorted_aids],
            rotation=90, fontsize=5,
        )
    else:
        ax2.set_xticks([])
    ax2.set_title("Cluster id (top) vs static group (bottom)", fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return True


# =====================================================================
# CLI
# =====================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="adaptive_network_analysis",
        description="Adaptive-network post-hoc analysis bundle (C.2/3/5/7).",
    )
    p.add_argument("--eval-root", type=Path, required=True,
                   help="Directory containing tasks/<id>/ folders.")
    p.add_argument("--sweep-axis", type=str, default=None,
                   help="Config key to sweep along (dotted path).  When "
                        "omitted, the first non-constant axis is auto-picked "
                        "(use --list-axes to see the candidates).")
    p.add_argument("--output", type=Path, default=Path("analysis_adaptive"),
                   help="Output directory for JSON report and PNG plots.")
    p.add_argument("--only", choices=["mean-field", "bifurcation",
                                      "cluster-sync", "critical"],
                   help="Run only one analysis.")
    p.add_argument("--cluster-threshold", type=float, default=0.6,
                   help="Pearson |r| threshold for cluster-sync linkage.")
    p.add_argument("--p-critical", type=float, default=None,
                   help="Override critical point for C.7 (else auto-detect).")
    p.add_argument("--bin-count", type=int, default=8,
                   help="Number of quantile bins for bifurcation continuation.")
    p.add_argument("--filter", type=str, default=None,
                   help="Stratify runs by key=value predicates "
                        "(comma-separated, dotted keys allowed).")
    p.add_argument("--list-axes", action="store_true",
                   help="Print every non-constant config key + its values "
                        "and exit (no analysis run).")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip PNG generation; write report.json only.")
    args = p.parse_args(argv)

    runs = load_eval_root(args.eval_root)
    if not runs:
        print(f"No tasks found under {args.eval_root}", file=sys.stderr)
        return 2

    if args.list_axes:
        axes = discover_axes(runs)
        if not axes:
            print("No varying axes found across runs (all configs identical).")
            return 0
        print(f"Varying config keys across {len(runs)} runs:")
        for k in sorted(axes):
            vals = sorted(axes[k])
            shown = ", ".join(vals[:6])
            more = f" (+{len(vals) - 6} more)" if len(vals) > 6 else ""
            print(f"  {k:<40s}  {len(vals)} values: {{{shown}}}{more}")
        return 0

    runs = filter_runs(runs, args.filter)
    if not runs:
        print(f"No runs match filter {args.filter!r}", file=sys.stderr)
        return 2
    if args.filter:
        print(f"Filter '{args.filter}' kept {len(runs)} runs.")

    # Auto-pick a sweep axis when unspecified.
    if args.sweep_axis is None:
        args.sweep_axis = _pick_sweep_axis(runs)
        if args.sweep_axis is None:
            print("Could not auto-pick a numeric sweep axis; pass --sweep-axis "
                  "or run --list-axes.", file=sys.stderr)
            return 2
        print(f"Auto-selected sweep axis: {args.sweep_axis}")

    args.output.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "n_runs": len(runs),
        "axis": args.sweep_axis,
        "filter": args.filter or "",
    }

    sel = args.only
    plt_avail = _try_import_matplotlib()[0] is not None
    do_plots = (not args.no_plots) and plt_avail
    if not args.no_plots and not plt_avail:
        print("[plots] matplotlib unavailable, skipping PNG generation.",
              file=sys.stderr)

    written_plots: list[str] = []
    mf: MeanFieldFit | None = None
    cont: BifurcationContinuation | None = None
    fit: CriticalExponentFit | None = None

    if sel in (None, "mean-field"):
        try:
            mf = fit_mean_field_reduction(runs)
            report["mean_field_fit"] = asdict(mf)
            print(f"[mean-field] gamma={mf.gamma:.4f} eta_inf={mf.eta_infinity:.4f} "
                  f"nu_topo={mf.nu_topo:.4f} rms={mf.rms_residual:.4f} "
                  f"n_runs={mf.n_runs}")
            if do_plots:
                ok = plot_mean_field(runs, mf,
                                     output=plots_dir / "mean_field.png")
                if ok:
                    written_plots.append("mean_field.png")
        except Exception as exc:
            report["mean_field_error"] = str(exc)
            print(f"[mean-field] failed: {exc}", file=sys.stderr)

    if sel in (None, "bifurcation"):
        try:
            cont = continue_bifurcation(
                runs, axis=args.sweep_axis, bin_count=args.bin_count
            )
            report["bifurcation"] = {
                "axis": cont.axis,
                "points": [asdict(pt) for pt in cont.points],
                "detected": cont.detected,
            }
            print(f"[bifurcation] n_points={len(cont.points)} "
                  f"detected={len(cont.detected)} on axis={args.sweep_axis}")
            for d in cont.detected:
                print(f"  - {d}")
            if do_plots and cont.points:
                ok = plot_bifurcation(cont, output=plots_dir / "bifurcation.png")
                if ok:
                    written_plots.append("bifurcation.png")
        except Exception as exc:
            report["bifurcation_error"] = str(exc)
            print(f"[bifurcation] failed: {exc}", file=sys.stderr)

    if sel in (None, "critical"):
        if cont is None:
            try:
                cont = continue_bifurcation(
                    runs, axis=args.sweep_axis, bin_count=args.bin_count
                )
            except Exception:
                cont = None
        if cont is not None:
            try:
                fit = fit_critical_exponents(cont, p_critical=args.p_critical)
                report["critical_exponents"] = asdict(fit)
                print(f"[critical] p_c={fit.p_critical:.4f} "
                      f"beta={fit.beta:.4f} gamma={fit.gamma:.4f} "
                      f"rms_eta={fit.rms_eta:.4f} rms_var={fit.rms_var:.4f}")
                if do_plots:
                    ok = plot_critical_exponents(
                        cont, fit, output=plots_dir / "critical_exponents.png"
                    )
                    if ok:
                        written_plots.append("critical_exponents.png")
            except Exception as exc:
                report["critical_error"] = str(exc)
                print(f"[critical] failed: {exc}", file=sys.stderr)

    if sel in (None, "cluster-sync"):
        cluster_results: list[dict] = []
        plotted_examples = 0
        for r in runs:
            try:
                cs = cluster_synchronisation(
                    r, threshold=args.cluster_threshold,
                )
                cluster_results.append({
                    "task_id": r.task_id,
                    "n_devices": cs.n_devices,
                    "n_clusters": cs.n_clusters,
                    "dynamic_vs_static": cs.dynamic_vs_static_score,
                    "notes": cs.notes,
                })
                # Plot the first 3 tasks with non-empty trajectories.
                if do_plots and cs.n_devices > 0 and plotted_examples < 3:
                    out_png = plots_dir / f"cluster_sync_{r.task_id}.png"
                    if plot_cluster_synchronisation(r, cs, output=out_png):
                        written_plots.append(out_png.name)
                        plotted_examples += 1
            except Exception as exc:
                cluster_results.append({"task_id": r.task_id,
                                        "error": str(exc)})
        report["cluster_synchronisation"] = cluster_results
        n_with_data = sum(1 for c in cluster_results
                          if c.get("n_devices", 0) > 0)
        print(f"[cluster-sync] tasks_with_traj={n_with_data}/{len(runs)}")

    if written_plots:
        report["plots"] = written_plots
        print(f"[plots] wrote {len(written_plots)} PNG(s) to {plots_dir}")

    out = args.output / "report.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nWrote {out}")
    return 0


def _is_numeric_str(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# Per-run identifiers and seeds: look like sweep candidates but aren't.
_AXIS_NUISANCE = frozenset({"task_id", "seed", "base_seed"})


def _pick_sweep_axis(runs: list[TaskRun]) -> str | None:
    """Choose the most plausible numeric sweep axis automatically.

    Prefers numeric axes with 3-12 distinct values, then any with >=2;
    otherwise returns None.  Excludes the nuisance axes in
    ``_AXIS_NUISANCE``.
    """
    axes = discover_axes(runs)
    candidates: list[tuple[str, int]] = []
    for k, vals in axes.items():
        if k in _AXIS_NUISANCE:
            continue
        numeric = [v for v in vals if _is_numeric_str(v)]
        if len(numeric) >= 2:
            candidates.append((k, len(numeric)))
    if not candidates:
        return None
    # Prefer 3..12 distinct values; penalise too many or too few.
    def score(item):
        _, n = item
        if 3 <= n <= 12:
            return (0, n)
        if n == 2:
            return (1, n)
        return (2, n)
    candidates.sort(key=score)
    return candidates[0][0]


if __name__ == "__main__":
    sys.exit(main())
