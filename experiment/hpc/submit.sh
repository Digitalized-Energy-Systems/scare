#!/usr/bin/env bash
# Thin wrapper around `python -m experiment.hpc.submit`.
#
# All Slurm settings (partition, nodelist, time, mem, max_concurrent, ...)
# live in <campaign_dir>/config.json — generated from the JSON config you
# passed to `python -m experiment.hpc.plan`. Edit the JSON, rebuild the
# campaign, and resubmit.
#
# Usage:
#   bash experiment/hpc/submit.sh <campaign_dir> [--only failed|timeout|missing|incomplete|all] [--dry-run]
#
# Examples:
#   bash experiment/hpc/submit.sh experiment/_runs/restoration_baseline_20260505-120000
#   bash experiment/hpc/submit.sh experiment/_runs/foo --only failed
#   bash experiment/hpc/submit.sh experiment/_runs/foo --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
    echo "Override with PYTHON_BIN=/path/to/python bash $0 ..." >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m experiment.hpc.submit "$@"
