#!/usr/bin/env bash
# Run a campaign locally end-to-end: plan -> run -> aggregate -> plot.
#
# Use this for development / smoke-testing on a workstation.  The on-disk
# layout (campaign dir, tasks/, summary.csv, plots/) matches what Slurm
# would produce, so the same plot/report tooling works in both modes.
#
# Usage:
#   bash scripts/run_local.sh <config.json> [--workers N]
#
# Examples:
#   bash scripts/run_local.sh experiment/configs/eval_smoke.json
#   bash scripts/run_local.sh experiment/configs/route_a_eval.json --workers 4
#
# Re-run a previously planned campaign without re-planning:
#   CAMPAIGN_DIR=experiment/_runs/eval/eval_smoke_20260519-193242 \
#       bash scripts/run_local.sh -
#
# Skip the plotting step (e.g. when iterating on results):
#   SKIP_PLOT=1 bash scripts/run_local.sh experiment/configs/eval_smoke.json

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash scripts/run_local.sh <config.json> [--workers N]" >&2
    exit 2
fi

CONFIG="$1"
shift

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$REPO_ROOT"

# Reuse an existing campaign dir if CAMPAIGN_DIR is exported, otherwise
# build a fresh one from the supplied config.  ``plan`` prints the path
# of the campaign it just created on stdout; we capture it.
if [ -n "${CAMPAIGN_DIR:-}" ]; then
    echo ">>> Reusing campaign dir: $CAMPAIGN_DIR"
else
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: config file not found: $CONFIG" >&2
        exit 1
    fi
    echo ">>> Planning campaign from $CONFIG"
    CAMPAIGN_DIR="$("$PYTHON_BIN" -m experiment.hpc.plan "$CONFIG" | tail -n 1)"
    if [ ! -d "$CAMPAIGN_DIR" ]; then
        echo "ERROR: plan step did not produce a campaign dir (got: $CAMPAIGN_DIR)" >&2
        exit 1
    fi
    echo "    Created: $CAMPAIGN_DIR"
fi

echo ">>> Running tasks locally"
"$PYTHON_BIN" -m experiment.hpc.run_local --campaign-dir "$CAMPAIGN_DIR" "$@"

if [ "${SKIP_PLOT:-0}" != "1" ]; then
    echo ">>> Aggregating + plotting"
    bash "$REPO_ROOT/scripts/plot.sh" "$CAMPAIGN_DIR"
fi

echo
echo "Campaign dir: $CAMPAIGN_DIR"
