#!/usr/bin/env bash
# Plan + submit a campaign to the HPC.
#
# Wraps the existing two-step flow:
#   1. python -m experiment.hpc.plan <config.json>
#        -> creates experiment/_runs/<campaign_dir>/{config.json,manifest.jsonl,tasks/}
#   2. bash experiment/hpc/submit.sh <campaign_dir>
#        -> activates the conda env on the head node, submits the Slurm array
#
# Slurm settings (partition, nodelist, time, mem, max_concurrent ...) live
# in the JSON config -> edit the config + re-plan to change them.
#
# Usage:
#   bash scripts/submit_campaign.sh <config.json>
#   bash scripts/submit_campaign.sh experiment/configs/eval_smoke.json
#
# Re-submit only the failed/missing tasks of an already-planned campaign:
#   bash scripts/submit_campaign.sh - --only failed       CAMPAIGN_DIR=…
#   CAMPAIGN_DIR=experiment/_runs/eval/eval_smoke_20260519-193242 \
#       bash scripts/submit_campaign.sh - --only failed
#
# Dry-run (print sbatch command without submitting):
#   bash scripts/submit_campaign.sh experiment/configs/eval_smoke.json --dry-run
module load hpc-env/13.1
module load Miniforge3/26.1.0-0
conda activate cmres_env

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash scripts/submit_campaign.sh <config.json|-> [submit-flags]" >&2
    echo "  use '-' for the config path when CAMPAIGN_DIR is already set" >&2
    exit 2
fi

CONFIG="$1"
shift

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$REPO_ROOT"

if [ -n "${CAMPAIGN_DIR:-}" ]; then
    echo ">>> Reusing campaign dir: $CAMPAIGN_DIR"
elif [ "$CONFIG" = "-" ]; then
    echo "ERROR: pass an explicit config path, or export CAMPAIGN_DIR" >&2
    exit 1
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

echo ">>> Submitting to Slurm"
bash "$REPO_ROOT/experiment/hpc/submit.sh" "$CAMPAIGN_DIR" "$@"

echo
echo "Campaign dir: $CAMPAIGN_DIR"
echo
echo "When the array finishes, plot with:"
echo "  bash scripts/plot.sh $CAMPAIGN_DIR"
