#!/usr/bin/env bash
# Submit a plotting / report job for a finished campaign to Slurm.
#
# Mirrors scripts/submit_campaign.sh: activates the conda env on the
# head node, then invokes `python -m experiment.hpc.submit_plot`, which
# sbatches a single-task job that runs scripts/plot.sh on a compute
# node (partition / nodelist / account inherited from the campaign's
# config.json).
#
# Usage:
#   bash scripts/submit_plot.sh <campaign_dir> [--skip-aggregate] [--dry-run]
#
# Examples:
#   bash scripts/submit_plot.sh experiment/_runs/eval/eval_smoke_20260519-193242
#   bash scripts/submit_plot.sh experiment/_runs/foo --skip-aggregate
#   bash scripts/submit_plot.sh experiment/_runs/foo --dry-run
#
# Slurm sizing for the plot job comes from the campaign's config.json:
#   * "slurm"      — defaults inherited from the per-task array config
#   * "slurm_eval" — optional overrides applied on top of "slurm"; any
#                    SlurmConfig field is overridable (partition,
#                    nodelist, account, qos, cpus, extra_sbatch_args,
#                    ...). Set a key to null to drop it.
# Edit the JSON and resubmit to change sizing.
#
# Note: scripts/submit_campaign.sh already chains an aggregator after
# the array (when slurm.aggregate=true), so plotting an already-aggregated
# campaign should use --skip-aggregate to avoid redoing summary.csv.
module load hpc-env/13.1
module load Miniforge3/26.1.0-0
conda activate cmres_env

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash scripts/submit_plot.sh <campaign_dir> [--skip-aggregate] [--dry-run]" >&2
    exit 2
fi

CAMPAIGN_DIR="$1"
shift

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -d "$CAMPAIGN_DIR" ]; then
    echo "ERROR: campaign dir not found: $CAMPAIGN_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m experiment.hpc.submit_plot "$CAMPAIGN_DIR" "$@"
