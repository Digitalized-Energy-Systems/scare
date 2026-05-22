#!/usr/bin/env bash
# Aggregate + plot a campaign run.
#
# Given the path to a campaign directory (the one that holds tasks/, the
# config.json, the manifest.jsonl), this:
#   1. materialises summary.csv via experiment.hpc.aggregate
#   2. renders plots/<experiment>/*.png and REPORT.md via experiment.eval.report
#
# Both steps are idempotent — re-running overwrites artefacts in place.
#
# Usage:
#   bash scripts/plot.sh <campaign_dir>
#   bash scripts/plot.sh experiment/_runs/eval/eval_smoke_20260519-193242
#
# Skip aggregation (e.g. when summary.csv is already current) with:
#   SKIP_AGGREGATE=1 bash scripts/plot.sh <campaign_dir>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash scripts/plot.sh <campaign_dir>" >&2
    exit 2
fi

CAMPAIGN_DIR="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -d "$CAMPAIGN_DIR" ]; then
    echo "ERROR: campaign dir not found: $CAMPAIGN_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"

if [ "${SKIP_AGGREGATE:-0}" != "1" ]; then
    echo ">>> Aggregating per-task artefacts → summary.csv"
    "$PYTHON_BIN" -m experiment.hpc.aggregate --campaign-dir "$CAMPAIGN_DIR"
fi

echo ">>> Rendering plots + REPORT.md"
"$PYTHON_BIN" -m experiment.eval.report --campaign-dir "$CAMPAIGN_DIR"

echo
echo "Done.  Artefacts:"
echo "  $CAMPAIGN_DIR/summary.csv"
echo "  $CAMPAIGN_DIR/REPORT.md"
echo "  $CAMPAIGN_DIR/plots/"
