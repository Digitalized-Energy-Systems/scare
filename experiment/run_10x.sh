#!/usr/bin/env bash
# Run the restoration scenario across grids; summarise per-run + per-grid.
set -u
cd "$(dirname "$0")/.."

if [ $# -gt 0 ]; then
    GRIDS=("$@")
else
    GRIDS=(urban urban_ties industrial regional)
fi
RUNS_PER_GRID="${RUNS_PER_GRID:-10}"

mkdir -p experiment/_runs
SUMMARY="experiment/_runs/summary.csv"
echo "grid,run,solver_failures,first_zero_factor" >"$SUMMARY"

for grid in "${GRIDS[@]}"; do
    fail_total=0
    zero_runs=0
    for i in $(seq 1 "$RUNS_PER_GRID"); do
        out="experiment/_runs/${grid}_run_${i}.log"
        timeout 120 python experiment/restoration.py --grid "$grid" --no-html >"$out" 2>&1 || true
        fails=$(grep -c "Pyomo solve failed" "$out" || true)
        first_zero=$(grep "factor=0.0000" "$out" | head -1 || true)
        echo "run=$i  grid=$grid  solver_failures=$fails  first_zero_factor=${first_zero:-none}"
        echo "${grid},${i},${fails},\"${first_zero:-}\"" >>"$SUMMARY"
        fail_total=$((fail_total + fails))
        [ -n "$first_zero" ] && zero_runs=$((zero_runs + 1))
    done
    echo "---  grid=$grid  total_solver_failures=${fail_total}  runs_with_zero_factor=${zero_runs}/${RUNS_PER_GRID}"
done

echo
echo "Summary written to $SUMMARY"
