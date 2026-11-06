#!/usr/bin/env bash
# All experiments (configs/final.json), on the CPU, each step only after the previous one has succeeded:
#   1. all methods (10 methods x 3 datasets x 3 seeds) + checks
#   2. parameter sensitivity (ISIC, seed 42)
#   3. summary (results/final/SUMMARY.md)
# usage (in the project's Python environment):  bash scripts/run_all.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results/final"
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"
export CUDA_VISIBLE_DEVICES="" CLUSTPY_DEVICE=cpu MPLBACKEND=Agg
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

say "STEP 1 started: all methods (90 runs)"
python scripts/run_experiments.py main > "$OUT/step1_main.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then say "STEP 1 FAILED (exit $rc): see $OUT/main_progress.log and $OUT/check_main.txt"; exit 1; fi
say "STEP 1 finished: all 90 runs done and all checks passed"

say "STEP 2 started: parameter sensitivity (seed 42)"
python scripts/run_experiments.py sensitivity > "$OUT/step2_sensitivity.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then say "STEP 2 FAILED (exit $rc): see $OUT/step2_sensitivity.log"; exit 2; fi
say "STEP 2 finished: lambda / sigma stage, best parameters, d stage"

python scripts/run_experiments.py summary > "$OUT/step3_summary.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then say "STEP 3 FAILED (exit $rc)"; exit 3; fi
say "ALL STEPS FINISHED: summary in $OUT/SUMMARY.md"
