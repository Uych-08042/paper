#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE205_ROOT:-$repo/output/stage205/fit_source_crossfit_shrinkage_v1}"
python -u -B tools/stage205_crossfit_shrinkage.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage204-bundle "${STAGE204_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage204_results_20261126T011205Z_1583527.txt}" \
  --output-dir "$run" "$@"
if [[ ! -f "$run/stage205_decision.json" ]]; then exit 0; fi
bundle="$run/stage205_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for name in stage205_decision.json manifest.json stage205_calibration.csv stage205_policy_summary.csv stage205_paired.csv \
      stage205_seed_summary.csv stage205_error_summary.csv stage205_source_bridge.csv stage205_inner_scope.csv \
      stage205_inner_statistics.csv stage205_inner_schedule.csv stage205_inner_support.csv stage205_descriptor_audit.csv \
      stage205_training_history.csv stage205_inner_predictions.csv stage205_frame_predictions.csv \
      stage205_policy_video.csv stage205_error_video.csv; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-205] result bundle: %s\n' "$bundle"
