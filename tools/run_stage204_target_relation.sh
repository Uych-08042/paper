#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE204_ROOT:-$repo/output/stage204/target_search_relation_v1}"
python -u -B tools/stage204_target_relation.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage203-bundle "${STAGE203_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage203_results_20261125T162346Z_1448196.txt}" \
  --output-dir "$run" "$@"
if [[ ! -f "$run/stage204_decision.json" ]]; then exit 0; fi
bundle="$run/stage204_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for name in stage204_decision.json manifest.json stage204_model_scope.csv stage204_fit_summary.csv stage204_source_bridge.csv \
      stage204_pair_support.csv stage204_pair_schedule.csv stage204_policy_summary.csv stage204_paired.csv stage204_error_summary.csv \
      stage204_error_video.csv stage204_policy_video.csv stage204_training_history.csv stage204_frame_predictions.csv \
      stage204_seed_summary.csv stage204_descriptor_audit.csv; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-204] result bundle: %s\n' "$bundle"
