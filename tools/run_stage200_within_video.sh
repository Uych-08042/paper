#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE200_ROOT:-$repo/output/stage200/matched_within_video_v1}"
python -u -B tools/stage200_within_video.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage199-bundle "${STAGE199_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage199_results_20261125T113420Z_1204543.txt}" \
  --output-dir "$run" "$@"
if [[ ! -f "$run/stage200_decision.json" ]]; then exit 0; fi
bundle="$run/stage200_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for name in stage200_decision.json manifest.json stage200_model_scope.csv stage200_fit_summary.csv stage200_source_bridge.csv \
      stage200_pair_support.csv stage200_pair_schedule.csv stage200_policy_summary.csv stage200_paired.csv stage200_error_summary.csv \
      stage200_error_video.csv stage200_policy_video.csv stage200_training_history.csv stage200_frame_predictions.csv; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-200] result bundle: %s\n' "$bundle"
