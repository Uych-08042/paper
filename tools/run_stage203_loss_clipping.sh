#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE203_ROOT:-$repo/output/stage203/loss_clipping_factorial_v1}"
python -u -B tools/stage203_loss_clipping.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage202-bundle "${STAGE202_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage202_results_20261125T154651Z_1417349.txt}" \
  --output-dir "$run" "$@"
if [[ ! -f "$run/stage203_decision.json" ]]; then exit 0; fi
bundle="$run/stage203_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for name in stage203_decision.json manifest.json stage203_model_scope.csv stage203_fit_summary.csv stage203_source_bridge.csv \
      stage203_pair_support.csv stage203_pair_schedule.csv stage203_policy_summary.csv stage203_paired.csv stage203_error_summary.csv \
      stage203_error_video.csv stage203_policy_video.csv stage203_training_history.csv stage203_frame_predictions.csv \
      stage203_seed_summary.csv stage203_control_replay.csv; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-203] result bundle: %s\n' "$bundle"
