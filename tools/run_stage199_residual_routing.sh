#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE199_ROOT:-$repo/output/stage199/fit_centered_interaction_v1}"
python -u -B tools/stage199_residual_routing.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage198-bundle "${STAGE198_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage198_results_447f9394ced9.txt}" \
  --output-dir "$run" "$@"
if [[ ! -f "$run/stage199_decision.json" ]]; then exit 0; fi
bundle="$run/stage199_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for name in stage199_decision.json manifest.json stage199_model_scope.csv stage199_fit_summary.csv \
      stage199_source_bridge.csv stage199_policy_summary.csv stage199_paired.csv stage199_error_summary.csv \
      stage199_error_video.csv stage199_policy_video.csv stage199_training_history.csv stage199_frame_predictions.csv; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-199] result bundle: %s\n' "$bundle"
