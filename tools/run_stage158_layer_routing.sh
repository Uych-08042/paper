#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE158_ROOT:-$repo/output/stage158/independent_layer_continuation_v1}"
printf '[Stage-158] started %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -u -B tools/stage158_layer_routing.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage134 "${STAGE134_ROOT:-$repo/output/stage134/native_peak_continuation_v1}" \
  --stage137 "${STAGE137_ROOT:-$repo/output/stage137/shared_crop_layers_v2}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE158_GPU:-0}" --progress-interval "${STAGE158_PROGRESS_SECONDS:-10}" --output-dir "$run"
cat -- "$run/stage158_summary.csv"

bundle="$run/stage158_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv windows.csv choices.csv folds.csv; do
    printf '\n===== stage158_%s =====\n' "$file"
    cat -- "$run/stage158_$file"
  done
} > "$bundle"
printf 'Stage-158 result bundle: %s\n' "$bundle"
