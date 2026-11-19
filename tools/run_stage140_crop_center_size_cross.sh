#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE140_ROOT:-$repo/output/stage140/crop_center_size_cross_v1}"
python -B tools/stage140_crop_center_size_cross.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage139 "${STAGE139_ROOT:-$repo/output/stage139/post_write_diagnostic_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE140_GPU:-0}" --output-dir "$run"
python -B tools/stage140_crop_center_size_cross.py --check-results --output-dir "$run"
cat -- "$run/stage140_summary.csv"
bundle="$run/stage140_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv frames.csv effects.csv scope.csv; do
    printf '\n===== stage140_%s =====\n' "$file"
    cat -- "$run/stage140_$file"
  done
} > "$bundle"
printf 'Stage-140 result bundle: %s\n' "$bundle"
