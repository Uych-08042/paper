#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE139_ROOT:-$repo/output/stage139/post_write_diagnostic_v1}"
python -B tools/stage139_post_write_diagnostic.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage138 "${STAGE138_ROOT:-$repo/output/stage138/cross_layer_continuation_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE139_GPU:-0}" --output-dir "$run"
python -B tools/stage139_post_write_diagnostic.py --check-results --output-dir "$run"
cat -- "$run/stage139_mechanism_summary.csv"
bundle="$run/stage139_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json mechanism_summary.csv replay.csv coverage.csv scope.csv; do
    printf '\n===== stage139_%s =====\n' "$file"
    cat -- "$run/stage139_$file"
  done
} > "$bundle"
printf 'Stage-139 result bundle: %s\n' "$bundle"
