#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE157_ROOT:-$repo/output/stage157/matched_writeback_v1}"

python -B tools/stage157_depth_writeback.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage155-root "${STAGE155_ROOT:-$repo/output/stage155/isolated_l10_adaptation_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE157_GPU:-0}" --output-dir "$run"
cat -- "$run/stage157_summary.csv"

bundle="$run/stage157_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv cases.csv; do
    printf '\n===== stage157_%s =====\n' "$file"
    cat -- "$run/stage157_$file"
  done
} > "$bundle"
printf 'Stage-157 result bundle: %s\n' "$bundle"
