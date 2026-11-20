#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE150_ROOT:-$repo/output/stage150/gray_write_continuation_v1}"

python -B tools/stage150_gray_write_continuation.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage133 "${STAGE133_ROOT:-$repo/output/stage133/native_failure_audit_v1}" \
  --stage143 "${STAGE143_ROOT:-$repo/output/stage143/fixed_pool_quality_v1}" \
  --stage149 "${STAGE149_ROOT:-$repo/output/stage149/score_ablation_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE150_GPU:-0}" --output-dir "$run"
python -B tools/stage150_gray_write_continuation.py --check-results --output-dir "$run"

cat -- "$run/stage150_summary.csv"
bundle="$run/stage150_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv windows.csv timeline.csv; do
    printf '\n===== stage150_%s =====\n' "$file"
    cat -- "$run/stage150_$file"
  done
} > "$bundle"
printf 'Stage-150 result bundle: %s\n' "$bundle"
