#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE152_ROOT:-$repo/output/stage152/center_closed_loop_v1}"

python -B tools/stage152_center_closed_loop.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage143 "${STAGE143_ROOT:-$repo/output/stage143/fixed_pool_quality_v1}" \
  --stage149 "${STAGE149_ROOT:-$repo/output/stage149/score_ablation_v1}" \
  --stage151 "${STAGE151_ROOT:-$repo/output/stage151/write_geometry_cross_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE152_GPU:-0}" --output-dir "$run"
python -B tools/stage152_center_closed_loop.py --check-results --output-dir "$run"
cat -- "$run/stage152_policy_summary.csv"

bundle="$run/stage152_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv video_summary.csv failures.csv switches.csv native_shadow.csv; do
    printf '\n===== stage152_%s =====\n' "$file"
    cat -- "$run/stage152_$file"
  done
} > "$bundle"
printf 'Stage-152 result bundle: %s\n' "$bundle"
