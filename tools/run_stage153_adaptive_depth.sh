#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE153_ROOT:-$repo/output/stage153/serial_one_block_v1}"

python -B tools/stage153_adaptive_depth.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE153_GPU:-0}" --output-dir "$run"
python -B tools/stage153_adaptive_depth.py --check-results --output-dir "$run"
cat -- "$run/stage153_policy_summary.csv"

bundle="$run/stage153_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv video_summary.csv failures.csv routing.csv; do
    printf '\n===== stage153_%s =====\n' "$file"
    cat -- "$run/stage153_$file"
  done
} > "$bundle"
printf 'Stage-153 result bundle: %s\n' "$bundle"
