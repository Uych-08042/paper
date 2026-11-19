#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE136_ROOT:-$repo/output/stage136/same_crop_peak_closed_loop_v1}"
python -B tools/stage136_peak_closed_loop.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage135 "${STAGE135_ROOT:-$repo/output/stage135/cached_peak_gate_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE136_GPU:-0}" --output-dir "$run"
python -B tools/stage136_peak_closed_loop.py --check-results --output-dir "$run"
cat -- "$run/stage136_policy_summary.csv"
bundle="$run/stage136_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv video_summary.csv failures.csv; do
    printf '\n===== stage136_%s =====\n' "$file"
    cat -- "$run/stage136_$file"
  done
} > "$bundle"
printf 'Stage-136 result bundle: %s\n' "$bundle"
