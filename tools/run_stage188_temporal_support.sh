#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}"

python -u -B tools/stage188_temporal_support.py \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage187-root "${STAGE187_ROOT:-$repo/output/stage187/crop_consensus_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE188_GPU:-0}" --progress-interval "${STAGE188_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage188_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json cases.csv support.csv candidates.csv choices.csv summary.csv paired.csv; do
    printf '\n===== stage188_%s =====\n' "$file"
    cat -- "$run/stage188_$file"
  done
} > "$bundle"
printf '[Stage-188] result bundle: %s\n' "$bundle"
