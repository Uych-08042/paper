#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE185_ROOT:-$repo/output/stage185/writeback_components_v1}"

python -u -B tools/stage185_writeback_components.py \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage184-root "${STAGE184_ROOT:-$repo/output/stage184/center_mapping_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE185_GPU:-0}" --progress-interval "${STAGE185_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage185_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json cases.csv windows.csv timing.csv response.csv summary.csv effects.csv focus.csv; do
    printf '\n===== stage185_%s =====\n' "$file"
    cat -- "$run/stage185_$file"
  done
} > "$bundle"
printf '[Stage-185] result bundle: %s\n' "$bundle"
