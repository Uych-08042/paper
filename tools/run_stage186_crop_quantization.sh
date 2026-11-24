#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE186_ROOT:-$repo/output/stage186/crop_quantization_v1}"

python -u -B tools/stage186_crop_quantization.py \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage185-root "${STAGE185_ROOT:-$repo/output/stage185/writeback_components_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE186_GPU:-0}" --progress-interval "${STAGE186_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage186_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json cases.csv probes.csv windows.csv pairs.csv summary.csv focus.csv; do
    printf '\n===== stage186_%s =====\n' "$file"
    cat -- "$run/stage186_$file"
  done
} > "$bundle"
printf '[Stage-186] result bundle: %s\n' "$bundle"
