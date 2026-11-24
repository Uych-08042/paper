#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE183_ROOT:-$repo/output/stage183/search_feedback_v1}"

python -u -B tools/stage183_search_feedback.py \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage176-root "${STAGE176_ROOT:-$repo/output/stage176/layer_event_sweep_v1}" \
  --stage182-root "${STAGE182_ROOT:-$repo/output/stage182/layer_opportunity_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE183_GPU:-0}" --progress-interval "${STAGE183_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage183_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json cases.csv scan.csv scan_layers.csv transitions.csv points.csv anchors.csv effects.csv tails.csv opportunities.csv summary.csv; do
    printf '\n===== stage183_%s =====\n' "$file"
    cat -- "$run/stage183_$file"
  done
} > "$bundle"
printf '[Stage-183] result bundle: %s\n' "$bundle"
