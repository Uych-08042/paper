#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE164_ROOT:-$repo/output/stage164/l6_local_match_gain_v1}"

python -u -B tools/stage164_local_match_gain.py \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage159-root "${STAGE159_ROOT:-$repo/output/stage159/layer_selector_supervision_v1}" \
  --stage163-root "${STAGE163_ROOT:-$repo/output/stage163/history_gain_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE164_GPU:-0}" --progress-interval "${STAGE164_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage164_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv folds.csv focus.csv choices.csv feature_audit.csv; do
    printf '\n===== stage164_%s =====\n' "$file"
    cat -- "$run/stage164_$file"
  done
} > "$bundle"
printf '[Stage-164] result bundle: %s\n' "$bundle"
