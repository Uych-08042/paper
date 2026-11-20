#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE159_ROOT:-$repo/output/stage159/layer_selector_supervision_v1}"
printf '[Stage-159] preparing data and runtime (%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python -u -B tools/stage159_layer_selector.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE159_GPU:-0}" --progress-interval "${STAGE159_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage159_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv fit.csv support.csv choices.csv windows.csv; do
    printf '\n===== stage159_%s =====\n' "$file"
    cat -- "$run/stage159_$file"
  done
} > "$bundle"
printf '[Stage-159] result bundle: %s\n' "$bundle"
