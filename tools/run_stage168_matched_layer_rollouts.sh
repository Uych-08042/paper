#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE168_ROOT:-$repo/output/stage168/matched_layer_rollouts_v1}"

python -u -B tools/stage168_matched_layer_rollouts.py \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage159-root "${STAGE159_ROOT:-$repo/output/stage159/layer_selector_supervision_v1}" \
  --stage165-root "${STAGE165_ROOT:-$repo/output/stage165/layer_reproposal_v1}" \
  --stage166-root "${STAGE166_ROOT:-$repo/output/stage166/pairwise_layer_ranking_v1}" \
  --stage167-root "${STAGE167_ROOT:-$repo/output/stage167/online_layer_ranking_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE168_GPU:-0}" --progress-interval "${STAGE168_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage168_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json windows.csv branches.csv contrasts.csv recovery_windows.csv; do
    if [[ -f "$run/stage168_$file" ]]; then
      printf '\n===== stage168_%s =====\n' "$file"
      cat -- "$run/stage168_$file"
    fi
  done
} > "$bundle"
printf '[Stage-168] result bundle: %s\n' "$bundle"
