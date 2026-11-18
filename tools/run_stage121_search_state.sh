#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage121_search_state.py \
      --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
      --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
      --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
      --stage112 "${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}" \
      --stage113 "${STAGE113_ROOT:-$repo/output/stage113/frozen_generated_shared_closed_loop_v1}" \
      --output_dir "$run" --gpu "${STAGE121_GPU:-0}" \
      --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage121_search_state.py --check-results --output_dir "$run"
bundle="$run/stage121_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv seed_video.csv scope.csv recovery.csv; do
    printf '\n===== stage121_%s =====\n' "$file"
    cat -- "$run/analysis/stage121_$file"
  done
} > "$bundle"
printf 'Stage-121 result bundle: %s\n' "$bundle"
