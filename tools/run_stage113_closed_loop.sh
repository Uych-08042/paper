#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage103="${STAGE103_ROOT:-$repo/output/stage103/four_candidate_verification_v1}"
stage112="${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}"
run="${STAGE113_ROOT:-$repo/output/stage113/frozen_generated_shared_closed_loop_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage113_frozen_closed_loop.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage103 "$stage103" --stage112 "$stage112" \
      --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE113_GPU:-0}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage113_frozen_closed_loop.py --check-results --output_dir "$run"
bundle="$run/stage113_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json model_scope.csv policy_summary.csv paired.csv video_summary.csv seed_video.csv scope.csv recovery.csv seed_effects.csv; do
    printf '\n===== stage113_%s =====\n' "$file"
    cat -- "$run/analysis/stage113_$file"
  done
} > "$bundle"
printf 'Stage-113 result bundle: %s\n' "$bundle"
