#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage103="${STAGE103_ROOT:-$repo/output/stage103/four_candidate_verification_v1}"
stage109="${STAGE109_ROOT:-$repo/output/stage109/gt_free_sampled_commit_branches_v1}"
run="${STAGE110_ROOT:-$repo/output/stage110/calibrated_search_proposals_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage110_search_proposals.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage103 "$stage103" --stage109 "$stage109" --output_dir "$run" \
      --checkpoint "$checkpoint" --gpu "${STAGE110_GPU:-0}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage110_search_proposals.py --check-results --output_dir "$run"
bundle="$run/stage110_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json sampling_scope.csv fit_support.csv prediction_quality.csv prediction_summary.csv \
    state_supply.csv policy_summary.csv paired.csv scope.csv; do
    printf '\n===== stage110_%s =====\n' "$file"
    cat -- "$run/analysis/stage110_$file"
  done
} > "$bundle"
printf 'Stage-110 result bundle: %s\n' "$bundle"
