#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage103="${STAGE103_ROOT:-$repo/output/stage103/four_candidate_verification_v1}"
stage110="${STAGE110_ROOT:-$repo/output/stage110/calibrated_search_proposals_v1}"
stage111="${STAGE111_ROOT:-$repo/output/stage111/cached_ppo_slot_audit_v1}"
run="${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage112_shared_scoring.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage103 "$stage103" --stage110 "$stage110" --stage111 "$stage111" \
      --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE112_GPU:-0}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage112_shared_scoring.py --check-results --output_dir "$run"
bundle="$run/stage112_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json fold_support.csv sampling_scope.csv training_scope.csv policy_summary.csv paired.csv collection_scope.csv state_decisions.csv; do
    printf '\n===== stage112_%s =====\n' "$file"
    cat -- "$run/analysis/stage112_$file"
  done
} > "$bundle"
printf 'Stage-112 result bundle: %s\n' "$bundle"
