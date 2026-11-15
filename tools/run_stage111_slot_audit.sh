#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage103="${STAGE103_ROOT:-$repo/output/stage103/four_candidate_verification_v1}"
stage110="${STAGE110_ROOT:-$repo/output/stage110/calibrated_search_proposals_v1}"
run="${STAGE111_ROOT:-$repo/output/stage111/cached_ppo_slot_audit_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage111_slot_audit.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage103 "$stage103" --stage110 "$stage110" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage111_slot_audit.py --check-results --output_dir "$run"
bundle="$run/stage111_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv sensitivity_summary.csv permutation_summary.csv state_summary.csv scope.csv; do
    printf '\n===== stage111_%s =====\n' "$file"
    cat -- "$run/analysis/stage111_$file"
  done
} > "$bundle"
printf 'Stage-111 result bundle: %s\n' "$bundle"
