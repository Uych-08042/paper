#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage112="${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}"
stage113="${STAGE113_ROOT:-$repo/output/stage113/frozen_generated_shared_closed_loop_v1}"
stage115="${STAGE115_ROOT:-$repo/output/stage115/matched_one_q0_continuation_v1}"
run="${STAGE116_ROOT:-$repo/output/stage116/cached_online_q0_evidence_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage116_online_evidence.py --stage76 "$stage76" --stage89 "$stage89" --stage93 "$stage93" \
      --stage112 "$stage112" --stage113 "$stage113" --stage115 "$stage115" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage116_online_evidence.py --check-results --output_dir "$run"
bundle="$run/stage116_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json population_scope.csv population_summary.csv anchor_evidence.csv evidence_summary.csv discrimination.csv policy_summary.csv paired.csv; do
    printf '\n===== stage116_%s =====\n' "$file"
    cat -- "$run/analysis/stage116_$file"
  done
} > "$bundle"
printf 'Stage-116 result bundle: %s\n' "$bundle"
