#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE147_ROOT:-$repo/output/stage147/trigger_coverage_v1}"
python -B tools/stage147_trigger_coverage.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage133 "${STAGE133_ROOT:-$repo/output/stage133/native_failure_audit_v1}" \
  --stage139 "${STAGE139_ROOT:-$repo/output/stage139/post_write_diagnostic_v1}" \
  --stage142 "${STAGE142_ROOT:-$repo/output/stage142/dense_candidate_recall_v1}" \
  --stage143 "${STAGE143_ROOT:-$repo/output/stage143/fixed_pool_quality_v1}" \
  --stage144 "${STAGE144_ROOT:-$repo/output/stage144/pairwise_candidate_ranking_v1}" \
  --stage146 "${STAGE146_ROOT:-$repo/output/stage146/conflict_appearance_v1}" \
  --output-dir "$run"
cat -- "$run/stage147_exclusions.csv"
bundle="$run/stage147_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json coverage.csv exclusions.csv focus.csv scope.csv; do
    printf '\n===== stage147_%s =====\n' "$file"
    cat -- "$run/stage147_$file"
  done
} > "$bundle"
printf 'Stage-147 result bundle: %s\n' "$bundle"
