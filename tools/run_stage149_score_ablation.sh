#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE149_ROOT:-$repo/output/stage149/score_ablation_v1}"
python -B tools/stage149_score_ablation.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage133 "${STAGE133_ROOT:-$repo/output/stage133/native_failure_audit_v1}" \
  --stage139 "${STAGE139_ROOT:-$repo/output/stage139/post_write_diagnostic_v1}" \
  --stage142 "${STAGE142_ROOT:-$repo/output/stage142/dense_candidate_recall_v1}" \
  --stage143 "${STAGE143_ROOT:-$repo/output/stage143/fixed_pool_quality_v1}" \
  --stage144 "${STAGE144_ROOT:-$repo/output/stage144/pairwise_candidate_ranking_v1}" \
  --stage148 "${STAGE148_ROOT:-$repo/output/stage148/overlap_appearance_v1}" \
  --output-dir "$run"
cat -- "$run/stage149_actions.csv"
bundle="$run/stage149_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json actions.csv funnel.csv policy.csv paired.csv stable.csv selected.csv focus.csv scope.csv; do
    printf '\n===== stage149_%s =====\n' "$file"
    cat -- "$run/stage149_$file"
  done
} > "$bundle"
printf 'Stage-149 result bundle: %s\n' "$bundle"
