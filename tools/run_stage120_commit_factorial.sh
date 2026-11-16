#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE120_ROOT:-$repo/output/stage120/support_appearance_commit_factorial_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage120_commit_factorial.py \
      --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
      --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
      --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
      --stage112 "${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}" \
      --stage113 "${STAGE113_ROOT:-$repo/output/stage113/frozen_generated_shared_closed_loop_v1}" \
      --stage115 "${STAGE115_ROOT:-$repo/output/stage115/matched_one_q0_continuation_v1}" \
      --stage117 "${STAGE117_ROOT:-$repo/output/stage117/gt_free_sampled_q0_continuation_v1}" \
      --stage118 "${STAGE118_ROOT:-$repo/output/stage118/provider_isolated_commit_learning_v1}" \
      --stage119 "${STAGE119_ROOT:-$repo/output/stage119/fixed_commit_support_diagnostic_v1}" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage120_commit_factorial.py --check-results --output_dir "$run"
bundle="$run/stage120_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json appearance_scope.csv appearance_evidence.csv model_scope.csv fit_metrics.csv policy_summary.csv paired.csv support.csv sample_predictions.csv; do
    printf '\n===== stage120_%s =====\n' "$file"
    cat -- "$run/analysis/stage120_$file"
  done
} > "$bundle"
printf 'Stage-120 result bundle: %s\n' "$bundle"
