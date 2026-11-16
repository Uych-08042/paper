#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage112="${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}"
stage113="${STAGE113_ROOT:-$repo/output/stage113/frozen_generated_shared_closed_loop_v1}"
stage115="${STAGE115_ROOT:-$repo/output/stage115/matched_one_q0_continuation_v1}"
stage116="${STAGE116_ROOT:-$repo/output/stage116/cached_online_q0_evidence_v1}"
stage117="${STAGE117_ROOT:-$repo/output/stage117/gt_free_sampled_q0_continuation_v1}"
run="${STAGE118_ROOT:-$repo/output/stage118/provider_isolated_commit_learning_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage118_nested_commit.py --stage112 "$stage112" --stage113 "$stage113" --stage115 "$stage115" \
      --stage116 "$stage116" --stage117 "$stage117" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage118_nested_commit.py --check-results --output_dir "$run"
bundle="$run/stage118_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json fold_scope.csv model_scope.csv fit_metrics.csv policy_summary.csv paired.csv support.csv sample_predictions.csv; do
    printf '\n===== stage118_%s =====\n' "$file"
    cat -- "$run/analysis/stage118_$file"
  done
} > "$bundle"
printf 'Stage-118 result bundle: %s\n' "$bundle"
