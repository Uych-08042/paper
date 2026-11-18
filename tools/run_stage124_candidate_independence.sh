#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE124_ROOT:-$repo/output/stage124/candidate_independence_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage124_candidate_independence.py \
      --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
      --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
      --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
      --stage112 "${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}" \
      --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
      --stage122 "${STAGE122_ROOT:-$repo/output/stage122/cached_alternate_reranking_v1}" \
      --output-dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage124_candidate_independence.py --check-results --output-dir "$run"
bundle="$run/stage124_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json model_scope.csv policy_summary.csv paired.csv video_effects.csv decisions.csv; do
    printf '\n===== stage124_%s =====\n' "$file"
    cat -- "$run/analysis/stage124_$file"
  done
} > "$bundle"
printf 'Stage-124 result bundle: %s\n' "$bundle"
