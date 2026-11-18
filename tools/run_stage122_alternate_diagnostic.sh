#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE122_ROOT:-$repo/output/stage122/cached_alternate_reranking_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage122_alternate_diagnostic.py \
      --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
      --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
      --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
      --stage112 "${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}" \
      --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
      --output-dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage122_alternate_diagnostic.py --check-results --output-dir "$run"
bundle="$run/stage122_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv trajectories.csv activations.csv retained_scores.csv relation_swaps.csv; do
    printf '\n===== stage122_%s =====\n' "$file"
    cat -- "$run/analysis/stage122_$file"
  done
} > "$bundle"
printf 'Stage-122 result bundle: %s\n' "$bundle"
