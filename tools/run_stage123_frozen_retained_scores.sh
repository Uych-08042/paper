#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE123_ROOT:-$repo/output/stage123/frozen_retained_scores_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage123_frozen_retained_scores.py \
      --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
      --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
      --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
      --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
      --stage122 "${STAGE122_ROOT:-$repo/output/stage122/cached_alternate_reranking_v1}" \
      --output-dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage123_frozen_retained_scores.py --check-results --output-dir "$run"
bundle="$run/stage123_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv video_effects.csv decisions.csv; do
    printf '\n===== stage123_%s =====\n' "$file"
    cat -- "$run/analysis/stage123_$file"
  done
} > "$bundle"
printf 'Stage-123 result bundle: %s\n' "$bundle"
