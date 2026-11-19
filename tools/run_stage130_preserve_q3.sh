#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run="${STAGE130_ROOT:-$repo/output/stage130/preserve_q3_single_zoom_v1}"
python -B tools/stage130_preserve_q3.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage112 "${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage127 "${STAGE127_ROOT:-$repo/output/stage127/trigger_crop_coverage_v1}" \
  --stage129 "${STAGE129_ROOT:-$repo/output/stage129/coarse_to_fine_v1}" \
  --output-dir "$run"
cat -- "$run/stage130_summary.csv"
bundle="$run/stage130_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv points.csv; do
    printf '\n===== stage130_%s =====\n' "$file"
    cat -- "$run/stage130_$file"
  done
} > "$bundle"
printf 'Stage-130 result bundle: %s\n' "$bundle"
