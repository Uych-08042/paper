#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run="${STAGE127_ROOT:-$repo/output/stage127/trigger_crop_coverage_v1}"
python -B tools/stage127_crop_coverage.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage125 "${STAGE125_ROOT:-$repo/output/stage125/alternate_trigger_continuation_v1}" \
  --output-dir "$run"
cat -- "$run/stage127_summary.csv"
bundle="$run/stage127_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv points.csv crops.csv; do
    printf '\n===== stage127_%s =====\n' "$file"
    cat -- "$run/stage127_$file"
  done
} > "$bundle"
printf 'Stage-127 result bundle: %s\n' "$bundle"
