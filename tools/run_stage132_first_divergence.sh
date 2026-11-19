#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run="${STAGE132_ROOT:-$repo/output/stage132/first_search_divergence_v1}"
python -B tools/stage132_first_divergence.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage131 "${STAGE131_ROOT:-$repo/output/stage131/q3_zoom_closed_loop_v1}" \
  --output-dir "$run"
cat -- "$run/stage132_summary.csv"
bundle="$run/stage132_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv trajectories.csv first_events.csv recovery.csv; do
    printf '\n===== stage132_%s =====\n' "$file"
    cat -- "$run/stage132_$file"
  done
} > "$bundle"
printf 'Stage-132 result bundle: %s\n' "$bundle"
