#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run="${STAGE133_ROOT:-$repo/output/stage133/native_failure_audit_v1}"
python -B tools/stage133_native_failure_audit.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage131 "${STAGE131_ROOT:-$repo/output/stage131/q3_zoom_closed_loop_v1}" \
  --output-dir "$run"
cat -- "$run/stage133_summary.csv"
bundle="$run/stage133_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv video.csv events.csv anchors.csv; do
    printf '\n===== stage133_%s =====\n' "$file"
    cat -- "$run/stage133_$file"
  done
} > "$bundle"
printf 'Stage-133 result bundle: %s\n' "$bundle"
