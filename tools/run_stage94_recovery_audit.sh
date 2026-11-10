#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
run="${STAGE94_ROOT:-$repo/output/stage94/appearance_recovery_audit_v1}"

python -B tools/stage94_recovery_audit.py --stage76 "$stage76" --stage89 "$stage89" --stage93 "$stage93" --output_dir "$run"

bundle="$run/stage94_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json events.csv; do
    printf '\n===== stage94_%s =====\n' "$file"
    cat -- "$run/stage94_$file"
  done
} > "$bundle"
printf 'Stage-94 result bundle: %s\n' "$bundle"
