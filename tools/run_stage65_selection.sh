#!/usr/bin/env bash
set -euo pipefail
stage65_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage65_repo"
export STAGE64_ROOT="${STAGE64_ROOT:-$PWD/output/stage64/causal_candidate_reliability_v1}"
export STAGE65_ROOT="${STAGE65_ROOT:-$PWD/output/stage65/cached_same_state_selection_v1}"
test -f "$STAGE64_ROOT/analysis/stage64_decision.json"
python -B tools/stage65_selection_audit.py --input "$STAGE64_ROOT" --output_dir "$STAGE65_ROOT"
stage65_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage65_dir"
stage65_report="$stage65_dir/stage65_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage65_files=(stage65_diagnostic.json stage65_policy_summary.csv stage65_matched_summary.csv stage65_feature_shift.csv stage65_scope.csv stage65_focus.csv)
for stage65_file in "${stage65_files[@]}"; do test -f "$STAGE65_ROOT/analysis/$stage65_file"; done
{
  for stage65_file in "${stage65_files[@]}"; do
    printf '\n===== %s =====\n' "$stage65_file"
    cat -- "$STAGE65_ROOT/analysis/$stage65_file"
    printf '\n'
  done
} > "$stage65_report"
printf 'Stage-65 result bundle: %s\n' "$stage65_report"
