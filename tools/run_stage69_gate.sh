#!/usr/bin/env bash
set -euo pipefail
stage69_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage69_repo"
export STAGE68_ROOT="${STAGE68_ROOT:-$PWD/output/stage68/fit_paired_takeover_advantage_v1}"
export STAGE69_ROOT="${STAGE69_ROOT:-$PWD/output/stage69/cached_paired_gate_audit_v1}"
test -f "$STAGE68_ROOT/analysis/stage68_decision.json"
python -B tools/stage69_gate_audit.py --input "$STAGE68_ROOT" --output_dir "$STAGE69_ROOT"
stage69_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage69_dir"
stage69_report="$stage69_dir/stage69_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage69_files=(stage69_diagnostic.json stage69_fit_prediction.csv stage69_fit_sign_groups.csv stage69_paired_case.csv stage69_paired_policy.csv stage69_same_state_summary.csv stage69_matched_summary.csv stage69_candidate_summary.csv stage69_feature_shift.csv stage69_anchors.csv stage69_focus.csv stage69_divergences.csv stage69_scope.csv)
for stage69_file in "${stage69_files[@]}"; do test -f "$STAGE69_ROOT/analysis/$stage69_file"; done
{
  for stage69_file in "${stage69_files[@]}"; do
    printf '\n===== %s =====\n' "$stage69_file"
    cat -- "$STAGE69_ROOT/analysis/$stage69_file"
    printf '\n'
  done
} > "$stage69_report"
printf 'Stage-69 result bundle: %s\n' "$stage69_report"
