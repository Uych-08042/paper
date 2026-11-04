#!/usr/bin/env bash
set -euo pipefail
stage72_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage72_repo"
export STAGE71_ROOT="${STAGE71_ROOT:-$PWD/output/stage71/bounded_on_policy_search_ppo_v1}"
export STAGE72_ROOT="${STAGE72_ROOT:-$PWD/output/stage72/cached_ppo_diagnostic_v1}"
test -f "$STAGE71_ROOT/analysis/stage71_decision.json"
python -B tools/stage72_ppo_audit.py --input "$STAGE71_ROOT" --output_dir "$STAGE72_ROOT"
stage72_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage72_dir"
stage72_report="$stage72_dir/stage72_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage72_files=(stage72_diagnostic.json stage72_same_state_summary.csv stage72_matched_summary.csv stage72_candidate_summary.csv stage72_training_summary.csv stage72_optimization_summary.csv stage72_focus.csv stage72_divergences.csv stage72_anchors.csv stage72_scope.csv)
for stage72_file in "${stage72_files[@]}"; do test -f "$STAGE72_ROOT/analysis/$stage72_file"; done
{
  for stage72_file in "${stage72_files[@]}"; do
    printf '\n===== %s =====\n' "$stage72_file"
    cat -- "$STAGE72_ROOT/analysis/$stage72_file"
    printf '\n'
  done
} > "$stage72_report"
printf 'Stage-72 result bundle: %s\n' "$stage72_report"
