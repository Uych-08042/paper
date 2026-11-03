#!/usr/bin/env bash
# CPU-only audit of existing candidate caches; does not run inference or Git mutations.
set -euo pipefail
stage62_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage62_repo"
export STAGE61_L40_ROOT="${STAGE61_L40_ROOT:-$PWD/output/stage61/v1_history_matched_recrop_l40_v1}"
export STAGE62_ROOT="${STAGE62_ROOT:-$PWD/output/stage62/cached_candidate_selection_v1}"
test -f "$STAGE61_L40_ROOT/analysis/stage61_l40_decision.json"
python -B tools/stage62_candidate_audit.py --input "$STAGE61_L40_ROOT" --output_dir "$STAGE62_ROOT"

stage62_report_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage62_report_dir"
stage62_report="$stage62_report_dir/stage62_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage62_files=(stage62_diagnostic.json stage62_policy_summary.csv stage62_scope.csv stage62_transitions.csv stage62_focus.csv)
for stage62_file in "${stage62_files[@]}"; do
  test -f "$STAGE62_ROOT/analysis/$stage62_file"
done
{
  for stage62_file in "${stage62_files[@]}"; do
    printf '\n===== %s =====\n' "$stage62_file"
    cat -- "$STAGE62_ROOT/analysis/$stage62_file"
    printf '\n'
  done
} > "$stage62_report"
printf 'Stage-62 result bundle: %s\n' "$stage62_report"
printf 'Detailed seed/video/timeline/candidate CSVs: %s/analysis\n' "$STAGE62_ROOT"
