#!/usr/bin/env bash
set -euo pipefail
stage67_repair_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage67_repair_repo"
export STAGE67_ROOT="${STAGE67_ROOT:-$PWD/output/stage67/matched_veto_continuation_v1}"
test -f "$STAGE67_ROOT/collection/run.json"
python -B tools/stage67_summary_repair.py --input "$STAGE67_ROOT"
stage67_repair_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage67_repair_dir"
stage67_repair_report="$stage67_repair_dir/stage67_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage67_repair_files=(stage67_diagnostic.json stage67_summary_repair.json stage67_anchors.csv stage67_case_summary.csv stage67_branch_summary.csv stage67_contrasts.csv stage67_scope.csv stage67_cached_video.csv stage67_divergences.csv)
for stage67_repair_file in "${stage67_repair_files[@]}"; do test -f "$STAGE67_ROOT/analysis/$stage67_repair_file"; done
{
  for stage67_repair_file in "${stage67_repair_files[@]}"; do
    printf '\n===== %s =====\n' "$stage67_repair_file"
    cat -- "$STAGE67_ROOT/analysis/$stage67_repair_file"
    printf '\n'
  done
} > "$stage67_repair_report"
printf 'Stage-67 result bundle: %s\n' "$stage67_repair_report"
