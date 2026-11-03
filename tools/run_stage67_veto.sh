#!/usr/bin/env bash
set -euo pipefail
stage67_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage67_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE66_ROOT="${STAGE66_ROOT:-$PWD/output/stage66/native_q0_takeover_v1}"
export STAGE67_ROOT="${STAGE67_ROOT:-$PWD/output/stage67/matched_veto_continuation_v1}"
stage67_gpu="${STAGE67_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE66_ROOT/analysis/stage66_decision.json"
python -B tools/stage67_veto_experiment.py --mode prepare --input "$STAGE66_ROOT" --output_dir "$STAGE67_ROOT" --gpu "$stage67_gpu"
python -B tools/stage67_veto_experiment.py --mode smoke --input "$STAGE67_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage67_gpu"
python -B tools/stage67_veto_experiment.py --mode collect --input "$STAGE67_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage67_gpu"
python -B tools/stage67_veto_summary.py --input "$STAGE67_ROOT"
stage67_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage67_dir"
stage67_report="$stage67_dir/stage67_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage67_files=(stage67_diagnostic.json stage67_anchors.csv stage67_case_summary.csv stage67_branch_summary.csv stage67_contrasts.csv stage67_scope.csv stage67_cached_video.csv stage67_divergences.csv)
for stage67_file in "${stage67_files[@]}"; do test -f "$STAGE67_ROOT/analysis/$stage67_file"; done
{
  for stage67_file in "${stage67_files[@]}"; do
    printf '\n===== %s =====\n' "$stage67_file"
    cat -- "$STAGE67_ROOT/analysis/$stage67_file"
    printf '\n'
  done
} > "$stage67_report"
printf 'Stage-67 result bundle: %s\n' "$stage67_report"
