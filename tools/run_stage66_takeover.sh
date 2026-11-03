#!/usr/bin/env bash
set -euo pipefail
stage66_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage66_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE65_ROOT="${STAGE65_ROOT:-$PWD/output/stage65/cached_same_state_selection_v1}"
export STAGE66_ROOT="${STAGE66_ROOT:-$PWD/output/stage66/native_q0_takeover_v1}"
stage66_gpu="${STAGE66_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE65_ROOT/analysis/stage65_diagnostic.json"
python -B tools/stage66_takeover_experiment.py --mode prepare --input "$STAGE65_ROOT" --output_dir "$STAGE66_ROOT" --gpu "$stage66_gpu"
python -B tools/stage66_takeover_experiment.py --mode fit --input "$STAGE66_ROOT"
python -B tools/stage66_takeover_experiment.py --mode smoke --input "$STAGE66_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage66_gpu"
python -B tools/stage66_takeover_experiment.py --mode evaluate --input "$STAGE66_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage66_gpu"
python -B tools/stage66_takeover_summary.py --input "$STAGE66_ROOT"
stage66_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage66_dir"
stage66_report="$stage66_dir/stage66_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage66_files=(stage66_decision.json stage66_policy_summary.csv stage66_paired.csv stage66_calibration.csv stage66_scope.csv stage66_focus.csv stage66_video_influence.csv)
for stage66_file in "${stage66_files[@]}"; do test -f "$STAGE66_ROOT/analysis/$stage66_file"; done
{
  for stage66_file in "${stage66_files[@]}"; do
    printf '\n===== %s =====\n' "$stage66_file"
    cat -- "$STAGE66_ROOT/analysis/$stage66_file"
    printf '\n'
  done
} > "$stage66_report"
printf 'Stage-66 result bundle: %s\n' "$stage66_report"
