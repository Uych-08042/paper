#!/usr/bin/env bash
set -euo pipefail
stage70_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage70_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE69_ROOT="${STAGE69_ROOT:-$PWD/output/stage69/cached_paired_gate_audit_v1}"
export STAGE70_ROOT="${STAGE70_ROOT:-$PWD/output/stage70/fit_history_source_v1}"
stage70_gpu="${STAGE70_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE69_ROOT/analysis/stage69_diagnostic.json"
python -B tools/stage70_history_experiment.py --mode prepare --input "$STAGE69_ROOT" --output_dir "$STAGE70_ROOT" --gpu "$stage70_gpu"
python -B tools/stage70_history_experiment.py --mode source_bridge --input "$STAGE70_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage70_gpu"
python -B tools/stage70_history_experiment.py --mode history_smoke --input "$STAGE70_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage70_gpu"
python -B tools/stage70_history_experiment.py --mode histories --input "$STAGE70_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage70_gpu"
python -B tools/stage70_history_experiment.py --mode resolve --input "$STAGE70_ROOT"
python -B tools/stage70_history_experiment.py --mode pair_smoke --input "$STAGE70_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage70_gpu"
python -B tools/stage70_history_experiment.py --mode collect --input "$STAGE70_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage70_gpu"
python -B tools/stage70_history_experiment.py --mode fit --input "$STAGE70_ROOT"
python -B tools/stage70_history_experiment.py --mode smoke --input "$STAGE70_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage70_gpu"
python -B tools/stage70_history_experiment.py --mode evaluate --input "$STAGE70_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage70_gpu"
python -B tools/stage70_history_summary.py --input "$STAGE70_ROOT"
stage70_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage70_dir"
stage70_report="$stage70_dir/stage70_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage70_files=(stage70_decision.json stage70_policy_summary.csv stage70_paired.csv stage70_factor_summary.csv stage70_calibration.csv stage70_fit_summary.csv stage70_pair_support.csv stage70_history_support.csv stage70_paired_probe.csv stage70_focus.csv stage70_scope.csv stage70_video_influence.csv)
for stage70_file in "${stage70_files[@]}"; do test -f "$STAGE70_ROOT/analysis/$stage70_file"; done
{
  for stage70_file in "${stage70_files[@]}"; do
    printf '\n===== %s =====\n' "$stage70_file"
    cat -- "$STAGE70_ROOT/analysis/$stage70_file"
    printf '\n'
  done
} > "$stage70_report"
printf 'Stage-70 result bundle: %s\n' "$stage70_report"
