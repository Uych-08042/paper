#!/usr/bin/env bash
set -euo pipefail
stage73_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage73_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE72_ROOT="${STAGE72_ROOT:-$PWD/output/stage72/cached_ppo_diagnostic_v1}"
export STAGE73_ROOT="${STAGE73_ROOT:-$PWD/output/stage73/separate_gradient_clip_ppo_v1}"
stage73_gpu="${STAGE73_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE72_ROOT/analysis/stage72_diagnostic.json"
python -B tools/stage73_ppo_experiment.py --mode prepare --input "$STAGE72_ROOT" --output_dir "$STAGE73_ROOT" --gpu "$stage73_gpu"
for stage73_mode in source_bridge train_smoke train smoke evaluate; do
  python -B tools/stage73_ppo_experiment.py --mode "$stage73_mode" --input "$STAGE73_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage73_gpu"
done
python -B tools/stage73_ppo_summary.py --input "$STAGE73_ROOT"
stage73_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage73_dir"
stage73_report="$stage73_dir/stage73_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage73_files=(stage73_decision.json stage73_policy_summary.csv stage73_paired.csv stage73_seed_effects.csv stage73_factor_summary.csv stage73_calibration.csv stage73_training_scope.csv stage73_fit_support.csv stage73_optimization_summary.csv stage73_training_summary.csv stage73_focus.csv stage73_scope.csv stage73_video_influence.csv)
for stage73_file in "${stage73_files[@]}"; do test -f "$STAGE73_ROOT/analysis/$stage73_file"; done
{
  for stage73_file in "${stage73_files[@]}"; do
    printf '\n===== %s =====\n' "$stage73_file"
    cat -- "$STAGE73_ROOT/analysis/$stage73_file"
    printf '\n'
  done
} > "$stage73_report"
printf 'Stage-73 result bundle: %s\n' "$stage73_report"
