#!/usr/bin/env bash
set -euo pipefail
stage71_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage71_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE70_ROOT="${STAGE70_ROOT:-$PWD/output/stage70/fit_history_source_v1}"
export STAGE71_ROOT="${STAGE71_ROOT:-$PWD/output/stage71/bounded_on_policy_search_ppo_v1}"
stage71_gpu="${STAGE71_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE70_ROOT/analysis/stage70_decision.json"
python -B tools/stage71_ppo_experiment.py --mode prepare --input "$STAGE70_ROOT" --output_dir "$STAGE71_ROOT" --gpu "$stage71_gpu"
for stage71_mode in source_bridge train_smoke train smoke evaluate; do
  python -B tools/stage71_ppo_experiment.py --mode "$stage71_mode" --input "$STAGE71_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage71_gpu"
done
python -B tools/stage71_ppo_summary.py --input "$STAGE71_ROOT"
stage71_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage71_dir"
stage71_report="$stage71_dir/stage71_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage71_files=(stage71_decision.json stage71_policy_summary.csv stage71_paired.csv stage71_seed_effects.csv stage71_calibration.csv stage71_training_scope.csv stage71_fit_support.csv stage71_focus.csv stage71_scope.csv stage71_video_influence.csv)
for stage71_file in "${stage71_files[@]}"; do test -f "$STAGE71_ROOT/analysis/$stage71_file"; done
{
  for stage71_file in "${stage71_files[@]}"; do
    printf '\n===== %s =====\n' "$stage71_file"
    cat -- "$STAGE71_ROOT/analysis/$stage71_file"
    printf '\n'
  done
} > "$stage71_report"
printf 'Stage-71 result bundle: %s\n' "$stage71_report"
