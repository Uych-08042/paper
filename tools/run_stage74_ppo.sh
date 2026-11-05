#!/usr/bin/env bash
set -euo pipefail
stage74_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage74_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE73_ROOT="${STAGE73_ROOT:-$PWD/output/stage73/separate_gradient_clip_ppo_v1}"
export STAGE74_ROOT="${STAGE74_ROOT:-$PWD/output/stage74/direct_four_candidate_ppo_v1}"
stage74_gpu="${STAGE74_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE73_ROOT/analysis/stage73_decision.json"
python -B tools/stage74_ppo_experiment.py --mode prepare --input "$STAGE73_ROOT" --output_dir "$STAGE74_ROOT" --gpu "$stage74_gpu"
for stage74_mode in source_bridge train_smoke train smoke evaluate; do
  python -B tools/stage74_ppo_experiment.py --mode "$stage74_mode" --input "$STAGE74_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage74_gpu"
done
python -B tools/stage74_ppo_summary.py --input "$STAGE74_ROOT"
stage74_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage74_dir"
stage74_report="$stage74_dir/stage74_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage74_files=(stage74_decision.json stage74_policy_summary.csv stage74_paired.csv stage74_seed_effects.csv stage74_training_scope.csv stage74_fit_support.csv stage74_training_summary.csv stage74_optimization_summary.csv stage74_candidate_summary.csv stage74_focus.csv stage74_scope.csv stage74_video_influence.csv)
for stage74_file in "${stage74_files[@]}"; do test -f "$STAGE74_ROOT/analysis/$stage74_file"; done
{
  for stage74_file in "${stage74_files[@]}"; do
    printf '\n===== %s =====\n' "$stage74_file"
    cat -- "$STAGE74_ROOT/analysis/$stage74_file"
    printf '\n'
  done
} > "$stage74_report"
printf 'Stage-74 result bundle: %s\n' "$stage74_report"
