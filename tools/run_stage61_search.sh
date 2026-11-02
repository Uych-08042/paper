#!/usr/bin/env bash
# Fixed Stage-61 workflow and a portable, cat-produced result bundle. No Git mutations.
set -euo pipefail

stage61_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage61_repo_root"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE60_ROOT="${STAGE60_ROOT:-$PWD/output/stage60/causal_search_state_value_v1}"
export STAGE61_ROOT="${STAGE61_ROOT:-$PWD/output/stage61/v1_history_matched_recrop_v1}"
stage61_gpu="${STAGE61_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE60_ROOT/manifest.json"

python -B tracking/stage61_search_experiment.py --mode prepare --input "$STAGE60_ROOT" --output_dir "$STAGE61_ROOT"
python -B tracking/stage61_search_experiment.py --mode smoke --input "$STAGE61_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage61_gpu"
python -B tracking/stage61_search_experiment.py --mode evaluate --input "$STAGE61_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage61_gpu"
python -B tracking/stage61_search_summary.py --input "$STAGE61_ROOT"

stage61_report_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage61_report_dir"
stage61_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
stage61_report="$stage61_report_dir/stage61_results_${stage61_stamp}_$$.txt"
stage61_files=(stage61_decision.json stage61_policy_summary.csv stage61_paired.csv
  stage61_video_influence.csv stage61_scope_summary.csv stage61_proposal_summary.csv
  stage61_video_summary.csv stage61_seed_video_summary.csv stage61_events.csv)
for stage61_file in "${stage61_files[@]}"; do
  test -f "$STAGE61_ROOT/analysis/$stage61_file"
done
{
  for stage61_file in "${stage61_files[@]}"; do
    printf '\n===== %s =====\n' "$stage61_file"
    cat -- "$STAGE61_ROOT/analysis/$stage61_file"
    printf '\n'
  done
} > "$stage61_report"
printf 'Stage-61 result bundle: %s\n' "$stage61_report"
