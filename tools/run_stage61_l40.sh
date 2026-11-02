#!/usr/bin/env bash
# All six fixed Stage-61 arms are newly executed on L40; no Git mutations.
set -euo pipefail

stage61_l40_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage61_l40_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE61_ROOT="${STAGE61_ROOT:-$PWD/output/stage61/v1_history_matched_recrop_v1}"
export STAGE61_L40_ROOT="${STAGE61_L40_ROOT:-$PWD/output/stage61/v1_history_matched_recrop_l40_v1}"
stage61_l40_gpu="${STAGE61_L40_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE61_ROOT/manifest.json"

python -B tools/stage61_l40_experiment.py --mode prepare --input "$STAGE61_ROOT" --output_dir "$STAGE61_L40_ROOT" --gpu "$stage61_l40_gpu"
python -B tools/stage61_l40_experiment.py --mode smoke --input "$STAGE61_L40_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage61_l40_gpu"
python -B tools/stage61_l40_experiment.py --mode evaluate --input "$STAGE61_L40_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage61_l40_gpu"
python -B tools/stage61_l40_summary.py --input "$STAGE61_L40_ROOT"

stage61_l40_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage61_l40_dir"
stage61_l40_report="$stage61_l40_dir/stage61_l40_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage61_l40_files=(stage61_l40_decision.json stage61_l40_policy_summary.csv stage61_l40_paired.csv
  stage61_l40_video_influence.csv stage61_l40_scope_summary.csv stage61_l40_proposal_summary.csv
  stage61_l40_video_summary.csv stage61_l40_seed_video_summary.csv stage61_l40_events.csv stage61_l40_device_bridge.csv)
for stage61_l40_file in "${stage61_l40_files[@]}"; do
  test -f "$STAGE61_L40_ROOT/analysis/$stage61_l40_file"
done
for stage61_l40_mode in smoke evaluate; do
  test -f "$STAGE61_L40_ROOT/diagnostics/stage61_runtime_${stage61_l40_mode}.json"
done
{
  for stage61_l40_file in "${stage61_l40_files[@]}"; do
    printf '\n===== %s =====\n' "$stage61_l40_file"
    cat -- "$STAGE61_L40_ROOT/analysis/$stage61_l40_file"
    printf '\n'
  done
  for stage61_l40_mode in smoke evaluate; do
    printf '\n===== stage61_runtime_%s.json =====\n' "$stage61_l40_mode"
    cat -- "$STAGE61_L40_ROOT/diagnostics/stage61_runtime_${stage61_l40_mode}.json"
    printf '\n'
  done
} > "$stage61_l40_report"
printf 'Stage-61 L40 result bundle: %s\n' "$stage61_l40_report"
