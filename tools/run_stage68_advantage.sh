#!/usr/bin/env bash
set -euo pipefail
stage68_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage68_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE67_ROOT="${STAGE67_ROOT:-$PWD/output/stage67/matched_veto_continuation_v1}"
export STAGE68_ROOT="${STAGE68_ROOT:-$PWD/output/stage68/fit_paired_takeover_advantage_v1}"
stage68_gpu="${STAGE68_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE67_ROOT/analysis/stage67_summary_repair.json"
python -B tools/stage68_advantage_experiment.py --mode prepare --input "$STAGE67_ROOT" --output_dir "$STAGE68_ROOT" --gpu "$stage68_gpu"
python -B tools/stage68_advantage_experiment.py --mode pair_smoke --input "$STAGE68_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage68_gpu"
python -B tools/stage68_advantage_experiment.py --mode collect --input "$STAGE68_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage68_gpu"
python -B tools/stage68_advantage_experiment.py --mode fit --input "$STAGE68_ROOT"
python -B tools/stage68_advantage_experiment.py --mode smoke --input "$STAGE68_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage68_gpu"
python -B tools/stage68_advantage_experiment.py --mode evaluate --input "$STAGE68_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage68_gpu"
python -B tools/stage68_advantage_summary.py --input "$STAGE68_ROOT"
stage68_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage68_dir"
stage68_report="$stage68_dir/stage68_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage68_files=(stage68_decision.json stage68_policy_summary.csv stage68_paired.csv stage68_calibration.csv stage68_fit_summary.csv stage68_pair_support.csv stage68_scope.csv stage68_focus.csv stage68_video_influence.csv)
for stage68_file in "${stage68_files[@]}"; do test -f "$STAGE68_ROOT/analysis/$stage68_file"; done
{
  for stage68_file in "${stage68_files[@]}"; do
    printf '\n===== %s =====\n' "$stage68_file"
    cat -- "$STAGE68_ROOT/analysis/$stage68_file"
    printf '\n'
  done
} > "$stage68_report"
printf 'Stage-68 result bundle: %s\n' "$stage68_report"
