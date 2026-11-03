#!/usr/bin/env bash
set -euo pipefail
stage64_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage64_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE63_ROOT="${STAGE63_ROOT:-$PWD/output/stage63/matched_one_choice_continuation_v1}"
export STAGE64_ROOT="${STAGE64_ROOT:-$PWD/output/stage64/causal_candidate_reliability_v1}"
stage64_gpu="${STAGE64_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE63_ROOT/analysis/stage63_diagnostic.json"
python -B tools/stage64_reliability_experiment.py --mode prepare --input "$STAGE63_ROOT" --output_dir "$STAGE64_ROOT" --gpu "$stage64_gpu"
python -B tools/stage64_reliability_experiment.py --mode fit --input "$STAGE64_ROOT"
python -B tools/stage64_reliability_experiment.py --mode smoke --input "$STAGE64_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage64_gpu"
python -B tools/stage64_reliability_experiment.py --mode evaluate --input "$STAGE64_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage64_gpu"
python -B tools/stage64_reliability_summary.py --input "$STAGE64_ROOT"
stage64_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage64_dir"
stage64_report="$stage64_dir/stage64_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage64_files=(stage64_decision.json stage64_policy_summary.csv stage64_paired.csv stage64_scope.csv stage64_fit_support.csv stage64_video_influence.csv stage64_focus.csv)
for stage64_file in "${stage64_files[@]}"; do test -f "$STAGE64_ROOT/analysis/$stage64_file"; done
{
  for stage64_file in "${stage64_files[@]}"; do
    printf '\n===== %s =====\n' "$stage64_file"
    cat -- "$STAGE64_ROOT/analysis/$stage64_file"
    printf '\n'
  done
} > "$stage64_report"
printf 'Stage-64 result bundle: %s\n' "$stage64_report"
