#!/usr/bin/env bash
# Matched one-choice short continuations on L40; preserve all source stages.
set -euo pipefail
stage63_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stage63_repo"
export CHECKPOINT="${CHECKPOINT:-$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export STAGE62_ROOT="${STAGE62_ROOT:-$PWD/output/stage62/cached_candidate_selection_v1}"
export STAGE63_ROOT="${STAGE63_ROOT:-$PWD/output/stage63/matched_one_choice_continuation_v1}"
stage63_gpu="${STAGE63_GPU:-0}"
test -f "$CHECKPOINT"
test -f "$STAGE62_ROOT/analysis/stage62_diagnostic.json"

python -B tools/stage63_choice_experiment.py --mode prepare --input "$STAGE62_ROOT" --output_dir "$STAGE63_ROOT" --gpu "$stage63_gpu"
python -B tools/stage63_choice_experiment.py --mode smoke --input "$STAGE63_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage63_gpu"
python -B tools/stage63_choice_experiment.py --mode collect --input "$STAGE63_ROOT" --checkpoint "$CHECKPOINT" --gpu "$stage63_gpu"
python -B tools/stage63_choice_summary.py --input "$STAGE63_ROOT"

stage63_dir="$PWD/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$stage63_dir"
stage63_report="$stage63_dir/stage63_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
stage63_files=(stage63_diagnostic.json stage63_case_summary.csv stage63_contrasts.csv
  stage63_scope.csv stage63_branch_summary.csv stage63_dense_focus.csv)
for stage63_file in "${stage63_files[@]}"; do
  test -f "$STAGE63_ROOT/analysis/$stage63_file"
done
{
  for stage63_file in "${stage63_files[@]}"; do
    printf '\n===== %s =====\n' "$stage63_file"
    cat -- "$STAGE63_ROOT/analysis/$stage63_file"
    printf '\n'
  done
} > "$stage63_report"
printf 'Stage-63 result bundle: %s\n' "$stage63_report"
printf 'Detailed timelines and response-map caches: %s\n' "$STAGE63_ROOT"
