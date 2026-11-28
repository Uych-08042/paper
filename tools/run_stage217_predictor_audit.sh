#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run="${STAGE217_ROOT:-$repo/output/stage217/cached_predictor_objective_v1}"
python -u -B tools/stage217_predictor_audit.py \
  --stage214-root "${STAGE214_ROOT:-$repo/output/stage214/learned_search_state_v1}" \
  --stage215-root "${STAGE215_REPLAY_ROOT:-$repo/output/stage215/same_model_gpu_replay_v1}" \
  --stage216-root "${STAGE216_ROOT:-$repo/output/stage216/common_state_successor_v1}" \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --output-dir "$run" "$@"
bundle="$run/stage217_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage217_decision.json manifest.json stage217_support.csv stage217_model_scope.csv stage217_constants.csv stage217_normalization.csv stage217_group_metrics.csv stage217_policy_summary.csv stage217_paired.csv stage217_prediction_replay.csv stage217_execution.csv)
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-217] result bundle: %s\n' "$bundle"
