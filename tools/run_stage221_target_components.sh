#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run="${STAGE221_ROOT:-$repo/output/stage221/frozen_target_components_v1}"
python -u -B tools/stage221_target_components.py \
  --stage220-root "${STAGE220_ROOT:-$repo/output/stage220/causal_second_difference_v1}" \
  --stage219-root "${STAGE219_ROOT:-$repo/output/stage219/matched_mlp_input_blocks_v1}" \
  --stage218-root "${STAGE218_ROOT:-$repo/output/stage218/matched_forecast_capacity_v1}" \
  --stage217-root "${STAGE217_ROOT:-$repo/output/stage217/cached_predictor_objective_v1}" \
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
bundle="$run/stage221_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage221_decision.json manifest.json stage221_support.csv stage221_scope.csv stage221_cells.csv stage221_actions.csv stage221_group_metrics.csv stage221_policy_summary.csv stage221_paired.csv stage221_source_replay.csv stage221_execution.csv)
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-221] result bundle: %s\n' "$bundle"
