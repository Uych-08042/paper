#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run="${STAGE222_ROOT:-$repo/output/stage222/cached_localization_evidence_v1}"
python -u -B tools/stage222_localization_evidence.py \
  --stage221-root "${STAGE221_ROOT:-$repo/output/stage221/frozen_target_components_v1}" \
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
bundle="$run/stage222_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage222_decision.json manifest.json stage222_selection.csv stage222_cells.csv stage222_group_metrics.csv stage222_policy_summary.csv stage222_primary.csv stage222_source_replay.csv stage222_execution.csv)
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-222] result bundle: %s\n' "$bundle"
