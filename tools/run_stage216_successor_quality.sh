#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run="${STAGE216_ROOT:-$repo/output/stage216/common_state_successor_v1}"
python -u -B tools/stage216_successor_quality.py \
  --stage214-root "${STAGE214_ROOT:-$repo/output/stage214/learned_search_state_v1}" \
  --stage215-root "${STAGE215_REPLAY_ROOT:-$repo/output/stage215/same_model_gpu_replay_v1}" \
  --stage208-root "${STAGE208_ROOT:-$repo/output/stage208/paired_state_interaction_v1}" \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE216_GPU:-0}" --workers "${STAGE216_WORKERS:-2}" --output-dir "$run" "$@"
bundle="$run/stage216_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage216_decision.json manifest.json stage216_selection.csv stage216_model_scope.csv stage216_source_replay.csv stage216_cells.csv stage216_policy_summary.csv stage216_paired.csv stage216_alignment.csv stage216_calls.csv stage216_smoke.csv stage216_execution.csv)
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-216] result bundle: %s\n' "$bundle"
