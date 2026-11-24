#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE181_ROOT:-$repo/output/stage181/paired_center_audit_v1}"

python -u -B tools/stage181_paired_center_audit.py \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage167-root "${STAGE167_ROOT:-$repo/output/stage167/online_layer_ranking_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage176-root "${STAGE176_ROOT:-$repo/output/stage176/layer_event_sweep_v1}" \
  --stage177-root "${STAGE177_ROOT:-$repo/output/stage177/native_proposal_audit_v1}" \
  --stage180-root "${STAGE180_ROOT:-$repo/output/stage180/center_match_audit_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE181_GPU:-0}" --progress-interval "${STAGE181_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage181_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json cases.csv signals.csv outcomes.csv association.csv sentinels.csv; do
    printf '\n===== stage181_%s =====\n' "$file"
    cat -- "$run/stage181_$file"
  done
} > "$bundle"
printf '[Stage-181] result bundle: %s\n' "$bundle"
