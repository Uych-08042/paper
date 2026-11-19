#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE137_ROOT:-$repo/output/stage137/shared_crop_layers_v1}"
python -B tools/stage137_shared_crop_layers.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage133 "${STAGE133_ROOT:-$repo/output/stage133/native_failure_audit_v1}" \
  --stage134 "${STAGE134_ROOT:-$repo/output/stage134/native_peak_continuation_v1}" \
  --stage136 "${STAGE136_ROOT:-$repo/output/stage136/same_crop_peak_closed_loop_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE137_GPU:-0}" --output-dir "$run"
python -B tools/stage137_shared_crop_layers.py --check-results --output-dir "$run"
cat -- "$run/stage137_summary.csv"
bundle="$run/stage137_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv cases.csv layers.csv; do
    printf '\n===== stage137_%s =====\n' "$file"
    cat -- "$run/stage137_$file"
  done
} > "$bundle"
printf 'Stage-137 result bundle: %s\n' "$bundle"
