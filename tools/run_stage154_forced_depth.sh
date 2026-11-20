#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE154_ROOT:-$repo/output/stage154/forced_serial_depth_v1}"

python -B tools/stage154_forced_depth_continuation.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage134 "${STAGE134_ROOT:-$repo/output/stage134/native_peak_continuation_v1}" \
  --stage137 "${STAGE137_ROOT:-$repo/output/stage137/shared_crop_layers_v2}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE154_GPU:-0}" --output-dir "$run"
python -B tools/stage154_forced_depth_continuation.py --check-results --output-dir "$run"
cat -- "$run/stage154_summary.csv"

bundle="$run/stage154_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv anchors.csv windows.csv timeline.csv; do
    printf '\n===== stage154_%s =====\n' "$file"
    cat -- "$run/stage154_$file"
  done
} > "$bundle"
printf 'Stage-154 result bundle: %s\n' "$bundle"
