#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE155_ROOT:-$repo/output/stage155/isolated_l10_adaptation_v1}"

python -B tools/stage155_adapt_serial_l10.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage134 "${STAGE134_ROOT:-$repo/output/stage134/native_peak_continuation_v1}" \
  --stage154 "${STAGE154_ROOT:-$repo/output/stage154/forced_serial_depth_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE155_GPU:-0}" --output-dir "$run"
python -B tools/stage155_adapt_serial_l10.py --check-results --output-dir "$run"
cat -- "$run/stage155_summary.csv"

bundle="$run/stage155_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json training.csv fit_support.csv summary.csv anchors.csv windows.csv; do
    printf '\n===== stage155_%s =====\n' "$file"
    cat -- "$run/stage155_$file"
  done
} > "$bundle"
printf 'Stage-155 result bundle: %s\n' "$bundle"
