#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run="${STAGE134_ROOT:-$repo/output/stage134/native_peak_continuation_v1}"
python -B tools/stage134_native_peak_continuation.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage133 "${STAGE133_ROOT:-$repo/output/stage133/native_failure_audit_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE134_GPU:-0}" --output-dir "$run"
python -B tools/stage134_native_peak_continuation.py --check-results --output-dir "$run"
cat -- "$run/stage134_summary.csv"
bundle="$run/stage134_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv windows.csv scope.csv; do
    printf '\n===== stage134_%s =====\n' "$file"
    cat -- "$run/stage134_$file"
  done
} > "$bundle"
printf 'Stage-134 result bundle: %s\n' "$bundle"
