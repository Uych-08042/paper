#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run="${STAGE135_ROOT:-$repo/output/stage135/cached_peak_gate_v1}"
python -B tools/stage135_peak_gate.py \
  --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage134 "${STAGE134_ROOT:-$repo/output/stage134/native_peak_continuation_v1}" \
  --output-dir "$run"
cat -- "$run/stage135_summary.csv"
bundle="$run/stage135_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv decisions.csv; do
    printf '\n===== stage135_%s =====\n' "$file"
    cat -- "$run/stage135_$file"
  done
} > "$bundle"
printf 'Stage-135 result bundle: %s\n' "$bundle"
