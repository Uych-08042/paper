#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
run="${STAGE91_ROOT:-$repo/output/stage91/frozen_template_appearance_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
case "${1:-run}" in
  run)
    if [[ ! -f "$run/analysis/stage91_decision.json" ]]; then
      python -B tools/stage91_template_appearance.py --stage76 "$stage76" --stage89 "$stage89" \
        --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE91_GPU:-0}"
    fi
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage91_template_appearance.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage91_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json signal_support.csv paired.csv appearance.csv; do
    printf '\n===== stage91_%s =====\n' "$file"
    cat -- "$run/analysis/stage91_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-91 result bundle: %s\n' "$bundle"
