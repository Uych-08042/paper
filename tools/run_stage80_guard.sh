#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage77="${STAGE77_ROOT:-$repo/output/stage77/sgla_search_decode_diagnostic_v1}"
stage79="${STAGE79_ROOT:-$repo/output/stage79/scale_action_cross_v1}"
run="${STAGE80_ROOT:-$repo/output/stage80/sgla_scale_consensus_full_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
case "${1:-run}" in
  run)
    if [[ ! -f "$run/analysis/stage80_decision.json" ]]; then
      python -B tools/stage80_guard_experiment.py --stage76 "$stage76" --stage77 "$stage77" --stage79 "$stage79" \
        --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE80_GPU:-0}"
    fi
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage80_guard_experiment.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage80_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv video_summary.csv seed_video_summary.csv scope.csv; do
    printf '\n===== stage80_%s =====\n' "$file"
    cat -- "$run/analysis/stage80_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-80 result bundle: %s\n' "$bundle"
