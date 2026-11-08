#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage77="${STAGE77_ROOT:-$repo/output/stage77/sgla_search_decode_diagnostic_v1}"
stage78="${STAGE78_ROOT:-$repo/output/stage78/sgla_scale_feedback_v1}"
run="${STAGE79_ROOT:-$repo/output/stage79/scale_action_cross_v1}"
confirmation="${STAGE79_CONFIRMATION:-$repo/output/stage79/confirmation_v1.json}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"

freeze_confirmation() {
  local exclusions=()
  if [[ -n "${STAGE79_EXCLUDE_FILE:-}" ]]; then
    exclusions=(--exclude-file "$STAGE79_EXCLUDE_FILE")
  fi
  python -B tools/stage79_confirmation.py \
    --dataset-root "${VISDRONE_SOT_ROOT:-$HOME/workspace/datasets/VisDrone-SOT/VisDrone2019-SOT-train}" \
    --stage27 "${STAGE27_ROOT:-$repo/output/stage27/uav_failure_diagnostic_v1}/manifest.json" \
    --stage31 "${STAGE31_ROOT:-$repo/output/stage31/continuous_identity_pilot_v1}/manifest.json" \
    --stage76 "$stage76/manifest.json" --output "$confirmation" "${exclusions[@]}"
}

case "${1:-run}" in
  --freeze-only) freeze_confirmation; exit ;;
  run)
    if [[ ! -f "$run/analysis/stage79_decision.json" ]]; then
      freeze_confirmation
      python -B tools/stage79_scale_cross.py --stage76 "$stage76" --stage77 "$stage77" --stage78 "$stage78" \
        --confirmation "$confirmation" --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE79_GPU:-0}"
    fi
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--freeze-only|--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage79_scale_cross.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage79_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json candidates.csv branches.csv timeline.csv; do
    printf '\n===== stage79_%s =====\n' "$file"
    cat -- "$run/analysis/stage79_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-79 result bundle: %s\n' "$bundle"
