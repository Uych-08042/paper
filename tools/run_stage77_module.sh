#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source_root="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
run_root="${STAGE77_ROOT:-$repo/output/stage77/sgla_search_decode_diagnostic_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
if [[ $# -eq 0 ]]; then
  python -B tools/stage77_module_diagnostic.py --input "$source_root" --output_dir "$run_root" \
    --checkpoint "$checkpoint" --gpu "${STAGE77_GPU:-0}"
elif [[ $# -ne 1 || $1 != --pack-only ]]; then
  printf 'Usage: %s [--pack-only]\n' "$0" >&2
  exit 2
fi

files=(diagnostic.json summary.csv states.csv candidates.csv)
python -B - "$run_root" "${files[@]:1}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
analysis = root / "analysis"

def load_sealed(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = record.pop("fingerprint")
    actual = hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if record["schema"] != "stage77_sgla_search_decode_diagnostic_v1" or actual != fingerprint:
        raise SystemExit(f"Invalid Stage-77 record: {path}")
    return dict(record, fingerprint=fingerprint)

plan = load_sealed(root / "manifest.json")
report = load_sealed(analysis / "stage77_diagnostic.json")
if (report["status"] != "COMPLETE_SGLA_MODULE_DIAGNOSTIC" or report["plan_fingerprint"] != plan["fingerprint"]
        or report["states"] != plan["states"] or report["search_calls"] != plan["expected_search_calls"]):
    raise SystemExit("Need the completed Stage-77 diagnostic for this run")
tables = {f"stage77_{name}": hashlib.sha256((analysis / f"stage77_{name}").read_bytes()).hexdigest()
          for name in sys.argv[2:]}
if tables != report["table_sha256"]:
    raise SystemExit("Stage-77 table hashes differ from the diagnostic")
PY

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage77_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in "${files[@]}"; do
    printf '\n===== stage77_%s =====\n' "$file"
    cat -- "$run_root/analysis/stage77_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-77 result bundle: %s\n' "$bundle"
