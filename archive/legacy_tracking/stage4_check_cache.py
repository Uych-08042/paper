"""Audit complete Stage-4 caches before any router training (NumPy only)."""

import argparse
import json
from pathlib import Path

from stage4_common import (
    ACTIONS, IOU_PROTOCOL, SCHEMA, audit_cache, cache_path, fingerprint,
    load_manifest, write_csv,
)


def check_directory(directory):
    directory = Path(directory).expanduser().resolve()
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if (run.get("schema") != SCHEMA or run.get("iou_protocol") != IOU_PROTOCOL
            or run.get("actions") != list(ACTIONS)
            or run.get("fingerprint") != fingerprint({key: value for key, value in run.items() if key != "fingerprint"})):
        raise ValueError("Invalid run metadata, action order or fingerprint")
    manifest = run["manifest"]
    load_manifest(manifest)
    if not run["trajectories"] or not set(run["trajectories"]).issubset({"sgla", "fast"}):
        raise ValueError("Invalid rollout policies")
    expected = {cache_path(directory, entry, trajectory)
                for entry in manifest["sequences"] for trajectory in run["trajectories"]}
    found = set(directory.rglob("*.npz"))
    if expected != found:
        missing, extra = sorted(expected - found), sorted(found - expected)
        raise ValueError("Incomplete/mixed cache: {} missing, {} unexpected. Examples: {}".format(
            len(missing), len(extra), [str(path) for path in (missing + extra)[:5]]))
    reports = [audit_cache(cache_path(directory, entry, trajectory), entry, trajectory, run)
               for entry in manifest["sequences"] for trajectory in run["trajectories"]]
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    reports = check_directory(args.input)
    output = Path(args.input).expanduser().resolve() / "cache_summary.csv"
    write_csv(output, reports)
    print("PASS: disjoint video splits, complete clips, finite causal features, state continuity, IoU/delta labels")
    print("split  trajectory  videos  decisions  valid")
    for split, trajectory in sorted({(row["split"], row["trajectory"]) for row in reports}):
        rows = [row for row in reports if row["split"] == split and row["trajectory"] == trajectory]
        print("{:<6} {:<11} {:>6} {:>10} {:>6}".format(
            split, trajectory, len(rows), sum(row["decision_frames"] for row in rows),
            sum(row["valid_frames"] for row in rows)))
    print("Summary: {}".format(output))
    print("These are training clips, not benchmark AUC or a deployed router result.")


if __name__ == "__main__":
    main()
