"""Freeze a train-source-only pilot for persistent FFN compression branches."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import fingerprint, load_manifest, write_json
from stage8_intervention_common import select_anchor_offsets
from stage19_compression_common import SCHEMA, nonempty_path


def build_manifest(source, calibration=12, check=12, anchors=4, horizon=20, seed=2057):
    source = load_manifest(source)
    if min(calibration, check, anchors) < 1 or horizon < 5:
        raise ValueError("Need positive split/anchor counts and horizon >= 5")
    eligible = []
    for entry in source["sequences"]:
        if entry["split"] != "train":
            continue
        try:
            select_anchor_offsets(entry["stop"] - entry["start"] - 1, horizon, anchors)
        except ValueError:
            continue
        eligible.append(entry)
    eligible.sort(key=lambda entry: entry["sequence"])
    if len(eligible) < calibration + check:
        raise ValueError("Only {} eligible source TRAIN videos; need {}. Source val is excluded.".format(
            len(eligible), calibration + check))
    order = np.random.default_rng(seed).permutation(len(eligible))[:calibration + check]
    entries = [dict(eligible[int(index)], split="train" if row < calibration else "val")
               for row, index in enumerate(order)]
    body = {key: source[key] for key in ("schema", "dataset", "root", "clip_frames")}
    body.update(seed=int(seed), sequences=entries, purpose=SCHEMA,
                source_manifest_fingerprint=source["fingerprint"],
                excluded_source_val_sequences=sorted(entry["sequence"] for entry in source["sequences"]
                                                     if entry["split"] == "val"),
                compression_design={"anchors": int(anchors), "horizon": int(horizon),
                                    "calibration_horizon": 5, "primary_horizon": int(horizon),
                                    "split_roles": {"train": "calibration", "val": "development_check"},
                                    "analysis_role": "development_only_mechanism_pilot"})
    body["fingerprint"] = fingerprint(body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", type=nonempty_path, required=True)
    parser.add_argument("--output", type=nonempty_path, required=True)
    parser.add_argument("--calibration_sequences", type=int, default=12)
    parser.add_argument("--check_sequences", type=int, default=12)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2057)
    args = parser.parse_args()
    path = Path(args.output).expanduser().resolve()
    if path.exists():
        raise FileExistsError("Manifest exists; use a new output path: {}".format(path))
    data = build_manifest(args.source_manifest, args.calibration_sequences,
                          args.check_sequences, args.anchors, args.horizon, args.seed)
    write_json(path, data)
    print("Manifest: {} ({} calibration / {} development-check videos)".format(
        path, args.calibration_sequences, args.check_sequences))
    print("All selected videos come from source train; source val is excluded.")
    print("Fingerprint: {}".format(data["fingerprint"]))


if __name__ == "__main__":
    main()
