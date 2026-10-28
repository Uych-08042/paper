"""Create a Stage-4-compatible holdout manifest excluding prior manifests."""

import argparse
from pathlib import Path

from stage4_common import fingerprint, load_manifest, write_json
from stage4_manifest import build_manifest, subset_names, training_names


def exclusion_records(paths, dataset):
    records = []
    excluded = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        manifest = load_manifest(path)
        if manifest["dataset"] != dataset:
            raise ValueError(
                "Excluded manifest dataset {} does not match {}: {}".format(
                    manifest["dataset"], dataset, path
                )
            )
        names = {entry["sequence"] for entry in manifest["sequences"]}
        excluded.update(names)
        records.append(
            {
                "name": path.name,
                "fingerprint": manifest["fingerprint"],
                "num_sequences": len(names),
            }
        )
    return excluded, sorted(records, key=lambda row: (row["name"], row["fingerprint"]))


def build_holdout_manifest(
    dataset,
    root,
    names,
    reserve_count,
    holdout_count,
    clip_frames,
    seed,
    exclude_paths,
):
    excluded, records = exclusion_records(exclude_paths, dataset)
    available = [name for name in names if name not in excluded]
    data = build_manifest(
        dataset,
        root,
        available,
        reserve_count,
        holdout_count,
        clip_frames,
        seed,
    )
    selected = {entry["sequence"] for entry in data["sequences"]}
    overlap = selected.intersection(excluded)
    if overlap:
        raise RuntimeError(
            "Holdout manifest overlaps excluded videos: {}".format(
                ", ".join(sorted(overlap))
            )
        )
    data["purpose"] = "stage7_frozen_temporal_holdout"
    data["excluded_manifests"] = records
    data["excluded_sequence_count"] = len(excluded)
    data["fingerprint"] = fingerprint(
        {key: value for key, value in data.items() if key != "fingerprint"}
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=("got10k_train", "lasot_train")
    )
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--sequence_file",
        default=None,
        help="Optional installed-sequence list for a partial training root.",
    )
    parser.add_argument(
        "--exclude_manifest",
        nargs="+",
        required=True,
        help="One or more prior manifests whose videos must not be selected.",
    )
    parser.add_argument(
        "--reserve_sequences",
        type=int,
        default=1,
        help="Compatibility train split; Stage-7 evaluation uses only val clips.",
    )
    parser.add_argument("--holdout_sequences", type=int, default=50)
    parser.add_argument("--clip_frames", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            "Manifest exists; use a new output path or --overwrite"
        )
    root = Path(args.root).expanduser().resolve()
    names = (
        subset_names(args.dataset, args.sequence_file)
        if args.sequence_file
        else training_names(args.dataset, root)
    )
    data = build_holdout_manifest(
        args.dataset,
        root,
        names,
        args.reserve_sequences,
        args.holdout_sequences,
        args.clip_frames,
        args.seed,
        args.exclude_manifest,
    )
    write_json(output, data)
    print(
        "Manifest: {} ({} reserve / {} holdout videos)".format(
            output, args.reserve_sequences, args.holdout_sequences
        )
    )
    print(
        "Excluded: {} videos from {} manifest(s); selected overlap=0".format(
            data["excluded_sequence_count"], len(data["excluded_manifests"])
        )
    )
    print("Fingerprint: {}".format(data["fingerprint"]))


if __name__ == "__main__":
    main()
