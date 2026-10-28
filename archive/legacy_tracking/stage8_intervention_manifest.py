"""Create a fresh development manifest for Stage-8B interventions."""

import argparse
from pathlib import Path

from stage4_common import fingerprint, load_annotations, write_json
from stage4_manifest import build_manifest, subset_names, training_names
from stage7_holdout_manifest import exclusion_records
from stage8_intervention_common import DEFAULT_HORIZONS


def build_intervention_manifest(
    dataset,
    root,
    names,
    train_count,
    val_count,
    clip_frames,
    minimum_clip_frames,
    seed,
    exclude_paths,
    horizons,
    anchor_count,
):
    horizons = tuple(sorted(set(int(value) for value in horizons)))
    if not horizons or horizons[0] < 1:
        raise ValueError("Horizons must be positive")
    if minimum_clip_frames < max(horizons) + anchor_count:
        raise ValueError(
            "minimum_clip_frames must be at least max(horizons)+anchor_count"
        )
    excluded, exclusion_details = exclusion_records(exclude_paths, dataset)
    root = Path(root).expanduser().resolve()
    available = sorted(set(names).difference(excluded))
    eligible = []
    missing = 0
    short = 0
    for name in available:
        try:
            _, ground_truth, _ = load_annotations(root, dataset, name)
        except FileNotFoundError:
            missing += 1
            continue
        if min(clip_frames, len(ground_truth)) < minimum_clip_frames:
            short += 1
            continue
        eligible.append(name)
    data = build_manifest(
        dataset,
        root,
        eligible,
        train_count,
        val_count,
        clip_frames,
        seed,
    )
    selected = {entry["sequence"] for entry in data["sequences"]}
    if selected.intersection(excluded):
        raise RuntimeError("Stage-8B manifest overlaps an excluded manifest")
    data.update(
        {
            "purpose": "stage8_l9_causal_intervention_development",
            "excluded_manifests": exclusion_details,
            "excluded_sequence_count": len(excluded),
            "available_sequence_count": len(available),
            "eligible_sequence_count": len(eligible),
            "missing_sequence_count": missing,
            "short_sequence_count": short,
            "minimum_clip_frames": int(minimum_clip_frames),
            "intervention_design": {
                "anchor_count": int(anchor_count),
                "horizons": list(horizons),
                "control": "fast_then_fast",
                "treatment": "single_fixed_l9_then_fast",
            },
        }
    )
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
    parser.add_argument("--sequence_file", default=None)
    parser.add_argument("--exclude_manifest", nargs="+", required=True)
    parser.add_argument("--train_sequences", type=int, default=120)
    parser.add_argument("--val_sequences", type=int, default=40)
    parser.add_argument("--clip_frames", type=int, default=256)
    parser.add_argument("--minimum_clip_frames", type=int, default=64)
    parser.add_argument("--anchors", type=int, default=12)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
    )
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError("Manifest exists; use a new output or --overwrite")
    root = Path(args.root).expanduser().resolve()
    names = (
        subset_names(args.dataset, args.sequence_file)
        if args.sequence_file
        else training_names(args.dataset, root)
    )
    data = build_intervention_manifest(
        args.dataset,
        root,
        names,
        args.train_sequences,
        args.val_sequences,
        args.clip_frames,
        args.minimum_clip_frames,
        args.seed,
        args.exclude_manifest,
        args.horizons,
        args.anchors,
    )
    write_json(output, data)
    print(
        "Manifest: {} ({} train / {} val; {} anchors; horizons {})".format(
            output,
            args.train_sequences,
            args.val_sequences,
            args.anchors,
            "/".join(str(value) for value in sorted(set(args.horizons))),
        )
    )
    print(
        "Excluded {} used videos; eligible {} of {} remaining; selected overlap=0".format(
            data["excluded_sequence_count"],
            data["eligible_sequence_count"],
            data["available_sequence_count"],
        )
    )
    print(
        "Unselected eligible reserve: {} videos".format(
            data["eligible_sequence_count"]
            - args.train_sequences
            - args.val_sequences
        )
    )
    print("Fingerprint: {}".format(data["fingerprint"]))


if __name__ == "__main__":
    main()
