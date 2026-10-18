"""Create a fresh video-disjoint manifest for Stage-9 schedule interventions."""

import argparse
from pathlib import Path

from stage4_common import fingerprint, load_annotations, write_json
from stage4_manifest import build_manifest, subset_names, training_names
from stage7_holdout_manifest import exclusion_records
from stage9_schedule_common import (
    DEFAULT_EVALUATION_HORIZONS,
    DEFAULT_HORIZON,
    POLICIES,
    PRIMARY_COMPARISON,
    PRIMARY_HORIZON,
    SCHEDULES,
)


def nonempty_path(value):
    if not str(value).strip():
        raise argparse.ArgumentTypeError(
            "path must not be empty; check that the referenced shell variable is exported"
        )
    return value


def build_schedule_manifest(
    dataset,
    root,
    names,
    pilot_count,
    confirm_count,
    clip_frames,
    minimum_clip_frames,
    seed,
    exclude_paths,
    anchor_count,
):
    if pilot_count < 1 or confirm_count < 1:
        raise ValueError("Need positive pilot and confirmatory sequence counts")
    required = DEFAULT_HORIZON + anchor_count
    if minimum_clip_frames < required:
        raise ValueError(
            "minimum_clip_frames must be at least horizon+anchors ({})".format(
                required
            )
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
    requested = pilot_count + confirm_count
    if len(eligible) < requested:
        if len(eligible) < 2:
            split_advice = (
                "At least two fresh videos are required for positive pilot and "
                "confirmatory splits."
            )
        else:
            suggested_confirm = min(confirm_count, len(eligible) - 1)
            suggested_pilot = len(eligible) - suggested_confirm
            split_advice = (
                "If already using {}, the largest valid split is "
                "--pilot_sequences {} --confirm_sequences {}."
            ).format(required, suggested_pilot, suggested_confirm)
        raise ValueError(
            "Only {} fresh videos satisfy the {}-frame minimum; requested {}. "
            "The frozen H{}/{}-anchor design permits a minimum of {} frames. "
            "{}".format(
                len(eligible),
                minimum_clip_frames,
                requested,
                DEFAULT_HORIZON,
                anchor_count,
                required,
                split_advice,
            )
        )
    data = build_manifest(
        dataset,
        root,
        eligible,
        pilot_count,
        confirm_count,
        clip_frames,
        seed,
    )
    selected = {entry["sequence"] for entry in data["sequences"]}
    if selected.intersection(excluded):
        raise RuntimeError("Stage-9 manifest overlaps an excluded manifest")
    data.update(
        {
            "purpose": "stage9_equal_cost_schedule_intervention",
            "split_roles": {
                "train": "pilot",
                "val": "confirmatory",
            },
            "excluded_manifests": exclusion_details,
            "excluded_sequence_count": len(excluded),
            "available_sequence_count": len(available),
            "eligible_sequence_count": len(eligible),
            "missing_sequence_count": missing,
            "short_sequence_count": short,
            "minimum_clip_frames": int(minimum_clip_frames),
            "schedule_design": {
                "anchor_count": int(anchor_count),
                "horizon": DEFAULT_HORIZON,
                "evaluation_horizons": list(DEFAULT_EVALUATION_HORIZONS),
                "policies": list(POLICIES),
                "schedules": {
                    policy: list(SCHEDULES[policy]) for policy in POLICIES
                },
                "primary_horizon": PRIMARY_HORIZON,
                "primary_comparison": list(PRIMARY_COMPARISON),
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
    parser.add_argument("--root", required=True, type=nonempty_path)
    parser.add_argument("--sequence_file", default=None, type=nonempty_path)
    parser.add_argument(
        "--exclude_manifest", nargs="+", required=True, type=nonempty_path
    )
    parser.add_argument("--pilot_sequences", type=int, default=10)
    parser.add_argument("--confirm_sequences", type=int, default=40)
    parser.add_argument("--clip_frames", type=int, default=128)
    parser.add_argument("--minimum_clip_frames", type=int, default=52)
    parser.add_argument("--anchors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2034)
    parser.add_argument("--output", required=True, type=nonempty_path)
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
    data = build_schedule_manifest(
        args.dataset,
        root,
        names,
        args.pilot_sequences,
        args.confirm_sequences,
        args.clip_frames,
        args.minimum_clip_frames,
        args.seed,
        args.exclude_manifest,
        args.anchors,
    )
    write_json(output, data)
    print(
        "Manifest: {} ({} pilot / {} confirmatory videos; overlap=0)".format(
            output, args.pilot_sequences, args.confirm_sequences
        )
    )
    print(
        "Excluded {} used videos; eligible {} of {} remaining".format(
            data["excluded_sequence_count"],
            data["eligible_sequence_count"],
            data["available_sequence_count"],
        )
    )
    print(
        "Design: {} anchors, H{}, equal-cost schedules {}".format(
            args.anchors,
            DEFAULT_HORIZON,
            " | ".join(
                "{}={}".format(
                    policy,
                    ",".join(str(value) for value in SCHEDULES[policy]) or "none",
                )
                for policy in POLICIES
            ),
        )
    )
    print("Fingerprint: {}".format(data["fingerprint"]))


if __name__ == "__main__":
    main()
