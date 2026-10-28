"""Create fixed video-level train/val clips from installed training datasets."""

import argparse
import random
from pathlib import Path

import numpy as np

from stage4_common import SCHEMA, file_hash, fingerprint, frame_path, load_annotations, write_json


def training_names(dataset, root):
    specs = Path(__file__).resolve().parents[1] / "lib/train/data_specs"
    if dataset == "lasot_train":
        return (specs / "lasot_train_split.txt").read_text().splitlines()
    names = (root / "list.txt").read_text().splitlines()
    # Use the repository training subset, never the official val/test folders.
    indices = np.loadtxt(str(specs / "got10k_train_split.txt"), dtype=int, ndmin=1)
    if indices.max() >= len(names) or not all(name.startswith("GOT-10k_Train_") for name in names):
        raise ValueError("--root must be the original GOT-10k train directory with its full list.txt")
    return [names[index] for index in indices]


def subset_names(dataset, path):
    """Load an explicit installed subset without weakening split isolation."""
    specs = Path(__file__).resolve().parents[1] / "lib/train/data_specs"
    names = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    names = [name.strip() for name in names if name.strip()]
    if not names:
        raise ValueError("--sequence_file is empty")
    if dataset == "lasot_train":
        allowed = set((specs / "lasot_train_split.txt").read_text().splitlines())
    else:
        indices = np.loadtxt(str(specs / "got10k_train_split.txt"), dtype=int, ndmin=1)
        allowed = {"GOT-10k_Train_{:06d}".format(int(index) + 1) for index in indices}
    rejected = sorted(set(names).difference(allowed))
    if rejected:
        raise ValueError(
            "Explicit subset contains sequence(s) outside the repository training split: {}".format(
                ", ".join(rejected[:3])
            )
        )
    return names


def build_manifest(dataset, root, names, train_count, val_count, clip_frames, seed):
    root = Path(root).expanduser().resolve()
    if train_count < 1 or val_count < 1 or clip_frames < 2:
        raise ValueError("Need positive train/val counts and at least two clip frames")
    rng = random.Random(seed)
    names = sorted(set(name.strip() for name in names if name.strip()))
    rng.shuffle(names)
    entries = []
    missing = 0
    for name in names:
        if "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("Unsafe dataset entry")
        try:
            directory, gt, valid = load_annotations(root, dataset, name)
        except FileNotFoundError:
            missing += 1
            continue
        length = min(clip_frames, len(gt))
        starts = np.flatnonzero(valid[:len(gt) - length + 1])
        if not len(starts):
            continue
        start = int(starts[rng.randrange(len(starts))])
        stop = start + length
        absent = [frame_path(directory, dataset, index) for index in range(start, stop)
                  if not frame_path(directory, dataset, index).is_file()]
        if absent:
            raise FileNotFoundError("Missing image: {}".format(absent[0]))
        entries.append({"sequence": name, "split": "train" if len(entries) < train_count else "val",
                        "start": start, "stop": stop, "annotation_sha256": None})
        entries[-1]["annotation_sha256"] = file_hash(directory / "groundtruth.txt")
        entries[-1]["visible_sha256"] = fingerprint(valid.tolist())
        if len(entries) == train_count + val_count:
            break
    if len(entries) != train_count + val_count:
        raise ValueError("Only {} usable installed training videos; need {} ({} missing annotations)".format(
            len(entries), train_count + val_count, missing))
    data = {"schema": SCHEMA, "dataset": dataset, "root": str(root), "seed": seed,
            "clip_frames": clip_frames, "sequences": entries}
    data["fingerprint"] = fingerprint(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("got10k_train", "lasot_train"))
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--sequence_file",
        default=None,
        help=("Optional text file containing one installed training sequence per line. "
              "This permits a partial dataset root and is checked against the repository train split."),
    )
    parser.add_argument("--train_sequences", type=int, default=5)
    parser.add_argument("--val_sequences", type=int, default=2)
    parser.add_argument("--clip_frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError("Manifest exists; use a new output path or --overwrite")
    root = Path(args.root).expanduser().resolve()
    names = (subset_names(args.dataset, args.sequence_file) if args.sequence_file
             else training_names(args.dataset, root))
    data = build_manifest(args.dataset, root, names,
                          args.train_sequences, args.val_sequences, args.clip_frames, args.seed)
    write_json(args.output, data)
    print("Manifest: {} ({} train / {} val videos; no video overlap)".format(
        args.output, args.train_sequences, args.val_sequences))
    print("Fingerprint: {}".format(data["fingerprint"]))


if __name__ == "__main__":
    main()
