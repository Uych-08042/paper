"""Freeze 24 source-group-distinct UAV clips after CPU image decoding."""

import argparse
from pathlib import Path

from stage4_common import file_hash, fingerprint, write_csv
from stage27_uav_common import (DESIGN, SCHEMA, inside, nonempty_path,
    publish_identical, read_boxes, read_rgb, scan_dataset, seal, select_entries)


def build_manifest(root, sequences=24, seed=2091, reader=read_rgb,
                   expected_sequences=86, expected_frames=69941):
    root, inventory = scan_dataset(root, expected_sequences, expected_frames)
    selected = select_entries(inventory, sequences, seed)
    audited = []
    for index, entry in enumerate(selected):
        frames = []
        for frame in entry["frames"]:
            path = inside(root, frame["file"])
            before = file_hash(path)
            image = reader(path)
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
                raise ValueError("Invalid decoded image: " + str(path))
            if file_hash(path) != before:
                raise ValueError("Image changed during audit: " + str(path))
            frames.append(dict(frame, sha256=before, height=int(image.shape[0]), width=int(image.shape[1])))
        boxes = read_boxes(inside(root, entry["annotation"]))
        if file_hash(inside(root, entry["annotation"])) != entry["annotation_sha256"]:
            raise ValueError("Annotation changed during audit")
        audited.append(dict(entry, frames=frames, initial_box=boxes[0].tolist(),
                            ground_truth=boxes.tolist()))
        print("[image audit] {}/{} {}: {} decoded frames".format(
            index + 1, len(selected), entry["sequence"], len(frames)), flush=True)
    return seal(dict(schema=SCHEMA, role="descriptive_development_dataset_manifest",
        dataset="visdrone2019_sot_train", root=str(root), seed=seed,
        selected_sequences=sequences, dataset_sequences=len(inventory),
        dataset_frames=sum(len(e["frames"]) for e in inventory),
        inventory_fingerprint=fingerprint(inventory), design=DESIGN,
        source_group_rule="sequence prefix uavNNNNNNN; conservative proxy, not verified physical-video identity",
        selection_rule="uniform source-group sampling then one clip/group; no outcome-based selection",
        initialization="first frame GT once; full remaining video; no GT resets",
        visibility="unknown: positive xywh is geometric validity, not evidence of visibility",
        image_audit_scope="all frames of selected videos decoded and SHA256 hashed; other videos structure-only",
        interpretation="development diagnosis only; pretraining overlap not ruled out; no generalization claim",
        code_sha256={name: file_hash(Path(__file__).parent / name)
                     for name in ("stage27_uav_common.py", "stage27_uav_manifest.py", "validation_metrics.py")},
        sequences=audited))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=nonempty_path, required=True)
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--sequences", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2091)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset_root, args.sequences, args.seed)
    output = Path(args.output_dir).expanduser().resolve()
    created = publish_identical(output / "manifest.json", manifest)
    rows = [dict(sequence=e["sequence"], source_group_proxy=e["source_group_proxy"],
                 frames=len(e["frames"]), tracked_frames=len(e["frames"]) - 1,
                 annotation_sha256=e["annotation_sha256"]) for e in manifest["sequences"]]
    write_csv(output / "stage27_selected_sequences.csv", rows)
    print("Manifest: {} ({})".format(output / "manifest.json", "created" if created else "identical, reused"))
    print("Selected {} full videos / {} tracked frames; selected images decoded.".format(
        len(rows), sum(r["tracked_frames"] for r in rows)))
    print("Fingerprint: " + manifest["fingerprint"])
    print("Dataset preparation only. Next: original SGLA collection; no training or predictions yet.")


if __name__ == "__main__":
    main()
