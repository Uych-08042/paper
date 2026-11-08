"""Freeze a VisDrone-SOT confirmation roster from names and frame inventory only."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracking"))
from stage4_common import file_hash, fingerprint
from stage27_uav_common import IMAGE_SUFFIXES, frame_number, publish_identical, seal, source_group
from stage66_takeover_common import sealed


SCHEMA = "stage79_visdrone_confirmation_v1"
# Also exclude sequence IDs mentioned in the versioned historical notes and reports.
DOCUMENTED_GROUPS = (
    "uav0000001 uav0000002 uav0000003 uav0000016 uav0000085 uav0000089 uav0000090 "
    "uav0000126 uav0000147 uav0000170 uav0000173 uav0000176 uav0000204 uav0000226 "
    "uav0000235 uav0000237 uav0000304 uav0000307 uav0000329 uav0000342"
).split()


def inventory(root):
    sequences, annotations = root / "sequences", root / "annotations"
    if not sequences.is_dir() or not annotations.is_dir():
        raise ValueError(f"Need the VisDrone-SOT train inventory: {root}")
    rows = []
    for folder in sorted(p for p in sequences.iterdir() if p.is_dir()):
        images = sorted(frame_number(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if (len(images) < 2 or any(b != a + 1 for a, b in zip(images, images[1:]))
                or not (annotations / (folder.name + ".txt")).is_file()):
            raise ValueError(f"Incomplete sequence inventory: {folder.name}")
        rows.append(dict(sequence=folder.name, source_group_proxy=source_group(folder.name),
                         frames=len(images), first_image=images[0], last_image=images[-1]))
    if len(rows) != 86 or sum(row["frames"] for row in rows) != 69941:
        raise ValueError("Need the complete 86-sequence / 69,941-frame train inventory")
    return rows


def roster(rows, excluded):
    groups = {}
    for entry in sorted(rows, key=lambda r: r["sequence"]):
        group = entry["source_group_proxy"]
        if group not in excluded:
            groups.setdefault(group, entry)
    if len(groups) < 12:
        raise ValueError(f"Only {len(groups)} eligible source groups; confirmation roster needs at least 12")
    return list(groups.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--stage27", type=Path, required=True)
    p.add_argument("--stage31", type=Path, required=True)
    p.add_argument("--stage76", type=Path, required=True)
    p.add_argument("--exclude-file", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.dataset_root.expanduser().resolve()
    stage76 = args.stage76.resolve()
    m76 = sealed(stage76, "stage76_four_candidate_horizon_supervision_v1")
    m27 = sealed(args.stage27, "stage27_uav_failure_diagnostic_v1")
    m31 = sealed(args.stage31, "stage31_continuous_identity_pilot_v1")
    if len(m27["sequences"]) != 24 or len(m31["sequences"]) != 14 or Path(m27["root"]).resolve() != root or Path(m31["root"]).resolve() != root:
        raise ValueError("Historical manifests must match the 24/14 exclusions on the same dataset")
    paths = [args.stage27, args.stage31, stage76, *args.exclude_file]
    reasons = dict(stage27=sorted({source_group(e["sequence"]) for e in m27["sequences"]}),
                   stage31=sorted({source_group(e["sequence"]) for e in m31["sequences"]}),
                   stage76=sorted({source_group(e["sequence"]) for e in m76["units"]}),
                   documented=DOCUMENTED_GROUPS)
    extra = {line.strip() for path in args.exclude_file for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if any(not re.fullmatch(r"uav[0-9]+", group) for group in extra):
        raise ValueError("Each additional exclusion line must be a source group such as uav0000204")
    reasons["additional"] = sorted(extra)
    rows = inventory(root)
    available = {r["source_group_proxy"] for r in rows}
    if len(reasons["stage76"]) != 12 or not set(reasons["stage27"] + reasons["stage31"] + reasons["stage76"]) <= available:
        raise ValueError("Historical group bindings do not match the train inventory")
    selected = roster(rows, {group for values in reasons.values() for group in values})
    plan = seal(dict(schema=SCHEMA, role="within_visdrone_confirmation_roster", root=str(root),
        inventory_fingerprint=fingerprint(rows), exclusion_sources={str(path.resolve()): file_hash(path) for path in paths},
        excluded_groups_by_source=reasons, sequences=selected,
        selection="all eligible source-group proxies, lexicographically first sequence per group",
        metrics=["video_macro_success_auc", "low_iou_fraction", "longest_failure_fraction", "new_failures_on_native_stable"],
        initial_box_only_at_inference=True, labels_read=False, images_decoded=False, pretraining_overlap_verified=False))
    publish_identical(args.output, plan)
    print(f"Confirmation roster: {len(selected)} source groups. Fingerprint: {plan['fingerprint']}")


if __name__ == "__main__":
    main()
