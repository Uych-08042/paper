"""Dependency-light helpers for frozen Stage-7 temporal holdout runs."""

import numpy as np

from stage4_common import file_hash, fingerprint, frame_path, load_annotations
from stage6_temporal_layout import (
    LAYOUT_POLICIES,
    active_policy,
    default_early_fraction,
    expected_candidate_blocks,
    split_frames,
    split_frames_for_layout,
)


BASELINE_POLICIES = ("fast", "sgla")
# Freeze the action set selected before seeing Stage-7 outcomes. Alternative
# layouts remain Stage-6 ablations and are deliberately unavailable here.
HOLDOUT_POLICIES = ("fast", "uniform_p5", "late_p10_exact", "sgla")


class ManifestSequence:
    """Minimal Sequence-compatible view over one fixed manifest clip."""

    def __init__(self, manifest, entry):
        directory, ground_truth, valid = load_annotations(
            manifest["root"], manifest["dataset"], entry["sequence"]
        )
        if (
            file_hash(directory / "groundtruth.txt")
            != entry["annotation_sha256"]
            or fingerprint(valid.tolist()) != entry["visible_sha256"]
        ):
            raise ValueError(
                "Annotations changed since manifest creation: {}".format(
                    entry["sequence"]
                )
            )

        start, stop = entry["start"], entry["stop"]
        if stop > len(ground_truth) or not valid[start]:
            raise ValueError(
                "Invalid clip bounds or initialization: {}".format(
                    entry["sequence"]
                )
            )
        frames = [
            frame_path(directory, manifest["dataset"], frame_id)
            for frame_id in range(start, stop)
        ]
        missing = [path for path in frames if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing image: {}".format(missing[0]))

        self.dataset = manifest["dataset"]
        self.name = entry["sequence"]
        self.frames = [str(path) for path in frames]
        self.ground_truth_rect = np.asarray(
            ground_truth[start:stop], dtype=np.float64
        ).copy()
        self.target_visible = np.asarray(valid[start:stop], dtype=bool).copy()
        self.manifest_start = start
        self.manifest_stop = stop
        self.manifest_split = entry["split"]

    def init_info(self):
        return {"init_bbox": self.ground_truth_rect[0].tolist()}


def assignment_for_policy(sequence, policy, early_fraction=None):
    """Build the assignment contract consumed by the Stage-6 runner."""
    if policy not in HOLDOUT_POLICIES:
        raise ValueError("Unknown holdout policy: {}".format(policy))
    tracked_frames = len(sequence.frames) - 1
    if policy in LAYOUT_POLICIES:
        fraction = (
            default_early_fraction(policy)
            if early_fraction is None
            else early_fraction
        )
        early_frames, late_frames = split_frames_for_layout(
            tracked_frames, fraction, policy
        )
        blocks = expected_candidate_blocks(
            tracked_frames, early_frames, policy
        )
    else:
        if early_fraction is not None:
            raise ValueError(
                "--early_fraction applies only to temporal layout policies"
            )
        early_frames, late_frames = split_frames(tracked_frames, 0.5)
        blocks = 0 if policy == "fast" else tracked_frames

    return {
        "dataset": sequence.dataset,
        "sequence": sequence.name,
        "early_frames": early_frames,
        "late_frames": late_frames,
        "router_policy": policy,
        "router_candidate_blocks": blocks,
        "fold": -1,
    }


def resolve_policy(frame_id, early_frames, policy):
    """Resolve fixed baselines or one of the frozen temporal layouts."""
    if policy in BASELINE_POLICIES:
        return policy
    return active_policy(frame_id, early_frames, policy)


def select_entries(manifest, split="val", sequence=None):
    entries = [
        entry
        for entry in manifest["sequences"]
        if (split == "all" or entry["split"] == split)
        and (sequence is None or entry["sequence"] == sequence)
    ]
    if not entries:
        raise ValueError("No manifest clips match --split/--sequence")
    return entries
