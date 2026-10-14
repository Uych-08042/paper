"""Pure helpers for equal-cost temporal placement ablations."""

import math

from stage3_periodic import PERIODIC_L9_BUDGETS, periodic_l9_active


LAYOUT_POLICIES = {
    "early_p10": "periodic_l9_10",
    "late_p10": "periodic_l9_10",
    "uniform_p5": "periodic_l9_05",
}


def split_frames(tracked_frames, early_fraction):
    if tracked_frames < 2:
        raise ValueError("Temporal layouts require at least two tracked frames")
    if not 0.0 < early_fraction < 1.0:
        raise ValueError("early_fraction must be strictly between 0 and 1")
    early_frames = int(math.floor(tracked_frames * early_fraction + 1e-12))
    early_frames = min(max(early_frames, 1), tracked_frames - 1)
    return early_frames, tracked_frames - early_frames


def active_policy(frame_id, early_frames, layout):
    if layout not in LAYOUT_POLICIES:
        raise ValueError("Unknown temporal layout: {}".format(layout))
    if frame_id < 1:
        return "fast"
    if layout == "early_p10":
        return LAYOUT_POLICIES[layout] if frame_id <= early_frames else "fast"
    if layout == "late_p10":
        return "fast" if frame_id <= early_frames else LAYOUT_POLICIES[layout]
    return LAYOUT_POLICIES[layout]


def candidate_blocks(frame_id, early_frames, layout):
    policy = active_policy(frame_id, early_frames, layout)
    if policy == "fast":
        return 0
    if policy not in PERIODIC_L9_BUDGETS:
        raise ValueError("No temporal-layout cost rule for {}".format(policy))
    return int(periodic_l9_active(policy, frame_id))


def expected_candidate_blocks(tracked_frames, early_frames, layout):
    return sum(
        candidate_blocks(frame_id, early_frames, layout)
        for frame_id in range(1, tracked_frames + 1)
    )
