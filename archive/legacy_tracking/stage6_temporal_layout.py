"""Pure helpers for equal-cost temporal placement ablations."""

import math

from stage3_periodic import PERIODIC_L9_BUDGETS, periodic_l9_active


LAYOUT_POLICIES = {
    "early_p10": "periodic_l9_10",
    "late_p10": "periodic_l9_10",
    "late_p10_exact": "periodic_l9_10",
    "late_p20": "periodic_l9_20",
    "late_p20_exact": "periodic_l9_20",
    "late_p30": "periodic_l9_30",
    "late_p30_exact": "periodic_l9_30",
    "late_p50": "periodic_l9_50",
    "late_p50_exact": "periodic_l9_50",
    "uniform_p5": "periodic_l9_05",
}

DEFAULT_EARLY_FRACTIONS = {
    "early_p10": 0.5,
    "late_p10": 0.5,
    "late_p10_exact": 0.5,
    "late_p20": 0.75,
    "late_p20_exact": 0.75,
    "late_p30": 5.0 / 6.0,
    "late_p30_exact": 5.0 / 6.0,
    "late_p50": 0.9,
    "late_p50_exact": 0.9,
    "uniform_p5": 0.5,
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
    if layout.startswith("early_"):
        return LAYOUT_POLICIES[layout] if frame_id <= early_frames else "fast"
    if layout.startswith("late_"):
        return "fast" if frame_id <= early_frames else LAYOUT_POLICIES[layout]
    return LAYOUT_POLICIES[layout]


def default_early_fraction(layout):
    if layout not in DEFAULT_EARLY_FRACTIONS:
        raise ValueError("Unknown temporal layout: {}".format(layout))
    return DEFAULT_EARLY_FRACTIONS[layout]


def split_frames_for_layout(tracked_frames, early_fraction, layout):
    nominal_early, _ = split_frames(tracked_frames, early_fraction)
    if not layout.endswith("_exact"):
        return nominal_early, tracked_frames - nominal_early

    policy = LAYOUT_POLICIES[layout]
    target_blocks = sum(
        int(periodic_l9_active("periodic_l9_05", frame_id))
        for frame_id in range(1, tracked_frames + 1)
    )
    total_policy_blocks = sum(
        int(periodic_l9_active(policy, frame_id))
        for frame_id in range(1, tracked_frames + 1)
    )
    required_prefix_blocks = total_policy_blocks - target_blocks
    prefix_blocks = 0
    candidates = []
    for frame_id in range(1, tracked_frames):
        prefix_blocks += int(periodic_l9_active(policy, frame_id))
        if prefix_blocks == required_prefix_blocks:
            candidates.append(frame_id)
    if not candidates:
        raise ValueError(
            "Cannot exactly match the P5 budget for {} over {} frames".format(
                layout, tracked_frames
            )
        )
    early_frames = min(
        candidates,
        key=lambda value: (abs(value - nominal_early), value),
    )
    return early_frames, tracked_frames - early_frames


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
