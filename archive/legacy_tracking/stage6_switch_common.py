"""Pure helpers shared by Stage-6 switched closed-loop validation."""

import csv
import math
from pathlib import Path


ALLOWED_POLICIES = {
    "fast",
    "periodic_l9_05",
    "periodic_l9_10",
    "periodic_l9_20",
    "periodic_l9_30",
    "periodic_l9_50",
    "sgla",
}
REQUIRED_ASSIGNMENT_FIELDS = {
    "early_fraction",
    "budget_percent",
    "dataset",
    "sequence",
    "early_frames",
    "late_frames",
    "router_policy",
    "router_candidate_blocks",
    "fold",
}


def select_assignments(rows, early_fraction, budget_percent):
    selected = {}
    for row in rows:
        if not math.isclose(
            float(row["early_fraction"]),
            float(early_fraction),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            continue
        if not math.isclose(
            float(row["budget_percent"]),
            float(budget_percent),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            continue
        policy = row["router_policy"]
        if policy not in ALLOWED_POLICIES:
            raise ValueError("Unsupported assigned policy: {}".format(policy))
        key = (row["dataset"], row["sequence"])
        if key in selected:
            raise ValueError("Duplicate assignment for {}/{}".format(*key))
        early_frames = int(row["early_frames"])
        late_frames = int(row["late_frames"])
        candidate_blocks = int(row["router_candidate_blocks"])
        if early_frames < 1 or late_frames < 1 or candidate_blocks < 0:
            raise ValueError("Invalid assignment counts for {}/{}".format(*key))
        selected[key] = {
            "dataset": key[0],
            "sequence": key[1],
            "early_frames": early_frames,
            "late_frames": late_frames,
            "router_policy": policy,
            "router_candidate_blocks": candidate_blocks,
            "fold": int(row["fold"]),
        }
    if not selected:
        raise ValueError(
            "No assignments match early_fraction={} and budget_percent={}".format(
                early_fraction, budget_percent
            )
        )
    return selected


def load_assignments(path, early_fraction, budget_percent):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Assignment CSV missing: {}".format(path))
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_ASSIGNMENT_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Assignment CSV is missing fields: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        rows = list(reader)
    return path, select_assignments(rows, early_fraction, budget_percent)


def active_policy(frame_id, early_frames, assigned_policy):
    if frame_id < 1:
        return "fast"
    return "fast" if frame_id <= early_frames else assigned_policy
