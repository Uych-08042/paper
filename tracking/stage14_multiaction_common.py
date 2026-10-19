"""Dependency-light helpers for Stage-14 multi-action value audits."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from stage4_common import IOU_PROTOCOL, file_hash, fingerprint, load_manifest, tracking_iou
from stage8_intervention_common import select_anchor_offsets


SCHEMA = "stage14_multiaction_counterfactual_v1"
ACTION_NAMES = ("fast", "l7", "l8", "l9", "l10", "l11", "l12")
CANDIDATE_LAYERS = tuple(range(7, 13))


def discounted_policy_values(
    policy_iou,
    valid_gt,
    horizon=20,
    gamma=0.9,
    min_valid_fraction=0.8,
):
    """Compute normalized discounted returns for every one-step action branch."""
    policy_iou = np.asarray(policy_iou, dtype=np.float64)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    horizon = int(horizon)
    gamma = float(gamma)
    min_valid_fraction = float(min_valid_fraction)
    if policy_iou.ndim != 3 or policy_iou.shape[1] != len(ACTION_NAMES):
        raise ValueError("policy_iou must have shape [anchor,7,time]")
    if valid_gt.shape != (policy_iou.shape[0], policy_iou.shape[2]):
        raise ValueError("valid_gt must match policy_iou anchor/time dimensions")
    if horizon < 2 or horizon > policy_iou.shape[2]:
        raise ValueError("Long-horizon value must fit the collected window")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0,1]")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie in (0,1]")

    valid = valid_gt[:, :horizon]
    values = policy_iou[:, :, :horizon]
    if not np.isfinite(values[:, :, :][np.broadcast_to(valid[:, None, :], values.shape)]).all():
        raise ValueError("Valid policy IoUs contain non-finite values")
    coverage = valid.mean(axis=1)
    eligible = valid[:, 0] & (coverage + 1e-12 >= min_valid_fraction)
    discounts = gamma ** np.arange(horizon, dtype=np.float64)
    denominator = np.sum(valid * discounts[None, :], axis=1)
    numerator = np.sum(
        np.where(valid[:, None, :], values, 0.0) * discounts[None, None, :],
        axis=2,
    )
    returns = np.full((len(values), len(ACTION_NAMES)), np.nan, dtype=np.float64)
    returns[eligible] = numerator[eligible] / denominator[eligible, None]
    return (
        returns,
        returns[:, 1:] - returns[:, :1],
        coverage.astype(np.float64),
        eligible,
    )


def multiaction_cache_path(output_dir, entry):
    return (
        Path(output_dir)
        / entry["split"]
        / "{}__stage14_multiaction.npz".format(entry["sequence"])
    )


def make_run(
    manifest,
    runtime_identity,
    pool_grid,
    anchor_count,
    horizon,
    gamma,
    min_valid_fraction,
    verify_actions,
):
    horizon = int(horizon)
    gamma = float(gamma)
    min_valid_fraction = float(min_valid_fraction)
    if anchor_count < 1 or horizon < 2:
        raise ValueError("anchor_count must be positive and horizon must exceed one")
    if not 0.0 < gamma <= 1.0 or not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("Invalid discount or valid-fraction requirement")
    directory = Path(__file__).resolve().parent
    files = (
        "stage14_multiaction_common.py",
        "stage14_multiaction_collect.py",
        "stage4_common.py",
        "stage4_runtime.py",
        "stage8_intervention_common.py",
        "validation_metrics.py",
        "stage1_validate.py",
    )
    run = {
        "schema": SCHEMA,
        "analysis_role": "development_only_multiaction_counterfactual_audit",
        "manifest": manifest,
        "runtime": runtime_identity,
        "pool_grid": int(pool_grid),
        "anchor_count": int(anchor_count),
        "horizon": horizon,
        "gamma": gamma,
        "min_valid_fraction": min_valid_fraction,
        "actions": list(ACTION_NAMES),
        "candidate_layers": list(CANDIDATE_LAYERS),
        "branch_semantics": "one_selected_action_then_fast_for_remaining_horizon",
        "verify_actions": bool(verify_actions),
        "iou_protocol": IOU_PROTOCOL,
        "code_sha256": {name: file_hash(directory / name) for name in files},
    }
    run["fingerprint"] = fingerprint(run)
    return run


def load_multiaction_run(directory):
    directory = Path(directory).expanduser().resolve()
    path = directory / "run.json"
    if not path.is_file():
        raise FileNotFoundError("Stage-14 run metadata missing: {}".format(path))
    run = json.loads(path.read_text(encoding="utf-8"))
    digest = run.get("fingerprint")
    body = {key: value for key, value in run.items() if key != "fingerprint"}
    if (
        run.get("schema") != SCHEMA
        or run.get("iou_protocol") != IOU_PROTOCOL
        or run.get("actions") != list(ACTION_NAMES)
        or run.get("candidate_layers") != list(CANDIDATE_LAYERS)
        or fingerprint(body) != digest
    ):
        raise ValueError("Invalid Stage-14 run metadata or fingerprint")
    load_manifest(run["manifest"])
    return directory, run


def _expected_shapes(entry, run):
    tracked = entry["stop"] - entry["start"] - 1
    anchors = run["anchor_count"]
    horizon = run["horizon"]
    dim = run["runtime"]["embed_dim"]
    side = int(run["runtime"]["lens_x"] ** 0.5)
    if side ** 2 != run["runtime"]["lens_x"]:
        raise ValueError("Expected a square search token grid")
    return {
        "frame_id": (anchors,),
        "anchor_offset": (anchors,),
        "anchor_fraction": (anchors,),
        "token_features": (anchors, (2 + run["pool_grid"] ** 2) * dim),
        "fast_score_map": (anchors, 1, side, side),
        "selector_prob": (anchors, len(CANDIDATE_LAYERS)),
        "selected_layer": (anchors,),
        "previous_state": (anchors, 4),
        "previous_fast_box": (anchors, 4),
        "fast_box": (anchors, 4),
        "candidate_boxes": (anchors, len(CANDIDATE_LAYERS), 4),
        "action_boxes": (anchors, len(ACTION_NAMES), 4),
        "image_hw": (anchors, 2),
        "resize_factor": (anchors,),
        "baseline_boxes": (tracked, 4),
        "baseline_gt": (tracked, 4),
        "baseline_valid_gt": (tracked,),
        "baseline_iou": (tracked,),
        "policy_boxes": (anchors, len(ACTION_NAMES), horizon, 4),
        "control_probe_boxes": (horizon, 4),
        "gt": (anchors, horizon, 4),
        "valid_gt": (anchors, horizon),
        "policy_iou": (anchors, len(ACTION_NAMES), horizon),
        "delta_iou": (anchors, len(CANDIDATE_LAYERS), horizon),
        "valid_fraction": (anchors,),
        "eligible": (anchors,),
        "discounted_value": (anchors, len(ACTION_NAMES)),
        "discounted_delta": (anchors, len(CANDIDATE_LAYERS)),
    }


def _require_allclose(actual, expected, message, path, atol=1e-9):
    if not np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True):
        difference = np.abs(
            np.asarray(actual, dtype=np.float64)
            - np.asarray(expected, dtype=np.float64)
        )
        finite = difference[np.isfinite(difference)]
        maximum = float(finite.max()) if finite.size else float("nan")
        raise ValueError("{} in {} (max abs diff {})".format(message, path, maximum))


def load_and_audit_cache(path, entry, run):
    """Load one cache and verify every shared-state action branch."""
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        data = {
            key: archive[key]
            for key in archive.files
            if key != "metadata_json"
        }
    if (
        metadata.get("run_fingerprint") != run["fingerprint"]
        or metadata.get("entry") != entry
    ):
        raise ValueError("Cache metadata mismatch: {}".format(path))
    shapes = _expected_shapes(entry, run)
    if set(data) != set(shapes):
        raise ValueError("Unexpected/missing Stage-14 fields: {}".format(path))
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(
                "Bad {} shape in {}: {} != {}".format(key, path, data[key].shape, shape)
            )

    nullable = {"baseline_gt", "gt", "discounted_value", "discounted_delta"}
    for key, values in data.items():
        if key not in nullable and not np.isfinite(values).all():
            raise ValueError("Non-finite {} in {}".format(key, path))
    for key in ("frame_id", "anchor_offset", "selected_layer", "image_hw"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("{} must contain integers".format(key))
    for key in ("baseline_valid_gt", "valid_gt", "eligible"):
        if data[key].dtype != np.bool_:
            raise ValueError("{} must be a boolean mask".format(key))

    tracked = entry["stop"] - entry["start"] - 1
    offsets = select_anchor_offsets(tracked, run["horizon"], run["anchor_count"])
    if not np.array_equal(data["anchor_offset"], offsets):
        raise ValueError("Anchor offsets changed in {}".format(path))
    if not np.array_equal(data["frame_id"], entry["start"] + offsets):
        raise ValueError("Absolute anchor frames changed in {}".format(path))
    _require_allclose(
        data["anchor_fraction"], offsets / tracked, "Anchor fractions changed", path, 1e-12
    )

    initial = np.asarray(metadata.get("init_box"), dtype=np.float64)
    if initial.shape != (4,) or not np.isfinite(initial).all() or np.any(initial[2:] <= 0):
        raise ValueError("Invalid Stage-14 initialization in {}".format(path))
    previous = np.stack(
        [initial if offset == 1 else data["baseline_boxes"][offset - 2] for offset in offsets]
    )
    _require_allclose(data["previous_state"], previous, "Causal states changed", path)
    _require_allclose(data["previous_fast_box"], previous, "Previous Fast boxes changed", path)
    _require_allclose(
        data["fast_box"], data["baseline_boxes"][offsets - 1],
        "Anchor Fast boxes disagree with baseline", path,
    )

    control = np.stack(
        [
            data["baseline_boxes"][offset - 1 : offset - 1 + run["horizon"]]
            for offset in offsets
        ]
    )
    gt = np.stack(
        [data["baseline_gt"][offset - 1 : offset - 1 + run["horizon"]] for offset in offsets]
    )
    valid = np.stack(
        [
            data["baseline_valid_gt"][offset - 1 : offset - 1 + run["horizon"]]
            for offset in offsets
        ]
    )
    _require_allclose(data["policy_boxes"][:, 0], control, "Fast branch is not baseline", path)
    _require_allclose(
        data["control_probe_boxes"], control[0],
        "Independent Fast replay disagrees with baseline", path, 1e-6,
    )
    _require_allclose(data["gt"], gt, "Ground-truth windows are misaligned", path)
    if not np.array_equal(data["valid_gt"], valid):
        raise ValueError("Visibility windows are misaligned in {}".format(path))
    _require_allclose(
        data["action_boxes"], data["policy_boxes"][:, :, 0],
        "Action boxes disagree with first branch step", path,
    )
    _require_allclose(
        data["action_boxes"][:, 0], data["fast_box"],
        "Fast action box changed", path,
    )
    _require_allclose(
        data["action_boxes"][:, 1:], data["candidate_boxes"],
        "Candidate action boxes changed", path,
    )

    expected_baseline_iou = tracking_iou(
        data["baseline_boxes"], data["baseline_gt"], data["baseline_valid_gt"]
    )
    expected_policy_iou = tracking_iou(
        data["policy_boxes"], data["gt"][:, None], data["valid_gt"][:, None]
    )
    for key, expected in (
        ("baseline_iou", expected_baseline_iou),
        ("policy_iou", expected_policy_iou),
        ("delta_iou", expected_policy_iou[:, 1:] - expected_policy_iou[:, :1]),
    ):
        _require_allclose(data[key], expected, "{} label mismatch".format(key), path, 1e-10)
    returns, deltas, coverage, eligible = discounted_policy_values(
        expected_policy_iou,
        data["valid_gt"],
        run["horizon"],
        run["gamma"],
        run["min_valid_fraction"],
    )
    _require_allclose(data["valid_fraction"], coverage, "Valid fractions changed", path, 1e-12)
    if not np.array_equal(data["eligible"], eligible):
        raise ValueError("Eligibility changed in {}".format(path))
    _require_allclose(data["discounted_value"], returns, "Discounted values changed", path, 1e-10)
    _require_allclose(data["discounted_delta"], deltas, "Discounted deltas changed", path, 1e-10)

    probabilities = data["selector_prob"]
    selected = data["selected_layer"]
    if (
        np.any((probabilities < 0.0) | (probabilities > 1.0))
        or np.any((selected < 7) | (selected > 12))
        or not np.allclose(
            probabilities[np.arange(len(selected)), selected - 7],
            probabilities.max(axis=1),
            rtol=0,
            atol=1e-7,
        )
    ):
        raise ValueError("Selector outputs are inconsistent in {}".format(path))
    for key in ("baseline_boxes", "policy_boxes", "action_boxes"):
        if np.any(data[key][..., 2:] <= 0):
            raise ValueError("Invalid predicted boxes in {}".format(path))
    if np.any(data["image_hw"] <= 0) or np.any(data["resize_factor"] <= 0):
        raise ValueError("Invalid image size or resize factor in {}".format(path))

    eligible_delta = data["discounted_delta"][eligible]
    oracle = np.argmax(eligible_delta, axis=1) if len(eligible_delta) else np.asarray([], dtype=int)
    distribution = Counter(int(index) + 7 for index in oracle.tolist())
    valid_baseline = data["baseline_iou"][data["baseline_valid_gt"]]
    report = {
        "split": entry["split"],
        "sequence": entry["sequence"],
        "tracked_frames": tracked,
        "anchors": run["anchor_count"],
        "eligible_anchors": int(eligible.sum()),
        "fast_mean_iou": float(valid_baseline.mean()) if valid_baseline.size else "",
        "control_probe_max_abs_diff": float(np.max(np.abs(data["control_probe_boxes"] - control[0]))),
        "oracle_layer_distribution": "|".join(
            "l{}:{}".format(layer, distribution.get(layer, 0)) for layer in CANDIDATE_LAYERS
        ),
    }
    return report, data, metadata
