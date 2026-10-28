"""Dependency-light helpers for Stage-8B single-L9 interventions."""

import json
from pathlib import Path

import numpy as np

from stage4_common import (
    IOU_PROTOCOL,
    file_hash,
    fingerprint,
    load_manifest,
    tracking_iou,
)


SCHEMA = "stage8_l9_intervention_v1"
DEFAULT_HORIZONS = (1, 5, 10, 20, 40)
OVERLAP_THRESHOLDS = np.arange(21, dtype=np.float64) * 0.05


def select_anchor_offsets(tracked_frames, max_horizon, anchor_count):
    """Choose outcome-independent anchors across the usable clip interval."""
    tracked_frames = int(tracked_frames)
    max_horizon = int(max_horizon)
    anchor_count = int(anchor_count)
    if tracked_frames < 1 or max_horizon < 1 or anchor_count < 1:
        raise ValueError("Tracked frames, horizon and anchor count must be positive")
    last_anchor = tracked_frames - max_horizon + 1
    if last_anchor < anchor_count:
        raise ValueError(
            "Need at least {} valid anchor positions, got {} from {} tracked "
            "frames and horizon {}".format(
                anchor_count, last_anchor, tracked_frames, max_horizon
            )
        )
    offsets = (
        np.floor(
            (np.arange(anchor_count, dtype=np.float64) + 0.5)
            * last_anchor
            / anchor_count
        ).astype(np.int64)
        + 1
    )
    if len(np.unique(offsets)) != anchor_count:
        raise RuntimeError("Anchor construction produced duplicate offsets")
    if offsets[0] < 1 or offsets[-1] > last_anchor:
        raise RuntimeError("Anchor construction left the usable interval")
    return offsets


def _require_allclose(actual, expected, message, path, atol=1e-9):
    if not np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True):
        difference = np.abs(
            np.asarray(actual, dtype=np.float64)
            - np.asarray(expected, dtype=np.float64)
        )
        finite = difference[np.isfinite(difference)]
        maximum = float(finite.max()) if finite.size else float("nan")
        raise ValueError(
            "{} in {} (max abs diff {})".format(message, path, maximum)
        )


def _window_auc(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return float("nan")
    return 100.0 * float(
        np.mean(values[:, None] > OVERLAP_THRESHOLDS[None, :])
    )


def intervention_labels(control_iou, treatment_iou, valid_gt, horizons):
    """Compute cumulative and post-anchor effects for each causal horizon."""
    control_iou = np.asarray(control_iou, dtype=np.float64)
    treatment_iou = np.asarray(treatment_iou, dtype=np.float64)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    horizons = tuple(int(value) for value in horizons)
    if (
        control_iou.shape != treatment_iou.shape
        or control_iou.shape != valid_gt.shape
        or control_iou.ndim != 2
    ):
        raise ValueError("Intervention IoU arrays must have matching [anchor, time] shapes")
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("Horizons must be positive, sorted and unique")
    if horizons[0] < 1 or horizons[-1] > control_iou.shape[1]:
        raise ValueError("Horizons exceed the collected intervention window")

    shape = (control_iou.shape[0], len(horizons))
    output = {
        "control_mean_iou": np.full(shape, np.nan, dtype=np.float64),
        "treatment_mean_iou": np.full(shape, np.nan, dtype=np.float64),
        "mean_iou_delta": np.full(shape, np.nan, dtype=np.float64),
        "control_auc": np.full(shape, np.nan, dtype=np.float64),
        "treatment_auc": np.full(shape, np.nan, dtype=np.float64),
        "auc_delta": np.full(shape, np.nan, dtype=np.float64),
        "post_anchor_mean_iou_delta": np.full(shape, np.nan, dtype=np.float64),
        "post_anchor_auc_delta": np.full(shape, np.nan, dtype=np.float64),
        "terminal_iou_delta": np.full(shape, np.nan, dtype=np.float64),
        "failure_reduction_02": np.full(shape, np.nan, dtype=np.float64),
        "valid_frames_by_horizon": np.zeros(shape, dtype=np.int64),
    }
    for anchor_index in range(control_iou.shape[0]):
        for horizon_index, horizon in enumerate(horizons):
            valid = valid_gt[anchor_index, :horizon]
            output["valid_frames_by_horizon"][anchor_index, horizon_index] = int(
                valid.sum()
            )
            if valid.any():
                control = control_iou[anchor_index, :horizon][valid]
                treatment = treatment_iou[anchor_index, :horizon][valid]
                output["control_mean_iou"][anchor_index, horizon_index] = float(
                    control.mean()
                )
                output["treatment_mean_iou"][anchor_index, horizon_index] = float(
                    treatment.mean()
                )
                output["mean_iou_delta"][anchor_index, horizon_index] = float(
                    (treatment - control).mean()
                )
                output["control_auc"][anchor_index, horizon_index] = _window_auc(
                    control
                )
                output["treatment_auc"][anchor_index, horizon_index] = _window_auc(
                    treatment
                )
                output["auc_delta"][anchor_index, horizon_index] = (
                    output["treatment_auc"][anchor_index, horizon_index]
                    - output["control_auc"][anchor_index, horizon_index]
                )
                output["failure_reduction_02"][anchor_index, horizon_index] = float(
                    np.mean(control < 0.2) - np.mean(treatment < 0.2)
                )
            if horizon > 1:
                post_valid = valid_gt[anchor_index, 1:horizon]
                if post_valid.any():
                    post_control = control_iou[anchor_index, 1:horizon][post_valid]
                    post_treatment = treatment_iou[anchor_index, 1:horizon][post_valid]
                    output["post_anchor_mean_iou_delta"][
                        anchor_index, horizon_index
                    ] = float((post_treatment - post_control).mean())
                    output["post_anchor_auc_delta"][
                        anchor_index, horizon_index
                    ] = _window_auc(post_treatment) - _window_auc(post_control)
            if valid_gt[anchor_index, horizon - 1]:
                output["terminal_iou_delta"][anchor_index, horizon_index] = (
                    treatment_iou[anchor_index, horizon - 1]
                    - control_iou[anchor_index, horizon - 1]
                )
    return output


def intervention_cache_path(output_dir, entry):
    return (
        Path(output_dir)
        / entry["split"]
        / "{}__l9_intervention.npz".format(entry["sequence"])
    )


def make_run(
    manifest,
    runtime_identity,
    pool_grid,
    anchor_count,
    horizons,
    verify_l9,
):
    horizons = tuple(int(value) for value in horizons)
    if anchor_count < 1:
        raise ValueError("anchor_count must be positive")
    if not horizons or tuple(sorted(set(horizons))) != horizons or horizons[0] < 1:
        raise ValueError("horizons must be positive, sorted and unique")
    directory = Path(__file__).resolve().parent
    files = (
        "stage8_intervention_common.py",
        "stage8_intervention_collect.py",
        "stage8_intervention_manifest.py",
        "stage4_common.py",
        "stage4_runtime.py",
        "validation_metrics.py",
        "stage1_validate.py",
    )
    run = {
        "schema": SCHEMA,
        "manifest": manifest,
        "runtime": runtime_identity,
        "pool_grid": int(pool_grid),
        "anchor_count": int(anchor_count),
        "horizons": list(horizons),
        "max_horizon": max(horizons),
        "treatment": "single_fixed_l9_then_fast",
        "control": "fast_then_fast",
        "verify_l9": bool(verify_l9),
        "iou_protocol": IOU_PROTOCOL,
        "code_sha256": {name: file_hash(directory / name) for name in files},
        "feature_fields": [
            "token_features",
            "fast_score_map",
            "selector_prob",
            "selected_layer",
            "previous_state",
            "previous_fast_box",
            "fast_box",
            "l9_box",
            "image_hw",
            "resize_factor",
            "anchor_offset",
            "anchor_fraction",
        ],
        "label_fields": [
            "control_iou",
            "treatment_iou",
            "delta_iou",
            "control_mean_iou",
            "treatment_mean_iou",
            "mean_iou_delta",
            "control_auc",
            "treatment_auc",
            "auc_delta",
            "post_anchor_mean_iou_delta",
            "post_anchor_auc_delta",
            "terminal_iou_delta",
            "failure_reduction_02",
        ],
    }
    run["fingerprint"] = fingerprint(run)
    return run


def load_intervention_run(directory):
    directory = Path(directory).expanduser().resolve()
    path = directory / "run.json"
    if not path.is_file():
        raise FileNotFoundError("Intervention run metadata missing: {}".format(path))
    run = json.loads(path.read_text(encoding="utf-8"))
    digest = run.get("fingerprint")
    body = {key: value for key, value in run.items() if key != "fingerprint"}
    if (
        run.get("schema") != SCHEMA
        or run.get("iou_protocol") != IOU_PROTOCOL
        or fingerprint(body) != digest
    ):
        raise ValueError("Invalid Stage-8B run metadata or fingerprint")
    load_manifest(run["manifest"])
    return directory, run


def _expected_shapes(entry, run):
    tracked = entry["stop"] - entry["start"] - 1
    anchors = run["anchor_count"]
    horizon = run["max_horizon"]
    num_horizons = len(run["horizons"])
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
        "selector_prob": (anchors, 6),
        "selected_layer": (anchors,),
        "previous_state": (anchors, 4),
        "previous_fast_box": (anchors, 4),
        "fast_box": (anchors, 4),
        "l9_box": (anchors, 4),
        "image_hw": (anchors, 2),
        "resize_factor": (anchors,),
        "baseline_boxes": (tracked, 4),
        "baseline_gt": (tracked, 4),
        "baseline_valid_gt": (tracked,),
        "baseline_iou": (tracked,),
        "control_boxes": (anchors, horizon, 4),
        "control_probe_boxes": (horizon, 4),
        "treatment_boxes": (anchors, horizon, 4),
        "gt": (anchors, horizon, 4),
        "valid_gt": (anchors, horizon),
        "control_iou": (anchors, horizon),
        "treatment_iou": (anchors, horizon),
        "delta_iou": (anchors, horizon),
        "horizons": (num_horizons,),
        "control_mean_iou": (anchors, num_horizons),
        "treatment_mean_iou": (anchors, num_horizons),
        "mean_iou_delta": (anchors, num_horizons),
        "control_auc": (anchors, num_horizons),
        "treatment_auc": (anchors, num_horizons),
        "auc_delta": (anchors, num_horizons),
        "post_anchor_mean_iou_delta": (anchors, num_horizons),
        "post_anchor_auc_delta": (anchors, num_horizons),
        "terminal_iou_delta": (anchors, num_horizons),
        "failure_reduction_02": (anchors, num_horizons),
        "valid_frames_by_horizon": (anchors, num_horizons),
    }


def load_and_audit_cache(path, entry, run):
    """Load one cache and verify its causal branch and label invariants."""
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
        raise ValueError("Unexpected/missing intervention fields: {}".format(path))
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(
                "Bad {} shape in {}: {} != {}".format(
                    key, path, data[key].shape, shape
                )
            )

    finite_fields = set(data).difference(
        {
            "baseline_gt",
            "baseline_iou",
            "gt",
            "control_iou",
            "treatment_iou",
            "delta_iou",
            "control_mean_iou",
            "treatment_mean_iou",
            "mean_iou_delta",
            "control_auc",
            "treatment_auc",
            "auc_delta",
            "post_anchor_mean_iou_delta",
            "post_anchor_auc_delta",
            "terminal_iou_delta",
            "failure_reduction_02",
        }
    )
    for key in finite_fields:
        if not np.isfinite(data[key]).all():
            raise ValueError("Non-finite {} in {}".format(key, path))
    for key in ("frame_id", "anchor_offset", "selected_layer", "image_hw", "horizons"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("{} must contain integers".format(key))
    for key in ("baseline_valid_gt", "valid_gt"):
        if data[key].dtype != np.bool_:
            raise ValueError("{} must be a boolean mask".format(key))

    tracked = entry["stop"] - entry["start"] - 1
    expected_offsets = select_anchor_offsets(
        tracked, run["max_horizon"], run["anchor_count"]
    )
    if not np.array_equal(data["anchor_offset"], expected_offsets):
        raise ValueError("Anchor offsets changed in {}".format(path))
    if not np.array_equal(data["frame_id"], entry["start"] + expected_offsets):
        raise ValueError("Absolute anchor frames changed in {}".format(path))
    if not np.array_equal(data["horizons"], np.asarray(run["horizons"])):
        raise ValueError("Horizon list changed in {}".format(path))
    _require_allclose(
        data["anchor_fraction"],
        expected_offsets / tracked,
        "Anchor fractions changed",
        path,
        atol=1e-12,
    )

    initial = np.asarray(metadata.get("init_box"), dtype=np.float64)
    if initial.shape != (4,) or not np.isfinite(initial).all() or np.any(initial[2:] <= 0):
        raise ValueError("Invalid intervention initialization in {}".format(path))
    expected_previous = np.stack(
        [
            initial if offset == 1 else data["baseline_boxes"][offset - 2]
            for offset in expected_offsets
        ]
    )
    _require_allclose(
        data["previous_state"],
        expected_previous,
        "Causal previous states changed",
        path,
    )
    _require_allclose(
        data["previous_fast_box"],
        expected_previous,
        "Previous Fast boxes changed",
        path,
    )
    _require_allclose(
        data["fast_box"],
        data["baseline_boxes"][expected_offsets - 1],
        "Anchor Fast boxes disagree with the baseline",
        path,
    )
    expected_control = np.stack(
        [
            data["baseline_boxes"][offset - 1 : offset - 1 + run["max_horizon"]]
            for offset in expected_offsets
        ]
    )
    expected_gt = np.stack(
        [
            data["baseline_gt"][offset - 1 : offset - 1 + run["max_horizon"]]
            for offset in expected_offsets
        ]
    )
    expected_valid = np.stack(
        [
            data["baseline_valid_gt"][
                offset - 1 : offset - 1 + run["max_horizon"]
            ]
            for offset in expected_offsets
        ]
    )
    _require_allclose(
        data["control_boxes"], expected_control, "Control is not a Fast baseline slice", path
    )
    _require_allclose(
        data["control_probe_boxes"],
        expected_control[0],
        "Independent Fast replay disagrees with the baseline",
        path,
        atol=1e-6,
    )
    _require_allclose(
        data["control_boxes"][:, 0],
        data["fast_box"],
        "Control anchor boxes disagree with Fast",
        path,
    )
    _require_allclose(
        data["treatment_boxes"][:, 0],
        data["l9_box"],
        "Treatment anchor boxes disagree with fixed L9",
        path,
    )
    _require_allclose(data["gt"], expected_gt, "GT windows are misaligned", path)
    if not np.array_equal(data["valid_gt"], expected_valid):
        raise ValueError("Visibility windows are misaligned in {}".format(path))

    expected_baseline_iou = tracking_iou(
        data["baseline_boxes"], data["baseline_gt"], data["baseline_valid_gt"]
    )
    expected_control_iou = tracking_iou(
        data["control_boxes"], data["gt"], data["valid_gt"]
    )
    expected_treatment_iou = tracking_iou(
        data["treatment_boxes"], data["gt"], data["valid_gt"]
    )
    for key, expected in (
        ("baseline_iou", expected_baseline_iou),
        ("control_iou", expected_control_iou),
        ("treatment_iou", expected_treatment_iou),
        ("delta_iou", expected_treatment_iou - expected_control_iou),
    ):
        if not np.allclose(data[key], expected, rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError("{} label mismatch in {}".format(key, path))
    labels = intervention_labels(
        expected_control_iou,
        expected_treatment_iou,
        data["valid_gt"],
        run["horizons"],
    )
    for key, expected in labels.items():
        if not np.allclose(data[key], expected, rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError("{} horizon label mismatch in {}".format(key, path))

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
        raise ValueError("Selector probabilities are inconsistent in {}".format(path))
    for key in ("baseline_boxes", "control_boxes", "treatment_boxes"):
        if np.any(data[key][..., 2:] <= 0):
            raise ValueError("Invalid predicted boxes in {}".format(path))
    if (
        np.any(data["control_probe_boxes"][..., 2:] <= 0)
        or np.any(data["image_hw"] <= 0)
        or np.any(data["resize_factor"] <= 0)
    ):
        raise ValueError("Invalid replay box, image size or resize factor in {}".format(path))

    valid_baseline_iou = data["baseline_iou"][data["baseline_valid_gt"]]
    report = {
        "split": entry["split"],
        "sequence": entry["sequence"],
        "tracked_frames": tracked,
        "anchors": run["anchor_count"],
        "valid_anchor_frames": int(data["valid_gt"][:, 0].sum()),
        "fast_mean_iou": (
            float(valid_baseline_iou.mean()) if valid_baseline_iou.size else ""
        ),
        "control_probe_max_abs_diff": float(
            np.max(np.abs(data["control_probe_boxes"] - expected_control[0]))
        ),
    }
    for horizon_index, horizon in enumerate(run["horizons"]):
        values = data["auc_delta"][:, horizon_index]
        values = values[np.isfinite(values)]
        report["mean_auc_delta_h{}".format(horizon)] = (
            float(values.mean()) if values.size else ""
        )
    return report, data, metadata
