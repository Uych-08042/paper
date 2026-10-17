"""Dependency-light helpers for Stage-9 equal-cost schedule interventions."""

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
from stage8_intervention_common import select_anchor_offsets


SCHEMA = "stage9_equal_cost_schedule_v1"
POLICIES = ("fast", "burst4", "paired4", "uniform4")
SCHEDULES = {
    "fast": (),
    "burst4": (0, 1, 2, 3),
    "paired4": (0, 1, 10, 11),
    "uniform4": (0, 5, 10, 15),
}
DEFAULT_HORIZON = 40
DEFAULT_EVALUATION_HORIZONS = (5, 10, 20, 40)
PRIMARY_HORIZON = 20
COMMON_TAIL = (16, 20)
PERSISTENCE_WINDOW = (20, 40)
PRIMARY_COMPARISON = ("uniform4", "burst4")
OVERLAP_THRESHOLDS = np.arange(21, dtype=np.float64) * 0.05


def validate_schedules(schedules, horizon):
    """Validate ordered schedules and return their boolean action mask."""
    if tuple(schedules) != POLICIES:
        raise ValueError("Stage-9 policy order changed")
    horizon = int(horizon)
    if horizon < 1 or schedules["fast"]:
        raise ValueError("Fast must have an empty schedule and horizon must be positive")
    costs = []
    mask = np.zeros((len(POLICIES), horizon), dtype=bool)
    for policy_index, policy in enumerate(POLICIES):
        offsets = tuple(schedules[policy])
        if any(type(value) is not int for value in offsets):
            raise ValueError("Schedule offsets must be integers")
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError("Schedule offsets must be sorted and unique")
        if offsets and (offsets[0] != 0 or offsets[-1] >= horizon):
            raise ValueError("Every treatment must start at zero and fit the horizon")
        if policy != "fast":
            costs.append(len(offsets))
        if offsets:
            mask[policy_index, list(offsets)] = True
    if not costs or len(set(costs)) != 1 or costs[0] != 4:
        raise ValueError("Every Stage-9 treatment must use exactly four L9 blocks")
    if max(max(schedules[policy]) for policy in POLICIES[1:]) >= COMMON_TAIL[0]:
        raise ValueError("The common tail must begin after every intervention")
    if not 0 <= COMMON_TAIL[0] < COMMON_TAIL[1] <= horizon:
        raise ValueError("Invalid common-tail interval")
    if not 0 <= PERSISTENCE_WINDOW[0] < PERSISTENCE_WINDOW[1] <= horizon:
        raise ValueError("Invalid persistence interval")
    return mask


def window_auc(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return float("nan")
    return 100.0 * float(
        np.mean(values[:, None] > OVERLAP_THRESHOLDS[None, :])
    )


def schedule_labels(policy_iou, valid_gt, horizons):
    """Compute policy metrics for cumulative and common post-action windows."""
    policy_iou = np.asarray(policy_iou, dtype=np.float64)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    horizons = tuple(int(value) for value in horizons)
    if policy_iou.ndim != 3 or valid_gt.shape != (
        policy_iou.shape[0],
        policy_iou.shape[2],
    ):
        raise ValueError("Expected policy IoU [anchor, policy, time] and valid [anchor, time]")
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("Evaluation horizons must be positive, sorted and unique")
    if horizons[0] < 1 or horizons[-1] > policy_iou.shape[2]:
        raise ValueError("Evaluation horizons exceed the collected window")

    shape = (policy_iou.shape[0], policy_iou.shape[1], len(horizons))
    output = {
        "policy_mean_iou": np.full(shape, np.nan, dtype=np.float64),
        "policy_auc": np.full(shape, np.nan, dtype=np.float64),
        "policy_failure_rate_02": np.full(shape, np.nan, dtype=np.float64),
        "policy_terminal_iou": np.full(shape, np.nan, dtype=np.float64),
        "valid_frames_by_horizon": np.zeros(
            (policy_iou.shape[0], len(horizons)), dtype=np.int64
        ),
        "common_tail_mean_iou": np.full(
            policy_iou.shape[:2], np.nan, dtype=np.float64
        ),
        "common_tail_auc": np.full(
            policy_iou.shape[:2], np.nan, dtype=np.float64
        ),
        "persistence_mean_iou": np.full(
            policy_iou.shape[:2], np.nan, dtype=np.float64
        ),
        "persistence_auc": np.full(
            policy_iou.shape[:2], np.nan, dtype=np.float64
        ),
    }
    for anchor_index in range(policy_iou.shape[0]):
        for horizon_index, horizon in enumerate(horizons):
            valid = valid_gt[anchor_index, :horizon]
            output["valid_frames_by_horizon"][anchor_index, horizon_index] = int(
                valid.sum()
            )
            if valid.any():
                for policy_index in range(policy_iou.shape[1]):
                    values = policy_iou[anchor_index, policy_index, :horizon][valid]
                    output["policy_mean_iou"][
                        anchor_index, policy_index, horizon_index
                    ] = float(values.mean())
                    output["policy_auc"][
                        anchor_index, policy_index, horizon_index
                    ] = window_auc(values)
                    output["policy_failure_rate_02"][
                        anchor_index, policy_index, horizon_index
                    ] = float(np.mean(values < 0.2))
            if valid_gt[anchor_index, horizon - 1]:
                output["policy_terminal_iou"][
                    anchor_index, :, horizon_index
                ] = policy_iou[anchor_index, :, horizon - 1]

        for prefix, interval in (
            ("common_tail", COMMON_TAIL),
            ("persistence", PERSISTENCE_WINDOW),
        ):
            start, stop = interval
            valid = valid_gt[anchor_index, start:stop]
            if not valid.any():
                continue
            for policy_index in range(policy_iou.shape[1]):
                values = policy_iou[
                    anchor_index, policy_index, start:stop
                ][valid]
                output[prefix + "_mean_iou"][anchor_index, policy_index] = float(
                    values.mean()
                )
                output[prefix + "_auc"][anchor_index, policy_index] = window_auc(
                    values
                )
    return output


def schedule_cache_path(output_dir, entry):
    return (
        Path(output_dir)
        / entry["split"]
        / "{}__stage9_schedule.npz".format(entry["sequence"])
    )


def _serialized_schedules(schedules):
    return [
        {"policy": policy, "l9_offsets": list(schedules[policy])}
        for policy in POLICIES
    ]


def schedules_from_run(run):
    schedules = {
        row["policy"]: tuple(int(value) for value in row["l9_offsets"])
        for row in run["schedules"]
    }
    validate_schedules(schedules, run["horizon"])
    return schedules


def make_run(
    manifest,
    runtime_identity,
    anchor_count,
    horizon,
    evaluation_horizons,
    schedules,
    verify_l9,
):
    evaluation_horizons = tuple(int(value) for value in evaluation_horizons)
    action_mask = validate_schedules(schedules, horizon)
    if anchor_count < 1:
        raise ValueError("anchor_count must be positive")
    if (
        not evaluation_horizons
        or tuple(sorted(set(evaluation_horizons))) != evaluation_horizons
        or evaluation_horizons[0] < 1
        or evaluation_horizons[-1] > horizon
    ):
        raise ValueError("Evaluation horizons must be positive, sorted and fit the window")
    if PRIMARY_HORIZON not in evaluation_horizons:
        raise ValueError("The primary H20 endpoint must be collected")
    directory = Path(__file__).resolve().parent
    files = (
        "stage9_schedule_common.py",
        "stage9_schedule_manifest.py",
        "stage9_schedule_collect.py",
        "stage4_common.py",
        "stage4_runtime.py",
        "validation_metrics.py",
        "stage1_validate.py",
    )
    run = {
        "schema": SCHEMA,
        "manifest": manifest,
        "runtime": runtime_identity,
        "anchor_count": int(anchor_count),
        "horizon": int(horizon),
        "evaluation_horizons": list(evaluation_horizons),
        "policies": list(POLICIES),
        "schedules": _serialized_schedules(schedules),
        "equal_treatment_blocks": int(action_mask[1].sum()),
        "primary_horizon": PRIMARY_HORIZON,
        "primary_comparison": list(PRIMARY_COMPARISON),
        "common_tail": list(COMMON_TAIL),
        "persistence_window": list(PERSISTENCE_WINDOW),
        "verify_l9": bool(verify_l9),
        "iou_protocol": IOU_PROTOCOL,
        "code_sha256": {name: file_hash(directory / name) for name in files},
    }
    run["fingerprint"] = fingerprint(run)
    return run


def load_schedule_run(directory):
    directory = Path(directory).expanduser().resolve()
    path = directory / "run.json"
    if not path.is_file():
        raise FileNotFoundError("Stage-9 run metadata missing: {}".format(path))
    run = json.loads(path.read_text(encoding="utf-8"))
    digest = run.get("fingerprint")
    body = {key: value for key, value in run.items() if key != "fingerprint"}
    if (
        run.get("schema") != SCHEMA
        or run.get("iou_protocol") != IOU_PROTOCOL
        or fingerprint(body) != digest
    ):
        raise ValueError("Invalid Stage-9 run metadata or fingerprint")
    load_manifest(run["manifest"])
    if (
        run.get("policies") != list(POLICIES)
        or run.get("primary_horizon") != PRIMARY_HORIZON
        or run.get("primary_comparison") != list(PRIMARY_COMPARISON)
        or run.get("common_tail") != list(COMMON_TAIL)
        or run.get("persistence_window") != list(PERSISTENCE_WINDOW)
    ):
        raise ValueError("Frozen Stage-9 design changed")
    schedules_from_run(run)
    return directory, run


def _expected_shapes(entry, run):
    tracked = entry["stop"] - entry["start"] - 1
    anchors = run["anchor_count"]
    policies = len(run["policies"])
    horizon = run["horizon"]
    evaluations = len(run["evaluation_horizons"])
    return {
        "frame_id": (anchors,),
        "anchor_offset": (anchors,),
        "anchor_fraction": (anchors,),
        "previous_state": (anchors, 4),
        "baseline_boxes": (tracked, 4),
        "baseline_gt": (tracked, 4),
        "baseline_valid_gt": (tracked,),
        "baseline_iou": (tracked,),
        "control_probe_boxes": (horizon, 4),
        "action_mask": (policies, horizon),
        "policy_boxes": (anchors, policies, horizon, 4),
        "gt": (anchors, horizon, 4),
        "valid_gt": (anchors, horizon),
        "policy_iou": (anchors, policies, horizon),
        "horizons": (evaluations,),
        "policy_mean_iou": (anchors, policies, evaluations),
        "policy_auc": (anchors, policies, evaluations),
        "policy_failure_rate_02": (anchors, policies, evaluations),
        "policy_terminal_iou": (anchors, policies, evaluations),
        "valid_frames_by_horizon": (anchors, evaluations),
        "common_tail_mean_iou": (anchors, policies),
        "common_tail_auc": (anchors, policies),
        "persistence_mean_iou": (anchors, policies),
        "persistence_auc": (anchors, policies),
    }


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


def load_and_audit_cache(path, entry, run):
    """Load one Stage-9 cache and recompute every alignment and metric."""
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
        raise ValueError("Unexpected/missing Stage-9 fields: {}".format(path))
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(
                "Bad {} shape in {}: {} != {}".format(
                    key, path, data[key].shape, shape
                )
            )

    for key in (
        "frame_id",
        "anchor_offset",
        "horizons",
    ):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("{} must contain integers".format(key))
    for key in ("baseline_valid_gt", "valid_gt", "action_mask"):
        if data[key].dtype != np.bool_:
            raise ValueError("{} must be a boolean mask".format(key))
    finite_fields = set(data).difference(
        {
            "baseline_gt",
            "baseline_iou",
            "gt",
            "policy_iou",
            "policy_mean_iou",
            "policy_auc",
            "policy_failure_rate_02",
            "policy_terminal_iou",
            "common_tail_mean_iou",
            "common_tail_auc",
            "persistence_mean_iou",
            "persistence_auc",
        }
    )
    for key in finite_fields:
        if not np.isfinite(data[key]).all():
            raise ValueError("Non-finite {} in {}".format(key, path))

    schedules = schedules_from_run(run)
    expected_mask = validate_schedules(schedules, run["horizon"])
    if not np.array_equal(data["action_mask"], expected_mask):
        raise ValueError("Action mask changed in {}".format(path))
    tracked = entry["stop"] - entry["start"] - 1
    offsets = select_anchor_offsets(tracked, run["horizon"], run["anchor_count"])
    if not np.array_equal(data["anchor_offset"], offsets):
        raise ValueError("Anchor offsets changed in {}".format(path))
    if not np.array_equal(data["frame_id"], entry["start"] + offsets):
        raise ValueError("Absolute anchor frames changed in {}".format(path))
    if not np.array_equal(
        data["horizons"], np.asarray(run["evaluation_horizons"])
    ):
        raise ValueError("Evaluation horizons changed in {}".format(path))
    _require_allclose(
        data["anchor_fraction"],
        offsets / tracked,
        "Anchor fractions changed",
        path,
        atol=1e-12,
    )

    initial = np.asarray(metadata.get("init_box"), dtype=np.float64)
    if (
        initial.shape != (4,)
        or not np.isfinite(initial).all()
        or np.any(initial[2:] <= 0)
    ):
        raise ValueError("Invalid Stage-9 initialization in {}".format(path))
    expected_previous = np.stack(
        [
            initial if offset == 1 else data["baseline_boxes"][offset - 2]
            for offset in offsets
        ]
    )
    _require_allclose(
        data["previous_state"], expected_previous, "Causal previous states changed", path
    )
    expected_control = np.stack(
        [
            data["baseline_boxes"][offset - 1 : offset - 1 + run["horizon"]]
            for offset in offsets
        ]
    )
    expected_gt = np.stack(
        [
            data["baseline_gt"][offset - 1 : offset - 1 + run["horizon"]]
            for offset in offsets
        ]
    )
    expected_valid = np.stack(
        [
            data["baseline_valid_gt"][offset - 1 : offset - 1 + run["horizon"]]
            for offset in offsets
        ]
    )
    _require_allclose(
        data["policy_boxes"][:, 0],
        expected_control,
        "Fast policy is not the baseline slice",
        path,
    )
    _require_allclose(
        data["control_probe_boxes"],
        expected_control[0],
        "Independent Fast replay disagrees with the baseline",
        path,
        atol=1e-6,
    )
    _require_allclose(data["gt"], expected_gt, "GT windows are misaligned", path)
    if not np.array_equal(data["valid_gt"], expected_valid):
        raise ValueError("Visibility windows are misaligned in {}".format(path))

    expected_baseline_iou = tracking_iou(
        data["baseline_boxes"], data["baseline_gt"], data["baseline_valid_gt"]
    )
    expected_policy_iou = tracking_iou(
        data["policy_boxes"], data["gt"][:, None, :, :], data["valid_gt"][:, None, :]
    )
    _require_allclose(
        data["baseline_iou"], expected_baseline_iou, "Baseline IoU changed", path
    )
    _require_allclose(
        data["policy_iou"], expected_policy_iou, "Policy IoU changed", path
    )
    labels = schedule_labels(
        expected_policy_iou, data["valid_gt"], run["evaluation_horizons"]
    )
    for key, expected in labels.items():
        _require_allclose(data[key], expected, "{} changed".format(key), path)
    if np.any(data["baseline_boxes"][..., 2:] <= 0) or np.any(
        data["policy_boxes"][..., 2:] <= 0
    ):
        raise ValueError("Invalid predicted boxes in {}".format(path))

    valid_baseline = data["baseline_iou"][data["baseline_valid_gt"]]
    report = {
        "split": entry["split"],
        "sequence": entry["sequence"],
        "tracked_frames": tracked,
        "anchors": run["anchor_count"],
        "valid_anchor_frames": int(data["valid_gt"][:, 0].sum()),
        "equal_treatment_blocks": run["equal_treatment_blocks"],
        "fast_mean_iou": (
            float(valid_baseline.mean()) if valid_baseline.size else ""
        ),
        "control_probe_max_abs_diff": float(
            np.max(np.abs(data["control_probe_boxes"] - expected_control[0]))
        ),
    }
    primary_index = run["evaluation_horizons"].index(PRIMARY_HORIZON)
    for policy_index, policy in enumerate(POLICIES):
        values = data["policy_auc"][:, policy_index, primary_index]
        values = values[np.isfinite(values)]
        report["{}_h20_auc".format(policy)] = (
            float(values.mean()) if values.size else ""
        )
    return report, data, metadata
