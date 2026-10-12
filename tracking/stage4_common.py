"""Small, dependency-light helpers for Stage-4 experiment files."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from validation_metrics import IOU_PROTOCOL, tracking_iou


SCHEMA = "stage4_ab_v1"
ACTIONS = ("fast", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), suffix=".tmp",
                                     encoding="utf-8", delete=False) as handle:
        handle.write(content)
        temporary = handle.name
    os.replace(temporary, str(path))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty report")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def load_annotations(root, dataset, sequence):
    root = Path(root)
    if dataset == "got10k_train":
        directory = root / sequence
        visible_files = (directory / "absence.label", directory / "cover.label")
    elif dataset == "lasot_train":
        directory = root / sequence.rsplit("-", 1)[0] / sequence
        visible_files = (directory / "full_occlusion.txt", directory / "out_of_view.txt")
    else:
        raise ValueError("Only explicit training sources are supported")
    gt = np.loadtxt(str(directory / "groundtruth.txt"), delimiter=",", ndmin=2)
    if gt.shape[1] != 4 or len(gt) < 2:
        raise ValueError("Invalid xywh annotation: {}".format(directory))
    labels = [np.loadtxt(str(path), delimiter=",", ndmin=1).reshape(-1) for path in visible_files]
    if any(len(label) != len(gt) for label in labels):
        raise ValueError("Visibility/GT length mismatch: {}".format(directory))
    visible = ((labels[0] == 0) & (labels[1] > 0)) if dataset == "got10k_train" else ((labels[0] == 0) & (labels[1] == 0))
    valid = visible & np.isfinite(gt).all(axis=1) & (gt[:, 2:] > 0).all(axis=1)
    return directory, gt, valid


def frame_path(directory, dataset, index):
    directory = Path(directory)
    if dataset == "lasot_train":
        directory = directory / "img"
    return directory / "{:08d}.jpg".format(index + 1)


def load_manifest(path):
    data = path if isinstance(path, dict) else json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA or data.get("dataset") not in ("got10k_train", "lasot_train"):
        raise ValueError("Unsupported manifest schema or training source")
    digest = data.get("fingerprint")
    body = {key: value for key, value in data.items() if key != "fingerprint"}
    if fingerprint(body) != digest:
        raise ValueError("Manifest fingerprint mismatch; regenerate the manifest")
    seen = set()
    for entry in data["sequences"]:
        name = entry["sequence"]
        if name in seen or Path(name).name != name or "/" in name or "\\" in name or name in ("", ".", ".."):
            raise ValueError("Duplicate or unsafe sequence name: {}".format(name))
        seen.add(name)
        if (entry["split"] not in ("train", "val")
                or type(entry["start"]) is not int or type(entry["stop"]) is not int
                or not 0 <= entry["start"] < entry["stop"] - 1):
            raise ValueError("Invalid split/clip: {}".format(name))
    if not seen:
        raise ValueError("Empty manifest")
    if {entry["split"] for entry in data["sequences"]} != {"train", "val"}:
        raise ValueError("Manifest must contain separate train and val videos")
    return data


def cache_path(output_dir, entry, trajectory):
    return Path(output_dir) / entry["split"] / "{}__{}.npz".format(entry["sequence"], trajectory)


def write_cache(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=str(path.parent), suffix=".tmp", delete=False) as handle:
        temporary = handle.name
        np.savez_compressed(handle, metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)
    os.replace(temporary, str(path))


def audit_cache(path, entry, trajectory, run):
    """Validate causal history and labels without loading any tracker dependencies."""
    with np.load(str(path), allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        if (meta["run_fingerprint"] != run["fingerprint"] or meta["entry"] != entry
                or meta["trajectory"] != trajectory):
            raise ValueError("Cache metadata mismatch: {}".format(path))
        data = {key: archive[key] for key in archive.files if key != "metadata_json"}
    n = entry["stop"] - entry["start"] - 1
    expected_frames = np.arange(entry["start"] + 1, entry["stop"])
    if not np.array_equal(data["frame_id"], expected_frames):
        raise ValueError("Wrong decision frame IDs: {}".format(path))
    dim = run["runtime"]["embed_dim"]
    side = int(run["runtime"]["lens_x"] ** 0.5)
    shapes = {
        "token_features": (n, (2 + run["pool_grid"] ** 2) * dim),
        "fast_score_map": (n, 1, side, side), "selector_prob": (n, 6),
        "previous_state": (n, 4), "previous_fast_box": (n, 4), "previous_action": (n,),
        "fast_box": (n, 4), "selected_layer": (n,), "action_boxes": (n, 7, 4),
        "gt": (n, 4), "valid_gt": (n,), "iou": (n, 7), "delta": (n, 6),
        "executed_action": (n,), "image_hw": (n, 2), "resize_factor": (n,),
        "frame_id": (n,),
    }
    if run["save_full_tokens"]:
        shapes["full_tokens"] = (n, run["runtime"]["lens_z"] + run["runtime"]["lens_x"], dim)
    if set(data) != set(shapes):
        raise ValueError("Unexpected/missing cache fields: {}".format(path))
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError("Bad {} shape: {} != {}".format(key, data[key].shape, shape))
        if key not in ("gt", "delta") and not np.isfinite(data[key]).all():
            raise ValueError("Non-finite {} in {}".format(key, path))
    if data["valid_gt"].dtype != np.bool_:
        raise ValueError("valid_gt must be a boolean mask")
    for key in ("frame_id", "previous_action", "selected_layer", "executed_action", "image_hw"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("{} must contain integers".format(key))
    if (np.any(data["action_boxes"][:, :, 2:] <= 0) or np.any(data["resize_factor"] <= 0)
            or np.any(data["image_hw"] <= 0)):
        raise ValueError("Invalid box/image dimensions or resize factor")
    layers = data["selected_layer"]
    if np.any((layers < 7) | (layers > 12)):
        raise ValueError("Invalid selected layer")
    probs = data["selector_prob"]
    if np.any((probs < 0) | (probs > 1)) or not np.allclose(
            probs[np.arange(n), layers - 7], probs.max(axis=1), rtol=0, atol=1e-7):
        raise ValueError("Selected layer disagrees with selector probabilities")
    executed = np.zeros(n, dtype=int) if trajectory == "fast" else layers - 6
    if not np.array_equal(data["executed_action"], executed):
        raise ValueError("Executed action disagrees with rollout policy")
    initial = np.asarray(meta["init_box"], dtype=float)
    if initial.shape != (4,) or not np.isfinite(initial).all() or np.any(initial[2:] <= 0):
        raise ValueError("Invalid initial box")
    previous = np.concatenate((initial[None], data["action_boxes"][np.arange(n - 1), executed[:-1]]))
    previous_fast = np.concatenate((initial[None], data["fast_box"][:-1]))
    for key, expected in (("previous_state", previous), ("previous_fast_box", previous_fast),
                          ("fast_box", data["action_boxes"][:, 0])):
        if not np.allclose(data[key], expected, rtol=0, atol=1e-9):
            raise ValueError("Causal state continuity failed for {}".format(key))
    if not np.array_equal(data["previous_action"], np.r_[-1, executed[:-1]]):
        raise ValueError("Causal action history failed")
    valid = data["valid_gt"]
    expected_iou = tracking_iou(data["action_boxes"], data["gt"][:, None], valid[:, None])
    if np.any(expected_iou[valid] < 0) or not np.allclose(data["iou"], expected_iou, rtol=0, atol=1e-10):
        raise ValueError("IoU label mismatch")
    expected_delta = np.where(valid[:, None], expected_iou[:, 1:] - expected_iou[:, :1], np.nan)
    if not np.allclose(data["delta"], expected_delta, rtol=0, atol=1e-10, equal_nan=True):
        raise ValueError("Delta label mismatch")
    iou = data["iou"][valid]
    return {"split": entry["split"], "sequence": entry["sequence"], "trajectory": trajectory,
            "decision_frames": n, "valid_frames": int(valid.sum()),
            "fast_mean_iou": float(iou[:, 0].mean()) if len(iou) else "",
            "best_action_mean_iou": float(iou.max(axis=1).mean()) if len(iou) else "",
            "iou_protocol": IOU_PROTOCOL}
