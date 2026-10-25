"""Geometry and data integrity for a descriptive UAV failure audit."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_json
from validation_metrics import tracking_iou


SCHEMA = "stage27_uav_failure_diagnostic_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DESIGN = dict(failure_iou=0.2, failure_run=3, recovery_iou=0.5,
              recovery_run=3, review_before=10, review_after=30,
              outside_coverage=0.1, inside_coverage=0.9,
              input_size_edges=[16, 32], motion_edges=[0.005, 0.02],
              motion_max_side=640, motion_min_tracks=20,
              motion_min_inliers=12, motion_min_inlier_ratio=0.5)


def nonempty_path(value):
    if not str(value).strip():
        raise argparse.ArgumentTypeError("Empty path; export the shell variable first")
    return value


def seal(body):
    result = dict(body)
    result["fingerprint"] = fingerprint(body)
    return result


def load_sealed(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in data.items() if k != "fingerprint"}
    if data.get("schema") != SCHEMA or data.get("fingerprint") != fingerprint(body):
        raise ValueError("Schema or fingerprint mismatch: " + str(path))
    return data


def publish_identical(path, data):
    path = Path(path)
    if path.exists():
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError("Existing artifact differs; use a new path: " + str(path))
        return False
    write_json(path, data)
    return True


def inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset path escapes root: " + str(relative))
    return path


def frame_number(path):
    match = re.fullmatch(r"(?:img)?([0-9]+)", Path(path).stem, re.IGNORECASE)
    if match is None:
        raise ValueError("Unsupported frame filename: " + str(path))
    return int(match.group(1))


def source_group(name):
    match = re.fullmatch(r"(uav[0-9]+)_[0-9]+_s", name)
    if match is None:
        raise ValueError("Cannot infer source-video group from: " + name)
    return match.group(1)


def read_boxes(path):
    rows = [list(map(float, re.split(r"[,\s]+", line.strip())))
            for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError("Expected one xywh box per line: " + str(path))
    boxes = np.asarray(rows, dtype=np.float64)
    valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
    # This is geometric validity, NOT visibility or absence supervision.
    if not valid.all():
        raise ValueError("This frozen pilot requires finite positive boxes: " + str(path))
    return boxes


def read_rgb(path):
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None or not bgr.size:
        raise ValueError("Cannot decode image: " + str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def scan_dataset(root, expected_sequences=86, expected_frames=69941):
    root = Path(root).expanduser().resolve()
    if not (root / "sequences").is_dir() or not (root / "annotations").is_dir():
        raise ValueError("Root must contain sequences/ and annotations/: " + str(root))
    entries = []
    for directory in sorted(p for p in (root / "sequences").iterdir() if p.is_dir()):
        annotation = inside(root, "annotations/" + directory.name + ".txt")
        boxes = read_boxes(annotation)
        images = sorted((p for p in directory.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=frame_number)
        ids = [frame_number(p) for p in images]
        if len(images) != len(boxes) or len(images) < 2:
            raise ValueError("Frame/annotation count mismatch: " + directory.name)
        if any(b != a + 1 for a, b in zip(ids, ids[1:])):
            raise ValueError("Duplicate or discontinuous frame IDs: " + directory.name)
        frames = []
        for path in images:
            path = inside(root, path.relative_to(root))
            size = path.stat().st_size
            if not size:
                raise ValueError("Empty image: " + str(path))
            frames.append(dict(file=path.relative_to(root).as_posix(), bytes=size,
                               source_frame_number=frame_number(path)))
        entries.append(dict(sequence=directory.name, source_group_proxy=source_group(directory.name),
                            annotation=annotation.relative_to(root).as_posix(),
                            annotation_sha256=file_hash(annotation), frames=frames))
    extras = {p.stem for p in (root / "annotations").glob("*.txt")} - {e["sequence"] for e in entries}
    total = sum(len(e["frames"]) for e in entries)
    if extras or len(entries) != expected_sequences or total != expected_frames:
        raise ValueError("Expected {} sequences / {} frames; found {} / {}; orphan annotations={}".format(
            expected_sequences, expected_frames, len(entries), total, sorted(extras)))
    return root, entries


def select_entries(entries, count=24, seed=2091):
    groups = {}
    for entry in sorted(entries, key=lambda e: e["sequence"]):
        groups.setdefault(entry["source_group_proxy"], []).append(entry)
    if not 1 <= count <= len(groups):
        raise ValueError("Need {} distinct source-group proxies; only {} available".format(count, len(groups)))
    rng = np.random.default_rng(seed)
    names = sorted(groups)
    selected = []
    for index in rng.permutation(len(names))[:count]:
        choices = groups[names[int(index)]]
        selected.append(choices[int(rng.integers(len(choices)))])
    return sorted(selected, key=lambda e: e["sequence"])


def load_manifest(path, check_code=True):
    data = path if isinstance(path, dict) else load_sealed(path)
    body = {k: v for k, v in data.items() if k != "fingerprint"}
    if data.get("schema") != SCHEMA or data.get("fingerprint") != fingerprint(body):
        raise ValueError("Manifest fingerprint mismatch")
    if data.get("role") != "descriptive_development_dataset_manifest" or data.get("design") != DESIGN:
        raise ValueError("Not the frozen Stage-27 dataset design")
    entries = data["sequences"]
    names = [e["sequence"] for e in entries]
    groups = [e["source_group_proxy"] for e in entries]
    if (not entries or len(names) != data["selected_sequences"] or len(set(names)) != len(names)
            or len(set(groups)) != len(groups)):
        raise ValueError("Duplicate/missing sequence or source group")
    for entry in entries:
        if source_group(entry["sequence"]) != entry["source_group_proxy"] or len(entry["frames"]) < 2:
            raise ValueError("Invalid selected sequence")
        gt = np.asarray(entry["ground_truth"], float)
        if (gt.shape != (len(entry["frames"]), 4) or not np.isfinite(gt).all()
                or (gt[:, 2:] <= 0).any() or not np.array_equal(gt[0], entry["initial_box"])):
            raise ValueError("Invalid frozen ground truth")
        for frame in entry["frames"]:
            inside(data["root"], frame["file"])
            if not frame.get("sha256") or min(frame["height"], frame["width"]) < 2:
                raise ValueError("Missing image audit")
    if check_code:
        for name, digest in data["code_sha256"].items():
            if file_hash(Path(__file__).parent / name) != digest:
                raise ValueError("Manifest source changed; use the frozen code: " + name)
    return data


def checked_image(root, frame, reader=read_rgb):
    path = inside(root, frame["file"])
    if file_hash(path) != frame["sha256"]:
        raise ValueError("Image bytes changed after freeze: " + str(path))
    image = reader(path)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != (frame["height"], frame["width"]):
        raise ValueError("Image dimensions changed: " + str(path))
    return image


def intersection_area(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:])
                            - np.maximum(a[:2], b[:2])).prod())


def crop_geometry(previous, image_hw, search_factor, search_size):
    x, y, w, h = np.asarray(previous, float)
    if not np.isfinite([x, y, w, h, search_factor]).all() or min(w, h, search_factor, search_size) <= 0:
        raise ValueError("Invalid crop parent/configuration")
    size = math.ceil(math.sqrt(w * h) * search_factor)
    x1, y1 = round(x + w / 2 - size / 2), round(y + h / 2 - size / 2)
    height, width = map(int, image_hw)
    # Preserve sample_target's historical '+1' right/bottom padding convention.
    xp, yp = max(0, -x1), max(0, -y1)
    xq, yq = max(x1 + size - width + 1, 0), max(y1 + size - height + 1, 0)
    usable = np.array([x1 + xp, y1 + yp, max(0, size - xp - xq), max(0, size - yp - yq)], float)
    return np.array([x1, y1, size, size], float), usable, search_size / size


def scene_geometry(previous, gt, image_hw, search_factor, search_size):
    crop, usable, factor = crop_geometry(previous, image_hw, search_factor, search_size)
    gt = np.asarray(gt, float)
    area = float(gt[2:].prod())
    height, width = image_hw
    center = gt[:2] + gt[2:] / 2
    return dict(crop=crop, usable=usable, factor=factor,
                coverage=intersection_area(gt, usable) / area,
                gt_image_fraction=intersection_area(gt, [0, 0, width, height]) / area,
                input_wh=gt[2:] * factor,
                center_in_usable=bool(((center >= usable[:2]) & (center < usable[:2] + usable[2:])).all()),
                padding_fraction=1 - float(usable[2:].prod()) / float(crop[2:].prod()))


def estimate_motion(previous, current, old_boxes, new_boxes):
    """Offline GT-masked background-motion proxy; never used by the tracker."""
    import cv2
    unknown = np.array([np.nan, np.nan, 0, 0, 0], dtype=float)
    if previous.shape != current.shape:
        return unknown
    h, w = current.shape[:2]
    scale = min(1., DESIGN["motion_max_side"] / max(h, w))
    size = (max(2, round(w * scale)), max(2, round(h * scale)))
    xy_scale = np.array([size[0] / w, size[1] / h])
    old_gray = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY), size)
    new_gray = cv2.resize(cv2.cvtColor(current, cv2.COLOR_RGB2GRAY), size)
    masks = []
    for boxes in (old_boxes, new_boxes):
        mask = np.full(old_gray.shape, 255, np.uint8)
        for box in boxes:
            x, y, bw, bh = np.asarray(box, float) * np.tile(xy_scale, 2)
            lo = np.floor([x - bw * .2, y - bh * .2]).astype(int)
            hi = np.ceil([x + bw * 1.2, y + bh * 1.2]).astype(int)
            cv2.rectangle(mask, tuple(lo), tuple(hi), 0, -1)
        masks.append(mask)
    points = cv2.goodFeaturesToTrack(old_gray, maxCorners=300, qualityLevel=.01,
                                    minDistance=7, mask=masks[0])
    if points is None:
        return unknown
    moved, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, points, None)
    if moved is None or status is None:
        return unknown
    p, q = points.reshape(-1, 2), moved.reshape(-1, 2)
    good = status.ravel().astype(bool) & np.isfinite(p).all(1) & np.isfinite(q).all(1)
    good &= (q[:, 0] >= 0) & (q[:, 0] < size[0]) & (q[:, 1] >= 0) & (q[:, 1] < size[1])
    p, q = p[good], q[good]
    allowed = masks[1][np.minimum(q[:, 1].astype(int), size[1] - 1),
                       np.minimum(q[:, 0].astype(int), size[0] - 1)] > 0
    p, q = p[allowed], q[allowed]
    unknown[2] = len(p)
    if len(p) < DESIGN["motion_min_tracks"]:
        return unknown
    transform, inlier = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
                         ransacReprojThreshold=3., maxIters=2000, confidence=.99)
    if transform is None or inlier is None or not np.isfinite(transform).all():
        return unknown
    count = int(inlier.sum())
    ratio = count / len(p)
    unknown[3:] = count, ratio
    if count < DESIGN["motion_min_inliers"] or ratio < DESIGN["motion_min_inlier_ratio"]:
        return unknown
    points = np.array([[x * (size[0] - 1), y * (size[1] - 1)]
                       for x in (.2, .5, .8) for y in (.2, .5, .8)])
    projected = points @ transform[:, :2].T + transform[:, 2]
    background = np.median(np.linalg.norm(projected - points, axis=1)) / math.hypot(*size)
    old, new = np.asarray(old_boxes[0], float), np.asarray(new_boxes[0], float)
    center = (old[:2] + old[2:] / 2) * xy_scale
    residual = np.linalg.norm(transform[:, :2] @ center + transform[:, 2]
                              - (new[:2] + new[2:] / 2) * xy_scale)
    residual /= max(1., math.sqrt(float(old[2:].prod() * xy_scale.prod())))
    return np.array([background, residual, len(p), count, ratio], float)


def sharpness_proxy(image, gt):
    import cv2
    h, w = image.shape[:2]
    x, y, bw, bh = gt
    x1, y1 = max(0, math.floor(x)), max(0, math.floor(y))
    x2, y2 = min(w, math.ceil(x + bw)), min(h, math.ceil(y + bh))
    if min(x2 - x1, y2 - y1) < 8:
        return np.nan
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def audit_trace(data, entry, run):
    n = len(entry["frames"]) - 1
    shapes = dict(frame_id=(n,), previous=(n, 4), boxes=(n, 4), gt=(n, 4),
                  image_hw=(n, 2), crop=(n, 4), usable=(n, 4), factor=(n,),
                  coverage=(n,), gt_image_fraction=(n,), input_wh=(n, 2),
                  center_in_usable=(n,), padding_fraction=(n,), layer=(n,),
                  raw_score_peak=(n,), motion=(n, 5), sharpness=(n,), iou=(n,))
    if set(data) != set(shapes) or any(np.shape(data[k]) != shape for k, shape in shapes.items()):
        raise ValueError("Incomplete trace: " + entry["sequence"])
    if not np.array_equal(data["frame_id"], np.arange(1, n + 1)):
        raise ValueError("Trace must cover the full video after initialization")
    for key in set(shapes) - {"motion", "sharpness"}:
        if not np.isfinite(data[key]).all():
            raise ValueError("Nonfinite trace field: " + key)
    if np.isinf(data["motion"]).any() or np.isinf(data["sharpness"]).any():
        raise ValueError("Proxy fields may use NaN, not infinity")
    initial = np.asarray(entry["initial_box"], float)
    np.testing.assert_allclose(data["previous"], np.vstack([initial, data["boxes"][:-1]]), atol=0, rtol=0)
    if not np.isin(data["layer"], np.arange(7, 13)).all() or (data["boxes"][:, 2:] <= 0).any():
        raise ValueError("Invalid executed layer or box")
    np.testing.assert_allclose(data["iou"], tracking_iou(data["boxes"], data["gt"]), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(data["gt"], np.asarray(entry["ground_truth"])[1:])
    for i in range(n):
        expected_hw = [entry["frames"][i + 1]["height"], entry["frames"][i + 1]["width"]]
        np.testing.assert_array_equal(data["image_hw"][i], expected_hw)
        expected = scene_geometry(data["previous"][i], data["gt"][i], expected_hw,
                                  run["search_factor"], run["search_size"])
        for key in ("crop", "usable", "factor", "coverage", "gt_image_fraction", "input_wh",
                    "center_in_usable", "padding_fraction"):
            np.testing.assert_allclose(data[key][i], expected[key], atol=1e-10, rtol=0)


def success_auc(iou):
    values = np.asarray(iou, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("AUC requires nonempty geometrically valid tracked frames")
    return float(100 * (values[:, None] > np.linspace(0, 1, 21)).mean())


def coverage_stratum(value):
    return "outside_lt_10pct" if value < .1 else "partial_10_to_90pct" if value < .9 else "inside_ge_90pct"


def size_stratum(value):
    return "lt_16px" if value < 16 else "16_to_32px" if value < 32 else "ge_32px"


def motion_stratum(value):
    if not np.isfinite(value):
        return "unknown"
    return "low_lt_0.5pct_diag" if value < .005 else "medium_0.5_to_2pct_diag" if value < .02 else "high_ge_2pct_diag"


def failure_events(iou):
    """Start after 3 low-IoU frames; recover after 3 IoU>=.5 frames.

    Indices are zero-based rows of the tracked-frame trace, not source image IDs.
    Invalid gaps censor an active event rather than bridging it.
    """
    events, active, low, good = [], None, 0, 0
    for index, value in enumerate(iou):
        if not np.isfinite(value) or value < 0:
            if active is not None:
                events.append(dict(active, end=index - 1, recovery=None, censor="invalid_gap"))
            active, low, good = None, 0, 0
            continue
        if active is None:
            low = low + 1 if value < DESIGN["failure_iou"] else 0
            if low == DESIGN["failure_run"]:
                active = dict(start=index - 2, confirmation=index)
                good = 0
        else:
            good = good + 1 if value >= DESIGN["recovery_iou"] else 0
            if good == DESIGN["recovery_run"]:
                events.append(dict(active, end=index - 3, recovery=index - 2, censor="none"))
                active, low, good = None, 0, 0
    if active is not None:
        events.append(dict(active, end=len(iou) - 1, recovery=None, censor="video_end"))
    return events


def load_trace(path, entry, run):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    if metadata != dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"]):
        raise ValueError("Trace identity mismatch: " + str(path))
    audit_trace(data, entry, run)
    return data
