"""Stage-46: causal background compensation for native SGLA search centers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_cache, write_json
from stage27_uav_common import checked_image, publish_identical
from stage28_response_common import load_source

SCHEMA = "stage46_sgla_camera_search_v1"
METHODS = ("sgla", "velocity", "camera_only", "camera_velocity")
DESIGN = dict(videos=12, seed=2127, smoke_frames=64, ema=.5, max_shift_side=.5,
    motion_max_side=640, target_mask_factor=2., corners=500, corner_quality=.01,
    corner_distance=8, min_tracks=30, min_inliers=20, min_inlier_ratio=.6,
    fb_error=1.5, ransac_error=2., median_error=1.5, min_cells=4,
    scale_min=.9, scale_max=1.1, rotation_deg=10., max_camera_diag=.15,
    bootstrap_samples=5000, bootstrap_seed=2128, auc_practice_target=.5,
    replay_box_atol=.001, replay_peak_atol=.00001)
FILES = ("stage46_motion_common.py", "stage46_motion_runtime.py", "stage46_motion_experiment.py",
         "stage46_motion_summary.py", "stage28_response_common.py")
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


def seal(value):
    return dict(value, fingerprint=fingerprint(value))


def load_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint(body):
        raise ValueError("Stage-46 schema/fingerprint mismatch: " + str(path))
    return value


def code_hashes():
    return {n: file_hash(Path(__file__).with_name(n)) for n in FILES}


def resolve_source(path):
    path = Path(path).expanduser().resolve()
    found = [p for p in (path, path/"cache") if (p/"run.json").is_file()]
    if len(found) != 1:
        raise ValueError("Need Stage-27 cache with run.json/traces: {}. Set paths after cd ~/workspace/paper.".format(path))
    return found[0]


def choose(traces):
    names = sorted(traces, key=lambda n: fingerprint([DESIGN["seed"], n]))
    if len(names) < DESIGN["videos"]:
        raise ValueError("Need 12 complete development source groups")
    return names[:DESIGN["videos"]]


def make_plan(source):
    directory, run, traces, hashes = load_source(resolve_source(source))
    names = choose(traces)
    entries = [traces[n][0] for n in names]
    if len({e["source_group_proxy"] for e in entries}) != len(entries):
        raise ValueError("Source groups are not distinct")
    return seal(dict(schema=SCHEMA, role="consumed_development_native_sgla_motion_pilot",
        source_input=str(directory), source_run_fingerprint=run["fingerprint"],
        source_run_sha256=file_hash(directory/"run.json"), source_trace_sha256=hashes,
        design=DESIGN, methods=list(METHODS), sequences=names,
        tracked_frames=sum(len(e["frames"])-1 for e in entries), code_sha256=code_hashes(),
        training_performed=False, original_checking_labels_loaded=False,
        selection="12 of the consumed Stage-27 groups, SHA order of seed/name; no outcome filtering",
        background_estimation="new current/previous images, masked by OWN previous prediction; never reuse Stage-27 GT-masked proxies")), (directory, run, traces, hashes)


def load_plan(root):
    plan = load_sealed(Path(root)/"manifest.json")
    expected, inputs = make_plan(plan["source_input"])
    if plan != expected:
        raise ValueError("Frozen Stage-46 source/code/design changed")
    return plan, inputs


def center(box):
    box = np.asarray(box, float)
    return box[:2] + box[2:]/2


def warp(matrix, point):
    matrix = np.asarray(matrix, float)
    return matrix[:, :2] @ np.asarray(point, float) + matrix[:, 2]


def bounded(vector, limit):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    return vector * min(1., limit/max(norm, 1e-12)), norm > limit


class MotionState:
    """Own boxes only. Camera matrices map previous-image coordinates to current."""
    def __init__(self, box):
        self.previous = np.asarray(box, float).copy()
        if self.previous.shape != (4,) or not np.isfinite(self.previous).all() or min(self.previous[2:]) <= 0:
            raise ValueError("Invalid initial box")
        self.raw = np.zeros(2)
        self.relative = np.zeros(2)

    def propose(self, method, matrix, valid, image_hw):
        if method not in METHODS:
            raise ValueError("Unknown Stage-46 policy")
        previous = self.previous
        c0 = center(previous)
        camera = warp(matrix, c0)-c0 if valid else np.zeros(2)
        if method == "sgla":
            shift = np.zeros(2)
        elif method == "velocity" or (method == "camera_velocity" and not valid):
            shift = self.raw.copy()
        elif method == "camera_only":
            shift = camera.copy()
        else:
            shift = camera + matrix[:, :2] @ self.relative
        proposed = c0 + shift
        limited, cap = bounded(shift, DESIGN["max_shift_side"]*math.sqrt(previous[2]*previous[3]))
        final = c0+limited
        if method != "sgla":
            final = np.clip(final, [0., 0.], np.asarray(image_hw[::-1], float)-1)
        parent = previous.copy()
        parent[:2] += final-c0
        return dict(parent=parent, raw_velocity=self.raw.copy(), relative_velocity=self.relative.copy(),
                    proposal=proposed, shift=final-c0, cap_hit=cap, camera_displacement=camera)

    def advance(self, box, matrix, valid):
        box = np.asarray(box, float)
        if not np.isfinite(box).all() or min(box[2:]) <= 0:
            raise ValueError("Invalid SGLA prediction")
        alpha = DESIGN["ema"]
        c0, c1 = center(self.previous), center(box)
        self.raw = (1-alpha)*self.raw + alpha*(c1-c0)
        if valid:
            self.relative = ((1-alpha)*(matrix[:, :2] @ self.relative) + alpha*(c1-warp(matrix, c0)))
        else:
            self.relative = np.zeros(2)  # do not carry an unknown camera residual through a failed fit
        self.previous = box.copy()


def gray_image(image):
    import cv2
    h, w = image.shape[:2]
    ratio = min(1., DESIGN["motion_max_side"]/max(h, w))
    shape = (max(2, round(w*ratio)), max(2, round(h*ratio)))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if shape != (w, h):
        gray = cv2.resize(gray, shape, interpolation=cv2.INTER_AREA)
    return gray, np.array([shape[0]/w, shape[1]/h], float)


def background_mask(shape, box, scales):
    mask = np.full(shape, 255, np.uint8)
    c0, size = center(box)*scales, np.asarray(box[2:])*scales*DESIGN["target_mask_factor"]
    x0, y0 = np.floor(c0-size/2).astype(int)
    x1, y1 = np.ceil(c0+size/2).astype(int)
    x0, x1 = np.clip([x0,x1], 0, shape[1])
    y0, y1 = np.clip([y0,y1], 0, shape[0])
    mask[y0:y1,x0:x1] = 0
    return mask


def estimate_camera(previous, current, own_box, image_hw, seed):
    """LK forward/backward + partial-affine RANSAC; no current GT or predicted output."""
    import cv2
    g0, scales0 = previous
    g1, scales1 = current
    stats = np.zeros(6, float)  # corners, surviving tracks, inliers, ratio, median residual, grid cells
    def reject(reason):
        return dict(matrix=IDENTITY.copy(), valid=False, stats=stats.copy(), reason=reason)
    if g0.shape != g1.shape or not np.array_equal(scales0, scales1):
        return reject(1)
    cv2.setRNGSeed(int(seed) % (2**31-1))
    points = cv2.goodFeaturesToTrack(g0, DESIGN["corners"], DESIGN["corner_quality"],
        DESIGN["corner_distance"], mask=background_mask(g0.shape, own_box, scales0))
    if points is None:
        return reject(2)
    stats[0] = len(points)
    if len(points) < DESIGN["min_tracks"]:
        return reject(2)
    options = dict(winSize=(21,21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 30, .01))
    forward, s1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, points, None, **options)
    if forward is None or s1 is None:
        return reject(3)
    p, q = points.reshape(-1,2), forward.reshape(-1,2)
    ok_forward = ((s1.ravel() == 1) & np.isfinite(q).all(1) &
                  (q[:,0] >= 0) & (q[:,0] < g1.shape[1]) & (q[:,1] >= 0) & (q[:,1] < g1.shape[0]))
    p, q = p[ok_forward], q[ok_forward]
    if len(p) < DESIGN["min_tracks"]:
        return reject(3)
    backward, s0, _ = cv2.calcOpticalFlowPyrLK(g1, g0, q.reshape(-1,1,2), None, **options)
    if backward is None or s0 is None:
        return reject(3)
    back = backward.reshape(-1,2)
    keep = ((s0.ravel() == 1) & np.isfinite(back).all(1) &
            (np.linalg.norm(p-back, axis=1) <= DESIGN["fb_error"]) &
            (q[:,0] >= 0) & (q[:,0] < g1.shape[1]) & (q[:,1] >= 0) & (q[:,1] < g1.shape[0]))
    p, q = p[keep], q[keep]
    stats[1] = len(p)
    if len(p) < DESIGN["min_tracks"]:
        return reject(3)
    matrix, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
        ransacReprojThreshold=DESIGN["ransac_error"], maxIters=2000, confidence=.99, refineIters=10)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return reject(4)
    ok = inliers.ravel().astype(bool)
    stats[2:4] = [int(ok.sum()), float(ok.mean())]
    residual = np.linalg.norm(p @ matrix[:,:2].T + matrix[:,2]-q, axis=1)
    stats[4] = float(np.median(residual[ok])) if ok.any() else 0.
    cells = np.minimum(3, (p[ok] / np.array(g0.shape[::-1]) * 4).astype(int))
    stats[5] = len({tuple(x) for x in cells})
    if (stats[2] < DESIGN["min_inliers"] or stats[3] < DESIGN["min_inlier_ratio"] or
        stats[4] > DESIGN["median_error"] or stats[5] < DESIGN["min_cells"]):
        return reject(5)
    # Conjugate the full affine by the actual x/y resize factors, including rounding.
    transform = np.diag([*scales0, 1.])
    full = np.eye(3)
    full[:2] = matrix
    full = np.linalg.inv(transform) @ full @ transform
    result = full[:2]
    scale = math.sqrt(abs(np.linalg.det(result[:,:2])))
    angle = abs(math.degrees(math.atan2(result[1,0], result[0,0])))
    middle = np.asarray(image_hw[::-1], float)/2
    displacement = np.linalg.norm(warp(result, middle)-middle)/np.linalg.norm(image_hw)
    if not DESIGN["scale_min"] <= scale <= DESIGN["scale_max"] or angle > DESIGN["rotation_deg"] or displacement > DESIGN["max_camera_diag"]:
        return reject(6)
    return dict(matrix=result, valid=True, stats=stats, reason=0)


def cache_path(root, sequence, method, smoke=False):
    return Path(root)/("smoke/traces" if smoke else "traces")/(sequence+"__"+method+".npz")


def metadata(run, sequence, method, smoke):
    return dict(run_fingerprint=run["fingerprint"], sequence=sequence, method=method, smoke=smoke)


def save_result(path, data, meta):
    path = Path(path)
    write_cache(path, data, meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, meta=meta, sha256=file_hash(path))))


def load_result(path, meta):
    path = Path(path)
    receipt = path.with_suffix(".receipt.json")
    if not receipt.exists():
        return None  # uncommitted interrupted video is recomputed
    record = load_sealed(receipt)
    if not path.exists() or record.get("meta") != meta or record.get("sha256") != file_hash(path):
        raise ValueError("Committed Stage-46 trace changed: " + str(path))
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != meta:
            raise ValueError("Mixed trace metadata")
        return {k:f[k] for k in f.files if k != "metadata_json"}


def audit_video(data, frames, initial_box, method, reference=None):
    n = len(frames)-1
    shapes = dict(frame_id=(n,), previous=(n,4), parents=(n,4), boxes=(n,4), image_hw=(n,2),
        layer=(n,), peak=(n,), matrix=(n,2,3), motion_valid=(n,), motion_reason=(n,), motion_stats=(n,6),
        raw_velocity=(n,2), relative_velocity=(n,2), proposal=(n,2), shift=(n,2), cap_hit=(n,),
        camera_displacement=(n,2), motion_ms=(n,), network_ms=(n,), pipeline_ms=(n,),
        block_calls=(n,12), head_calls=(n,), selector_calls=(n,))
    if set(data) != set(shapes) or any(np.shape(data[k]) != s for k,s in shapes.items()):
        raise ValueError("Trace shape/coverage mismatch")
    if not all(np.isfinite(x).all() for x in data.values()):
        raise ValueError("Non-finite trace")
    for key in ("motion_valid", "cap_hit"):
        if data[key].dtype != np.bool_:
            raise ValueError("Invalid boolean field " + key)
    for key in ("frame_id", "image_hw", "layer", "motion_reason", "block_calls", "head_calls", "selector_calls"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("Invalid integer field " + key)
    np.testing.assert_array_equal(data["frame_id"], np.arange(1,n+1))
    if not np.isin(data["layer"], np.arange(7,13)).all() or (data["boxes"][:,2:] <= 0).any():
        raise ValueError("Invalid SGLA layer/box")
    np.testing.assert_array_equal(data["head_calls"], np.ones(n, int))
    np.testing.assert_array_equal(data["selector_calls"], np.ones(n, int))
    expected = np.zeros((n,12), int)
    expected[:,:6] = 1
    expected[np.arange(n),data["layer"]-1] = 1
    np.testing.assert_array_equal(data["block_calls"], expected)
    state = MotionState(initial_box)
    for i in range(n):
        np.testing.assert_array_equal(data["previous"][i], state.previous)
        hw = (frames[i+1]["height"],frames[i+1]["width"])
        np.testing.assert_array_equal(data["image_hw"][i], hw)
        valid, matrix = data["motion_valid"][i], data["matrix"][i]
        if not valid:
            np.testing.assert_array_equal(matrix, IDENTITY)
        if valid != (data["motion_reason"][i] == 0):
            raise ValueError("Camera fit status mismatch")
        stats = data["motion_stats"][i]
        if valid and (stats[0] < DESIGN["min_tracks"] or stats[1] < DESIGN["min_tracks"] or
            stats[2] < DESIGN["min_inliers"] or stats[3] < DESIGN["min_inlier_ratio"] or
            stats[4] > DESIGN["median_error"] or stats[5] < DESIGN["min_cells"]):
            raise ValueError("Accepted camera fit lacks quality support")
        p = state.propose(method, matrix, valid, hw)
        for key in ("raw_velocity", "relative_velocity", "proposal", "shift", "cap_hit", "camera_displacement"):
            np.testing.assert_allclose(data[key][i], p[key], atol=1e-9, rtol=0)
        np.testing.assert_allclose(data["parents"][i], p["parent"], atol=1e-9, rtol=0)
        state.advance(data["boxes"][i], matrix, valid)
    if reference is not None and method == "sgla":
        np.testing.assert_allclose(data["boxes"], reference["boxes"][:n], atol=DESIGN["replay_box_atol"], rtol=0)
        # Official SGLA IoU truncates predicted coordinates. Small floating-point
        # tolerance must not silently permit a different integer prediction/AUC.
        np.testing.assert_array_equal(np.trunc(data["boxes"]), np.trunc(reference["boxes"][:n]))
        np.testing.assert_allclose(data["peak"], reference["raw_score_peak"][:n], atol=DESIGN["replay_peak_atol"], rtol=0)
        np.testing.assert_array_equal(data["layer"], reference["layer"][:n])
