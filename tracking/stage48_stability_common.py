"""Fixed native-SGLA head-only, two-view stability development pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import crop_geometry, publish_identical
import stage47_search_common as prior

SCHEMA = "stage48_native_head_stability_oof_v1"
ARMS = ("supervised", "rank", "shape", "rank_shape")
SEEDS = (2132, 2133)
DESIGN = dict(folds=2, fold_seed=2131, samples_per_video=48, updates=256, smoke_updates=8,
    batch_pairs=2, save_every=128, learning_rate=1e-5, weight_decay=1e-4, gradient_clip=1.,
    rank_margin=.2, rank_weight=.1, shape_weight=.25, size_pair_weight=.25,
    positive_iou=.5, negative_iou=.1, log_size_beta=.1, smoke_frames=32,
    bootstrap_samples=5000, bootstrap_seed=2134, practical_auc_gain=.5,
    reliable_mean_floor=-.5, minimum_rank_pairs=16, minimum_rank_videos=2,
    views="native parent and one +/-1 original-image pixel translation; identical views across arms",
    training="Only original CENTER head parameters; backbone/selector/candidates and all BN buffers frozen")
FILES = ("stage48_stability_common.py", "stage48_stability_runtime.py", "stage48_stability_experiment.py",
         "stage48_stability_summary.py", "test_stage48_stability.py")


def seal(body):
    return dict(body, fingerprint=fingerprint(body))


def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint(body):
        raise ValueError("Stage-48 schema/fingerprint mismatch: " + str(path))
    return value


def code_hashes():
    return {n: file_hash(Path(__file__).with_name(n)) for n in FILES +
        ("stage18_tracking_loss.py", "stage47_search_common.py", "stage47_search_collect.py", "stage47_search_summary.py")}


def fold_design(entries):
    if len(entries) != 12 or len({e["source_group_proxy"] for e in entries}) != 12:
        raise ValueError("Need twelve distinct consumed development source groups")
    names = sorted((e["sequence"] for e in entries), key=lambda n: fingerprint([DESIGN["fold_seed"], n]))
    return [dict(fold=f, fit=sorted(names[(1-f)*6:(2-f)*6]), held_out=sorted(names[f*6:(f+1)*6])) for f in range(2)]


def make_plan(root):
    root = Path(root).expanduser().resolve()
    old, inputs = prior.load_plan(root)
    run = prior.read_sealed(root / "run.json")
    report = prior.read_sealed(root / "analysis/stage47_diagnostic.json")
    if (report.get("status") != "COMPLETE_MATCHED_SEARCH_RESPONSE_DIAGNOSTIC" or
        report.get("plan_fingerprint") != old["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
        report.get("training_performed") is not False or report.get("replay_head_calls") != old["expected_head_calls"]):
        raise ValueError("Need the unchanged completed Stage-47 diagnostic")
    bindings = dict(old["source_file_sha256"])
    bindings.update(report["cache_sha256"])
    bindings.update({str(root / "analysis" / n): d for n, d in report["table_sha256"].items()})
    for name in ("manifest.json", "run.json", "analysis/stage47_diagnostic.json"):
        bindings[str(root / name)] = file_hash(root / name)
    prior.prior.verify_files(bindings)
    for event in old["events"]:
        data = prior.read_cache(prior.path_for(root, event), prior.metadata(run, event))
        if data is None:
            raise ValueError("Missing Stage-47 replay")
        prior.audit_response(data, event, inputs, np.asarray(run["window"], np.float32))
    source46 = prior.motion.load_sealed(Path(old["source_input"]) / "manifest.json")
    entries = [inputs["traces"][n][0] for n in source46["sequences"]]
    folds = fold_design(entries)
    plan = seal(dict(schema=SCHEMA, role="consumed_development_native_head_two_factor_oof",
        source_stage47=str(root), source_stage46=old["source_input"], source_stage47_fingerprint=report["fingerprint"],
        source_file_sha256=bindings, code_sha256=code_hashes(), design=DESIGN, seeds=list(SEEDS), arms=list(ARMS),
        entries=entries, folds=folds, runtime=inputs["source"]["runtime"], runtime_identity=inputs["old_run"]["runtime"],
        root=inputs["source"]["manifest"]["root"], search_factor=inputs["source"]["search_factor"],
        search_size=inputs["source"]["search_size"], window=run["window"],
        original_checking_labels_loaded=False, independent_validation=False,
        interpretation="All twelve videos previously informed design; OOF is development evidence, not a fresh benchmark"))
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json")
    if plan != make_plan(plan["source_stage47"]):
        raise ValueError("Frozen Stage-48 source/code/design changed")
    return plan


def sample_frames(entry):
    n = len(entry["frames"]) - 1
    return np.unique(np.linspace(1, n, min(n, DESIGN["samples_per_video"])).round().astype(int)).tolist()


def pair_parents(parent, sequence, frame):
    values = np.tile(np.asarray(parent, float), (2, 1))
    direction = int(fingerprint([sequence, frame])[:8], 16) % 4
    values[1, direction // 2] += -1. if direction % 2 == 0 else 1.
    return values


def source_native(plan, entry):
    root = Path(plan["source_stage46"])
    run = prior.motion.load_sealed(root / "run.json")
    path = prior.motion.cache_path(root, entry["sequence"], "sgla")
    if file_hash(path) != plan["source_file_sha256"][str(path)]:
        raise ValueError("Frozen native trace changed")
    data = prior.motion.load_result(path, prior.motion.metadata(run, entry["sequence"], "sgla", False))
    prior.motion.audit_video(data, entry["frames"], entry["initial_box"], "sgla")
    return data


def cache_path(root, entry):
    return Path(root) / "features" / (entry["sequence"] + "__stage48_features.npz")


def save_cache(path, data, meta):
    write_cache(path, data, meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=meta, sha256=file_hash(path))))


def read_cache(path, meta):
    receipt = path.with_suffix(".receipt.json")
    if not receipt.exists():
        return None
    record = read_sealed(receipt)
    if record.get("metadata") != meta or not path.is_file() or file_hash(path) != record.get("sha256"):
        raise ValueError("Committed Stage-48 cache changed: " + str(path))
    with np.load(path, allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item())) != meta:
            raise ValueError("Mixed Stage-48 cache")
        return {k: a[k] for k in a.files if k != "metadata_json"}


def feature_meta(plan, entry):
    return dict(plan_fingerprint=plan["fingerprint"], sequence=entry["sequence"], role="frozen_native_features")


def audit_features(data, plan, entry, reference):
    from stage28_response_common import dense_boxes
    n, s, dim = len(sample_frames(entry)), int(plan["runtime"]["lens_x"] ** .5), plan["runtime"]["embed_dim"]
    k = (n, 2)
    shapes = dict(frame_id=(n,), features=k+(dim,s,s), parents=k+(4,), crop=k+(4,), raw=k+(s,s),
        size=k+(2,s,s), offset=k+(2,s,s), boxes=k+(4,), layer=k, target=k+(4,), gt_wh=k+(2,),
        positive=k+(s,s), negative=k+(s,s), size_cell=k, size_valid=k)
    if set(data) != set(shapes) or any(data[x].shape != shape for x, shape in shapes.items()):
        raise ValueError("Invalid Stage-48 feature pack shape")
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Non-finite feature pack")
    for name in ("positive", "negative", "size_valid"):
        if data[name].dtype != np.bool_:
            raise ValueError("Feature masks must be boolean")
    np.testing.assert_array_equal(data["frame_id"], sample_frames(entry))
    from stage27_uav_common import tracking_iou
    for i, frame in enumerate(data["frame_id"]):
        gt = np.asarray(entry["ground_truth"][int(frame)], float)
        hw = [entry["frames"][int(frame)]["height"], entry["frames"][int(frame)]["width"]]
        expected = pair_parents(reference["previous"][int(frame)-1], entry["sequence"], int(frame))
        np.testing.assert_array_equal(data["parents"][i], expected)
        np.testing.assert_allclose(data["boxes"][i,0], reference["boxes"][int(frame)-1], atol=.001, rtol=0)
        np.testing.assert_array_equal(np.trunc(data["boxes"][i,0]), np.trunc(reference["boxes"][int(frame)-1]))
        if data["layer"][i,0] != reference["layer"][int(frame)-1]:
            raise ValueError("Native feature selected-layer replay failed")
        for v in range(2):
            crop, _, factor = crop_geometry(expected[v], hw, plan["search_factor"], plan["search_size"])
            np.testing.assert_array_equal(data["crop"][i,v], crop)
            target = np.r_[(gt[:2]-crop[:2])/crop[2], gt[2:]/crop[2]]
            np.testing.assert_allclose(data["target"][i,v], target, atol=1e-10, rtol=0)
            np.testing.assert_array_equal(data["gt_wh"][i,v], gt[2:])
            boxes = dense_boxes(data["size"][i,v], data["offset"][i,v], expected[v], factor, plan["search_size"], hw)
            cell = int(np.argmax(data["raw"][i,v] * np.asarray(plan["window"])))
            if np.max(np.abs(boxes[cell]-data["boxes"][i,v])) > .001:
                raise ValueError("Feature dense/native decode mismatch")
            boxes[cell] = data["boxes"][i,v]
            overlap = tracking_iou(boxes, gt).reshape(s,s)
            np.testing.assert_array_equal(data["positive"][i,v], overlap >= DESIGN["positive_iou"])
            np.testing.assert_array_equal(data["negative"][i,v], overlap < DESIGN["negative_iou"])
            center = target[:2]+target[2:]/2
            valid = bool(((center >= 0)&(center < 1)).all())
            xy = np.clip(np.round(center*s).astype(int), 0, s-1)
            if data["size_valid"][i,v] != valid or data["size_cell"][i,v] != xy[1]*s+xy[0]:
                raise ValueError("GT size assignment changed")


def fit_pool(root, plan, fold):
    pool, bindings, support_videos = [], {}, set()
    for entry in plan["entries"]:
        if entry["sequence"] not in plan["folds"][fold]["fit"]:
            continue  # Held-out feature/label arrays are never loaded by fitting.
        path = cache_path(root, entry)
        data = read_cache(path, feature_meta(plan, entry))
        if data is None:
            raise ValueError("Collect all frozen feature packs before training")
        audit_features(data, plan, entry, source_native(plan, entry))
        bindings[str(path)] = file_hash(path)
        bindings[str(path.with_suffix(".receipt.json"))] = file_hash(path.with_suffix(".receipt.json"))
        for i in range(len(data["frame_id"])):
            sample = {k:data[k][i] for k in data if k != "frame_id"}
            sample.update(sequence=entry["sequence"], frame=int(data["frame_id"][i]))
            pool.append(sample)
    supported = [i for i,p in enumerate(pool) if (p["positive"].any((1,2)) & p["negative"].any((1,2))).any() and p["size_valid"].any()]
    support_videos = {pool[i]["sequence"] for i in supported}
    if len(supported) < DESIGN["minimum_rank_pairs"] or len(support_videos) < DESIGN["minimum_rank_videos"]:
        raise ValueError("Insufficient fit-only ranking support; preserve packs and inspect support")
    return pool, bindings, supported
