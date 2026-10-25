"""Small GT-free candidate-identity baseline; frozen backbone, explicit causal state."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_common import (checked_image, inside, read_boxes, read_rgb, scan_dataset,
                                seal, source_group, tracking_iou)
from stage30_branch_response_common import (load_inputs as load_stage30_inputs, load_point,
    load_sealed as load_stage30, point_filename, validate_plan as validate_stage30_plan)


SCHEMA = "stage31_continuous_identity_pilot_v1"
METHODS = ("sgla", "motion", "appearance", "static_ranker", "identity")
FEATURES = ("initial_cos", "trusted_cos", "recent_cos", "previous_other_max_cos", "identity_margin",
            "motion_cost", "scale_cost", "relative_response", "weighted_response")
DESIGN = dict(fit_videos=8, checking_videos=6, seed=2096, max_candidates=5,
    trusted_ema=0.1, update_initial_cos=0.5, update_identity_margin=0.02,
    motion_scale_weight=0.25, ranking_l2=0.1, ranking_iterations=50,
    minimum_pair_frames=100, minimum_pair_videos=4, bootstrap_samples=5000,
    minimum_alternative_ratio=0.005, minimum_alternative_videos=2)


def load_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint(body):
        raise ValueError("Stage-31 schema/fingerprint mismatch: " + str(path))
    return value


def code_hashes():
    names = ("stage31_identity_common.py", "stage31_identity_manifest.py", "stage31_identity_engine.py",
             "stage31_identity_fit.py", "stage31_identity_evaluate.py", "stage31_identity_summary.py")
    return {n: file_hash(Path(__file__).parent / n) for n in names}


def load_evidence(directory):
    directory = Path(directory).expanduser().resolve()
    plan, run = load_stage30(directory / "plan.json"), load_stage30(directory / "run.json")
    report = load_stage30(directory / "analysis" / "stage30_response_diagnostic.json")
    inputs = load_stage30_inputs(plan["source_input"])
    validate_stage30_plan(plan, inputs)
    if (run.get("role") != "corrected_branch_single_frame_response_replay" or run.get("plan") != plan
            or report.get("role") != "descriptive_corrected_branch_response_audit"
            or report.get("status") != "COMPLETE_CORRECTED_BRANCH_RESPONSE_AUDIT"
            or report.get("plan_fingerprint") != plan["fingerprint"]
            or report.get("audited_points") != len(plan["points"]) or report.get("missing_points") != []):
        raise ValueError("Need the complete, matched Stage-30 diagnostic")
    hashes = {}
    for point in plan["points"]:
        path = directory / "responses" / point_filename(point)
        load_point(path, point, inputs, run)
        hashes[path.name] = file_hash(path)
    if report.get("cache_sha256") != hashes:
        raise ValueError("Stage-30 response hashes changed")
    return directory, report, inputs["source"]


def build_contract(evidence, reader=read_rgb, expected_sequences=86, expected_frames=69941):
    directory, report, source = load_evidence(evidence)
    root, inventory = scan_dataset(source["manifest"]["root"], expected_sequences, expected_frames)
    excluded = {e["source_group_proxy"] for e in source["manifest"]["sequences"]}
    groups = {}
    for entry in inventory:
        if entry["source_group_proxy"] not in excluded:
            groups.setdefault(entry["source_group_proxy"], []).append(entry)
    needed = DESIGN["fit_videos"] + DESIGN["checking_videos"]
    print("Fresh source groups: {}; need {} ({} fit / {} checking).".format(
        len(groups), needed, DESIGN["fit_videos"], DESIGN["checking_videos"]), flush=True)
    if len(groups) < needed:
        raise ValueError("Only {} fresh groups after excluding {}; frozen 8/6 design needs 14. Do not reuse diagnostic groups.".format(len(groups), len(excluded)))
    rng = np.random.default_rng(DESIGN["seed"])
    names = sorted(groups)
    selected = []
    for ordinal, index in enumerate(rng.permutation(len(names))[:needed]):
        choices = sorted(groups[names[int(index)]], key=lambda e: e["sequence"])
        entry = dict(choices[int(rng.integers(len(choices)))])
        split = "fit" if ordinal < DESIGN["fit_videos"] else "checking"
        frames = []
        for frame in entry["frames"]:
            path = inside(root, frame["file"])
            digest = file_hash(path)
            image = reader(path)
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2 or file_hash(path) != digest:
                raise ValueError("Image audit failed: " + str(path))
            frames.append(dict(frame, sha256=digest, height=int(image.shape[0]), width=int(image.shape[1])))
        gt = read_boxes(inside(root, entry["annotation"]))
        if file_hash(inside(root, entry["annotation"])) != entry["annotation_sha256"]:
            raise ValueError("Annotation changed during manifest audit")
        selected.append(dict(entry, split=split, frames=frames, initial_box=gt[0].tolist()))
        print("[image audit] {} {}: {} full tracked frames".format(split, entry["sequence"], len(frames) - 1), flush=True)
    return seal(dict(schema=SCHEMA, role="frozen_fresh_group_identity_pilot", design=DESIGN, root=str(root),
        evidence_input=str(directory), evidence_fingerprint=report["fingerprint"],
        evidence_sha256=file_hash(directory / "analysis" / "stage30_response_diagnostic.json"),
        source_stage27_fingerprint=source["fingerprint"], runtime=source["runtime"],
        source_code_sha256=source["code_sha256"],
        search_factor=source["search_factor"], search_size=source["search_size"],
        excluded_groups=sorted(excluded), available_fresh_groups=len(groups),
        inventory_fingerprint=fingerprint(inventory), sequences=sorted(selected, key=lambda e: (e["split"], e["sequence"])),
        code_sha256=code_hashes(),
        annotation_use="Plan audits geometry; fit reads fit labels only. Checking labels are evaluation-only, never actor inputs.",
        interpretation="Small development pilot on source-group proxies absent from Stage-27; pretraining overlap not ruled out",
        novelty="Candidate association is a baseline/feasibility test, not a new compression method claim"))


def load_contract(path):
    contract = load_sealed(path)
    if contract.get("role") != "frozen_fresh_group_identity_pilot" or contract.get("design") != DESIGN:
        raise ValueError("Not the frozen Stage-31 design")
    if contract.get("code_sha256") != code_hashes():
        raise ValueError("Stage-31 code changed; preserve the old run and use a new contract")
    for name, digest in contract["source_code_sha256"].items():
        if file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen original SGLA source changed: " + name)
    entries = contract["sequences"]
    names, groups = [e["sequence"] for e in entries], [e["source_group_proxy"] for e in entries]
    if len(set(names)) != len(entries) or len(set(groups)) != len(entries) or set(groups) & set(contract["excluded_groups"]):
        raise ValueError("Duplicate or previously consumed source groups")
    for split, number in (("fit", DESIGN["fit_videos"]), ("checking", DESIGN["checking_videos"])):
        if sum(e["split"] == split for e in entries) != number:
            raise ValueError("Frozen split size changed")
    for e in entries:
        if source_group(e["sequence"]) != e["source_group_proxy"] or len(e["frames"]) < 2:
            raise ValueError("Invalid source-group entry")
        if e["split"] not in ("fit", "checking") or "ground_truth" in e:
            raise ValueError("Full labels must stay out of the training contract")
    evidence = Path(contract["evidence_input"]) / "analysis" / "stage30_response_diagnostic.json"
    if file_hash(evidence) != contract["evidence_sha256"]:
        raise ValueError("Frozen Stage-30 evidence changed")
    return contract


def ground_truth(contract, entry):
    path = inside(contract["root"], entry["annotation"])
    if file_hash(path) != entry["annotation_sha256"]:
        raise ValueError("Frozen annotation changed")
    gt = read_boxes(path)
    if len(gt) != len(entry["frames"]) or not np.array_equal(gt[0], entry["initial_box"]):
        raise ValueError("Annotation coverage/initialization changed")
    return gt


def normalize(values):
    values = np.asarray(values, float)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def roi_weights(boxes, crop, side):
    """Area-weighted pooling on actual crop patches; outside boxes use nearest cell."""
    boxes, crop = np.asarray(boxes, float), np.asarray(crop, float)
    step = crop[2] / side
    left, top = crop[0] + np.arange(side) * step, crop[1] + np.arange(side) * step
    yy, xx = np.meshgrid(top + step / 2, left + step / 2, indexing="ij")
    result = []
    for box in boxes:
        wx = np.maximum(0, np.minimum(left + step, box[0] + box[2]) - np.maximum(left, box[0]))
        wy = np.maximum(0, np.minimum(top + step, box[1] + box[3]) - np.maximum(top, box[1]))
        weight = wy[:, None] * wx[None]
        if weight.sum() == 0:
            center = box[:2] + box[2:] / 2
            weight.flat[int(np.argmin((xx - center[0]) ** 2 + (yy - center[1]) ** 2))] = 1
        result.append((weight.ravel() / weight.sum()).astype(np.float32))
    return np.asarray(result, np.float32)


class IdentityState:
    """All state updates use selected predictions and descriptors, never labels."""
    def __init__(self, initial_box, anchor):
        self.anchor = normalize(anchor)
        self.trusted = self.anchor.copy()
        self.recent = self.anchor.copy()
        self.others = np.empty((0, len(self.anchor)))
        self.box = np.asarray(initial_box, float).copy()
        self.velocity = np.zeros(2)

    def features(self, boxes, descriptors, scores):
        boxes, descriptors = np.asarray(boxes, float), normalize(descriptors)
        center = boxes[:, :2] + boxes[:, 2:] / 2
        prediction = self.box[:2] + self.box[2:] / 2 + self.velocity
        motion = np.linalg.norm((center - prediction) / self.box[2:], axis=1)
        scale = np.abs(np.log(boxes[:, 2:] / self.box[2:])).mean(1)
        recent = descriptors @ self.recent
        other = (descriptors @ self.others.T).max(1) if len(self.others) else np.zeros(len(boxes))
        return np.stack([descriptors @ self.anchor, descriptors @ self.trusted, recent, other,
            recent - other, motion, scale, np.asarray(scores) / max(float(np.max(scores)), 1e-12), scores], axis=1)

    def advance(self, boxes, descriptors, features, chosen):
        descriptor = normalize(descriptors[chosen])
        new_box = np.asarray(boxes[chosen], float)
        displacement = new_box[:2] + new_box[2:] / 2 - self.box[:2] - self.box[2:] / 2
        self.velocity = .5 * self.velocity + .5 * displacement
        self.box, self.recent = new_box.copy(), descriptor.copy()
        if (features[chosen, 0] >= DESIGN["update_initial_cos"]
                and features[chosen, 4] >= DESIGN["update_identity_margin"]):
            eta = DESIGN["trusted_ema"]
            self.trusted = normalize((1 - eta) * self.trusted + eta * descriptor)
        self.others = normalize(np.delete(np.asarray(descriptors), chosen, axis=0))


def select_candidate(method, features, model=None):
    if method == "sgla":
        return 0
    if method == "motion":
        score = -features[:, 5] - DESIGN["motion_scale_weight"] * features[:, 6]
    elif method == "appearance":
        score = features[:, 0]  # Permanent initial identity only; no geometric cues.
    elif method in ("identity", "static_ranker"):
        if model is None:
            raise ValueError("Frozen identity model required")
        prefix = "static_" if method == "static_ranker" else ""
        score = (features / np.asarray(model[prefix + "feature_scale"])) @ np.asarray(model[prefix + "weights"])
    else:
        raise ValueError("Unknown method: " + method)
    return int(np.argmax(score))  # Ties keep the original higher response rank.


def labels_for(boxes, gt):
    overlap = tracking_iou(boxes, gt)
    labels = np.full(len(boxes), -1, np.int64)
    labels[overlap >= .5], labels[overlap < .1] = 1, 0
    return labels


def fit_ranker(records):
    pairs, weights, video_count, frame_count = [], [], 0, 0
    for record in records:
        video = []
        for f, mask, labels in zip(record["features"], record["valid"], record["labels"]):
            pos, neg = np.flatnonzero(mask & (labels == 1)), np.flatnonzero(mask & (labels == 0))
            if len(pos) and len(neg):
                diff = (f[pos, None] - f[neg][None]).reshape(-1, len(FEATURES))
                video.append(diff)
        if video:
            video_count += 1
            frame_count += len(video)
            pairs.extend(video)
            weights.extend([np.full(len(v), 1 / len(video) / len(v)) for v in video])
    if frame_count < DESIGN["minimum_pair_frames"] or video_count < DESIGN["minimum_pair_videos"]:
        raise ValueError("Insufficient fit-only ranking support: {} frames / {} videos; need >=100 / >=4".format(frame_count, video_count))
    x, weight = np.vstack(pairs).astype(float), np.concatenate(weights)
    weight /= weight.sum()
    scale = np.maximum(np.sqrt((weight[:, None] * x ** 2).sum(0)), .001)
    x = x / scale
    coefficient = np.zeros(len(FEATURES))
    penalty = DESIGN["ranking_l2"]
    def loss(w):
        return float(weight @ np.logaddexp(0, -(x @ w)) + .5 * penalty * (w @ w))
    for _ in range(DESIGN["ranking_iterations"]):
        z = x @ coefficient
        probability = 1 / (1 + np.exp(np.clip(z, -60, 60)))
        gradient = -(x.T @ (weight * probability)) + penalty * coefficient
        hessian = x.T @ ((weight * probability * (1 - probability))[:, None] * x) + penalty * np.eye(len(FEATURES))
        step = np.linalg.solve(hessian, gradient)
        rate, before = 1., loss(coefficient)
        while rate > 1e-6 and loss(coefficient - rate * step) > before:
            rate *= .5
        coefficient -= rate * step
        if np.linalg.norm(rate * step) < 1e-8:
            break
    if not np.isfinite(coefficient).all():
        raise ValueError("Nonfinite ranking model")
    return dict(feature_names=list(FEATURES), feature_scale=scale.tolist(), weights=coefficient.tolist(),
        fit_pair_frames=frame_count, fit_pair_videos=video_count, fit_pairs=len(x),
        final_pairwise_loss=loss(coefficient), fit_pair_accuracy=float(np.clip(weight @ ((x @ coefficient) > 0), 0, 1)))
