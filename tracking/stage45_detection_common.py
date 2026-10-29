"""Stage-45: a self-contained XS-VID development pilot, separate from frozen SOT runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

SCHEMA = "stage45_xsvid_temporal_signal_pilot_v1"
UPSTREAM_COMMIT = "9e51ac319baf0d3793694e770e049585648bae42"
WEIGHT_SHA256 = "214a4fcb277ac527ee7c83718fc28d7dac8464f7fb82bb8a6794049206a1caa4"
NAMES = ["bicycle-static", "bicycle-person", "car", "person", "ignore", "truck", "boat"]
ARMS = ("native", "capacity", "signals")
DESIGN = dict(seed=2125, fit_videos=8, checking_videos=8, updates=128,
              clip_frames=8, burn_in=2, save_every=16, image_size=1024,
              lr=0.001, weight_decay=0., grad_clip=1., hidden=16,
              confidence=0.001, nms_iou=0.7, max_det=300,
              history_box_confidence=0.25, age_cap=6, smoke_frames=8,
              bootstrap_samples=256, bootstrap_seed=2126)
FILES = ("stage45_detection_common.py", "stage45_temporal_gate.py",
         "stage45_detection_runtime.py", "stage45_detection_experiment.py",
         "stage45_detection_summary.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seal(value):
    value = dict(value)
    value["fingerprint"] = fingerprint(value)
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_sealed(path):
    value = read_json(path)
    raw = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("fingerprint") != fingerprint(raw) or value.get("schema") != SCHEMA:
        raise ValueError("Invalid Stage-45 sealed record: " + str(path))
    return value


def atomic_json(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError("Completed Stage-45 artifact changed: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def code_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in FILES}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_upstream(root):
    root = Path(root).resolve()
    if git(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("Use pinned YOLOFT commit " + UPSTREAM_COMMIT)
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("YOLOFT tracked sources changed; use a clean separate checkout")
    return str(root)


def safe_image(root, name):
    name = str(name).replace("\\", "/")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or ":" in name:
        raise ValueError("Unsafe image path: " + name)
    root = Path(root).resolve()
    paths = {q.resolve() for q in (root / name, root / "images" / name) if q.is_file()}
    if len(paths) != 1:
        raise FileNotFoundError("Missing/ambiguous release image: " + name)
    path = paths.pop()
    if root not in path.parents:
        raise ValueError("Image escapes dataset root")
    return path


def annotation_index(data):
    categories = {c["id"]: c["name"] for c in data["categories"]}
    if categories != dict(enumerate(NAMES)):
        raise ValueError("Expected canonical XS-VID seven original class slots, including ignore=4")
    images = {i["id"]: i for i in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("Duplicate image IDs")
    annotations = {k: [] for k in images}
    ids = set()
    for a in data["annotations"]:
        box = a["bbox"]
        if a["id"] in ids or a["image_id"] not in images or a["category_id"] not in categories:
            raise ValueError("Invalid annotation identity")
        if len(box) != 4 or not all(isinstance(v, (int, float)) and abs(v) < 1e8 for v in box) or min(box[2:]) <= 0:
            raise ValueError("Invalid annotation box")
        ids.add(a["id"])
        annotations[a["image_id"]].append(a)
    return images, annotations


def grouped(data):
    images, _ = annotation_index(data)
    groups = {}
    for image in images.values():
        if any(k not in image for k in ("video_id", "file_name", "width", "height")) or not any(k in image for k in ("frame_index", "frame_id")):
            raise ValueError("Need real video_id/frame_id metadata")
        image = dict(image, frame_id=image.get("frame_index", image.get("frame_id")))
        if not str(image["video_id"]).isdecimal() or not isinstance(image["frame_id"], int):
            raise ValueError("Expected canonical numeric video IDs and integer frame indices")
        if image["width"] != DESIGN["image_size"] or image["height"] != DESIGN["image_size"]:
            raise ValueError("Stage-45 uses canonical 1024-square media, without resize or tiling")
        groups.setdefault(str(image["video_id"]), []).append(image)
    for frames in groups.values():
        frames.sort(key=lambda i: i["frame_id"])
        numbers = [i["frame_id"] for i in frames]
        if len(set(numbers)) != len(numbers) or any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
            raise ValueError("Pilot requires contiguous, unique video frame indices")
    return groups


def choose(groups, count, tag):
    candidates = [k for k, v in groups.items() if len(v) >= DESIGN["clip_frames"]]
    candidates.sort(key=lambda k: fingerprint([DESIGN["seed"], tag, k]))
    if len(candidates) < count:
        raise ValueError("Not enough complete video groups for " + tag)
    return candidates[:count]


def prepare(data_root, upstream, weights, output):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    upstream = verify_upstream(upstream)
    weights = Path(weights).resolve()
    if digest(weights) != WEIGHT_SHA256:
        raise ValueError("Need official tensor-only VID release vid.pt, not SGLA/MOT/SOT weights")
    if output == data_root or data_root in output.parents or output in data_root.parents:
        raise ValueError("Use a separate Stage-45 output directory")
    paths = {s: data_root / "annotations" / n for s, n in
             (("fit", "train.json"), ("checking", "val_dev.json"))}
    raw = {s: read_json(p) for s, p in paths.items()}
    groups = {s: grouped(v) for s, v in raw.items()}
    checking_names = {i["file_name"] for i in raw["checking"]["images"]}
    checking_ids = set(groups["checking"])
    fit = {k: v for k, v in groups["fit"].items()
           if k not in checking_ids and not any(i["file_name"] in checking_names for i in v)}
    chosen = {"fit": choose(fit, DESIGN["fit_videos"], "fit"),
              "checking": choose(groups["checking"], DESIGN["checking_videos"], "checking")}
    videos = {}
    seen_hashes = {}
    for split in chosen:
        videos[split] = []
        for vid in chosen[split]:
            frames = []
            for item in groups[split][vid]:
                path = safe_image(data_root, item["file_name"])
                sha = digest(path)
                if sha in seen_hashes and seen_hashes[sha] != (split, vid):
                    raise ValueError("Duplicate image bytes across selected video groups")
                seen_hashes[sha] = (split, vid)
                frames.append(dict(item, path=str(path), sha256=sha))
            videos[split].append(dict(video_id=vid, frames=frames))
    plan = seal(dict(schema=SCHEMA, role="pretrained_development_pilot_not_external_validation",
                     data_root=str(data_root), upstream=upstream, upstream_commit=UPSTREAM_COMMIT,
                     weights=str(weights), weights_sha256=WEIGHT_SHA256, design=DESIGN,
                     code_sha256=code_hashes(), annotations={s: dict(path=str(p), sha256=digest(p))
                                                             for s, p in paths.items()},
                     videos=videos, checkpoint_training_overlap="Released detector saw XS-VID training data; val_dev derives from that partition. No unseen-pretraining claim.",
                     grouping="video_id proxy; independent original captures/scenes not established",
                     test_annotations_opened=False))
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    atomic_json(output / "manifest.json", plan, immutable=True)
    print("Stage-45: 8 fit / 8 checking complete video proxies; one seed; 2 x 128 gate-only updates")
    print("Development with pretrained overlap; official test not opened. Plan: " + plan["fingerprint"])
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json")
    if plan["design"] != DESIGN or plan["code_sha256"] != code_hashes():
        raise ValueError("Frozen Stage-45 code/design changed")
    verify_upstream(plan["upstream"])
    for item in [*plan["annotations"].values(), dict(path=plan["weights"], sha256=plan["weights_sha256"])]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Frozen Stage-45 source changed: " + item["path"])
    return plan


def schedule(plan, step):
    key = int(fingerprint([DESIGN["seed"], step])[:16], 16)
    videos = plan["videos"]["fit"]
    video = videos[key % len(videos)]
    start = (key // len(videos)) % (len(video["frames"]) - DESIGN["clip_frames"] + 1)
    return video, video["frames"][start:start + DESIGN["clip_frames"]]


def result_path(root, arm, video):
    return Path(root) / "evaluation" / arm / ("video_" + video["video_id"] + ".json")


def committed(path, meta):
    path = Path(path)
    receipt = path.with_suffix(".receipt.json")
    if not path.exists():
        if receipt.exists():
            raise ValueError("Result missing behind receipt: " + str(path))
        return None
    if not receipt.exists():
        return None  # interrupted, uncommitted video is recomputed from frame zero
    record = read_sealed(receipt)
    if record["meta"] != meta or record["sha256"] != digest(path):
        raise ValueError("Completed Stage-45 result changed: " + str(path))
    return read_json(path)


def commit_result(path, value, meta):
    atomic_json(path, value)
    atomic_json(Path(path).with_suffix(".receipt.json"),
                seal(dict(schema=SCHEMA, meta=meta, sha256=digest(path))), immutable=True)
