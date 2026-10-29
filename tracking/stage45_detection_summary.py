"""Official XS-VID pooled AP, cached paired video bootstrap and error diagnostics."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import importlib.util
import io
import math
from pathlib import Path
import sys
import types

import stage45_detection_common as c


def official_evaluator(upstream):
    """Load the pinned scorer and its real mask dependency without importing the detector."""
    root = Path(c.verify_upstream(upstream)) / "ultralytics/data"
    package_name = "stage45_xsvid_evaluation"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".cocoeval_xs_vid", root / "cocoeval_xs_vid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COCOeval


def build_evaluator(dataset, predictions, evaluator_class):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(dataset)
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = dict(images=copy.deepcopy(dataset["images"]),
                              categories=copy.deepcopy(dataset["categories"]), annotations=[])
            dt.createIndex()
        ev = evaluator_class(gt, dt, "bbox")
        # Preserve every original category and annotation-level crowd/ignore flag,
        # exactly as the official score_detection.py does (including ignore slot 4).
        ev.evaluate()
        ev.accumulate()
    return ev


def ap(ev, area="all"):
    values = ev.eval["precision"][:, :, :, ev.params.areaRngLbl.index(area), ev.params.maxDets.index(100)]
    values = values[values > -1]
    return 100*float(values.mean()) if values.size else None


def cached_sample(ev, image_ids):
    """Duplicate complete per-image evaluations, then use the unchanged official accumulator.

    Matching is image-local. Repeated copies count as separate images/positives,
    so this recomputes pooled AP of a video bootstrap, not an average of video APs.
    """
    sampled = copy.copy(ev)
    original = ev._paramsEval
    positions = {int(x): i for i, x in enumerate(original.imgIds)}
    indices = [positions[int(x)] for x in image_ids]
    n, areas = len(original.imgIds), len(original.areaRng)
    # Restrict accumulation to the two predeclared endpoints to keep bootstrap inexpensive.
    selected = [original.areaRngLbl.index(a) for a in ("all", "0-12")]
    sampled.evalImgs = [ev.evalImgs[k*areas*n+a*n+i]
                        for k in range(len(original.catIds)) for a in selected for i in indices]
    p = copy.deepcopy(original)
    p.imgIds = list(range(len(image_ids)))
    p.areaRng, p.areaRngLbl = [original.areaRng[a] for a in selected], ["all", "0-12"]
    p.maxDets = [100]
    sampled.params, sampled._paramsEval = p, copy.deepcopy(p)
    with contextlib.redirect_stdout(io.StringIO()):
        sampled.accumulate()
    return {"AP": ap(sampled), "AP_es": ap(sampled, "0-12")}


def paired_bootstrap(evaluators, videos, samples=None):
    import numpy as np
    samples = c.DESIGN["bootstrap_samples"] if samples is None else samples
    rng = np.random.default_rng(c.DESIGN["bootstrap_seed"])
    draws = []
    for i in range(samples):
        selected = rng.integers(0, len(videos), len(videos))
        images = [f["id"] for index in selected for f in videos[int(index)]["frames"]]
        draws.append({arm: cached_sample(ev, images) for arm, ev in evaluators.items()})
        if (i+1) % 32 == 0:
            print("[paired video bootstrap] {}/{}".format(i+1, samples), flush=True)
    rows = []
    for left, right in (("signals", "capacity"), ("signals", "native"), ("capacity", "native")):
        for name, area in (("AP", "all"), ("AP_es", "0-12")):
            values = [d[left][name]-d[right][name] for d in draws
                      if d[left][name] is not None and d[right][name] is not None]
            a, b = ap(evaluators[left], area), ap(evaluators[right], area)
            rows.append(dict(left=left, right=right, metric=name, videos=len(videos),
                delta=None if a is None or b is None else a-b,
                ci_low=float(np.percentile(values, 2.5)) if values else None,
                ci_high=float(np.percentile(values, 97.5)) if values else None,
                bootstrap_draws=samples, supported_draws=len(values)))
    return rows


def overlap(box, other):
    x, y, w, h = box
    u, v, a, b = other
    intersection = max(0., min(x+w, u+a)-max(x, u)) * max(0., min(y+h, v+b)-max(y, v))
    return intersection / max(1e-12, w*h+a*b-intersection), intersection/max(1e-12, w*h)


def miss_diagnostics(video, records, annotations):
    """Supplementary score=.25/IoU=.5 matching; not the official AP/ignore protocol."""
    tp = fp = count = tiny = tiny_tp = long_misses = 0
    runs, lengths = {}, []
    durations, gains = [], []
    identities_available = True
    for frame_index, (frame, record) in enumerate(zip(video["frames"], records)):
        gt = [a for a in annotations[frame["id"]] if a["category_id"] != 4 and
              not a.get("ignore", 0) and not a.get("iscrowd", 0)]
        ignore = [a["bbox"] for a in annotations[frame["id"]] if a["category_id"] == 4 or
                  a.get("ignore", 0) or a.get("iscrowd", 0)]
        used = set()
        for p in sorted(record["boxes"], key=lambda b: -b[4]):
            if p[4] < .25 or int(p[5]) == 4:
                continue
            box = [p[0], p[1], p[2]-p[0], p[3]-p[1]]
            candidates = [(overlap(box, a["bbox"])[0], j) for j, a in enumerate(gt)
                          if j not in used and a["category_id"] == int(p[5])]
            best = max(candidates, default=(0., -1))
            if best[0] >= .5:
                used.add(best[1])
                tp += 1
            elif not any(overlap(box, b)[1] >= .5 for b in ignore):
                fp += 1
        active = set()
        for j, a in enumerate(gt):
            count += 1
            if a["bbox"][2]*a["bbox"][3] <= 144:
                tiny += 1
                tiny_tp += j in used
            if "track_id" not in a:
                identities_available = False
                continue
            key = (a["category_id"], a["track_id"])
            active.add(key)
            if j in used:
                if runs.get(key, 0):
                    lengths.append(runs[key])
                runs[key] = 0
            else:
                runs[key] = runs.get(key, 0) + 1
        for key in list(runs):
            if key not in active:
                if runs[key]:
                    lengths.append(runs[key])
                del runs[key]
        if frame_index >= c.DESIGN["smoke_frames"]:
            durations.append(record["ms"])
        gains.extend(g["changed_fraction"] for g in record["gains"])
    lengths.extend(x for x in runs.values() if x)
    if identities_available:
        long_misses = sum(x for x in lengths if x >= 5)
    return dict(video_id=video["video_id"], frames=len(records), true_positives=tp, false_positives=fp,
                ground_truth=count, recall=tp/count if count else None,
                tiny_ground_truth=tiny, tiny_recall=tiny_tp/tiny if tiny else None,
                track_ids_available=identities_available,
                longest_miss_frames=max(lengths, default=0) if identities_available else None,
                miss_frames_in_runs_ge5=long_misses if identities_available else None,
                mean_gate_changed_fraction=sum(gains)/len(gains) if gains else 0.,
                timed_frames=len(durations), durations=durations)


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    import numpy as np
    root = Path(root)
    plan = c.load_plan(root)
    run = c.read_sealed(root / "evaluation/run.json")
    if run["plan"] != plan["fingerprint"]:
        raise ValueError("Mixed evaluation plan")
    if set(run["models"]) != {"capacity", "signals"}:
        raise ValueError("Need both frozen gate models")
    for arm, expected in run["models"].items():
        path = root / "training" / arm / ("step{:04d}.pt".format(c.DESIGN["updates"]))
        if c.digest(path) != expected:
            raise ValueError("Evaluated gate checkpoint changed")
    full_gt = c.read_json(plan["annotations"]["checking"]["path"])
    _, annotations = c.annotation_index(full_gt)
    videos = plan["videos"]["checking"]
    image_ids = {f["id"] for v in videos for f in v["frames"]}
    gt = dict(info={}, categories=full_gt["categories"],
              images=[i for i in full_gt["images"] if i["id"] in image_ids],
              annotations=[a for a in full_gt["annotations"] if a["image_id"] in image_ids])
    scorer = official_evaluator(plan["upstream"])
    evaluators, policies, video_rows, evidence = {}, [], [], {}
    for arm in c.ARMS:
        predictions, diagnostics = [], []
        for video in videos:
            meta = dict(run=run["fingerprint"], arm=arm, video_id=video["video_id"])
            path = c.result_path(root, arm, video)
            result = c.committed(path, meta)
            if result is None:
                raise ValueError("Need all 24 complete video results before analysis: " + str(path))
            records = result["frames"]
            if result["meta"] != meta or len(records) != len(video["frames"]):
                raise ValueError("Mixed/incomplete video")
            evidence[str(path.relative_to(root))] = c.digest(path)
            for f, record in zip(video["frames"], records):
                if record["image_id"] != f["id"] or record["frame_id"] != f["frame_id"] or record["head_calls"] != 1:
                    raise ValueError("Unordered image IDs or hidden extra detector calls")
                for p in record["boxes"]:
                    if len(p) != 6 or not all(math.isfinite(x) for x in p) or not 0 <= p[4] <= 1 or int(p[5]) != p[5] or not 0 <= p[5] <= 6:
                        raise ValueError("Malformed prediction")
                    if p[2] <= p[0] or p[3] <= p[1]:
                        raise ValueError("Degenerate predicted box")
                    predictions.append(dict(image_id=f["id"], category_id=int(p[5]), score=round(p[4], 5),
                        bbox=[round(p[0], 3), round(p[1], 3), round(p[2]-p[0], 3), round(p[3]-p[1], 3)]))
            detail = miss_diagnostics(video, records, annotations)
            diagnostics.append(detail)
            video_rows.append(dict(arm=arm, **{k: v for k, v in detail.items() if k != "durations"}))
        ev = build_evaluator(gt, predictions, scorer)
        evaluators[arm] = ev
        for row, video in zip(video_rows[-len(videos):], videos):
            diagnostic = cached_sample(ev, [f["id"] for f in video["frames"]])
            row.update(video_AP_diagnostic=diagnostic["AP"], video_AP_es_diagnostic=diagnostic["AP_es"])
        times = [t for d in diagnostics for t in d["durations"]]
        policies.append(dict(arm=arm, videos=len(videos), AP=ap(ev), AP_es=ap(ev, "0-12"),
            AP_rs=ap(ev, "12-20"), AP_gs=ap(ev, "20-32"),
            recall=sum(d["true_positives"] for d in diagnostics)/max(1, sum(d["ground_truth"] for d in diagnostics)),
            false_positives=sum(d["false_positives"] for d in diagnostics),
            median_network_nms_ms=float(np.median(times)) if times else None,
            p95_network_nms_ms=float(np.percentile(times, 95)) if times else None))
        print("{} AP={} AP_es={}".format(arm, policies[-1]["AP"], policies[-1]["AP_es"]), flush=True)
    paired = paired_bootstrap(evaluators, videos)
    primary = next(r for r in paired if (r["left"], r["right"], r["metric"]) == ("signals", "capacity", "AP_es"))
    overall = next(r for r in paired if (r["left"], r["right"], r["metric"]) == ("signals", "capacity", "AP"))
    native = next(r for r in paired if (r["left"], r["right"], r["metric"]) == ("signals", "native", "AP"))
    tiny_support = [r for r in video_rows if r["arm"] == "native" and r["tiny_ground_truth"] > 0]
    support = len(tiny_support) >= 4 and sum(r["tiny_ground_truth"] for r in tiny_support) >= 200
    promising = (support and primary["delta"] is not None and primary["delta"] >= 1. and
                 overall["delta"] >= -.3 and native["delta"] >= -.3)
    decision = c.seal(dict(schema=c.SCHEMA, status="PROMISING_DEVELOPMENT_SIGNAL" if promising else "REVIEW_DEVELOPMENT_SIGNAL_RESULT",
        plan=plan["fingerprint"], run=run["fingerprint"], primary=primary, overall=overall,
        practice_target="AP_es +1 point vs equal-capacity control; overall AP >= -0.3 vs control and native",
        tiny_support=support, tiny_supported_videos=len(tiny_support),
        tiny_gt_instances=sum(r["tiny_ground_truth"] for r in tiny_support),
        seed=c.DESIGN["seed"], new_models=2, full_trajectories=len(videos)*len(c.ARMS),
        evidence_sha256=evidence, tables=["stage45_policy_summary.csv", "stage45_video_summary.csv", "stage45_paired.csv"],
        test_evaluated=False, external_generalization_established=False,
        next_step="Review video effects, continuous misses and gate use; develop the signal design or plan an independent UAV check before tuning on checking outcomes",
        warnings=["Pretrained detector saw source training data; val_dev is developmental, not unseen-pretraining evidence.",
                  "Eight video IDs are source proxies; shared original captures are not excluded by metadata alone.",
                  "One seed and 256 paired video bootstrap draws; intervals condition on the frozen two trained gates.",
                  "AP is pooled with the pinned official scorer; bootstrap duplicates complete image evaluations before official accumulation.",
                  "Score-.25 recall/miss matching uses its documented supplemental ignore-overlap rule; it is not the AP protocol.",
                  "Age saturates at six frames; consistency is colocated cosine, not optical-flow alignment quality.",
                  "Three signals are tested together; no separate signal causality or UAV-specific novelty is established.",
                  "Timing includes synchronized FP32 network, gate feature extraction and NMS; excludes image loading/conversion, logging and first eight frames of each video. No deployment speed claim."]))
    folder = root / "analysis"
    folder.mkdir(exist_ok=True)
    csv_write(folder / "stage45_policy_summary.csv", policies)
    csv_write(folder / "stage45_video_summary.csv", video_rows)
    csv_write(folder / "stage45_paired.csv", paired)
    c.atomic_json(folder / "stage45_decision.json", decision)
    print("Stage-45:", decision["status"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    summarize(p.parse_args().input)
