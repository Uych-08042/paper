"""Compare writeback rules under the fixed Stage-156 low-peak trigger."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np

import stage156_depth_routing_audit as routing


ARM = "adapted_l10"
TRIGGER = "native_hann_peak"
MIN_OVERLAP = .30
SCHEMA = "stage157_depth_writeback_v1"
SOURCE_DIR = Path(__file__).resolve().parents[1]/"docs/experiments/06_sgla_uav_tracking/results"
SOURCES = ("stage76", "stage89", "stage93", "stage155_root")


def protected_choice(effect):
    return (effect["tail32"] > 0 and not effect["lost_good_run"]
            and not effect["new_low_run"] and not effect["lost_anchor_good"])


def choices(peaks, overlap):
    increased = bool(peaks[1] > peaks[0])
    return dict(peak_gain=increased, peak_gain_overlap=increased and overlap >= MIN_OVERLAP)


def reports(points, trigger, observed=None):
    """Every veto consumes the same extra block as direct writeback."""
    selected = {p["case_id"] for p, active in zip(points, trigger, strict=True) if active}
    if observed is not None and set(observed) != selected:
        raise ValueError("Need a paired exposure for every fixed trigger")
    policies = ["native", "direct_write", "protected_hindsight"]
    if observed is not None:
        policies += ["peak_gain", "peak_gain_overlap"]
    summary, cases = [], []
    for policy in policies:
        use = []
        for p, active in zip(points, trigger, strict=True):
            cid = p["case_id"]
            effect = p["outcomes"][ARM]
            evidence = observed[cid] if observed is not None and active else None
            decision = False
            if active:
                if policy == "direct_write":
                    decision = True
                elif policy == "protected_hindsight":
                    decision = protected_choice(effect)
                elif policy in ("peak_gain", "peak_gain_overlap"):
                    decision = choices(evidence["peaks"], evidence["overlap"])[policy]
            use.append(float(decision))
            cases.append(dict(case_id=cid, sequence=p["sequence"], frame=p["frame"], kind=p["kind"],
                              fold=p["fold"], policy=policy, fixed_trigger=bool(active), write_l10=decision,
                              native_hann_peak=float(evidence["peaks"][0]) if evidence else None,
                              serial_hann_peak=float(evidence["peaks"][1]) if evidence else None,
                              box_overlap=evidence["overlap"] if evidence else None, **effect))
        rows = routing.score(points, ARM, policy, use, TRIGGER)
        for row in rows:
            active = sum(t for p, t in zip(points, trigger) if row["kind"] in ("all", p["kind"]))
            row["l10_writes"] = int(row.pop("extra_blocks"))
            row["extra_blocks"] = 0 if policy == "native" else int(active)
            row["extra_head_calls"] = row["extra_blocks"]
        summary.extend(rows)
    direct = {r["kind"]:r for r in summary if r["policy"] == "direct_write"}
    for row in summary:
        row["tail32_delta_vs_direct"] = row["tail32"]-direct[row["kind"]]["tail32"]
    return {"summary":summary, "cases":cases}


def expose(native, case, entry, parent, l10, root, stage, reader):
    """Run one crop and a shared L1-L6/L9 path, then run both heads before writeback."""
    rt = native.rt
    first = case["frame"]
    frames = entry["frames"]
    template = reader(root, frames[0])
    native.initialize(template, entry["initial_box"])
    image = reader(root, frames[first])
    rt.tracker.state = parent.tolist()
    rt.tracker.frame_id = first-1
    native.blocks[:] = 0
    native.heads = native.selectors = 0
    crop_hashes = []
    sample = rt.s1.sample_target

    def capture(*args, **kwargs):
        result = sample(*args, **kwargs)
        crop_hashes.append(sha256(np.ascontiguousarray(result[0]).tobytes()).hexdigest())
        return result

    with ExitStack() as stack:
        stack.enter_context(rt.torch.no_grad())
        stack.enter_context(patch.object(rt.s1, "sample_target", capture))
        stack.callback(l10.register_forward_hook(native._block(9)).remove)
        search, factor = rt.prepare(image)
        tokens = rt.prefix(search)
        layer = rt.selected_layer(rt.selector(tokens))
        if layer != 9:
            raise ValueError(f"Stage-157 expects L9 on the fixed roster: {case['case_id']}")
        tokens = rt.backbone.blocks[layer-1](tokens)
        outputs = (rt.head(tokens), rt.head(l10(tokens)))
        boxes = np.array([rt.decode(output, factor, image) for output in outputs], float)
        peaks = np.array([float((rt.tracker.output_window*output["score_map"]).max().item()) for output in outputs])
    counts = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(native.blocks, counts)
    np.testing.assert_array_equal(rt.tracker.state, parent)
    if (len(crop_hashes) != 1 or native.heads != 2 or native.selectors != 1
            or not np.isfinite(peaks).all() or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any()):
        raise ValueError("Invalid paired depth exposure")
    # Split accounting by route for the existing trajectory audit. Both routes share the real prefix above.
    native_counts = counts.copy()
    native_counts[9] = 0
    paths = [[dict(boxes=box, parent=parent, selected=layer, added=added,
                   block_calls=blocks, crop_calls=1, head_calls=1, selector_calls=1)]
             for box, added, blocks in zip(boxes, (0, 10), (native_counts, counts), strict=True)]
    ids = np.arange(first, first+case["successors"]+1)
    with ExitStack() as stack:
        observer = stage.prior.DepthObserver(native, stack)
        observer.initialize(template, entry["initial_box"], "native")
        for frame in ids[1:]:
            image = reader(root, frames[frame])
            for path in paths:
                path.append(observer.step(image, path[-1]["boxes"], int(frame)))
    data = {k:np.asarray([[row[k] for row in path] for path in paths]) for k in stage.prior.FIELDS}
    return dict(data, frame_id=ids, anchor_crop_sha256=np.array(crop_hashes*2), peaks=peaks,
                anchor_calls=np.array([1, 2, 1, int(counts.sum())]))


def load_model(stage, native, root, report, fold):
    from stage48_stability_experiment import load_torch

    l10 = stage.isolated_l10(native.rt)
    name = f"training/fold{fold}/step{stage.DESIGN['updates']:06d}.pt"
    path = root/name
    stage.io.verify({str(path):report["artifact_sha256"][name]})
    checkpoint = load_torch(native.rt.torch, path)
    if checkpoint["metadata"]["fold"] != fold or checkpoint["metadata"]["plan_fingerprint"] != report["plan_fingerprint"]:
        raise ValueError("L10 checkpoint provenance changed")
    l10.load_state_dict(checkpoint["l10"], strict=True)
    return l10.requires_grad_(False)


def paired_effect(stage, data, reference, case, entry, same_runtime, atol):
    np.testing.assert_array_equal(data["anchor_calls"], [1, 2, 1, 8])
    np.testing.assert_array_equal(data["added"][:, 0], [0, 10])
    stage.prior.audit(data, reference=reference, case=case, same_runtime=same_runtime, atol=atol)
    replay_error = float(np.abs(data["boxes"]-reference["boxes"]).max())
    if same_runtime:
        np.testing.assert_allclose(data["boxes"], reference["boxes"], atol=atol, rtol=0)
    anchor, w, _ = stage.prior.describe(data, case, entry["ground_truth"], replay_error)
    if not all(r["valid_successors"] == r["horizon"] for r in w):
        raise ValueError("Incomplete paired-writeback successors")
    r = w[-1]
    return dict(anchor_delta=anchor["anchor_delta"], new_anchor_good=anchor["new_good"],
                lost_anchor_good=anchor["lost_good"], tail8=w[0]["tail_mean_delta"], tail32=r["tail_mean_delta"],
                new_good_run=r["serial_extra_first_good_run"] is not None and r["native_first_good_run"] is None,
                lost_good_run=r["native_first_good_run"] is not None and r["serial_extra_first_good_run"] is None,
                new_low_run=r["serial_extra_first_low_run"] is not None and r["native_first_low_run"] is None)


def replay(args, points, trigger, root):
    import stage155_adapt_serial_l10 as stage
    from stage28_response_common import continuous_iou
    import torch

    c, io, base = stage.c, stage.io, stage.base
    r76, r89, r93, r155 = [getattr(args, key).resolve() for key in SOURCES]
    if any(root == p or p in root.parents or root in p.parents for p in (r76,r89,r93,r155)):
        raise ValueError("Use a separate Stage-157 output directory")
    stage.check_results(r155)
    source = c.c66.sealed(r155/"manifest.json", stage.SCHEMA)
    report = c.c66.sealed(r155/"stage155_decision.json", stage.SCHEMA)
    if report["fingerprint"] != routing.FINGERPRINTS[155]:
        raise ValueError("Need the archived Stage-155 source")
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    ap = base.e63.c.actor_plan(old)
    specs = {s["unit"]["sequence"]:s for s in p89["units"]}
    cases = {r["case_id"]:r for r in source["cases"]}
    folds = {g:r["fold"] for r in source["folds"] for g in r["held"]}
    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    diffs = base.s109.check_runtime(source["runtime"], runtime)
    active = [p for p, t in zip(points, trigger, strict=True) if t]
    plan = io.seal(dict(schema=SCHEMA, trigger=TRIGGER, minimum_overlap=MIN_OVERLAP,
                        triggered=[p["case_id"] for p in active], runtime=runtime, runtime_differences=diffs,
                        source155=report["fingerprint"], code_sha256={p.name:c.file_hash(p)
                            for p in (Path(__file__), Path(routing.__file__))}))
    c.publish_identical(root/"manifest.json", plan)
    frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    args.parameter = runtime["tracker"]["parameter"]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    native,_ = base.s103.s76.startup.checked_runtime(args, frozen, root)
    models, observed, artifacts = {}, {}, {}
    for p in active:
        cid, seq, fold = p["case_id"], p["sequence"], p["fold"]
        if folds[seq.split("_", 1)[0]] != fold:
            raise ValueError("Held-out fold changed")
        if fold not in models:
            models[fold] = load_model(stage, native, r155, report, fold)
        l10 = models[fold]
        model_digest = stage.digest(l10)
        name = f"branches/{cid}.npz"
        meta = dict(plan_fingerprint=plan["fingerprint"], case_id=cid, adapted_l10_sha256=model_digest)
        previous = c.read_cache(r155/name, dict(plan_fingerprint=source["fingerprint"], case_id=cid,
                                                fold=fold, adapted_l10_sha256=model_digest))
        if previous is None:
            raise ValueError(f"Missing Stage-155 branch for {cid}")
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, specs[seq])[0]
        case = cases[cid]
        path = root/name
        path.parent.mkdir(exist_ok=True)
        data = c.read_cache(path, meta)
        fresh = data is None
        if fresh:
            data = expose(native, case, entry, previous["parent"][0, 0], l10, ap["root"], stage, stage.checked_image)
        reference = stage.pair(previous, 0, 2)
        reference.update(layer=reference["selected"], crop_sha256=reference["anchor_crop_sha256"][:, None])
        p["outcomes"][ARM] = paired_effect(stage, data, reference, case, entry, not diffs, ap["design"]["box_atol"])
        if not diffs:
            np.testing.assert_allclose(data["peaks"][0], p["native_hann_peak"], atol=1e-6, rtol=0)
        first_boxes = data["boxes"][:, 0]
        observed[cid] = dict(peaks=data["peaks"], overlap=float(continuous_iou(first_boxes[0], first_boxes[1:])[0]))
        if stage.digest(l10) != model_digest:
            raise ValueError("Stage-155 L10 copy changed during evaluation")
        if fresh:
            c.save(path, data, meta)
        for item in (path, path.with_suffix(".receipt.json")):
            artifacts[item.relative_to(root).as_posix()] = c.file_hash(item)
        print(f"[Stage-157] {cid} Hann={data['peaks']} IoU={observed[cid]['overlap']:.3f}", flush=True)
    native.assert_frozen()
    n = len(active)
    return observed, dict(plan_fingerprint=plan["fingerprint"], runtime=runtime, runtime_differences=diffs,
                          evaluated_anchors=n, successor_searches=sum(2*cases[p["case_id"]]["successors"] for p in active),
                          paired_anchor_head_calls=2*n, paired_anchor_block_calls=8*n, artifact_sha256=artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage137", type=Path, default=SOURCE_DIR/"stage137_results_20261119T123230Z_433433.txt")
    parser.add_argument("--stage155", type=Path, default=SOURCE_DIR/"stage155_results_20261120T134722Z_730065.txt")
    for name in SOURCES:
        parser.add_argument("--"+name.replace("_", "-"), type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cached-bound", action="store_true", help="Run the protected reference with CSV bundles only")
    parser.set_defaults(mode="stage157_depth_writeback")
    args = parser.parse_args()
    points = routing.matched_points(args.stage137, args.stage155)
    trigger, _, _ = routing.routed(points, TRIGGER)
    if sum(trigger) != 16 or sum(v for p,v in zip(points,trigger) if p["kind"] == "onset") != 14:
        raise ValueError("Expected the frozen Stage-156 trigger roster")
    root = args.output_dir.expanduser().resolve()
    import stage155_adapt_serial_l10 as stage
    from stage27_uav_collect import writer_lock
    from stage46_shift_audit import publish_table

    with writer_lock(root):
        status = "COMPLETE_STAGE157_CACHED_BOUND" if args.cached_bound else "COMPLETE_STAGE157_MATCHED_WRITEBACK"
        decision = root/"stage157_decision.json"
        if decision.exists():
            existing = stage.c.c66.sealed(decision, SCHEMA)
            if existing["status"] != status:
                raise ValueError("Use separate directories for cached bound and paired writeback")
            stage.io.verify({str(root/k):v for k,v in (existing["table_sha256"]|existing["artifact_sha256"]).items()})
            print(f"Stage-157: {status}")
            return
        if args.cached_bound:
            observed, scope = None, dict(evaluated_anchors=0, successor_searches=0, paired_anchor_head_calls=0,
                                         paired_anchor_block_calls=0, artifact_sha256={})
        else:
            if args.checkpoint is None or any(getattr(args, n) is None for n in SOURCES):
                parser.error("Paired writeback requires the Stage-76/89/93/155 directories and original checkpoint")
            observed, scope = replay(args, points, trigger, root)
        tables = {}
        for name, rows in reports(points, trigger, observed).items():
            path = root/f"stage157_{name}.csv"
            publish_table(path, rows)
            tables[path.name] = stage.c.file_hash(path)
        stage.c.publish_identical(decision, stage.io.seal(dict(schema=SCHEMA, status=status, trigger=TRIGGER,
            cases=len(points), triggered=int(sum(trigger)), source155=routing.FINGERPRINTS[155],
            table_sha256=tables, new_full_trajectories=0, training_updates=0, **scope)))
        print(f"Stage-157: {status}")


if __name__ == "__main__":
    main()
