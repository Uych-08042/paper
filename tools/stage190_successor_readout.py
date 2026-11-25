"""Replay the first failed successor and separate crop, grid and readout evidence."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage189_continuation_audit as prior
from stage27_uav_collect import writer_lock
from stage27_uav_common import inside, read_boxes, scene_geometry
from stage28_response_common import dense_boxes


old, c, io, online = prior.prior, prior.c, prior.io, prior.online
SCHEMA = "stage190_successor_readout_v1"
SOURCE189 = "6fd02d8652fd71a15952e2a5eee677bf081966a810286bd5b2652db4271d25a0"
PRIMARY = "uav0000173_00781_s__onset__f000502"
CONTROL = "uav0000107_01763_s__onset__f000161"
BRANCHES = ("native", "opposite_center", "observed", "opposite_pixels")
GOOD = online.TRACKING["recovery_iou"]
BOX_ATOL = old.source.BOX_ATOL


class FixedRoute(old.crop_stage.center.CenterObserver):
    def route(self, module, args, output):
        super().route(module, args, output)
        if self.chosen == self.fixed_layer:
            return None
        self.chosen = self.fixed_layer
        self.switched = True
        forced = self.rt.torch.zeros_like(output)
        forced[:, old.source.LAYERS.index(self.chosen)] = 1
        return forced


def jobs(root, dataset, manifest, decision, report_root, report):
    clips = {r["sequence"]: r for r in manifest["clips"]}
    receipts = old.risk.read_verified(report_root, report, 189, "branches")
    receipts = {(r["case_id"], r["branch"]): r for r in receipts}
    result = []
    for key, anchor, frame, names in ((PRIMARY, 502, 503, BRANCHES),
                                      (CONTROL, 161, 161, ("current", "opposite_center"))):
        seq = key.split("__", 1)[0]
        clip = clips[seq]
        annotation = inside(dataset, clip["annotation"])
        if c.file_hash(annotation) != clip["annotation_sha256"]:
            raise ValueError(f"Stage-188 annotation changed at {seq}")
        gt = read_boxes(annotation)
        if len(gt) != len(clip["frames"]) or len(gt) != clip["tracked_frames"] + 1:
            raise ValueError("Stage-188 image and annotation counts differ")
        trace = prior.read_cache(root, decision, "native", seq)
        old.previous.audit(trace, clip["tracked_frames"], gt[0], "native", 0.)
        probe = prior.read_cache(root, decision, "probe", key)
        probe_names = list(probe["names"])
        if key == PRIMARY:
            tail = prior.read_cache(root, decision, "tail", key)
            paths = tail["paths"][tail["inverse"]]
            np.testing.assert_array_equal(paths[:, 0], probe["boxes"])
            refs = {name: paths[probe_names.index(name), :2] for name in names[1:]}
            refs["native"] = trace["boxes"][anchor - 1:frame]
            inputs = [(name, *refs[name]) for name in names]
            layer = int(trace["layer"][frame - 1])
        else:
            inputs = [(name, probe["parent"][probe_names.index(name)], probe["boxes"][probe_names.index(name)])
                      for name in names]
            layer = int(old.geometry_stage.LAYER)
            np.testing.assert_array_equal(probe["layer"], [layer] * len(probe_names))
        for name, parent, ref in inputs:
            receipt = receipts[key, name]
            output_field = "successor1_iou" if frame > anchor else "anchor_iou"
            np.testing.assert_allclose(online.tracking_iou(ref, gt[frame]), float(receipt[output_field]), atol=1e-12, rtol=0.)
            if key == PRIMARY and name != "native" and online.tracking_iou(parent, gt[anchor]) < GOOD:
                raise ValueError("Expected a good archived writeback on the anchor")
        result.append(dict(key=key, sequence=seq, frame=frame, layer=layer, inputs=inputs, gt=gt,
                           initialize=clip["frames"][0], image=clip["frames"][frame]))
    return result


def measure(data, target):
    raw, window = data["raw"], data["window"]
    hann = (raw * window).ravel()
    if raw.shape != window.shape or not np.isfinite(hann).all():
        raise ValueError("Invalid CENTER response")
    geo = scene_geometry(data["parent"], target, data["image_hw"], data["search_factor"], data["search_size"])
    np.testing.assert_array_equal(geo["crop"], data["crop"])
    np.testing.assert_allclose(geo["factor"], data["factor"], atol=1e-12, rtol=0.)
    boxes = dense_boxes(data["size"], data["offset"], data["parent"], geo["factor"], data["search_size"], data["image_hw"])
    selected, raw_top = int(hann.argmax()), int(raw.argmax())
    np.testing.assert_allclose(boxes[selected], data["boxes"], atol=BOX_ATOL, rtol=0.)
    np.testing.assert_allclose(hann[selected], data["hann_peak"], atol=1e-6, rtol=0.)
    boxes[selected] = data["boxes"]
    iou = online.tracking_iou(boxes, target)
    best = int(iou.argmax())
    is_good = iou >= GOOD
    order = np.argsort(-hann, kind="stable")
    good_ranks = np.flatnonzero(is_good[order]) + 1
    left, top, side, _ = map(int, geo["crop"])
    visible = geo["gt_image_fraction"]
    coverage = float(np.clip(geo["coverage"] / visible, 0., 1.)) if visible else None
    return dict(crop_left=left, crop_top=top, crop_side=side, coverage_visible=coverage,
                gt_image_fraction=float(visible), center_inside=geo["center_in_usable"],
                input_short_side_px=float(min(geo["input_wh"])), selected_cell=selected,
                selected_iou=float(iou[selected]), hann_peak=float(hann[selected]), raw_cell=raw_top,
                raw_top_iou=float(iou[raw_top]), best_grid_cell=best, best_grid_iou=float(iou[best]),
                good_grid_cells=int(is_good.sum()), first_good_hann_rank=int(good_ranks[0]) if len(good_ranks) else None,
                readout_miss=bool(is_good.any() and not is_good[selected]),
                raw_good_hann_bad=bool(is_good[raw_top] and not is_good[selected]))


def evaluate(tracker, dataset, items, exact_runtime, progress):
    rows = []
    used_calls = 0
    for job in items:
        f = job["frame"]
        gt = job["gt"]
        template = old.checked_frame(dataset, job["initialize"])
        image = old.checked_frame(dataset, job["image"])
        for name, parent, ref in job["inputs"]:
            with ExitStack() as stack:
                if job["key"] == PRIMARY:
                    natural = old.crop_stage.center.CenterObserver(tracker, stack)
                    data = natural.observe(template, image, gt[0], parent, f)
                else:
                    fixed = FixedRoute(tracker, stack)
                    fixed.fixed_layer = job["layer"]
                    data = fixed.observe(template, image, gt[0], parent, f)
                used_calls += 1
            error = float(np.max(np.abs(data["boxes"] - ref)))
            if exact_runtime and (error > BOX_ATOL or (name == "native" and data["layer"] != job["layer"])):
                raise ValueError(f"Archived output did not replay at {job['key']}/{name}: {error}")
            archive_layer = int(data["layer"])
            progress.advance()
            if archive_layer != job["layer"]:
                with ExitStack() as stack:
                    fixed = FixedRoute(tracker, stack)
                    fixed.fixed_layer = job["layer"]
                    data = fixed.observe(template, image, gt[0], parent, f)
                    used_calls += 1
            if job["key"] == PRIMARY:
                progress.advance()
            np.testing.assert_equal(data["layer"], job["layer"])
            np.testing.assert_array_equal(data["parent"], parent)
            if data["head_calls"] != 1 or data["crop_calls"] != 1:
                raise ValueError("Expected one search and one head for every observed output")
            data["search_factor"] = tracker.rt.tracker.params.search_factor
            rows.append(dict(sequence=job["sequence"], frame=f, branch=name,
                             replayed_source_layer=archive_layer, fixed_layer=job["layer"],
                             archive_box_max_abs_error=error, archived_box_reproduced=error <= BOX_ATOL,
                             archived_iou=float(online.tracking_iou(ref, gt[f])),
                             parent_x=float(parent[0]), parent_y=float(parent[1]), parent_w=float(parent[2]), parent_h=float(parent[3]),
                             **measure(data, gt[f])))
    return rows, used_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("stage188-root", "stage189-root", "dataset-root", "checkpoint", "output-dir"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.set_defaults(mode="stage190_successor_readout")
    args = parser.parse_args()
    a, b, dataset, target = (p.expanduser().resolve() for p in (args.stage188_root, args.stage189_root, args.dataset_root, args.output_dir))
    if any(target == p or target in p.parents or p in target.parents for p in (a, b, dataset)):
        parser.error("Use a separate Stage-190 result directory")
    decision = c.c66.sealed(a / "stage188_decision.json", old.SCHEMA)
    manifest = c.c66.sealed(a / "manifest.json", old.SCHEMA)
    report = c.c66.sealed(b / "stage189_decision.json", prior.SCHEMA)
    if (decision["fingerprint"] != prior.SOURCE188 or report["fingerprint"] != SOURCE189
            or report["source_stage188"] != decision["fingerprint"] or report["source_plan_fingerprint"] != manifest["fingerprint"]
            or decision["plan_fingerprint"] != manifest["fingerprint"] or decision["runtime"] != manifest["runtime"]):
        raise ValueError("Need the frozen Stage-188 and Stage-189 runs")

    with writer_lock(target):
        path = target / "stage190_decision.json"
        if path.exists():
            done = c.c66.sealed(path, SCHEMA)
            if done["source_stage189"] != SOURCE189:
                raise ValueError("Existing result is bound to another input")
            io.verify({str(target / name): digest for name, digest in done["table_sha256"].items()})
            return
        items = jobs(a, dataset, manifest, decision, b, report)

        import torch

        torch.set_num_threads(1)
        runtime_source = manifest["runtime"]
        runtime = old.base.s103.s76.device_runtime(dict(runtime_identity=runtime_source), torch, args.gpu)
        differences = old.base.s109.check_runtime(runtime_source, runtime)
        plan = io.seal(dict(schema=SCHEMA, source_stage188=prior.SOURCE188, source_stage189=SOURCE189,
                            runtime=runtime, runtime_differences=differences, audit_sha256=c.file_hash(Path(__file__))))
        c.publish_identical(target / "manifest.json", plan)
        args.parameter = runtime["tracker"]["parameter"]
        progress = OverallProgress("Stage-190", len(BRANCHES) * 2 + 2 + 1, unit="work")
        progress.report("Loading frozen SGLA")
        tracker, _ = old.base.s103.s76.startup.checked_runtime(args, dict(fingerprint=plan["fingerprint"],
                                                                 runtime_identity=runtime, runtime=runtime["tracker"]), target)
        progress.report("Replaying source boxes and measuring candidate supply")
        rows, calls = evaluate(tracker, dataset, items, not differences, progress)
        tracker.assert_frozen()
        hashes = old.gain.prior.publish(target, dict(arms=rows), stage="stage190")
        c.publish_identical(path, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE190_SUCCESSOR_READOUT_AUDIT",
                                             source_stage189=SOURCE189, plan_fingerprint=plan["fingerprint"],
                                             diagnostic_rows=len(rows), search_calls=calls, training_performed=False,
                                             runtime_differences=differences, table_sha256=hashes)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
