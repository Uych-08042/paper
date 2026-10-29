"""Stage-47: prepare, smoke and collect independent native-SGLA search interventions."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, nonempty_path, publish_identical
from stage28_response_common import cpu_array
import stage47_search_common as c


class Capture:
    """Observe the original public tracker; do not recompute or replace its output."""
    def __init__(self, native):
        self.native, self.rt, self.maps = native, native.rt, None
        self.handle = self.rt.net.register_forward_hook(self._output)

    def _output(self, module, args, output):
        self.maps = {key: cpu_array(output[key]).astype(np.float32) for key in ("score_map", "size_map", "offset_map")}

    def initialize(self, image, box):
        self.native.initialize(image, box)

    def capture(self, image, parent, frame):
        tracker = self.rt.tracker
        namespace = tracker.track.__func__.__globals__
        original = namespace["sample_target"]
        self.maps, sampling = None, []
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            np.testing.assert_array_equal(args[1], parent)
            patch_image, factor, mask = result
            sampling.append(dict(factor=float(factor),
                crop_sha256=hashlib.sha256(np.ascontiguousarray(patch_image).tobytes()).hexdigest(),
                mask_sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()))
            return result
        tracker.frame_id = frame - 1
        with patch.dict(namespace, sample_target=observed):
            result = self.native.step(image, parent)
        if self.maps is None or len(sampling) != 1 or tracker.frame_id != frame:
            raise ValueError("Need one original sample/forward/decode at the requested frame")
        side = self.rt.search_side
        return dict(boxes=result["box"], layer=result["layer"], peak=result["peak"],
            raw=self.maps["score_map"].reshape(side, side), size=self.maps["size_map"].reshape(2, side, side),
            offset=self.maps["offset_map"].reshape(2, side, side),
            block_calls=result["block_calls"], head_calls=result["head_calls"], selector_calls=result["selector_calls"],
            **sampling[0])

    def assert_frozen(self):
        self.native.assert_frozen()


def prepare(source, root):
    source, root = Path(source).expanduser().resolve(), Path(root).expanduser().resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("Use a separate new Stage-47 output directory")
    plan, _ = c.load_source(source)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-47 directory")
        publish_identical(root / "manifest.json", plan)
    print("Frozen Stage-47: {} cases / {} current-frame forward calls; no continuation".format(len(plan["events"]), plan["expected_head_calls"]))
    print("Plan: " + plan["fingerprint"])


def collect_case(observer, event, snapshots, frames, dataset_root, initial_box, search_factor, search_size, image_reader=checked_image):
    """No GT labels are passed to the replay actor; snapshots are saved own-history states."""
    observer.initialize(image_reader(dataset_root, frames[0]), initial_box)
    results, hws = [], []
    for frame in event["frames"]:
        image = image_reader(dataset_root, frames[frame])
        hws.append(image.shape[:2])
        histories = []
        for history in c.HISTORIES:
            parents = c.proposals(snapshots[history], frame - 1, image.shape[:2])
            treatments = []
            for parent in parents:
                result = observer.capture(image, parent, frame)
                crop = crop_geometry(parent, image.shape[:2], search_factor, search_size)[0]
                treatments.append(dict(parents=parent, crop=crop, **result))
            histories.append(treatments)
        results.append(histories)
    keys = results[0][0][0]
    data = {key: np.asarray([[[r[key] for r in treatments] for treatments in histories] for histories in results]) for key in keys}
    data.update(frame_id=np.asarray(event["frames"], int), image_hw=np.asarray(hws, int))
    observer.assert_frozen()
    return data


def execute(root, plan, inputs, observer, identity, window, smoke=False, case_id=None):
    window = np.asarray(window, np.float32)
    if identity != inputs["old_run"]["runtime"]:
        raise ValueError("Need the exact frozen Stage-46 runtime and model state")
    run = c.seal(dict(schema=c.SCHEMA, role="independent_current_frame_search_replay", plan_fingerprint=plan["fingerprint"],
        runtime=identity, window=window.tolist(), histories=list(c.HISTORIES), treatments=list(c.TREATMENTS)))
    publish_identical(root / "run.json", run)
    if not smoke:
        record = c.read_sealed(root / "smoke/stage47_smoke_decision.json")
        if record.get("run_fingerprint") != run["fingerprint"] or record.get("status") != "PASS_MATCHED_SEARCH_REPLAY_SMOKE":
            raise ValueError("Need a passing Stage-47 smoke for this exact run")
    events = plan["events"][:1] if smoke else plan["events"]
    if case_id:
        events = [e for e in events if e["case_id"] == case_id]
        if not events:
            raise ValueError("Case is outside the fixed collection selection")
    integrity = []
    for event in events:
        entry = inputs["traces"][event["sequence"]][0]
        path = c.path_for(root, event)
        data = c.read_cache(path, c.metadata(run, event))
        if data is None:
            data = collect_case(observer, event, {h: inputs["cached"][event["sequence"], h] for h in c.HISTORIES},
                entry["frames"], inputs["source"]["manifest"]["root"], entry["initial_box"],
                inputs["source"]["search_factor"], inputs["source"]["search_size"])
            audit = c.audit_response(data, event, inputs, window)
            c.save(path, data, c.metadata(run, event))
        else:
            audit = c.audit_response(data, event, inputs, window)
        integrity.append(audit)
        print("[done, audited] {}: {} replays; source box max error {:.6g}".format(event["case_id"], audit["head_calls"], audit["maximum_recorded_box_error"]), flush=True)
    observer.assert_frozen()
    c.prior.verify_files(plan["source_file_sha256"])
    if smoke:
        publish_identical(root / "smoke/stage47_smoke_decision.json", c.seal(dict(schema=c.SCHEMA,
            status="PASS_MATCHED_SEARCH_REPLAY_SMOKE", role="execution_only_not_method_gain", run_fingerprint=run["fingerprint"],
            case_id=events[0]["case_id"], integrity=integrity[0], training_performed=False, continuation_performed=False)))
        print("Stage-47 smoke PASS; first case reused by the full collection")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("prepare", "smoke", "collect"))
    p.add_argument("--input", required=True, type=nonempty_path)
    p.add_argument("--output_dir", type=nonempty_path)
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    p.add_argument("--parameter", default="deit_distilled")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--case_id", choices=("E01", "E02", "E03", "E04"))
    args = p.parse_args()
    if args.mode == "prepare":
        if not args.output_dir:
            p.error("prepare needs --output_dir")
        prepare(args.input, args.output_dir)
        return
    root = Path(args.input).expanduser().resolve()
    plan, inputs = c.load_plan(root)
    if not args.checkpoint or not str(args.checkpoint).strip():
        p.error("Use --checkpoint or CHECKPOINT with original SGLA weights")
    from stage46_motion_experiment import make_runtime
    with writer_lock(root):
        native, identity = make_runtime(args, inputs["source"])
        observer = Capture(native)
        window = cpu_array(native.rt.tracker.output_window).reshape(native.rt.search_side, native.rt.search_side)
        execute(root, plan, inputs, observer, identity, window, args.mode == "smoke", args.case_id)
    print("Next: python tracking/stage47_search_summary.py --input \"{}\"".format(root))


if __name__ == "__main__":
    main()
