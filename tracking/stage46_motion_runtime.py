"""Run the original tracker.track; instrument its actual block/selector/head calls."""
from __future__ import annotations

import hashlib
import time

import numpy as np

from stage46_motion_common import (DESIGN, MotionState, audit_video, checked_image,
                                   estimate_camera, fingerprint, gray_image)


def state_digest(net):
    result = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class NativeObserver:
    def __init__(self, rt):
        self.rt = rt
        self.handles = []
        self.blocks, self.heads, self.selectors = np.zeros(12,int), 0, 0
        self.peak = None
        for index, block in enumerate(rt.net.backbone.blocks):
            self.handles.append(block.register_forward_hook(self._block(index)))
        self.handles.append(rt.net.box_head.register_forward_hook(self._head))
        self.handles.append(rt.net.backbone.MLP.register_forward_hook(self._selector))
        self.handles.append(rt.net.register_forward_hook(self._output))
        self.digest = state_digest(rt.net)

    def _block(self, index):
        def hook(module, args, output):
            self.blocks[index] += 1
        return hook

    def _head(self, module, args, output):
        self.heads += 1

    def _selector(self, module, args, output):
        self.selectors += 1

    def _output(self, module, args, output):
        self.peak = float(output["score_map"].detach().amax().item())

    def initialize(self, image, box):
        self.rt.initialize(image, box)

    def synchronize(self):
        self.rt.torch.cuda.synchronize()

    def step(self, image, parent):
        self.blocks[:] = 0
        self.heads = self.selectors = 0
        self.peak = None
        self.rt.tracker.state = np.asarray(parent, float).tolist()
        self.synchronize()
        start = time.perf_counter()
        # This is the original public tracking path, including native selector,
        # Hann response, CENTER decode, image clipping and frame counter.
        with self.rt.torch.no_grad():
            box = self.rt.tracker.track(image)["target_bbox"]
        self.synchronize()
        elapsed = 1000*(time.perf_counter()-start)
        chosen = np.flatnonzero(self.blocks[6:])
        if (self.heads != 1 or self.selectors != 1 or len(chosen) != 1 or
            not np.array_equal(self.blocks[:6], np.ones(6,int)) or self.blocks[6:].sum() != 1):
            raise ValueError("Not one native SGLA call: six prefix blocks + one selected candidate + one head")
        if self.peak is None:
            raise ValueError("Missing original response observation")
        box = np.asarray(box, float)
        np.testing.assert_array_equal(box, self.rt.tracker.state)
        return dict(box=box, layer=int(chosen[0])+7, peak=self.peak,
                    block_calls=self.blocks.copy(), head_calls=self.heads,
                    selector_calls=self.selectors, network_ms=elapsed)

    def assert_frozen(self):
        if state_digest(self.rt.net) != self.digest:
            raise ValueError("Frozen SGLA checkpoint or BN state changed")


def run_video(observer, dataset_root, frames, initial_box, sequence, method,
              image_reader=checked_image, estimator=estimate_camera):
    """Actor inputs contain image records and the initial box; no future GT API."""
    first = image_reader(dataset_root, frames[0])
    observer.initialize(first, initial_box)
    state = MotionState(initial_box)
    old_gray = gray_image(first)
    rows = []
    key = int(fingerprint([DESIGN["seed"], sequence])[:8], 16)
    for frame_id, frame in enumerate(frames[1:], 1):
        image = image_reader(dataset_root, frame)
        observer.synchronize()
        start = time.perf_counter()
        new_gray = gray_image(image)
        motion = estimator(old_gray, new_gray, state.previous.copy(), image.shape[:2], key+frame_id)
        motion_ms = 1000*(time.perf_counter()-start)
        policy = state.propose(method, motion["matrix"], motion["valid"], image.shape[:2])
        result = observer.step(image, policy["parent"])
        state_before = state.previous.copy()
        state.advance(result["box"], motion["matrix"], motion["valid"])
        observer.synchronize()
        elapsed = 1000*(time.perf_counter()-start)
        rows.append(dict(frame_id=frame_id, previous=state_before, parents=policy["parent"], boxes=result["box"],
            image_hw=image.shape[:2], layer=result["layer"], peak=result["peak"], matrix=motion["matrix"],
            motion_valid=motion["valid"], motion_reason=motion["reason"], motion_stats=motion["stats"],
            **{k:policy[k] for k in ("raw_velocity", "relative_velocity", "proposal", "shift", "cap_hit", "camera_displacement")},
            motion_ms=motion_ms, network_ms=result["network_ms"], pipeline_ms=elapsed,
            block_calls=result["block_calls"], head_calls=result["head_calls"], selector_calls=result["selector_calls"]))
        old_gray = new_gray
        if frame_id % 250 == 0:
            print("[{}] {} {}/{}".format(method, sequence, frame_id, len(frames)-1), flush=True)
    data = {k:np.asarray([row[k] for row in rows]) for k in rows[0]}
    for k in ("frame_id", "image_hw", "layer", "motion_reason", "block_calls", "head_calls", "selector_calls"):
        data[k] = data[k].astype(np.int64)
    for k in ("motion_valid", "cap_hit"):
        data[k] = data[k].astype(bool)
    audit_video(data, frames, initial_box, method)
    observer.assert_frozen()
    return data
