"""Single native SGLA forward plus pooled candidate descriptors and causal association."""

import json
import time

import numpy as np

from stage27_uav_common import checked_image, crop_geometry, tracking_iou
from stage28_response_common import bind_native_top1, cpu_array, dense_boxes, peak_candidates
from stage31_identity_common import (DESIGN, FEATURES, IdentityState, ground_truth, labels_for,
                                      normalize, roi_weights, select_candidate)


class NativeObserver:
    def __init__(self, runtime):
        self.rt, self.identity = runtime, runtime.identity

    def initialize(self, image, box):
        self.rt.initialize(image, box)

    def set_state(self, box, frame):
        self.rt.tracker.state, self.rt.tracker.frame_id = np.asarray(box, float).tolist(), int(frame)

    def synchronize(self):
        self.rt.torch.cuda.synchronize()

    def _forward(self, image, features, verify=False):
        rt = self.rt
        previous = np.asarray(rt.tracker.state, float).copy()
        with rt.torch.no_grad():
            search, factor = rt.prepare(image)
            tokens = rt.prefix(search)
            layer = rt.selected_layer(rt.selector(tokens))
            if features:
                chosen = rt.backbone.blocks[layer - 1](tokens.clone())
                normalized = rt.backbone.norm(rt.s1.recover_tokens(chosen, rt.lens_z, rt.lens_x,
                                                                   mode=rt.backbone.cat_mode))
                output = rt.net.forward_head(normalized, None)
                grid = normalized[0, rt.lens_z:]
            else:
                output, grid = rt.candidate(tokens, layer), None
            if verify:
                rt.verify(search, output)
            native = np.asarray(rt.decode(output, factor, image), float)
            peak = float(output["score_map"].detach().amax().item())
        np.testing.assert_array_equal(rt.tracker.state, previous)
        crop, _, expected_factor = crop_geometry(previous, image.shape[:2], rt.tracker.params.search_factor,
                                                rt.tracker.params.search_size)
        np.testing.assert_allclose(factor, expected_factor, atol=1e-12, rtol=0)
        return output, grid, dict(native_box=native, factor=factor, layer=layer, raw_score_peak=peak, crop=crop)

    def _pool(self, grid, boxes, crop):
        rt = self.rt
        weights = roi_weights(boxes, crop, rt.search_side)
        with rt.torch.no_grad():
            weights = rt.torch.as_tensor(weights, device=grid.device, dtype=grid.dtype)
            result = weights @ grid
        return normalize(cpu_array(result))

    def initial_anchor(self, image, box):
        _, grid, packet = self._forward(image, True)
        return self._pool(grid, np.asarray(box)[None], packet["crop"])[0]

    def observe(self, image, features):
        output, grid, packet = self._forward(image, features, verify=self.rt.tracker.frame_id == 0)
        if features:
            side = self.rt.search_side
            raw = cpu_array(output["score_map"]).reshape(side, side)
            window = cpu_array(self.rt.tracker.output_window).reshape(side, side)
            boxes = dense_boxes(cpu_array(output["size_map"]).reshape(2, side, side),
                cpu_array(output["offset_map"]).reshape(2, side, side), self.rt.tracker.state,
                packet["factor"], self.rt.tracker.params.search_size, image.shape[:2])
            boxes = bind_native_top1(raw, window, boxes, packet["native_box"])
            cells = peak_candidates(raw * window, boxes)
            packet.update(boxes=boxes[cells], scores=(raw * window).ravel()[cells],
                          descriptors=self._pool(grid, boxes[cells], packet["crop"]))
        else:
            packet.update(boxes=packet["native_box"][None], scores=np.array([packet["raw_score_peak"]]),
                          descriptors=np.zeros((1, self.identity["embed_dim"])))
        return packet


def run_video(observer, contract, entry, method, model=None, fit_collect=False,
              image_reader=checked_image, labels=None):
    """Labels are logged after the predicted state has been written. Actor APIs never accept GT."""
    root = contract["root"]
    gt = ground_truth(contract, entry) if labels is None else np.asarray(labels, float)
    n, k, dim = len(entry["frames"]) - 1, DESIGN["max_candidates"], observer.identity["embed_dim"]
    initial = image_reader(root, entry["frames"][0])
    observer.initialize(initial, entry["initial_box"])
    need_features = fit_collect or method != "sgla"
    anchor = observer.initial_anchor(initial, entry["initial_box"]) if need_features else np.zeros(dim)
    state = IdentityState(entry["initial_box"], anchor) if need_features else None
    data = dict(frame_id=np.arange(1, n + 1, dtype=np.int64), previous=np.zeros((n, 4)), boxes=np.zeros((n, 4)),
        native_boxes=np.zeros((n, 4)), candidate_boxes=np.zeros((n, k, 4)), descriptors=np.zeros((n, k, dim)),
        scores=np.zeros((n, k)), features=np.zeros((n, k, len(FEATURES))), valid=np.zeros((n, k), bool),
        chosen=np.zeros(n, np.int64), layer=np.zeros(n, np.int64), factor=np.zeros(n), raw_score_peak=np.zeros(n),
        image_hw=np.zeros((n, 2), np.int64), elapsed_ms=np.zeros(n), anchor=np.asarray(anchor, float),
        gt=gt[1:].copy(), iou=np.zeros(n), labels=np.full((n, k), -1, np.int64))
    previous = np.asarray(entry["initial_box"], float).copy()
    for row, frame in enumerate(range(1, n + 1)):
        image = image_reader(root, entry["frames"][frame])
        observer.synchronize()
        start = time.perf_counter()
        packet = observer.observe(image, need_features)
        boxes, descriptors, scores = packet["boxes"], packet["descriptors"], packet["scores"]
        features = state.features(boxes, descriptors, scores) if need_features else np.zeros((1, len(FEATURES)))
        chosen = select_candidate(method, features, model)
        box = boxes[chosen].copy()
        observer.set_state(box, frame)
        if need_features:
            state.advance(boxes, descriptors, features, chosen)
        observer.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        count = len(boxes)
        if not 1 <= count <= k:
            raise ValueError("Candidate count outside the frozen top-5 design")
        data["previous"][row], data["boxes"][row] = previous, box
        data["native_boxes"][row] = packet["native_box"]
        data["candidate_boxes"][row] = packet["native_box"]  # Finite padding, excluded by valid.
        for key, value in (("candidate_boxes", boxes), ("descriptors", descriptors), ("scores", scores), ("features", features)):
            data[key][row, :count] = value
        data["valid"][row, :count] = True
        data["chosen"][row] = chosen
        for key in ("layer", "factor", "raw_score_peak"):
            data[key][row] = packet[key]
        data["image_hw"][row], data["elapsed_ms"][row] = image.shape[:2], elapsed
        # No GT-derived label or error is passed to the observer, scorer, prototype or crop parent.
        data["labels"][row, :count] = labels_for(boxes, gt[frame])
        data["iou"][row] = tracking_iou(box, gt[frame])
        previous = box
        if frame % 250 == 0:
            print("[{}] {} {}/{}".format(method, entry["sequence"], frame, n), flush=True)
    return data


def audit_video(data, contract, entry, method, model=None, fit_collect=False, labels=None):
    n, k, dim = len(entry["frames"]) - 1, DESIGN["max_candidates"], contract["runtime"]["embed_dim"]
    shapes = dict(frame_id=(n,), previous=(n, 4), boxes=(n, 4), native_boxes=(n, 4), candidate_boxes=(n, k, 4),
        descriptors=(n, k, dim), scores=(n, k), features=(n, k, len(FEATURES)), valid=(n, k), chosen=(n,), layer=(n,),
        factor=(n,), raw_score_peak=(n,), image_hw=(n, 2), elapsed_ms=(n,), anchor=(dim,), gt=(n, 4), iou=(n,), labels=(n, k))
    if set(data) != set(shapes) or any(np.shape(data[key]) != shape for key, shape in shapes.items()):
        raise ValueError("Video cache shape/coverage mismatch")
    if not all(np.isfinite(a).all() for a in data.values()):
        raise ValueError("Nonfinite video cache")
    for key in ("frame_id", "chosen", "layer", "image_hw", "labels"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("Expected integer " + key)
    if data["valid"].dtype != np.bool_:
        raise ValueError("Candidate validity mask must be boolean")
    gt = ground_truth(contract, entry) if labels is None else np.asarray(labels, float)
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, n + 1))
    np.testing.assert_array_equal(data["gt"], gt[1:])
    np.testing.assert_array_equal(data["previous"], np.vstack([entry["initial_box"], data["boxes"][:-1]]))
    np.testing.assert_array_equal(data["native_boxes"], data["candidate_boxes"][:, 0])
    np.testing.assert_allclose(data["iou"], tracking_iou(data["boxes"], gt[1:]), atol=1e-12, rtol=0)
    if (data["boxes"][:, 2:] <= 0).any() or not np.isin(data["layer"], np.arange(7, 13)).all():
        raise ValueError("Invalid predicted box or native selected layer")
    need_features = fit_collect or method != "sgla"
    state = IdentityState(entry["initial_box"], data["anchor"]) if need_features else None
    if need_features:
        np.testing.assert_allclose(np.linalg.norm(data["anchor"]), 1., atol=1e-6, rtol=0)
    for i in range(n):
        count = int(data["valid"][i].sum())
        if not 1 <= count <= k or not np.array_equal(data["valid"][i], np.arange(k) < count):
            raise ValueError("Non-prefix candidate mask")
        boxes, desc, scores = data["candidate_boxes"][i, :count], data["descriptors"][i, :count], data["scores"][i, :count]
        features = state.features(boxes, desc, scores) if need_features else np.zeros((1, len(FEATURES)))
        if need_features:
            np.testing.assert_allclose(np.linalg.norm(desc, axis=1), 1., atol=1e-6, rtol=0)
        np.testing.assert_allclose(data["features"][i, :count], features, atol=1e-9, rtol=0)
        choice = select_candidate(method, features, model)
        np.testing.assert_equal(data["chosen"][i], choice)
        np.testing.assert_array_equal(data["boxes"][i], boxes[choice])
        expected = np.full(k, -1, np.int64)
        expected[:count] = labels_for(boxes, gt[i + 1])
        np.testing.assert_array_equal(data["labels"][i], expected)
        if need_features:
            state.advance(boxes, desc, features, choice)
        else:
            np.testing.assert_equal(count, 1)
        hw = [entry["frames"][i + 1]["height"], entry["frames"][i + 1]["width"]]
        np.testing.assert_array_equal(data["image_hw"][i], hw)
        _, _, factor = crop_geometry(data["previous"][i], hw, contract["search_factor"], contract["search_size"])
        np.testing.assert_equal(data["factor"][i], factor)


def cache_metadata(run, entry, method):
    return dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"], split=entry["split"], method=method)


def load_video(path, contract, entry, method, run, model=None, fit_collect=False):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    if meta != cache_metadata(run, entry, method):
        raise ValueError("Video cache identity mismatch")
    audit_video(data, contract, entry, method, model, fit_collect)
    return data
