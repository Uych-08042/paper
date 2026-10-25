"""Fresh-group split, GT-free state, pairwise fit, full-video and gate tests."""

import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import crop_geometry, seal
from stage30_branch_response_common import point_filename, point_metadata
from stage31_identity_common import (DESIGN, FEATURES, METHODS, IdentityState, build_contract,
    fit_ranker, ground_truth, load_contract, roi_weights, select_candidate)
from stage31_identity_engine import NativeObserver, audit_video, cache_metadata, run_video
from stage31_identity_fit import load_model
from stage31_identity_summary import decide, video_metrics
from test_stage27_uav_diagnostic import fake_image
from test_stage30_branch_response import point_fixture


HERE = Path(__file__).resolve().parent
TORCH = importlib.util.find_spec("torch") is not None


class FakeObserver:
    def __init__(self, identity):
        self.identity = identity

    def initialize(self, image, box):
        self.box, self.frame = np.asarray(box, float).copy(), 0

    def initial_anchor(self, image, box):
        return np.array([1., 0.])

    def synchronize(self):
        pass

    def set_state(self, box, frame):
        self.box, self.frame = np.asarray(box).copy(), frame

    def observe(self, image, features):
        boxes = np.array([[20., 20, 10, 10], [40., 20, 10, 10]])
        if not features:
            boxes = boxes[:1]
        _, _, factor = crop_geometry(self.box, image.shape[:2], 4, 256)
        return dict(native_box=boxes[0].copy(), boxes=boxes, descriptors=np.eye(2)[[1, 0]][:len(boxes)] if features else np.zeros((1, 2)),
                    scores=np.array([.9, .8])[:len(boxes)], factor=factor, layer=9, raw_score_peak=.9)


def fixture(root):
    inputs, plan, run, caches, _ = point_fixture(root)
    evidence = root / "stage30"
    write_json(evidence / "plan.json", plan)
    write_json(evidence / "run.json", run)
    hashes = {}
    for point in plan["points"]:
        path = evidence / "responses" / point_filename(point)
        write_cache(path, caches[path.name], point_metadata(run, point))
        hashes[path.name] = file_hash(path)
    write_json(evidence / "analysis" / "stage30_response_diagnostic.json", seal(dict(schema=plan["schema"],
        role="descriptive_corrected_branch_response_audit", status="COMPLETE_CORRECTED_BRANCH_RESPONSE_AUDIT",
        plan_fingerprint=plan["fingerprint"], audited_points=6, missing_points=[], cache_sha256=hashes)))
    dataset = Path(inputs["source"]["manifest"]["root"])
    for video in range(4, 18):
        name = "uav{:07d}_00000_s".format(video)
        directory = dataset / "sequences" / name
        directory.mkdir(parents=True)
        for frame in range(48):
            (directory / ("img{:07d}.jpg".format(frame + 1))).write_bytes(bytes([frame + 1]) * 12)
        (dataset / "annotations" / (name + ".txt")).write_text("\n".join(["40,20,10,10"] * 48))
    with redirect_stdout(io.StringIO()):
        contract = build_contract(evidence, reader=fake_image, expected_sequences=17, expected_frames=816)
    body = {k: v for k, v in contract.items() if k != "fingerprint"}
    body["runtime"] = dict(body["runtime"], embed_dim=2)  # Explicit synthetic descriptor dimension.
    contract = seal(body)
    output = root / "stage31"
    write_json(output / "manifest.json", contract)
    from stage27_uav_common import checked_image
    reader = lambda directory, frame: checked_image(directory, frame, reader=fake_image)
    fit_run = seal(dict(schema=contract["schema"], role="native_fit_candidate_features", contract_fingerprint=contract["fingerprint"],
                        runtime=contract["runtime"], trajectory="synthetic native SGLA"))
    write_json(output / "fit_run.json", fit_run)
    records = []
    for entry in contract["sequences"]:
        if entry["split"] == "fit":
            data = run_video(FakeObserver(contract["runtime"]), contract, entry, "sgla", fit_collect=True, image_reader=reader)
            audit_video(data, contract, entry, "sgla", fit_collect=True)
            write_cache(output / "fit_cache" / (entry["sequence"] + ".npz"), data, cache_metadata(fit_run, entry, "sgla"))
            records.append(data)
    cmd = [sys.executable, "-B", str(HERE / "stage31_identity_fit.py"), "--input", str(output), "--mode", "train"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    model = load_model(output, contract)
    checking_run = seal(dict(schema=contract["schema"], role="frozen_full_video_identity_checking",
        contract_fingerprint=contract["fingerprint"], model_fingerprint=model["fingerprint"], methods=list(METHODS)))
    write_json(output / "checking_run.json", checking_run)
    for entry in contract["sequences"]:
        if entry["split"] == "checking":
            for method in METHODS:
                data = run_video(FakeObserver(contract["runtime"]), contract, entry, method, model, image_reader=reader)
                audit_video(data, contract, entry, method, model)
                write_cache(output / "checking" / (entry["sequence"] + "__" + method + ".npz"), data,
                            cache_metadata(checking_run, entry, method))
    return output, contract, model, records, reader


class ArrayTests(unittest.TestCase):
    def test_actual_crop_roi_and_subcell_fallback(self):
        weights = roi_weights([[10, 10, 20, 20], [0, 0, 1, 1]], [0, 0, 40, 40], 4)
        expected = np.zeros(16)
        expected[[5, 6, 9, 10]] = .25
        np.testing.assert_array_equal(weights[0], expected)
        self.assertEqual(weights[1, 0], 1)
        np.testing.assert_allclose(weights.sum(1), 1)
        outside = roi_weights([[100, 100, 1, 1]], [0, 0, 40, 40], 4)
        self.assertEqual(outside[0, 15], 1)

    def test_state_uses_history_and_never_labels(self):
        state = IdentityState([40, 20, 10, 10], [1, 0])
        boxes, desc, scores = np.array([[20., 20, 10, 10], [40., 20, 10, 10]]), np.array([[0., 1], [1., 0]]), np.array([.9, .8])
        features = state.features(boxes, desc, scores)
        self.assertEqual(select_candidate("appearance", features), 1)
        self.assertEqual(select_candidate("motion", features), 1)
        state.advance(boxes, desc, features, 1)
        again = state.features(boxes, desc, scores)
        self.assertGreater(again[1, 4], again[0, 4])
        np.testing.assert_array_equal(state.box, boxes[1])

    def test_pairwise_fit_minimum_support_and_direction(self):
        records = []
        for _ in range(4):
            features = np.zeros((25, 5, len(FEATURES)))
            features[:, 0, 0] = 1
            valid = np.zeros((25, 5), bool)
            valid[:, :2] = True
            labels = np.full((25, 5), -1)
            labels[:, 0], labels[:, 1] = 1, 0
            records.append(dict(features=features, valid=valid, labels=labels))
        model = fit_ranker(records)
        self.assertGreater(model["weights"][0], 0)
        self.assertEqual(model["fit_pair_accuracy"], 1)
        with self.assertRaises(ValueError):
            fit_ranker(records[:3])

    def test_static_control_cannot_use_history_features(self):
        model = dict(static_feature_scale=[1.] * len(FEATURES), static_weights=[1., 0., 0., 0., 0., 0., 0., .1, 0.])
        features = np.zeros((2, len(FEATURES)))
        features[1, 0] = 1
        self.assertEqual(select_candidate("static_ranker", features, model), 1)
        features[0, 1:7] = 10000
        features[1, 1:7] = -10000
        self.assertEqual(select_candidate("static_ranker", features, model), 1)

    @unittest.skipUnless(TORCH, "GPU descriptor helper comparison requires PyTorch (CPU tensors suffice)")
    def test_torch_roi_pool_matches_known_grid(self):
        import torch
        observer = NativeObserver.__new__(NativeObserver)
        observer.rt = SimpleNamespace(torch=torch, search_side=4)
        grid = torch.tensor(np.stack([np.arange(16), np.ones(16)], axis=1), dtype=torch.float32)
        result = observer._pool(grid, [[10, 10, 20, 20]], [0, 0, 40, 40])
        expected = np.array([7.5, 1.])
        np.testing.assert_allclose(result[0], expected / np.linalg.norm(expected), atol=1e-7)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=HERE)
        cls.root = Path(cls.temporary.name)
        cls.output, cls.contract, cls.model, cls.records, reader = fixture(cls.root)
        cls.reader = staticmethod(reader)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_disjoint_fresh_split_no_future_gt_in_manifest(self):
        contract = load_contract(self.output / "manifest.json")
        groups = {e["source_group_proxy"] for e in contract["sequences"]}
        self.assertFalse(groups & set(contract["excluded_groups"]))
        self.assertEqual(len(groups), 14)
        self.assertEqual(sum(e["split"] == "fit" for e in contract["sequences"]), 8)
        self.assertTrue(all("ground_truth" not in e for e in contract["sequences"]))
        self.assertFalse(self.model["checking_labels_used"])
        self.assertEqual(self.model["fit_pair_videos"], 8)

    def test_future_gt_changes_do_not_change_actor_outputs(self):
        entry = next(e for e in self.contract["sequences"] if e["split"] == "checking")
        original = ground_truth(self.contract, entry)
        changed = original.copy()
        changed[1:, 0] += 10
        for method in METHODS:
            a = run_video(FakeObserver(self.contract["runtime"]), self.contract, entry, method, self.model,
                          image_reader=self.reader, labels=original)
            b = run_video(FakeObserver(self.contract["runtime"]), self.contract, entry, method, self.model,
                          image_reader=self.reader, labels=changed)
            np.testing.assert_array_equal(a["boxes"], b["boxes"])
            np.testing.assert_array_equal(a["chosen"], b["chosen"])
            audit_video(b, self.contract, entry, method, self.model, labels=changed)
        self.assertFalse(np.array_equal(a["iou"], b["iou"]))

    def test_corrupted_continuity_or_features_rejected(self):
        entry = next(e for e in self.contract["sequences"] if e["split"] == "fit")
        data = self.records[0]
        for key in ("previous", "features", "chosen", "labels"):
            bad = {k: v.copy() for k, v in data.items()}
            bad[key].flat[0] += 1
            with self.assertRaises((ValueError, AssertionError)):
                audit_video(bad, self.contract, entry, "sgla", fit_collect=True)

    def test_full_summary_and_incomplete_never_passes(self):
        cmd = [sys.executable, "-B", str(HERE / "stage31_identity_summary.py"), "--input", str(self.output)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.output / "analysis" / "stage31_decision.json").read_text())
        self.assertEqual(report["checking_videos"], 6)
        # Fixed appearance suffices in this synthetic data, so temporal value must not be claimed.
        self.assertFalse(report["pilot_pass"])
        path = next((self.output / "checking").glob("*.npz"))
        saved = path.read_bytes()
        try:
            path.unlink()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            result = subprocess.run(cmd + ["--allow_incomplete"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((self.output / "analysis_pilot" / "stage31_decision.json").read_text())
            self.assertEqual(report["status"], "PILOT_INCOMPLETE")
        finally:
            path.write_bytes(saved)

    def test_model_is_immutable_on_repeated_fit(self):
        path = self.output / "model.json"
        before = file_hash(path)
        cmd = [sys.executable, "-B", str(HERE / "stage31_identity_fit.py"), "--input", str(self.output), "--mode", "train"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, file_hash(path))


class GateTests(unittest.TestCase):
    def test_explicit_pass_and_harm_guard(self):
        rows = []
        for i in range(6):
            for method in METHODS:
                rows.append(dict(sequence=str(i), method=method, failure_events=0,
                    alternative_frames=2 if method == "identity" else 0, tracked_frames=100,
                    low_iou_fraction=.01 if method == "identity" else .02))
        pairs = [dict(right=m, mean_delta=.2, ci_low=.01) for m in METHODS if m != "identity"]
        self.assertTrue(decide(rows, pairs, True)["pilot_pass"])
        self.assertFalse(decide(rows, pairs, False)["pilot_pass"])
        bad = copy.deepcopy(rows)
        next(r for r in bad if r["method"] == "identity")["failure_events"] = 1
        self.assertFalse(decide(bad, pairs, True)["pilot_pass"])


if __name__ == "__main__":
    unittest.main()
