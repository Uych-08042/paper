"""CPU mechanism tests; real PyTorch FFN surgery tests run when torch is installed."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import fingerprint, write_cache
from stage19_compression_collect import collect_clip
from stage19_compression_common import load_and_audit_cache, make_run, nonempty_path
from stage19_compression_manifest import build_manifest
from stage19_compression_summary import analyze, sequence_scores
from stage19_ffn_compression import compress_ffn, linear_slices, make_masks


def source_manifest():
    result = {"schema": "stage4_ab_v1", "dataset": "got10k_train", "root": "/unused/train",
              "clip_frames": 50, "seed": 9, "sequences": [
                  {"sequence": "GOT-10k_Train_{:06d}".format(i + 1),
                   "split": "train" if i < 8 else "val", "start": 0, "stop": 50,
                   "annotation_sha256": "fixture", "visible_sha256": "fixture"} for i in range(10)]}
    result["fingerprint"] = fingerprint(result)
    return result


class FakeRuntime:
    """Crop-dependent dynamics: a persistent prediction shift accumulates in free rollouts."""
    def __init__(self):
        self.torch = SimpleNamespace(no_grad=nullcontext)
        self.tracker = SimpleNamespace(state=None, frame_id=0)
        self.identity = {"embed_dim": 2}
        self.bias = 0.0
        self.forward_bias = []
        self.verify_calls = 0

    def read(self, path):
        return np.full((2, 2, 3), int(Path(path).stem), dtype=np.uint8)

    def initialize(self, image, box):
        self.tracker.state = list(box)
        self.tracker.frame_id = 0

    def prepare(self, image):
        return np.asarray(self.tracker.state).copy(), 1.0

    def prefix(self, search):
        self.forward_bias.append(self.bias)
        return search

    def selector(self, tokens):
        return tokens

    def selected_layer(self, probabilities):
        return 7 + int(probabilities[0]) % 6

    def candidate(self, tokens, layer):
        result = tokens.copy()
        result[0] = 0.8 * tokens[0] + 0.2 * 10 + self.bias
        return {"box": result}

    def decode(self, output, factor, image):
        return output["box"].tolist()

    def verify(self, search, output):
        self.verify_calls += 1


@contextmanager
def fake_surgery(rt, spec, verify=True):
    previous = rt.bias
    rt.bias = len(spec["drop"]) / spec["width"] * (1.2 + spec["layer"] * 0.3)
    try:
        yield {"removed_parameters": len(spec["drop"]) * 5, "masked_equivalence_max_abs_diff": 0.0}
    finally:
        rt.bias = previous


class MaskTests(unittest.TestCase):
    def test_equal_cost_disjoint_reproducible_groups(self):
        masks = make_masks([128] * 6, seed=7)
        self.assertEqual(masks, make_masks([128] * 6, seed=7))
        self.assertNotEqual(masks, make_masks([128] * 6, seed=8))
        self.assertEqual(len(masks), 12)
        for index in range(0, 12, 2):
            self.assertEqual(len(masks[index]["drop"]), 32)
            self.assertFalse(set(masks[index]["drop"]) & set(masks[index + 1]["drop"]))

    def test_slice_preserves_output_bias_and_removes_hidden_units(self):
        rng = np.random.default_rng(2)
        x, w1, b1, w2, b2 = (rng.normal(size=shape) for shape in ((3, 4), (8, 4), (8,), (4, 8), (4,)))
        drop = [1, 3, 5]
        keep = linear_slices(w1, b1, w2, drop)
        hidden = np.maximum(x @ w1.T + b1, 0)
        expected = hidden.copy()
        expected[:, drop] = 0
        actual = np.maximum(x @ w1[keep].T + b1[keep], 0) @ w2[:, keep].T + b2
        np.testing.assert_allclose(actual, expected @ w2.T + b2)
        self.assertEqual(linear_slices(w1, b1, w2, []).size, 8)

    def test_invalid_compression_groups_rejected(self):
        for fraction in (0, 1, 0.1, float("nan")):
            with self.assertRaises(ValueError):
                make_masks([128] * 6, fraction=fraction)
        for drop in ([1, 1], [-1], [8], [1.5], list(range(8))):
            with self.assertRaises(ValueError):
                linear_slices(np.zeros((8, 4)), None, np.zeros((4, 8)), drop)


class ManifestTests(unittest.TestCase):
    def test_source_val_excluded_and_video_splits_disjoint(self):
        result = build_manifest(source_manifest(), 2, 2)
        source_train = {row["sequence"] for row in source_manifest()["sequences"] if row["split"] == "train"}
        self.assertTrue(all(row["sequence"] in source_train for row in result["sequences"]))
        self.assertEqual(len({row["sequence"] for row in result["sequences"]}), 4)
        self.assertEqual(result, build_manifest(source_manifest(), 2, 2))

    def test_short_inventory_cannot_borrow_validation_videos(self):
        with self.assertRaisesRegex(ValueError, "Source val is excluded"):
            build_manifest(source_manifest(), 5, 4)
        with self.assertRaises(Exception):
            nonempty_path("")


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.manifest = build_manifest(source_manifest(), 1, 1, anchors=2)
        self.entry = self.manifest["sequences"][0]
        self.rt = FakeRuntime()
        self.run = make_run(self.manifest, self.rt.identity, [128] * 6, 0.25, 2, 2057)
        self.gt = np.tile([10.0, 10.0, 10.0, 10.0], (50, 1))
        self.data, self.initial = collect_clip(self.rt, Path("unused"), "got10k_train", self.entry,
                                               self.gt, np.ones(50, dtype=bool), self.run, fake_surgery)

    def audit(self, data):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            path = Path(temp) / "fixture.npz"
            write_cache(path, data, {"entry": self.entry, "run_fingerprint": self.run["fingerprint"], "init_box": self.initial})
            return load_and_audit_cache(path, self.entry, self.run)

    def test_persistent_compression_isolates_feedback_and_restores(self):
        self.audit(self.data)
        self.assertEqual(self.rt.bias, 0.0)
        self.assertEqual(self.rt.verify_calls, 1)
        active = np.count_nonzero(np.asarray(self.rt.forward_bias))
        self.assertEqual(active, 2 * 2 * 12 * 20)
        np.testing.assert_allclose(self.data["sync_boxes"][:, :, 0], self.data["free_boxes"][:, :, 0])
        self.assertTrue(np.all(self.data["free_boxes"][:, :, -1, 0] > self.data["sync_boxes"][:, :, -1, 0]))
        result = sequence_scores(self.data, 20)
        self.assertTrue(np.all(result["feedback_damage"] > 0))

    def test_audit_rejects_corrupted_states_labels_and_restoration(self):
        for key, index in (("sync_states", (0, 0, 1, 0)), ("free_states", (0, 0, 1, 0)),
                           ("post_restore_probe_boxes", (1, 0)), ("identity_boxes", (1, 0)),
                           ("free_iou", (0, 0, 0)), ("removed_parameters", (0,))):
            broken = {name: value.copy() for name, value in self.data.items()}
            broken[key][index] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(broken)

    def test_incomplete_data_cannot_select_a_structure(self):
        record = {"entry": self.entry, "data": self.data}
        _, _, paired, decision = analyze([record], self.run, missing=1, bootstrap_samples=10)
        self.assertEqual(decision["status"], "PENDING_INCOMPLETE")
        self.assertFalse(paired)
        self.assertNotIn("calibration_selected_masks", decision)

    def test_structure_choice_never_reads_check_outcomes(self):
        calibration = {name: value.copy() for name, value in self.data.items()}
        check = {name: value.copy() for name, value in self.data.items()}
        # Synthetic summary inputs give different optimal structures by split.
        calibration["free_iou"][:] = 0.5
        calibration["sync_iou"][:] = 0.5
        calibration["free_iou"][:, 1] = 0.9
        records = [{"entry": self.entry, "data": calibration},
                   {"entry": self.manifest["sequences"][1], "data": check}]
        first = analyze(records, self.run, bootstrap_samples=10)[-1]
        check["free_iou"] = 1 - check["free_iou"]
        second = analyze(records, self.run, bootstrap_samples=10)[-1]
        self.assertEqual(first["calibration_selected_masks"], second["calibration_selected_masks"])
        self.assertEqual(first["status"], "PILOT_TOO_SMALL")

    def test_visibility_mask_and_anchor_cluster_weighting(self):
        data = {name: value.copy() for name, value in self.data.items()}
        data["valid"][0, :5] = False
        result = sequence_scores(data, 5)
        self.assertEqual(result["eligible_anchors"], 1)

    def test_cli_summary_and_resume_identity(self):
        from stage4_collect import publish_run
        from stage19_compression_common import cache_path
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            directory = Path(temp) / "cache"
            publish_run(directory, self.run)
            publish_run(directory, self.run)
            changed = dict(self.run, seed=0)
            with self.assertRaisesRegex(ValueError, "Run configuration changed"):
                publish_run(directory, changed)
            for entry in self.manifest["sequences"]:
                write_cache(cache_path(directory, entry), self.data,
                            {"entry": entry, "run_fingerprint": self.run["fingerprint"], "init_box": self.initial})
            output = Path(temp) / "analysis"
            script = Path(__file__).with_name("stage19_compression_summary.py")
            process = subprocess.run([sys.executable, "-B", str(script), "--input", str(directory),
                                      "--output_dir", str(output), "--bootstrap_samples", "20"],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            decision = json.loads((output / "stage19_decision.json").read_text())
            self.assertEqual(decision["status"], "PILOT_TOO_SMALL")
            self.assertTrue((output / "stage19_mask_summary.csv").is_file())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class RealSurgeryTests(unittest.TestCase):
    def test_physical_tensors_zero_deletion_and_exception_restoration(self):
        import torch
        class Mlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 128)
                self.fc2 = torch.nn.Linear(128, 4)
            def forward(self, value):
                return self.fc2(torch.nn.functional.gelu(self.fc1(value)))
        mlp = Mlp().eval()
        rt = SimpleNamespace(torch=torch, backbone=SimpleNamespace(blocks=[SimpleNamespace(mlp=mlp)]))
        original1, original2 = mlp.fc1, mlp.fc2
        spec = make_masks([128], per_layer=1)[0]
        with compress_ffn(rt, dict(spec, drop=[])) as info:
            self.assertEqual(info["removed_parameters"], 0)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with compress_ffn(rt, spec) as info:
                self.assertEqual(mlp.fc1.weight.shape, (96, 4))
                self.assertEqual(mlp.fc2.weight.shape, (4, 96))
                self.assertEqual(info["removed_parameters"], 32 * 9)
                raise RuntimeError("fixture failure")
        self.assertIs(mlp.fc1, original1)
        self.assertIs(mlp.fc2, original2)


if __name__ == "__main__":
    unittest.main()
