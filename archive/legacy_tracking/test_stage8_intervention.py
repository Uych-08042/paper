"""CPU-only tests for Stage-8B causal intervention collection."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import fingerprint, write_cache, write_json
from stage4_manifest import build_manifest
from stage8_intervention_collect import collect_interventions
from stage8_intervention_common import (
    intervention_cache_path,
    intervention_labels,
    load_and_audit_cache,
    make_run,
    select_anchor_offsets,
)
from stage8_intervention_manifest import build_intervention_manifest
from stage8_intervention_summary import anchor_rows, summarize_effects


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeRuntime:
    def __init__(self):
        self.torch = SimpleNamespace(no_grad=nullcontext)
        self.tracker = SimpleNamespace(state=None, frame_id=0)
        self.identity = {"lens_z": 2, "lens_x": 4, "embed_dim": 2}
        self.candidate_calls = 0
        self.fast_runs = 0
        self.fixed_l9_runs = 0

    def read(self, path):
        return np.full((32, 32, 3), int(Path(path).stem), dtype=np.uint8)

    def initialize(self, image, box):
        self.tracker.state = list(box)
        self.tracker.frame_id = 0

    def prepare(self, image):
        return float(image[0, 0, 0]) + self.tracker.state[0], 1.0

    def prefix(self, search):
        return FakeTensor(np.full((1, 6, 2), search))

    def head(self, tokens):
        return {
            "score_map": FakeTensor(np.full((1, 1, 2, 2), 0.5)),
            "box": np.asarray(self.tracker.state) + [0.5, 0.0, 0.0, 0.0],
        }

    def selector(self, tokens):
        return FakeTensor([[0.1, 0.2, 0.9, 0.4, 0.5, 0.6]])

    def selected_layer(self, probabilities):
        return 9

    def candidate(self, tokens, layer):
        self.assert_layer(layer)
        self.candidate_calls += 1
        return {
            "score_map": FakeTensor(np.full((1, 1, 2, 2), 0.6)),
            "box": np.asarray(self.tracker.state) + [1.0, 0.0, 0.0, 0.0],
        }

    @staticmethod
    def assert_layer(layer):
        if layer != 9:
            raise AssertionError("Only L9 is expected")

    def pool(self, tokens, grid):
        return FakeTensor(
            np.full((1, (2 + grid ** 2) * 2), tokens.value.mean())
        )

    def decode(self, output, factor, image):
        return output["box"].tolist()

    def run_image(self, image, mode):
        search, factor = self.prepare(image)
        tokens = self.prefix(search)
        if mode == "fast":
            self.fast_runs += 1
            output = self.head(tokens)
        elif mode == "fixed_l9":
            self.fixed_l9_runs += 1
            output = self.candidate(tokens, 9)
        else:
            raise ValueError("Unexpected fake runtime mode: {}".format(mode))
        return self.decode(output, factor, image)


def create_dataset(root, names, length=12):
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        boxes = ["{},{},10,10".format(10 + index * 0.5, 10) for index in range(length)]
        (directory / "groundtruth.txt").write_text(
            "\n".join(boxes) + "\n", encoding="utf-8"
        )
        (directory / "absence.label").write_text(
            "0\n" * length, encoding="utf-8"
        )
        (directory / "cover.label").write_text(
            "8\n" * length, encoding="utf-8"
        )
        for index in range(1, length + 1):
            (directory / "{:08d}.jpg".format(index)).write_bytes(b"image")


class InterventionCommonTests(unittest.TestCase):
    def test_anchor_offsets_are_fixed_unique_and_fit_horizon(self):
        offsets = select_anchor_offsets(63, 40, 12)
        self.assertEqual(len(offsets), 12)
        self.assertEqual(len(set(offsets.tolist())), 12)
        self.assertGreaterEqual(offsets[0], 1)
        self.assertLessEqual(offsets[-1] + 40 - 1, 63)

    def test_horizon_labels_separate_immediate_and_future_effect(self):
        control = np.asarray([[0.5, 0.5, 0.5]])
        treatment = np.asarray([[0.7, 0.4, 0.4]])
        labels = intervention_labels(
            control, treatment, np.ones_like(control, dtype=bool), (1, 3)
        )
        self.assertGreater(labels["mean_iou_delta"][0, 0], 0.0)
        self.assertLess(labels["post_anchor_mean_iou_delta"][0, 1], 0.0)
        self.assertTrue(np.isnan(labels["post_anchor_auc_delta"][0, 0]))


class InterventionManifestTests(unittest.TestCase):
    def test_fresh_manifest_excludes_every_prior_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            names = ["GOT-10k_Train_{:06d}".format(i) for i in range(1, 8)]
            create_dataset(root, names)
            prior = build_manifest("got10k_train", root, names[:2], 1, 1, 8, 4)
            prior_path = Path(temporary) / "prior.json"
            write_json(prior_path, prior)
            manifest = build_intervention_manifest(
                "got10k_train",
                root,
                names,
                2,
                1,
                10,
                6,
                7,
                [prior_path],
                (1, 3),
                3,
            )
            selected = {entry["sequence"] for entry in manifest["sequences"]}
            used = {entry["sequence"] for entry in prior["sequences"]}
            self.assertFalse(selected.intersection(used))
            self.assertEqual(
                manifest["purpose"],
                "stage8_l9_causal_intervention_development",
            )
            self.assertEqual(manifest["intervention_design"]["anchor_count"], 3)


class InterventionCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        create_dataset(
            self.root / "dataset",
            ["GOT-10k_Train_000001", "GOT-10k_Train_000002"],
        )
        self.entry = {
            "sequence": "GOT-10k_Train_000001",
            "split": "train",
            "start": 0,
            "stop": 12,
            "annotation_sha256": None,
            "visible_sha256": None,
        }
        self.rt = FakeRuntime()
        directory = self.root / "dataset" / self.entry["sequence"]
        self.gt = np.loadtxt(directory / "groundtruth.txt", delimiter=",", ndmin=2)
        self.valid = np.ones(len(self.gt), dtype=bool)
        self.arrays, self.initial = collect_interventions(
            self.rt,
            directory,
            "got10k_train",
            self.entry,
            self.gt,
            self.valid,
            pool_grid=2,
            anchor_count=3,
            horizons=(1, 3),
            verify_l9=True,
        )

    def test_control_windows_are_exact_fast_baseline_slices(self):
        offsets = self.arrays["anchor_offset"]
        expected = np.stack(
            [self.arrays["baseline_boxes"][offset - 1 : offset + 2] for offset in offsets]
        )
        np.testing.assert_allclose(self.arrays["control_boxes"], expected)
        np.testing.assert_allclose(
            self.arrays["control_probe_boxes"], expected[0]
        )
        self.assertTrue(
            np.any(
                np.abs(
                    self.arrays["treatment_boxes"]
                    - self.arrays["control_boxes"]
                )
                > 0
            )
        )
        self.assertEqual(self.rt.candidate_calls, 4)
        self.assertEqual(self.rt.fixed_l9_runs, 1)
        self.assertEqual(self.rt.fast_runs, 9)

    def test_cache_roundtrip_and_summary(self):
        second = dict(
            self.entry, sequence="GOT-10k_Train_000002", split="val"
        )
        run_manifest = {
            "schema": "stage4_ab_v1",
            "dataset": "got10k_train",
            "root": str((self.root / "dataset").resolve()),
            "seed": 1,
            "clip_frames": 12,
            "sequences": [self.entry, second],
        }
        run_manifest["fingerprint"] = fingerprint(run_manifest)
        run = make_run(run_manifest, self.rt.identity, 2, 3, (1, 3), True)
        path = intervention_cache_path(self.root / "cache", self.entry)
        write_cache(
            path,
            self.arrays,
            {
                "run_fingerprint": run["fingerprint"],
                "entry": self.entry,
                "init_box": self.initial,
            },
        )
        report, data, metadata = load_and_audit_cache(path, self.entry, run)
        self.assertEqual(report["anchors"], 3)
        records = [
            {
                "entry": self.entry,
                "path": path,
                "report": report,
                "data": data,
                "metadata": metadata,
            }
        ]
        rows = anchor_rows(records, (1, 3))
        summary = summarize_effects(rows, (1, 3), 20, 9)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["num_anchors"], 3)
        self.assertIn("post_anchor_auc_delta_ci_low", summary[1])

        bad_arrays = {key: value.copy() for key, value in self.arrays.items()}
        bad_arrays["control_probe_boxes"][0, 0] += 1.0
        bad_path = self.root / "bad_replay.npz"
        write_cache(
            bad_path,
            bad_arrays,
            {
                "run_fingerprint": run["fingerprint"],
                "entry": self.entry,
                "init_box": self.initial,
            },
        )
        with self.assertRaisesRegex(ValueError, "replay disagrees"):
            load_and_audit_cache(bad_path, self.entry, run)


if __name__ == "__main__":
    unittest.main()
