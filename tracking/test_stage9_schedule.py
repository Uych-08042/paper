"""CPU-only tests for Stage-9 equal-cost schedule interventions."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import fingerprint, write_cache, write_json
from stage4_manifest import build_manifest
from stage9_schedule_collect import collect_schedules
from stage9_schedule_common import (
    DEFAULT_EVALUATION_HORIZONS,
    DEFAULT_HORIZON,
    POLICIES,
    SCHEDULES,
    load_and_audit_cache,
    make_run,
    schedule_cache_path,
    schedule_labels,
    validate_schedules,
)
from stage9_schedule_manifest import build_schedule_manifest
from stage9_schedule_summary import paired_summary, policy_summary
from stage9_schedule_summary import primary_decision


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)


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
        return {"box": np.asarray(self.tracker.state) + [0.5, 0.0, 0.0, 0.0]}

    def candidate(self, tokens, layer):
        if layer != 9:
            raise AssertionError("Only L9 is expected")
        self.candidate_calls += 1
        return {"box": np.asarray(self.tracker.state) + [1.0, 0.0, 0.0, 0.0]}

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
            raise ValueError("Unexpected mode: {}".format(mode))
        return self.decode(output, factor, image)


def create_dataset(root, names, length=64):
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        boxes = [
            "{},{},10,10".format(10 + index * 0.5, 10)
            for index in range(length)
        ]
        (directory / "groundtruth.txt").write_text(
            "\n".join(boxes) + "\n", encoding="utf-8"
        )
        (directory / "absence.label").write_text("0\n" * length, encoding="utf-8")
        (directory / "cover.label").write_text("8\n" * length, encoding="utf-8")
        for index in range(1, length + 1):
            (directory / "{:08d}.jpg".format(index)).write_bytes(b"image")


class ScheduleCommonTests(unittest.TestCase):
    def test_frozen_schedules_have_equal_h20_cost(self):
        mask = validate_schedules(SCHEDULES, DEFAULT_HORIZON)
        self.assertEqual(mask.shape, (4, 40))
        self.assertEqual(int(mask[0].sum()), 0)
        self.assertEqual([int(row.sum()) for row in mask[1:]], [4, 4, 4])
        self.assertEqual([int(row[:20].sum()) for row in mask[1:]], [4, 4, 4])

    def test_labels_separate_common_tail_and_persistence(self):
        iou = np.full((1, 4, 40), 0.5, dtype=np.float64)
        iou[:, 3, 16:20] = 0.7
        iou[:, 3, 20:40] = 0.6
        labels = schedule_labels(
            iou, np.ones((1, 40), dtype=bool), DEFAULT_EVALUATION_HORIZONS
        )
        self.assertGreater(labels["common_tail_auc"][0, 3], labels["common_tail_auc"][0, 0])
        self.assertGreater(labels["persistence_auc"][0, 3], labels["persistence_auc"][0, 0])

    def test_incomplete_cache_cannot_produce_a_decision(self):
        decision = primary_decision([], [Path("missing.npz")])
        self.assertEqual(decision["status"], "PENDING_INCOMPLETE")


class ScheduleManifestTests(unittest.TestCase):
    def test_manifest_excludes_prior_videos(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            names = ["GOT-10k_Train_{:06d}".format(i) for i in range(1, 9)]
            create_dataset(root, names)
            prior = build_manifest("got10k_train", root, names[:2], 1, 1, 64, 4)
            prior_path = Path(temporary) / "prior.json"
            write_json(prior_path, prior)
            manifest = build_schedule_manifest(
                "got10k_train",
                root,
                names,
                2,
                1,
                64,
                52,
                7,
                [prior_path],
                12,
            )
            selected = {entry["sequence"] for entry in manifest["sequences"]}
            used = {entry["sequence"] for entry in prior["sequences"]}
            self.assertFalse(selected.intersection(used))
            self.assertEqual(
                manifest["purpose"], "stage9_equal_cost_schedule_intervention"
            )
            self.assertEqual(manifest["schedule_design"]["primary_horizon"], 20)

    def test_short_inventory_error_reports_largest_valid_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            names = ["GOT-10k_Train_{:06d}".format(i) for i in range(1, 5)]
            create_dataset(root, names)
            prior = build_manifest("got10k_train", root, names[:2], 1, 1, 64, 4)
            prior_path = Path(temporary) / "prior.json"
            write_json(prior_path, prior)
            with self.assertRaisesRegex(
                ValueError,
                r"largest valid split is --pilot_sequences 1 --confirm_sequences 1",
            ):
                build_schedule_manifest(
                    "got10k_train",
                    root,
                    names,
                    2,
                    1,
                    64,
                    52,
                    7,
                    [prior_path],
                    12,
                )

    def test_empty_inventory_error_does_not_suggest_invalid_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            names = ["GOT-10k_Train_000001", "GOT-10k_Train_000002"]
            create_dataset(root, names)
            prior = build_manifest("got10k_train", root, names, 1, 1, 64, 4)
            prior_path = Path(temporary) / "prior.json"
            write_json(prior_path, prior)
            with self.assertRaisesRegex(
                ValueError,
                r"At least two fresh videos are required",
            ):
                build_schedule_manifest(
                    "got10k_train",
                    root,
                    names,
                    1,
                    1,
                    64,
                    52,
                    7,
                    [prior_path],
                    12,
                )


class ScheduleCollectionTests(unittest.TestCase):
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
            "stop": 64,
            "annotation_sha256": None,
            "visible_sha256": None,
        }
        self.rt = FakeRuntime()
        directory = self.root / "dataset" / self.entry["sequence"]
        self.gt = np.loadtxt(directory / "groundtruth.txt", delimiter=",", ndmin=2)
        self.valid = np.ones(len(self.gt), dtype=bool)
        self.arrays, self.initial = collect_schedules(
            self.rt,
            directory,
            "got10k_train",
            self.entry,
            self.gt,
            self.valid,
            anchor_count=1,
            horizon=DEFAULT_HORIZON,
            evaluation_horizons=DEFAULT_EVALUATION_HORIZONS,
            schedules=SCHEDULES,
            verify_l9=True,
        )

    def _run(self):
        second = dict(
            self.entry, sequence="GOT-10k_Train_000002", split="val"
        )
        manifest = {
            "schema": "stage4_ab_v1",
            "dataset": "got10k_train",
            "root": str((self.root / "dataset").resolve()),
            "seed": 1,
            "clip_frames": 64,
            "sequences": [self.entry, second],
        }
        manifest["fingerprint"] = fingerprint(manifest)
        return make_run(
            manifest,
            self.rt.identity,
            1,
            DEFAULT_HORIZON,
            DEFAULT_EVALUATION_HORIZONS,
            SCHEDULES,
            True,
        )

    def test_fast_policy_is_exact_baseline_and_cost_is_equal(self):
        offset = int(self.arrays["anchor_offset"][0])
        expected = self.arrays["baseline_boxes"][offset - 1 : offset + 39]
        np.testing.assert_allclose(self.arrays["policy_boxes"][0, 0], expected)
        np.testing.assert_allclose(self.arrays["control_probe_boxes"], expected)
        self.assertEqual(
            [int(row.sum()) for row in self.arrays["action_mask"]],
            [0, 4, 4, 4],
        )
        self.assertEqual(self.rt.candidate_calls, 11)
        self.assertEqual(self.rt.fixed_l9_runs, 10)
        self.assertEqual(self.rt.fast_runs, 149)

    def test_cache_roundtrip_and_clustered_summary(self):
        run = self._run()
        path = schedule_cache_path(self.root / "cache", self.entry)
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
        self.assertEqual(report["equal_treatment_blocks"], 4)
        record = {
            "entry": self.entry,
            "path": path,
            "report": report,
            "data": data,
            "metadata": metadata,
        }
        val_entry = dict(
            self.entry, sequence="GOT-10k_Train_000002", split="val"
        )
        val_path = schedule_cache_path(self.root / "cache", val_entry)
        write_cache(
            val_path,
            self.arrays,
            {
                "run_fingerprint": run["fingerprint"],
                "entry": val_entry,
                "init_box": self.initial,
            },
        )
        val_report, val_data, val_metadata = load_and_audit_cache(
            val_path, val_entry, run
        )
        val_record = {
            "entry": val_entry,
            "path": val_path,
            "report": val_report,
            "data": val_data,
            "metadata": val_metadata,
        }
        rng = np.random.default_rng(9)
        policies = policy_summary([record, val_record], run, 20, rng)
        paired = paired_summary([record, val_record], run, 20, rng)
        self.assertTrue(
            any(row["horizon"] == 20 and row["policy"] == "uniform4" for row in policies)
        )
        self.assertTrue(
            any(
                row["horizon"] == 20
                and row["left_policy"] == "uniform4"
                and row["right_policy"] == "burst4"
                and row["equal_cost_at_horizon"] == 1
                for row in paired
            )
        )
        self.assertIn(primary_decision(paired, [])["status"], ("PASS", "FAIL"))


if __name__ == "__main__":
    unittest.main()
