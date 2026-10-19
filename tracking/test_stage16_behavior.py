"""CPU-only tests for Stage-16 continuous behavior collection and audits."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import fingerprint, write_cache, write_json
from stage16_behavior_audit import audit_behavior_dataset
from stage16_behavior_collect import collect_continuous_behavior
from stage16_behavior_common import (
    ACTION_NAMES,
    behavior_cache_path,
    behavior_specs,
    behavior_seed,
    load_and_audit_cache,
    make_behavior_schedule,
    make_run,
)


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

    def read(self, path):
        return np.full((32, 32, 3), int(Path(path).stem), dtype=np.uint8)

    def initialize(self, image, box):
        self.tracker.state = list(box)
        self.tracker.frame_id = 0

    def prepare(self, image):
        return float(image[0, 0, 0]) + self.tracker.state[0], 1.0

    def prefix(self, search):
        return FakeTensor(np.full((1, 6, 2), search))

    def spatial_state(self, tokens, grid):
        template = FakeTensor(np.full((1, 2, 2), tokens.value.mean()))
        search = FakeTensor(np.full((1, grid, grid, 2), tokens.value.mean()))
        return template, search

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
        shift = 0.2 * (layer - 6)
        return {
            "score_map": FakeTensor(np.full((1, 1, 2, 2), 0.6)),
            "box": np.asarray(self.tracker.state) + [shift, 0.0, 0.0, 0.0],
        }

    def decode(self, output, factor, image):
        return output["box"].tolist()

    def run_image(self, image, mode):
        search, factor = self.prepare(image)
        tokens = self.prefix(search)
        if mode != "fast":
            raise ValueError("Unexpected fake runtime mode")
        return self.decode(self.head(tokens), factor, image)


def create_dataset(root, names, length=18):
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        boxes = ["{},{},10,10".format(10 + index * 0.25, 10) for index in range(length)]
        (directory / "groundtruth.txt").write_text("\n".join(boxes) + "\n", encoding="utf-8")
        (directory / "absence.label").write_text("0\n" * length, encoding="utf-8")
        (directory / "cover.label").write_text("8\n" * length, encoding="utf-8")
        for index in range(1, length + 1):
            (directory / "{:08d}.jpg".format(index)).write_bytes(b"image")


class BehaviorScheduleTests(unittest.TestCase):
    def test_schedule_is_exact_balanced_and_deterministic(self):
        first = make_behavior_schedule(31, 30.0, 16)
        second = make_behavior_schedule(31, 30.0, 16)
        third = make_behavior_schedule(31, 30.0, 17)
        np.testing.assert_array_equal(first["action"], second["action"])
        self.assertFalse(np.array_equal(first["action"], third["action"]))
        self.assertEqual(int(np.sum(first["action"] > 0)), 9)
        np.testing.assert_array_equal(first["action_cost"], first["action"] > 0)
        candidate_counts = first["action_counts"][1:]
        self.assertLessEqual(int(candidate_counts.max() - candidate_counts.min()), 1)
        self.assertTrue(np.all(first["action_propensity"] > 0.0))
        np.testing.assert_allclose(first["behavior_probability"].sum(axis=1), 1.0)
        self.assertEqual(first["candidate_remaining_before"][0], 9)
        self.assertEqual(first["candidate_used_before"][0], 0)


class ContinuousBehaviorCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
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
            "stop": 18,
            "annotation_sha256": None,
            "visible_sha256": None,
        }
        second = dict(self.entry, sequence="GOT-10k_Train_000002", split="val")
        self.manifest = {
            "schema": "stage4_ab_v1",
            "dataset": "got10k_train",
            "root": str((self.root / "dataset").resolve()),
            "seed": 1,
            "clip_frames": 18,
            "sequences": [self.entry, second],
        }
        self.manifest["fingerprint"] = fingerprint(self.manifest)
        self.runtime = FakeRuntime()
        self.run = make_run(
            self.manifest,
            self.runtime.identity,
            state_grid=2,
            behavior_rates=(0, 50, 100),
            behavior_repeats=2,
            seed=16,
            verify_actions=True,
            save_full_tokens=False,
        )
        directory = self.root / "dataset" / self.entry["sequence"]
        self.directory = directory
        self.gt = np.loadtxt(directory / "groundtruth.txt", delimiter=",", ndmin=2)
        self.valid = np.ones(len(self.gt), dtype=bool)
        self.arrays, self.initial = collect_continuous_behavior(
            self.runtime,
            directory,
            "got10k_train",
            self.entry,
            self.gt,
            self.valid,
            self.run,
        )

    def _write(self, arrays=None, name=None):
        path = (
            behavior_cache_path(self.root / "cache", self.entry)
            if name is None
            else self.root / name
        )
        write_cache(
            path,
            self.arrays if arrays is None else arrays,
            {
                "run_fingerprint": self.run["fingerprint"],
                "entry": self.entry,
                "init_box": self.initial,
            },
        )
        return path

    def test_actions_feed_the_next_state_and_all_actions_are_covered(self):
        episodes = len(behavior_specs((0, 50, 100), 2))
        self.assertEqual(self.arrays["behavior_action"].shape, (episodes, 17))
        self.assertEqual(set(np.unique(self.arrays["behavior_action"])), set(range(7)))
        for episode in range(episodes):
            expected = np.concatenate(
                (
                    np.asarray(self.initial)[None],
                    self.arrays["executed_box"][episode, :-1],
                )
            )
            np.testing.assert_allclose(self.arrays["previous_state"][episode], expected)
        report, _, _ = load_and_audit_cache(
            self._write(), self.entry, self.run
        )
        self.assertEqual(report["episodes"], episodes)
        self.assertGreater(report["candidate_action_effect_ratio"], 0.99)
        self.assertGreater(report["state_divergence_vs_fast_ratio"], 0.0)

        write_json(self.root / "cache" / "run.json", self.run)
        _, sequences, trajectories, coverage, decision, missing = (
            audit_behavior_dataset(self.root / "cache", allow_incomplete=True)
        )
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(trajectories), episodes)
        self.assertEqual(len(coverage), len(ACTION_NAMES))
        self.assertEqual(decision["status"], "PILOT_PASS")
        self.assertEqual(len(missing), 1)

    def test_ground_truth_changes_labels_not_behavior_or_state(self):
        changed_gt = self.gt.copy()
        changed_gt[1:, 0] += 20.0
        arrays, _ = collect_continuous_behavior(
            FakeRuntime(),
            self.directory,
            "got10k_train",
            self.entry,
            changed_gt,
            self.valid,
            self.run,
        )
        for key in (
            "behavior_action",
            "previous_state",
            "action_boxes",
            "template_summary",
            "search_token_grid",
        ):
            np.testing.assert_allclose(arrays[key], self.arrays[key])
        self.assertFalse(np.array_equal(arrays["action_iou"], self.arrays["action_iou"]))

    def test_audit_rejects_schedule_and_continuity_tampering(self):
        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed["behavior_action"][1, 0] = (
            changed["behavior_action"][1, 0] + 1
        ) % len(ACTION_NAMES)
        with self.assertRaisesRegex(ValueError, "Behavior schedule changed"):
            load_and_audit_cache(
                self._write(changed, "bad_schedule.npz"), self.entry, self.run
            )

        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed["previous_state"][1, 2, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "not written to the next state"):
            load_and_audit_cache(
                self._write(changed, "bad_state.npz"), self.entry, self.run
            )


if __name__ == "__main__":
    unittest.main()
