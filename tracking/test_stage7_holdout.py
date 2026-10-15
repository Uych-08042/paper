"""CPU-only tests for the frozen Stage-7 holdout experiment."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from stage4_common import load_manifest, write_json
from stage4_manifest import build_manifest
from stage7_holdout_common import (
    ManifestSequence,
    assignment_for_policy,
    resolve_policy,
)
from stage7_holdout_manifest import build_holdout_manifest


def create_got_dataset(root, names, length=40):
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "groundtruth.txt").write_text(
            "10,10,20,20\n" * length, encoding="utf-8"
        )
        (directory / "absence.label").write_text(
            "0\n" * length, encoding="utf-8"
        )
        (directory / "cover.label").write_text(
            "8\n" * length, encoding="utf-8"
        )
        for index in range(1, length + 1):
            (directory / "{:08d}.jpg".format(index)).write_bytes(b"image")


class HoldoutManifestTests(unittest.TestCase):
    def test_excluded_manifest_is_disjoint_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            names = [
                "GOT-10k_Train_{:06d}".format(index)
                for index in range(1, 9)
            ]
            create_got_dataset(root, names)
            pilot = build_manifest(
                "got10k_train", root, names[:3], 1, 1, 20, 2026
            )
            pilot_path = Path(temporary) / "pilot.json"
            write_json(pilot_path, pilot)

            holdout = build_holdout_manifest(
                "got10k_train",
                root,
                names,
                1,
                3,
                20,
                2027,
                [pilot_path],
            )
            pilot_names = {
                entry["sequence"] for entry in pilot["sequences"]
            }
            holdout_names = {
                entry["sequence"] for entry in holdout["sequences"]
            }
            self.assertFalse(pilot_names.intersection(holdout_names))
            self.assertEqual(holdout["excluded_sequence_count"], 2)
            self.assertEqual(holdout["purpose"], "stage7_frozen_temporal_holdout")

            holdout_path = Path(temporary) / "holdout.json"
            write_json(holdout_path, holdout)
            self.assertEqual(load_manifest(holdout_path), holdout)

    def test_manifest_sequence_uses_only_the_frozen_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            names = ["GOT-10k_Train_000001", "GOT-10k_Train_000002"]
            create_got_dataset(root, names, length=30)
            manifest = build_manifest(
                "got10k_train", root, names, 1, 1, 12, 7
            )
            entry = manifest["sequences"][1]
            sequence = ManifestSequence(manifest, entry)
            self.assertEqual(len(sequence.frames), 12)
            self.assertEqual(sequence.ground_truth_rect.shape, (12, 4))
            self.assertTrue(sequence.target_visible.all())
            self.assertEqual(
                sequence.init_info()["init_bbox"], [10.0, 10.0, 20.0, 20.0]
            )
            self.assertEqual(
                Path(sequence.frames[0]).name,
                "{:08d}.jpg".format(entry["start"] + 1),
            )


class HoldoutPolicyTests(unittest.TestCase):
    def setUp(self):
        self.sequence = SimpleNamespace(
            dataset="got10k_train",
            name="clip",
            frames=[None] * 100,
        )

    def test_fixed_baseline_costs(self):
        fast = assignment_for_policy(self.sequence, "fast")
        sgla = assignment_for_policy(self.sequence, "sgla")
        self.assertEqual(fast["router_candidate_blocks"], 0)
        self.assertEqual(sgla["router_candidate_blocks"], 99)
        self.assertEqual(fast["early_frames"] + fast["late_frames"], 99)
        self.assertEqual(resolve_policy(70, 49, "fast"), "fast")
        self.assertEqual(resolve_policy(2, 49, "sgla"), "sgla")

    def test_exact_late_cost_matches_uniform_per_clip(self):
        uniform = assignment_for_policy(self.sequence, "uniform_p5")
        late = assignment_for_policy(self.sequence, "late_p10_exact")
        self.assertEqual(
            late["router_candidate_blocks"],
            uniform["router_candidate_blocks"],
        )
        self.assertEqual(
            resolve_policy(late["early_frames"], late["early_frames"], late["router_policy"]),
            "fast",
        )
        self.assertEqual(
            resolve_policy(late["early_frames"] + 1, late["early_frames"], late["router_policy"]),
            "periodic_l9_10",
        )

    def test_baselines_reject_temporal_boundary_override(self):
        with self.assertRaisesRegex(ValueError, "applies only"):
            assignment_for_policy(self.sequence, "fast", 0.5)


if __name__ == "__main__":
    unittest.main()
