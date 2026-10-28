"""Tests for the Stage-17A safe residual-advantage screen."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage16_recurrent_cql import SCALAR_NAMES
from stage17_residual_advantage import (
    RESIDUAL_SCALAR_NAMES,
    budget_diagnostic,
    build_advantage_model,
    collate_advantage_clips,
    grouped_outer_folds,
    residual_loss,
    rounded_budget,
    safe_cap_actions,
    validate_cli_paths,
)


def make_item(seed=1, episodes=2, steps=5, token_dim=3):
    rng = np.random.default_rng(seed)
    return {
        "search_grid": rng.normal(
            size=(episodes, steps, 2, 2, token_dim)
        ).astype(np.float32),
        "template": rng.normal(
            size=(episodes, steps, 2, token_dim)
        ).astype(np.float32),
        "score_map": rng.normal(
            size=(episodes, steps, 1, 4, 4)
        ).astype(np.float32),
        "scalar": rng.normal(
            size=(episodes, steps, len(SCALAR_NAMES))
        ).astype(np.float32),
        "action_reward": rng.uniform(
            size=(episodes, steps, 7)
        ).astype(np.float32),
        "valid": np.ones((episodes, steps), dtype=bool),
        "sequence": "s{}".format(seed),
        "split": "train",
    }


class PathTests(unittest.TestCase):
    def test_rejects_empty_shell_paths_and_non_file_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "freeze.json"
            manifest.write_text("{}", encoding="utf-8")
            paths = dict(
                input=directory, freeze_manifest=str(manifest),
                prepared_dir=directory, output_dir=str(Path(directory) / "new_output"),
            )
            validate_cli_paths(SimpleNamespace(**paths))
            for name in paths:
                for empty in ("", "   "):
                    with self.subTest(name=name, empty=empty):
                        invalid = dict(paths, **{name: empty})
                        with self.assertRaisesRegex(ValueError, "--{} is empty".format(name)):
                            validate_cli_paths(SimpleNamespace(**invalid))
            for invalid_path in (directory, str(Path(directory) / "missing.json")):
                with self.subTest(manifest=invalid_path):
                    invalid = dict(paths, freeze_manifest=invalid_path)
                    with self.assertRaisesRegex(ValueError, "--freeze_manifest.*file"):
                        validate_cli_paths(SimpleNamespace(**invalid))


class CollationTests(unittest.TestCase):
    def test_scalar_columns_preserve_time_episode_and_padding(self):
        array_backend = SimpleNamespace(
            from_numpy=np.asarray, as_tensor=np.asarray, long=np.int64
        )
        mean = np.linspace(-1.0, 1.0, len(RESIDUAL_SCALAR_NAMES), dtype=np.float32)
        scale = np.linspace(0.5, 2.0, len(RESIDUAL_SCALAR_NAMES), dtype=np.float32)
        for steps, shorter_steps in ((5, 3), (len(mean), len(mean))):
            with self.subTest(steps=steps):
                items = [make_item(steps=steps), make_item(seed=2, steps=shorter_steps)]
                for clip, item in enumerate(items):
                    for episode in range(2):
                        for frame in range(item["scalar"].shape[1]):
                            item["scalar"][episode, frame] = (
                                clip * 100000 + episode * 10000 + frame * 100
                                + np.arange(len(SCALAR_NAMES))
                            )
                batch = collate_advantage_clips(items, array_backend, mean, scale)
                self.assertEqual(batch["scalar"].shape, (4, steps, len(mean)))
                np.testing.assert_array_equal(
                    batch["lengths"], [steps, steps, shorter_steps, shorter_steps]
                )
                self.assertEqual(batch["sequences"], ["s1", "s1", "s2", "s2"])
                for clip, item in enumerate(items):
                    valid_steps = item["scalar"].shape[1]
                    for episode in range(2):
                        row = clip * 2 + episode
                        for column, name in enumerate(RESIDUAL_SCALAR_NAMES):
                            expected = (
                                item["scalar"][episode, :, SCALAR_NAMES.index(name)]
                                - mean[column]
                            ) / scale[column]
                            np.testing.assert_allclose(
                                batch["scalar"][row, :valid_steps, column], expected
                            )
                        np.testing.assert_array_equal(
                            batch["mask"][row], np.arange(steps) < valid_steps
                        )
                        np.testing.assert_array_equal(
                            batch["target"][row, :valid_steps],
                            item["action_reward"][episode, :, 1:]
                            - item["action_reward"][episode, :, :1],
                        )
                        self.assertFalse(batch["valid"][row, valid_steps:].any())
                        self.assertFalse(batch["target"][row, valid_steps:].any())


class FoldAndPolicyTests(unittest.TestCase):
    def test_grouped_outer_folds_cover_each_video_once(self):
        entries = [
            {"sequence": "s{:02d}".format(index), "split": "train"}
            for index in range(17)
        ]
        folds = grouped_outer_folds(entries, 5, 19)
        test_names = []
        for _, train, test in folds:
            train_names = {entry["sequence"] for entry in train}
            fold_names = {entry["sequence"] for entry in test}
            self.assertFalse(train_names & fold_names)
            test_names.extend(fold_names)
        self.assertEqual(sorted(test_names), sorted(entry["sequence"] for entry in entries))

    def test_safe_cap_never_forces_or_exceeds_budget(self):
        scores = np.asarray([0.4, -0.1, 0.3, 0.2, 0.1, 0.5])
        selected = safe_cap_actions(scores, 30.0)
        self.assertEqual(rounded_budget(len(scores), 30.0), 2)
        np.testing.assert_array_equal(selected, [True, False, True, False, False, False])
        self.assertFalse(safe_cap_actions(np.full(10, -0.1), 50.0).any())

    def test_budget_diagnostic_rewards_safe_non_l9_choice(self):
        episodes, steps, actions = 2, 10, 6
        mean = np.full((episodes, steps, actions), -0.2, dtype=np.float32)
        std = np.zeros_like(mean)
        target = np.zeros_like(mean)
        mean[:, :3, 0] = 0.2
        target[:, :3, 0] = 0.1
        target[:, :, 2] = -0.02
        record = {
            "sequence": "s0",
            "fold": 0,
            "mean": mean,
            "std": std,
            "target": target,
            "valid": np.ones((episodes, steps), dtype=bool),
            "behavior_ids": ("p30_r0", "p30_r1"),
        }
        row = budget_diagnostic([record], 30.0, 1.0, 0.0, 100, 7)
        self.assertEqual(row["candidate_blocks"], 6)
        self.assertEqual(row["candidate_cap_blocks"], 6)
        self.assertGreater(row["gain_vs_fast"], 0.0)
        self.assertGreater(row["lift_vs_uniform_l9"], 0.0)
        self.assertIn("l7:6", row["action_distribution"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class ModelTests(unittest.TestCase):
    def test_model_is_causal_and_residual_loss_is_finite(self):
        import torch
        import torch.nn as nn

        mean = np.zeros(len(RESIDUAL_SCALAR_NAMES), dtype=np.float32)
        scale = np.ones(len(RESIDUAL_SCALAR_NAMES), dtype=np.float32)
        batch = collate_advantage_clips([make_item()], torch, mean, scale)
        config = {
            "token_dim": 3,
            "scalar_dim": len(RESIDUAL_SCALAR_NAMES),
            "hidden_dim": 16,
            "spatial_channels": 4,
            "num_actions": 6,
        }
        torch.manual_seed(3)
        model = build_advantage_model(torch, nn, config).eval()
        with torch.no_grad():
            original = model(batch)
            changed = dict(batch)
            changed["search_grid"] = batch["search_grid"].clone()
            changed["search_grid"][:, 3:] += 100.0
            altered = model(changed)
        torch.testing.assert_close(original[:, :3], altered[:, :3])

        prediction = model(batch)
        weighted, plain, _ = residual_loss(torch, prediction, batch, 0.01, 3.0)
        self.assertTrue(torch.isfinite(weighted))
        self.assertTrue(torch.isfinite(plain))
        weighted.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
