"""CPU tests for Stage-16 cache freezing and recurrent CQL primitives."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage4_common import fingerprint, write_cache, write_json
from stage16_behavior_collect import collect_continuous_behavior
from stage16_behavior_common import behavior_cache_path, make_run
from stage16_freeze_cache import build_freeze_manifest, verify_frozen_cache
from stage16_recurrent_cql import (
    SCALAR_NAMES,
    build_q_model,
    collate_clips,
    discounted_returns,
    grouped_fit_tune,
    recurrent_cql_loss,
    scalar_features,
    success_reward,
)
from test_stage16_behavior import FakeRuntime, create_dataset


class Namespace:
    gamma = 0.95
    cql_temperature = 1.0
    cql_alpha = 0.1
    bc_weight = 0.05
    policy_temperature = 1.0


def _audit_decision(run):
    decision = {
        "schema": "stage16_behavior_audit_v1",
        "status": "PASS_BEHAVIOR_DATA",
        "source_run_fingerprint": run["fingerprint"],
        "audited_clips": 2,
        "expected_clips": 2,
        "missing_clips": 0,
        "checks": {"complete": True},
    }
    decision["fingerprint"] = fingerprint(decision)
    return decision


class FreezeContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        names = ["GOT-10k_Train_000001", "GOT-10k_Train_000002"]
        create_dataset(self.root / "dataset", names, length=18)
        entries = [
            {
                "sequence": names[0],
                "split": "train",
                "start": 0,
                "stop": 18,
                "annotation_sha256": None,
                "visible_sha256": None,
            },
            {
                "sequence": names[1],
                "split": "val",
                "start": 0,
                "stop": 18,
                "annotation_sha256": None,
                "visible_sha256": None,
            },
        ]
        manifest = {
            "schema": "stage4_ab_v1",
            "dataset": "got10k_train",
            "root": str((self.root / "dataset").resolve()),
            "seed": 1,
            "clip_frames": 18,
            "sequences": entries,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        runtime = FakeRuntime()
        self.run = make_run(
            manifest,
            runtime.identity,
            state_grid=2,
            behavior_rates=(0, 50, 100),
            behavior_repeats=2,
            seed=16,
            verify_actions=False,
        )
        self.cache = self.root / "cache"
        write_json(self.cache / "run.json", self.run)
        for entry in entries:
            directory = self.root / "dataset" / entry["sequence"]
            gt = np.loadtxt(directory / "groundtruth.txt", delimiter=",", ndmin=2)
            arrays, initial = collect_continuous_behavior(
                FakeRuntime(), directory, "got10k_train", entry, gt,
                np.ones(len(gt), dtype=bool), self.run
            )
            write_cache(
                behavior_cache_path(self.cache, entry),
                arrays,
                {
                    "run_fingerprint": self.run["fingerprint"],
                    "entry": entry,
                    "init_box": initial,
                },
            )
        self.audit = self.root / "audit" / "stage16_behavior_decision.json"
        write_json(self.audit, _audit_decision(self.run))

    def test_freeze_detects_byte_changes(self):
        frozen = build_freeze_manifest(self.cache, self.audit)
        path = self.root / "frozen.json"
        write_json(path, frozen)
        directory, run, loaded = verify_frozen_cache(path)
        self.assertEqual(directory, self.cache.resolve())
        self.assertEqual(run["fingerprint"], self.run["fingerprint"])
        self.assertEqual(loaded["num_files"], 2)

        target = behavior_cache_path(self.cache, self.run["manifest"]["sequences"][0])
        with target.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "size changed"):
            verify_frozen_cache(path)


class FeatureAndReturnTests(unittest.TestCase):
    def test_discounted_returns_and_success_reward(self):
        reward = np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32)
        valid = np.asarray([[True, True, True]])
        result = discounted_returns(reward, valid, 0.5)
        np.testing.assert_allclose(result, [[1.25, 0.5, 0.0]])
        score = success_reward(np.asarray([0.0, 0.5, 1.0]))
        self.assertLess(score[0], score[1])
        self.assertLess(score[1], score[2])

    def test_group_split_is_video_disjoint(self):
        entries = [
            {"sequence": "s{}".format(index), "split": "train"}
            for index in range(10)
        ] + [{"sequence": "v0", "split": "val"}]
        fit, tune = grouped_fit_tune(entries, 0.2, 7)
        self.assertEqual(len(fit), 8)
        self.assertEqual(len(tune), 2)
        self.assertFalse(
            {entry["sequence"] for entry in fit}
            & {entry["sequence"] for entry in tune}
        )

    def test_scalar_feature_contract(self):
        episodes, steps = 2, 4
        image_hw = np.asarray([[100, 200]] * steps)
        boxes = np.ones((episodes, steps, 4), dtype=np.float64)
        boxes[..., 2:] = 10.0
        data = {
            "previous_action": np.asarray([[-1, 0, 1, 2], [-1, 6, 5, 4]]),
            "selected_layer": np.full((episodes, steps), 9),
            "selector_prob": np.full((episodes, steps, 6), 1.0 / 6.0),
            "previous_state": boxes,
            "previous_fast_box": boxes + 1.0,
            "fast_box": boxes + 2.0,
            "image_hw": image_hw,
            "candidate_used_before": np.asarray([[0, 0, 1, 1], [0, 1, 1, 2]]),
            "candidate_remaining_before": np.asarray([[1, 1, 0, 0], [2, 1, 1, 0]]),
            "remaining_candidate_rate": np.full((episodes, steps), 0.25),
            "resize_factor": np.ones((episodes, steps)),
        }
        features = scalar_features(data)
        self.assertEqual(features.shape, (episodes, steps, len(SCALAR_NAMES)))
        self.assertTrue(np.isfinite(features).all())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class RecurrentCqlTests(unittest.TestCase):
    def _item(self, seed, steps=5, episodes=2, token_dim=3):
        rng = np.random.default_rng(seed)
        action = rng.integers(0, 7, size=(episodes, steps), dtype=np.int64)
        return {
            "search_grid": rng.normal(size=(episodes, steps, 2, 2, token_dim)).astype(np.float32),
            "template": rng.normal(size=(episodes, steps, 2, token_dim)).astype(np.float32),
            "score_map": rng.normal(size=(episodes, steps, 1, 4, 4)).astype(np.float32),
            "scalar": rng.normal(size=(episodes, steps, len(SCALAR_NAMES))).astype(np.float32),
            "action": action,
            "action_propensity": np.full((episodes, steps), 1.0 / 7.0, dtype=np.float32),
            "behavior_probability": np.full((episodes, steps, 7), 1.0 / 7.0, dtype=np.float32),
            "reward": rng.uniform(size=(episodes, steps)).astype(np.float32),
            "action_reward": rng.uniform(size=(episodes, steps, 7)).astype(np.float32),
            "valid": np.ones((episodes, steps), dtype=bool),
            "done": np.broadcast_to(np.arange(steps) == steps - 1, (episodes, steps)).copy(),
            "remaining_count": np.full((episodes, steps), 2, dtype=np.int64),
            "native_action": np.full((episodes, steps), 3, dtype=np.int64),
            "sequence": "s{}".format(seed),
            "split": "train",
        }

    def test_recurrent_model_is_causal_and_loss_is_finite(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        item = self._item(1)
        mean = np.zeros(len(SCALAR_NAMES), dtype=np.float32)
        scale = np.ones(len(SCALAR_NAMES), dtype=np.float32)
        batch = collate_clips([item], torch, mean, scale)
        config = {
            "token_dim": 3,
            "scalar_dim": len(SCALAR_NAMES),
            "hidden_dim": 16,
            "spatial_channels": 4,
            "num_actions": 7,
        }
        torch.manual_seed(3)
        model = build_q_model(torch, nn, config).eval()
        with torch.no_grad():
            original = model(batch)
            changed = dict(batch)
            changed["search_grid"] = batch["search_grid"].clone()
            changed["search_grid"][:, 3:] += 100.0
            altered = model(changed)
        torch.testing.assert_close(original[:, :3], altered[:, :3])

        q = model(batch)
        target_q = q.detach().clone()
        losses = recurrent_cql_loss(torch, functional, q, target_q, batch, Namespace())
        self.assertTrue(all(torch.isfinite(value) for value in losses[:4]))
        losses[0].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
