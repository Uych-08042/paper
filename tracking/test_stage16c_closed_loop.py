"""CPU tests for the frozen Stage-16C deployment and decision contracts."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_json
from stage16_behavior_common import make_run
from stage16_recurrent_cql import SCALAR_NAMES, build_q_model, scalar_features
from stage16c_closed_loop_summary import make_decision
from stage16c_closed_loop_validate import (
    FrozenTavrPolicy,
    candidate_budget,
    single_step_scalar_features,
    uniform_exact_schedule,
)
from stage16c_freeze_policy import build_policy_bundle, load_policy_bundle


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PolicyBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        entries = [
            {"sequence": "train0", "split": "train", "start": 0, "stop": 8},
            {"sequence": "val0", "split": "val", "start": 0, "stop": 8},
        ]
        manifest = {
            "schema": "stage4_ab_v1",
            "dataset": "got10k_train",
            "root": str((self.root / "dataset").resolve()),
            "sequences": entries,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        runtime = {
            "parameter": "fake",
            "checkpoint_sha256": "0" * 64,
            "config": "fake",
            "torch": "fake",
            "cuda": "fake",
            "device": "fake",
            "lens_z": 2,
            "lens_x": 16,
            "embed_dim": 6,
        }
        behavior_run = make_run(
            manifest,
            runtime,
            state_grid=4,
            behavior_rates=(0, 30, 100),
            behavior_repeats=2,
            seed=1,
            verify_actions=False,
        )
        cache = self.root / "cache"
        write_json(cache / "run.json", behavior_run)
        frozen = {
            "schema": "stage16_frozen_cache_v1",
            "source_cache": str(cache.resolve()),
            "source_run_fingerprint": behavior_run["fingerprint"],
            "source_manifest_fingerprint": manifest["fingerprint"],
            "run_json_sha256": file_hash(cache / "run.json"),
        }
        frozen["fingerprint"] = fingerprint(frozen)
        self.freeze_path = self.root / "freeze.json"
        write_json(self.freeze_path, frozen)

        self.stage16b = self.root / "stage16b"
        prepared = self.stage16b / "prepared"
        prepared.mkdir(parents=True)
        prepared_contract = {
            "schema": "stage16b_prepared_v1",
            "source_run_fingerprint": behavior_run["fingerprint"],
            "source_freeze_fingerprint": frozen["fingerprint"],
            "scalar_names": list(SCALAR_NAMES),
        }
        prepared_contract["fingerprint"] = fingerprint(prepared_contract)
        write_json(prepared / "stage16b_prepared_run.json", prepared_contract)
        np.savez_compressed(
            prepared / "stage16b_token_pca.npz",
            mean=np.zeros(6, dtype=np.float32),
            components=np.eye(3, 6, dtype=np.float32),
            explained_variance_ratio=np.ones(3, dtype=np.float32) / 3,
            sample_count=np.asarray(10),
        )
        np.savez_compressed(
            prepared / "stage16b_scalar_normalization.npz",
            mean=np.zeros(len(SCALAR_NAMES), dtype=np.float32),
            scale=np.ones(len(SCALAR_NAMES), dtype=np.float32),
            names=np.asarray(SCALAR_NAMES),
        )
        seeds = [1, 2, 3]
        checkpoints = []
        for seed in seeds:
            path = self.stage16b / "stage16b_seed_{}.pt".format(seed)
            path.write_bytes("checkpoint-{}".format(seed).encode())
            checkpoints.append(str(path.resolve()))
        experiment = {
            "schema": "stage16b_recurrent_cql_v1",
            "source_run_fingerprint": behavior_run["fingerprint"],
            "source_freeze_fingerprint": frozen["fingerprint"],
            "prepared_fingerprint": prepared_contract["fingerprint"],
            "seeds": seeds,
            "primary_budget": 30.0,
        }
        experiment["fingerprint"] = fingerprint(experiment)
        write_json(self.stage16b / "stage16b_experiment.json", experiment)
        _write_csv(
            self.stage16b / "stage16b_budget_summary.csv",
            [
                {"split": "train", "budget_percent": 30.0, "threshold": -0.25},
                {"split": "val", "budget_percent": 30.0, "threshold": 99.0},
            ],
        )
        model_config = {
            "token_dim": 3,
            "scalar_dim": len(SCALAR_NAMES),
            "hidden_dim": 8,
            "spatial_channels": 4,
            "num_actions": 7,
        }
        decision = {
            "schema": "stage16b_recurrent_cql_v1",
            "status": "PASS_OFFLINE_Q_SCREEN",
            "source_run_fingerprint": behavior_run["fingerprint"],
            "source_freeze_fingerprint": frozen["fingerprint"],
            "prepared_fingerprint": prepared_contract["fingerprint"],
            "experiment_fingerprint": experiment["fingerprint"],
            "seeds": seeds,
            "model_config": model_config,
            "screen": {"primary_budget": 30.0},
            "artifacts": {
                "checkpoints": checkpoints,
                "token_pca": str((prepared / "stage16b_token_pca.npz").resolve()),
                "scalar_normalization": str(
                    (prepared / "stage16b_scalar_normalization.npz").resolve()
                ),
            },
            "code_sha256": {
                "stage16_recurrent_cql.py": file_hash(
                    Path(__file__).resolve().parent / "stage16_recurrent_cql.py"
                )
            },
        }
        decision["fingerprint"] = fingerprint(decision)
        write_json(self.stage16b / "stage16b_decision.json", decision)

    def test_bundle_uses_train_threshold_and_detects_artifact_change(self):
        bundle = build_policy_bundle(self.stage16b, self.freeze_path)
        self.assertEqual(bundle["candidate_advantage_threshold"], -0.25)
        self.assertEqual(bundle["state_grid"], 4)
        path = self.root / "policy.json"
        write_json(path, bundle)
        _, loaded = load_policy_bundle(path)
        self.assertEqual(loaded["fingerprint"], bundle["fingerprint"])

        checkpoint = self.stage16b / "stage16b_seed_1.pt"
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "size changed|content changed"):
            load_policy_bundle(path)


class OnlineFeatureTests(unittest.TestCase):
    def test_single_step_features_match_training_builder(self):
        episodes, steps = 1, 4
        image_hw = np.asarray([[100, 200]] * steps, dtype=np.int64)
        previous_state = np.asarray(
            [[[10, 20, 30, 40], [11, 21, 30, 40], [12, 22, 30, 40], [13, 23, 30, 40]]],
            dtype=np.float64,
        )
        previous_fast = previous_state + 1.0
        fast_box = previous_state + 2.0
        selector = np.asarray(
            [[[0.1, 0.2, 0.3, 0.15, 0.15, 0.1]]] * steps,
            dtype=np.float32,
        ).reshape(episodes, steps, 6)
        data = {
            "previous_action": np.asarray([[-1, 0, 3, 1]], dtype=np.int64),
            "selected_layer": np.asarray([[9, 9, 9, 9]], dtype=np.int64),
            "selector_prob": selector,
            "previous_state": previous_state,
            "previous_fast_box": previous_fast,
            "fast_box": fast_box,
            "image_hw": image_hw,
            "candidate_used_before": np.asarray([[0, 0, 1, 1]], dtype=np.int64),
            "candidate_remaining_before": np.asarray([[2, 2, 1, 1]], dtype=np.int64),
            "remaining_candidate_rate": np.asarray([[0.5, 2 / 3, 0.5, 1.0]]),
            "resize_factor": np.asarray([[1.0, 1.1, 1.2, 1.3]]),
        }
        expected = scalar_features(data)
        for step in range(steps):
            actual = single_step_scalar_features(
                selector[0, step],
                9,
                previous_state[0, step],
                previous_fast[0, step],
                fast_box[0, step],
                data["previous_action"][0, step],
                image_hw[step],
                data["candidate_used_before"][0, step],
                data["candidate_remaining_before"][0, step],
                steps,
                step,
                data["resize_factor"][0, step],
            )
            np.testing.assert_allclose(actual, expected[0, step], rtol=0, atol=1e-7)

    def test_exact_budget_schedule(self):
        for steps in (3, 17, 79):
            count = candidate_budget(steps, 30.0)
            schedule = uniform_exact_schedule(steps, count)
            self.assertEqual(int(schedule.sum()), count)


class DecisionTests(unittest.TestCase):
    def test_decision_requires_fast_and_matched_cost_superiority(self):
        run = {"fingerprint": "run", "manifest_fingerprint": "manifest", "split": "val"}
        bundle = {
            "fingerprint": "bundle",
            "primary_budget_percent": 30.0,
        }
        summaries = [
            {
                "method": "tavrtrack_p30",
                "num_sequences": 40,
                "candidate_blocks_per_frame": 0.29,
                "active_candidate_layers": 5,
            }
        ]
        paired = [
            {"reference": "fast", "mean_delta": 0.3, "ci_low": 0.1},
            {"reference": "uniform_l9_p30", "mean_delta": 0.2, "ci_low": 0.05},
            {"reference": "sgla", "mean_delta": -0.1, "ci_low": -0.3},
        ]
        passed = make_decision(run, bundle, summaries, paired, True, 100, 1)
        self.assertEqual(passed["status"], "PASS_CLOSED_LOOP_DEVELOPMENT")
        paired[1] = {
            "reference": "uniform_l9_p30",
            "mean_delta": 0.02,
            "ci_low": -0.05,
        }
        failed = make_decision(run, bundle, summaries, paired, True, 100, 1)
        self.assertEqual(failed["status"], "FAIL_ADAPTIVE_VALUE")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class RecurrentDeploymentTests(unittest.TestCase):
    def test_projection_reproduces_prepared_float16_contract(self):
        import torch

        deployer = FrozenTavrPolicy.__new__(FrozenTavrPolicy)
        deployer.torch = torch
        deployer.pca_mean = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32)
        deployer.pca_components = torch.tensor(
            [[0.3, 0.7, -0.2], [-0.6, 0.1, 0.5]], dtype=torch.float32
        )
        values = torch.tensor(
            [[0.123456, -0.765432, 1.234567]], dtype=torch.float32
        )
        source = values.to(dtype=torch.float16).float()
        expected = (
            (source - deployer.pca_mean) @ deployer.pca_components.t()
        ).to(dtype=torch.float16).float()
        torch.testing.assert_close(deployer._project(values), expected, rtol=0, atol=0)

    def test_online_steps_equal_full_causal_forward(self):
        import torch
        import torch.nn as nn

        config = {
            "token_dim": 3,
            "scalar_dim": len(SCALAR_NAMES),
            "hidden_dim": 8,
            "spatial_channels": 4,
            "num_actions": 7,
        }
        torch.manual_seed(4)
        model = build_q_model(torch, nn, config).eval()
        batch = {
            "search_grid": torch.randn(1, 5, 4, 4, 3),
            "template": torch.randn(1, 5, 2, 3),
            "score_map": torch.randn(1, 5, 1, 16, 16),
            "scalar": torch.randn(1, 5, len(SCALAR_NAMES)),
        }
        with torch.no_grad():
            expected = model(batch)
            deployer = FrozenTavrPolicy.__new__(FrozenTavrPolicy)
            deployer.torch = torch
            hidden = None
            observed = []
            for step in range(5):
                one = {key: value[:, step : step + 1] for key, value in batch.items()}
                q, hidden = deployer._step_model(model, one, hidden)
                observed.append(q)
            observed = torch.stack(observed, dim=1)
        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main()
