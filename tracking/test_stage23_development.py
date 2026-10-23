"""Protocol/evaluation/statistics tests, with optional CPU PyTorch integration."""

import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from stage4_common import file_hash, write_csv, write_json
from stage23_branch_credit import ARMS
from stage23_development import (CONTROLS, METRIC, SCHEMA, SEEDS, evaluate_video, read_sealed,
                                 sealed, train_arm, update_policy, valid_auc, verify_smoke)
from stage23_development_summary import audit_evaluation, load_complete, paired_statistics, progress, summarize


class ProtocolTests(unittest.TestCase):
    def test_metric_excludes_initial_and_invalid_frames(self):
        self.assertAlmostEqual(valid_auc([1., -1.], np.array([True, False])), 100 * 20 / 21)
        self.assertAlmostEqual(valid_auc([0., 1.], np.ones(2, bool)), 100 * 10 / 21)
        for values, valid in (([1.], np.array([False])), ([np.nan], np.array([True])), ([2.], np.array([True]))):
            with self.assertRaises(ValueError):
                valid_auc(values, valid)

    def test_report_integrity_and_partial_progress(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp)
            write_json(path / "stage23b_contract.json", sealed(dict(schema=SCHEMA)))
            _, rows = progress(path)
            self.assertEqual(len(rows), 9)
            self.assertFalse(any(r["completed"] for r in rows))
            write_json(path / "bad.json", dict(schema=SCHEMA, fingerprint="tampered"))
            with self.assertRaises(ValueError):
                read_sealed(path / "bad.json")

    def smoke_fixture(self, path):
        names = ("stage23_branch_credit.py", "stage23_branch_ppo.py", "stage21_counterfactual_runtime.py",
                 "stage21_counterfactual_math.py", "stage18_environment.py", "stage18_policy_runtime.py")
        contract = sealed(dict(schema="stage23_branch_credit_smoke_v1", bundle_fingerprint="bundle",
                               code_sha256={n: file_hash(Path(__file__).parent / n) for n in names}))
        write_json(path / "stage23_contract.json", contract)
        rows = []
        for arm in ARMS:
            write_json(path / arm / "last.pt", {"test": "opaque checkpoint bytes, never unpickled here"})
            rows.append(dict(arm=arm, live_transitions=1024, branch_transitions=4480, optimizer_steps=16,
                actor_changed=True, frozen_representation_unchanged=True, initial_model_sha256="initial",
                maximum_replay_error=2e-7, checkpoint_sha256=file_hash(path / arm / "last.pt"), seconds=10.))
        write_json(path / "stage23_decision.json", sealed(dict(status="PASS_BRANCH_PPO_SMOKE",
            contract_fingerprint=contract["fingerprint"], arms=rows)))

    def test_passed_smoke_requires_original_artifact_bytes(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp)
            self.smoke_fixture(path)
            evidence = verify_smoke(path, "bundle")
            self.assertGreater(evidence["rough_training_seconds"], 0)
            with self.assertRaisesRegex(ValueError, "matching"):
                verify_smoke(path, "other bundle")
            write_json(path / ARMS[0] / "last.pt", {"test": "changed"})
            with self.assertRaisesRegex(ValueError, "artifacts"):
                verify_smoke(path, "bundle")

    def test_cached_evaluation_rejects_budget_and_auc_tampering(self):
        contract = {"fingerprint": "contract"}
        row = dict(contract_fingerprint="contract", policy_sha256="policy", method="policy", sequence="video",
                   metric=METRIC, frames=4, valid_frames=4, boxes=[[0., 0., 20., 20.]] * 4,
                   valid=[True] * 4, overlaps=[1.] * 4, actions=[0, 0, 0, 1], candidate_blocks=1,
                   auc=valid_auc([1.] * 4, np.ones(4, bool)), mean_crop_model_decode_ms=1.)
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "video.json"
            write_json(path, sealed(row))
            audit_evaluation(path, contract, "policy", "policy")
            write_json(path, sealed(dict(row, actions=[1, 0, 0, 0])))
            with self.assertRaisesRegex(ValueError, "budget"):
                audit_evaluation(path, contract, "policy", "policy")
            write_json(path, sealed(dict(row, auc=100.)))
            with self.assertRaisesRegex(ValueError, "AUC"):
                audit_evaluation(path, contract, "policy", "policy")

    def test_complete_nine_run_audit_and_changed_checkpoint_rejection(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            contract = sealed(dict(schema=SCHEMA, tune=["a", "b"], bundle_fingerprint="bundle",
                                   bootstrap_samples=100, analysis_seed=2077))
            def evaluations(directory, method, policy_id):
                references = []
                for name in contract["tune"]:
                    actions = ([-1] * 4 if method == "sgla" else [0] * 4 if method in ("fast", "memory_fast")
                               else [0, 0, 0, int(method[1:]) - 6 if method.startswith("l") else 1])
                    path = directory / (name + ".json")
                    write_json(path, sealed(dict(contract_fingerprint=contract["fingerprint"], policy_sha256=policy_id,
                        sequence=name, method=method, metric=METRIC, frames=4, valid_frames=4, valid=[True] * 4,
                        boxes=[[0., 0., 20., 20.]] * 4, overlaps=[1.] * 4, actions=actions,
                        candidate_blocks=sum(a != 0 for a in actions), auc=100 * 20 / 21, mean_crop_model_decode_ms=1.)))
                    references.append(dict(sequence=name, sha256=file_hash(path)))
                return references
            controls = {method: evaluations(root / "controls" / method, method, "bundle") for method in CONTROLS}
            write_json(root / "controls_complete.json", sealed(dict(contract_fingerprint=contract["fingerprint"],
                                                                    evaluations=controls)))
            for seed in SEEDS:
                for arm in ARMS:
                    path = root / ("seed" + str(seed)) / arm
                    write_json(path / "last.pt", {"opaque": "no pickle load in CPU audit"})
                    digest = file_hash(path / "last.pt")
                    refs = evaluations(path / "evaluation", "policy", digest)
                    write_csv(path / "history.csv", [dict(seed=seed, arm=arm, steps=256, anchor_count=4,
                        branch_transitions=1120, optimizer_steps=4, valid_transitions=256, likelihood_replay_error=1e-7,
                        probability_replay_error=1e-7, chosen_replay_error=0, live_candidate_blocks=70,
                        branch_candidate_blocks=300, valid_anchors=4, seconds=1.)])
                    write_json(path / "complete.json", sealed(dict(schema=SCHEMA, contract_fingerprint=contract["fingerprint"],
                        seed=seed, arm=arm, steps=256, frozen_representation_unchanged=True, actor_changed=True,
                        initial_model_sha256="initial" + str(seed), final_model_sha256="trained",
                        checkpoint_sha256=digest, history_sha256=file_hash(path / "history.csv"), evaluations=refs)))
            with patch("stage23_development_summary.STEPS", 256):
                trajectories, costs = load_complete(root, contract)
                self.assertEqual(len(costs), 9)
                self.assertEqual(len(trajectories[ARMS[-1]]), 3)
                _, _, decision = summarize(trajectories, contract)
                self.assertEqual(decision["status"], "NO_CREDIT_EFFECT_ESTABLISHED")
                write_json(root / "seed2074" / ARMS[0] / "last.pt", {"changed": True})
                with self.assertRaisesRegex(ValueError, "integrity"):
                    load_complete(root, contract)


class StatisticsTests(unittest.TestCase):
    def test_seed_repeats_are_not_additional_videos(self):
        left = np.array([[1., -1., 2., -2.]])
        once = paired_statistics(left, np.zeros_like(left), 1000)
        repeated = paired_statistics(np.repeat(left, 3, 0), np.zeros((3, 4)), 1000)
        self.assertEqual(once["ci_low"], repeated["ci_low"])
        self.assertEqual(once["ci_high"], repeated["ci_high"])
        self.assertEqual(repeated["seed_mean_std"], 0.)

    def test_seed_pairing_and_positive_seed_count(self):
        left = np.array([[2., 2., 2.], [0., 0., 0.], [1., 1., 1.]])
        right = np.ones((3, 3))
        result = paired_statistics(left, right, 500)
        self.assertEqual(result["mean_delta"], 0.)
        self.assertEqual(result["positive_seed_means"], 1)
        self.assertEqual(result["ties"], 3)
        self.assertLess(result["seed_video_ci_low"], 0)
        self.assertGreater(result["seed_video_ci_high"], 0)

    def trajectories(self, values):
        result = {}
        for method in ARMS + CONTROLS:
            result[method] = [[dict(auc=float(v), frames=100, candidate_blocks=0, actions=[0] * 100,
                                   mean_crop_model_decode_ms=1.) for v in values[method]]
                              for _ in range(3 if method in ARMS else 1)]
        return result

    def test_credit_success_is_distinct_from_competitive_policy(self):
        values = {method: [80.] * 5 for method in ARMS + CONTROLS}
        values.update({ARMS[1]: [81.] * 5, ARMS[2]: [82.] * 5, "l9": [83.] * 5})
        contract = dict(tune=list(range(5)), bootstrap_samples=500, analysis_seed=2077)
        _, _, decision = summarize(self.trajectories(values), contract)
        self.assertEqual(decision["status"], "CREDIT_EFFECT_ONLY_NOT_COMPETITIVE")
        values[ARMS[2]] = [81.] * 5
        _, _, decision = summarize(self.trajectories(values), contract)
        self.assertEqual(decision["status"], "NO_CREDIT_EFFECT_ESTABLISHED")
        values[ARMS[2]] = [84.] * 5
        _, _, decision = summarize(self.trajectories(values), contract)
        self.assertEqual(decision["status"], "DEVELOPMENT_POLICY_GAIN_SUPPORTED")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.old_threads)

    def make_system(self):
        from test_stage18_temporal import ModelEnvironmentTests
        from stage18_policy_runtime import configure_policy
        torch, modules, rt = ModelEnvironmentTests().make_system()
        configure_policy(modules)
        modules.eval()
        rt.read = lambda x: x
        source = dict(sequence="fit", paths=[np.zeros((32, 32, 3)) for _ in range(65)],
                      gt=np.tile([0, 0, 20, 20], (65, 1)), valid=np.ones(65, bool),
                      terminal=True, bootstrap_path=None)
        return torch, modules, rt, source

    def test_greedy_evaluation_has_no_branches_or_extra_heads(self):
        torch, modules, rt, source = self.make_system()
        policy = evaluate_video(rt, modules, source, "policy")
        self.assertEqual(rt.tracker.head_calls, 64)
        self.assertEqual(policy["candidate_blocks"], 0)  # Initial actor argmax is Fast.
        static = evaluate_video(rt, modules, source, "l7")
        self.assertEqual(rt.tracker.head_calls, 128)
        self.assertEqual(static["candidate_blocks"], 64 * 3 // 10)
        self.assertEqual(policy["auc"], static["auc"])

    def test_full_update_matches_smoke_on_valid_data_and_handles_no_labels(self):
        from stage23_branch_credit import BranchCollector
        from stage23_branch_ppo import update_policy as smoke_update
        torch, modules, rt, source = self.make_system()
        data = BranchCollector(rt, modules, [source], SEEDS[0]).collect(256)
        other = copy.deepcopy(modules)
        opt = lambda m: torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.0003, eps=.00001)
        smoke_update(torch, modules, opt(modules), data, ARMS[-1], SEEDS[0], 1)
        update_policy(torch, other, opt(other), data, ARMS[-1], SEEDS[0], 1)
        for a, b in zip(modules.parameters(), other.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
        torch, modules, rt, source = self.make_system()
        data = BranchCollector(rt, modules, [source], SEEDS[0]).collect(256)
        for anchor in data["counterfactual"]:
            anchor["valid"] = False
        before = copy.deepcopy(modules.state_dict())
        data["valid"][:] = False
        result = update_policy(torch, modules, opt(modules), data, ARMS[-1], SEEDS[0], 1)
        self.assertEqual(result["optimizer_steps"], 0)
        for name, value in modules.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_atomic_resume_matches_uninterrupted_weights(self):
        from stage23_branch_credit import BranchCollector
        torch, initial, rt, source = self.make_system()
        factory = lambda *args: copy.deepcopy(initial)
        original_collect = BranchCollector.collect
        calls = []
        def interrupt_second(collector, count):
            calls.append(count)
            if len(calls) == 2:
                raise RuntimeError("test interruption")
            return original_collect(collector, count)
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            with patch("stage23_development.STEPS", 512), patch("stage23_development.build_model", factory), \
                    patch.object(torch.cuda, "synchronize", lambda: None):
                with patch.object(BranchCollector, "collect", interrupt_second):
                    with self.assertRaisesRegex(RuntimeError, "test interruption"):
                        train_arm(rt, {}, root, [source], [source], ARMS[-1], SEEDS[0], root / "resume", {"fingerprint": "test"})
                train_arm(rt, {}, root, [source], [source], ARMS[-1], SEEDS[0], root / "resume", {"fingerprint": "test"})
                train_arm(rt, {}, root, [source], [source], ARMS[-1], SEEDS[0], root / "continuous", {"fingerprint": "test"})
            a, b = read_sealed(root / "resume" / "complete.json"), read_sealed(root / "continuous" / "complete.json")
            self.assertEqual(a["final_model_sha256"], b["final_model_sha256"])


if __name__ == "__main__":
    unittest.main()
