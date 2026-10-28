"""Stage-26 contracts and real-torch current-version gradient/replay tests."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from stage4_common import file_hash, write_cache, write_csv, write_json
from stage26_value_common import (ANCHORS, ARMS, FILES, HORIZON, LIB_FILES, SCHEMA, SEEDS,
    branch_actions, example_schedule, make_contract, pairwise_value_loss, planned_jobs, post_values,
    read_sealed, sealed, split_videos, validate_sample, verify_contract)
from stage26_value_runtime import current_losses
from stage26_value_summary import paired_row, summarize_smoke, validate_route


def fixture_contract(mode="smoke"):
    names = ["v{:03d}".format(i) for i in range(95)]
    train, check, unused = split_videos(names)
    groups = dict(train=train, check=check, unused_fit=unused, excluded_tune=["old_tune"], excluded_validation=["old_val"])
    entries = [dict(sequence=n, start=0, stop=130, split="train") for n in names]
    return make_contract(dict(fingerprint="bundle"), dict(fingerprint="manifest", sequences=entries),
                         groups, "stage25_failure", {"embed_dim": 8}, mode)


def fixture_sample():
    actions, anchors = example_schedule(64, SEEDS[0], "v", 0)
    choices = branch_actions(actions, int(anchors[0]), SEEDS[0], "v", 0)
    rewards = np.random.default_rng(3).random((2, 6, 20))
    rewards[1, :, 0] = rewards[0, :, 0]
    valid = np.ones(20, bool)
    q, eligible = post_values(rewards, valid)
    boxes = np.tile([1., 2., 10., 20.], (20, 1))
    data = dict(rewards=rewards, valid=valid, costs=np.full((2, 6), np.count_nonzero(choices[0]), dtype=int),
        continuation=choices, replay_boxes=boxes, live_boxes=boxes.copy(), target=q.mean(0))
    meta = dict(eligible=eligible, teacher_prefix_sha256="a" * 64, teacher_head_sha256="b" * 64)
    return data, meta


class ContractTests(unittest.TestCase):
    def test_split_is_fixed_disjoint_and_does_not_resample_for_labels(self):
        names = ["v{:03d}".format(i) for i in range(95)]
        a, b, c = split_videos(names)
        self.assertEqual((len(a), len(b), len(c)), (32, 16, 47))
        self.assertEqual((a, b, c), split_videos(names))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a + b + c), set(names))

    def test_duplicate_or_insufficient_source_ids_rejected(self):
        for names in (["v"] * 95, [str(i) for i in range(47)]):
            with self.assertRaises(ValueError):
                split_videos(names)

    def test_smoke_and_full_preserve_same_base_split(self):
        smoke, full = fixture_contract(), fixture_contract("full")
        self.assertEqual(smoke["groups"], full["groups"])
        self.assertEqual(smoke["training"], full["training"][:3])
        self.assertEqual(smoke["checking"], [])
        self.assertEqual(full["checking"], full["groups"]["check"])
        self.assertEqual(smoke["expected_attempts"], 12)
        self.assertEqual(full["expected_attempts"], 256)
        self.assertFalse(full["ppo_trained"])

    def test_jobs_match_canonical_cost_and_complete_horizon(self):
        for mode in ("smoke", "full"):
            contract = fixture_contract(mode)
            jobs = planned_jobs(contract, SEEDS[0])
            self.assertEqual(len(jobs), contract["expected_attempts"])
            for name, round_index, anchor, actions in jobs:
                self.assertTrue(actions[anchor] > 0)
                self.assertLessEqual(anchor + 20, len(actions))
                np.testing.assert_array_equal((actions > 0).cumsum(), 3 * np.arange(1, len(actions) + 1) // 10)
            self.assertEqual([j[:3] for j in jobs], [j[:3] for j in planned_jobs(contract, SEEDS[0])])

    def test_contract_code_hashes_and_no_torch_import(self):
        contract = fixture_contract()
        with tempfile.TemporaryDirectory() as d, patch.dict("sys.modules", {"torch": None}):
            write_json(Path(d) / "stage26_contract.json", contract)
            self.assertEqual(verify_contract(d), contract)
            bad = copy.deepcopy(contract)
            bad["code_sha256"][FILES[0]] = "changed"
            bad = sealed({k: v for k, v in bad.items() if k != "fingerprint"})
            write_json(Path(d) / "stage26_contract.json", bad)
            with self.assertRaisesRegex(ValueError, "dependency changed"):
                verify_contract(d)


class TargetTests(unittest.TestCase):
    def test_action_frame_is_excluded(self):
        data, _ = fixture_sample()
        before = post_values(data["rewards"], data["valid"])[0]
        data["rewards"][:, :, 0] = .99
        after = post_values(data["rewards"], data["valid"])[0]
        np.testing.assert_array_equal(before, after)

    def test_visibility_normalization_does_not_create_false_damage(self):
        rewards = np.full((2, 6, 20), .7)
        valid = np.ones(20, bool)
        valid[3:6] = False
        rewards[..., ~valid] = 0
        q, eligible = post_values(rewards, valid)
        np.testing.assert_allclose(q, .7)
        self.assertTrue(eligible)
        valid[0] = False
        rewards[:, :, 0] = 0
        self.assertFalse(post_values(rewards, valid)[1])

    def test_invalid_reward_or_first_repeat_mismatch_rejected(self):
        data, _ = fixture_sample()
        data["rewards"][1, 0, 0] += .01
        with self.assertRaises(ValueError):
            post_values(data["rewards"], data["valid"])

    def test_sample_replay_cost_and_target_tampering(self):
        data, meta = fixture_sample()
        validate_sample(data, meta)
        for name in ("target", "costs", "replay_boxes"):
            changed = copy.deepcopy(data)
            changed[name].flat[0] += .1 if name != "costs" else 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_sample(changed, meta)

    def test_stale_model_returns_rejected_before_touching_runtime(self):
        data, meta = fixture_sample()
        with patch("stage26_value_runtime.model_digest", return_value="new_version"):
            with self.assertRaisesRegex(ValueError, "stale returns"):
                current_losses(None, None, None, object(), None, data, meta, ARMS[1])


class AnalysisTests(unittest.TestCase):
    def test_paired_seeds_do_not_create_48_independent_videos(self):
        left, right = np.ones((3, 16)), np.zeros((3, 16))
        row = paired_row(left, right, "test", family=6)
        self.assertEqual(row["wins"], 16)
        self.assertEqual(row["positive_seed_means"], 3)
        self.assertEqual(row["mean_delta"], 1)

    def test_checking_metric_and_fixed_layer_schedule_are_audited(self):
        from stage23_development import valid_auc
        actions, _ = example_schedule(64, 1, "x", 0)
        actions[actions > 0] = 3
        valid = np.ones(64, bool)
        record = dict(actions=actions.tolist(), boxes=np.tile([0., 0., 10., 10.], (64, 1)).tolist(),
            valid=valid.tolist(), overlaps=np.full(64, .8).tolist(), frames=64,
            candidate_blocks=int((actions > 0).sum()), auc=valid_auc(np.full(64, .8), valid))
        validate_route(record, "l9", 64)
        record["actions"][3] = 1
        with self.assertRaises(ValueError):
            validate_route(record, "l9", 64)

    def test_complete_smoke_is_reconstructed_without_loading_checkpoints(self):
        contract = fixture_contract()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root / "stage26_contract.json", contract)
            for arm in ARMS:
                directory = root / ("seed" + str(SEEDS[0])) / arm
                directory.mkdir(parents=True)
                identity = dict(contract_fingerprint=contract["fingerprint"], seed=SEEDS[0], arm=arm,
                                initial_prefix_sha256="a" * 64, initial_head_sha256="b" * 64)
                history = []
                prefix, head = identity["initial_prefix_sha256"], identity["initial_head_sha256"]
                for i, (name, round_index, anchor, actions) in enumerate(planned_jobs(contract, SEEDS[0])):
                    data, metadata = fixture_sample()
                    data["continuation"] = branch_actions(actions, anchor, SEEDS[0], name, round_index)
                    data["costs"][:] = np.count_nonzero(data["continuation"][0])
                    metadata.update(identity=identity, attempt=i, sequence=name, round=round_index, anchor=anchor,
                                    teacher_prefix_sha256=prefix, teacher_head_sha256=head)
                    sample = directory / "samples" / ("{:05d}.npz".format(i))
                    write_cache(sample, data, metadata)
                    next_prefix, next_head = ("{:064x}".format(i + 1), "{:064x}".format(i + 100))
                    history.append(dict(attempt=i, sequence=name, round=round_index, anchor=anchor, eligible=True,
                        optimizer_steps=1, branch_frames=240, prefix_frames=anchor + 1, replay_frames=19,
                        prefix_before_sha256=prefix, head_before_sha256=head, prefix_after_sha256=next_prefix,
                        head_after_sha256=next_head, sample_sha256=file_hash(sample), full_h20_replay_error=0.,
                        maximum_gradient_norm=.3, seconds=.1, rank_loss=.2, tracking_loss=.3,
                        rank_prefix_gradient=0. if arm == ARMS[0] else .1, tracking_prefix_gradient=.2,
                        differentiable_replay_error=0.))
                    prefix, head = next_prefix, next_head
                write_csv(directory / "history.csv", history)
                (directory / "last.pt").write_bytes(b"test-only opaque checkpoint: do not deserialize")
                training = sealed(dict(schema=SCHEMA, **identity, attempts=len(history), optimizer_steps=len(history),
                    branch_frames=len(history) * 240, maximum_rank_prefix_gradient=0. if arm == ARMS[0] else .1,
                    maximum_tracking_prefix_gradient=.2, maximum_replay_error=0., prefix_changed=True, head_changed=True,
                    frozen_runtime_and_memory_unchanged=True, frozen_memory_sha256="c" * 64,
                    final_prefix_sha256=prefix, final_head_sha256=head, checkpoint_sha256=file_hash(directory / "last.pt"),
                    history_sha256=file_hash(directory / "history.csv")))
                write_json(directory / "training_complete.json", training)
                write_json(directory / "complete.json", sealed(dict(schema=SCHEMA, contract_fingerprint=contract["fingerprint"],
                    seed=SEEDS[0], arm=arm, training=training, evaluations=[])))
            with patch.dict("sys.modules", {"torch": None}):
                result = summarize_smoke(root, contract)
            self.assertEqual(result["status"], "PASS_VALUE_REPRESENTATION_SMOKE")
            self.assertFalse(result["checking_labels_loaded"])
            broken = root / ("seed" + str(SEEDS[0])) / ARMS[0] / "samples/00000.npz"
            broken.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                summarize_smoke(root, contract)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: real gradient/replay tests must run on server")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def system(self):
        import torch
        from stage18_temporal_model import build_temporal_modules
        from stage26_value_runtime import build_value_head, trainable_prefix
        torch.manual_seed(1)
        nn = torch.nn
        net = nn.Module()
        net.backbone = nn.Module()
        net.backbone.blocks = nn.Sequential(*[nn.Linear(8, 8) for _ in range(12)])
        net.box = nn.Linear(8, 4)
        net.score = nn.Linear(8, 1)
        tracker = SimpleNamespace(state=[0., 0., 10., 10.], frame_id=0,
                                  params=SimpleNamespace(search_size=32), head_calls=0, prefix_calls=0)
        def initialize(image, box):
            tracker.state, tracker.frame_id = list(box), 0
        def prefix(image):
            tracker.prefix_calls += 1
            value = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) / 100 + float(image[0, 0, 0]) / 100
            for block in net.backbone.blocks[:6]:
                value = block(value)
            return value
        def head(tokens):
            tracker.head_calls += 1
            search = tokens[:, 4:]
            return dict(pred_boxes=net.box(search.mean(1)).sigmoid()[:, None],
                        score_map=net.score(search).sigmoid().reshape(1, 1, 2, 2))
        rt = SimpleNamespace(torch=torch, net=net, backbone=net.backbone, tracker=tracker,
            initialize=initialize, read=lambda path: np.full((32, 32, 3), float(path)),
            prepare=lambda image: (image, 1.), prefix=prefix, head=head,
            decode=lambda output, factor, image: list(tracker.state), identity={"embed_dim": 8})
        selected = trainable_prefix(rt)
        modules = build_temporal_modules(torch, nn, 8, 4, 2).eval().requires_grad_(False)
        with torch.no_grad():
            modules.read_out.weight.fill_(.01)
        value_head = build_value_head(torch, 8).eval()
        source = dict(sequence="toy", paths=list(range(65)), gt=np.tile([0., 0., 10., 10.], (65, 1)), valid=np.ones(65, bool))
        return torch, rt, selected, modules, value_head, source

    def test_pairwise_loss_common_offset_and_target_detachment(self):
        import torch
        prediction = torch.randn(1, 6, requires_grad=True)
        target = torch.randn(6, requires_grad=True)
        first = pairwise_value_loss(torch, prediction, target)
        second = pairwise_value_loss(torch, prediction + 10, target + 10)
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
        first.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_value_gradient_reaches_real_prefix_only_in_B(self):
        from stage26_value_runtime import collect_example, current_losses
        for arm in ARMS:
            torch, rt, prefix, modules, head, source = self.system()
            actions, anchors = example_schedule(64, SEEDS[0], source["sequence"], 0)
            data, meta, context = collect_example(rt, modules, head, prefix, source, actions, int(anchors[0]), SEEDS[0], 0)
            loss, metrics = current_losses(rt, modules, head, prefix, context, data, meta, arm)
            self.assertGreater(metrics["tracking_prefix_gradient"], 0)
            if arm == "stop_gradient":
                self.assertEqual(metrics["rank_prefix_gradient"], 0)
            else:
                self.assertGreater(metrics["rank_prefix_gradient"], 0)
            loss.backward()
            self.assertTrue(all(p.grad is None for p in rt.backbone.blocks[:4].parameters()))
            self.assertTrue(all(p.grad is None for p in rt.backbone.blocks[6:].parameters()))
            self.assertTrue(all(p.grad is None for p in modules.parameters()))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in prefix.parameters()))

    def test_one_head_and_one_prefix_per_greedy_deployed_frame(self):
        from stage26_value_runtime import value_environment
        _, rt, _, modules, head, source = self.system()
        counts = [0] * 12
        for index, block in enumerate(rt.backbone.blocks):
            block.register_forward_hook(lambda module, args, output, i=index: counts.__setitem__(i, counts[i] + 1))
        env = value_environment(rt, modules, head)
        env.reset(rt.read(0), source["gt"][0])
        calls = []
        for index in range(1, 65):
            calls.append(int(env.step(rt.read(index))["action"] > 0))
        self.assertEqual(rt.tracker.head_calls, 64)
        self.assertEqual(rt.tracker.prefix_calls, 64)
        self.assertEqual(counts[:6], [64] * 6)
        self.assertEqual(sum(counts[6:]), 19)
        np.testing.assert_array_equal(np.cumsum(calls), 3 * np.arange(1, 65) // 10)

    def test_actual_adam_updates_change_prefix_but_preserve_other_runtime_weights(self):
        from stage26_value_runtime import collect_example, current_losses
        from stage26_value_train import frozen_parameters, verify_frozen_runtime
        from stage23_branch_ppo import model_digest
        torch, rt, prefix, modules, head, source = self.system()
        frozen, memory_id = frozen_parameters(rt), model_digest(modules)
        before = model_digest(prefix)
        actions, anchors = example_schedule(64, SEEDS[0], source["sequence"], 0)
        data, meta, context = collect_example(rt, modules, head, prefix, source, actions, int(anchors[0]), SEEDS[0], 0)
        loss, _ = current_losses(rt, modules, head, prefix, context, data, meta, ARMS[1])
        optimizer = torch.optim.AdamW(list(prefix.parameters()) + list(head.parameters()), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertNotEqual(before, model_digest(prefix))
        verify_frozen_runtime(torch, rt, frozen, modules, memory_id)
        with self.assertRaisesRegex(ValueError, "stale"):
            current_losses(rt, modules, head, prefix, context, data, meta, ARMS[1])

    def test_training_atomic_resume_does_not_recollect_committed_examples(self):
        from stage26_value_train import cpu_state, train_arm
        torch, rt, prefix, modules, head, source = self.system()
        with tempfile.TemporaryDirectory() as d:
            directory, repdir = Path(d) / "arm", Path(d) / "rep"
            repdir.mkdir()
            torch.save(dict(epoch=1, contract_fingerprint="rep", model=modules.state_dict()), repdir / "best.pt")
            bundle = dict(model=dict(embed_dim=8, lens_z=4, search_side=2), best_epoch=1,
                          representation_contract_fingerprint="rep")
            contract = dict(fixture_contract(), training=["toy"], clip_steps={"toy": 64}, expected_attempts=4, rounds=1)
            contract = sealed({k: v for k, v in contract.items() if k != "fingerprint"})
            original = cpu_state(prefix)
            _, _, _, first = train_arm(rt, bundle, repdir, [source], ARMS[0], SEEDS[0], directory, contract, original)
            with patch("stage26_value_train.collect_example", side_effect=AssertionError("Recollected committed examples")):
                _, _, _, second = train_arm(rt, bundle, repdir, [source], ARMS[0], SEEDS[0], directory, contract, original)
            self.assertEqual(first, second)
            self.assertEqual(first["attempts"], 4)

    def test_interrupted_training_matches_uninterrupted_parameters(self):
        import stage26_value_train as trainer
        from stage26_value_train import cpu_state
        from stage18_representation_train import load_checkpoint
        torch, rt, prefix, modules, head, source = self.system()
        with tempfile.TemporaryDirectory() as d:
            repdir = Path(d) / "rep"
            repdir.mkdir()
            torch.save(dict(epoch=1, contract_fingerprint="rep", model=modules.state_dict()), repdir / "best.pt")
            bundle = dict(model=dict(embed_dim=8, lens_z=4, search_side=2), best_epoch=1,
                          representation_contract_fingerprint="rep")
            contract = dict(fixture_contract(), training=["toy"], clip_steps={"toy": 64}, expected_attempts=4, rounds=1)
            contract = sealed({k: v for k, v in contract.items() if k != "fingerprint"})
            original = cpu_state(prefix)
            baseline, interrupted = Path(d) / "baseline", Path(d) / "interrupted"
            trainer.train_arm(rt, bundle, repdir, [source], ARMS[1], SEEDS[0], baseline, contract, original)
            collect, calls = trainer.collect_example, []
            def interrupt(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                return collect(*args, **kwargs)
            with patch.object(trainer, "collect_example", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "interruption"):
                trainer.train_arm(rt, bundle, repdir, [source], ARMS[1], SEEDS[0], interrupted, contract, original)
            saved = load_checkpoint(torch, interrupted / "last.pt")
            self.assertEqual(saved["cursor"], 1)
            trainer.train_arm(rt, bundle, repdir, [source], ARMS[1], SEEDS[0], interrupted, contract, original)
            expected, actual = load_checkpoint(torch, baseline / "last.pt"), load_checkpoint(torch, interrupted / "last.pt")
            for field in ("prefix", "head"):
                for name, value in expected[field].items():
                    self.assertTrue(torch.equal(value, actual[field][name]), field + ":" + name)


if __name__ == "__main__":
    unittest.main()
