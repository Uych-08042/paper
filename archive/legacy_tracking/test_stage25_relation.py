"""CPU integrity tests plus real-torch probe checks when torch is installed."""

import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage18_environment import TokenCredit
from stage25_relation_collect import simulate
from stage25_relation_common import (ANCHORS, CODE_FILES, COLLECTION_SEED, GAMMA, HORIZON,
    RELATION_DIM, REPEATS, SCHEMA, anchor_indices, build_probe, cache_path, continuation,
    discounted_values, keyed_seed, load_cache, raw_dimensions, relation_features, schedule,
    sealed, validate_cache)
from stage25_relation_screen import dataset, grouped_folds, normalized, run, standardizer, summarize


def fixture(sequence="fit_a", dim=3):
    steps = 64
    actions = schedule(steps, keyed_seed(COLLECTION_SEED, sequence, "behavior"))
    anchors = anchor_indices(actions)
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(ANCHORS, raw_dimensions(dim))).astype(np.float32)
    relation = []
    for i, anchor in enumerate(anchors):
        raw[i, -18:-4] = np.r_[np.zeros(4), .8, (3 * (int(anchor) + 1) % 10 + 10) / 20,
                                np.eye(8)[int(actions[anchor - 1]) + 1]]
        raw[i, -4:] = [.375, .375, .25, .25]
        relation.append(relation_features(raw[i, :64 * dim].reshape(dim, 8, 8),
                                         raw[i, 64 * dim:65 * dim], raw[i, -4:]))
    boxes = np.tile([10., 20., 15., 30.], (steps, 1))
    data = dict(anchor=anchors, raw=raw, relation=np.stack(relation),
        rewards=rng.random((ANCHORS, REPEATS, 6, HORIZON)), valid=np.ones((ANCHORS, HORIZON), bool),
        live_actions=actions, live_boxes=boxes, live_valid=np.ones(steps, bool),
        replay_boxes=np.stack([boxes[a:a + HORIZON] for a in anchors]), replay_error=np.zeros(ANCHORS),
        branch_cost=np.stack([np.full((REPEATS, 6), np.count_nonzero(actions[a:a + HORIZON])) for a in anchors]))
    data["rewards"][:, 1, :, 0] = data["rewards"][:, 0, :, 0]
    contract = sealed(dict(schema=SCHEMA, embed_dim=dim, fit=[sequence], excluded_tune=["tune_a"],
        excluded_validation=["val_a"], anchors=ANCHORS, horizon=HORIZON, repeats=REPEATS,
        gamma=GAMMA, seed=COLLECTION_SEED, clip_steps={sequence: steps},
        code_sha256={n: file_hash(Path(__file__).parent / n) for n in CODE_FILES}))
    return contract, data


class ScheduleTests(unittest.TestCase):
    def test_cost_balance_prefix_and_reproducibility(self):
        for steps in (64, 79, 128):
            actions = schedule(steps, 10)
            np.testing.assert_array_equal(actions, schedule(steps, 10))
            np.testing.assert_array_equal((actions > 0).cumsum(), 3 * np.arange(1, steps + 1) // 10)
            counts = np.bincount(actions[actions > 0], minlength=7)[1:]
            self.assertLessEqual(np.ptp(counts), 1)
            budget = TokenCredit()
            for action in actions:
                budget.advance()
                budget.spend(int(action))

    def test_anchor_and_continuation_selection_is_label_independent(self):
        actions = schedule(64, 7)
        anchors = anchor_indices(actions)
        self.assertEqual(len(set(anchors)), ANCHORS)
        self.assertTrue(np.all(anchors + HORIZON <= len(actions)))
        for anchor in anchors:
            paths = continuation(actions, anchor, 100 + int(anchor))
            np.testing.assert_array_equal(paths[0], actions[anchor:anchor + HORIZON])
            np.testing.assert_array_equal(paths[0] > 0, paths[1] > 0)

    def test_short_or_non_candidate_anchor_rejected(self):
        with self.assertRaises(ValueError):
            anchor_indices(schedule(30, 1))
        with self.assertRaises(ValueError):
            continuation(schedule(64, 1), 0, 2)


class RelationTests(unittest.TestCase):
    def test_finite_deterministic_encoding_from_pre_action_fields(self):
        grid = np.random.default_rng(1).normal(size=(3, 8, 8))
        template = np.array([1., 0., 0.])
        first = relation_features(grid, template, [.375, .375, .25, .25])
        self.assertEqual(first.shape, (RELATION_DIM,))
        np.testing.assert_array_equal(first, relation_features(grid, template, [.375, .375, .25, .25]))

    def test_target_distractor_maps_have_declared_meaning(self):
        grid = np.zeros((2, 8, 8))
        grid[1] = 1
        grid[0, 3:5, 3:5], grid[1, 3:5, 3:5] = 1, 0
        relation = relation_features(grid, [1, 0], [.375, .375, .25, .25])
        maps = relation[:256].reshape(4, 8, 8)
        self.assertAlmostEqual(maps[1, 3, 3], 1)
        self.assertAlmostEqual(maps[2, 3, 3], 0)
        np.testing.assert_allclose(maps[3], maps[1] - maps[2])

    def test_degenerate_maps_and_outside_roi_are_finite(self):
        for roi in ([0, 0, 1, 1], [2, 2, .01, .01]):
            self.assertTrue(np.isfinite(relation_features(np.zeros((3, 8, 8)), np.zeros(3), roi)).all())

    def test_invalid_state_rejected(self):
        for roi in ([0, 0, -1, 1], [0, np.nan, 1, 1]):
            with self.assertRaises(ValueError):
                relation_features(np.ones((3, 8, 8)), np.ones(3), roi)


class CacheTests(unittest.TestCase):
    def test_roundtrip_and_lineage(self):
        contract, data = fixture()
        validate_cache(data, contract, "fit_a")
        with tempfile.TemporaryDirectory() as d:
            path = cache_path(d, "fit_a")
            write_cache(path, data, dict(contract_fingerprint=contract["fingerprint"], sequence="fit_a"))
            loaded = load_cache(path, contract, "fit_a")
            np.testing.assert_array_equal(loaded["raw"], data["raw"])
            with self.assertRaises(ValueError):
                load_cache(path, dict(contract, fingerprint="changed"), "fit_a")

    def test_cache_tampering_detected(self):
        contract, data = fixture()
        for field, mutate in (
            ("relation", lambda a: a.__setitem__((0, 0), a[0, 0] + .1)),
            ("branch_cost", lambda a: a.__setitem__((0, 0, 0), a[0, 0, 0] + 1)),
            ("live_actions", lambda a: a.__setitem__(0, 1)),
            ("replay_boxes", lambda a: a.__setitem__((0, 10, 0), a[0, 10, 0] + 1)),
            ("raw", lambda a: a.__setitem__((0, -13), 0)),
        ):
            changed = copy.deepcopy(data)
            mutate(changed[field])
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_cache(changed, contract, "fit_a")

    def test_visibility_mask_and_target_normalization(self):
        rewards = np.full((1, REPEATS, 6, HORIZON), .7)
        valid = np.ones((1, HORIZON), bool)
        valid[0, 5:8] = False
        rewards[..., 5:8] = 0
        q, eligible = discounted_values(rewards, valid)
        np.testing.assert_allclose(q, .7)
        self.assertTrue(eligible[0])
        valid[0, 0], rewards[..., 0] = False, 0
        self.assertFalse(discounted_values(rewards, valid)[1][0])
        rewards[..., 0] = .1
        with self.assertRaises(ValueError):
            discounted_values(rewards, valid)

    def test_audit_never_opens_tune_or_validation_files(self):
        contract, data = fixture()
        with tempfile.TemporaryDirectory() as d:
            cache, audit = Path(d) / "cache", Path(d) / "audit"
            write_json(cache / "stage25_contract.json", contract)
            write_cache(cache_path(cache, "fit_a"), data,
                        dict(contract_fingerprint=contract["fingerprint"], sequence="fit_a"))
            for name in ("tune_a", "val_a"):
                write_cache(cache_path(cache, name), {"bad": np.array([np.nan])}, {})
            with patch.dict("sys.modules", {"torch": None}):
                result = run(cache, audit, audit_only=True)
            self.assertEqual(result["status"], "PASS_BRANCH_DATA_INTEGRITY")
            self.assertFalse(result["models_trained"])

    def test_incomplete_data_rejected_before_model_loading(self):
        contract, data = fixture()
        contract = sealed({k: v for k, v in dict(contract, fit=["fit_a", "fit_b"],
                          clip_steps={"fit_a": 64, "fit_b": 64}).items() if k != "fingerprint"})
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "cache"
            write_json(cache / "stage25_contract.json", contract)
            write_cache(cache_path(cache, "fit_a"), data,
                        dict(contract_fingerprint=contract["fingerprint"], sequence="fit_a"))
            with self.assertRaisesRegex(ValueError, "Complete every"), patch.dict("sys.modules", {"torch": None}):
                run(cache, Path(d) / "screen")


class ScreenTests(unittest.TestCase):
    def test_video_folds_and_train_only_standardizer(self):
        names = np.repeat(["a", "b", "c", "d", "e"], 3)
        seen = []
        for train, test, _ in grouped_folds(names, 5, 1):
            self.assertFalse(set(names[train]) & set(names[test]))
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(names))))
        mean, scale = standardizer(np.array([[1., 2.], [3., 2.]]), np.ones(2))
        np.testing.assert_array_equal(mean, [2, 2])
        self.assertTrue(np.isfinite(normalized(np.array([[100., 2.]]), mean, scale)).all())

    def test_per_state_action_centering_not_video_difficulty(self):
        _, data = fixture()
        values = dataset([("a", data)])
        self.assertEqual(values["q"].shape, (ANCHORS, REPEATS, 6))
        centered = values["q"].mean(1)
        centered -= centered.mean(1, keepdims=True)
        np.testing.assert_allclose(centered.sum(1), 0, atol=1e-12)

    def test_uniform_predictions_fail_screen_without_forced_diversity(self):
        names = np.repeat(["a", "b", "c", "d", "e"], 6)
        q = np.full((30, 2, 6), .5)
        q[:, :, 2] += .1
        scores = np.zeros((30, 6))
        scores[:, 2] = 1
        data = dict(names=names, q=q, post_q=q.copy())
        _, _, result = summarize(data, dict(raw=scores, relational=scores), np.full(30, 2))
        self.assertNotEqual(result["status"], "PASS_RELATIONAL_STATE_SCREEN")

    def test_known_state_dependent_gain_passes_all_static_controls(self):
        names = np.repeat(["a", "b", "c", "d", "e"], 6)
        q = np.full((30, 2, 6), .5)
        for i in range(30):
            q[i, :, i % 6] += .1
        _, _, result = summarize(dict(names=names, q=q, post_q=q.copy()),
                                 dict(raw=np.zeros((30, 6)), relational=q.mean(1)), np.zeros(30, int))
        self.assertEqual(result["status"], "PASS_RELATIONAL_STATE_SCREEN")

    def test_immediate_gain_cannot_pass_when_post_action_values_are_flat(self):
        names = np.repeat(["a", "b", "c", "d", "e"], 6)
        q = np.full((30, 2, 6), .5)
        for i in range(30):
            q[i, :, i % 6] += .1
        _, _, result = summarize(dict(names=names, q=q, post_q=np.full_like(q, .5)),
                                 dict(raw=np.zeros((30, 6)), relational=q.mean(1)), np.zeros(30, int))
        self.assertFalse(result["post_anchor_transfer_pass"])
        self.assertNotEqual(result["status"], "PASS_RELATIONAL_STATE_SCREEN")


class FakeState:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return FakeState(self.value)


class FakeEnvironment:
    def __init__(self, fail=False):
        self.budget = TokenCredit()
        self.box, self.frame = np.array([0., 0., 10., 10.]), 0
        self.actor_view = SimpleNamespace(uniform=None, forced=None, last=None)
        self.rng, self.fail = FakeState(5), fail
        self.rt = SimpleNamespace(torch=SimpleNamespace(get_rng_state=lambda: self.rng,
            set_rng_state=lambda v: setattr(self, "rng", v), cuda=SimpleNamespace(is_available=lambda: False)))

    def snapshot(self):
        return copy.deepcopy(dict(box=self.box, motion=np.zeros(4), frame_id=self.frame, previous_action=0,
                                  confidence=.5, memory=None, budget=self.budget))

    def restore(self, state):
        self.box, self.frame, self.budget = state["box"].copy(), state["frame_id"], copy.deepcopy(state["budget"])

    def step(self, image, action):
        self.budget.advance()
        self.budget.spend(action)
        np.random.random()
        random.random()
        self.rng.value += 1
        if self.fail and action == 6:
            raise RuntimeError("injected branch failure")
        self.box[0] += action * .001
        self.frame += 1
        self.actor_view.last = dict(action=action)
        return dict(box=self.box.copy(), action=action)


class BranchTests(unittest.TestCase):
    def test_restore_actual_history_rng_and_equal_cost_on_success_and_failure(self):
        from stage21_counterfactual_runtime import assert_snapshot_equal
        for fail in (False, True):
            env = FakeEnvironment(fail)
            actions = schedule(64, 0)
            anchor = int(anchor_indices(actions)[0])
            for action in actions[:anchor]:
                env.step(None, int(action))
            before = env.snapshot()
            env.fail = False
            actual = env.step(None, int(actions[anchor]))
            env.fail = fail
            after = env.snapshot()
            transient = env.actor_view.last
            numpy, python, torch_rng = np.random.get_state(), random.getstate(), env.rng.value
            args = (env, before, after, actual, [None] * HORIZON,
                    np.tile([0, 0, 10, 10], (HORIZON, 1)), np.ones(HORIZON, bool), continuation(actions, anchor, 1))
            if fail:
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    simulate(*args)
            else:
                rewards, costs, error, boxes = simulate(*args)
                self.assertEqual(rewards.shape, (2, 6, 20))
                self.assertTrue(np.all(costs == np.count_nonzero(actions[anchor:anchor + HORIZON])))
                self.assertEqual(error, 0)
                self.assertEqual(boxes.shape, (20, 4))
            assert_snapshot_equal(env.snapshot(), after)
            self.assertIs(env.actor_view.last, transient)
            self.assertEqual(env.rng.value, torch_rng)
            self.assertEqual(repr(np.random.get_state()), repr(numpy))
            self.assertEqual(random.getstate(), python)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable: run these tests on the training server")
class TorchTests(unittest.TestCase):
    def test_probe_is_centered_and_matched_capacity(self):
        torch.manual_seed(1)
        raw = torch.randn(4, raw_dimensions(3))
        relation = torch.randn(4, RELATION_DIM)
        first = build_probe(torch, 3)
        second = build_probe(torch, 3)
        second.load_state_dict(first.state_dict())
        self.assertEqual(sum(p.numel() for p in first.parameters()), sum(p.numel() for p in second.parameters()))
        prediction = first(raw, relation)
        self.assertEqual(tuple(prediction.shape), (4, 6))
        self.assertTrue(torch.allclose(prediction.sum(1), torch.zeros(4), atol=1e-6))
        loss = prediction.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters()))
        with torch.no_grad():
            self.assertTrue(torch.equal(first(raw, torch.zeros_like(relation)), second(raw, torch.zeros_like(relation))))

    def test_real_optimizer_train_predict_and_no_test_label_access(self):
        import stage25_relation_screen as screen
        _, arrays = fixture(dim=3)
        data = dataset([("a", arrays), ("b", arrays)])
        train, test = np.arange(8), np.arange(8, 16)
        with patch.object(screen, "EPOCHS", 1):
            first, _, _ = screen.train_predict(torch, data, train, test, "relational", 1, 3, "cpu")
            changed = copy.deepcopy(data)
            changed["q"][test] = 999
            second, _, _ = screen.train_predict(torch, changed, train, test, "relational", 1, 3, "cpu")
        np.testing.assert_array_equal(first, second)

    def test_full_screen_checkpoint_publication_and_resume(self):
        import stage25_relation_screen as screen
        contract, first = fixture("fit_a")
        _, second = fixture("fit_b")
        contract = sealed({k: v for k, v in dict(contract, fit=["fit_a", "fit_b"],
                          clip_steps={"fit_a": 64, "fit_b": 64}).items() if k != "fingerprint"})
        with tempfile.TemporaryDirectory() as d:
            cache, output = Path(d) / "cache", Path(d) / "screen"
            write_json(cache / "stage25_contract.json", contract)
            for name, data in (("fit_a", first), ("fit_b", second)):
                write_cache(cache_path(cache, name), data,
                            dict(contract_fingerprint=contract["fingerprint"], sequence=name))
            with patch.object(screen, "EPOCHS", 1), patch.object(screen, "FOLDS", 2), patch.object(screen, "MODEL_SEEDS", (1,)):
                result = run(cache, output)
                with patch.object(screen, "train_predict", side_effect=AssertionError("Resume retrained a completed fold")):
                    resumed = run(cache, output)
            self.assertEqual(result, resumed)
            self.assertFalse(result["tune_labels_loaded"])
            self.assertTrue((output / "folds/0/raw/1/model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
