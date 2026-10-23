"""CPU contracts/rollout audits plus optional real-PyTorch student-gradient tests."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stage4_common import file_hash, fingerprint, tracking_iou, write_cache, write_json
from stage20_compression_common import (ARMS, METHODS, SCHEMA, accept_teacher, audit_trace,
                                        auc, channel_plan, compact_copy, crop_parent,
                                        design_split, distillation_loss, factorial_effect,
                                        forward_model, load_contract, paired_interval, state_digest)
from stage20_compression_summary import audit_training, read_scores, summarize
from stage20_student_train import evaluate, train_clip


def manifest_fixture():
    value = dict(schema="stage4_ab_v1", dataset="got10k_train", root="unused",
                 sequences=[dict(sequence="video{:03d}".format(i), split="train" if i < 10 else "val",
                                 start=0, stop=17) for i in range(12)])
    value["fingerprint"] = fingerprint(value)
    return value


def trace_fixture():
    boxes = np.tile([0., 0., 10., 10.], (16, 1))
    valid = np.ones(16, dtype=bool)
    return dict(frame_id=np.arange(1, 17), previous=boxes.copy(), boxes=boxes,
                gt=boxes.copy(), valid=valid, iou=tracking_iou(boxes, boxes, valid),
                layer=np.full(16, 9, dtype=np.int64), latency_ms=np.ones(16))


class ContractTests(unittest.TestCase):
    def test_split_never_uses_original_val_and_smoke_is_subset_of_full(self):
        source = manifest_fixture()
        full = design_split(source, "full", seed=8, check_sequences=4)
        smoke = design_split(source, "smoke", seed=8, check_sequences=4)
        self.assertEqual(smoke["fit"], full["fit"][:2])
        self.assertEqual(smoke["check"], full["check"][:2])
        fit, check = ({x["sequence"] for x in full[key]} for key in ("fit", "check"))
        self.assertFalse(fit & check)
        self.assertFalse((fit | check) & set(full["excluded_source_val"]))
        with self.assertRaises(ValueError):
            design_split(source, "full", check_sequences=9)

    def test_channel_ranking_is_deterministic_physical_half_all_twelve(self):
        fc1, fc2 = np.ones((64, 4)), np.ones((4, 64))
        specs = channel_plan([(fc1, fc2)] * 12)
        self.assertEqual([s["layer"] for s in specs], list(range(1, 13)))
        self.assertTrue(all(s["drop"] == list(range(32, 64)) for s in specs))
        fc1[63] *= 5
        ranked = channel_plan([(fc1, fc2)] * 12)
        self.assertNotIn(63, ranked[0]["drop"])
        self.assertEqual(ranked, channel_plan([(fc1, fc2)] * 12))
        with self.assertRaises(ValueError):
            channel_plan([(fc1, fc2)] * 6)

    def test_crop_factorial_control_and_no_aliasing(self):
        reference, student = np.array([1, 2, 3, 4]), np.array([10, 20, 3, 4])
        for arm in ARMS:
            parent = crop_parent(arm, reference, student)
            np.testing.assert_array_equal(parent, reference if arm.endswith("_reference") else student)
            parent[:] = 0
        self.assertEqual(reference[0], 1)
        self.assertEqual(student[0], 10)

    def test_teacher_acceptance_no_invisible_or_offcrop_pseudolabels(self):
        box = [2, 2, 10, 10]
        self.assertTrue(accept_teacher([.1, .1, .2, .2], box, box, True))
        self.assertFalse(accept_teacher([.1, .1, .2, .2], box, box, False))
        self.assertFalse(accept_teacher([1.1, .1, .2, .2], box, box, True))
        self.assertFalse(accept_teacher([.1, .1, .2, .2], [30, 30, 10, 10], box, True))

    def test_trace_rejects_reference_crop_during_autonomous_evaluation(self):
        data = trace_fixture()
        audit_trace(data, [0, 0, 10, 10], np.arange(1, 17))
        data["previous"][5, 0] += .01
        with self.assertRaisesRegex(ValueError, "own crop"):
            audit_trace(data, [0, 0, 10, 10], np.arange(1, 17))

    def test_trace_rejects_changed_labels(self):
        data = trace_fixture()
        data["iou"][0] = .5
        with self.assertRaisesRegex(ValueError, "IoU"):
            audit_trace(data, [0, 0, 10, 10], np.arange(1, 17))

    def test_metrics_and_interaction(self):
        self.assertAlmostEqual(auc([1, -1, 0]), 100 * 20 / 42)
        effect = factorial_effect(dict(kd_student=np.array([5., 6.]), supervised_student=np.array([3., 4.]),
                                       kd_reference=np.array([4., 5.]), supervised_reference=np.array([3., 4.])))
        np.testing.assert_array_equal(effect, [1., 1.])
        self.assertEqual(paired_interval(effect)["ci_low"], 1.)

    def test_cli_help_without_torch_and_empty_path_rejected(self):
        for file in ("stage20_student_train.py", "stage20_compression_summary.py"):
            path = Path(__file__).parent / file
            result = subprocess.run([sys.executable, "-B", str(path), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).parent / "stage20_student_train.py"),
                                 "--source_manifest", "", "--output_dir", "unused"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Empty path", result.stderr)


class EvaluationTests(unittest.TestCase):
    def test_evaluator_uses_own_predictions_not_gt(self):
        class Runtime:
            torch = SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None))
            tracker = SimpleNamespace(state=None)

            def initialize(self, image, box):
                self.tracker.state = list(box)

            def prepare(self, image):
                return np.asarray(self.tracker.state).copy(), 1.

            def decode(self, output, factor, image):
                return output

        class Model:
            def eval(self):
                return self

        def predict(rt, model, search):
            search[0] += 1
            return search, 8

        source = dict(start=0, stop=5, gt=np.tile([0., 0., 10., 10.], (5, 1)), valid=np.ones(5, dtype=bool))
        with patch("stage20_student_train.forward_model", predict):
            data = evaluate(Runtime(), Model(), source, [np.zeros((4, 4, 3))] * 5)
        np.testing.assert_array_equal(data["boxes"][:, 0], [1, 2, 3, 4])
        np.testing.assert_array_equal(data["previous"][:, 0], [0, 1, 2, 3])


class DiskAuditTests(unittest.TestCase):
    def fixture(self, root):
        sources = [dict(sequence="a", start=0, stop=17, gt=np.tile([0., 0., 10., 10.], (17, 1)).tolist(),
                        valid=[True] * 17)]
        analysis = dict(primary_contrasts=["kd_student_minus_kd_reference", "kd_student_minus_supervised_student"],
                        bootstrap_samples=100, seed=1, latency_warmup_frames=5,
                        teacher_auc_noninferiority_margin=.5, target_parameter_reduction=.25,
                        target_latency_reduction=.1)
        contract = dict(schema=SCHEMA, mode="smoke", seeds=[1, 2], initial_state_sha256="initial",
                        runtime=dict(checkpoint_sha256="teacher"), architecture=dict(parameter_reduction=.3),
                        sources=dict(fit=sources, check=sources), training=dict(epochs=1, accumulation_frames=4),
                        analysis=analysis)
        contract["fingerprint"] = fingerprint(contract)
        write_json(root / "stage20_contract.json", contract)
        artifacts = []
        for seed in contract["seeds"]:
            for arm in ARMS:
                path = root / "checkpoints" / ("seed{}".format(seed)) / arm / "last.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                artifacts.append(dict(seed=seed, method=arm, sha256=file_hash(path), epoch=1,
                                      frames=16, valid_frames=16, updates=4, teacher_accepted=1,
                                      kd_loss_sum=1., final_state_sha256="changed"))
            for method in METHODS:
                digest = "teacher" if method == "teacher" else "initial" if method == "compact_initial" else file_hash(path)
                metadata = dict(seed=seed, method=method, sequence="a", checkpoint_sha256=digest,
                                contract_fingerprint=contract["fingerprint"])
                write_cache(root / "evaluation" / ("seed{}".format(seed)) / method / "a.npz", trace_fixture(), metadata)
        write_json(root / "stage20_artifacts.json", dict(contract_fingerprint=contract["fingerprint"],
                                                       teacher_unchanged=True, checkpoints=artifacts))
        return contract

    def test_end_to_end_cpu_summary_and_seed_clustering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            contract = self.fixture(root)
            self.assertEqual(load_contract(root)[1], contract)
            artifacts = audit_training(root, contract)
            rows = read_scores(root, contract, artifacts)
            policies, paired, seeds, decision = summarize(rows, contract)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(policies), 6)
            self.assertEqual(paired[0]["num_videos"], 1)
            self.assertEqual(paired[0]["num_seeds"], 2)
            self.assertEqual(len(seeds), 12)
            self.assertEqual(decision["status"], "PASS_MATCHED_TRAINING_SMOKE")
            full = dict(contract, mode="full")
            self.assertEqual(summarize(rows, full)[-1]["status"], "FAIL_DEVELOPMENT_FEASIBILITY")
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).parent / "stage20_compression_summary.py"),
                                     "--input", str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_modified_contract_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            contract = self.fixture(root)
            contract["mode"] = "full"
            write_json(root / "stage20_contract.json", contract)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_contract(root)

    def test_mismatched_training_budget_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            contract = self.fixture(root)
            path = root / "stage20_artifacts.json"
            data = json.loads(path.read_text())
            data["checkpoints"][0]["updates"] += 1
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "updates differ"):
                audit_training(root, contract)

    def test_tampered_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            contract = self.fixture(root)
            (root / "checkpoints/seed1/kd_student/last.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Checkpoint missing/changed"):
                audit_training(root, contract)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed locally")
class TorchTests(unittest.TestCase):
    def test_forward_executes_prefix_plus_one_independent_candidate(self):
        import torch

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return x + self.fc(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Module()
                bb = self.backbone
                bb.patch_embed = torch.nn.Linear(4, 4)
                bb.pos_embed_z = torch.nn.Parameter(torch.zeros(1, 1, 4))
                bb.pos_embed_x = torch.nn.Parameter(torch.zeros(1, 2, 4))
                bb.pos_drop, bb.norm = torch.nn.Identity(), torch.nn.LayerNorm(4)
                bb.blocks = torch.nn.ModuleList([Block() for _ in range(12)])
                bb.MLP = torch.nn.Linear(3, 6)
                with torch.no_grad():
                    bb.MLP.weight.zero_()
                    bb.MLP.bias.zero_()
                    bb.MLP.bias[3] = 1
                bb.cat_mode = "direct"

            def forward_head(self, tokens, unused):
                return tokens[:, -2:]

        model = Model()
        rt = SimpleNamespace(torch=torch, lens_z=1, lens_x=2,
                             tracker=SimpleNamespace(z_dict1=SimpleNamespace(tensors=torch.randn(1, 1, 4))),
                             selected_layer=lambda p: int(p.argmax(1).item()) + 7,
                             s1=SimpleNamespace(combine_tokens=lambda z, x, mode: torch.cat([z, x], dim=1),
                                                recover_tokens=lambda x, z, s, mode: x))
        output, layer = forward_model(rt, model, torch.randn(1, 2, 4))
        output.square().mean().backward()
        self.assertEqual(layer, 10)
        self.assertEqual([b.calls for b in model.backbone.blocks], [1] * 6 + [0, 0, 0, 1, 0, 0])
        self.assertIsNone(model.backbone.MLP.weight.grad)
        self.assertIsNotNone(model.backbone.blocks[9].fc.weight.grad)

    def test_actual_optimizer_and_teacher_query_share_each_executed_crop(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(value))

        class Runtime:
            def __init__(self):
                self.torch = torch
                self.net = Model(0.).requires_grad_(False)
                self.tracker = SimpleNamespace(state=None, params=SimpleNamespace(search_size=20))
                self.inputs = []

            def read(self, path):
                return np.zeros((20, 20, 3))

            def initialize(self, image, box):
                self.tracker.state = list(box)

            def prepare(self, image):
                return torch.tensor(self.tracker.state, dtype=torch.float32), 1.

            def decode(self, output, factor, image):
                value = np.asarray(self.tracker.state, dtype=float).copy()
                value[0] += float(output["delta"].detach())
                return value

        def predict(rt, model, search):
            rt.inputs.append((model is rt.net, search.detach().numpy().copy()))
            return dict(score_map=torch.sigmoid(model.value).expand(1, 1, 2, 2),
                        pred_boxes=(model.value + .2).expand(1, 1, 4), delta=model.value), 9

        def loss_fn(torch, output, target, valid):
            return output["pred_boxes"].square().mean(), {}

        for arm in ARMS:
            rt, student = Runtime(), Model(.1)
            source = dict(sequence="fixture", paths=[None] * 5,
                          gt=np.tile([0., 0., 10., 10.], (5, 1)), valid=np.ones(5, dtype=bool))
            optimizer = torch.optim.SGD(student.parameters(), lr=.01)
            with patch("stage20_student_train.forward_model", predict), patch("stage20_student_train.tracking_loss", loss_fn):
                result = train_clip(rt, student, source, arm, optimizer, 2)
            self.assertEqual(result["updates"], 2)
            self.assertEqual(result["teacher_accepted"], 4)
            self.assertNotEqual(float(student.value.detach()), .1)
            self.assertIsNone(rt.net.value.grad)
            for index in range(0, len(rt.inputs), 3):
                self.assertTrue(rt.inputs[index + 1][0])
                self.assertFalse(rt.inputs[index + 2][0])
                np.testing.assert_array_equal(rt.inputs[index + 1][1], rt.inputs[index + 2][1])
            self.assertEqual(result["crop_shift_sum"] == 0, arm.endswith("_reference"))

    def test_real_dense_surgery_gradients_and_teacher_restoration(self):
        import torch
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Sequential()
                self.mlp.add_module("fc1", torch.nn.Linear(4, 64))
                self.mlp.add_module("act", torch.nn.GELU())
                self.mlp.add_module("fc2", torch.nn.Linear(64, 4))

        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.blocks = torch.nn.ModuleList([Block() for _ in range(12)])
        model.backbone.MLP = torch.nn.Linear(4, 6)
        before = state_digest(model)
        specs = channel_plan([(b.mlp.fc1.weight.detach().numpy(), b.mlp.fc2.weight.detach().numpy())
                              for b in model.backbone.blocks])
        student, audits = compact_copy(torch, model, specs)
        self.assertEqual(before, state_digest(model))
        self.assertTrue(all(b.mlp.fc1.out_features == 32 for b in student.backbone.blocks))
        self.assertTrue(all(not p.requires_grad for p in student.backbone.MLP.parameters()))
        x = torch.randn(1, 3, 4)
        for block in student.backbone.blocks:
            x = x + block.mlp(x)
        x.square().mean().backward()
        self.assertTrue(all(b.mlp.fc1.weight.grad is not None for b in student.backbone.blocks))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(sum(a["removed_parameters"] for a in audits),
                         sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in student.parameters()))

    def test_kd_teacher_detached_and_identical_output_zero(self):
        import torch
        student = dict(score_map=torch.rand(1, 1, 4, 4, requires_grad=True),
                       pred_boxes=torch.rand(1, 1, 4, requires_grad=True))
        teacher = dict(score_map=torch.rand(1, 1, 4, 4, requires_grad=True),
                       pred_boxes=torch.rand(1, 1, 4, requires_grad=True))
        loss = distillation_loss(torch, student, teacher)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student["score_map"].grad)
        self.assertIsNone(teacher["score_map"].grad)
        self.assertIsNone(teacher["pred_boxes"].grad)
        self.assertAlmostEqual(float(distillation_loss(torch, teacher, teacher)), 0., places=5)


if __name__ == "__main__":
    unittest.main()
