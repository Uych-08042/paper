"""Real CPU gradients, held-out isolation, resume, immutable sources and OOF guards."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_common import seal
from stage34_identity_common import prepare_plan as prepare_source
from test_stage34_identity_representation import fake_image, source_fixture, torch_runtime
from stage35_preserve_common import ARMS, BASELINES, DESIGN, fold_samples, load_plan, prepare
import stage35_preserve_experiment as experiment
from stage35_preserve_runtime import forward_student, losses, make_student, parameters, preserve_loss
from stage35_preserve_summary import compare_rows, decide, summarize

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


def fixture(root):
    source, prior, _ = source_fixture(root)
    old_root = root / "stage34"
    old = prepare_source(source, prior, old_root, fake_image, 20, 500)
    write_json(old_root / "analysis/stage34_decision.json", seal(dict(schema=old["schema"],
        status="NO_SPATIAL_REPRESENTATION_GAIN_ESTABLISHED", plan_fingerprint=old["fingerprint"])))
    # Stage-35 must work even when every consumed checking annotation is absent.
    for e in old["checking"]:
        (Path(old["root"]) / e["annotation"]).unlink()
    return prepare(old_root, root / "stage35")


def synthetic_rows():
    rows = []
    for video in range(8):
        for method in BASELINES + ARMS:
            auc = dict(fast_initial=60., sgla=62., supervised=60.1, preserve=60.2, correct=60.3, preserve_correct=61., distill_all_correct=60.5)[method]
            reliable = dict(fast_initial=80., sgla=80., supervised=79.8, preserve=80., correct=79.5, preserve_correct=80., distill_all_correct=79.8)[method]
            failure = dict(fast_initial=20., sgla=20., supervised=20.1, preserve=20.2, correct=21., preserve_correct=21.1, distill_all_correct=20.4)[method]
            for seed in (DESIGN["seeds"] if method in ARMS else ["baseline"]):
                rows.append(dict(sequence=str(video), method=method, seed=seed, auc=auc,
                    low_iou_fraction=0., failure_events=0, instrumented_ms=1., fast_reliable_frames=20,
                    teacher_failure_frames=10, fast_reliable_auc=reliable, teacher_failure_auc=failure))
    return rows


class DataTests(unittest.TestCase):
    def test_fold_isolation_source_immutability_and_support(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            plan = fixture(root)
            before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "stage35" not in p.parts}
            held = []
            for f in plan["folds"]:
                samples, weights = fold_samples(plan, f["fold"])
                self.assertFalse(set(f["fit"]) & set(f["held_out"]))
                self.assertEqual(len(f["fit"]), 6)
                self.assertTrue({s["sequence"] for s in samples} <= set(f["fit"]))
                self.assertAlmostEqual(sum(w for s, w in zip(samples, weights) if s["cohort"] == "hard"), .5)
                held += f["held_out"]
            self.assertEqual(len(held), len(set(held)))
            self.assertEqual(load_plan(root / "stage35"), plan)
            self.assertEqual(before, {str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "stage35" not in p.parts})
            with mock.patch.dict(DESIGN, minimum_fit_cohort_frames=999999):
                with self.assertRaisesRegex(ValueError, "Insufficient fixed fold support"):
                    prepare(root / "stage34", root / "insufficient")

    def test_decision_demands_preservation_and_correction_not_easy_average(self):
        with mock.patch.dict(DESIGN, bootstrap_samples=64):
            rows = synthetic_rows()
            averaged, paired = compare_rows(rows)
            report, passed = decide(rows, averaged, paired)
            self.assertTrue(passed)
            self.assertEqual(report["preservation_effect"]["videos"], 8)  # NOT 16 seeds-as-videos.
            # All videos still avoid failure, but a one-video easy-state drop fails the new guard.
            bad = [dict(r) for r in rows]
            for r in bad:
                if r["sequence"] == "0" and r["method"] == "preserve_correct":
                    r["fast_reliable_auc"] = 78.9
            a, p = compare_rows(bad)
            self.assertFalse(decide(bad, a, p)[0]["retention_guard_pass"])
            # Missing hard support must never pass by dropping unsupported groups.
            bad = [dict(r, teacher_failure_auc=None) if int(r["sequence"]) >= 3 else dict(r) for r in rows]
            a, p = compare_rows(bad)
            self.assertEqual(decide(bad, a, p)[0]["status"], "INSUFFICIENT_OOF_SUPPORT")
            bad = [dict(r, auc=62.) if r["method"] == "distill_all_correct" else dict(r) for r in rows]
            a, p = compare_rows(bad)
            self.assertFalse(decide(bad, a, p)[0]["mechanism_pass"])


@unittest.skipUnless(TORCH, "PyTorch required for actual CPU gradient tests")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def test_plain_path_gate_gradients_and_detached_teacher(self):
        torch = self.torch
        rt = torch_runtime(torch)
        image = np.ones((64, 96, 3), np.uint8)
        rt.initialize(image, [40, 20, 10, 10])
        search, _ = rt.prepare(image)
        net = make_student(rt)
        with torch.no_grad():
            a, b = rt.head(rt.prefix(search)), forward_student(rt, net, search)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        self.assertFalse(hasattr(net, "spatial_identity"))
        self.assertFalse(hasattr(net.backbone, "MLP"))
        self.assertEqual(len(net.backbone.blocks), 6)
        for arm in ARMS:
            for reliable in (False, True):
                net = make_student(rt)
                total, supervised, kd, correction = losses(rt, net, search, torch.tensor([[.25, .25, .25, .25]]), arm, reliable)
                self.assertTrue(bool(torch.isfinite(total)))
                self.assertEqual(float(kd.detach()) > 0, arm == "distill_all_correct" or (reliable and arm in ("preserve", "preserve_correct")))
                self.assertEqual(float(correction.detach()) > 0, not reliable and arm in ("correct", "preserve_correct", "distill_all_correct"))
                total.backward()
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in parameters(net).values() if p.grad is not None), 0)
                self.assertTrue(all(p.grad is None for p in rt.net.parameters()))
                self.assertTrue(all(p.grad is None for n, p in net.named_parameters() if n not in parameters(net)))
        s = dict(score_map=torch.rand(1, 1, 4, 4, requires_grad=True),
            size_map=torch.rand(1, 2, 4, 4, requires_grad=True), offset_map=torch.rand(1, 2, 4, 4, requires_grad=True))
        t = {k: torch.rand_like(v, requires_grad=True) for k, v in s.items()}
        preserve_loss(rt, net, s, t).backward()
        self.assertTrue(all(v.grad is None for v in t.values()))
        self.assertTrue(all(v.grad is not None and bool(v.grad.isfinite().all()) for v in s.values()))

    def test_full_workflow_resume_oof_model_isolation_and_tamper_rejection(self):
        torch = self.torch
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            with mock.patch.dict(DESIGN, updates=8, smoke_updates=8, save_every=2, bootstrap_samples=64):
                plan = fixture(root)
                output = root / "stage35"
                rt = torch_runtime(torch)
                sources = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "stage35" not in p.parts}
                def reader(root, frame):
                    return fake_image(Path(root) / frame["file"])
                self.assertEqual(experiment.train(rt, plan, output, DESIGN["seeds"][0], True, reader)["status"], "PASS_CORRECTIVE_DISTILLATION_SMOKE")
                with self.assertRaises(FileNotFoundError):
                    experiment.evaluation_models(rt, plan, output)
                # Each training call must open only its six training groups, never its two held-out groups.
                original_train = experiment.train_arm
                def isolated(rt, plan, directory, fold, seed, arm, smoke=False, image_reader=None):
                    def isolated_reader(root, frame):
                        self.assertIn(Path(frame["file"]).parent.name, plan["folds"][fold]["fit"])
                        return reader(root, frame)
                    return original_train(rt, plan, directory, fold, seed, arm, smoke, isolated_reader)
                original_save = experiment.save_step
                def interrupt(torch, directory, value):
                    path = original_save(torch, directory, value)
                    if value["step"] == 2:
                        raise RuntimeError("injected interruption")
                    return path
                with mock.patch.object(experiment, "train_arm", side_effect=isolated):
                    with mock.patch.object(experiment, "save_step", side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                            experiment.train(rt, plan, output, DESIGN["seeds"][0], False, reader)
                    for seed in DESIGN["seeds"]:
                        self.assertEqual(experiment.train(rt, plan, output, seed, False, reader)["status"], "COMPLETE_MATCHED_OOF_TRAINING")
                # Adam state + fixed sample stream produces identical resumed final tensors.
                reference = original_train(rt, plan, root / "reference", 0, DESIGN["seeds"][0], "supervised", False, reader)
                resumed = output / "training/seed2110/fold0/supervised/step000008.pt"
                a = experiment.load_checkpoint(torch, Path(reference["checkpoint"]))["parameters"]
                b = experiment.load_checkpoint(torch, resumed)["parameters"]
                for key in a:
                    torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
                old_hashes = {str(p): file_hash(p) for p in (output / "training").rglob("*") if p.is_file()}
                experiment.train(rt, plan, output, DESIGN["seeds"][0], False, reader)
                self.assertEqual(old_hashes, {str(p): file_hash(p) for p in (output / "training").rglob("*") if p.is_file()})
                experiment.evaluate(rt, plan, output, reader)
                self.assertEqual(len(list((output / "oof").glob("*.npz"))), 96)
                report = summarize(output)
                self.assertEqual(report["videos"], 8)
                self.assertFalse(report["original_checking_labels_loaded"])
                experiment.status(output, plan)
                self.assertEqual(sources, {str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "stage35" not in p.parts and "reference" not in p.parts})
                # Directory iteration order differs across filesystems. Corrupt
                # a deliberate SECOND-seed checkpoint, then resume its owner;
                # resuming seed[0] does not audit another seed's training tree.
                corrupt_seed = DESIGN["seeds"][1]
                corrupt = (output / "training" / ("seed{}".format(corrupt_seed)) / "fold0" /
                    "preserve_correct" / "step{:06d}.pt".format(DESIGN["updates"]))
                receipt = experiment.load_sealed(corrupt.with_suffix(".json"))
                owner_seed = receipt["metadata"]["seed"]
                self.assertEqual(owner_seed, corrupt_seed)
                self.assertNotEqual(owner_seed, DESIGN["seeds"][0])
                self.assertEqual(file_hash(corrupt), receipt["checkpoint_sha256"])
                corrupt.write_bytes(corrupt.read_bytes() + b"tampered")
                self.assertNotEqual(file_hash(corrupt), receipt["checkpoint_sha256"])
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    experiment.train(rt, plan, output, owner_seed, False, reader)
                # Evaluation must audit ALL seeds before opening OOF outcomes.
                with self.assertRaisesRegex(ValueError, "Frozen checkpoint changed"):
                    experiment.evaluation_models(rt, plan, output)


if __name__ == "__main__":
    unittest.main()
