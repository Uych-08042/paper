"""Actual CPU training/resume, OOF isolation, GT-free checking and frozen evidence."""

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import crop_geometry, seal
import stage40_crop_common as common
import stage40_crop_experiment as experiment
import stage40_crop_summary as summary
from stage40_crop_common import ARMS, DESIGN, SEEDS

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


class MathTests(unittest.TestCase):
    def test_target_mapping_uses_actual_rounded_crop_and_keeps_full_unclipped_gt(self):
        plan = dict(search_factor=4., search_size=256)
        parent = np.array([11.1, 8.3, 15.3, 17.2])
        gt = np.array([-200., 2., 45., 20.])
        for label in common.prior.PERTURBATIONS:
            shifted = common.prior.perturb(parent, label)
            crop, _, factor = crop_geometry(shifted, [64, 64], 4., 256)
            roi, actual = common.target_roi(gt, shifted, [64, 64], plan)
            np.testing.assert_allclose(np.r_[roi[:2] * crop[2] + crop[:2], roi[2:] * crop[2]], gt, atol=1e-12)
            self.assertEqual(actual, factor)
            self.assertLess(roi[0], 0.)

    def test_video_pairing_retention_and_secondary_stress_do_not_count_seeds_as_videos(self):
        plan = dict(fit=[dict(sequence=str(i), source_group_proxy="g" + str(i)) for i in range(8)],
            tasks={str(i): [2, 5, 7] for i in range(8)})
        rows, stress = [], []
        for entry in plan["fit"]:
            for arm in ARMS + ("fast_initial", "sgla"):
                seeds = SEEDS if arm in ARMS else ("shared",)
                for seed in seeds:
                    value = dict(clean=62., augmented=64., fast_initial=60., sgla=61.)[arm]
                    rows.append(dict(sequence=entry["sequence"], arm=arm, seed=seed, auc=value,
                        reliable_auc=value, low_iou_fraction=0., failure_events=0, longest_failure_fraction=0.))
            for method in common.methods():
                for anchor in plan["tasks"][entry["sequence"]]:
                    loss = .01 if method.startswith("augmented") else .03
                    stress.append(dict(sequence=entry["sequence"], method=method, anchor=anchor,
                        clean_iou=.8, perturbed_iou=.8 - loss, post_iou_loss=loss))
        with mock.patch.dict(DESIGN, bootstrap_samples=64):
            tables, decision = summary.analyze(rows, stress, plan)
        self.assertTrue(decision["pilot_pass"])
        self.assertTrue(decision["augmentation_stability_mechanism_supported"])
        self.assertEqual(decision["primary"]["videos"], 8)
        self.assertEqual(decision["primary"]["mean_delta"], 2.)
        self.assertEqual(len(tables["stress_video_summary"]), 16)
        changed = copy.deepcopy(rows)
        next(r for r in changed if r["sequence"] == "0" and r["arm"] == "augmented")["failure_events"] = 1
        with mock.patch.dict(DESIGN, bootstrap_samples=64):
            _, rejected = summary.analyze(changed, stress, plan)
        self.assertFalse(rejected["retention_guard_pass"])
        with self.assertRaisesRegex(ValueError, "complete unmixed"):
            summary.analyze(rows + rows[:1], stress, plan)


def fixture(root, torch):
    """Remote Stage-39 provenance is the only mocked boundary; all new training is real."""
    from test_stage37_crossed import source_fixture
    from stage35_preserve_experiment import collect_trace
    _, rt, reader, _, _ = source_fixture(root, torch)
    original = root / "stage35"
    source = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    source.pop("fingerprint")
    samples = []
    for entry in source["fit"]:
        trace = collect_trace(rt, source, entry, "fast_initial", image_reader=reader)
        write_cache(original / "oof" / (entry["sequence"] + "__fast_initial.npz"), trace, {})
        for frame in range(1, 7):
            parent = trace["previous"][frame - 1]
            roi, factor = common.target_roi([20., 20., 12., 12.], parent, [64, 64], source)
            samples.append(dict(sequence=entry["sequence"], frame_id=frame, cohort="easy" if frame % 2 else "hard",
                previous=parent.tolist(), factor=factor, target_roi=roi.tolist()))
    source["samples"] = samples
    source = seal(source)
    write_json(original / "manifest.json", source)
    stage39 = root / "stage39"
    stage39.mkdir()
    write_json(stage39 / "manifest.json", seal(dict(schema=common.prior.SCHEMA, role="synthetic_external_boundary")))
    old = dict(source_input=str(original), source_runtime=rt.identity,
        source_frozen_neural_model_sha256={"sgla": experiment.state_digest(rt.net)}, runtime_seed=2116,
        window=np.ones((4, 4)).tolist(), tasks={e["sequence"]: common.prior.anchors(e) for e in source["fit"]})
    bindings = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    def adapter(unused):
        common.verify_bindings(bindings)
        return old, source, dict(fingerprint="synthetic-completed-negative-stage39"), dict(bindings)
    return stage39, rt, reader, adapter, bindings


@unittest.skipUnless(TORCH, "PyTorch needed for actual CPU training and evaluation")
class TorchTests(unittest.TestCase):
    def test_real_training_resume_exactness_and_complete_oof_evaluation_with_tamper_rejection(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.dict(DESIGN, updates=3, smoke_updates=2, batch_size=1, save_every=1, bootstrap_samples=64), \
                mock.patch.dict(common.prior.DESIGN, horizon=3, smoke_horizon=2):
            root = Path(temporary)
            source, rt, reader, adapter, bindings = fixture(root, torch)
            output = root / "stage40"
            with mock.patch.object(common, "source_info", side_effect=adapter):
                plan = common.prepare(source, output)
                self.assertEqual(plan, common.load_plan(output))
                # Same parent/sample schedule; perturbation labels do not select on GT outcomes.
                samples, draws, perturbations = common.schedule(plan, 0, SEEDS[0], 3)
                self.assertTrue(all(s["sequence"] in plan["folds"][0]["fit"] for s in samples))
                self.assertTrue(all(s["sequence"] not in plan["folds"][0]["held_out"] for s in samples))
                experiment.smoke(rt, plan, output, reader)
                experiment.smoke_gate(output, plan)
                # Resume after an actual committed intermediate optimizer step: exact tensor result.
                reference = experiment.train_arm(rt, plan, root / "uninterrupted", 0, SEEDS[0], "augmented", image_reader=reader)
                save = experiment.save_step
                def interrupt(torch_api, directory, value):
                    saved = save(torch_api, directory, value)
                    if value["step"] == 2:
                        raise RuntimeError("test interruption")
                    return saved
                with mock.patch.object(experiment, "save_step", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "test interruption"):
                    experiment.train_arm(rt, plan, root / "resumed", 0, SEEDS[0], "augmented", image_reader=reader)
                resumed = experiment.train_arm(rt, plan, root / "resumed", 0, SEEDS[0], "augmented", image_reader=reader)
                a = experiment.load_checkpoint(torch, reference["checkpoint"])
                b = experiment.load_checkpoint(torch, resumed["checkpoint"])
                for key in a["parameters"]:
                    self.assertTrue(torch.equal(a["parameters"][key], b["parameters"][key]))
                self.assertEqual(a["history"], b["history"])
                # No held-out image or annotation can enter this fold's training reads.
                seen = set()
                def training_reader(data_root, frame):
                    seen.add(Path(frame["file"]).parent.name)
                    return reader(data_root, frame)
                experiment.train_arm(rt, plan, root / "isolation", 0, SEEDS[1], "clean", image_reader=training_reader)
                self.assertTrue(seen <= set(plan["folds"][0]["fit"]))
                experiment.train(rt, plan, output, SEEDS[0], image_reader=reader)
                with self.assertRaises(FileNotFoundError):
                    experiment.evaluate(rt, plan, output, image_reader=reader)
                experiment.train(rt, plan, output, SEEDS[1], image_reader=reader)
                models, model_bindings = experiment.evaluation_models(rt, plan, output)
                self.assertEqual(len(models), 16)
                entry = plan["fit"][0]
                experiment.evaluate(rt, plan, output, entry["sequence"], reader)
                with self.assertRaisesRegex(ValueError, "all 32 full"):
                    summary.summarize(output)
                net = models[(experiment.fold_for(plan, entry), "augmented_seed2110")]
                before = experiment.collect_trace(rt, plan, entry, "augmented_seed2110", net, reader)
                future = Path(plan["root"]) / "annotations/other_future.txt"
                future.write_text("20,20,12,12\n" + "\n".join(["50,50,5,5"] * 12), encoding="utf-8")
                changed = dict(entry, annotation="annotations/other_future.txt", annotation_sha256=file_hash(future))
                after = experiment.collect_trace(rt, plan, changed, "augmented_seed2110", net, reader)
                np.testing.assert_array_equal(before["boxes"], after["boxes"])
                self.assertFalse(np.array_equal(before["gt"], after["gt"]))
                anchor = plan["tasks"][entry["sequence"]][0]
                before = experiment.collect_stress(rt, plan, entry, "augmented_seed2110", anchor, net, 3, reader)
                after = experiment.collect_stress(rt, plan, changed, "augmented_seed2110", anchor, net, 3, reader)
                np.testing.assert_array_equal(before["boxes"], after["boxes"])
                bad = dict(before, previous=before["previous"].copy())
                bad["previous"][1, 1, 0] += 1
                with self.assertRaises(AssertionError):
                    common.audit_stress(bad, plan, entry, anchor, 3, common.baseline(plan, entry, "sgla"))
                experiment.evaluate(rt, plan, output, image_reader=reader)
                result = summary.summarize(output)
                self.assertEqual(result["videos"], 8)
                self.assertEqual(result["new_models"], 16)
                self.assertFalse(result["novel_method_established"])
                self.assertLess((output / "analysis/stage40_decision.json").stat().st_size, 12000)
                self.assertEqual(result, summary.summarize(output))
                common.verify_bindings(bindings)
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    experiment.evaluate(rt, plan, output, image_reader=reader)
                    self.assertEqual(spy.call_count, 0)
                path = common.result_path(output, entry, "augmented_seed2110", anchor)
                path.with_suffix(".json").unlink()
                experiment.evaluate(rt, plan, output, entry["sequence"], reader)
                self.assertEqual(result, summary.summarize(output))
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "Stage-40 result changed"):
                    summary.summarize(output)
                path.write_bytes(original)
                checkpoint = next(Path(k) for k in model_bindings if k.endswith(".pt"))
                checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper checkpoint")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    summary.summarize(output)


if __name__ == "__main__":
    unittest.main()
