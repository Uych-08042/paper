"""Source reuse, independent geometry channels and actual CPU closed-loop checks."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash
import stage37_crossed_common as old_common
import stage37_crossed_experiment as old_experiment
import stage37_crossed_summary as old_summary
from stage38_geometry_common import ARMS, DESIGN, REFERENCE, channels, compose, methods, prepare, verify_bindings
import stage38_geometry_common as common
import stage38_geometry_experiment as experiment
import stage38_geometry_summary as summary

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


class MathTests(unittest.TestCase):
    def test_size_offset_sources_are_independent_and_response_is_always_student(self):
        outputs = [dict(score_map=np.full((1, 1, 4, 4), .1 + i * .2),
            size_map=np.full((1, 2, 4, 4), .2 + i * .2), offset_map=np.full((1, 2, 4, 4), .3 + i * .2)) for i in range(2)]
        original = copy.deepcopy(outputs)
        for arm in ARMS:
            packet = compose(outputs, arm)
            size, offset = channels(arm)
            self.assertIs(packet["score_map"], outputs[0]["score_map"])
            self.assertIs(packet["size_map"], outputs[size]["size_map"])
            self.assertIs(packet["offset_map"], outputs[offset]["offset_map"])
        for a, b in zip(outputs, original):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(len(methods("full")), 4)
        self.assertEqual(len(methods("smoke")), 8)

    def test_factorial_effects_video_pairing_family_intervals_and_stability_counts(self):
        rows = []
        for video in range(8):
            for seed in DESIGN["seeds"]:
                for arm in ARMS + (REFERENCE,):
                    gain = dict(student=0., teacher_size=4. if seed == 2110 else 0.,
                        teacher_offset=1., teacher_geometry=4. if seed == 2110 else 2., sgla_reference=5.)[arm]
                    rows.append(dict(sequence=str(video), seed=seed, arm=arm, auc=60 + gain,
                        low_iou_fraction=0., failure_events=int(video == 0 and seed == 2110 and arm == "teacher_offset"),
                        longest_failure_fraction=0., student_reliable_auc=70 + gain, sgla_reliable_auc=70 + gain))
        with mock.patch.dict(DESIGN, bootstrap_samples=64):
            tables, residual = summary.analyze(rows)
        focus = {r["arm"]: r for r in tables["focus"]}
        self.assertEqual(focus["teacher_size"]["gain_vs_student"], 2.)
        self.assertEqual(focus["teacher_size"]["family_ci_low"], 2.)
        self.assertEqual(focus["teacher_size"]["seed_mean_deltas"], "4.00000000|0.00000000")
        self.assertEqual(focus["teacher_offset"]["introduced_failure_videos"], 1)
        self.assertTrue(all(r["videos"] == 8 for r in tables["paired"]))
        factors = {r["effect"]: r["mean_auc_delta"] for r in tables["factor_summary"]}
        self.assertEqual(factors, dict(size_order_average=2., offset_order_average=1., size_offset_interaction=0.))
        self.assertEqual(residual, 0.)
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            summary.analyze(rows + [rows[0]])


@unittest.skipUnless(TORCH, "PyTorch needed for actual CPU source and new-rollout checks")
class RolloutTests(unittest.TestCase):
    def test_completed_source_controls_new_rollouts_gt_isolation_resume_and_tampering(self):
        import torch
        from test_stage37_crossed import source_fixture
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.dict(old_common.DESIGN, smoke_frames=5, bootstrap_samples=64), \
                mock.patch.dict(DESIGN, smoke_frames=5, bootstrap_samples=64):
            root = Path(temporary)
            stage36, rt, reader, adapter, original_bindings = source_fixture(root, torch)
            old_root, output = root / "stage37", root / "stage38"
            # Only the earlier external-data boundary is synthetic; Stage-37/38 provenance is real.
            with mock.patch.object(old_common, "source_info", side_effect=adapter):
                old_plan = old_common.prepare(stage36, old_root)
                old_experiment.collect(rt, old_plan, old_root, "smoke", image_reader=reader)
                old_experiment.collect(rt, old_plan, old_root, "full", image_reader=reader)
                old_summary.summarize(old_root)
                frozen_old = {str(p): file_hash(p) for p in old_root.rglob("*") if p.is_file()}
                plan = prepare(old_root, output)
                self.assertEqual(plan["source_plan_fingerprint"], old_plan["fingerprint"])
                self.assertEqual(plan["source_model_plan_fingerprint"], old_plan["source_plan_fingerprint"])
                with self.assertRaises(FileNotFoundError):
                    experiment.collect(rt, plan, output, "full", image_reader=reader)
                experiment.collect(rt, plan, output, "smoke", image_reader=reader)
                experiment.smoke_gate(output, plan)
                smoke = common.read_sealed(output / "smoke/stage38_smoke.json", common.SCHEMA)
                self.assertEqual(len(smoke["endpoint_parity"]), 4)
                self.assertTrue(all(p["max_box_error"] == 0 for p in smoke["endpoint_parity"]))
                entry = plan["fit"][0]
                experiment.collect(rt, plan, output, "full", entry["sequence"], reader)
                with self.assertRaisesRegex(ValueError, "all 32 NEW"):
                    summary.summarize(output)
                models = experiment.load_models(rt, dict(plan, fingerprint=plan["source_model_plan_fingerprint"]), plan["source_input"])
                net = models[(old_common.fold_for(plan, entry), "supervised_seed2110")]
                method = "teacher_size_seed2110"
                parents = []
                prepare_original = rt.prepare
                def prepare_spy(image):
                    parents.append(np.asarray(rt.tracker.state).copy())
                    return prepare_original(image)
                with mock.patch.object(rt, "prepare", side_effect=prepare_spy):
                    before = experiment.collect_trace(rt, plan, entry, method, net, image_reader=reader)
                np.testing.assert_array_equal(parents, before["previous"])
                annotation = Path(plan["root"]) / "annotations/other_future.txt"
                annotation.write_text("20,20,12,12\n" + "\n".join(["50,50,5,5"] * 12), encoding="utf-8")
                changed = dict(entry, annotation="annotations/other_future.txt", annotation_sha256=file_hash(annotation))
                read_original = experiment.read_boxes
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    def read_after_rollout(path):
                        self.assertEqual(spy.call_count, 12)
                        return read_original(path)
                    with mock.patch.object(experiment, "read_boxes", side_effect=read_after_rollout):
                        after = experiment.collect_trace(rt, plan, changed, method, net, image_reader=reader)
                np.testing.assert_array_equal(before["boxes"], after["boxes"])
                self.assertFalse(np.array_equal(before["gt"], after["gt"]))
                bad = dict(before, previous=before["previous"].copy())
                bad["previous"][1, 0] += 1
                with self.assertRaises(AssertionError):
                    common.audit_trace(bad, plan, entry, method)
                bad = dict(before, offset=before["offset"].copy())
                bad["offset"][0, 1, 0, 0, 0] = np.nan  # Even an unused source map must be valid.
                with self.assertRaisesRegex(ValueError, "Invalid geometry"):
                    common.audit_trace(bad, plan, entry, method)
                experiment.collect(rt, plan, output, "full", image_reader=reader)
                result = summary.summarize(output)
                self.assertEqual(result["new_full_trajectories"], 32)
                self.assertEqual(result["reused_endpoint_trajectories"], 32)
                self.assertEqual(result["shared_reference_trajectories"], 8)
                self.assertEqual(result["videos"], 8)
                self.assertTrue(all(r["reliable_supported_videos"] <= 8 for r in result["focus"]))
                self.assertIsNone(result["adoption_gate"])
                self.assertLessEqual(result["factorial_maximum_residual"], 1e-10)
                self.assertEqual(len(list((output / "full/traces").glob("*.npz"))), 32)
                self.assertTrue(all("teacher_size" in p.name or "teacher_offset" in p.name
                    for p in (output / "full/traces").glob("*.npz")))
                verify_bindings(original_bindings)
                verify_bindings(frozen_old)
                self.assertEqual(result, summary.summarize(output))
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    experiment.collect(rt, plan, output, "full", image_reader=reader)
                    self.assertEqual(spy.call_count, 0)
                path = output / "full/traces" / (entry["sequence"] + "__" + method + ".npz")
                path.with_suffix(".json").unlink()
                experiment.status(output, plan)
                experiment.collect(rt, plan, output, "full", entry["sequence"], reader)
                self.assertEqual(result, summary.summarize(output))
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "Stage-38 trace changed"):
                    summary.summarize(output)
                path.write_bytes(original)
                old_path = old_root / "full/traces" / (entry["sequence"] + "__student_seed2110.npz")
                old_path.write_bytes(old_path.read_bytes() + b"tamper source")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    common.load_plan(output)


if __name__ == "__main__":
    unittest.main()
