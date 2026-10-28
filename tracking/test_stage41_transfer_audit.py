"""Exact metric accounting, geometric time order and read-only cached provenance."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash
from stage27_uav_common import success_auc, tracking_iou
import stage41_transfer_audit as audit

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


class MathTests(unittest.TestCase):
    def test_auc_partition_exhausts_frames_and_reproduces_strict_threshold_auc(self):
        clean = np.array([1., .05, .75, .1, .4, .2, .5, 0.])
        augmented = np.array([.6, .8, 0., .15, .35, .6, .5, 0.])
        rows, residual = audit.auc_partition(clean, augmented)
        self.assertEqual(sum(r["frames"] for r in rows), len(clean))
        self.assertEqual(len(rows), 5)
        self.assertLessEqual(residual, 1e-12)
        self.assertAlmostEqual(sum(r["contribution_to_full_auc"] for r in rows), success_auc(augmented) - success_auc(clean))
        self.assertAlmostEqual(audit.frame_auc(np.array([.05]))[0], 100 / 21)
        self.assertTrue(all(r["conditional_auc_delta"] is None for r in audit.auc_partition(np.ones(5), np.ones(5))[0] if r["frames"] == 0))

    def test_reduced_stress_loss_can_come_from_lower_unperturbed_accuracy(self):
        clean, augmented = np.full((7, 20), .7), np.full((7, 20), .65)
        clean[0], augmented[0] = .8, .6
        row, residual = audit.stress_accounting(clean, augmented)
        self.assertAlmostEqual(row["loss_delta"], -.15)
        self.assertAlmostEqual(row["perturbed_iou_delta"], -.05)
        self.assertAlmostEqual(row["contribution_from_unperturbed_decline"], .2)
        self.assertLessEqual(residual, 1e-12)
        common = dict(clean_auc=60., augmented_auc=61., auc_delta=1., low_iou_fraction_delta=0.,
            clean_failure_events=0., augmented_failure_events=0., clean_longest_failure_fraction=0., augmented_longest_failure_fraction=0.)
        common.update({c + "_auc_contribution": 1. if c == "both_reliable" else 0. for c in audit.COHORTS})
        seeds = [dict(sequence=str(v), seed=s, **common, **row) for v in range(8) for s in audit.source.SEEDS]
        with mock.patch.dict(audit.CONFIG, bootstrap_samples=64):
            videos, paired, focus = audit.aggregate(seeds)
        self.assertEqual(len(videos), 8)
        self.assertTrue(all(r["videos"] == 8 for r in paired))
        self.assertEqual(focus["reduced_loss_without_absolute_perturbed_gain_videos"], 8)
        self.assertEqual(focus["seed_consistency"]["both_positive"], 8)
        with self.assertRaisesRegex(ValueError, "eight videos"):
            audit.aggregate(seeds + seeds[:1])

    def test_own_parent_coverage_follows_output_error_and_event_indices_are_source_frames(self):
        n = 18
        gt = np.tile([20., 20., 10., 10.], (n, 1))
        boxes = gt.copy()
        boxes[9:, 0] = 60.
        entry = dict(sequence="synthetic", initial_box=gt[0].tolist(), frames=[dict(height=120, width=120)] * (n + 1))
        def data(pred):
            return dict(frame_id=np.arange(1, n + 1), boxes=pred, previous=np.vstack([gt[0], pred[:-1]]), gt=gt,
                iou=tracking_iou(pred, gt))
        clean, augmented = data(gt.copy()), data(boxes)
        plan = dict(search_factor=4., search_size=256)
        row, parts, events, timeline, phases, residual = audit.trajectory_audit(entry, 2110, clean, augmented, plan)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["onset_frame"], 10)
        self.assertEqual(events[0]["confirmed_frame"], 12)
        self.assertEqual(events[0]["onset_coverage"], 1.)
        self.assertEqual(events[0]["onset_parent_center_error_gt_units"], 0.)
        self.assertEqual(events[0]["onset_output_center_error_gt_units"], 4.)
        self.assertEqual(events[0]["onset_center_error_vector_jump_gt_units"], 4.)
        after = next(r for r in timeline if r["anchor_kind"] == "first_failure" and r["arm"] == "augmented" and r["offset"] == 1)
        self.assertEqual(after["coverage"], 0.)
        clean_row = next(r for r in phases if r["anchor_kind"] == "first_failure" and r["arm"] == "clean" and r["phase"] == "onset")
        self.assertEqual(clean_row["mean_iou"], 1.)
        self.assertLess(row["harmed_reliable_to_low_auc_contribution"], 0.)
        self.assertLessEqual(residual, 1e-12)


@unittest.skipUnless(TORCH, "CPU PyTorch needed only to create real completed Stage-40 test artifacts")
class SourceTests(unittest.TestCase):
    def test_completed_real_source_readonly_reuse_and_tamper_rejection(self):
        import torch
        import stage40_crop_common as common
        import stage40_crop_experiment as experiment
        import stage40_crop_summary as summary
        from test_stage40_crop_augmentation import fixture
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.dict(common.DESIGN, updates=3, smoke_updates=2, batch_size=1, save_every=1, bootstrap_samples=64), \
                mock.patch.dict(common.prior.DESIGN, horizon=3, smoke_horizon=2), \
                mock.patch.dict(audit.CONFIG, bootstrap_samples=64):
            root = Path(temporary)
            original, rt, reader, adapter, original_bindings = fixture(root, torch)
            source_root, output = root / "stage40", root / "stage41"
            with mock.patch.object(common, "source_info", side_effect=adapter):
                plan = common.prepare(original, source_root)
                experiment.smoke(rt, plan, source_root, reader)
                for seed in common.SEEDS:
                    experiment.train(rt, plan, source_root, seed, image_reader=reader)
                experiment.evaluate(rt, plan, source_root, image_reader=reader)
                source_report = summary.summarize(source_root)
                frozen_source = {str(p): file_hash(p) for p in source_root.rglob("*") if p.is_file()}
                with mock.patch.object(experiment, "train", side_effect=AssertionError("No training in cached audit")), \
                        mock.patch.object(experiment, "evaluate", side_effect=AssertionError("No inference in cached audit")), \
                        mock.patch.object(summary, "summarize", side_effect=AssertionError("Never rewrite source summary")):
                    report = audit.audit(source_root, output)
                    self.assertEqual(report["status"], "COMPLETE_CACHED_TRANSFER_AUDIT")
                    self.assertEqual(report["source_status"], source_report["status"])
                    self.assertFalse(report["new_inference_performed"])
                    self.assertFalse(report["training_performed"])
                    self.assertFalse(report["image_decoding_performed"])
                    self.assertIsNone(report["adoption_gate"])
                    self.assertLessEqual(report["maximum_auc_partition_residual"], 1e-10)
                    self.assertLessEqual(report["maximum_stress_accounting_residual"], 1e-10)
                    self.assertEqual(report["paired_trajectories"], 16)
                    self.assertLess((output / "stage41_diagnostic.json").stat().st_size, 18000)
                    stamps = {str(p): p.stat().st_mtime_ns for p in output.iterdir() if p.is_file()}
                    self.assertEqual(report, audit.audit(source_root, output))
                    self.assertEqual(stamps, {str(p): p.stat().st_mtime_ns for p in output.iterdir() if p.is_file()})
                    audit.verify_bindings(frozen_source)
                    audit.verify_bindings(original_bindings)
                    with self.assertRaisesRegex(ValueError, "separate"):
                        audit.audit(source_root, source_root / "analysis/stage41")
                    with mock.patch.dict(audit.CONFIG, before=11), self.assertRaisesRegex(ValueError, "source/code/configuration changed"):
                        audit.audit(source_root, output)
                    table = output / "stage41_video_summary.csv"
                    saved = table.read_bytes()
                    table.write_bytes(saved + b"tamper")
                    with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                        audit.audit(source_root, output)
                    table.write_bytes(saved)
                    source_cache = next((source_root / "evaluation/traces").glob("*.npz"))
                    source_cache.write_bytes(source_cache.read_bytes() + b"tamper source")
                    with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                        audit.audit(source_root, output)


if __name__ == "__main__":
    unittest.main()
