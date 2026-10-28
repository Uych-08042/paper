"""Factorial, censoring, real frozen continuations, provenance and resume checks."""

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
from stage27_uav_common import tracking_iou
import stage42_state_common as common
import stage42_state_experiment as experiment
import stage42_state_summary as summary

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


class MathTests(unittest.TestCase):
    def test_failure_tasks_deduplicate_owners_and_record_end_censoring(self):
        values = np.ones(80)
        values[10:30] = 0.
        entry = dict(sequence="x", frames=[{}] * 81)
        tasks, skipped = common.video_tasks(entry, 2110, {a:dict(iou=values) for a in common.ARMS})
        onset = next(t for t in tasks if t["frame_id"] == 11)
        self.assertEqual(len(onset["memberships"]), 2)
        self.assertEqual(len({t["frame_id"] for t in tasks}), len(tasks))
        self.assertEqual(len(tasks), 6)
        self.assertEqual(skipped, [])
        last = next(t for t in tasks if t["frame_id"] == 60)
        self.assertEqual(last["frames"], 21)
        short, skipped = common.video_tasks(dict(sequence="y", frames=[{}] * 6), 2111,
            {a:dict(iou=np.array([1., 1., 0., 0., 0.])) for a in common.ARMS})
        self.assertEqual(len(skipped), 4)
        self.assertTrue(all(s["reason"] == "after_video_end" for s in skipped))

    def test_parent_and_model_effects_close_and_short_horizons_remain_missing(self):
        effect, residual = summary.factorial([[.5, .1], [.6, .2]])
        self.assertAlmostEqual(effect["augmented_clean_parent_minus_augmented_own"], .4)
        self.assertAlmostEqual(effect["clean_augmented_parent_minus_clean_own"], -.4)
        self.assertAlmostEqual(effect["model_order_average"], .1)
        self.assertAlmostEqual(effect["parent_order_average"], -.4)
        self.assertAlmostEqual(effect["native_augmented_minus_clean"], -.3)
        self.assertLessEqual(residual, 1e-12)
        missing, _ = summary.factorial([[None, None], [None, None]])
        self.assertTrue(all(v is None for v in missing.values()))
        metrics = summary.branch_metrics(np.ones(12))
        self.assertIsNone(metrics[summary.primary_key()])
        self.assertEqual(metrics["post_available_mean_iou"], 1.)
        self.assertEqual(metrics["early_1_10_mean_iou"], 1.)
        self.assertIsNone(metrics["late_21_40_mean_iou"])
        offsets = summary.branch_metrics(np.array([0., 0., 0., .8, .8, .8, .1, .1, .1]))
        self.assertEqual(offsets["first_three_reliable_offset"], 3)
        self.assertEqual(offsets["first_relapse_offset"], 6)

    def test_group_aggregation_requires_two_supported_seeds_not_zero_filling(self):
        entries = [dict(sequence=str(i)) for i in range(8)]
        tasks = [dict(sequence=e["sequence"], seed=s, frame_id=10,
            memberships=[dict(kind="first_failure", owner="augmented", lag=0)]) for e in entries for s in common.training.SEEDS]
        plan = dict(fit=entries, tasks=tasks)
        values = [(t, {k:None if t["sequence"] == "0" and t["seed"] == 2111 else .1 for k in summary.EFFECTS}) for t in tasks]
        with mock.patch.dict(common.DESIGN, bootstrap_samples=64):
            grouped, videos = summary.aggregate(plan, values)
        main = next(r for r in grouped if r["cohort"] == "augmented_first_failure" and r["lag"] == 0 and r["effect"] == summary.EFFECTS[0])
        self.assertEqual(main["videos"], 7)
        self.assertAlmostEqual(main["mean_delta"], .1)
        self.assertEqual(main["excluded_videos"], "0")
        missing = next(r for r in videos if r["sequence"] == "0" and r["cohort"] == main["cohort"] and r["lag"] == 0 and r["effect"] == main["effect"])
        self.assertEqual(missing["supported_seed_count"], 1)
        self.assertIsNone(missing["mean_delta"])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            summary.aggregate(plan, values + values[:1])


@unittest.skipUnless(TORCH, "CPU PyTorch needed for actual frozen source-model replay")
class RolloutTests(unittest.TestCase):
    def test_real_source_diagonals_causal_swap_gt_isolation_resume_and_tamper_rejection(self):
        import torch
        import stage40_crop_common as c40
        import stage40_crop_experiment as e40
        import stage40_crop_summary as s40
        import stage41_transfer_audit as a41
        from test_stage40_crop_augmentation import fixture
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.dict(c40.DESIGN, updates=3, smoke_updates=2, batch_size=1, save_every=1, bootstrap_samples=64), \
                mock.patch.dict(c40.prior.DESIGN, horizon=3, smoke_horizon=2), \
                mock.patch.dict(a41.CONFIG, bootstrap_samples=64), \
                mock.patch.dict(common.DESIGN, followup_frames=3, smoke_frames=2, bootstrap_samples=64):
            root = Path(temporary)
            old, rt, reader, adapter, original_bindings = fixture(root, torch)
            r40, r41, output = root / "stage40", root / "stage41", root / "stage42"
            with mock.patch.object(c40, "source_info", side_effect=adapter):
                p40 = c40.prepare(old, r40)
                e40.smoke(rt, p40, r40, reader)
                for seed in c40.SEEDS:
                    e40.train(rt, p40, r40, seed, image_reader=reader)
                e40.evaluate(rt, p40, r40, image_reader=reader)
                s40.summarize(r40)
                a41.audit(r40, r41)
                frozen = {str(p): file_hash(p) for r in (r40, r41) for p in r.rglob("*") if p.is_file()}
                plan = common.prepare(r41, output)
                self.assertEqual(plan, common.load_plan(output))
                self.assertTrue(any(t["frames"] < 4 for t in plan["tasks"]))
                with self.assertRaises(FileNotFoundError):
                    experiment.collect(rt, plan, output, "full", image_reader=reader)
                experiment.collect(rt, plan, output, "smoke", image_reader=reader)
                experiment.smoke_gate(output, plan)
                smoke = common.read_sealed(output / "smoke/stage42_smoke.json", common.SCHEMA)
                self.assertTrue(all(p["max_box_error"] == 0 for p in smoke["native_parity"]))
                entry = plan["fit"][0]
                experiment.collect(rt, plan, output, "full", entry["sequence"], reader)
                with self.assertRaisesRegex(ValueError, "complete four-corner"):
                    summary.summarize(output)
                models, _ = e40.evaluation_models(rt, p40, r40)
                task = next(t for t in plan["tasks"] if t["sequence"] == entry["sequence"] and t["frames"] == 4)
                traces = common.source_traces(plan, entry, task["seed"])
                nets = {a:models[(experiment.fold_for(plan, entry), a + "_seed" + str(task["seed"]))] for a in common.ARMS}
                old_read = experiment.read_boxes
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    def read_after_branches(path):
                        self.assertEqual(spy.call_count, 4 * task["frames"])
                        return old_read(path)
                    with mock.patch.object(experiment, "read_boxes", side_effect=read_after_branches):
                        before = experiment.collect_unit(rt, plan, entry, task, nets, traces, image_reader=reader)
                # Changing future labels changes evaluation, never the four executed trajectories.
                future = Path(plan["root"]) / "annotations/other_future.txt"
                future.write_text("20,20,12,12\n" + "\n".join(["50,50,5,5"] * 12), encoding="utf-8")
                changed = dict(entry, annotation="annotations/other_future.txt", annotation_sha256=file_hash(future))
                changed_traces = copy.deepcopy(traces)
                for arm in common.ARMS:
                    changed_traces[arm]["gt"] = np.tile([50., 50., 5., 5.], (12, 1))
                    changed_traces[arm]["iou"] = tracking_iou(changed_traces[arm]["boxes"], changed_traces[arm]["gt"])
                after = experiment.collect_unit(rt, plan, changed, task, nets, changed_traces, image_reader=reader)
                np.testing.assert_array_equal(before["boxes"], after["boxes"])
                np.testing.assert_array_equal(before["previous"], after["previous"])
                self.assertFalse(np.array_equal(before["gt"], after["gt"]))
                np.testing.assert_array_equal(before["previous"][0, 0, 0], before["previous"][1, 0, 0])
                bad = dict(before, previous=before["previous"].copy())
                bad["previous"][1, 0, 1, 0] += 1
                with self.assertRaises(AssertionError):
                    common.audit_data(bad, plan, entry, task, traces)
                experiment.collect(rt, plan, output, "full", image_reader=reader)
                report = summary.summarize(output)
                self.assertEqual(report["status"], "COMPLETE_FROZEN_STATE_SWAP_DIAGNOSTIC")
                self.assertFalse(report["training_performed"])
                self.assertIsNone(report["adoption_gate"])
                self.assertEqual(report["maximum_native_box_error"], 0.)
                self.assertEqual(report["maximum_native_iou_error"], 0.)
                self.assertLessEqual(report["maximum_factorial_residual"], 1e-10)
                self.assertEqual(report["unique_anchor_packs"], len(plan["tasks"]))
                self.assertGreater(report["short_followup_anchors"], 0)
                common.verify_bindings(original_bindings)
                common.verify_bindings(frozen)
                self.assertEqual(report, summary.summarize(output))
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    experiment.collect(rt, plan, output, "full", image_reader=reader)
                    self.assertEqual(spy.call_count, 0)
                path = common.task_path(output, "full", task)
                path.with_suffix(".json").unlink()
                experiment.status(output, plan)
                experiment.collect(rt, plan, output, "full", entry["sequence"], reader)
                self.assertEqual(report, summary.summarize(output))
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "Stage-42 branch changed"):
                    summary.summarize(output)
                path.write_bytes(original)
                checkpoint = next((r40 / "training").rglob("step000003.pt"))
                checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper model")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    common.load_plan(output)


if __name__ == "__main__":
    unittest.main()
