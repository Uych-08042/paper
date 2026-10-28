"""Matched interventions, video aggregation and real CPU frozen-branch workflow."""

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
from stage39_perturb_common import DESIGN, MODELS, PERTURBATIONS, anchors, perturb
import stage39_perturb_common as common
import stage39_perturb_experiment as experiment
import stage39_perturb_summary as summary

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


class MathTests(unittest.TestCase):
    def test_intervention_is_parent_relative_center_preserving_and_not_gt_dependent(self):
        parent = np.array([10., 20., 16., 25.])
        original = parent.copy()
        np.testing.assert_array_equal(perturb(parent, "clean"), parent)
        for axis in ("x", "y"):
            pair = [perturb(parent, axis + "_" + sign) for sign in ("minus", "plus")]
            np.testing.assert_allclose((pair[0] + pair[1]) / 2, parent, atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(pair[0][:2] - parent[:2]), 3.)
        for name, scale in (("scale_down", .8), ("scale_up", 1.25)):
            result = perturb(parent, name)
            np.testing.assert_array_equal(result[2:], parent[2:] * scale)
            np.testing.assert_allclose(result[:2] + result[2:] / 2, parent[:2] + parent[2:] / 2)
        np.testing.assert_array_equal(parent, original)
        with self.assertRaises(ValueError):
            perturb(parent, "choose_by_GT")
        entry = dict(frames=[{}] * 101)
        self.assertEqual(anchors(entry), [20, 40, 60])

    def test_excess_is_difference_of_losses_not_absolute_accuracy_and_seeds_are_paired(self):
        plan = dict(fit=[dict(sequence=str(i), source_group_proxy="g" + str(i)) for i in range(8)],
            tasks={str(i): [10, 20, 30] for i in range(8)})
        rows = []
        for entry, anchor in common.tasks(plan, "full"):
            for m in MODELS:
                for p in PERTURBATIONS[1:]:
                    clean = .6 if m == "sgla" else .9
                    loss = dict(sgla=.01, fast_initial=.04, supervised_seed2110=.02, supervised_seed2111=.06)[m]
                    rows.append(dict(sequence=entry["sequence"], anchor=anchor, model=m, perturbation=p,
                        clean_first_iou=clean, perturbed_first_iou=clean - loss, first_iou_loss=loss,
                        clean_post_iou=clean, perturbed_post_iou=clean - loss, post_iou_loss=loss,
                        late_iou_loss=loss, post_low_iou_increase=0.))
        with mock.patch.dict(DESIGN, bootstrap_samples=64):
            tables, primary, checks = summary.analyze(rows, plan)
        self.assertAlmostEqual(primary["mean_delta"], .03)
        self.assertTrue(all(checks.values()))
        self.assertEqual(primary["videos"], 8)
        averaged = next(r for r in tables["paired"] if r["contrast"] == "supervised_mean_minus_sgla" and r["metric"] == "post_iou_loss")
        self.assertAlmostEqual(averaged["mean_delta"], .03)
        self.assertEqual(averaged["videos"], 8)
        with self.assertRaisesRegex(ValueError, "complete unmixed"):
            summary.analyze(rows + rows[:1], plan)
        # Positive mean dominated by one video fails direction and leave-one-out checks.
        altered = copy.deepcopy(rows)
        for r in altered:
            if r["model"] == "fast_initial":
                r["post_iou_loss"] = .5 if r["sequence"] == "0" else 0.
        with mock.patch.dict(DESIGN, bootstrap_samples=64):
            _, _, bad = summary.analyze(altered, plan)
        self.assertFalse(bad["cross_video_direction_pass"])
        self.assertFalse(bad["leave_one_video_out_pass"])

    def test_primary_excludes_injected_frame_but_retains_negative_losses(self):
        data = dict(iou=np.full((4, 7, 20), .5))
        data["iou"][:, 1:, 0] = 0.
        data["iou"][1, 1:, 1:] = .6
        rows = summary.branch_rows(dict(sequence="x", source_group_proxy="g"), 10, data)
        teacher = next(r for r in rows if r["model"] == "sgla")
        fast = next(r for r in rows if r["model"] == "fast_initial")
        self.assertEqual(teacher["first_iou_loss"], .5)
        self.assertEqual(teacher["post_iou_loss"], 0.)
        self.assertAlmostEqual(fast["post_iou_loss"], -.1)


@unittest.skipUnless(TORCH, "PyTorch needed for actual CPU source/branch checks")
class RolloutTests(unittest.TestCase):
    def test_full_provenance_weight_identity_gt_isolation_continuity_resume_and_tampering(self):
        import torch
        import stage37_crossed_common as c37
        import stage37_crossed_experiment as e37
        import stage37_crossed_summary as s37
        import stage38_geometry_common as c38
        import stage38_geometry_experiment as e38
        import stage38_geometry_summary as s38
        from test_stage37_crossed import source_fixture
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.dict(c37.DESIGN, smoke_frames=3, bootstrap_samples=64), \
                mock.patch.dict(c38.DESIGN, smoke_frames=3, bootstrap_samples=64), \
                mock.patch.dict(DESIGN, horizon=3, smoke_horizon=2, bootstrap_samples=64):
            root = Path(temporary)
            old, rt, reader, adapter, bindings = source_fixture(root, torch)
            r37, r38, output = root / "stage37", root / "stage38", root / "stage39"
            with mock.patch.object(c37, "source_info", side_effect=adapter):
                p37 = c37.prepare(old, r37)
                e37.collect(rt, p37, r37, "smoke", image_reader=reader)
                e37.collect(rt, p37, r37, "full", image_reader=reader)
                s37.summarize(r37)
                p38 = c38.prepare(r37, r38)
                e38.collect(rt, p38, r38, "smoke", image_reader=reader)
                e38.collect(rt, p38, r38, "full", image_reader=reader)
                s38.summarize(r38)
                frozen_sources = {str(p): file_hash(p) for r in (r37, r38) for p in r.rglob("*") if p.is_file()}
                plan = common.prepare(r38, output)
                self.assertEqual(plan, common.load_plan(output))
                with self.assertRaises(FileNotFoundError):
                    experiment.collect(rt, plan, output, "full", image_reader=reader)
                experiment.collect(rt, plan, output, "smoke", image_reader=reader)
                experiment.smoke_gate(output, plan)
                entry = plan["fit"][0]
                experiment.collect(rt, plan, output, "full", entry["sequence"], reader)
                with self.assertRaisesRegex(ValueError, "all 24"):
                    summary.summarize(output)
                models, _ = experiment.frozen_models(rt, plan)
                fast = models["fast_initial"]
                self.assertEqual(len(fast.backbone.blocks), 6)
                self.assertFalse(hasattr(fast.backbone, "MLP"))
                self.assertTrue(all(torch.equal(v, rt.net.state_dict()[k]) for k, v in fast.state_dict().items()))
                ref = common.source_trace(plan, entry)
                anchor = plan["tasks"][entry["sequence"]][0]
                # All 28 branches finish before any future label read.
                old_read = experiment.read_boxes
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    def read_after_branches(path):
                        self.assertEqual(spy.call_count, 28 * DESIGN["horizon"])
                        return old_read(path)
                    with mock.patch.object(experiment, "read_boxes", side_effect=read_after_branches):
                        data = experiment.collect_pack(rt, plan, entry, anchor, models, ref, 3, reader)
                changed_annotation = Path(plan["root"]) / "annotations/changed_future.txt"
                changed_annotation.write_text("20,20,12,12\n" + "\n".join(["50,50,5,5"] * 12), encoding="utf-8")
                changed = dict(entry, annotation="annotations/changed_future.txt", annotation_sha256=file_hash(changed_annotation))
                changed_ref = dict(ref, iou=tracking_iou(ref["boxes"], np.tile([50., 50., 5., 5.], (12, 1))))
                after = experiment.collect_pack(rt, plan, changed, anchor, models, changed_ref, 3, reader)
                np.testing.assert_array_equal(data["boxes"], after["boxes"])
                np.testing.assert_array_equal(data["previous"], after["previous"])
                self.assertFalse(np.array_equal(data["gt"], after["gt"]))
                bad = dict(data, previous=data["previous"].copy())
                bad["previous"][1, 1, 1, 0] += 1
                with self.assertRaises(AssertionError):
                    common.audit(bad, plan, entry, anchor, 3, ref)
                bad = dict(data, raw=data["raw"].copy())
                bad["raw"][0, 0, 0, 0, 0] = np.nan
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    common.audit(bad, plan, entry, anchor, 3, ref)
                experiment.collect(rt, plan, output, "full", image_reader=reader)
                report = summary.summarize(output)
                self.assertEqual(report["videos"], 8)
                self.assertEqual(report["unique_branch_trajectories"], 672)
                self.assertFalse(report["training_performed"])
                self.assertFalse(report["learned_method_gain_established"])
                common.verify_bindings(bindings)
                common.verify_bindings(frozen_sources)
                self.assertEqual(report, summary.summarize(output))
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as spy:
                    experiment.collect(rt, plan, output, "full", image_reader=reader)
                    self.assertEqual(spy.call_count, 0)
                path = common.task_path(output, "full", entry, anchor)
                path.with_suffix(".json").unlink()
                experiment.status(output, plan)
                experiment.collect(rt, plan, output, "full", entry["sequence"], reader)
                self.assertEqual(report, summary.summarize(output))
                saved = path.read_bytes()
                path.write_bytes(saved + b"tamper")
                with self.assertRaisesRegex(ValueError, "Stage-39 branch changed"):
                    summary.summarize(output)
                path.write_bytes(saved)
                checkpoint = Path(plan["source_input"]) / "training/seed2111/fold3.pt"
                checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper checkpoint")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    common.load_plan(output)


if __name__ == "__main__":
    unittest.main()
