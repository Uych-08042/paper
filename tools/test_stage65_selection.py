"""Conditional-rate semantics and a real Stage-59..65 read-only cache workflow."""
import copy
import builtins
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import stage65_selection_common as c
import stage65_selection_audit as a


class Semantics(unittest.TestCase):
    def fixture(self):
        gt = np.array([40., 40., 20., 20.]); far = gt+[100, 0, 0, 0]
        boxes = np.array([[gt, far, gt, far], [far]*4, [far, gt, far, far], [far]*4])
        base = np.zeros((4, 4, 24)); f = np.zeros((4, 4, 64)); f[:, 1, 0] = 1.
        chosen = np.zeros(4, int)
        data = dict(candidate_boxes=boxes, boxes=boxes[:, 0], candidate_parents=np.tile(gt, (4, 4, 1)),
                    crop=np.tile([0., 0., 128., 128.], (4, 4, 1)), image_hw=np.tile([256, 256], (4, 1)),
                    features=base, selected=chosen)
        entry = dict(ground_truth=np.tile(gt, (5, 1)))
        models = {target:dict(target=target, scale=[1.]*64, weights=[sign]+[0.]*63)
                  for target, sign in (("quality", -1.), ("reliability", 1.))}
        return data, entry, f, models

    def test_matched_choices_harm_rescue_empty_support_and_gt_isolation(self):
        data, entry, f, models = self.fixture(); saved = copy.deepcopy(data)
        metrics, detail, choices, scores = c.metrics(data, entry, f, models, 1e-6)
        np.testing.assert_array_equal(choices["quality"], np.zeros(4))
        np.testing.assert_array_equal(choices["reliability"], np.ones(4))
        r = c.summarize(metrics["reliability"])
        self.assertEqual(r["q0_good_to_low_count"], 1); self.assertEqual(r["rescued_count"], 1)
        self.assertEqual(r["miss_given_good_decoded"], .5); self.assertEqual(r["low_given_good_q0"], 1.)
        self.assertEqual(r["rescue_given_opportunity"], 1.)
        mask = np.array([False, True, False, True])
        absent = c.summarize(metrics["reliability"], mask)
        self.assertIsNone(absent["miss_given_good_decoded"]); self.assertIsNone(absent["loss_given_good_q0"])
        empty = c.summarize(metrics["quality"], np.zeros(4, bool)); self.assertIsNone(empty["mean_selected_iou"])
        changed = dict(entry, ground_truth=np.tile([2., 2., 1., 1.], (5, 1)))
        other = c.metrics(data, changed, f, models, 1e-6)
        for rule in c.RULES:
            np.testing.assert_array_equal(choices[rule], other[2][rule]); np.testing.assert_array_equal(scores[rule], other[3][rule])
        for k in data: np.testing.assert_array_equal(data[k], saved[k])

    def test_weighted_fit_support_macro_support_and_missing_anchor(self):
        data, entry, f, models = self.fixture()
        ref = c.reference(f, np.ones(4), models); values = c.shifted(f, np.array([[.6, .7, .2, .19]]*4), ref)
        self.assertEqual(values["coordinate_outside_fit_fraction"], 0.)
        self.assertAlmostEqual(values["same_utility_different_iou_pair_fraction"], 1/3)
        changed = f.copy(); changed[:, 1, 0] = 10.
        self.assertGreater(c.shifted(changed, np.ones((4, 4)), ref)["pair_any_outside_fit_fraction"], 0)
        rows = [dict(sequence="v1", frame_states=100, x=.2), dict(sequence="v1", frame_states=100, x=None),
                dict(sequence="v2", frame_states=1, x=.8)]
        videos = c.aggregate(rows, ("sequence",), "seeds", ("x",)); macro = c.aggregate(videos, (), "videos", ("x",))[0]
        self.assertAlmostEqual(macro["x"], .5); self.assertEqual(videos[0]["x_supported_seeds"], 1)
        values = np.array([.8, .1, .1, .1, .6, .6, .6])
        self.assertEqual(c.anchor_frame("first_failure", values), 2)
        self.assertEqual(c.anchor_frame("first_recovery", values), 5)
        with self.assertRaisesRegex(ValueError, "no support"): c.anchor_frame("first_failure", np.ones(5))


try:
    import torch
    import cv2
except ImportError: torch = cv2 = None


@unittest.skipUnless(torch is not None and cv2 is not None, "Real CPU Torch/OpenCV required; skips are not execution validation")
class Workflow(unittest.TestCase):
    def test_real_stage64_chain_no_images_inference_refit_or_source_mutation(self):
        from test_stage64_reliability import Workflow as Previous
        import stage64_reliability_summary as s64
        original = s64.summarize; captured = {}

        def extend(root):
            result = original(root)
            if captured: return result
            captured["ran"] = True
            plan, _ = c.c64.load_plan(root)
            spec = dict(case_id="fixture", sequence=plan["entries"][0]["sequence"], origin="native", seed=0,
                        anchor_policy="reliability", anchor="fixed", frame=2)
            dest = Path(root).parent/"stage65"
            before = {str(p):p.read_bytes() for p in Path(root).parent.rglob("*") if p.is_file()}
            read_bytes = Path.read_bytes

            def refuse_images(path):
                if path.suffix.lower() in (".png", ".jpg", ".jpeg"): raise AssertionError("No image file reads in audit")
                return read_bytes(path)

            def guard_open(opener):
                def checked(path, *args, **kwargs):
                    if isinstance(path, (str, Path)) and Path(path).suffix.lower() in (".png", ".jpg", ".jpeg"):
                        raise AssertionError("No raw image reads or image hashing in audit")
                    return opener(path, *args, **kwargs)
                return checked

            with patch.object(c, "CASES", (spec,)), patch.dict("sys.modules", torch=None, cv2=None), \
                 patch.object(c.e64, "run_actor", side_effect=AssertionError("No new inference")), \
                 patch.object(Path, "read_bytes", refuse_images), \
                 patch.object(builtins, "open", side_effect=guard_open(builtins.open)), \
                 patch.object(io, "open", side_effect=guard_open(io.open)):
                report = a.audit(root, dest); self.assertEqual(report, a.audit(root, dest))
                self.assertEqual(report["source_trajectories"], 16)
                self.assertEqual(report["held_out_frame_states"], 64)
                self.assertEqual(report["fit_reference_frame_states"], 16)
                self.assertEqual(report["cached_decoded_box_occurrences"], 320)
                self.assertEqual(report["focus_rows"], 16)
                self.assertEqual(report["new_network_calls"], 0); self.assertFalse(report["training_performed"])
                self.assertTrue(report["old_fit_coefficients_reconstructed"])
                original_completed = c.e64.completed
                def changed_heads(*args):
                    run = copy.deepcopy(original_completed(*args))
                    run["observed_heads"][next(iter(run["observed_heads"]))] = "wrong-head"
                    return run
                with patch.object(c.e64, "completed", side_effect=changed_heads):
                    with self.assertRaisesRegex(ValueError, "head identity"): a.load_source(root)
                with self.assertRaisesRegex(ValueError, "separate"): a.audit(root, Path(root)/"bad")
                table = dest/"analysis/stage65_policy_summary.csv"; saved = table.read_bytes(); table.write_bytes(saved+b"tamper")
                with self.assertRaisesRegex(ValueError, "differs"): a.audit(root, dest)
                table.write_bytes(saved)
                source_table = Path(root)/"analysis/stage64_scope.csv"; saved = source_table.read_bytes()
                source_table.write_bytes(saved+b"tamper")
                with self.assertRaises(ValueError): a.load_source(root)
                source_table.write_bytes(saved)
                source_models = Path(root)/"models/training.json"; saved = source_models.read_bytes(); source_models.write_bytes(saved+b"tamper")
                with self.assertRaises(ValueError): a.load_source(root)
                source_models.write_bytes(saved)
            self.assertTrue(all(Path(p).read_bytes()==payload for p, payload in before.items()))
            return result

        with patch.object(s64, "summarize", side_effect=extend):
            Previous().test_real_source_chain_fit_isolation_controls_resume_and_frozen_weights()
        self.assertTrue(captured)


if __name__ == "__main__": unittest.main()
