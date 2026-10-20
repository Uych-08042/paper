"""Stage-18 supervised loss, crop geometry, masking, and gradient checks."""

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

# Support direct execution as well as unittest discovery from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.stage18_tracking_loss import tracking_loss


class ImportGeometryTests(unittest.TestCase):
    def test_import_without_torch_or_tracker_dependencies(self):
        code = """
import builtins
import importlib.util
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'numpy', 'lib'}:
        raise AssertionError('Eager tracker dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('loss_helper', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.tracking_loss)
"""
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(ROOT / "tracking/stage18_tracking_loss.py")],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
    def test_gt_uses_previous_executed_crop_without_clipping(self):
        from tracking.stage18_environment import executed_crop_roi

        # Crop center=(30,60), side=100, origin=(-20,10); GT leaves crop.
        target = executed_crop_roi([-40, 30, 40, 20], [20, 40, 20, 40], 2.56, 256)
        for actual, expected in zip(target, [-.2, .2, .4, .2]):
            self.assertAlmostEqual(float(actual), expected, places=6)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class TrackingLossTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

    def make_output(self, boxes=((.5, .5, .5, .5),), shape=(4, 4), probability=.5,
                    dtype=None):
        torch = self.torch
        dtype = dtype or torch.float64
        return dict(
            pred_boxes=torch.tensor(boxes, dtype=dtype).unsqueeze(1).requires_grad_(),
            score_map=torch.full((len(boxes), 1, *shape), probability, dtype=dtype,
                                 requires_grad=True))

    def target(self, boxes=((.25, .25, .5, .5),)):
        return self.torch.tensor(boxes, dtype=self.torch.float64)

    def assert_finite_gradients(self, loss, output):
        torch = self.torch
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for tensor in output.values():
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_perfect_box_known_focal_and_tensor_stats(self):
        output = self.make_output()
        loss, stats = tracking_loss(self.torch, output, self.target())
        # A 2x2-cell GT gives radius zero: one positive and 15 negatives.
        expected = 16 * .25 * math.log(2)
        self.assertAlmostEqual(float(loss), expected, places=10)
        self.assertEqual(float(stats["giou_sum"]), 0)
        self.assertEqual(float(stats["l1_sum"]), 0)
        self.assertEqual(float(stats["iou_sum"]), 1)
        for key in ("supervised_count", "regression_count", "center_count"):
            self.assertEqual(int(stats[key]), 1)
        for value in stats.values():
            self.assertTrue(self.torch.is_tensor(value))
            self.assertEqual(value.ndim, 0)
            self.assertFalse(value.requires_grad)
        self.assert_finite_gradients(loss, output)

        perfect = self.make_output(probability=0.)
        with self.torch.no_grad():
            perfect["score_map"][0, 0, 2, 2] = 1
        perfect_loss, _ = tracking_loss(self.torch, perfect, self.target())
        self.assertLess(float(perfect_loss), 1e-12)
        self.assert_finite_gradients(perfect_loss, perfect)

    def test_known_disjoint_predictions_and_loss_weights(self):
        output = self.make_output(boxes=((.25, .25, .25, .25),))
        target = self.target(((.5, .5, .25, .25),))
        loss, stats = tracking_loss(self.torch, output, target)
        # Equal boxes offset by .375: IoU=0, union=.125, enclosure=.390625.
        expected_giou = 1 + (.390625 - .125) / .390625
        self.assertAlmostEqual(float(stats["giou_sum"]), expected_giou)
        self.assertAlmostEqual(float(stats["l1_sum"]), .375)
        self.assertAlmostEqual(float(loss), 2 * expected_giou + 5 * .375
                               + float(stats["focal_sum"]))
        self.assert_finite_gradients(loss, output)

    def test_partial_crop_regression_keeps_full_gt(self):
        output = self.make_output(boxes=((.125, .5, .75, .5),))
        loss, stats = tracking_loss(self.torch, output,
                                    self.target(((-.25, .25, .75, .5),)))
        self.assertEqual(int(stats["regression_count"]), 1)
        self.assertEqual(float(stats["giou_sum"]), 0)
        self.assertEqual(float(stats["l1_sum"]), 0)
        self.assert_finite_gradients(loss, output)

    def test_offscreen_center_has_no_gaussian_tail_but_keeps_intersection_regression(self):
        torch = self.torch
        output = self.make_output(boxes=((-.125, .5, .75, .5),), shape=(16, 16))
        target = self.target(((-.5, .25, .75, .5),))
        loss, stats = tracking_loss(torch, output, target)
        self.assertEqual(int(stats["regression_count"]), 1)
        self.assertEqual(int(stats["center_count"]), 0)
        self.assertEqual(float(stats["l1_sum"]), 0)
        self.assertAlmostEqual(float(stats["focal_sum"]), 256 * .25 * math.log(2))
        self.assert_finite_gradients(loss, output)
        self.assertTrue((output["score_map"].grad > 0).all())

    def test_fully_offcrop_or_touching_skips_invented_regression_target(self):
        torch = self.torch
        targets = [(1.25, .25, .5, .5), (-1, .25, .5, .5),
                   (.25, 1.25, .5, .5), (.25, -1, .5, .5),
                   (1, .25, .5, .5), (-.5, .25, .5, .5)]
        for target in targets:
            with self.subTest(target=target):
                output = self.make_output()
                loss, stats = tracking_loss(torch, output, self.target((target,)))
                self.assertEqual(int(stats["supervised_count"]), 1)
                self.assertEqual(int(stats["regression_count"]), 0)
                self.assertEqual(int(stats["center_count"]), 0)
                self.assertEqual(float(stats["giou_sum"]), 0)
                self.assertEqual(float(stats["l1_sum"]), 0)
                self.assertAlmostEqual(float(loss), 16 * .25 * math.log(2))
                self.assert_finite_gradients(loss, output)
                self.assertEqual(float(output["pred_boxes"].grad.abs().sum()), 0)
                self.assertTrue((output["score_map"].grad > 0).all())

    def test_saturated_scores_are_finite_in_supported_dtypes(self):
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for probability in (0., 1.):
                for offcrop in (False, True):
                    with self.subTest(dtype=dtype, probability=probability, offcrop=offcrop):
                        output = self.make_output(probability=probability, dtype=dtype)
                        target = self.target(((1.25, .25, .5, .5),)) if offcrop else self.target()
                        loss, _ = tracking_loss(torch, output, target)
                        self.assert_finite_gradients(loss, output)

    def test_all_invalid_is_differentiable_zero_even_for_nan_labels_and_outputs(self):
        torch = self.torch
        for valid in (False, torch.tensor(False), torch.tensor([False])):
            with self.subTest(valid=valid):
                output = self.make_output(boxes=((math.nan,) * 4,), probability=math.nan)
                loss, stats = tracking_loss(torch, output,
                                            self.target(((math.nan, math.inf, -1, 0),)), valid)
                self.assertEqual(float(loss), 0)
                self.assertTrue(all(float(value) == 0 for value in stats.values()))
                self.assert_finite_gradients(loss, output)
                self.assertTrue(all(torch.equal(tensor.grad, torch.zeros_like(tensor))
                                    for tensor in output.values()))

    def test_mixed_batch_matches_single_frame_sums_and_masks_invalid_rows(self):
        torch = self.torch
        output = self.make_output(boxes=((.5, .5, .5, .5), (.5, .5, .5, .5),
                                        (math.nan,) * 4))
        with torch.no_grad():
            output["score_map"][2] = math.nan
        target = self.target(((.25, .25, .5, .5), (1.25, .25, .5, .5), (math.nan,) * 4))
        loss, stats = tracking_loss(torch, output, target, torch.tensor([True, True, False]))
        single_stats = [tracking_loss(torch, {key: value[i:i + 1] for key, value in output.items()},
                                      target[i:i + 1])[1] for i in range(2)]
        for key, value in stats.items():
            torch.testing.assert_close(value, sum(row[key] for row in single_stats))
        torch.testing.assert_close(loss.detach(), stats["loss_sum"] / 2)
        self.assertEqual(int(stats["supervised_count"]), 2)
        self.assertEqual(int(stats["regression_count"]), 1)
        self.assert_finite_gradients(loss, output)
        self.assertEqual(float(output["pred_boxes"].grad[1:].abs().sum()), 0)
        self.assertEqual(float(output["score_map"].grad[2].abs().sum()), 0)

    def test_heatmap_matches_existing_generator_at_actual_grid_size(self):
        torch = self.torch
        from lib.utils.focal_loss import FocalLoss
        from lib.utils.heapmap_utils import generate_heatmap

        for side in (8, 20):
            with self.subTest(side=side):
                target = self.target()
                output = self.make_output(shape=(side, side))
                _, stats = tracking_loss(torch, output, target)
                expected_map = generate_heatmap(target.unsqueeze(0), patch_size=side, stride=1)[0]
                expected = FocalLoss()(output["score_map"], expected_map[:, None])
                torch.testing.assert_close(stats["focal_sum"], expected.detach(), rtol=1e-6, atol=1e-6)

    def test_rectangular_grid_and_inside_edge_center_have_positive_supervision(self):
        torch = self.torch
        for x in (-.125, .874):
            with self.subTest(x=x):
                output = self.make_output(shape=(4, 8))
                target = self.target(((x, .25, .25, .5),))
                loss, stats = tracking_loss(torch, output, target)
                self.assertEqual(int(stats["center_count"]), 1)
                self.assert_finite_gradients(loss, output)
                self.assertEqual(int((output["score_map"].grad < 0).sum()), 1)

    def test_declared_valid_bad_geometry_and_contract_errors_are_explicit(self):
        torch = self.torch
        for target in ((math.nan, 0, .5, .5), (0, 0, 0, .5), (0, 0, .5, -1)):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "Valid GT"):
                tracking_loss(torch, self.make_output(), self.target((target,)))
        with self.assertRaisesRegex(ValueError, "target_xywh"):
            tracking_loss(torch, self.make_output(), self.target()[0])
        with self.assertRaisesRegex(TypeError, "valid"):
            tracking_loss(torch, self.make_output(), self.target(), valid=1)
        with self.assertRaisesRegex(ValueError, "valid"):
            tracking_loss(torch, self.make_output(), self.target(), valid=torch.ones(2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "pred_boxes"):
            output = self.make_output()
            output["pred_boxes"] = output["pred_boxes"].expand(-1, 2, -1)
            tracking_loss(torch, output, self.target())
        with self.assertRaisesRegex(ValueError, "probabilities"):
            tracking_loss(torch, self.make_output(probability=1.01), self.target())


if __name__ == "__main__":
    unittest.main()
