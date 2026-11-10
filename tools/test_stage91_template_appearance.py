"""Exercise ROI pooling and exact native replay with a hook that reads before mutation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage91_template_appearance as stage
import test_stage75_ppo_failure as fixture


class Appearance(unittest.TestCase):
    def test_visible_tokens_and_rgb_use_foreground_roi(self):
        image = np.zeros((8, 8, 3), np.uint8)
        image[:, 4:] = 255
        crop = np.array([0, 0, 8, 8])
        roi, coverage = stage.visible_roi([-2, 0, 6, 4], crop)
        np.testing.assert_array_equal(roi, [0, 0, 4, 4])
        self.assertAlmostEqual(coverage, 2 / 3)
        tokens = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
        np.testing.assert_array_equal(stage.descriptor(tokens, crop, roi), [1, 0])
        np.testing.assert_array_equal(stage.descriptor(tokens, crop, np.array([4, 0, 4, 4])), [0, 1])
        black = stage.histogram(image, roi, 8)
        white = stage.histogram(image, np.array([4, 0, 4, 4]), 8)
        self.assertAlmostEqual(black @ black, 1.)
        self.assertAlmostEqual(black @ white, 0.)

    def test_absent_appearance_keeps_ppo_and_shared_ranking_support(self):
        sample = dict(stratum="conflict", cohort="expanded", source_group="uav0000001", case="sample",
                      material_help=False, material_harm=True, tail_help=False, tail_harm=True,
                      ppo_only_recovery=True, challenger_only_recovery=False, followup_gain=-.4,
                      low_iou_delta_frames=8, patch_margin=None, rgb_margin=.5, hann_margin=.1,
                      velocity_margin=.1, consensus_margin=.1)
        support, paired = stage.summarize([sample], dict(stage.DESIGN, bootstrap_samples=16))
        row = next(r for r in support if r["cohort"] == "all" and r["method"] == "patch")
        self.assertEqual((row["eligible"], row["switches"], row["ppo_only_recovery_switched"]), (0, 0, 0))
        self.assertTrue(all(r["rank_windows"] == 0 and r["rank_auc"] is None for r in support))
        metric = next(r for r in paired if r["cohort"] == "all" and r["contrast"] == "patch_minus_ppo")
        self.assertEqual(metric["mean_delta"], 0.)


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Replay(unittest.TestCase):
    def test_hook_reads_before_position_encoding_and_preserves_frame(self):
        from test_stage61_search import TorchTests

        torch = fixture.torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            original, native, old_observer, _ = TorchTests().fixture(Path(directory))
            old_observer.handle.remove()
            old_observer.pre_handle.remove()
            observer = stage.e63.Capture(native)
            self.addCleanup(observer.handle.remove)
            rt = native.rt
            rt.lens_z, rt.lens_x = 4, 16
            rt.tracker.params.template_factor = 2.
            rt.tracker.params.template_size = 128
            rt.net.backbone.patch_embed = torch.nn.Identity()
            forward = rt.net.forward

            def hooked_forward():
                z = rt.net.backbone.patch_embed(torch.tensor([[[1., 0.]] * 4]))
                x = rt.net.backbone.patch_embed(torch.tensor([[[1., 0.]] * 16]))
                z.add_(10.)
                x.add_(20.)
                return forward()

            entry = original["plan"]["entries"][0]
            ap = stage.e74.e67.actor_plan(dict(original["plan"], actor_design=dict(stage.e63.source_common.DESIGN)))
            prefix = np.asarray([entry["initial_box"], np.asarray(entry["initial_box"]) + [24., 0., 0., 0.]])
            frames = np.array([2])

            def choose(row, _):
                return dict(row, selected=0, boxes=row["candidate_boxes"][0].copy())

            fixed = stage.s86.collect(observer, entry, prefix, frames, 0., ap, choose)
            item = dict(prefix=prefix, frames=frames, arms=[fixed], action=2)
            with patch.object(rt.net, "forward", side_effect=hooked_forward):
                scores = stage.observe(observer, entry, item, ap, choose, stage.DESIGN)
                np.testing.assert_allclose(scores["scores"], [[1., 0.], [1., 0.]], atol=1e-12, rtol=0)
                item["arms"] = [dict(fixed, selected=np.array([1]))]
                with self.assertRaisesRegex(AssertionError, "first-frame replay changed selected"):
                    stage.observe(observer, entry, item, ap, choose, stage.DESIGN)
            self.assertEqual(len(rt.net.backbone.patch_embed._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
