"""Check causal reference selection and exact slot replay with CPU fixtures."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage92_recent_appearance as stage
import test_stage75_ppo_failure as fixture


class References(unittest.TestCase):
    def test_references_use_past_frames_and_reject_ambiguous_history(self):
        boxes = np.tile([0., 0., 20., 20.], (20, 4, 1))
        boxes[14, 1, 0] = 15.  # Strong, spatially conflicting candidate at frame 15.
        boxes[11, 0, 0] = 15.  # Large accepted motion at frame 12.
        peaks = np.zeros((20, 4, len(stage.s76.branches.prior.FEATURES)))
        peaks[:, :, stage.s76.branches.prior.FEATURES.index("hann_peak")] = .5
        trace = dict(candidate_boxes=boxes, selected=np.zeros(20, int), features=peaks,
                     history_latest=np.tile([0., 0., 20., 20.], (20, 1)))
        self.assertEqual(stage.reference_frames(trace, 16, stage.DESIGN),
                         {"recent": [15, 12, 9], "gated": [14, 11, 8]})
        self.assertEqual(stage.reference_frames(trace, 1, stage.DESIGN), {"recent": [], "gated": []})

    def test_archived_stage91_matches_stage89_action_pairs(self):
        windows, report, _ = stage.s90.read_bundle(stage.s90.DEFAULT_INPUT)
        rows = stage.s90.decisions(windows, stage.DESIGN)
        fixed, _ = stage.load_fixed(stage.BUNDLE91, rows, report)
        self.assertEqual(len(fixed), 158)


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Replay(unittest.TestCase):
    def test_fixed_anchor_and_shared_reference_budget(self):
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
            rt.tracker.params.template_factor, rt.tracker.params.template_size = 2., 128
            rt.net.backbone.patch_embed = torch.nn.Identity()
            forward = rt.net.forward

            def hooked_forward():
                rt.net.backbone.patch_embed(torch.tensor([[[1., 0.]] * 4])).add_(10.)
                rt.net.backbone.patch_embed(torch.tensor([[[1., 0.]] * 16])).add_(20.)
                return forward()

            entry = original["plan"]["entries"][0]
            ap = stage.e74.e67.actor_plan(dict(original["plan"], actor_design=dict(stage.e63.source_common.DESIGN)))

            def choose(row, _):
                return dict(row, selected=0, boxes=row["candidate_boxes"][0].copy())

            trace = stage.s86.collect(observer, entry, np.asarray([entry["initial_box"]]), np.arange(1, 5), 0., ap, choose)
            refs = {"recent": [3, 2], "gated": [3]}
            with patch.object(rt.net, "forward", side_effect=hooked_forward) as replay:
                observed = stage.collect(observer, entry, trace, 4, (0, 2), refs, [1., 1.], ap, stage.DESIGN)
                self.assertEqual(replay.call_count, 4)
                np.testing.assert_array_equal(observed["counts"], [2, 1])
                np.testing.assert_allclose(observed["margins"], 0., atol=1e-12, rtol=0.)
                with self.assertRaisesRegex(AssertionError, "fixed appearance changed"):
                    stage.collect(observer, entry, trace, 4, (0, 2), refs, [1., .9], ap, stage.DESIGN)
            self.assertEqual(len(rt.net.backbone.patch_embed._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
