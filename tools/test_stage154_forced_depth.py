"""Exercise an actual forced block call and verify that the tails stay native."""
import unittest
from contextlib import ExitStack

import numpy as np

import stage154_forced_depth_continuation as stage
import test_stage153_adaptive_depth as fixtures


def sample_target(image, parent):
    return image, 1., None


class PairedFixture(fixtures.NativeFixture):
    def track(self, image):
        self.frame_id += 1
        pixels, _, _ = sample_target(image, self.state)
        tokens = float(pixels[0, 0])
        for block in self.backbone.blocks[:6]:
            tokens = block(tokens)
        selected = self.backbone.MLP(tokens).layer
        decoded = self.head(self.backbone.blocks[selected-1](tokens))
        self.state = [decoded + self.state[0]*.25, 0., 10., 10.]
        return {"target_bbox":self.state}


class ForcedDepth(unittest.TestCase):
    def test_exposure_uses_same_crop_and_one_head_then_preserves_native_tails(self):
        tracker = PairedFixture()
        entry = dict(frames=list(range(34)), initial_box=[0.,0.,10.,10.], sequence="fixture")
        case = dict(case_id="fixture", sequence="fixture", kind="onset", group="onset_with_good_native_grid",
                    frame=1, successors=32)
        parent = np.array(entry["initial_box"])
        with ExitStack() as stack:
            observer = stage.DepthObserver(tracker, stack)
            control = stage.collect(observer, entry, case, parent, None,
                                    reader=lambda _,frame:np.array([[frame]], np.uint8))
        self.assertEqual(control["added"][1,0], 10)
        self.assertEqual(control["block_calls"][1].sum(), (32+1)*7+1)
        self.assertEqual(int(control["head_calls"].sum()), 66)
        self.assertNotEqual(control["boxes"][0,1,0], control["boxes"][1,1,0])
        reference = dict(frame_id=control["frame_id"], boxes=control["boxes"][:1], layer=control["selected"][:1],
                         parent=control["parent"][:1], crop_sha256=control["anchor_crop_sha256"][:1,None])
        stage.audit(control, case, reference, True, 0.)
        truth = np.repeat(control["boxes"][1,:1], 34, axis=0)
        anchor, windows, frames = stage.describe(control, case, truth, 0.)
        self.assertTrue(anchor["evaluated"])
        self.assertTrue(anchor["changed_box"])
        self.assertEqual(len(frames), 66)
        self.assertEqual(len(windows), 2)
        control["added"][1,1] = 10
        with self.assertRaises(AssertionError):
            stage.audit(control, case, reference, True, 0.)


if __name__ == "__main__":
    unittest.main()
