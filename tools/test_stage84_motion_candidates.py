"""Exercise the matched crop budget and sealed replay with a real CPU CENTER head."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage84_motion_candidates as stage
import test_stage75_ppo_failure as fixture


class Layout(unittest.TestCase):
    def test_retains_q0_and_sizes_and_selects_forward_center(self):
        history = np.array([[x, 50., 20., 20.] for x in (10., 20., 30., 40., 50.)])
        ap = dict(design=dict(history=5), search_factor=4., search_size=256)
        hw = (160, 160)
        fixed, generated = stage.layouts(history, hw, ap, stage.DESIGN)
        np.testing.assert_array_equal(fixed[0], generated[0])
        np.testing.assert_array_equal(fixed[:, 2:], generated[:, 2:])
        self.assertEqual(generated[1, 0] + generated[1, 2]/2, 70.)
        self.assertEqual(len({tuple(stage.crop(p, hw, ap)) for p in generated}), 4)
        short = stage.layouts(history[:3], hw, ap, stage.DESIGN)
        np.testing.assert_array_equal(short[0], short[1])


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class MatchedReplay(unittest.TestCase):
    def test_real_center_source_bridge_and_resume(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        fixture.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            original, native, old_observer, _ = TorchTests().fixture(root)
            old_observer.handle.remove()
            old_observer.pre_handle.remove()
            observer = stage.e63.Capture(native)
            self.addCleanup(observer.handle.remove)
            original = dict(original["plan"], actor_design=dict(stage.e63.source_common.DESIGN))
            entry = original["entries"][0]
            ap = stage.s76.e.e67.actor_plan(original)
            n = len(entry["frames"])-1
            history = [np.array(entry["initial_box"], float)]
            updates = np.asarray([history[0] + [5*(i+1), 0, 0, 0] for i in range(n)])
            parents, crops = [], []
            for frame, box in zip(entry["frames"][1:], updates, strict=True):
                hw = (frame["height"], frame["width"])
                fixed = stage.s76.branches.prior.history_parents(history, ap["design"])
                parents.append(fixed)
                crops.append([stage.crop(p, hw, ap) for p in fixed])
                history.append(box)
            parents = np.asarray(parents)
            reference = stage.collect(observer, entry, np.stack((parents, parents), axis=1), ap)
            trace = dict(boxes=updates, candidate_parents=parents, crop=crops,
                         crop_sha256=reference["crop_hash"][:, 0], layer=reference["layers"][:, 0],
                         candidate_boxes=reference["boxes"][:, 0])
            units = [dict(sequence=entry["sequence"], origin="native", fold=0, seed=seed, policy="ppo")
                     for seed in (2171, 2172)]
            source = dict(original=original, report=dict(native_head_sha256=digest(native.rt.net.box_head)))
            design = dict(stage.DESIGN, bootstrap_samples=32)
            plan = stage.io.seal(dict(schema=stage.SCHEMA, units=units, states=2*n, design=design,
                runtime={}, source_stage76_fingerprint="fixture", source_sha256={}, code_sha256={}))
            output = root / "stage84"
            output.mkdir()
            stage.c.publish_identical(output / "manifest.json", plan)
            with (patch.object(stage.source_io, "read_source_trace", return_value=(entry, trace)),
                  contextlib.redirect_stdout(io.StringIO())):
                stage.run(output, plan, source, observer)
                stage.check_results(output)
                report = stage.c.c66.sealed(output / "analysis/stage84_decision.json", stage.SCHEMA)
                self.assertEqual(report["matched_states"], 2*n)
                self.assertEqual(report["logical_search_calls_per_arm"], 8*n)
                self.assertEqual(report["fixed_source_max_box_error"], 0.)
                self.assertGreater(report["changed_crop_states"], 0)
                self.assertLessEqual(report["physical_search_calls"], 14*n)
                with patch.object(observer, "capture", side_effect=AssertionError("Committed GPU replay repeated")):
                    stage.run(output, plan, source, observer)


if __name__ == "__main__":
    unittest.main()
