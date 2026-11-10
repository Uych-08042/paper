"""Check online memory feedback, recovery guards and full CPU fixture execution."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage93_appearance_closed_loop as stage
import test_stage75_ppo_failure as fixture


class Policy(unittest.TestCase):
    def test_only_final_accepted_descriptor_enters_future_memory(self):
        boxes = np.array([[0., 0., 20., 20.], [2., 0., 20., 20.], [0., 0., 20., 20.], [2., 0., 20., 20.]])
        features = np.zeros((4, len(stage.s76.branches.prior.FEATURES)))
        features[:, stage.s76.branches.prior.FEATURES.index("hann_peak")] = [.5, .49, .5, .49]
        row = dict(candidate_boxes=boxes, candidate_parents=boxes, history_latest=boxes[0], features=features, selected=0)
        memory = [(1, np.array([0., 1.])), (4, np.array([0., 1.]))]
        queries = np.eye(2)
        result = stage.choose_write(row, queries, np.array([1., 0.]), memory, 7, "gated", stage.DESIGN)
        self.assertEqual(result["selected"], 1)
        np.testing.assert_array_equal(result["refs"], [4, 1, -1])
        np.testing.assert_array_equal(memory[-1][1], queries[1])
        self.assertTrue(result["used_history"])
        keep = stage.choose_write(row, queries, np.array([1., 0.]), [], 7, "gated", stage.DESIGN)
        self.assertEqual(keep["selected"], 0)
        self.assertFalse(keep["used_history"])

    def test_recovery_window_is_anchored_to_ppo_recovery(self):
        baseline = np.r_[np.zeros(3), np.ones(20)]
        late = np.r_[np.zeros(19), np.ones(4)]
        aligned = np.r_[np.zeros(18), np.ones(5)]
        self.assertIsNone(stage.recovery_comparison(baseline, late, stage.DESIGN)[0]["matched_good_frame"])
        self.assertEqual(stage.recovery_comparison(baseline, aligned, stage.DESIGN)[0]["matched_good_frame"], 19)


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Replay(unittest.TestCase):
    def test_full_arms_and_resume(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        torch = fixture.torch
        torch.set_num_threads(1)
        c = stage.c
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            original, native, old_observer, _ = TorchTests().fixture(root)
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

            entry = dict(original["plan"]["entries"][0], sequence="uav0000126_07915_s")
            old = dict(original["plan"], actor_design=dict(stage.e63.source_common.DESIGN))
            unit = dict(sequence=entry["sequence"], origin="native", fold=0, policy="ppo", seed=2171)
            source = dict(original=old, report=dict(native_head_sha256=digest(native.rt.net.box_head)))
            actor = dict(models={}, prior={})
            root89, output = root / "stage89", root / "stage93"
            plan89 = dict(fingerprint="fixture89", units=[dict(unit=unit, cohort="expanded")])
            c.publish_identical(root89 / "entries" / (entry["sequence"] + ".json"), stage.io.seal(dict(
                schema=stage.s89.SCHEMA, plan_fingerprint=plan89["fingerprint"], entry=entry)))
            plan = stage.io.seal(dict(schema=stage.SCHEMA, trajectories=4, new_trajectories=4, reuse_ppo=False,
                reuse_native_historical=False, runtime={}, design=dict(stage.DESIGN, bootstrap_samples=16),
                source_sha256={}, code_sha256={}))
            c.publish_identical(output / "manifest.json", plan)

            def choose(row, *_):
                return dict(row, selected=0, boxes=row["candidate_boxes"][0].copy())

            with (patch.object(rt.net, "forward", side_effect=hooked_forward),
                  patch.object(stage.e74, "choose", side_effect=choose), patch.object(stage.e74, "learned_pack"),
                  contextlib.redirect_stdout(io.StringIO())):
                stage.collect_run(output, plan, source, actor, root89, plan89, observer)
                stage.check_results(output)
                report = c.c66.sealed(output / "analysis/stage93_decision.json", stage.SCHEMA)
                self.assertEqual((report["trajectories"], report["new_search_calls"], report["gated_switches"]), (4, 52, 0))
                with patch.object(observer, "capture", side_effect=AssertionError("Cached searches repeated")):
                    stage.collect_run(output, plan, source, actor, root89, plan89, observer)


if __name__ == "__main__":
    unittest.main()
