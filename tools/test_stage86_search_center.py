"""Verify the same-output experiment through a real CPU tracker fixture."""
import contextlib
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage86_search_center as stage
import test_stage75_ppo_failure as fixture


class Windows(unittest.TestCase):
    def test_prediction_rank_and_nonoverlap_on_shortest_video(self):
        boxes = np.tile([0., 0., 20., 20.], (145, 1))
        boxes[17:, 0] += 40
        boxes[65:, 1] += 20
        boxes[113:, 0] -= 60
        unit = dict(sequence="v0", origin="native", seed=2171, fold=0, policy="ppo")
        windows = stage.choose_windows(boxes, unit, stage.DESIGN)
        self.assertEqual([w["anchor"] for w in windows[::2]], [17, 65, 113])
        self.assertEqual([w["anchor"] for w in windows[1::2]], [1, 49, 97])
        self.assertEqual([w["predicted_displacement"] for w in windows[::2]], [2., 1., 3.])
        frames = [f for w in windows for f in range(w["anchor"] + 1, w["anchor"] + 17)]
        self.assertEqual(len(frames), len(set(frames)))


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Replay(unittest.TestCase):
    def test_real_center_write_fixed_parity_and_cache(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests
        import test_stage50_crossed

        torch = fixture.torch
        torch.set_num_threads(1)
        c, e74 = stage.c, stage.e74
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            original, native, old_observer, _ = TorchTests().fixture(root)
            old_observer.handle.remove()
            old_observer.pre_handle.remove()
            utils = Path(__file__).resolve().parents[1] / "lib/train/data/processing_utils.py"
            sample_target = runpy.run_path(str(utils))["sample_target"]
            observer = stage.e63.Capture(native)
            self.addCleanup(observer.handle.remove)
            entry = original["plan"]["entries"][0]
            for frame in entry["frames"]:
                path = Path(original["plan"]["root"]) / frame["file"]
                gradient = np.tile(np.arange(frame["width"], dtype=np.uint8), (frame["height"], 1))
                self.assertTrue(fixture.cv2.imwrite(str(path), np.repeat(gradient[..., None], 3, axis=2)))
                frame["sha256"] = c.file_hash(path)
            unit = dict(sequence=entry["sequence"], origin="native", fold=0, policy="ppo", seed=2171)
            old = dict(original["plan"], design=dict(c.DESIGN), actor_design=dict(stage.e63.source_common.DESIGN),
                       controller_design=dict(c.c66.DESIGN))
            pack = e74.learning.initial(torch, unit["seed"], dict(center=[0.] * 256, scale=[1.] * 256), old["design"])
            quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                           critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
            runtime = dict(tracker=dict(device="fixture"))
            source = dict(original=old, plan=dict(runtime=runtime), report=dict(native_head_sha256=digest(native.rt.net.box_head)))
            cases = [dict(case=s, unit=unit, time_bin=1, stratum=s, anchor=1, predicted_displacement=.2) for s in stage.STRATA]
            design = dict(stage.DESIGN, horizon=3, bootstrap_samples=32)
            plan = stage.io.seal(dict(schema=stage.SCHEMA, windows=cases, design=design, expected_search_calls=48,
                runtime=runtime, source_stage76_fingerprint="fixture", source_sha256={}, code_sha256={}))
            output = root / "stage86"
            output.mkdir()
            c.publish_identical(output / "manifest.json", plan)
            with (patch.object(test_stage50_crossed, "sample_target", sample_target),
                  patch.object(c, "fit_pack", return_value=quality),
                  patch.object(e74.source_experiment, "control_pack", return_value={"source": quality}),
                  patch.object(e74, "learned_pack", return_value=pack), contextlib.redirect_stdout(io.StringIO())):
                prefix = e74.run_actor(observer, entry, unit, old, {}, pack, limit=1)
                prefix["boxes"][0, 0] += 24  # Exercise a real shifted crop after a shared prediction jump.
                trace = e74.run_actor(observer, entry, unit, old, {}, pack, prefix=prefix)
                with patch.object(stage.previous, "read_source_trace", return_value=(entry, trace)):
                    stage.run(output, plan, source, {"models": {}, "prior": {}}, observer)
                    stage.check_results(output)
                    report = c.c66.sealed(output / "analysis/stage86_decision.json", stage.SCHEMA)
                    self.assertEqual((report["windows"], report["branches"], report["search_calls"]), (2, 4, 48))
                    self.assertEqual((report["source_max_box_error"], report["source_action_changes"]), (0., 0))
                    self.assertGreater(report["windows_with_changed_first_crop"], 0)
                    with patch.object(observer, "capture", side_effect=AssertionError("Committed replay repeated")):
                        stage.run(output, plan, source, {"models": {}, "prior": {}}, observer)
                    cache = output / "collection/high__midpoint.npz"
                    with cache.open("ab") as file:
                        file.write(b"changed")
                    with self.assertRaises(ValueError):
                        stage.check_results(output)


if __name__ == "__main__":
    unittest.main()
