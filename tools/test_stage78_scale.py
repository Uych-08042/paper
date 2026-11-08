"""Verify causal scale feedback, separated output and the real CENTER replay entry."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage78_scale_experiment as stage
import test_stage75_ppo_failure as fixture


class Geometry(unittest.TestCase):
    def test_only_growth_is_limited_and_center_is_preserved(self):
        history = np.array([[10., 15., 20., 40.]] * 4 + [[5., 5., 50., 60.]])
        box = np.array([10., 10., 80., 20.])
        limited = stage.bounded_box(box, history, stage.DESIGN)
        np.testing.assert_allclose(limited[2:], [25., 20.])
        np.testing.assert_allclose(limited[:2] + limited[2:] / 2, box[:2] + box[2:] / 2)
        small = np.array([10., 10., 16., 19.])
        np.testing.assert_array_equal(stage.bounded_box(small, history, stage.DESIGN), small)


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Continuation(unittest.TestCase):
    def test_real_center_matched_start_causal_update_and_verified_export(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        torch = fixture.torch
        torch.set_num_threads(1)
        c, e74 = stage.c, stage.e74
        with tempfile.TemporaryDirectory(prefix="stage78-", dir=Path(__file__).parent) as temp:
            folder = Path(temp)
            orig, native, old_observer, _ = TorchTests().fixture(folder)
            old_observer.handle.remove()
            old_observer.pre_handle.remove()
            runtime = dict(orig["plan"]["runtime_identity"], tracker=dict(device="cpu"))
            entry = dict(orig["plan"]["entries"][0], initial_box=[40., 40., 8., 20.])
            old = dict(orig["plan"], entries=[entry], runtime_identity=runtime, design=dict(c.DESIGN),
                       actor_design=dict(stage.e63.source_common.DESIGN), controller_design=dict(c.c66.DESIGN))
            unit = dict(sequence=entry["sequence"], origin="native", fold=0, seed=2172, policy="ppo")
            pack = e74.learning.initial(torch, 2172, dict(center=[0.] * 256, scale=[1.] * 256), old["design"])
            quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                           critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
            source_root = folder / "source"
            source = dict(root=source_root, original=old, plan=dict(runtime=runtime, fingerprint="source"), evidence={},
                          report=dict(source_training_fingerprint="fixture", native_head_sha256=digest(native.rt.net.box_head)))
            actor = dict(prior={}, models={"controllers": {c.job_key(unit): {"ppo": pack}}})
            case = dict(case="fixture", role="primary", unit=unit, first=1, last=4, focus=3)
            root = folder / "new"
            argv = ["stage78", "--input", str(source_root), "--diagnostic", str(folder / "previous"),
                    "--output_dir", str(root), "--checkpoint", "fixture"]
            hooks = tuple(native.rt.net._forward_hooks)
            observer = stage.e63.Capture(native)
            try:
                with (patch.object(c, "fit_pack", return_value=quality),
                      patch.object(e74.source_experiment, "control_pack", return_value={"source": quality})):
                    trace = e74.run_actor(observer, entry, unit, old, {}, pack)
                    for policy, controller in (("myopic", pack), ("q0", None)):
                        control = dict(unit, policy=policy)
                        baseline = e74.run_actor(observer, entry, control, old, {}, controller, limit=2)
                        actual = stage.collect(observer, entry["frames"], entry["initial_box"],
                                               np.asarray([entry["initial_box"]]), [1, 2], control, old, {},
                                               controller, stage.DESIGN, False)
                        for key in stage.FIELDS:
                            np.testing.assert_array_equal(actual[key], baseline[key])
            finally:
                observer.handle.remove()
            path = source_root / "evaluation" / (c.key(unit) + ".npz")
            c.save(path, trace, dict(plan_fingerprint="source", unit=unit, controller="fixture"))
            stage.s76.binding(source["evidence"], path)

            with (patch("sys.argv", argv), patch.object(stage, "cases", return_value=[case]),
                  patch.object(stage, "load_sources", return_value=(source, actor, dict(fingerprint="diagnosis"))),
                  patch.object(stage.s76, "device_runtime", return_value=runtime),
                  patch.object(stage.s76.startup, "checked_runtime", return_value=(native, runtime)),
                  patch.object(c, "fit_pack", return_value=quality), contextlib.redirect_stdout(io.StringIO())):
                stage.main()
                self.assertEqual(tuple(native.rt.net._forward_hooks), hooks)
                with patch.object(stage.e63.Capture, "capture", side_effect=AssertionError("Committed replay repeated")):
                    stage.main()

            tables = tuple(f"stage78_{n}.csv" for n in stage.TABLES)
            _, report, _ = stage.completed(root, stage.SCHEMA, stage.STATUS, tables)
            self.assertEqual((report["physical_search_calls"], report["block_calls"], report["source_max_box_error"]), (32, 224, 0.))
            self.assertEqual(report["source_action_disagreements"], 0)
            csv_path = root / "analysis/stage78_timeline.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            base, search, published = ([r for r in rows if r["arm"] == arm] for arm in stage.ARMS)
            self.assertEqual(search[0]["crop_changed"], "False")
            self.assertEqual(search[0]["iou"], base[0]["iou"])
            self.assertEqual(search[1]["crop_changed"], "True")
            self.assertNotEqual(published[0]["published_width_ratio"], base[0]["published_width_ratio"])
            for original, corrected in zip(base, published):
                for key in ("action", "crop_side", "raw_iou", "best_candidate_iou"):
                    self.assertEqual(original[key], corrected[key])
            csv_path.write_bytes(csv_path.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                stage.completed(root, stage.SCHEMA, stage.STATUS, tables)


if __name__ == "__main__":
    unittest.main()
