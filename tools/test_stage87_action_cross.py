"""Verify exact-box reuse and the first-action crossing through a CPU tracker."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage87_action_cross as stage
import test_stage75_ppo_failure as fixture


class Crossing(unittest.TestCase):
    def test_same_crossed_box_creates_one_continuation(self):
        bank = np.array([[[0., 0., 1., 1.], [10., 0., 1., 1.]]])
        fixed = dict(candidate_boxes=bank, selected=[0], boxes=bank[:, 0])
        other = bank + [10., 0., 0., 0.]
        midpoint = dict(candidate_boxes=other, selected=[1], boxes=other[:, 1])
        cells = stage.cross((fixed, midpoint))
        self.assertEqual([cell["ref"] for cell in cells.values()], ["ff", "fm", "fm", "mm"])


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Replay(unittest.TestCase):
    def test_bridge_and_one_new_tail_per_case_then_resume(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        fixture.torch.set_num_threads(1)
        s86, c = stage.s86, stage.c
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            original, native, old_observer, _ = TorchTests().fixture(root)
            old_observer.handle.remove()
            old_observer.pre_handle.remove()
            observer = stage.e63.Capture(native)
            self.addCleanup(observer.handle.remove)
            old = dict(original["plan"], actor_design=dict(stage.e63.source_common.DESIGN))
            entry = old["entries"][0]
            ap = stage.e74.e67.actor_plan(old)
            prefix = np.array([entry["initial_box"], np.asarray(entry["initial_box"]) + [24., 0., 0., 0.]])
            frames = np.array([2, 3, 4])
            unit = dict(sequence=entry["sequence"], origin="native", fold=0, policy="ppo", seed=2171)

            def choose(row, history, *_):
                q = 0 if np.array_equal(row["candidate_parents"][0], history[-1]) else 2
                return dict(row, selected=q, boxes=row["candidate_boxes"][q].copy())

            arms = [s86.collect(observer, entry, prefix, frames, mix, ap, choose) for mix in (0., .5)]
            cells = stage.cross(arms)
            self.assertEqual([cell["ref"] for cell in cells.values()], ["ff", "mm", "mf", "mm"])
            cases = [(dict(case=s, stratum=s, unit=unit, anchor=1), entry, frames, prefix, arms, cells) for s in s86.STRATA]
            root86, output = root / "source", root / "stage87"
            (root86 / "analysis").mkdir(parents=True)
            output.mkdir()
            for path in (root86 / "manifest.json", root86 / "analysis/stage86_decision.json"):
                path.write_text("{}\n", encoding="utf-8")
            runtime = dict(tracker=dict(device="fixture"))
            source = dict(original=old, report=dict(fingerprint="fixture", native_head_sha256=digest(native.rt.net.box_head)))
            manifest = dict(fingerprint="plan86", windows=[row[0] for row in cases], design=dict(s86.DESIGN, horizon=3))
            with patch.object(stage, "cases", return_value=iter(cases)):
                plan = stage.prepare(output, root86, manifest, dict(fingerprint="86", runtime=runtime), source)
            self.assertEqual(plan["new_branches"], ["high__mf", "low__mf"])
            c.publish_identical(output / "runtime.json", stage.io.seal(dict(
                schema=stage.SCHEMA, plan_fingerprint=plan["fingerprint"], runtime=runtime)))
            actor = dict(models={}, prior={})
            with (patch.object(stage.e74, "choose", side_effect=choose), patch.object(stage.e74, "learned_pack"),
                  contextlib.redirect_stdout(io.StringIO())):
                stage.run(output, plan, iter(cases), source, actor, observer)
                stage.check_results(output)
                report = c.c66.sealed(output / "analysis/stage87_decision.json", stage.SCHEMA)
                self.assertEqual((report["crossed_cells"], report["new_continuations"], report["search_calls"]), (8, 2, 32))
                with (output / "analysis/stage87_paired.csv").open(newline="") as file:
                    effects = {r["contrast"]: float(r["mean_delta"]) for r in csv.DictReader(file)
                               if r["stratum"] == "high" and r["metric"] == "mean_iou"}
                self.assertAlmostEqual(effects["natural_total"],
                                       effects["geometry_given_fixed_action"] + effects["action_given_midpoint_layout"])
                with patch.object(observer, "capture", side_effect=AssertionError("Cached inference repeated")):
                    stage.run(output, plan, iter(cases), source, actor, observer)


if __name__ == "__main__":
    unittest.main()
