"""Check module attribution, sealed trace replay and cached resume."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stage77_module_diagnostic as stage
import test_stage75_ppo_failure as fixture


class Classification(unittest.TestCase):
    def test_public_grid_and_crop_are_separate_readouts(self):
        low = dict(iou=.1, best_grid_iou=.15, gt_coverage=.4, gt_center_inside=False)
        good = dict(low, iou=.6, best_grid_iou=.7)
        grid = dict(low, best_grid_iou=.65)
        covered = dict(low, gt_coverage=.95, gt_center_inside=True)
        for rows, category in (([good, low], "selected_good"), ([low, good], "outer_choice_missed"),
                               ([low, grid], "decode_grid_has_good"), ([low, covered], "covered_grid_lacks_good"),
                               ([low, low], "crop_support_limited")):
            self.assertEqual(stage.classify(rows, 0, stage.DESIGN), category)


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class ModuleReplay(unittest.TestCase):
    def test_real_center_replay_sealed_receipts_and_resume(self):
        from test_stage61_search import TorchTests
        from stage48_stability_runtime import digest

        torch = fixture.torch
        torch.set_num_threads(1)
        quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                       critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
        s76, c = stage.s76, stage.c
        with tempfile.TemporaryDirectory(prefix="stage77-", dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            original, _, observer, _ = TorchTests().fixture(folder)
            try:
                entry = original["plan"]["entries"][0]
                runtime = original["plan"]["runtime_identity"]
                original_dir, source_dir, root = (folder / name for name in ("original", "source", "new"))
                root.mkdir()
                base = {k: v for k, v in original["plan"].items() if k != "fingerprint"}
                old = stage.io.seal(dict(base, schema=c.SCHEMA, design=dict(c.DESIGN),
                    actor_design=dict(stage.e63.source_common.DESIGN), controller_design=dict(c.c66.DESIGN)))
                d74 = stage.io.seal(dict(schema=c.SCHEMA, plan_fingerprint=old["fingerprint"],
                                         training_fingerprint="fixture", runtime=runtime))
                c.publish_identical(original_dir / "manifest.json", old)
                c.publish_identical(original_dir / "analysis/stage74_decision.json", d74)
                unit = dict(sequence=entry["sequence"], origin="native", fold=0, seed=2171, policy="horizon_ce")
                with patch.object(c, "fit_pack", return_value=quality):
                    norm = dict(center=[0.] * 256, scale=[1.] * 256)
                    pack = s76.e.learning.initial(torch, 2171, norm, old["design"])
                    trace = s76.e.run_actor(observer, entry, dict(unit, policy="myopic"), old, {}, pack)
                calls = int(trace["head_calls"].sum())
                p76 = stage.io.seal(dict(schema=s76.SCHEMA, source_input=str(original_dir), runtime=runtime,
                    units=[unit], eval_search_calls=calls, source_report_fingerprint=d74["fingerprint"],
                    source_training_fingerprint="fixture"))
                c.publish_identical(source_dir / "manifest.json", p76)
                path = source_dir / "evaluation" / (c.key(unit) + ".npz")
                c.save(path, trace, dict(plan_fingerprint=p76["fingerprint"], unit=unit, controller="fit_fixture"))
                evidence = {}
                s76.binding(evidence, path)
                report = stage.io.seal(dict(schema=s76.SCHEMA, status="COMPLETE_MATCHED_HORIZON_SUPERVISION_DEVELOPMENT",
                    plan_fingerprint=p76["fingerprint"], runtime=runtime, trajectories=1, evaluation_search_calls=calls,
                    source_report_fingerprint=d74["fingerprint"], source_training_fingerprint="fixture",
                    model_fingerprints={c.job_key(unit): "fit_fixture"}, native_head_sha256=digest(observer.rt.net.box_head),
                    evaluation_evidence_fingerprint=c.fingerprint(evidence)))
                c.publish_identical(source_dir / "analysis/stage76_decision.json", report)
                window = dict(group="fixture", unit=unit, frames=[2, 3], expected_first_failure=None)
                with patch.object(stage, "windows", return_value=[window]), contextlib.redirect_stdout(io.StringIO()):
                    source = stage.load_source(source_dir)
                    plan = stage.make_plan(source, runtime)
                    result = stage.run(root, plan, source, observer)
                    self.assertEqual((result["states"], result["search_calls"], result["block_calls"]), (2, 8, 56))
                    self.assertEqual(result["maximum_public_box_error_px"], 0.)
                    with patch.object(observer, "capture", side_effect=AssertionError("Committed replay repeated")):
                        self.assertEqual(stage.run(root, plan, source, observer), result)
                receipt = path.with_suffix(".receipt.json")
                receipt.write_bytes(receipt.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "evaluation evidence changed"):
                    stage.load_source(source_dir)
            finally:
                observer.handle.remove()
                observer.pre_handle.remove()


if __name__ == "__main__":
    unittest.main()
