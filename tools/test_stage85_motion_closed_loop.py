"""Exercise the frozen source and the four-search closed loop on CPU."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage85_motion_closed_loop as stage
import test_stage75_ppo_failure as fixture


class SourceGate(unittest.TestCase):
    def test_accepts_only_the_passing_unchanged_generator(self):
        proposal = stage.proposal
        units = [dict(sequence=f"v{i}", policy="ppo", seed=s) for i in range(12) for s in (2171, 2172)]
        source = dict(plan=dict(units=units), report=dict(fingerprint="fixture"), evidence={})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            analysis = root / "analysis"
            analysis.mkdir()
            generator = Path(proposal.__file__).resolve()
            plan = stage.io.seal(dict(schema=proposal.SCHEMA, states=27050, units=units, design=proposal.DESIGN,
                source_stage76_fingerprint="fixture", source_sha256={}, code_sha256={str(generator): stage.c.file_hash(generator)}))
            stage.c.publish_identical(root / "manifest.json", plan)
            hashes = {}
            for name in proposal.TABLES:
                path = analysis / f"stage84_{name}.csv"
                path.write_text("fixture\n", encoding="utf-8")
                hashes[path.name] = stage.c.file_hash(path)
            body = dict(schema=proposal.SCHEMA, status=proposal.STATUS, plan_fingerprint=plan["fingerprint"],
                        matched_states=27050, logical_search_calls_per_arm=108200, histories=24, videos=12,
                        source_stage76_fingerprint="fixture", confirmation_inference_calls=0, new_full_trajectories=0,
                        training_updates=0, candidate_supply_pass=True, fixed_source_max_box_error=0.,
                        fixed_source_layer_changes=0, gates=dict(best_iou_positive_ci=True, best_iou_video_wins=True,
                        good_candidate_fraction_increased=True), table_sha256=hashes)
            stage.c.publish_identical(analysis / "stage84_decision.json", stage.io.seal(body))
            stage.stage84_binding(root, source)
            original_hash = stage.c.file_hash
            with patch.object(stage.c, "file_hash", side_effect=lambda p: "changed" if Path(p) == generator else original_hash(p)):
                with self.assertRaisesRegex(ValueError, "unchanged generator"):
                    stage.stage84_binding(root, source)
            path = analysis / "stage84_decision.json"
            path.unlink()
            stage.c.publish_identical(path, stage.io.seal(dict(body, candidate_supply_pass=False)))
            with self.assertRaisesRegex(ValueError, "passing"):
                stage.stage84_binding(root, source)


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class ClosedLoop(unittest.TestCase):
    def test_real_center_same_actor_full_tracks_and_resume(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        torch = fixture.torch
        torch.set_num_threads(1)
        c, e74 = stage.c, stage.e74
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            original, native, old_observer, _ = TorchTests().fixture(root)
            old_observer.handle.remove()
            old_observer.pre_handle.remove()
            observer = stage.e63.Capture(native)
            self.addCleanup(observer.handle.remove)
            entry = original["plan"]["entries"][0]
            old = dict(original["plan"], design=dict(c.DESIGN),
                       actor_design=dict(stage.e63.source_common.DESIGN), controller_design=dict(c.c66.DESIGN))
            base = dict(sequence=entry["sequence"], origin="native", fold=0, policy="ppo")
            packs = {seed: e74.learning.initial(torch, seed, dict(center=[0.] * 256, scale=[1.] * 256), old["design"])
                     for seed in (2171, 2172)}
            quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                           critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
            units = [dict(base, policy="q0", seed=0, arm="native_sgla")]
            units += [dict(base, seed=seed, arm=arm) for seed in packs for arm in stage.ARMS[1:]]
            source = dict(original=old, report=dict(native_head_sha256=digest(native.rt.net.box_head)))
            n = len(entry["frames"])-1
            budget = n*17
            plan = stage.io.seal(dict(schema=stage.SCHEMA, units=units, design=dict(stage.DESIGN, bootstrap_samples=32),
                generator_design=stage.proposal.DESIGN, expected_search_calls=budget, runtime={},
                source_stage84_fingerprint="fixture", confirmation_fingerprint="roster", source_sha256={}, code_sha256={}))
            output = root / "stage85"
            output.mkdir()
            c.publish_identical(output / "manifest.json", plan)
            with (patch.object(c, "fit_pack", return_value=quality),
                  patch.object(e74.source_experiment, "control_pack", return_value={"source": quality})):
                refs = {seed: e74.run_actor(observer, entry, dict(base, seed=seed), old, {}, pack)
                        for seed, pack in packs.items()}
                refs[0] = e74.run_actor(observer, entry, dict(base, policy="q0", seed=0), old, {})
                with (patch.object(stage.proposal.source_io, "read_source_trace",
                                   side_effect=lambda s, w: (entry, refs[w["unit"]["seed"]])),
                      patch.object(e74, "learned_pack", side_effect=lambda models, unit, prior: packs[unit["seed"]]),
                      contextlib.redirect_stdout(io.StringIO())):
                    stage.run(output, plan, source, {"models": {}, "prior": {}}, observer)
                    stage.check_results(output)
                    report = c.c66.sealed(output / "analysis/stage85_decision.json", stage.SCHEMA)
                    self.assertEqual((report["trajectories"], report["search_calls"], report["block_calls"]), (5, budget, 7*budget))
                    self.assertEqual((report["max_source_box_error"], report["source_action_changes"]), (0., 0))
                    self.assertGreater(report["changed_crop_frames"], 0)
                    self.assertEqual(report["confirmation_inference_calls"], 0)
                    with patch.object(observer, "capture", side_effect=AssertionError("Committed replay repeated")):
                        stage.run(output, plan, source, {"models": {}, "prior": {}}, observer)


if __name__ == "__main__":
    unittest.main()
