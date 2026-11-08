"""Verify the consensus rule and the full-video entry with a small real CENTER head."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage80_guard_experiment as stage
import test_stage75_ppo_failure as fixture


class Consensus(unittest.TestCase):
    def test_guard_requires_near_tie_and_distinct_peer_support(self):
        boxes = np.array([[100., 100., 20., 20.], [12., 10., 20., 20.],
                          [11., 10., 20., 20.], [10., 10., 20., 20.]])
        logits = np.array([1., .7, .99, .5])
        self.assertEqual(stage.guard_action(boxes, np.ones(4, bool), logits, 0, stage.DESIGN), (2, True))
        logits[2] = .8
        self.assertEqual(stage.guard_action(boxes, np.ones(4, bool), logits, 0, stage.DESIGN), (0, False))
        logits[2] = .99
        self.assertEqual(stage.guard_action(boxes, np.array([1, 0, 1, 0], bool), logits, 0, stage.DESIGN), (0, False))


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class FullRun(unittest.TestCase):
    def test_native_and_paired_full_tracks_use_the_declared_calls(self):
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
            try:
                entry = dict(original["plan"]["entries"][0], initial_box=[40., 40., 8., 20.])
                old = dict(original["plan"], entries=[entry], design=dict(c.DESIGN),
                           actor_design=dict(stage.e63.source_common.DESIGN), controller_design=dict(c.c66.DESIGN))
                ppo = dict(sequence=entry["sequence"], origin="native", fold=0, seed=2172, policy="ppo")
                packs = {seed: e74.learning.initial(torch, seed, dict(center=[0.] * 256, scale=[1.] * 256), old["design"])
                         for seed in (2171, 2172)}
                quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                               critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
                units = [dict(ppo, policy="q0", seed=0, arm="native_sgla")]
                units += [dict(ppo, seed=seed, arm=arm) for seed in packs for arm in stage.ARMS[1:]]
                source = dict(original=old, report=dict(native_head_sha256=digest(native.rt.net.box_head)))
                budget = 4 * (1 + 6 * 4)
                plan = stage.io.seal(dict(schema=stage.SCHEMA, units=units, design=dict(stage.DESIGN, bootstrap_samples=30),
                    expected_search_calls=budget, runtime={}, source_stage79_fingerprint="stage79",
                    confirmation_fingerprint="roster", source_sha256={}, code_sha256={}))
                run = root / "new"
                run.mkdir()
                with (patch.object(c, "fit_pack", return_value=quality),
                      patch.object(e74.source_experiment, "control_pack", return_value={"source": quality})):
                    reference = {seed: e74.run_actor(observer, entry, dict(ppo, seed=seed), old, {}, pack)
                                 for seed, pack in packs.items()}
                    reference[0] = e74.run_actor(observer, entry, dict(ppo, policy="q0", seed=0), old, {})

                    def source_trace(_, window):
                        return entry, reference[window["unit"]["seed"]]

                    with (patch.object(stage.s78.previous, "read_source_trace", source_trace),
                          patch.object(stage.s78, "actor_pack", side_effect=lambda s, a, u: (u, packs.get(u["seed"]))),
                          contextlib.redirect_stdout(io.StringIO())):
                        stage.run(run, plan, source, {"prior": {}}, observer)
                        report = c.c66.sealed(run / "analysis/stage80_decision.json", stage.SCHEMA)
                        self.assertEqual((report["trajectories"], report["search_calls"], report["block_calls"]), (7, budget, 7*budget))
                        self.assertEqual((report["max_source_box_error"], report["source_action_changes"]), (0., 0))
                        self.assertEqual(report["confirmation_inference_calls"], 0)
                        with patch.object(observer, "capture", side_effect=AssertionError("Committed replay repeated")):
                            stage.run(run, plan, source, {"prior": {}}, observer)
            finally:
                observer.handle.remove()


if __name__ == "__main__":
    unittest.main()
