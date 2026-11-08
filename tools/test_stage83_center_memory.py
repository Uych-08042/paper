"""Verify center retention timing, crop identity and the full-track budget."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage83_center_memory as stage
import test_stage75_ppo_failure as fixture


class Memory(unittest.TestCase):
    def test_conflict_and_one_effective_extra_frame(self):
        ap = dict(design=dict(history=5), search_factor=4., search_size=256)
        hw = (160, 180)
        older = np.array([10., 10., 20., 30.])
        latest = np.array([20., 20., 20., 30.])
        other = np.array([90., 30., 20., 30.])
        boxes = np.array([latest, other, [21., 20., 20., 30.], [80., 30., 20., 30.]])
        mask = np.ones(4, bool)
        logits = np.array([1., .99, .7, .3])
        self.assertTrue(stage.conflict(boxes, mask, logits, 0, stage.DESIGN))
        self.assertFalse(stage.conflict(boxes, np.array([1, 0, 1, 1], bool), logits, 0, stage.DESIGN))
        self.assertFalse(stage.conflict(boxes, mask, np.array([1., .7, .99, .3]), 0, stage.DESIGN))

        history = [older, latest]
        anchor, remaining = stage.advance(history, other, None, 0, True, ap, stage.DESIGN)
        np.testing.assert_array_equal(anchor, stage.center(latest))
        first, changed = stage.search_parents(history, anchor, remaining, hw, ap)
        self.assertFalse(changed)
        np.testing.assert_array_equal(first, stage.s76.branches.prior.history_parents(history, ap["design"]))
        anchor, remaining = stage.advance(history, other, anchor, remaining, True, ap, stage.DESIGN)
        second, changed = stage.search_parents(history, anchor, remaining, hw, ap)
        ordinary = stage.s76.branches.prior.history_parents(history, ap["design"])
        self.assertTrue(changed)
        np.testing.assert_array_equal(second[:2], ordinary[:2])
        np.testing.assert_array_equal(second[:, 2:], ordinary[:, 2:])
        np.testing.assert_array_equal(second[2:, :2]+second[2:, 2:]/2, np.array([anchor, anchor]))
        anchor, remaining = stage.advance(history, other, anchor, remaining, True, ap, stage.DESIGN)
        final, changed = stage.search_parents(history, anchor, remaining, hw, ap)
        self.assertEqual(remaining, 0)
        self.assertFalse(changed)
        np.testing.assert_array_equal(final, stage.s76.branches.prior.history_parents(history, ap["design"]))


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class FullRun(unittest.TestCase):
    def test_real_center_full_tracks_and_cache(self):
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
                budget = (len(entry["frames"])-1) * 17
                plan = stage.io.seal(dict(schema=stage.SCHEMA, units=units, design=dict(stage.DESIGN, bootstrap_samples=30),
                    expected_search_calls=budget, runtime={}, source_stage82_fingerprint="stage82",
                    confirmation_fingerprint="roster", source_sha256={}, code_sha256={}))
                run = root / "new"
                run.mkdir()
                with (patch.object(c, "fit_pack", return_value=quality),
                      patch.object(e74.source_experiment, "control_pack", return_value={"source": quality})):
                    reference = {seed: e74.run_actor(observer, entry, dict(base, seed=seed), old, {}, pack)
                                 for seed, pack in packs.items()}
                    reference[0] = e74.run_actor(observer, entry, dict(base, policy="q0", seed=0), old, {})

                    def source_trace(_, window):
                        return entry, reference[window["unit"]["seed"]]

                    with (patch.object(stage.s78.previous, "read_source_trace", source_trace),
                          patch.object(stage.s78, "actor_pack", side_effect=lambda s, a, u: (u, packs.get(u["seed"]))),
                          patch.object(stage, "conflict", return_value=True), contextlib.redirect_stdout(io.StringIO())):
                        stage.run(run, plan, source, {"prior": {}}, observer)
                        report = c.c66.sealed(run / "analysis/stage83_decision.json", stage.SCHEMA)
                        self.assertEqual((report["trajectories"], report["search_calls"], report["block_calls"]), (5, budget, 7*budget))
                        self.assertEqual((report["max_source_box_error"], report["source_action_changes"]), (0., 0))
                        self.assertEqual(report["confirmation_inference_calls"], 0)
                        self.assertEqual(report["triggers"], 4)
                        unit = units[2]
                        path = run / "evaluation" / (c.key(stage.s82.source_unit(unit)) + "__center_memory_ppo.npz")
                        data = c.read_cache(path, dict(plan_fingerprint=plan["fingerprint"], unit=unit))
                        np.testing.assert_array_equal(data["remaining"], [0, 2, 1, 0])
                        np.testing.assert_array_equal(data["crop"][:2], reference[2171]["crop"][:2])
                        with patch.object(observer, "capture", side_effect=AssertionError("Committed replay repeated")):
                            stage.run(run, plan, source, {"prior": {}}, observer)
            finally:
                observer.handle.remove()


if __name__ == "__main__":
    unittest.main()
