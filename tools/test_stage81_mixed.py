"""Verify mixed search slot assignment and the full-track cache path."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage81_mixed_scale as stage
import test_stage75_ppo_failure as fixture


class Slots(unittest.TestCase):
    def test_preserves_raw_slots_and_reverts_when_scale_is_unchanged(self):
        ap = dict(design=dict(history=5), search_size=256, search_factor=4.)
        raw = [np.array([45., 40., 10., 20.]) for _ in range(4)] + [np.array([30., 40., 40., 20.])]
        limited = raw[:-1] + [np.array([40., 40., 20., 20.])]
        native = stage.s76.branches.prior.history_parents(raw, ap["design"])
        parents, mask = stage.mixed_parents(raw, limited, (160, 180), ap)
        np.testing.assert_array_equal(parents[[0, 2]], native[[0, 2]])
        np.testing.assert_array_equal(mask, [False, True, False, True])
        np.testing.assert_array_equal(parents[[1, 3], :2] + parents[[1, 3], 2:] / 2,
                                      native[[0, 2], :2] + native[[0, 2], 2:] / 2)
        parents, mask = stage.mixed_parents(raw, raw, (160, 180), ap)
        np.testing.assert_array_equal(parents, native)
        self.assertFalse(mask.any())
        existing_median = raw[:-1] + [np.array([45., 40., 10., 20.])]
        parents, mask = stage.mixed_parents(raw, existing_median, (160, 180), ap)
        np.testing.assert_array_equal(parents, native)
        self.assertFalse(mask.any())


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class FullRun(unittest.TestCase):
    def test_complete_real_center_tracks_and_resume(self):
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
                budget = (len(entry["frames"]) - 1) * (1 + 6 * 4)
                plan = stage.io.seal(dict(schema=stage.SCHEMA, units=units, design=dict(stage.DESIGN, bootstrap_samples=30),
                    expected_search_calls=budget, runtime={}, source_stage80_fingerprint="stage80",
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
                        report = c.c66.sealed(run / "analysis/stage81_decision.json", stage.SCHEMA)
                        self.assertEqual((report["trajectories"], report["search_calls"], report["block_calls"]), (7, budget, 7*budget))
                        self.assertEqual((report["max_source_box_error"], report["source_action_changes"]), (0., 0))
                        self.assertEqual(report["confirmation_inference_calls"], 0)
                        unit = units[2]
                        origin = {k: v for k, v in unit.items() if k != "arm"}
                        data = c.read_cache(run / "evaluation" / (c.key(origin) + "__mixed_ppo.npz"),
                                            dict(plan_fingerprint=plan["fingerprint"], unit=unit))
                        self.assertFalse(data["mixed"][0].any())
                        self.assertTrue(data["mixed"][1:].any())
                        self.assertFalse(data["mixed"][:, [0, 2]].any())
                        with patch.object(observer, "capture", side_effect=AssertionError("Committed replay repeated")):
                            stage.run(run, plan, source, {"prior": {}}, observer)
            finally:
                observer.handle.remove()


if __name__ == "__main__":
    unittest.main()
