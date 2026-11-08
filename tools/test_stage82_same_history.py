"""Verify identical-state replay, search reuse and the candidate/selection decomposition."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage82_same_history as stage
import test_stage75_ppo_failure as fixture


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class SameHistory(unittest.TestCase):
    def test_real_center_reproduces_both_history_providers(self):
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
                ap = e74.e67.actor_plan(old)
                base = dict(sequence=entry["sequence"], origin="native", fold=0, policy="ppo")
                packs = {seed: e74.learning.initial(torch, seed, dict(center=[0.] * 256, scale=[1.] * 256), old["design"])
                         for seed in (2171, 2172)}
                quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                               critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
                source = dict(original=old, report=dict(native_head_sha256=digest(native.rt.net.box_head)))
                design = dict(stage.DESIGN, bootstrap_samples=30)
                records = []
                with patch.object(c, "fit_pack", return_value=quality):
                    for seed, pack in packs.items():
                        action_unit = dict(base, seed=seed)

                        def choose(row, history):
                            return e74.choose(row, history, action_unit, old, {}, pack)

                        for provider in stage.PROVIDERS:
                            unit = dict(action_unit, arm=provider)
                            collector = stage.s81.collect_mixed if provider == "mixed_ppo" else stage.s80.collect
                            trace = collector(observer, entry, unit, ap, choose, design)
                            states = stage.active_states(entry, trace, provider, ap, design)
                            self.assertTrue(states)
                            self.assertGreater(states[0]["frame"], 1)
                            records.append((unit, entry, states))
                    units = [dict(u, tracked_frames=len(e["frames"])-1, active_frames=len(states),
                                  search_calls=sum(4+int(s["changed"].sum()) for s in states)) for u, e, states in records]
                    plan = stage.io.seal(dict(schema=stage.SCHEMA, units=units, design=design, runtime={}, videos=1,
                        source_stage81_fingerprint="stage81", confirmation_fingerprint="roster", source_sha256={}, code_sha256={}))
                    with (patch.object(stage.s78, "actor_pack", side_effect=lambda s, a, u: (u, packs[u["seed"]])),
                          contextlib.redirect_stdout(io.StringIO())):
                        stage.replay(root, plan, source, {"prior": {}}, records, observer)
                        report = c.c66.sealed(root / "analysis/stage82_decision.json", stage.SCHEMA)
                        self.assertTrue(report["factual_exact_bridge"])
                        self.assertEqual(report["search_calls"], sum(u["search_calls"] for u in units))
                        self.assertEqual(report["scanned_states"], sum(u["tracked_frames"] for u in units))
                        with (root / "analysis/stage82_active_frames.csv").open(newline="", encoding="utf-8") as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertEqual(len(rows), report["active_states"])
                        for row in rows:
                            self.assertAlmostEqual(float(row["selected_iou_gain"]),
                                                   float(row["best_iou_gain"]) + float(row["regret_reduction"]))
                        with patch.object(observer, "capture", side_effect=AssertionError("Committed replay repeated")):
                            stage.replay(root, plan, source, {"prior": {}}, records, observer)
            finally:
                observer.handle.remove()


if __name__ == "__main__":
    unittest.main()
