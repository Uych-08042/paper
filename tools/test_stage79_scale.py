"""Exercise group exclusion and same-history PPO intervention continuations."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage79_confirmation as confirmation
import stage79_scale_cross as stage
import test_stage75_ppo_failure as fixture


class Confirmation(unittest.TestCase):
    def test_manifest_exclusions_and_one_clip_per_group(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            def entry(index, suffix="00000"):
                return dict(sequence=f"uav{index:07d}_{suffix}_s", source_group_proxy=f"uav{index:07d}", frames=10)

            seqs = [entry(i) for i in range(1000, 1055)] + [entry(1043, "00100")]
            schemas = ("stage27_uav_failure_diagnostic_v1", "stage31_continuous_identity_pilot_v1",
                       "stage76_four_candidate_horizon_supervision_v1")
            bodies = (dict(sequences=seqs[:24]), dict(sequences=seqs[24:38]), dict(units=seqs[:12]))
            argv = ["confirmation", "--dataset-root", str(root), "--output", str(root / "roster.json")]
            for name, schema, body in zip(("stage27", "stage31", "stage76"), schemas, bodies):
                path = root / (name + ".json")
                confirmation.publish_identical(path, confirmation.seal(dict(body, schema=schema, root=str(root))))
                argv.extend(["--" + name, str(path)])
            extra = root / "extra.txt"
            extra.write_text("uav0001042\n", encoding="utf-8")
            argv.extend(["--exclude-file", str(extra)])
            with (patch("sys.argv", argv), patch.object(confirmation, "inventory", return_value=seqs),
                  contextlib.redirect_stdout(io.StringIO())):
                confirmation.main()
                confirmation.main()
                record = confirmation.sealed(root / "roster.json", confirmation.SCHEMA)
                self.assertEqual(len(record["sequences"]), 16)
                self.assertNotIn(entry(1042), record["sequences"])
                self.assertIn(entry(1043), record["sequences"])
                self.assertNotIn(entry(1043, "00100"), record["sequences"])
                self.assertFalse(record["labels_read"])
                extra.write_text("uav0001044\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Existing artifact differs"):
                    confirmation.main()


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Replay(unittest.TestCase):
    def test_real_center_natural_tails_match_stage78_and_cache(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        torch = fixture.torch
        torch.set_num_threads(1)
        c, e74, s78 = stage.c, stage.e74, stage.s78
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
                unit = dict(sequence=entry["sequence"], origin="native", fold=0, seed=2172, policy="ppo")
                pack = e74.learning.initial(torch, 2172, dict(center=[0.] * 256, scale=[1.] * 256), old["design"])
                quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                               critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
                prefix = np.asarray([entry["initial_box"]])
                case = dict(case="fixture", role="primary", unit=unit, first=1, last=4, focus=2)
                parent = dict(fingerprint="stage78", design=s78.DESIGN)
                with patch.object(c, "fit_pack", return_value=quality):
                    sources = {}
                    for scale in stage.SCALES:
                        data = s78.collect(observer, entry["frames"], entry["initial_box"], prefix, [1, 2, 3, 4], unit,
                                           old, {}, pack, s78.DESIGN, scale == "search_scale")
                        sources[scale] = data
                        c.save(root / "collection" / "fixture" / (scale + ".npz"), data,
                               dict(plan_fingerprint="stage78", case=case, arm=scale))
                    source_actions = [int(sources[scale]["selected"][1]) for scale in stage.SCALES]
                    actions = sorted({0, 2} | set(source_actions))
                    case = dict(case, anchor=2, actions=actions, source_actions=source_actions)
                    source = dict(original=old, report=dict(native_head_sha256=digest(native.rt.net.box_head)))
                    plan = stage.io.seal(dict(schema=stage.SCHEMA, design=s78.DESIGN, cases=[case], runtime={},
                        expected_search_calls=8 + 16 * len(actions), source_stage78_fingerprint="stage78",
                        confirmation_fingerprint="roster", source_sha256={}, code_sha256={}))
                    run = root / "new"
                    with (patch.object(s78.previous, "read_source_trace", return_value=(entry, sources["native"])),
                          patch.object(s78, "actor_pack", return_value=(unit, pack)), contextlib.redirect_stdout(io.StringIO())):
                        stage.replay(run, plan, source, {"prior": {}}, parent, root, observer)
                        report = c.c66.sealed(run / "analysis/stage79_decision.json", stage.SCHEMA)
                        self.assertGreaterEqual(report["natural_exact_branches"], 2)
                        self.assertEqual(report["physical_search_calls"], 8 + 16 * len(actions))
                        with patch.object(observer, "capture", side_effect=AssertionError("Replay cache was bypassed")):
                            stage.replay(run, plan, source, {"prior": {}}, parent, root, observer)
                    self.assertNotEqual(stage.alias(np.array([[1, 1, 3, 3]] * 4), 2), 2)
            finally:
                observer.handle.remove()


if __name__ == "__main__":
    unittest.main()
