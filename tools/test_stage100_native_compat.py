"""Regress real four-slot q0 records, immutable plans and repaired continuation."""
import contextlib
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import stage100_native_compat as compat
import stage100_online_scale as stage
import test_stage100_online_scale as previous
import test_stage97_trusted_scale as fixture


class NativeCompatibility(previous.OnlineScale):
    def source_traces(self):
        with (patch.object(stage.s93.s85, "checked_image", side_effect=self.reader),
              patch.object(stage.s86, "checked_image", side_effect=self.reader)):
            one = stage.s93.s85.collect(self.observer, self.entry, "native_sgla", self.ap, None, {})
            four = stage.s86.collect(self.observer, self.entry, np.asarray([self.entry["initial_box"]]),
                                    np.arange(1, 13), 0., self.ap, fixture.choose)
        return one, four

    def test_four_slot_q0_projection_preserves_every_input_and_rejects_bad_unused_calls(self):
        one, four = self.source_traces()
        before = {k: v.copy() for k, v in four.items()}
        with self.assertRaisesRegex(ValueError, "block_calls"):
            stage.e63.source_common.prior.audit_calls(four, (12, 1))
        view = compat.native_view(four, self.entry, self.ap)
        stage.e63.source_common.prior.audit_calls(view, (12, 1))
        for key in four:
            np.testing.assert_array_equal(four[key], before[key], err_msg=key)
        np.testing.assert_array_equal(view["boxes"], four["boxes"])
        np.testing.assert_array_equal(view["candidate_boxes"], four["candidate_boxes"][:, :1])
        self.assertEqual(int(view["head_calls"].sum()), 12)
        self.assertEqual(int(view["native_source_head_calls"]), 48)
        self.assertEqual(int(view["native_source_slots"]), 4)
        for value in (one, view):
            normalized = compat.native_view(value, self.entry, self.ap)
            np.testing.assert_array_equal(normalized["boxes"], value["boxes"])
        bad = {k: v.copy() for k, v in four.items()}
        bad["head_calls"][0, 3] = 0
        with self.assertRaises(ValueError):
            compat.native_view(bad, self.entry, self.ap)
        bad = {k: v.copy() for k, v in four.items()}
        bad["selected"][0] = 1
        with self.assertRaisesRegex(ValueError, "slot zero"):
            compat.native_view(bad, self.entry, self.ap)
        bad = {k: v.copy() for k, v in four.items()}
        bad["block_calls"] = bad["block_calls"][:, :3]
        with self.assertRaisesRegex(ValueError, "Unsupported native call shape"):
            compat.native_view(bad, self.entry, self.ap)

    def test_frozen_plan_continues_both_hardware_modes_with_mixed_native_formats(self):
        one, four = self.source_traces()
        gated = self.collect("gated")
        references = dict(ppo=four, gated=gated)
        unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
        specs = [dict(unit=unit, cohort="historical"),
                 dict(unit=dict(unit, seed=2172), cohort="historical"),
                 dict(unit=dict(unit, sequence="other"), cohort="expanded")]
        source = dict(original=self.old, report=dict(native_head_sha256="native"))
        frozen_paths = [Path(stage.__file__).resolve(), Path(stage.__file__).with_name("run_stage100_online_scale.sh").resolve()]
        code_hashes = {str(p): stage.c.file_hash(p) for p in frozen_paths}
        for reuse in (True, False):
            with self.subTest(reuse=reuse), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = stage.io.seal(dict(schema=stage.SCHEMA, design=dict(stage.DESIGN, bootstrap_samples=16),
                    runtime={}, runtime_differences=[], units=specs, trajectories=14,
                    new_trajectories=6 if reuse else 14, new_search_calls=288 if reuse else 600,
                    logical_search_calls=600, reuse_sources=reuse,
                    box_atol=self.old["actor_design"]["box_atol"], source_sha256={}, code_sha256=code_hashes))
                stage.c.publish_identical(root / "manifest.json", plan)
                manifest_bytes = (root / "manifest.json").read_bytes()
                prior_native = None
                if reuse:
                    native_unit = dict(unit, seed=0, policy="q0")
                    prior_native = root / "evaluation" / f"{stage.c.key(native_unit)}__native.npz"
                    stage.c.save(prior_native, four, dict(plan_fingerprint=plan["fingerprint"], unit=native_unit, arm="native"))
                    prior_native_bytes = prior_native.read_bytes()
                original_collect = stage.collect
                actual = lambda *args: original_collect(*args, image_reader=self.reader)
                args = (root, root, root, plan, {}, dict(units=specs), {}, source,
                        dict(models={}, prior={}), self.observer)
                with (patch.object(compat, "ORIGINAL_LOAD_NATIVE", side_effect=lambda *a: (
                          dict(a[-1]["unit"], seed=0, policy="q0"), four if a[-1]["cohort"] == "historical" else one)),
                      patch.object(stage.s95, "load_unit", side_effect=lambda *a: (
                          dict(self.entry, sequence=a[-1]["unit"]["sequence"]), references)),
                      patch.object(stage.e74, "learned_pack"),
                      patch.object(stage.e74, "choose", side_effect=lambda row,h,*_: fixture.choose(row,h)),
                      patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                      patch.object(stage.s93.s85, "checked_image", side_effect=self.reader),
                      patch.object(stage.s86, "checked_image", side_effect=self.reader),
                      patch.object(stage, "collect", side_effect=actual), contextlib.redirect_stdout(io.StringIO())):
                    compat.run(*args)
                    compat.check_results(root)
                    cache_hashes = {str(p): stage.c.file_hash(p) for p in (root / "evaluation").iterdir()}
                    with patch.object(self.observer, "capture", side_effect=AssertionError("Repeated inference")):
                        compat.run(*args)
                    self.assertTrue(all(stage.c.file_hash(p) == h for p, h in cache_hashes.items()))
                self.assertEqual((root / "manifest.json").read_bytes(), manifest_bytes)
                if prior_native is not None:
                    self.assertEqual(prior_native.read_bytes(), prior_native_bytes)
                self.assertTrue(all(stage.c.file_hash(p) == h for p, h in code_hashes.items()))
                report = stage.c.c66.sealed(root / "analysis/stage100_decision.json", stage.SCHEMA)
                self.assertEqual(report["new_search_calls"], 288 if reuse else 600)
                by_name = {r["sequence"]: r for r in report["native_source_views"]}
                self.assertEqual(by_name["test"]["source_head_calls"], 48)
                self.assertEqual(by_name["other"]["source_head_calls"], 12)
                self.assertEqual(by_name["test"]["q0_view_head_calls"], 12)
                receipt = stage.c.c66.sealed(root / compat.RECEIPT, compat.SCHEMA)
                self.assertFalse(receipt["manifest_edited"])
                self.assertEqual(receipt["adapter_inference_calls"], 0)
                cache = next((root / "evaluation").glob("*.npz"))
                cache.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    compat.check_results(root)


if __name__ == "__main__":
    import unittest
    unittest.main()
