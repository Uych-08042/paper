"""Exercise causal online scale episodes and full-video publication on CPU."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import stage100_online_scale as stage
import test_stage97_trusted_scale as fixture


class OnlineScale(unittest.TestCase):
    def setUp(self):
        self.old = dict(root="fixture", search_factor=5., search_size=100,
            window=np.ones((5, 5), np.float32), actor_design=dict(stage.e63.source_common.DESIGN))
        self.ap = stage.e63.c.actor_plan(self.old)
        initial = np.array([80., 80., 20., 20.])
        self.entry = dict(sequence="test", initial_box=initial, frames=[dict(height=200, width=200)]*13,
                          ground_truth=np.tile(initial, (13, 1)))
        self.reader = lambda *_: np.zeros((200, 200, 3), np.uint8)
        self.observer = fixture.Observer()
        self.observer.rt.tracker.params = SimpleNamespace(template_factor=2., template_size=100)
        self.cpu = patch.object(stage.s93.s91, "cpu_array", side_effect=lambda x: x.copy())
        self.cpu.start()
        self.addCleanup(self.cpu.stop)

    def collect(self, mode, entry=None, choose=fixture.choose):
        return stage.collect(self.observer, self.entry if entry is None else entry, self.ap, choose,
                             mode, stage.DESIGN, image_reader=self.reader)

    def test_episode_state_rearms_only_after_request_ends(self):
        previous, pulses = False, []
        for request in (False, True, True, True, False, True, False, True):
            enabled, episode = stage.activation("pulse", True, request, previous)
            self.assertEqual(enabled, episode)
            pulses.append(enabled)
            previous = request
        self.assertEqual(pulses, [False, True, False, False, False, True, False, True])
        self.assertEqual(stage.activation("continuous", True, False, True), (True, False))
        self.assertEqual(stage.activation("gated", True, True, False), (False, True))
        with self.assertRaises(ValueError):
            stage.activation("pulse", False, True, False)

    def test_causal_size_references_unchanged_centers_and_pixel_request(self):
        history = [np.array([80., 80., 30., 30.]), np.array([83., 83., 40., 40.])]
        sizes = [(1, np.array([10., 20.])), (2, np.array([10., 20.]))]
        ordinary, _, _, _ = stage.parents_for(history, sizes, 3, "gated", False, (200, 200), self.ap, stage.DESIGN)
        parents, refs, request, first = stage.parents_for(history, sizes, 3, "pulse", False, (200, 200), self.ap, stage.DESIGN)
        self.assertTrue(request and first)
        np.testing.assert_array_equal(refs, [1, 2, -1, -1, -1])
        np.testing.assert_array_equal(parents[[0, 2]], ordinary[[0, 2]])
        np.testing.assert_allclose(stage.s86.centers(parents), stage.s86.centers(ordinary), rtol=0, atol=1e-12)
        restored, refs, request, first = stage.parents_for(history, sizes, 3, "pulse", True, (200, 200), self.ap, stage.DESIGN)
        np.testing.assert_array_equal(restored, ordinary)
        self.assertTrue(request)
        self.assertFalse(first)
        self.assertTrue((refs == -1).all())
        with self.assertRaises(ValueError):
            stage.parents_for(history, sizes, 2, "pulse", False, (200, 200), self.ap, stage.DESIGN)

    def test_equal_recovery_loss_counts_do_not_hide_a_newly_lost_event(self):
        ident = dict(sequence="test", seed=2171, reference_arm="ppo", matched_good_frame=None)
        baseline = dict(ident, arm="gated", ppo_failure_frame=10, ppo_recovery_frame=20)
        replacement = dict(ident, arm="pulse", ppo_failure_frame=30, ppo_recovery_frame=40)
        self.assertEqual(stage.newly_lost_ppo_recoveries([baseline, replacement]), {("test", 2171, 30, 40)})
        self.assertFalse(stage.newly_lost_ppo_recoveries([baseline, dict(baseline, arm="pulse")]))

    def test_real_geometry_memory_rollout_labels_isolated_and_control_reconstructed(self):
        baseline = self.collect("gated")
        original = {k: baseline[k] for k in stage.s86.FIELDS+stage.s93.FIELDS} | dict(anchor=baseline["anchor"])
        decorated = stage.decorate_control(original, self.entry, self.ap, stage.DESIGN)
        for key in stage.FIELDS:
            np.testing.assert_array_equal(decorated[key], baseline[key], err_msg=key)
        for mode in ("gated", "continuous", "pulse"):
            data = self.collect(mode)
            stage.audit(data, self.entry, self.ap, fixture.choose, mode, stage.DESIGN)
            stage.shared_prefix(data, baseline)
            changed = dict(self.entry, ground_truth=np.full((13, 4), 999.))
            repeated = self.collect(mode, changed)
            for key in stage.FIELDS:
                np.testing.assert_array_equal(data[key], repeated[key], err_msg=key)
            np.testing.assert_array_equal(data["scale_refs"][:2], -np.ones((2, 5), int))
            for frame, refs in enumerate(data["scale_refs"], 1):
                self.assertTrue(np.all(refs[refs >= 0] < frame))
            if mode == "pulse":
                np.testing.assert_array_equal(data["scale_active"], data["scale_episode_start"])
                self.assertTrue(data["scale_crop_changed"].any())
                data["scale_refs"][3, 0] = 4
                with self.assertRaises(AssertionError):
                    stage.audit(data, self.entry, self.ap, fixture.choose, mode, stage.DESIGN)
        self.assertIsNone(self.observer.embed.hook)

    def test_full_pipeline_same_card_new_card_resume_and_cache_tampering(self):
        unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
        with (patch.object(stage.s93.s85, "checked_image", side_effect=self.reader),
              patch.object(stage.s86, "checked_image", side_effect=self.reader)):
            native = stage.s93.s85.collect(self.observer, self.entry, "native_sgla", self.ap, None, {})
            ppo = stage.s86.collect(self.observer, self.entry, np.asarray([self.entry["initial_box"]]),
                                   np.arange(1, 13), 0., self.ap, fixture.choose)
        gated = self.collect("gated")
        references = dict(ppo=ppo, gated=gated)
        source = dict(original=self.old, report=dict(native_head_sha256="native"))
        specs = [dict(unit=unit, cohort="historical"),
                 dict(unit=dict(unit, seed=2172), cohort="historical"),
                 dict(unit=dict(unit, sequence="other"), cohort="expanded")]
        for reuse in (True, False):
            with self.subTest(reuse=reuse), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                design = dict(stage.DESIGN, bootstrap_samples=16)
                plan = stage.io.seal(dict(schema=stage.SCHEMA, design=design, runtime={}, runtime_differences=[],
                    trajectories=14, new_trajectories=6 if reuse else 14,
                    new_search_calls=288 if reuse else 600, logical_search_calls=600,
                    reuse_sources=reuse, box_atol=self.old["actor_design"]["box_atol"], source_sha256={}, code_sha256={}))
                stage.c.publish_identical(root / "manifest.json", plan)
                original_collect = stage.collect
                actual = lambda *args: original_collect(*args, image_reader=self.reader)
                args = (root, root, root, plan, {}, dict(units=specs),
                        {}, source, dict(models={}, prior={}), self.observer)
                with (patch.object(stage, "load_native", side_effect=lambda *a: (dict(a[-1]["unit"], seed=0, policy="q0"), native)),
                      patch.object(stage.s95, "load_unit", side_effect=lambda *a: (dict(self.entry, sequence=a[-1]["unit"]["sequence"]), references)),
                      patch.object(stage.e74, "learned_pack"),
                      patch.object(stage.e74, "choose", side_effect=lambda row,h,*_: fixture.choose(row,h)),
                      patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                      patch.object(stage.s93.s85, "checked_image", side_effect=self.reader),
                      patch.object(stage.s86, "checked_image", side_effect=self.reader),
                      patch.object(stage, "collect", side_effect=actual), contextlib.redirect_stdout(io.StringIO())):
                    stage.run(*args)
                    stage.check_results(root)
                    with patch.object(self.observer, "capture", side_effect=AssertionError("Repeated inference")):
                        stage.run(*args)
                report = stage.c.c66.sealed(root / "analysis/stage100_decision.json", stage.SCHEMA)
                self.assertEqual(report["new_search_calls"], 288 if reuse else 600)
                self.assertEqual(report["maximum_source_control_box_error_px"], 0.)
                cache = next((root / "evaluation").glob("*.npz"))
                cache.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
