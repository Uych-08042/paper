"""Check past-only scale updates, prefix continuation and complete pilot publication."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import stage97_trusted_scale as stage
import test_stage95_response_diagnostic as fixture


class PatchGrid:
    def __init__(self):
        self.hook = None

    @contextlib.contextmanager
    def register_forward_hook(self, hook):
        self.hook = hook
        try:
            yield
        finally:
            self.hook = None

    def emit(self, grid):
        if self.hook is not None:
            self.hook(self, (), grid)


class Observer:
    def __init__(self):
        self.embed = PatchGrid()
        self.rt = SimpleNamespace(lens_z=4, lens_x=25,
            net=SimpleNamespace(backbone=SimpleNamespace(patch_embed=self.embed)),
            tracker=SimpleNamespace(state=None))
        self.calls = 0

    def initialize(self, *_):
        pass

    def capture(self, image, parent, frame):
        self.calls += 1
        self.embed.emit(np.tile([[[1., 0.]]], (1, 4, 1)))
        self.embed.emit(np.tile([[[1., 0.]]], (1, 25, 1)))
        data = fixture.shared()
        raw, size, offset = (data[k][0, 0] for k in ("raw", "size", "offset"))
        raw[2, 2] = .95
        offset[:] = .5
        if frame % 2 == 0:
            size[:] = .3
        _, _, factor = stage.crop_geometry(parent, image.shape[:2], 5., 100)
        dense = stage.s95.dense_boxes(size, offset, parent, factor, 100, image.shape[:2])
        blocks = np.zeros(12, int)
        blocks[:6], blocks[8] = 1, 1
        return dict(raw=raw, size=size, offset=offset, boxes=dense[int(raw.argmax())], layer=9,
                    block_calls=blocks, head_calls=1, selector_calls=1, crop_sha256="0"*64)


def choose(row, history):
    return dict(row, selected=0, history_latest=history[-1], boxes=row["candidate_boxes"][0].copy())


class TrustedScale(unittest.TestCase):
    def test_scale_uses_past_qualified_sizes_and_expires(self):
        history = [np.array([20., 20., 80., 80.]), np.array([22., 22., 100., 100.])]
        ap = dict(design=dict(stage.e63.source_common.DESIGN))
        memory = [(1, np.array([20., 30.])), (4, np.array([20., 30.]))]
        ordinary, _ = stage.parents_at(history, memory, 7, "control", ap, stage.DESIGN)
        revised, refs = stage.parents_at(history, memory, 7, "trusted_scale", ap, stage.DESIGN)
        np.testing.assert_array_equal(revised[[0, 2]], ordinary[[0, 2]])
        np.testing.assert_allclose(revised[[1, 3], 2:], [[20, 30], [20, 30]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(revised[:, :2]+revised[:, 2:]/2, ordinary[:, :2]+ordinary[:, 2:]/2)
        np.testing.assert_array_equal(refs, [1, 4, -1, -1, -1])
        expired, refs = stage.parents_at(history, memory, 30, "trusted_scale", ap, stage.DESIGN)
        np.testing.assert_array_equal(expired, ordinary)
        self.assertTrue((refs < 0).all())
        with self.assertRaises(ValueError):
            stage.parents_at(history, memory, 4, "trusted_scale", ap, stage.DESIGN)
        self.assertFalse(stage.scale_write(dict(memory_write=True, selected=1, ppo_selected=0),
                                           np.array([[1., 0.], [np.nan, np.nan]])))

    def test_matched_prefix_future_labels_and_full_pipeline_resume(self):
        old = dict(root="fixture", search_factor=5., search_size=100, window=np.ones((5, 5), np.float32),
                   actor_design=dict(stage.e63.source_common.DESIGN))
        initial = np.array([80., 80., 20., 20.])
        entry = dict(initial_box=initial, frames=[{}]*9, ground_truth=np.tile(initial, (9, 1)))
        empty = dict(boxes=np.empty((0, 4)), anchor=np.array([1., 0.]))
        reader = lambda *_: np.zeros((200, 200, 3), np.uint8)
        observer = Observer()
        original_collect = stage.collect

        def collect(*args):
            return original_collect(*args, image_reader=reader)

        with patch.object(stage.s93.s91, "cpu_array", side_effect=lambda x: x.copy()):
            trace = collect(observer, entry, empty, 1, 8, choose, "control", old, stage.DESIGN)
            tail = collect(observer, entry, trace, 3, 8, choose, "control", old, stage.DESIGN)
            for key in stage.FIELDS:
                np.testing.assert_array_equal(tail[key], trace[key][2:], err_msg=key)
            for arm in stage.ARMS:
                data = collect(observer, entry, trace, 3, 8, choose, arm, old, stage.DESIGN)
                stage.audit(data, trace, entry, 3, 8, choose, arm, old, stage.DESIGN)
                self.assertTrue(data["scale_written"].any())
                if arm == "trusted_scale":
                    self.assertTrue(data["scale_active"].any())
                changed = dict(entry, ground_truth=np.full((9, 4), 999.))
                repeat = collect(observer, changed, trace, 3, 8, choose, arm, old, stage.DESIGN)
                for key in stage.FIELDS:
                    np.testing.assert_array_equal(data[key], repeat[key])
            self.assertIsNone(observer.embed.hook)
            event = dict(sequence="test", seed="2171", outcome="retained", first_audit_frame="3",
                         last_audit_frame="8", ppo_recovery_frame="5", matched_good_frame="")
            unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
            source = dict(original=old, report=dict(native_head_sha256="native"))
            for reuse in (True, False):
                with self.subTest(reuse_control=reuse), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, events=[event],
                        reuse_control=reuse, source_sha256={}, code_sha256={}, runtime={}, runtime_differences=[]))
                    stage.c.publish_identical(root / "manifest.json", plan)
                    plan89 = dict(units=[dict(unit=unit)])
                    with (patch.object(stage.s95, "load_unit", return_value=(entry, dict(gated=trace))),
                          patch.object(stage.e74, "learned_pack"), patch.object(stage.e74, "choose", side_effect=lambda r,h,*_: choose(r,h)),
                          patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                          patch.object(stage, "collect", side_effect=collect), contextlib.redirect_stdout(io.StringIO())):
                        stage.run(root, plan, root, root, {}, plan89, {}, source, dict(models={}, prior={}), observer)
                        stage.check_results(root)
                    report = stage.c.c66.sealed(root / "analysis/stage97_decision.json", stage.SCHEMA)
                    self.assertEqual(report["new_search_calls"], 24 if reuse else 48)
                    self.assertEqual(report["maximum_control_source_box_error_px"], 0.)
                    with (patch.object(stage.s95, "load_unit", return_value=(entry, dict(gated=trace))),
                          patch.object(stage.e74, "learned_pack"), patch.object(stage.e74, "choose", side_effect=lambda r,h,*_: choose(r,h)),
                          patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                          patch.object(stage, "collect", side_effect=AssertionError("Repeated inference")),
                          contextlib.redirect_stdout(io.StringIO())):
                        stage.run(root, plan, root, root, {}, plan89, {}, source, dict(models={}, prior={}), observer)
                    cache = next((root / "branches").glob("*.npz"))
                    cache.write_bytes(b"tampered")
                    with self.assertRaises(ValueError):
                        stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
