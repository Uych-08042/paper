"""Test crossed geometry, native decoding, budgets and resumable source reuse."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage96_geometry_cross as stage
import test_stage95_response_diagnostic as fixture


class GeometryCross(unittest.TestCase):
    def test_slot_roles_and_diagonals_are_preserved(self):
        ppo = np.array([[1., 2., 20., 30.], [3., 4., 22., 32.],
                        [5., 6., 20., 30.], [7., 8., 22., 32.]])
        gated = ppo + [11., 13., 6., 8.]
        for cell in stage.CELLS:
            parents = stage.crossed_parents(ppo, gated, cell)
            centers = ppo if cell[0] == "P" else gated
            sizes = ppo if cell[1] == "P" else gated
            np.testing.assert_array_equal(parents[:, 2:], sizes[:, 2:])
            np.testing.assert_array_equal(parents[:, :2]+parents[:, 2:]/2, centers[:, :2]+centers[:, 2:]/2)
        np.testing.assert_array_equal(stage.crossed_parents(ppo, gated, "PP"), ppo)
        np.testing.assert_array_equal(stage.crossed_parents(ppo, gated, "GG"), gated)
        with self.assertRaises(ValueError):
            stage.crossed_parents(ppo, gated * np.nan, "PP")

    def test_factor_effects_reconstruct_both_paths(self):
        # Additive contributions of 0.2 (center), 0.3 (size), plus 0.4 interaction.
        values = dict(GG=0., PG=.2, GP=.3, PP=.9)
        rows = [dict(sequence="test", seed=1, frame=1, cohort="fixture", outcome="missed", roles="recovery",
                     gt_valid=True, cell=cell, **{key: value for key in stage.METRICS}) for cell, value in values.items()]
        factor, = stage.factorial(rows)
        self.assertAlmostEqual(factor["best_public_iou_interaction"], .4)
        self.assertAlmostEqual(factor["best_public_iou_center_average"]+factor["best_public_iou_size_average"], .9)
        self.assertAlmostEqual(factor["best_public_iou_center_at_gated_size"]+
            factor["best_public_iou_size_at_gated_center"]+factor["best_public_iou_interaction"], .9)
        with self.assertRaises(ValueError):
            stage.factorial(rows[:-1])

    def test_full_collection_with_and_without_source_reuse_and_resume(self):
        prototype = fixture.shared()
        old = dict(root="fixture", search_factor=5., search_size=100, window=np.ones((5, 5), np.float32),
                   actor_design=dict(stage.e63.source_common.DESIGN))
        image = np.zeros((200, 200, 3), np.uint8)
        unit = dict(sequence="test", seed=2171, policy="ppo", origin="native", fold=0)
        point = dict(sequence="test", seed=2171, frame=1, cohort="historical", outcome="missed", roles=["test"])
        entry = dict(initial_box=[30., 30., 20., 20.], frames=[{}, {}],
                     ground_truth=[[30., 30., 20., 20.], [40., 40., 20., 20.]])
        original_collect = stage.collect

        class Observer:
            def __init__(self):
                self.calls = 0

            def initialize(self, *_):
                pass

            def capture(self, current, parent, frame):
                self.calls += 1
                raw, size, offset = (prototype[k][0, 0] for k in ("raw", "size", "offset"))
                _, _, factor = stage.crop_geometry(parent, current.shape[:2], 5., 100)
                dense = stage.s95.dense_boxes(size, offset, parent, factor, 100, current.shape[:2])
                blocks = np.zeros(12, int)
                blocks[:6], blocks[8] = 1, 1
                return dict(raw=raw, size=size, offset=offset, boxes=dense[int(raw.argmax())], layer=9,
                            block_calls=blocks, head_calls=1, selector_calls=1, crop_sha256="0"*64)

        def collect(*args):
            return original_collect(*args, image_reader=lambda *_: image)

        for reuse in (True, False):
            with self.subTest(reuse_diagonals=reuse), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "new"
                root95 = Path(directory) / "source"
                observer = Observer()
                for history, parent in (("ppo", [30., 30., 20., 20.]), ("gated", [55., 65., 30., 40.])):
                    data = collect(observer, entry, np.tile(parent, (4, 1)), 1, old)
                    path = root95 / "collection" / f"{stage.c.key(unit)}__{history}__f000001.npz"
                    stage.c.save(path, data, dict(plan_fingerprint="fixture95", unit=unit, history=history, frame=1))
                (root95 / "analysis").mkdir(exist_ok=True)
                stage.s86.publish_table(root95 / "analysis" / "stage95_states.csv", [
                    dict(sequence="test", seed=2171, frame=1, history=h, source_selected=0) for h in ("ppo", "gated")
                ])
                plan = stage.io.seal(dict(schema=stage.SCHEMA, reuse_diagonals=reuse, points=[point],
                    units=[dict(unit=unit)], source_sha256={}, code_sha256={}, runtime={}, runtime_differences=[]))
                stage.c.publish_identical(root / "manifest.json", plan)
                plan95 = dict(fingerprint="fixture95", events=[dict(sequence="test", seed="2171", ppo_recovery_frame="1")])
                source = dict(report=dict(native_head_sha256="native"))
                with (patch.object(stage.c, "entry_for", return_value=entry),
                      patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                      patch.object(stage, "collect", side_effect=collect), contextlib.redirect_stdout(io.StringIO())):
                    stage.run(root, root95, plan, plan95, source, old, observer)
                report = stage.c.c66.sealed(root / "analysis/stage96_decision.json", stage.SCHEMA)
                expected = 8 if reuse else 16
                self.assertEqual((report["states"], report["logical_search_calls"], report["new_search_calls"]), (4, 16, expected))
                self.assertEqual(observer.calls, 8+expected)
                with (patch.object(stage.c, "entry_for", return_value=entry),
                      patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                      patch.object(stage, "collect", side_effect=AssertionError("Repeated inference")),
                      contextlib.redirect_stdout(io.StringIO())):
                    stage.run(root, root95, plan, plan95, source, old, observer)
                cache = next((root / "collection").glob("*.npz"))
                cache.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
