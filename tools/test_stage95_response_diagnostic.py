"""Exercise dense response diagnosis and immutable replay cache recovery."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import stage95_response_diagnostic as stage


def shared():
    parent = np.array([30., 30., 20., 20.])
    raw = np.full((5, 5), .01, np.float32)
    raw[1, 1], raw[3, 3] = .9, .8
    size = np.full((2, 5, 5), .2, np.float32)
    return dict(candidate_parents=np.tile(parent, (1, 4, 1)),
                candidate_boxes=np.tile([0., 0., 20., 20.], (1, 4, 1)),
                raw=np.tile(raw, (1, 4, 1, 1)), size=np.tile(size, (1, 4, 1, 1, 1)),
                offset=np.zeros((1, 4, 2, 5, 5), np.float32), image_hw=np.array([[200, 200]]),
                layer=np.full((1, 4), 9))


class ResponseDiagnosis(unittest.TestCase):
    def test_shared_probe_replay_checks_actual_geometry_and_reuses_cache(self):
        data = shared()
        old = dict(root="fixture", search_factor=5., search_size=100, window=np.ones((5, 5), np.float32),
                   actor_design=dict(stage.e63.source_common.DESIGN))
        image = np.zeros((200, 200, 3), np.uint8)

        class Observer:
            def __init__(self):
                self.rt = SimpleNamespace(tracker=SimpleNamespace(state=None))
                self.calls = 0

            def initialize(self, *_):
                pass

            def capture(self, current, parent, frame):
                self.calls += 1
                raw, size, offset = (data[k][0, 0] for k in ("raw", "size", "offset"))
                geometry = stage.scene_geometry(parent, np.array([40., 40., 20., 20.]),
                                                current.shape[:2], 5., 100)
                dense = stage.dense_boxes(size, offset, parent, geometry["factor"], 100, current.shape[:2])
                blocks = np.zeros(12, int)
                blocks[:6] = 1
                blocks[8] = 1
                return dict(raw=raw, size=size, offset=offset, boxes=dense[int(raw.argmax())],
                            layer=9, block_calls=blocks, head_calls=1, selector_calls=1, crop_sha256="0"*64)

        observer = Observer()
        initial = np.array([30., 30., 20., 20.])
        entry = dict(initial_box=initial, frames=[{}, {}], ground_truth=[initial, [40., 40., 20., 20.]])
        history = initial[None]
        unit = dict(sequence="test", policy="ppo", seed=2171, fold=0, origin="native")
        point = dict(sequence="test", seed=2171, frame=1, cohort="historical", outcome="missed", roles=["test"])
        original_collect = stage.e63.collect_shared

        def collect(*args):
            return original_collect(*args, image_reader=lambda *_: image)

        trace = collect(observer, entry["frames"], initial, history, 1, "history", stage.e63.c.actor_plan(old))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(stage.e63, "collect_shared", side_effect=collect):
                first = stage.replay(Path(tmp), dict(fingerprint="fixture"), point, unit, "gated", trace, entry, old, observer)
            self.assertEqual(first[0]["category"], "weighted_peaks_have_good")
            self.assertEqual(first[0]["max_public_box_error_px"], 0.)
            self.assertEqual(observer.calls, 8)
            with patch.object(stage.e63, "collect_shared", side_effect=AssertionError("Repeated inference")):
                second = stage.replay(Path(tmp), dict(fingerprint="fixture"), point, unit, "gated", trace, entry, old, observer)
            self.assertEqual(first[0], second[0])

    def test_real_decode_preserves_hidden_good_peak_and_exposes_shape_loss(self):
        data = shared()
        old = dict(search_factor=5., search_size=100, window=np.ones((5, 5), np.float32))
        truth = np.array([40., 40., 20., 20.])
        rows = stage.inspect(data, old, truth, 0)
        self.assertEqual(stage.classify(rows, 0), "weighted_peaks_have_good")
        self.assertTrue(rows[0]["gt_center_inside"])
        self.assertGreater(rows[0]["dense_best_gt_iou"], .99)
        self.assertEqual(rows[0]["weighted_gt_rank_in_top5"], 2)
        data["size"][:] = .8
        data["candidate_boxes"][:] = [0., 0., 50., 50.]
        rows = stage.inspect(data, old, truth, 0)
        self.assertEqual(stage.classify(rows, 0), "covered_dense_lacks_good")

    def test_target_outside_all_crops_and_invalid_labels_are_distinguished(self):
        data = shared()
        old = dict(search_factor=5., search_size=100, window=np.ones((5, 5), np.float32))
        rows = stage.inspect(data, old, np.array([170., 170., 20., 20.]), 0)
        self.assertEqual(stage.classify(rows, 0), "crop_support_limited")
        self.assertEqual(max(r["gt_coverage"] for r in rows), 0.)
        rows = stage.inspect(data, old, np.zeros(4), 0)
        self.assertEqual(stage.classify(rows, 0), "invalid_gt")

    def test_frozen_anchors_use_both_outcomes_and_collapse_duplicate_frames(self):
        root = Path(__file__).resolve().parents[1]
        import csv
        import io
        raw = (root / "docs/experiments/06_sgla_uav_tracking/results/stage94_results_20261110T080350Z_1181619.txt").read_text()
        events = list(csv.DictReader(io.StringIO(raw.split("===== stage94_events.csv =====")[1].strip())))
        points = stage.points_for(events)
        self.assertEqual(len(points), 137)
        self.assertEqual({r["outcome"] for r in points}, {"missed", "retained"})
        self.assertTrue(all(any(p["frame"] == int(e["ppo_recovery_frame"]) and p["sequence"] == e["sequence"]
            and p["seed"] == int(e["seed"]) for p in points) for e in events))

    def test_resume_verifies_caches_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = stage.io.seal(dict(schema=stage.SCHEMA, points=[{}], source_sha256={}, code_sha256={}))
            stage.c.publish_identical(root / "manifest.json", plan)
            caches = {}
            for name in ("ppo", "gated"):
                path = root / "collection" / f"{name}.npz"
                stage.c.save(path, dict(value=np.ones(1)), dict(history=name))
                caches[path.relative_to(root).as_posix()] = stage.c.file_hash(path)
            tables = {}
            for name in stage.TABLES:
                path = root / "analysis" / f"stage95_{name}.csv"
                path.parent.mkdir(exist_ok=True)
                stage.s86.publish_table(path, [dict(value=1)])
                tables[path.name] = stage.c.file_hash(path)
            report = stage.io.seal(dict(schema=stage.SCHEMA, status=stage.STATUS, plan_fingerprint=plan["fingerprint"],
                states=2, search_calls=8, block_calls=56, cache_sha256=caches, table_sha256=tables))
            stage.c.publish_identical(root / "analysis/stage95_decision.json", report)
            with patch.object(stage.e63, "Capture", side_effect=AssertionError("Model loaded during cache check")):
                stage.check_results(root)
            (root / "collection/gated.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
