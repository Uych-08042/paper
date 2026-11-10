"""Test held-out exclusion, deterministic sampling and expanded CPU continuation."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage89_conflict_sampling as stage
import test_stage75_ppo_failure as fixture


class Sampling(unittest.TestCase):
    def test_roster_selection_excludes_confirmation_and_is_order_independent(self):
        rows = [dict(sequence=f"uav{i:07d}_00000_s", source_group_proxy=f"uav{i:07d}", frames=100)
                for i in range(5)]
        groups = {r["source_group_proxy"] for r in rows}
        design = dict(stage.DESIGN, extra_groups=2)
        args = ({rows[0]["source_group_proxy"]}, {rows[1]["source_group_proxy"]}, groups, design)
        selected = stage.extra_sequences(rows, *args)
        self.assertEqual(selected, stage.extra_sequences(rows[::-1], *args))
        self.assertTrue(set(selected) <= {r["sequence"] for r in rows[2:]})

    def test_two_strata_use_only_prediction_and_do_not_overlap(self):
        boxes = np.tile(np.array([0., 0., 20., 20.]), (80, 4, 1))
        boxes[:32, 1, 0] = 12.
        boxes[32:64, 1, 0] = 4.
        boxes[64:, 1, 0] = 12.
        features = np.zeros((80, 4, len(stage.s76.branches.prior.FEATURES)))
        peak = stage.s76.branches.prior.FEATURES.index("hann_peak")
        features[:, 0, peak] = .6
        features[:, 1, peak] = .58
        features[64:, 1, peak] = .4
        trace = dict(boxes=boxes[:, 0], candidate_boxes=boxes, candidate_parents=boxes.copy(),
                     history_latest=boxes[:, 0], selected=np.zeros(80, dtype=int), features=features)
        design = dict(stage.DESIGN, horizon=4, windows_per_stratum=2)
        picked, eligible = stage.select_windows(trace, "fixture", design)
        self.assertEqual(eligible, {"conflict": 32, "ordinary": 32})
        self.assertEqual([r["stratum"] for r in picked], ["conflict", "conflict", "ordinary", "ordinary"])
        self.assertEqual(picked, stage.select_windows(trace, "fixture", design)[0])
        frames = [i for r in picked for i in range(r["first"], r["first"] + 4)]
        self.assertEqual(len(frames), len(set(frames)))


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Replay(unittest.TestCase):
    def test_expanded_trace_and_branches_resume(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        fixture.torch.set_num_threads(1)
        s86, c = stage.s86, stage.c
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            original, native, old_observer, _ = TorchTests().fixture(root)
            old_observer.handle.remove()
            old_observer.pre_handle.remove()
            observer = stage.e63.Capture(native)
            self.addCleanup(observer.handle.remove)
            sequence = "uav0000126_07915_s"
            dataset, output = root / "dataset", root / "stage89"
            images = dataset / "sequences" / sequence
            images.mkdir(parents=True)
            annotations = dataset / "annotations"
            annotations.mkdir()
            fixture_entry = original["plan"]["entries"][0]
            for f in fixture_entry["frames"]:
                (images / f["file"]).write_bytes((Path(original["plan"]["root"]) / f["file"]).read_bytes())
            (annotations / (sequence + ".txt")).write_text(
                "".join(",".join(map(str, b)) + "\n" for b in fixture_entry["ground_truth"]), encoding="utf-8")
            old = dict(original["plan"], root=str(dataset), actor_design=dict(stage.e63.source_common.DESIGN))
            unit = dict(sequence=sequence, origin="native", fold=0, policy="ppo", seed=2171)
            source = dict(original=old, report=dict(native_head_sha256=digest(native.rt.net.box_head)))
            actor = dict(models={}, prior={})
            plan = stage.io.seal(dict(schema=stage.SCHEMA, dataset=str(dataset), units=[dict(unit=unit, cohort="expanded")],
                runtime={}, design=dict(stage.DESIGN, horizon=3), source_sha256={}, code_sha256={}))
            c.publish_identical(output / "manifest.json", plan)

            def choose(row, history, *_):
                return dict(row, selected=0, boxes=row["candidate_boxes"][0].copy())

            with (patch.object(stage.e74, "choose", side_effect=choose), patch.object(stage.e74, "learned_pack"),
                  contextlib.redirect_stdout(io.StringIO())):
                stage.execute(output, plan, source, actor, observer)
                stage.check_results(output)
                report = c.c66.sealed(output / "analysis/stage89_decision.json", stage.SCHEMA)
                self.assertEqual(report["baseline_search_calls"], 4 * (len(fixture_entry["frames"]) - 1))
                self.assertGreater(report["windows"], 0)
                with patch.object(observer, "capture", side_effect=AssertionError("Cached inference repeated")):
                    stage.execute(output, plan, source, actor, observer)


if __name__ == "__main__":
    unittest.main()
