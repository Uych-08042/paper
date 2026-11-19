"""Check frozen case selection, one-time cached write and native suffix accounting."""
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import stage138_cross_layer_continuation as stage


class CrossLayerContinuation(unittest.TestCase):
    def test_real_archive_case_inventory(self):
        directory = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results"

        def table(file, section):
            text = (directory / file).read_text(encoding="utf-8")
            part = text.split(f"===== {section} =====\n", 1)[1].split("\n===== ", 1)[0]
            return list(csv.DictReader(io.StringIO(part)))

        rows137 = table("stage137_results_20261119T123230Z_433433.txt", "stage137_cases.csv")
        rows134 = table("stage134_results_20261119T100654Z_391381.txt", "stage134_windows.csv")
        cases = [dict(r, frame=int(r["frame"]), successors=int(r["successors"])) for r in rows137]
        selected = stage.freeze_cases({"cases": cases}, rows137, rows134)
        self.assertEqual([(r["sequence"][6:10], r["frame"]) for r in selected],
                         [("0080", 645), ("0085", 1541), ("0089", 1159), ("0147", 1113), ("0237", 242), ("0349", 408)])
        self.assertEqual(sum(r["successors"] for r in selected) * len(stage.ARMS), 960)

    def test_cached_write_suffix_and_resume(self):
        wrong = np.array([80., 80., 16., 16.])
        right = np.array([8., 8., 16., 16.])
        case = dict(case_id="fixture", sequence="fixture", kind="onset", frame=1, successors=3)
        entry = dict(sequence="fixture", initial_box=right, frames=list(range(5)), ground_truth=np.tile(right, (5, 1)))
        reference = dict(boxes=np.stack([np.tile(b, (4, 1)) for b in (wrong, wrong, wrong, right)]),
                         frame_id=np.arange(1, 5), layer=np.full((4, 4), 7))
        layers = dict(boxes=np.stack((right, wrong, wrong, wrong, wrong, wrong)))
        scored = dict(native_layer=9, best_other_layer=7, best_other_top1_iou=1., native_top1_iou=0., peak_choice_layer=7)
        ap = dict(root=None, design={"box_atol": 1e-6})
        plan = stage.io.seal(dict(schema=stage.SCHEMA, cases=[case], budget=15, runtime={}, runtime_differences=[]))

        def observe(_observer, _image, parent, _frame, _ap):
            return dict(boxes=parent.copy(), parent=parent, crop=np.zeros(4), crop_sha256="fixture", layer=7,
                        selector_calls=1, head_calls=1, block_calls=[1] * 7 + [0] * 5)

        with TemporaryDirectory() as directory:
            paths = Path(directory)
            root, r134, r137 = (paths / k for k in ("run", "r134", "r137"))
            for folder in (root, r134 / "branches", r137 / "observations"):
                folder.mkdir(parents=True)
            stage.c.publish_identical(root / "manifest.json", plan)
            stage.c.save(r134 / "branches/fixture.npz", reference, dict(plan_fingerprint="r134", case_id="fixture"))
            stage.c.save(r137 / "observations/fixture.npz", layers, dict(plan_fingerprint="r137", case_id="fixture"))
            observer, forward = Mock(), Mock(side_effect=observe)
            collect = stage.collect

            def run_collect(*args):
                return collect(*args, image_reader=lambda _root, frame: frame, observe=forward)

            with (patch.object(stage.probe, "score_case", return_value=(scored, [])),
                  patch.object(stage, "collect", side_effect=run_collect)):
                for _ in range(2):
                    stage.run(root, {"fingerprint": "r134"}, {"fingerprint": "r137"}, r134, r137, plan, observer, ap,
                              lambda _: entry)
            self.assertEqual(forward.call_count, 15)
            observer.initialize.assert_called_once()
            stage.check_results(root)
            with (root / "stage138_windows.csv").open(newline="") as handle:
                rows = {r["arm"]: r for r in csv.DictReader(handle)}
            self.assertEqual(rows[stage.ARMS[3]]["first_tail_good_run"], "2")
            self.assertEqual(rows[stage.ARMS[3]]["tail_delta_vs_peak_pool"], "1.0")
            self.assertEqual(rows[stage.ARMS[3]]["tail_delta_vs_dense"], "0.0")


if __name__ == "__main__":
    unittest.main()
