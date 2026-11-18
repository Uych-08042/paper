"""Check the three frozen scores, alternate invariance and diagnostic report."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage123_frozen_retained_scores as stage
from test_stage122_alternate_diagnostic import sample


class FrozenRetainedScores(unittest.TestCase):
    def test_same_bank_can_recover_alternate_or_lose_a_good_output(self):
        boxes = np.array([[0., 0., 5., 5.], [20., 0., 5., 5.], [40., 0., 5., 5.], [60., 0., 5., 5.]])
        a, b, hybrid = stage.compare_scores([.40, .35, .20, .95], [.80, .60, .25, .50], boxes)
        self.assertEqual((a,b), (0,3))
        np.testing.assert_array_equal(hybrid, [.40,.35,.20,.50])
        a, b, _ = stage.compare_scores([.70, .35, .20, .05], [.25, .60, .25, .65], boxes)
        self.assertEqual((a,b), (3,0))
        a, b, _ = stage.compare_scores([.50, .35, .20, .95], [.25, .60, .25, .50], boxes)
        self.assertEqual((a,b), (1,0))

    def test_cache_replay_isolates_the_point_and_exports_a_sealed_report(self):
        ref, alt, entry, scorer = sample()
        identity = dict(cohort="expanded", sequence="test", seed="2312", fold="0")
        _, events, scores, _ = stage.prior.diagnose(ref, alt, entry, scorer, 5, identity)
        point = {k:str(v) for k,v in events[0].items()}
        points = [(point, scores)]
        entry["sequence"] = "test"
        spec = dict(cohort="expanded", unit=dict(sequence="test"))
        jobs = [dict(spec=spec, seed=2312, frames=4, outer_fold=0, arm=arm)
                for arm in (stage.prior.CONTROL, stage.s121.TREATMENT)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            result = root / "result"
            result.mkdir()
            source_plan = dict(jobs=jobs, fingerprint="fixture")
            for job, data in zip(jobs, (ref,alt)):
                path = source_root/"evaluation"/(stage.base.job_key(job)+".npz")
                stage.c.save(path, data, dict(job=job, plan_fingerprint=source_plan["fingerprint"]))
            tables = stage.run(source_root, source_plan, {("test",2312):points}, lambda _:entry, 5)
            row = tables["decisions"][0]
            self.assertEqual((row["full_slot"],row["frozen_slot"]), (1,0))
            self.assertEqual((row["full_iou"],row["frozen_iou"]), (1.,0.))
            self.assertTrue(row["severe_loss"])
            with patch.object(stage, "DESIGN", dict(stage.DESIGN, points=1, added_good_points=0)):
                manifest = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN,
                                             source_stage122_fingerprint=stage.SOURCE122, inputs={}))
                stage.c.publish_identical(result/"manifest.json", manifest)
                stage.publish(result, manifest, tables)
                stage.check_results(result)
                (result/"analysis/stage123_summary.csv").write_text("corrupted")
                with self.assertRaises(ValueError):
                    stage.check_results(result)


if __name__ == "__main__":
    unittest.main()
