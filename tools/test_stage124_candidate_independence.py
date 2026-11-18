"""Check feature invariance and matching from training to the sealed reports."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

import stage124_candidate_independence as stage


class CandidateIndependence(unittest.TestCase):
    def test_companion_changes_leave_retained_independent_scores_fixed(self):
        boxes = np.array([[0, 0, 5, 5], [5, 0, 4, 4], [10, 0, 8, 6], [18, 0, 4, 4]], float)
        altered = boxes.copy()
        altered[-1] = [0, 0, 5, 5]
        base = np.zeros((4, 24))
        base[:, 1] = [.5, .8, .2, .9]
        history = boxes[:1].copy()
        a = stage.common.features(base, boxes, history)
        b = stage.common.features(base, altered, history)
        self.assertTrue(np.any(a[:3] != b[:3]))
        state = stage.common.initial(2312, "shared", stage.common.DESIGN)
        model = dict(state=state, norm=dict(center=np.zeros(64), scale=np.ones(64)),
                     architecture="shared", design=stage.common.DESIGN)
        np.testing.assert_array_equal(stage.common.scores(model, stage.inputs(a, "independent"))[:3],
                                      stage.common.scores(model, stage.inputs(b, "independent"))[:3])
        np.testing.assert_array_equal(stage.inputs(a, "independent")[:, [27,28,59,60]], a[:, [27,28,59,60]])
        self.assertTrue(np.any(a[:, list(stage.MASK_COLUMNS)]))
        self.assertFalse(np.any(stage.inputs(a, "independent")[:, list(stage.MASK_COLUMNS)]))

    def test_four_folds_train_paired_models_and_refuse_leakage(self):
        groups = [f"uav{i:07d}" for i in range(28)]
        provider = {fold:{"held_out_groups":groups[7*fold:7*(fold+1)]} for fold in range(4)}
        rng = np.random.default_rng(124)
        rows = {}
        boxes = np.array([[0, 0, 5, 5], [5, 0, 4, 4], [10, 0, 8, 6], [18, 0, 4, 4]], float)
        for i, group in enumerate(groups):
            fold = i // 7
            row = dict(group=group, sequence=group+"_00000_s", cohort="expanded", seed=2312,
                       source_arm=stage.SOURCES[0], frame=1, fold=fold, boxes=boxes.copy(),
                       x=rng.normal(size=(4,64)), y=np.array([.6,.3,.2,.1]),
                       recorded_iou=.6, recorded_action=0, fit_eligible=True)
            row.update({p:p in ("uniform","uniform_q0_good") for p in stage.POOLS})
            rows[group] = [row]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = {p:sum(r[p] for rr in rows.values() for r in rr) for p in stage.POOLS}
            plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, source_stage122_fingerprint=stage.previous.SOURCE122,
                                       states=len(groups), population_counts=counts, inputs={}))
            stage.c.publish_identical(root/"manifest.json", plan)
            stage.run(root, plan, rows, provider)
            d = stage.c.c66.sealed(root/"analysis/stage124_decision.json", stage.SCHEMA)
            self.assertEqual(d["primary"]["videos"], 28)
            self.assertEqual(d["row_counts"]["decisions"], 56)
            scope = stage.read_table(root/"analysis/stage124_model_scope.csv")
            for record in scope:
                peers = record["fit_groups"].split("|")
                self.assertEqual(set(peers), set(provider[int(record["provider_fold"])]["held_out_groups"]) - {record["group"]})
                self.assertEqual(int(record["examples"]), 6)
                self.assertEqual(int(record["parameters"]), 2113)
                self.assertEqual(int(record["optimizer_steps"]), stage.common.DESIGN["epochs"])
            stage.check_results(root)
            with self.assertRaisesRegex(ValueError, "six peer groups"):
                stage.training_split(1, groups[0], provider)


if __name__ == "__main__":
    unittest.main()
