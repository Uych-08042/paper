"""Verify matched training pairs and the frozen Stage-143 reference."""
import unittest

import numpy as np

import stage144_pairwise_candidate_ranking as stage


def candidates(ious):
    x = np.zeros((4, len(stage.fixed.FEATURES)))
    x[:, 0] = [.6, .9, .8, .7]
    x[:, 1] = [0., 0., 1., 2.]
    return dict(x=x, y=np.array(ious), native_iou=ious[0], pools={"peak": np.array([0, 1, 2]), "dense": np.array([0, 1, 3])})


class PairwiseCandidateRanking(unittest.TestCase):
    def test_coexisting_pairs_are_unique_and_risk_strata_have_correct_sign(self):
        rescue = candidates([.1, .05, .7, .8])
        features, gaps, (all_pairs, recover, protect) = stage.frame_pairs(rescue)
        self.assertEqual((all_pairs.sum(), recover.sum(), protect.sum()), (5, 4, 0))
        self.assertTrue((gaps[recover] > 0).all())
        self.assertTrue((features[recover, 1] > 0).all())
        stable = candidates([.8, .05, .3, .4])
        _, _, (_, recover, protect) = stage.frame_pairs(stable)
        self.assertEqual((recover.sum(), protect.sum()), (0, 1))

    def test_changing_held_labels_does_not_change_fitted_models(self):
        a, b = candidates([.1, .05, .7, .8]), candidates([.8, .05, .3, .4])
        bank = {"peer": stage.source_moments([a, b]), "held": stage.source_moments([a])}
        regression = dict(held_source="held", fit_sources=["peer"], mean=[0.] * 9, scale=[1.] * 9)
        expected = stage.fit_pairwise(bank, "held", regression)
        bank["held"] = stage.source_moments([b])
        np.testing.assert_equal(expected, stage.fit_pairwise(bank, "held", regression))
        self.assertEqual(expected["hard_pairwise"]["mixture"], [.5, .25, .25])
        scores = stage.fixed.score(expected["pairwise"], a["x"])
        self.assertGreater(scores[3], scores[1])

    def test_stage143_audit_checks_candidate_order_and_reproduced_action(self):
        row = dict(sequence="uav0000089_00920_s", frame=1159, dense_cells="2|1|3",
                   dense_quality_cell=1, dense_quality_iou=.4698)
        reference = {k: str(v) for k, v in row.items()}
        stage.audit_reference(row, reference)
        with self.assertRaisesRegex(ValueError, "dense_cells"):
            stage.audit_reference(dict(row, dense_cells="2|3|1"), reference)
        with self.assertRaisesRegex(ValueError, "dense_quality_cell"):
            stage.audit_reference(dict(row, dense_quality_cell=3), reference)


if __name__ == "__main__":
    unittest.main()
