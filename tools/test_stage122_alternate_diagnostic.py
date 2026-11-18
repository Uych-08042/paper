"""Exercise retained-candidate reranking with the actual frozen scoring code."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

import stage122_alternate_diagnostic as stage


def sample():
    common = stage.base.common
    state = dict(w1=np.zeros((64, 1)), b1=np.zeros(1), w2=np.ones((1, 1)), b2=np.zeros(1))
    state["w1"][[0, 31], 0] = 1
    scorer = dict(architecture="shared", state=state, norm=dict(center=np.zeros(64), scale=np.ones(64)),
                  design=common.DESIGN)
    boxes = np.array([[0., 0., 4., 4.], [12., 0., 4., 4.], [24., 0., 4., 4.], [48., 0., 4., 4.]])
    n = 4
    ref = {name: np.zeros(n) for name in stage.base.FIELDS}
    ref.update(frame_id=np.arange(1, n+1), candidate_boxes=np.tile(boxes, (n,1,1)),
               candidate_parents=np.tile(boxes, (n,1,1)), crop=np.tile(boxes, (n,1,1)),
               features=np.zeros((n,4,24)), unique_search_calls=np.full(n, 4),
               scores=np.zeros((n,4)), selected=np.zeros(n, int), boxes=np.zeros((n,4)))
    for name in ("crop_sha256", "layer", "raw", "size", "offset"):
        ref[name] = np.zeros((n,4))
    ref["features"][:,:,0] = [.52, .05, 0., -2.]
    ref["features"][:,:,1] = [.5, .8, .2, .9]
    alt = {k:v.copy() for k,v in ref.items()}
    alt["alternate_used"] = np.array([False, True, True, False])
    alt["features"][1:3,3,1] = .65
    for key in ("crop", "candidate_parents", "candidate_boxes"):
        alt[key][1:3,3,0] += 9.
    initial = boxes[0]
    for data in (ref, alt):
        for i in range(n):
            h = stage.history(data, initial, i, 5)
            x = common.features(data["features"][i], data["candidate_boxes"][i], h)
            data["scores"][i] = common.scores(scorer, x)
            data["selected"][i] = common.physical_choice(data["scores"][i], data["candidate_boxes"][i])
            data["boxes"][i] = data["candidate_boxes"][i, data["selected"][i]]
    return ref, alt, dict(initial_box=initial, ground_truth=np.tile(boxes[1], (n+1,1))), scorer


class AlternateDiagnostic(unittest.TestCase):
    def test_retained_winner_changes_without_using_slot4(self):
        ref, alt, entry, scorer = sample()
        row, activations, scores, swaps = stage.diagnose(ref, alt, entry, scorer, 5, {})
        self.assertEqual(row["first_activation"], 2)
        self.assertEqual(row["first_output_change"], 2)
        self.assertEqual(row["first_output_mechanism"], "retained_boxes_reranked")
        self.assertEqual(row["alternate_slot4_selections"], 0)
        self.assertEqual(row["subsequent_valid_frames"], 2)
        self.assertGreater(row["alternate_auc"], row["control_auc"])
        self.assertEqual([r["matched"] for r in activations], [True, False])
        self.assertEqual(len(scores), 3)
        rank = next(r for r in swaps if r["relation"] == "hann_rank")
        self.assertTrue(rank["insertion_matches_new"] and rank["restoration_matches_old"])
        self.assertFalse(any(r["insertion_matches_new"] for r in swaps if r is not rank))
        altered = dict(entry, ground_truth=np.tile([100.,0.,4.,4.], (5,1)))
        other = stage.diagnose(ref, alt, altered, scorer, 5, {})
        self.assertEqual(other[2:], (scores, swaps))
        self.assertEqual(other[0]["first_output_mechanism"], row["first_output_mechanism"])

    def test_sealed_report_and_tamper_detection(self):
        ref, alt, entry, scorer = sample()
        self.assertEqual(stage.mechanism(True, 0, 3, True), "alternate_box_selected")
        entry.update(sequence="test", frames=[None]*5)
        base_job = dict(spec=dict(cohort="expanded", unit=dict(sequence="test")), frames=4, outer_fold=0, seed=2312)
        jobs = {arm:dict(base_job, arm=arm) for arm in (stage.CONTROL, stage.source.TREATMENT)}
        models = {(0, "generator", 0): {"generator": {}}, (0, "generated_4", 2312): scorer}
        source_plan = dict(fingerprint="fixture")
        treatment = ("test", 2312, stage.source.TREATMENT)
        scope = {treatment: dict(frames="4", alternate_used_frames="2", alternate_selected_frames="0", first_alternate_frame="2")}
        metrics = {("test", 2312, stage.CONTROL): {"auc": 0.0}, treatment: {"auc": 100*10/21}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            root.mkdir()
            source = Path(directory) / "source"
            for arm, data in ((stage.CONTROL, ref), (stage.source.TREATMENT, alt)):
                job = jobs[arm]
                trace = source / "evaluation" / (stage.base.job_key(job) + ".npz")
                stage.c.save(trace, data, dict(plan_fingerprint=source_plan["fingerprint"], job=job))
            plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN,
                                      pairs=1, activations=2, inputs={}, source_stage121_fingerprint=stage.SOURCE121))
            stage.c.publish_identical(root / "manifest.json", plan)
            stage.run(root, source, plan, {("test", 2312):jobs}, models, lambda _:entry, 5, source_plan, scope, metrics)
            stage.check_results(root)
            path = root / "analysis/stage122_activations.csv"
            path.write_text("damaged", encoding="utf-8")
            with self.assertRaises(ValueError):
                stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
