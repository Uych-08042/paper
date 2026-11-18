"""Check the one-frame carry and exercise both cache modes without a GPU."""
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage121_search_state as stage
import test_stage113_frozen_closed_loop as fixture


class SearchState(unittest.TestCase):
    setUp = fixture.FrozenClosedLoop.setUp

    def test_ambiguity_replaces_only_last_distinct_crop(self):
        history = [np.array([60., 10., 10., 10.])]
        parents = np.array([history[0], [72., 10., 10., 10.], [84., 10., 10., 10.], [96., 10., 10., 10.]])
        previous = dict(selected=2, scores=np.array([.70, .1, .73, .3]),
                        candidate_boxes=np.array([[10., 10., 10., 10.], parents[1], history[0], parents[3]]))
        with patch.object(stage.base, "forecast_parents", side_effect=lambda *_: dict(candidate_parents=parents.copy())):
            row = stage.forecast(history, previous, (128, 128), self.ap, None)
            self.assertTrue(row["alternate_requested"] and row["alternate_used"])
            np.testing.assert_array_equal(row["candidate_parents"][:3], parents[:3])
            np.testing.assert_array_equal(row["candidate_parents"][3], previous["candidate_boxes"][0])
            for state in (None, dict(previous, selected=0), dict(previous, scores=np.array([.5, .1, .73, .3]))):
                unused = stage.forecast(history, state, (128, 128), self.ap, None)
                self.assertFalse(unused["alternate_used"])
                np.testing.assert_array_equal(unused["candidate_parents"], parents)
            previous["candidate_boxes"][0] = parents[3]
            duplicate = stage.forecast(history, previous, (128, 128), self.ap, None)
            self.assertTrue(duplicate["alternate_requested"])
            self.assertFalse(duplicate["alternate_used"])

    def test_full_rollout_reports_and_reuses_completed_controls(self):
        source = stage.base
        entry = self.entry
        spec = dict(cohort="expanded", unit=dict(sequence=entry["sequence"], origin="native"))
        owner = dict(spec=spec, frames=len(entry["frames"]) - 1, outer_fold=0)
        control_jobs = [dict(owner, arm=arm, seed=seed) for arm in stage.REFERENCES
                        for seed in ([0] if arm == "native" else source.DESIGN["seeds"])]
        jobs = control_jobs + [dict(owner, arm=stage.TREATMENT, seed=seed) for seed in source.DESIGN["seeds"]]
        models = {(0, "generator", 0): self.package}
        models.update({(0, bank, seed): self.model for bank in ("fixed_4", "generated_4")
                       for seed in source.DESIGN["seeds"]})
        source_plan = dict(fingerprint="fixture")
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(source.e63.source_experiment, "installed", return_value=contextlib.nullcontext("native"))):
            root = Path(directory)
            fresh = root / "fresh"
            cached = root / "cached"
            for reuse, trial in ((False, fresh), (True, cached)):
                plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, jobs=jobs, runtime={},
                                          source_stage113_fingerprint=stage.SOURCE113, reuse_controls=reuse,
                                          source_sha256={}, code_sha256={}, native_head_sha256="native"))
                trial.mkdir()
                stage.c.publish_identical(trial / "manifest.json", plan)
                before = self.observer.calls
                stage.run(trial, root / "source", source_plan, plan, models, lambda _: entry,
                          self.old, self.observer, image_reader=self.reader)
                report = stage.c.c66.sealed(trial / "analysis/stage121_decision.json", stage.SCHEMA)
                self.assertEqual(report["new_search_calls"], self.observer.calls - before)
                self.assertEqual(report["new_full_trajectories"], 2 if reuse else 7)
                self.assertEqual(report["evaluated_trajectories"], 7)
                stage.check_results(trial)
                if not reuse:
                    for job in control_jobs:
                        generator, scorer = source.job_models(job, models)
                        path = trial / "evaluation" / (source.job_key(job) + ".npz")
                        data = stage.c.read_cache(path, dict(plan_fingerprint=plan["fingerprint"], job=job))
                        meta = dict(plan_fingerprint=source_plan["fingerprint"], job=job,
                                    generator_fingerprint=self.package["fingerprint"] if generator is not None else None,
                                    scorer_fingerprint=scorer["fingerprint"] if scorer is not None else None)
                        stage.c.save(root / "source" / "evaluation" / path.name, data, meta)


if __name__ == "__main__":
    unittest.main()
