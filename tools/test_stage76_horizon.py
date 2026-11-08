"""Check label reversal, split isolation and resumed four-choice evaluation."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage76_horizon_experiment as stage
import test_stage75_ppo_failure as fixture


class HorizonLabels(unittest.TestCase):
    def test_immediate_and_horizon_targets_use_the_same_legal_choices(self):
        design = dict(horizon=3, anchors_per_video=3)
        self.assertEqual(stage.fixed_frames(6, design), [1, 2, 4])
        mask = np.array([[True, True, False, True]])
        data = dict(frame=np.array([1]), x=np.zeros((1, 256)), action_mask=mask,
                    baseline_action=np.array([0]), search_calls=np.array([28]),
                    path_iou=np.array([[[.9, 0., 0.], [.5, .8, .8], [1., 1., 1.], [.5, .8, .8]]]))
        target, _ = stage.labels(data, [1], design, 1e-6)
        self.assertEqual(target["immediate_ce"].tolist(), [0])
        self.assertEqual(target["horizon_ce"].tolist(), [1])
        with self.assertRaisesRegex(ValueError, "fit receipt"):
            stage.labels(dict(data, baseline_action=np.array([2])), [1], design, 1e-6)


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class HorizonReplay(unittest.TestCase):
    def test_fit_collect_train_and_evaluate_with_committed_receipts(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        torch = fixture.torch
        torch.set_num_threads(1)
        quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                       critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
        with tempfile.TemporaryDirectory(prefix="stage76-", dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            original, _, observer, _ = TorchTests().fixture(folder)
            try:
                entries = original["plan"]["entries"]
                names = [entry["sequence"] for entry in entries]
                self.assertEqual(len(names), 2)
                folds = [dict(fold=i, fit=[names[i]], held_out=[names[1-i]]) for i in range(2)]
                jobs = [dict(fold=i, seed=2171, episodes=[dict(sequence=names[i], pass_index=0)]) for i in range(2)]
                old = dict(original["plan"], folds=folds, jobs=jobs, design=dict(stage.c.DESIGN, seeds=[2171]),
                           actor_design=dict(stage.e.e63.source_common.DESIGN), controller_design=dict(stage.c.c66.DESIGN))
                source_root, root = folder / "source", folder / "new"
                root.mkdir()
                norm = dict(center=[0.] * 256, scale=[1.] * 256)
                pack = stage.e.learning.initial(torch, 2171, norm, old["design"])
                source = dict(root=source_root, plan=old, prior={}, bindings={}, report=dict(fingerprint="fixture"),
                              run=dict(observed_heads=dict(native=digest(observer.rt.net.box_head))),
                              models=dict(fingerprint="fixture", controllers={stage.c.job_key(job): dict(ppo=pack) for job in jobs}))
                with (patch.object(stage.c, "fit_pack", return_value=quality),
                      patch.object(stage.e.source_experiment, "control_pack", return_value=dict(source=quality)),
                      patch.dict(stage.DESIGN, horizon=2, anchors_per_video=2, epochs=2, minibatch=2),
                      contextlib.redirect_stdout(io.StringIO())):
                    for job, entry in zip(jobs, entries):
                        unit = dict(sequence=entry["sequence"], origin="native", fold=job["fold"], seed=job["seed"], policy="ppo")
                        before = stage.source_io.seal(dict(schema=stage.c.SCHEMA))
                        stage.c.publish_identical(stage.c.step_path(source_root, job, 0), before)
                        trace = stage.e.run_actor(observer, entry, unit, old, {}, pack, fit=True)
                        stage.c.save(stage.c.trace_path(source_root, job, 0), trace, stage.e.train_meta(old, job, 0, before))
                    plan = stage.make_plan(source, old["runtime_identity"])
                    self.assertEqual(plan["fit_states"], 4)
                    self.assertEqual(plan["fit_search_cap"], 80)
                    for group in plan["fit_groups"]:
                        self.assertNotIn(group["sequence"], folds[group["fold"]]["held_out"])
                    report = stage.run(root, plan, source, observer, torch)
                    self.assertEqual(report["trajectories"], 12)
                    self.assertEqual(report["evaluation_search_calls"], 192)
                    self.assertEqual(set(report["supervised_updates_per_job"].values()), {2})
                    with patch.object(observer, "capture", side_effect=AssertionError("Committed inference repeated")):
                        self.assertEqual(stage.run(root, plan, source, observer, torch), report)
            finally:
                observer.handle.remove()
                observer.pre_handle.remove()


if __name__ == "__main__":
    unittest.main()
