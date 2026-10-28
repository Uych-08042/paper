"""Causal recovery, exact compute budgets, baseline parity and frozen evidence."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash
import stage43_recovery_common as common
import stage43_recovery_experiment as experiment
import stage43_recovery_summary as summary

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


class ControllerTests(unittest.TestCase):
    def test_history_is_past_only_and_prefix_cap_cooldown_and_hold_are_exact(self):
        initial, jump = np.array([20.,20.,12.,12.]), np.array([60.,60.,12.,12.])
        c = common.Controller(initial, "history_search")
        for frame in range(1, 9):
            box, log = c.finish(frame, initial, initial, .8, (100,100))
            self.assertFalse(log["alarm"])
        # An alarm at t=9 is observable but cannot spend a not-yet-earned call.
        _, ninth = c.finish(9, initial, jump, .1, (100,100))
        self.assertTrue(ninth["alarm"])
        self.assertFalse(ninth["eligible"])
        self.assertEqual(c.history[-1][0], 8)
        view = c.inspect(10, jump, jump, .1, (100,100))
        np.testing.assert_array_equal(common.center(view["alternate_parent"]), common.center(initial))
        np.testing.assert_array_equal(view["alternate_parent"][2:], initial[2:] * 2)
        selected, log = c.finish(10, jump, jump, .1, (100,100), (initial, .7))
        np.testing.assert_array_equal(selected, initial)
        self.assertEqual(log["choice"], 1)
        self.assertEqual(c.history[-1][0], 10)
        spent = [10]
        for frame in range(11, 61):
            view = c.inspect(frame, selected, jump, .1, (100,100))
            alt = (initial, .7) if view["eligible"] else None
            selected, log = c.finish(frame, selected, jump, .1, (100,100), alt)
            if view["eligible"]:
                spent.append(frame)
            self.assertLessEqual(c.used, frame // 10)
        self.assertEqual(spent, [10,20,30,40,50,60])
        with self.assertRaisesRegex(ValueError, "trigger/budget"):
            c.finish(61, initial, jump, .1, (100,100), (initial,.7))
        hold = common.Controller(initial, "motion_hold")
        for frame in range(1,9):
            hold.finish(frame, initial, initial, .8, (100,100))
        selected, log = hold.finish(10, jump, jump, .1, (100,100))
        np.testing.assert_array_equal(selected, initial)
        self.assertEqual(log["choice"], 2)
        self.assertFalse(log["history_update"])
        self.assertEqual(hold.history[-1][0], 8)

    def test_local_and_history_proposals_only_differ_in_center_and_motion_is_bounded(self):
        initial = np.array([20.,20.,12.,12.])
        controllers = [common.Controller(initial,p) for p in ("local_search", "history_search")]
        for c in controllers:
            for frame in range(1,9):
                c.finish(frame, initial, initial + [frame,0,0,0], .8, (200,200))
        previous, wrong = initial + [20,0,0,0], initial + [80,80,0,0]
        a,b = [c.inspect(20,previous,wrong,.1,(200,200)) for c in controllers]
        np.testing.assert_array_equal(a["alternate_parent"][2:], previous[2:] * 2)
        np.testing.assert_array_equal(common.center(a["alternate_parent"]), common.center(previous))
        np.testing.assert_array_equal(common.center(b["alternate_parent"]), common.center(b["forecast"]))
        np.testing.assert_array_equal(controllers[1].forecast(100), controllers[1].forecast(20))
        self.assertEqual(a["score_floor"], .4)

    def test_video_gate_counts_seeds_inside_videos_and_rejects_stable_damage_and_no_use(self):
        plan = dict(fit=[dict(sequence=str(i)) for i in range(8)])
        rows = []
        for entry in plan["fit"]:
            for seed in common.SEEDS:
                for model in common.MODELS:
                    for policy in common.POLICIES:
                        gain = dict(native=0,motion_hold=.2,local_search=.3,history_search=1.)[policy]
                        rows.append(dict(sequence=entry["sequence"], seed=seed, model=model, policy=policy,
                            tracked_frames=100, auc=60. + gain, low_iou_fraction=.02 if policy == "history_search" else .03,
                            failure_events=0, longest_failure_fraction=0., reliable_frames=100, reliable_auc=70.,
                            baseline_stable=True, introduced_failure=False, alarm_frames=4, interventions=2 if policy != "native" else 0,
                            alternate_calls=2 if policy in ("local_search","history_search") else 0,
                            total_head_calls=102 if policy in ("local_search","history_search") else 100,
                            changed_frames=2 if policy != "native" else 0, changed_ratio=.02 if policy != "native" else 0., alternate_ratio=.02))
        with mock.patch.dict(common.DESIGN, bootstrap_samples=64):
            tables, good = summary.analyze(rows,plan)
            self.assertTrue(good["pilot_pass"])
            self.assertEqual(good["primary"]["videos"],8)
            self.assertEqual(good["primary"]["mean_delta"],1.)
            changed = copy.deepcopy(rows)
            next(r for r in changed if r["policy"] == "history_search" and r["model"] == "augmented")["introduced_failure"] = True
            self.assertFalse(summary.analyze(changed,plan)[1]["retention_guard_pass"])
            unused = copy.deepcopy(rows)
            for r in unused:
                r["changed_frames"] = 0
            self.assertFalse(summary.analyze(unused,plan)[1]["nontrivial_usage_pass"])
            with self.assertRaisesRegex(ValueError,"complete unmixed"):
                summary.analyze(rows + rows[:1],plan)


@unittest.skipUnless(TORCH, "CPU PyTorch needed for real frozen-model replay")
class RolloutTests(unittest.TestCase):
    def test_full_source_chain_gt_isolation_resume_parity_and_tamper(self):
        import torch
        import stage40_crop_common as c40
        import stage40_crop_experiment as e40
        import stage40_crop_summary as s40
        import stage41_transfer_audit as a41
        import stage42_state_common as c42
        import stage42_state_experiment as e42
        import stage42_state_summary as s42
        from test_stage40_crop_augmentation import fixture
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(torch.cuda,"is_available",return_value=False), \
                mock.patch.dict(c40.DESIGN,updates=3,smoke_updates=2,batch_size=1,save_every=1,bootstrap_samples=64), \
                mock.patch.dict(c40.prior.DESIGN,horizon=3,smoke_horizon=2), \
                mock.patch.dict(a41.CONFIG,bootstrap_samples=64), \
                mock.patch.dict(c42.DESIGN,followup_frames=3,smoke_frames=2,bootstrap_samples=64), \
                mock.patch.dict(common.DESIGN,smoke_frames=4,bootstrap_samples=64):
            root = Path(temporary)
            old,rt,reader,adapter,bindings = fixture(root,torch)
            r40,r41,r42,out = [root / n for n in ("stage40","stage41","stage42","stage43")]
            with mock.patch.object(c40,"source_info",side_effect=adapter):
                p40 = c40.prepare(old,r40)
                e40.smoke(rt,p40,r40,reader)
                for seed in c40.SEEDS:
                    e40.train(rt,p40,r40,seed,image_reader=reader)
                e40.evaluate(rt,p40,r40,image_reader=reader)
                s40.summarize(r40)
                a41.audit(r40,r41)
                p42 = c42.prepare(r41,r42)
                e42.collect(rt,p42,r42,"smoke",image_reader=reader)
                e42.collect(rt,p42,r42,"full",image_reader=reader)
                s42.summarize(r42)
                frozen = {str(p):file_hash(p) for r in (r40,r41,r42) for p in r.rglob("*") if p.is_file()}
                plan = common.prepare(r42,out)
                self.assertEqual(plan,common.load_plan(out))
                with self.assertRaises(FileNotFoundError):
                    experiment.collect(rt,plan,out,"full",image_reader=reader)
                experiment.collect(rt,plan,out,"smoke",image_reader=reader)
                experiment.smoke_gate(out,plan)
                e = plan["fit"][0]
                experiment.collect(rt,plan,out,"full",e["sequence"],reader)
                with self.assertRaisesRegex(ValueError,"128 complete"):
                    summary.summarize(out)
                nets,_ = e40.evaluation_models(rt,p40,r40)
                net = nets[(experiment.fold_for(plan,e),"augmented_seed2110")]
                observed = [0]
                def count_frame(*args,**kwargs):
                    observed[0] += 1
                hook = net.box_head.register_forward_hook(count_frame)
                read = experiment.read_boxes
                def after_all_frames(path):
                    self.assertGreaterEqual(observed[0],len(e["frames"]) - 1)
                    return read(path)
                with mock.patch.object(experiment,"read_boxes",side_effect=after_all_frames):
                    before = experiment.rollout(rt,plan,e,net,"history_search",image_reader=reader)
                hook.remove()
                future = Path(plan["root"]) / "annotations/changed_future.txt"
                future.write_text("20,20,12,12\n" + "\n".join(["50,50,5,5"] * 12),encoding="utf-8")
                other = dict(e,annotation="annotations/changed_future.txt",annotation_sha256=file_hash(future))
                after = experiment.rollout(rt,plan,other,net,"history_search",image_reader=reader)
                for key in before:
                    if key not in ("gt","iou"):
                        np.testing.assert_array_equal(before[key],after[key])
                self.assertFalse(np.array_equal(before["gt"],after["gt"]))
                # Exercise an actual second neural forward, even when tiny-fixture
                # natural responses produce no alarms. Only a synthetic score map
                # is substituted at the tenth native call; decoder math stays real.
                real_forward, forced_calls = experiment.forward_student, [0]
                def low_tenth_response(*args):
                    result = real_forward(*args)
                    forced_calls[0] += 1
                    if forced_calls[0] == 10:
                        result = dict(result, score_map=torch.full_like(result["score_map"],1e-6))
                    return result
                with mock.patch.object(experiment,"forward_student",side_effect=low_tenth_response):
                    exercised = experiment.rollout(rt,plan,e,net,"history_search",image_reader=reader)
                self.assertEqual(exercised["head_calls"][9],2)
                self.assertTrue(exercised["eligible"][9])
                self.assertGreater(exercised["alternate_score"][9],0)
                common.audit_control(exercised,e,"history_search")
                short_events = summary.event_rows(e,2110,"augmented","history_search",exercised)
                self.assertTrue(short_events)
                self.assertTrue(all(r["post40_mean_iou"] is None for r in short_events))
                bad = dict(before,previous=before["previous"].copy())
                bad["previous"][1,0] += 1
                with self.assertRaises(AssertionError):
                    common.audit_control(bad,e,"history_search")
                experiment.collect(rt,plan,out,"full",image_reader=reader)
                report = summary.summarize(out)
                self.assertEqual(report["full_trajectories"],128)
                self.assertEqual(report["maximum_native_box_error"],0.)
                self.assertEqual(report["maximum_native_iou_error"],0.)
                self.assertFalse(report["training_performed"])
                self.assertFalse(report["novel_method_established"])
                self.assertEqual(report,summary.summarize(out))
                common.verify_bindings(frozen)
                common.verify_bindings(bindings)
                with mock.patch.object(rt,"prepare",wraps=rt.prepare) as spy:
                    experiment.collect(rt,plan,out,"full",image_reader=reader)
                    self.assertEqual(spy.call_count,0)
                path = common.result_path(out,"full",e,2110,"augmented","history_search")
                path.with_suffix(".json").unlink()
                experiment.collect(rt,plan,out,"full",e["sequence"],reader)
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError,"Stage-43 trace changed"):
                    summary.summarize(out)
                path.write_bytes(original)
                checkpoint = next((r40 / "training").rglob("step000003.pt"))
                checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper checkpoint")
                with self.assertRaisesRegex(ValueError,"Frozen Stage-35 artifact changed"):
                    common.load_plan(out)


if __name__ == "__main__":
    unittest.main()
