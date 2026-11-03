"""Semantic controls and genuine small-CENTER source/continuation regression tests."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage63_choice_common as c
import stage63_choice_experiment as experiment
import stage63_choice_summary as summary


class ChoiceSemantics(unittest.TestCase):
    def test_anchor_rules_do_not_search_outcomes_or_silently_replace_missing_cases(self):
        m = dict(frame_id=np.arange(1, 7), q0_good_to_low=np.array([0, 0, 1, 1, 0, 0], bool),
                 available_selected_miss=np.array([0, 1, 0, 0, 1, 0], bool),
                 no_good_decoded=np.array([0, 0, 0, 1, 0, 0], bool), q0_good=np.array([1, 0, 1, 0, 0, 0], bool))
        self.assertEqual(c.select_frame(dict(case_id="a", selector="first_q0_good_to_low"), m, []), 3)
        self.assertEqual(c.select_frame(dict(case_id="b", selector="unique_window_miss", first=3, last=6), m, []), 5)
        self.assertEqual(c.select_frame(dict(case_id="c", selector="first_failure_no_good"), m, [dict(start=3)]), 4)
        with self.assertRaisesRegex(ValueError, "unique"):
            c.select_frame(dict(case_id="b", selector="unique_window_miss", first=1, last=6), m, [])
        with self.assertRaisesRegex(ValueError, "no support"):
            c.select_frame(dict(case_id="a", selector="first_q0_good_to_low"), dict(m, q0_good_to_low=np.zeros(6, bool)), [])
        with self.assertRaisesRegex(ValueError, "no good q0"):
            c.select_frame(dict(case_id="p", selector="fixed_good_q0", frame=4), m, [])

    def test_short_branch_gains_keep_immediate_and_future_choices_separate(self):
        rows = [dict(start_iou=a, mean_iou=b, following_mean_iou=b, endpoint_iou=b, low_iou_frames=l, good_frames=3-l)
                for a,b,l in ((.7, .2, 2), (.8, .3, 1), (.4, .8, 0), (.6, .5, 0))]
        case = dict(case_id="a", sequence="v", origin="native", seed=0, policy="history", purpose="fixture", frame=10,
                    recorded_choice=0, immediate_best=1)
        contrasts = summary.contrasts(case, rows)
        self.assertEqual([r["candidate"] for r in contrasts], [0, 1, 2])
        self.assertFalse(contrasts[0]["offline_GT_choice"])
        self.assertTrue(contrasts[1]["offline_GT_choice"])
        self.assertAlmostEqual(contrasts[1]["mean_iou_gain"], .1)
        self.assertAlmostEqual(contrasts[2]["mean_iou_gain"], .6)
        self.assertEqual(summary.first_three_good([.1, .5, .6, .7], 10), 11)
        self.assertIsNone(summary.first_three_good([.6, .7], 10))


try:
    import torch
    import cv2
except ImportError:
    torch = cv2 = None


@unittest.skipUnless(torch is not None and cv2 is not None, "Real CPU Torch/OpenCV required; skips do not establish the replay")
class RealCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def close(self, observer):
        observer.handle.remove()
        if hasattr(observer, "pre_handle"):
            observer.pre_handle.remove()

    def test_one_intervention_accepted_history_gt_isolation_dense_and_source_diagonal(self):
        from test_stage61_search import TorchTests as Fixture
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage63-actor-", dir=Path(__file__).parent) as folder:
            old, native, observer, _ = Fixture().fixture(Path(folder))
            plan = dict(old["plan"], actor_design=dict(experiment.source_common.DESIGN), design=dict(c.DESIGN, horizon=3))
            ap = c.actor_plan(plan)
            entry = old["plan"]["entries"][0]
            original = digest(native.rt.net)
            try:
                for policy in ("history", "recrop"):
                    trace = c.prior.source_actor.run_actor(observer, entry["frames"], entry["initial_box"], plan["root"], policy, ap)
                    frame = 2
                    history = experiment.history_at(trace, entry["initial_box"], frame, plan)
                    case = dict(case_id="actor", frame=frame, policy=policy, sequence=entry["sequence"], origin="native", seed=0,
                                purpose="fixture", recorded_choice=int(trace["selected"][frame-1]), immediate_best=0)
                    with patch.object(observer, "capture", wraps=observer.capture) as capture:
                        shared = experiment.collect_shared(observer, entry["frames"], entry["initial_box"], history, frame, policy, ap)
                        experiment.audit_shared(shared, case, entry, trace, plan)
                        tails = []
                        for q in range(4):
                            tail = experiment.collect_tail(observer, entry["frames"], entry["initial_box"], history, shared, q, frame, 3, policy, ap)
                            experiment.audit_tail(tail, shared, case, entry, trace, plan, q)
                            np.testing.assert_array_equal(tail["history_latest"][0], shared["candidate_boxes"][0, q])
                            tails.append(tail)
                        self.assertEqual(capture.call_count, 36)  # Four shared probes plus four actual eight-search tails.
                    self.assertEqual(int(shared["head_calls"].sum())+sum(int(t["head_calls"].sum()) for t in tails), 36)
                    self.assertGreater(len(np.unique(shared["candidate_boxes"][0], axis=0)), 1)
                    changed = copy.deepcopy(plan)
                    changed["entries"][0]["ground_truth"][1:] = [[130., 120., 10., 10.]]*4
                    again = experiment.collect_tail(observer, entry["frames"], entry["initial_box"], history, shared, 1, frame, 3, policy, c.actor_plan(changed))
                    for key in experiment.FIELDS:
                        np.testing.assert_array_equal(again[key], tails[1][key])
                    before = summary.branch_rows(plan, case, entry, shared, tails)[0]
                    altered = dict(entry, ground_truth=[[130., 120., 10., 10.]]*5)
                    after = summary.branch_rows(plan, case, altered, shared, tails)[0]
                    self.assertNotEqual([r["mean_iou"] for r in before], [r["mean_iou"] for r in after])
                    details = summary.dense_rows(plan, case, entry, shared)
                    self.assertEqual(len(details), 4)
                    self.assertTrue(all(0 <= r["best_dense_gt_iou"] <= 1 for r in details))
                    bad = copy.deepcopy(tails[0]); bad["candidate_parents"][0, 0, 0] += 1.
                    with self.assertRaises(AssertionError):
                        experiment.audit_tail(bad, shared, case, entry, trace, plan, 0)
                    bad = copy.deepcopy(shared); bad["candidate_boxes"][0, 0, 0] += .1
                    with self.assertRaises((AssertionError, ValueError)):
                        experiment.audit_shared(bad, case, entry, trace, plan)
                self.assertEqual(digest(native.rt.net), original)
            finally:
                self.close(observer)

    def test_full_sealed_source_chain_l40_binding_resume_immutability_and_tamper(self):
        from test_stage61_search import TorchTests as Fixture
        import stage59_parent_collect as e59
        import stage59_parent_summary as s59
        import stage60_state_common as c60
        import stage60_state_experiment as e60
        import stage60_state_summary as s60
        import stage61_search_common as c61
        import stage61_search_experiment as e61
        import stage61_l40_common as l40
        import stage61_l40_experiment as e40
        import stage61_l40_summary as s40
        from stage4_common import write_json, fingerprint
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage63-workflow-", dir=Path(__file__).parent) as directory, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory)
            old, native, observer, records = Fixture().fixture(folder)
            try:
                # Genuine CPU head payloads at a synthetic Stage-58 boundary; no GPU-result claim.
                for ordinal, (record, state) in enumerate(records):
                    path = old["root"]/("head"+str(ordinal)+".pth")
                    meta = dict(fixture=True, fold=record["fold"], seed=record["seed"], arm=record["arm"])
                    torch.save(dict(metadata=meta, step=1, head=state), path)
                    receipt = dict(schema="stage58_projected_endpoint_gradient_v1", metadata=meta, step=1, sha256=c.file_hash(path))
                    write_json(path.with_suffix(".receipt.json"), dict(receipt, fingerprint=fingerprint(receipt)))
                    record.update(checkpoint=str(path), checkpoint_sha256=c.file_hash(path), receipt_sha256=c.file_hash(path.with_suffix(".receipt.json")))
                    old["bindings"].update({str(path):record["checkpoint_sha256"], str(path.with_suffix(".receipt.json")):record["receipt_sha256"]})
                old["plan"]["entries"][0]["ground_truth"][1:] = [[1., 1., 1., 1.]]*4
                raw = old["plan"]["runtime"]
                raw.update(device="NVIDIA L20", torch=str(torch.__version__), cuda=str(torch.version.cuda))
                old["plan"]["runtime_identity"] = dict(tracker=copy.deepcopy(raw), gpu_uuid="source-l20", model_state_sha256=native.digest,
                                                        opencv=cv2.__version__, numpy=np.__version__)
                upstream = [dict(case_id="T01", sequence="v0", frame=2, purpose="fixture", anchor_method="endpoint_control_seed2132", anchor_kind="fixed_frame")]
                cached_cases = (c.prior.c.case("T01", "v0", "native", 0, "history", 1, "onset", "fixture"),)
                own_cases = (c.spec("T01", "v0", "native", 0, "recrop", "fixed_frame", "fixture", frame=2),
                             c.spec("T02", "v1", "endpoint", 2132, "history", "fixed_frame", "fixture", frame=2))
                with patch.object(c60.prior, "load_source", return_value=old), patch.object(c60.prior, "CONFIG", upstream), \
                     patch.dict(c60.prior.DESIGN, horizon=2), patch.dict(c60.DESIGN, horizon=2, anchors_per_video=2, smoke_frames=3, bootstrap_samples=30), \
                     patch.dict(c61.DESIGN, smoke_frames=3, bootstrap_samples=30), patch.object(c.prior.c, "CASES", cached_cases), \
                     patch.object(c, "CONFIG", own_cases), patch.dict(c.DESIGN, horizon=3):
                    root59 = folder/"stage59"; e59.prepare(old["root"], root59); p59, i59 = c60.prior.load_plan(root59)
                    banks = {f["fold"]:e59.HeadBank(native, records, f["fold"]) for f in old["plan"]["folds"]}
                    e59.execute(root59, p59, i59, observer, banks, old["plan"]["runtime_identity"], True)
                    e59.execute(root59, p59, i59, observer, banks, old["plan"]["runtime_identity"]); s59.summarize(root59)
                    root60 = folder/"stage60"; e60.prepare(root59, root60); p60, i60 = c60.load_plan(root60)
                    e60.smoke(p60, root60, i60, observer, records); e60.collect_fit(p60, root60, observer, records)
                    e60.fit_models(p60, root60, records); e60.evaluate(p60, root60, i60, observer, records); s60.summarize(root60)
                    root61 = folder/"stage61"; e61.prepare(root60, root61)
                    root40 = folder/"l40"; e40.prepare(root61, root40, dict(device="NVIDIA L40", gpu_uuid="source-l40"))
                    p40, i40 = l40.load_plan(root40)
                    e40.smoke(p40, root40, i40, observer, records); e40.evaluate(p40, root40, i40, observer, records); s40.summarize(root40)
                    for mode in ("smoke", "evaluate"):
                        write_json(root40/"diagnostics"/("stage61_runtime_"+mode+".json"), dict(schema="stage61_runtime_replay_diagnostic_v1",
                            mode=mode, status="EXACT_SOURCE_RUNTIME_VERIFIED", plan_fingerprint=p40["fingerprint"], mismatches=[],
                            verified_runtime_fingerprint=fingerprint(p40["runtime_identity"]), launcher_sha256=p40["code_sha256"]["stage61_runtime_replay.py"]))
                    root62 = folder/"stage62"; c.prior.audit(root40, root62)
                    frozen = {str(p):p.read_bytes() for d in (old["root"], root59, root60, root61, root40, root62) for p in d.rglob("*") if p.is_file()}
                    original = digest(native.rt.net)
                    root = folder/"stage63"
                    experiment.prepare(root62, root, dict(device="NVIDIA L40", gpu_uuid="new-l40"))
                    plan, inputs = c.load_plan(root)
                    self.assertEqual(plan["formal_search_calls"], 72)
                    self.assertEqual(plan["counterfactual_branches"], 8)
                    self.assertEqual([r["field"] for r in plan["declared_runtime_differences"]], ["runtime.gpu_uuid"])
                    loaded = experiment.load_records(inputs, torch)
                    self.assertEqual(len(loaded), 4)
                    with patch.object(observer, "capture", wraps=observer.capture) as capture:
                        experiment.execute(root, plan, inputs, observer, loaded, plan["runtime_identity"], True)
                        self.assertEqual(capture.call_count, 12)
                    with patch.object(observer, "capture", wraps=observer.capture) as capture:
                        experiment.execute(root, plan, inputs, observer, loaded, plan["runtime_identity"])
                        self.assertEqual(capture.call_count, 60)  # Smoke's shared/recorded branch is reused.
                    with patch.object(observer, "capture", side_effect=AssertionError("Committed probes must not repeat")):
                        experiment.execute(root, plan, inputs, observer, loaded, plan["runtime_identity"], True)
                        experiment.execute(root, plan, inputs, observer, loaded, plan["runtime_identity"])
                    with patch.dict("sys.modules", torch=None, cv2=None), patch.object(experiment, "checked_image", side_effect=AssertionError("Summary cannot read images")):
                        report = summary.summarize(root)
                        self.assertEqual(report, summary.summarize(root))
                    self.assertEqual(report["formal_search_calls"], 72)
                    self.assertEqual(report["additional_smoke_calls"], 0)
                    self.assertTrue(report["counterfactual_continuations_performed"])
                    self.assertFalse(report["training_performed"])
                    self.assertFalse(report["online_method_established"])
                    self.assertEqual(digest(native.rt.net), original)
                    self.assertTrue(all(Path(p).read_bytes() == content for p,content in frozen.items()))
                    for path in (root62/"analysis/stage62_candidates.csv", experiment.paths(root, plan["cases"][0], 0)):
                        saved = path.read_bytes(); path.write_bytes(saved+b"tamper")
                        with self.assertRaises(ValueError):
                            summary.summarize(root)
                        path.write_bytes(saved)
                    bad = copy.deepcopy(plan); bad["runtime_identity"]["tracker"]["device"] = "NVIDIA L20"
                    with self.assertRaises(ValueError):
                        experiment.execute(root, bad, inputs, observer, loaded, plan["runtime_identity"], True)
                    with self.assertRaises(ValueError):
                        c.make_plan(root62, dict(device="NVIDIA L20", gpu_uuid="other"))
                    path = root/"analysis/stage63_contrasts.csv"; path.write_bytes(path.read_bytes()+b"tamper")
                    with self.assertRaisesRegex(ValueError, "differs"):
                        summary.summarize(root)
            finally:
                self.close(observer)


if __name__ == "__main__":
    unittest.main()
