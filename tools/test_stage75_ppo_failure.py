"""Regression checks for the fixed anchors and forced-choice continuation."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage75_ppo_failure_common as c
import stage75_ppo_failure_experiment as experiment
from stage27_uav_collect import writer_lock


class FixedCases(unittest.TestCase):
    def test_failure_categories_and_tail_budget(self):
        self.assertEqual(c.frame_category([.7, .1, .6, .2], 1), "good_q0_replaced_by_low")
        self.assertEqual(c.frame_category([.3, .1, .6, .2], 1), "missed_good_candidate")
        self.assertEqual(c.frame_category([.1, .18, .1, .02], 1), "all_low")
        self.assertEqual(c.frame_category([.1, .49, .1, .02], 1), "no_good_candidate")
        self.assertEqual(c.frame_category([.1, .5, .1, .02], 1), "selected_good")

        lengths = dict(uav0000003_00000_s=144, uav0000304_00253_s=1725,
                       uav0000176_00000_s=775, uav0000307_04531_s=759, uav0000204_00000_s=921)
        distinct = np.array([[q, 0., 2., 2.] for q in range(4)])
        counts = {}
        for spec in c.CASES:
            size = lengths[spec["sequence"]]
            trace = dict(candidate_boxes=np.broadcast_to(distinct, (size, 4, 4)), selected=np.zeros(size, int))
            row = spec["reference_frame"] - 1
            case = c.resolve_case(spec, {}, dict(frames=range(size + 1)), trace,
                                  [dict(start=row, recovery=row)], "fixture")
            counts[case["case_id"]] = (case["horizon"], case["search_calls"])
        self.assertEqual(len(counts), 13)
        self.assertEqual([counts[key][0] for key in ("D01", "D02", "E01", "E02")], [7, 5, 15, 13])
        self.assertEqual(sum(count for _, count in counts.values()), 5092)
        trace = dict(candidate_boxes=np.tile(distinct[[0, 0, 2, 2]], (144, 1, 1)), selected=np.zeros(144, int))
        case = c.resolve_case(c.CASES[0], {}, dict(frames=range(145)), trace, [dict(start=137)], "fixture")
        self.assertEqual(case["legal_actions"], [0, 2])
        self.assertEqual(case["search_calls"], 52)
        with self.assertRaisesRegex(ValueError, "Source event differs"):
            c.resolve_case(c.CASES[0], {}, dict(frames=range(145)), trace, [dict(start=138)], "fixture")

    def test_branch_metrics_exclude_anchor_and_keep_short_tail_empty(self):
        entry = dict(ground_truth=[[0., 0., 4., 4.]] * 3)
        case = dict(case_id="metric", sequence="v", seed=2171, frame=1, horizon=2, purpose="fixture", recorded_action=0)
        anchor = dict(candidate_boxes=np.array([[[0., 0., 4., 4.], [12., 0., 4., 4.]]]))
        tails = {0: dict(boxes=np.array([[12., 0., 4., 4.]]), selected=np.array([1])),
                 1: dict(boxes=np.array([[0., 0., 4., 4.]]), selected=np.array([0]))}
        rows, frames = experiment.summarize_branches(case, entry, anchor, tails)
        self.assertEqual(rows[1]["anchor_gain"], -1)
        self.assertEqual(rows[1]["mean_gain"], 0)
        self.assertEqual(rows[1]["following_mean_gain"], 1)
        self.assertEqual([row["selected"] for row in frames], [0, 1, 1, 0])
        rows, _ = experiment.summarize_branches(dict(case, horizon=1), entry, anchor,
                                               {0: dict(boxes=np.empty((0, 4)), selected=np.empty(0, int))})
        self.assertIsNone(rows[0]["following_mean_gain"])


try:
    import cv2
    import torch
except ImportError:
    cv2 = torch = None


@unittest.skipUnless(torch is not None and cv2 is not None, "CPU Torch and OpenCV are required for CENTER replay")
class RealReplay(unittest.TestCase):
    def test_sealed_source_center_replay_and_interrupted_resume(self):
        from test_stage61_search import TorchTests
        from stage48_stability_runtime import digest
        import stage74_ppo_runtime as runtime

        torch.set_num_threads(1)
        # This fixture tests the Stage-75 boundary with real public predictions,
        # masks and causal features. Upstream training attestations are fixtures.
        with tempfile.TemporaryDirectory(prefix="stage75-workflow-", dir=Path(__file__).parent) as directory:
            folder = Path(directory)
            original, _, observer, _ = TorchTests().fixture(folder)
            quality = dict(quality=dict(target="quality", scale=[1.] * 64, weights=[0.] * 64),
                           critic=dict(bias=.5, center=[0.] * 64, scale=[1.] * 64, weights=[0.] * 64))
            try:
                plan, prior, models, run, bridge = self.fixture_source(folder / "stage74", original, observer, runtime, digest, quality)
                with (patch.object(c.source_common, "load_plan", return_value=(plan, prior)),
                      patch.object(c.source_common, "fit_pack", return_value=quality),
                      patch.object(c.source_experiment, "load_models", return_value=models),
                      patch.object(c.source_experiment, "completed", return_value=run),
                      patch.object(c.source_experiment, "bridge_gate", return_value=bridge)):
                    self.check_replay(folder / "stage74", observer, digest)
            finally:
                observer.handle.remove()
                observer.pre_handle.remove()

    def fixture_source(self, root, original, observer, runtime, digest, quality):
        root.mkdir()
        entry = copy.deepcopy(original["plan"]["entries"][0])
        entry["ground_truth"][1:] = [[1., 1., 1., 1.]] * (len(entry["frames"]) - 1)
        unit = dict(sequence=entry["sequence"], origin="native", fold=0)
        keys = [("q0", 0), ("quality", 0), ("future", 0)]
        keys += [(policy, seed) for policy in ("ppo", "myopic") for seed in (2171, 2172)]
        units = [dict(unit, policy=policy, seed=seed) for policy, seed in keys]
        stage74 = c.source_common
        schema = stage74.SCHEMA
        base = {key: value for key, value in original["plan"].items() if key != "fingerprint"}
        plan = c.seal(dict(base, schema=schema, entries=[entry], units=units,
                           design=dict(stage74.DESIGN), actor_design=dict(experiment.e63.source_common.DESIGN),
                           controller_design=dict(stage74.c66.DESIGN), source_file_sha256=original["bindings"],
                           code_sha256={}, formal_eval_calls=4 * (len(entry["frames"]) - 1) * len(units)))
        norm = dict(center=[0.] * 256, scale=[1.] * 256)
        q0 = runtime.initial(torch, 2171, norm, plan["design"])
        controllers = {}
        for seed in (2171, 2172):
            pack = runtime.initial(torch, seed, norm, plan["design"])
            pack["state"]["actor.2.bias"] = [0., 1., 2., 3.]
            controllers[stage74.job_key(dict(unit, seed=seed))] = dict(ppo=pack, myopic=q0)
        models = c.seal(dict(schema=schema, controllers=controllers, training_evidence_sha256={}))
        prior = dict(stage73=dict(report=dict(fingerprint="fixture_report"), models=dict(fingerprint="fixture_models")))
        bridge = c.seal(dict(schema=schema, cache_sha256={}))
        hashes = {}
        with patch.object(stage74, "fit_pack", return_value=quality):
            for provider in units:
                pack = controllers[stage74.job_key(provider)]["ppo"] if provider["policy"] == "ppo" else q0
                trace = c.source_experiment.run_actor(observer, entry, dict(provider, policy="ppo"), plan, prior, pack)
                path = stage74.eval_path(root, provider)
                stage74.save(path, trace, c.source_experiment.eval_meta(plan, provider, models))
                experiment.e63.source_experiment.bind(hashes, path)
        head = digest(observer.rt.net.box_head)
        run = c.seal(dict(schema=schema, cache_sha256=hashes, observed_heads={stage74.key(u): head for u in units}))
        table_hashes = {}
        (root / "analysis").mkdir()
        for name in c.SOURCE_TABLES:
            path = root / "analysis" / f"stage74_{name}.csv"
            path.write_bytes(b"fixture\r\ntrue\r\n")
            table_hashes[path.name] = c.file_hash(path)
        report = c.seal(dict(schema=schema, status="REVIEW_DIRECT_FOUR_CANDIDATE_PPO_PILOT",
                             plan_fingerprint=plan["fingerprint"], training_fingerprint=models["fingerprint"],
                             run_fingerprint=run["fingerprint"], runtime=plan["runtime_identity"],
                             source_report_fingerprint="fixture_report", source_training_fingerprint="fixture_models",
                             source_bindings_fingerprint=c.fingerprint(plan["source_file_sha256"]),
                             source_bridge_fingerprint=bridge["fingerprint"], training_evidence_fingerprint=c.fingerprint({}),
                             eval_evidence_fingerprint=c.fingerprint(hashes), full_evaluation_trajectories=len(units),
                             eval_search_calls=plan["formal_eval_calls"], actions=[0, 1, 2, 3], new_recovery_search_calls=0,
                             recovery_branch_present=False, sgla_training_performed=False, source_models_unchanged=True,
                             maximum_source_common_field_error=0., table_sha256=table_hashes))
        for name, artifact in (("manifest.json", plan), ("models/training.json", models), ("evaluation/run.json", run),
                               ("analysis/stage74_decision.json", report), ("smoke/bridge.json", bridge),
                               ("smoke/train.json", models), ("smoke/evaluate.json", run)):
            c.publish_identical(root / name, artifact)
        return plan, prior, models, run, bridge

    def check_replay(self, root74, observer, digest):
        source = c.load_source(root74)
        plan74 = source["plan"]
        entry = plan74["entries"][0]
        sequence = entry["sequence"]
        units = {(unit["policy"], unit["seed"]): unit for unit in plan74["units"] if unit["sequence"] == sequence}
        q0 = c.read_trace(source, units["q0", 0])
        failures = c.failure_events(c.tracking_iou(q0["boxes"], np.asarray(entry["ground_truth"])[1:]))
        self.assertTrue(failures)
        reference = failures[0]["start"] + 1
        ppo = c.read_trace(source, units["ppo", 2171])
        frame = next(frame for frame in range(2, len(ppo["frame_id"]))
                     if c.source_common.action_mask(ppo["candidate_boxes"][frame - 1]).sum() > 1)
        base = c.case_spec("active", sequence, 2171, reference, "fixture", "first_failure", "q0", 0)
        specs = (dict(base, offset=frame - reference),
                 dict(base, case_id="last", offset=len(ppo["frame_id"]) - reference))
        root = root74.parent / "stage75"
        frozen = digest(observer.rt.net)
        actual_capture = type(observer).capture.__get__(observer, type(observer))
        saved_tail = experiment.collect_tail

        def interrupted(*args, **kwargs):
            if args[5] != args[4]["recorded_action"]:
                raise RuntimeError("interrupted alternative")
            return saved_tail(*args, **kwargs)

        with patch.object(c, "CASES", specs), writer_lock(root), contextlib.redirect_stdout(io.StringIO()):
            plan, cached = experiment.prepare(root, source)
            self.assertEqual(cached["search_calls"], 0)
            self.assertEqual(cached["same_state_contexts"], 14)
            self.assertEqual(plan["cases"][1]["horizon"], 1)
            run = experiment.runtime_record(plan, source, plan74["runtime_identity"])
            with patch.object(observer, "capture", wraps=actual_capture) as calls:
                with patch.object(experiment, "collect_tail", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "interrupted alternative"):
                        experiment.replay(root, plan, source, cached, run, observer)
                self.assertEqual(calls.call_count, 4 * plan["cases"][0]["horizon"])
                done_calls = calls.call_count
                calls.reset_mock()
                result = experiment.replay(root, plan, source, cached, run, observer)
                self.assertEqual(calls.call_count, plan["search_calls"] - done_calls)
            self.assertEqual(result["exact_source_branches"], 2)
            self.assertEqual(result["training_updates"], 0)

            case = plan["cases"][0]
            directory = root / "collection/active"
            metadata = dict(schema=c.SCHEMA, plan_fingerprint=plan["fingerprint"],
                            run_fingerprint=run["fingerprint"], case_id="active")
            anchor = c.source_common.read_cache(directory / "anchor.npz", dict(metadata, kind="anchor"))
            action = next(action for action in case["legal_actions"] if action != case["recorded_action"])
            path = directory / f"q{action}.npz"
            meta = dict(metadata, kind="continuation", forced_action=action)
            tail = c.source_common.read_cache(path, meta)
            pack = c.source_experiment.learned_pack(source["models"], case["unit"], source["prior"])
            changed = dict(entry, ground_truth=object())
            with patch.object(observer, "capture", wraps=actual_capture):
                again = saved_tail(observer, changed, ppo, anchor, case, action, source, pack, experiment.checked_image)
            for name in experiment.FIELDS:
                np.testing.assert_array_equal(tail[name], again[name])
            np.testing.assert_array_equal(tail["history_latest"][0], anchor["candidate_boxes"][0, action])

            with patch.dict("sys.modules", torch=None, cv2=None):
                self.assertEqual((plan, cached), experiment.prepare(root, source))
                self.assertEqual(result, experiment.replay(root, plan, source, cached, run))
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    experiment.replay(root, plan, source, cached, run)
            finally:
                path.write_bytes(original)
            scope_path = root74 / "analysis/stage74_scope.csv"
            original = scope_path.read_bytes()
            try:
                scope_path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    c.load_source(root74)
            finally:
                scope_path.write_bytes(original)
            c.verify(source["bindings"])
            self.assertEqual(digest(observer.rt.net), frozen)


if __name__ == "__main__":
    unittest.main()
