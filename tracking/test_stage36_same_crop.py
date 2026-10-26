"""Fixed-window planning, score/geometry diagnostics and real CPU replay checks."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage20_compression_common import state_digest
from stage20_student_train import atomic_save
from stage27_uav_common import crop_geometry, seal
from stage28_response_common import clip_boxes, dense_boxes
from stage35_preserve_common import ARMS, BASELINES, DESIGN as SOURCE_DESIGN, load_plan as load_stage35
from stage35_preserve_experiment import collect_trace
from stage35_preserve_runtime import make_student, trainable_state
from stage35_supervised_audit import aggregate, audit as baseline_audit
from test_stage34_identity_representation import fake_image, torch_runtime
from test_stage35_supervised_audit import source_fixture
from stage36_same_crop_common import DESIGN, METHODS, load_inputs, prepare, video_plan
import stage36_same_crop_collect as collector
from stage36_same_crop_summary import describe, render_matrix, summarize

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


def response_example(geometry=False):
    side, n = 4, 1
    previous, gt = np.array([16., 16., 16., 16.]), np.array([16., 16., 16., 16.])
    raw = np.full((n, 4, 4, side, side), .01, np.float32)
    sizes = np.full((n, 4, 4, 2, side, side), .25, np.float32)
    offsets = np.ones_like(sizes)
    native = np.empty((n, 4, 4, 4))
    for parent in range(4):
        for evaluator in range(4):
            peak = (1, 1) if geometry or evaluator >= 2 else (3, 3)
            raw[0, parent, evaluator, peak[0], peak[1]] = .9
            if geometry and evaluator < 2:
                sizes[0, parent, evaluator] = .75
            boxes = dense_boxes(sizes[0, parent, evaluator], offsets[0, parent, evaluator], previous, 4., 256, [64, 64])
            native[0, parent, evaluator] = boxes[peak[0] * side + peak[1]]
    return dict(raw=raw, size=sizes, offset=offsets, native_boxes=native,
        previous=np.tile(previous, (1, 4, 1)), factor=np.full((1, 4), 4.),
        gt=gt[None], image_hw=np.array([[64, 64]]), window=np.ones((4, 4), np.float32),
        layer=np.tile([0, 9, 0, 0], (1, 4, 1)), frame_id=np.array([1]))


class MathTests(unittest.TestCase):
    def test_same_crop_swaps_separate_ranking_and_geometry_and_parent_accounting(self):
        video = dict(sequence="synthetic", frames=[dict(frame_id=1, controls=[1], anchors=[])])
        source = dict(search_factor=4, search_size=256)
        responses, hybrids, parents = describe(video, response_example(), source)
        self.assertEqual(len(responses), 16)
        for row in hybrids:
            self.assertEqual(row["full_gain"], 1.)
            self.assertEqual(row["score_only_gain"], 1.)
            self.assertEqual(row["geometry_only_gain"], 0.)
        self.assertTrue(all(r["parent_change_for_student"] == 0 and r["decomposition_max_residual"] == 0 for r in parents))
        _, hybrids, _ = describe(video, response_example(geometry=True), source)
        for row in hybrids:
            self.assertEqual(row["score_only_gain"], 0.)
            self.assertGreater(row["full_gain"], .5)
            self.assertAlmostEqual(row["geometry_only_gain"], row["full_gain"])

    def test_window_rule_keeps_controls_and_uses_first_same_direction_run(self):
        traces = {m: dict(iou=np.ones(20)) for m in METHODS}
        traces[METHODS[2]]["iou"][5:] = 0
        traces[METHODS[3]]["iou"][12:] = 0
        entry = dict(sequence="case", source_group_proxy="case")
        with mock.patch.dict(DESIGN, before=1, after=1):
            result = video_plan(entry, traces)
        frames = {r["frame_id"] for r in result["frames"]}
        self.assertTrue({5, 6, 7, 10, 12, 13, 14, 15} <= frames)
        self.assertEqual([a["frame_id"] for a in result["anchors"] if a["kind"] == "first_failure"], [6, 13])
        self.assertTrue(all(a["direction"] == "student_lower" for a in result["anchors"] if a["kind"] == "first_sustained_discrepancy"))


def canonical_runtime(torch):
    rt = torch_runtime(torch)
    def decode(output, factor, image):
        with torch.no_grad():
            p = (rt.net.box_head.cal_bbox(rt.tracker.output_window * output["score_map"], output["size_map"], output["offset_map"])
                .view(-1, 4).mean(0) * rt.tracker.params.search_size / factor).tolist()
        previous = np.asarray(rt.tracker.state)
        center = np.asarray(p[:2]) + previous[:2] + previous[2:] / 2 - .5 * rt.tracker.params.search_size / factor
        return clip_boxes(np.r_[center - np.asarray(p[2:]) / 2, p[2:]], image.shape[:2])
    rt.decode = decode
    return rt


def replay_fixture(root, torch):
    source_root = source_fixture(root)
    source = load_stage35(source_root)
    rt = canonical_runtime(torch)
    original_run = json.loads((source_root / "oof/run.json").read_text())
    model_bindings, models = {}, {}
    for seed in SOURCE_DESIGN["seeds"]:
        report_path = source_root / ("training/seed{}/stage35_training_decision.json".format(seed))
        report = json.loads(report_path.read_text())
        report.pop("fingerprint")
        report["teacher_sha256"] = state_digest(rt.net)
        for record in report["arms"]:
            path = Path(record["checkpoint"])
            if record["arm"] == "supervised":
                net = make_student(rt)
                with torch.no_grad():
                    net.backbone.blocks[4].ffn[0].weight[0, 0] += .02 * (1 + record["fold"]) * (seed - SOURCE_DESIGN["seeds"][0] + 1)
                initial = make_student(rt)
                fold = source["folds"][record["fold"]]
                metadata = dict(plan_fingerprint=source["fingerprint"], mode="full", arm="supervised", seed=seed,
                    fold=fold["fold"], fit_videos=fold["fit"], held_out_videos=fold["held_out"],
                    teacher_sha256=state_digest(rt.net), initial_student_sha256=state_digest(initial))
                atomic_save(torch, dict(step=SOURCE_DESIGN["updates"], metadata=metadata, parameters=trainable_state(net)), path)
                models[(fold["fold"], "supervised_seed" + str(seed))] = net.eval().requires_grad_(False)
                record["checkpoint_sha256"] = file_hash(path)
            model_bindings[str(path)] = file_hash(path)
        write_json(report_path, seal(report))
        model_bindings[str(report_path)] = file_hash(report_path)
    new_run = seal(dict(schema=source["schema"], plan_fingerprint=source["fingerprint"], methods=original_run["methods"], frozen_model_sha256=model_bindings))
    write_json(source_root / "oof/run.json", new_run)
    # Four actual model trajectories supply diagonal replay ground truth.
    cases = [e["sequence"] for e in source["fit"][:4]]
    reader = lambda root, frame: fake_image(Path(root) / frame["file"])
    hashes = {}
    for entry in source["fit"]:
        fold = next(f["fold"] for f in source["folds"] if entry["sequence"] in f["held_out"])
        for method in original_run["methods"]:
            path = source_root / "oof" / (entry["sequence"] + "__" + method + ".npz")
            if entry["sequence"] in cases and method in METHODS:
                data = collect_trace(rt, source, entry, method, models.get((fold, method)), reader)
            else:
                with np.load(path, allow_pickle=False) as archive:
                    data = {k: archive[k] for k in archive.files if k != "metadata_json"}
            write_cache(path, data, dict(schema=source["schema"], run_fingerprint=new_run["fingerprint"],
                sequence=entry["sequence"], fold=fold, method=method))
            hashes[path.name] = file_hash(path)
    decision_path = source_root / "analysis/stage35_decision.json"
    original = json.loads(decision_path.read_text())
    original.pop("fingerprint")
    original.update(oof_run_fingerprint=new_run["fingerprint"], cache_sha256=hashes)
    write_json(decision_path, seal(original))
    from stage35_supervised_audit import load_source
    from stage27_uav_summary import write_table
    _, _, rows, _ = load_source(source_root)
    averaged = aggregate(rows)
    policy = []
    for method in BASELINES + ARMS:
        values = [r for r in averaged if r["method"] == method]
        policy.append(dict(method=method, videos=8, **{k: float(np.mean([v[k] for v in values])) for k in ("auc", "low_iou_fraction", "instrumented_ms")}))
    write_table(source_root / "analysis/stage35_policy_summary.csv", policy)
    baseline_audit(source_root)
    return source_root, rt, cases, reader


@unittest.skipUnless(TORCH, "PyTorch required for actual CPU model replay tests")
class ReplayTests(unittest.TestCase):
    def test_actual_crossed_replay_pilot_resume_dense_parity_and_frozen_sources(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()), mock.patch.object(torch.cuda, "is_available", return_value=False):
            root = Path(temporary)
            source_root, rt, cases, reader = replay_fixture(root, torch)
            output = root / "stage36"
            originals = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
            with mock.patch.dict(DESIGN, cases=cases, before=1, after=1):
                plan = prepare(source_root, output)
                self.assertEqual(plan["planned_model_forwards"], plan["unique_frame_images"] * 16)
                with mock.patch.object(rt, "prepare", wraps=rt.prepare) as prepare_spy:
                    collector.collect(rt, output, cases[0], reader)
                self.assertEqual(prepare_spy.call_count, 4 * len(plan["videos"][0]["frames"]))
                pilot = summarize(output, reader)
                self.assertEqual(pilot["status"], "PILOT_REPLAY_ONLY")
                collector.collect(rt, output, None, reader)
                final = summarize(output, reader)
                self.assertEqual(final["status"], "COMPLETE_SAME_CROP_DIAGNOSTIC")
                self.assertEqual(final["audited_cases"], 4)
                self.assertLessEqual(final["maximum_parent_decomposition_residual"], 1e-12)
                self.assertFalse(final["training_performed"])
                self.assertIsNone(final["adoption_gate"])
                collector.verify_bindings(originals)
                caches = {str(p): file_hash(p) for p in (output / "responses").glob("*.npz")}
                collector.collect(rt, output, cases[0], reader)
                self.assertEqual(caches, {str(p): file_hash(p) for p in (output / "responses").glob("*.npz")})
                self.assertEqual(final, summarize(output, reader))
                files = list((output / "analysis/figures").glob("*.png"))
                self.assertTrue(files)
                from PIL import Image
                with Image.open(files[0]) as im:
                    self.assertEqual(im.size, (1520, 1260))
                frozen_plan, source, traces = load_inputs(output)
                run = collector.load_sealed(output / "run.json")
                video = plan["videos"][0]
                entry = next(e for e in source["fit"] if e["sequence"] == cases[0])
                path = output / "responses" / (cases[0] + "__stage36_same_crop.npz")
                data = collector.load_response(path, source, entry, traces[cases[0]], video, run)
                data["previous"][0, 0, 0] += 1
                with self.assertRaises(AssertionError):
                    collector.audit_response(data, source, entry, traces[cases[0]], video, run)
                original_replay_bytes = path.read_bytes()
                path.write_bytes(original_replay_bytes + b"changed replay")
                with self.assertRaisesRegex(ValueError, "Completed Stage-36 response/source changed"):
                    summarize(output, reader)
                path.write_bytes(original_replay_bytes)
                # Source changes are rejected before another model is replayed.
                model = source_root / ("training/seed2111/fold3/supervised/step{:06d}.pt".format(SOURCE_DESIGN["updates"]))
                model.write_bytes(model.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    load_inputs(output)


if __name__ == "__main__":
    unittest.main()
